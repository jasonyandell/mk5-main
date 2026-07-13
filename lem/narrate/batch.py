#!/usr/bin/env python3
"""Batch narration generator for Stage 1 STaR training data.

For each seed: plays all 10 declarations via E[Q] (N=10), renders narrations
from all 4 player perspectives, filters to trick-6 decisions with real choices
and meaningful E[Q] gaps, and outputs a JSONL dataset.

Usage:
    # Generate 100 seeds (= up to 100 × 10 × 4 = 4000 perspectives, filtered)
    python -m lem.narrate.batch --start-seed 0 --count 100 \
        --checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt \
        --output lem/data/narrations.jsonl

    # Quick test with 5 seeds
    python -m lem.narrate.batch --start-seed 0 --count 5 --output scratch/test_batch.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EVAL_SEED_START = 900000
EVAL_SEED_END = 909999

DEFAULT_CHECKPOINT = "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def dom_str(dom_id: int, domino_high, domino_low) -> str:
    return f"{domino_high[dom_id]}-{domino_low[dom_id]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch narration generator for STaR")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=100, help="Number of seeds to process")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--n-samples", type=int, default=10, help="E[Q] samples per decision")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Games per GPU batch (10 = one seed's worth of decls)")
    parser.add_argument("--eq-gap", type=float, default=1.0,
                        help="Min E[Q] gap between best and second-best to keep (points)")
    parser.add_argument("--primer", type=str, default="lem/rules/primer.md",
                        help="Rules primer to prepend")
    parser.add_argument("--bid", type=int, default=30,
                        help="Bid value for all games (default 30 — bid doesn't affect E[Q] play)")
    parser.add_argument("--allow-eval-seeds", action="store_true",
                        help="Allow generating from eval seed range (900000-909999). "
                        "Use this ONLY for eval dataset generation.")
    args = parser.parse_args()

    # Imports (heavy — do after arg parsing)
    import torch
    from forge.oracle.declarations import DECL_ID_TO_NAME, N_DECLS
    from forge.oracle.rng import deal_from_seed
    from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW
    from forge.eq.generate.pipeline import generate_eq_games_gpu
    from forge.eq.oracle import Stage1Oracle
    from lem.narrate.render import render_narration

    # Load primer
    primer_path = Path(args.primer)
    if not primer_path.exists():
        log(f"[error] Primer not found: {primer_path}")
        sys.exit(1)
    primer_text = primer_path.read_text().rstrip()

    # Warn if generating eval seeds
    end_seed = args.start_seed + args.count
    if args.start_seed < EVAL_SEED_END and end_seed > EVAL_SEED_START:
        log(f"[WARN] Seed range {args.start_seed}-{end_seed} overlaps eval seeds "
            f"{EVAL_SEED_START}-{EVAL_SEED_END}! Eval seeds will be SKIPPED.")

    # Load oracle
    log(f"[model] Loading oracle from {args.checkpoint}")
    oracle = Stage1Oracle(args.checkpoint, device="cuda", compile=False)

    # Prepare output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_decl_ids = list(range(N_DECLS))
    total_kept = 0
    total_skipped_forced = 0
    total_skipped_gap = 0
    total_skipped_eval = 0
    total_games = 0
    t_start = time.time()

    with open(output_path, "w") as f:
        for seed_offset in range(args.count):
            seed = args.start_seed + seed_offset

            # Skip eval seeds (unless explicitly allowed for eval generation)
            if not args.allow_eval_seeds and EVAL_SEED_START <= seed <= EVAL_SEED_END:
                total_skipped_eval += 1
                continue

            hands = deal_from_seed(seed)

            # Build batch: one game per declaration
            batch_hands = [hands] * N_DECLS
            batch_decls = all_decl_ids

            # Play all 10 games at once
            try:
                records = generate_eq_games_gpu(
                    model=oracle.model,
                    hands=batch_hands,
                    decl_ids=batch_decls,
                    n_samples=args.n_samples,
                    device="cuda",
                    seeds=[seed * 10 + d for d in range(N_DECLS)],
                )
            except Exception as e:
                log(f"[error] seed {seed} failed: {e}")
                continue

            total_games += len(records)

            # For each game, check all 4 narrator perspectives
            for record in records:
                for narrator in range(4):
                    # Find narrator's decisions
                    narrator_turns = [
                        (i, d) for i, d in enumerate(record.decisions)
                        if d.player == narrator
                    ]

                    # Look at the second-to-last turn (trick 6 region)
                    # Turn index 5 (0-based) = narrator's 6th play = trick 6
                    if len(narrator_turns) < 6:
                        continue

                    dec_idx, decision = narrator_turns[5]

                    # Filter: must have a real choice
                    n_legal = int(decision.legal_mask.sum().item())
                    if n_legal < 2:
                        total_skipped_forced += 1
                        continue

                    # Filter: E[Q] gap between best and second-best
                    eq_vals = decision.e_q.clone()
                    eq_vals[~decision.legal_mask] = float("-inf")
                    sorted_eq = eq_vals[decision.legal_mask].sort(descending=True).values
                    if len(sorted_eq) >= 2:
                        gap = (sorted_eq[0] - sorted_eq[1]).item()
                    else:
                        gap = 0.0

                    if gap < args.eq_gap:
                        total_skipped_gap += 1
                        continue

                    # Render narration truncated at this decision
                    try:
                        narration = render_narration(
                            record,
                            narrator=narrator,
                            bid=args.bid,
                            bidder=0,
                            stop_at_decision=dec_idx,
                        )
                    except (ValueError, AssertionError):
                        continue

                    # Prepend primer
                    full_prompt = primer_text + "\n\n---\n\n" + narration

                    # Extract E[Q] grading data
                    initial_hands = record.hands
                    bot_slot = decision.action_taken
                    bot_dom_id = initial_hands[narrator][bot_slot]
                    bot_eq = eq_vals[bot_slot].item()

                    best_slot = eq_vals.argmax().item()
                    best_dom_id = initial_hands[narrator][best_slot]
                    best_eq = eq_vals[best_slot].item()

                    # All legal actions with E[Q]
                    all_eq = {}
                    for slot in range(7):
                        if decision.legal_mask[slot]:
                            did = initial_hands[narrator][slot]
                            all_eq[dom_str(did, DOMINO_HIGH, DOMINO_LOW)] = round(eq_vals[slot].item(), 3)

                    # Legal action names
                    legal_actions = [
                        dom_str(initial_hands[narrator][s], DOMINO_HIGH, DOMINO_LOW)
                        for s in range(7) if decision.legal_mask[s]
                    ]

                    example = {
                        "seed": seed,
                        "decl_id": record.decl_id,
                        "decl_name": DECL_ID_TO_NAME[record.decl_id],
                        "narrator": narrator,
                        "decision_index": dec_idx,
                        "prompt": full_prompt,
                        "legal_actions": legal_actions,
                        "bot_action": dom_str(bot_dom_id, DOMINO_HIGH, DOMINO_LOW),
                        "bot_eq": round(bot_eq, 3),
                        "best_action": dom_str(best_dom_id, DOMINO_HIGH, DOMINO_LOW),
                        "best_eq": round(best_eq, 3),
                        "eq_gap": round(gap, 3),
                        "all_eq": all_eq,
                        "n_legal": n_legal,
                        "n_samples": args.n_samples,
                    }
                    f.write(json.dumps(example) + "\n")
                    total_kept += 1

            # Progress
            elapsed = time.time() - t_start
            seeds_done = seed_offset + 1
            rate = seeds_done / elapsed if elapsed > 0 else 0
            if seeds_done % 10 == 0 or seeds_done == args.count:
                log(
                    f"[progress] {seeds_done}/{args.count} seeds | "
                    f"{total_games} games | {total_kept} kept | "
                    f"{total_skipped_forced} forced | {total_skipped_gap} low-gap | "
                    f"{rate:.1f} seeds/s"
                )

    elapsed = time.time() - t_start
    log(f"\n[done] {total_kept} examples from {total_games} games "
        f"({args.count} seeds) in {elapsed:.0f}s")
    log(f"  Kept: {total_kept}")
    log(f"  Skipped (forced): {total_skipped_forced}")
    log(f"  Skipped (low gap): {total_skipped_gap}")
    log(f"  Skipped (eval seeds): {total_skipped_eval}")
    log(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
