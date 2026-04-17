#!/usr/bin/env python3
"""Generate trick-6 decision prompts with E[Q] grading info.

Each example: compact game state prompt + E[Q] data for K1 grading.
Mirrors the comprehension pipeline but for play decisions.

Usage:
    python -u -m lem.rules.generate_decisions \
        --start-seed 900000 --count 20 --allow-eval-seeds \
        --output lem/data/decisions_scout.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from lem.rules.generate_comprehension import (
    EVAL_SEED_START, EVAL_SEED_END,
    DEFAULT_CHECKPOINT, _t, _dom, _decl_name,
    extract_position, render_compact_prompt,
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def extract_decision(record, narrator: int, seed: int, min_eq_gap: float) -> dict | None:
    """Extract a decision with E[Q] data. Returns None if filtered out."""
    pos = extract_position(record, narrator, seed)
    if pos is None:
        return None

    # Find the narrator's 6th decision in the record
    narrator_turns = [
        (i, d) for i, d in enumerate(record.decisions) if d.player == narrator
    ]
    if len(narrator_turns) < 6:
        return None
    dec_idx, decision = narrator_turns[5]

    # E[Q] data
    eq_vals = decision.e_q.clone()
    eq_vals[~decision.legal_mask] = float("-inf")

    legal_slots = [s for s in range(7) if decision.legal_mask[s]]
    if len(legal_slots) < 2:
        return None

    sorted_eq = eq_vals[decision.legal_mask].sort(descending=True).values
    gap = (sorted_eq[0] - sorted_eq[1]).item()
    if gap < min_eq_gap:
        return None

    initial_hands = record.hands
    bot_slot = decision.action_taken
    bot_dom_id = initial_hands[narrator][bot_slot]
    bot_eq = eq_vals[bot_slot].item()

    best_slot = eq_vals.argmax().item()
    best_dom_id = initial_hands[narrator][best_slot]
    best_eq = eq_vals[best_slot].item()

    all_eq = {}
    for s in legal_slots:
        did = initial_hands[narrator][s]
        all_eq[_dom(did)] = round(eq_vals[s].item(), 3)

    prompt = render_compact_prompt(pos)

    # Structured state for the verifier (avoids parsing the prompt)
    remaining_hand = [_dom(d) for d in pos.remaining_hand]
    plays = []
    for trick in pos.completed_tricks:
        for i, play in enumerate(trick.plays):
            plays.append({
                "player": play.player,
                "dom": _dom(play.dom_id),
                "trick": trick.trick_num,
                "is_lead": i == 0,
            })
    for i, play in enumerate(pos.current_trick_plays):
        plays.append({
            "player": play.player,
            "dom": _dom(play.dom_id),
            "trick": 6,
            "is_lead": i == 0,
        })

    return {
        "seed": seed,
        "decl_id": pos.decl_id,
        "decl_name": _decl_name(pos.decl_id),
        "narrator": narrator,
        "partner": pos.partner,
        "prompt": prompt,
        "question": "What do you play and why?",
        "legal_actions": [_dom(d) for d in pos.legal_actions],
        "bot_action": _dom(bot_dom_id),
        "bot_eq": round(bot_eq, 3),
        "best_action": _dom(best_dom_id),
        "best_eq": round(best_eq, 3),
        "eq_gap": round(gap, 3),
        "all_eq": all_eq,
        "n_legal": len(legal_slots),
        "remaining_hand": remaining_hand,
        "plays": plays,
        "is_leading": pos.is_leading,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-seed", type=int, default=900000)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--min-eq-gap", type=float, default=1.0,
                        help="Min E[Q] gap between best and 2nd-best (points)")
    parser.add_argument("--allow-eval-seeds", action="store_true")
    parser.add_argument("--max-decisions", type=int, default=20,
                        help="Stop after this many decisions (for scout runs)")
    args = parser.parse_args()

    import torch
    from forge.oracle.declarations import N_DECLS
    from forge.oracle.rng import deal_from_seed
    from forge.eq.generate.pipeline import generate_eq_games_gpu
    from forge.eq.oracle import Stage1Oracle

    log(f"[model] Loading oracle from {args.checkpoint}")
    oracle = Stage1Oracle(args.checkpoint, device="cuda", compile=False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_decisions = 0
    t_start = time.time()

    with open(output_path, "w") as f:
        for seed_offset in range(args.count):
            if total_decisions >= args.max_decisions:
                break

            seed = args.start_seed + seed_offset
            if not args.allow_eval_seeds and EVAL_SEED_START <= seed <= EVAL_SEED_END:
                continue

            hands = deal_from_seed(seed)
            batch_hands = [hands] * N_DECLS
            batch_decls = list(range(N_DECLS))

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
                log(f"[error] seed {seed}: {e}")
                continue

            for record in records:
                for narrator in range(4):
                    if total_decisions >= args.max_decisions:
                        break
                    ex = extract_decision(record, narrator, seed, args.min_eq_gap)
                    if ex is None:
                        continue
                    f.write(json.dumps(ex) + "\n")
                    total_decisions += 1
                if total_decisions >= args.max_decisions:
                    break

            if (seed_offset + 1) % 5 == 0:
                log(f"[progress] {seed_offset + 1}/{args.count} seeds | "
                    f"{total_decisions} decisions")

    elapsed = time.time() - t_start
    log(f"\n[done] {total_decisions} decisions in {elapsed:.0f}s")
    log(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
