#!/usr/bin/env python3
"""Narrate a single Texas 42 game played by the E[Q] bot from one player's POV.

Usage:
    python -m lem.narrate --seed 42 --narrator 3
    python -m lem.narrate --seed 42 --narrator 3 --decl fives --bid 30
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_CHECKPOINT = "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Narrate one Texas 42 game from a player's seat."
    )
    parser.add_argument("--seed", type=int, required=True, help="Deal seed")
    parser.add_argument(
        "--narrator",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="Which player to narrate from (default: 3)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Q-value oracle checkpoint for E[Q] play",
    )
    parser.add_argument(
        "--decl",
        type=str,
        default=None,
        help="Force a declaration (e.g. 'fives', 'notrump'). Default: bidder picks.",
    )
    parser.add_argument(
        "--bid",
        type=int,
        default=None,
        help="Force a bid value (default: bidder picks, or 30 if --decl is forced).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="E[Q] sample count per decision (default: 10)",
    )
    parser.add_argument(
        "--bidder-samples",
        type=int,
        default=50,
        help="Simulations per trump during bidder evaluation (default: 50)",
    )
    parser.add_argument(
        "--stop-at-decision",
        type=int,
        default=None,
        help="Truncate before decision N (0-27) and emit a 'what do you play?' prompt. "
        "N must be one of the narrator's turns.",
    )
    parser.add_argument(
        "--stop-at-narrator-turn",
        type=int,
        default=None,
        help="Truncate at the narrator's K-th turn (0=first, 6=last). "
        "Robust to run-to-run variation in the play trajectory.",
    )
    parser.add_argument(
        "--with-primer",
        action="store_true",
        help="Prepend lem/rules/primer.md (the rules preamble) to the narration.",
    )
    parser.add_argument(
        "--show-eq",
        action="store_true",
        help="Include E[Q] counterfactuals on real choices (not yet wired — stay tuned)",
    )
    args = parser.parse_args()

    if args.show_eq:
        log("[note] --show-eq not yet wired; ignoring.")

    from forge.oracle.declarations import DECL_ID_TO_NAME, DECL_NAME_TO_ID, parse_decl_arg
    from forge.oracle.rng import deal_from_seed

    hands = deal_from_seed(args.seed)
    log(f"[deal] seed={args.seed}")
    log(f"[deal] P0 hand = {hands[0]}")
    log(f"[deal] P1 hand = {hands[1]}")
    log(f"[deal] P2 hand = {hands[2]}")
    log(f"[deal] P3 hand = {hands[3]}")

    # --- Resolve decl + bid ---
    if args.decl is not None:
        decl_id = parse_decl_arg(args.decl).decl_ids[0]
        bid = args.bid if args.bid is not None else 30
        log(f"[decl] forced: {DECL_ID_TO_NAME[decl_id]} @ {bid}")
    else:
        log("[bidder] running simulation to pick trump+bid for P0...")
        from forge.bidding.estimator import find_best_bid
        from forge.bidding.evaluate import run_evaluation
        from forge.bidding.inference import PolicyModel

        policy_model = PolicyModel()
        results, bidder_elapsed = run_evaluation(
            policy_model,
            hand=hands[0],
            n_samples=args.bidder_samples,
            seed=args.seed,
        )
        trump_name, bid, swing = find_best_bid(results)
        decl_id = DECL_NAME_TO_ID[trump_name]
        log(
            f"[bidder] picked {trump_name} @ {bid} "
            f"(mark swing {swing:+.2f}, {bidder_elapsed:.1f}s)"
        )

    # --- Load Q-value oracle and play the game ---
    log(f"[oracle] loading {args.checkpoint}")
    from forge.eq.oracle import Stage1Oracle

    oracle = Stage1Oracle(args.checkpoint, device="cuda", compile=False)

    log(f"[play] generating game at N={args.n_samples}")
    from forge.eq.generate.pipeline import generate_eq_games_gpu

    game_records = generate_eq_games_gpu(
        model=oracle.model,
        hands=[hands],
        decl_ids=[decl_id],
        n_samples=args.n_samples,
        device="cuda",
        seeds=[args.seed],
    )
    record = game_records[0]
    log(f"[play] got {len(record.decisions)} decisions")

    # --- Resolve stop index ---
    stop_at = args.stop_at_decision
    if args.stop_at_narrator_turn is not None:
        if stop_at is not None:
            log("[warn] both --stop-at-decision and --stop-at-narrator-turn set; using --stop-at-narrator-turn")
        narrator_turns = [
            i for i, d in enumerate(record.decisions) if d.player == args.narrator
        ]
        k = args.stop_at_narrator_turn
        if not (0 <= k < len(narrator_turns)):
            log(
                f"[error] narrator turn {k} out of range; narrator has "
                f"{len(narrator_turns)} turns (0..{len(narrator_turns)-1})"
            )
            sys.exit(1)
        stop_at = narrator_turns[k]
        log(f"[stop] resolved narrator turn {k} → decision index {stop_at}")

    # --- Render ---
    from lem.narrate.render import render_narration

    text = render_narration(
        record,
        narrator=args.narrator,
        bid=bid,
        bidder=0,
        stop_at_decision=stop_at,
    )

    if args.with_primer:
        from pathlib import Path

        primer_path = Path(__file__).resolve().parent.parent / "rules" / "primer.md"
        if not primer_path.exists():
            log(f"[warn] --with-primer set but {primer_path} does not exist; skipping")
        else:
            primer = primer_path.read_text()
            text = primer.rstrip() + "\n\n---\n\n" + text

    print(text)


if __name__ == "__main__":
    main()
