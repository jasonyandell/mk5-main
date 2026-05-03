"""Build reentry-preservation snapshot corpus.

Generates 200 mid-game snapshots from random plays on fresh deals.  Each
snapshot satisfies the "single reentry, vulnerable off" shape:
  - It is the bidder's (player 0) turn to play.
  - The bidder holds exactly 1 trump remaining.
  - The bidder holds dominoes in at least 2 distinct non-trump suits.

This shape is the canonical mid-game position for Ch 03 reentry-preservation:
the bidder has one trump left (their reentry to the trump suit), and must
decide whether to burn it now or preserve it.

Output: snapshots.jsonl  (one JSON line per snapshot)

Usage:
    python build_reentry_corpus.py [--n-snapshots 200] [--max-seeds 5000] \
                                   [--seed 0] [--output snapshots.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from forge.eq.game_tensor import GameStateTensor, SNAPSHOT_SCHEMA_VERSION
from forge.oracle.declarations import (
    DOUBLES_TRUMP,
    DOUBLES_SUIT,
    NOTRUMP,
    N_DECLS,
    PIP_TRUMP_IDS,
)
from forge.oracle.rng import deal_from_seed
from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW, DOMINO_IS_DOUBLE


def is_trump(domino_id: int, decl_id: int) -> bool:
    """True if domino_id is in the called suit for decl_id."""
    if decl_id in PIP_TRUMP_IDS:
        return decl_id in (DOMINO_HIGH[domino_id], DOMINO_LOW[domino_id])
    if decl_id in (DOUBLES_TRUMP, DOUBLES_SUIT):
        return bool(DOMINO_IS_DOUBLE[domino_id])
    if decl_id == NOTRUMP:
        return False
    raise ValueError(f"Unknown decl_id: {decl_id}")


def hand_pip_suits(hand_dominoes: list[int], decl_id: int) -> set[int]:
    """Return the set of non-trump pip suits represented in the hand."""
    suits: set[int] = set()
    for did in hand_dominoes:
        if is_trump(did, decl_id):
            continue
        # A domino belongs to its HIGH pip suit (following convention)
        suits.add(int(DOMINO_HIGH[did]))
    return suits


def check_reentry_shape(
    state: GameStateTensor,
    decl_id: int,
    bidder: int,
) -> bool:
    """Return True if the state is a valid reentry-preservation snapshot.

    Conditions:
    1. It is the bidder's turn.
    2. Exactly 1 trump remains in the bidder's hand.
    3. At least 2 distinct non-trump suits remain in the bidder's hand.
    """
    # 1. Bidder's turn?
    current = int(state.current_player[0].item())
    if current != bidder:
        return False

    # 2. Bidder hand dominoes
    hand = state.hands[0, bidder]  # (7,) int8, -1 for empty
    remaining = [int(d) for d in hand.tolist() if d >= 0]

    if not remaining:
        return False

    n_trumps = sum(1 for d in remaining if is_trump(d, decl_id))
    if n_trumps != 1:
        return False

    # 3. At least 2 non-trump pip suits
    off_suits = hand_pip_suits(remaining, decl_id)
    if len(off_suits) < 2:
        return False

    return True


def simulate_game_collecting_snapshots(
    deal: list[list[int]],
    decl_id: int,
    bidder: int,
    rng: random.Random,
    device: str,
    max_snapshots_per_game: int,
) -> list[dict]:
    """Replay a random game, collect all reentry-shape snapshots."""
    bid_value = 30  # Fixed for corpus generation

    state = GameStateTensor.from_deals(
        hands=[deal],
        decl_ids=[decl_id],
        device=device,
        bidders=[bidder],
    )

    collected: list[dict] = []

    while state.active_games().any() and len(collected) < max_snapshots_per_game:
        if check_reentry_shape(state, decl_id, bidder):
            snaps = state.to_snapshot(bid_values=[bid_value])
            collected.extend(snaps)

        # Random legal action
        legal = state.legal_actions()[0]
        legal_slots = legal.nonzero(as_tuple=True)[0].tolist()
        if not legal_slots:
            break
        slot = rng.choice(legal_slots)
        state = state.apply_actions(torch.tensor([slot], device=device))

    return collected


def build_corpus(
    n_snapshots: int,
    max_seeds: int,
    seed: int,
    device: str,
) -> list[dict]:
    """Mine reentry-shape snapshots from fresh deals."""
    rng = random.Random(seed)
    snapshots: list[dict] = []

    seeds_tried = 0
    for base_seed in range(max_seeds):
        if len(snapshots) >= n_snapshots:
            break

        seeds_tried += 1
        deal = deal_from_seed(base_seed)

        # Try pip-trump declarations (0–6) — NOTRUMP and doubles have no
        # meaningful "reentry" since there is no trump suit.
        # Shuffle to avoid bias toward low decl_ids.
        pip_decls = list(PIP_TRUMP_IDS)
        rng.shuffle(pip_decls)

        for decl_id in pip_decls:
            if len(snapshots) >= n_snapshots:
                break

            collected = simulate_game_collecting_snapshots(
                deal=deal,
                decl_id=decl_id,
                bidder=0,  # P0 is always bidder in the corpus
                rng=rng,
                device=device,
                max_snapshots_per_game=3,
            )
            snapshots.extend(collected)

            if len(snapshots) % 20 == 0 and len(snapshots) > 0:
                print(
                    f"  [{len(snapshots):3d}/{n_snapshots}] seed={base_seed} decl={decl_id}",
                    flush=True,
                )

    print(
        f"Mined {len(snapshots)} snapshots from {seeds_tried} seeds.",
        flush=True,
    )
    return snapshots[:n_snapshots]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-snapshots", type=int, default=200)
    parser.add_argument("--max-seeds", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "snapshots.jsonl"),
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building reentry-preservation corpus ({args.n_snapshots} snapshots)…", flush=True)
    snapshots = build_corpus(
        n_snapshots=args.n_snapshots,
        max_seeds=args.max_seeds,
        seed=args.seed,
        device=args.device,
    )

    if len(snapshots) < args.n_snapshots:
        print(
            f"Warning: only found {len(snapshots)} qualifying snapshots "
            f"(wanted {args.n_snapshots}). Increase --max-seeds.",
            flush=True,
        )

    with out_path.open("w") as fh:
        for snap in snapshots:
            fh.write(json.dumps(snap) + "\n")

    print(f"Wrote {len(snapshots)} snapshots to {out_path}", flush=True)

    # Write manifest
    import hashlib
    content = out_path.read_bytes()
    sha = hashlib.sha256(content).hexdigest()

    manifest = {
        "corpus": "reentry_preservation",
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "n_snapshots": len(snapshots),
        "seed": args.seed,
        "max_seeds_tried": args.max_seeds,
        "output": str(out_path),
        "sha256": sha,
        "shape_filter": {
            "bidder_turn": True,
            "n_trumps_remaining": 1,
            "min_distinct_off_suits": 2,
        },
        "claim_tested": "ch03-reentry-preservation",
    }
    manifest_path = out_path.parent / "manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote manifest to {manifest_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
