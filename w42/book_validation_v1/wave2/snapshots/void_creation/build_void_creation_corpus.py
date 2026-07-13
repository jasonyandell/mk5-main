"""
Build void_creation snapshot corpus.

Filter: setter (left or right opponent of bidder) on lead, with the option to
play a non-following tile that would void them in a suit they currently hold
>= 1 tile in.  Decision must be at trick >= 1.

"Setter" = team 1 (players 1 and 3).
"On lead" = trick_plays is empty ([−1,−1,−1,−1]) AND player is setter.
"Has void-creation option" = player holds >= 2 suits, at least one of which
they could void by leading from it (i.e., they hold exactly 1 tile in that
suit, and leading it would expose it).

Practical filter: setter is leading AND they hold >= 2 off-suit groups where
at least one group has exactly 1 domino (voiding candidate).

Target: 500 snapshots.
Tests: ch05-void-creation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = "/Users/jason/code/mk5-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SNAPSHOTS_DIR = Path(PROJECT_ROOT) / "w42/book_validation_v1/wave2/snapshots"
sys.path.insert(0, str(SNAPSHOTS_DIR))

from corpus_replayer import (
    iter_decisions,
    suits_held,
    BIDDER,
)
from forge.eq.game_tensor import GameStateTensor

OUTPUT_DIR = Path(__file__).parent
CORPUS_DIR = Path(PROJECT_ROOT) / "gus/data"
TARGET = 500

FILTER_DEF = {
    "role": "setter (player 1 or 3)",
    "position": "on lead (trick_plays empty)",
    "min_trick_number": 1,
    "void_creation_condition": "holds >= 2 distinct suits, at least 1 suit has exactly 1 domino",
    "note": "Tests ch05-void-creation: leading to void a suit",
}

SETTER_PLAYERS = {1, 3}  # team 1 = opponents


def passes_filter(dp) -> bool:
    """Return True if this is a void-creation opportunity for the setter."""
    # Must be a setter player
    if dp.player not in SETTER_PLAYERS:
        return False
    # Trick >= 1
    if dp.trick_number < 1:
        return False
    # Must be on lead (no trick plays yet)
    n_trick = dp.state_before.n_trick_plays()
    if n_trick != 0:
        return False
    # Player must hold >= 2 distinct suits with >= 1 tile each
    suit_map = suits_held(dp.state_before, dp.player)
    # Count non-empty suits
    non_empty_suits = {s: tiles for s, tiles in suit_map.items() if tiles}
    if len(non_empty_suits) < 2:
        return False
    # At least one suit has exactly 1 domino (voiding candidate)
    singleton_suits = [s for s, tiles in non_empty_suits.items() if len(tiles) == 1]
    if not singleton_suits:
        return False
    return True


def build_corpus(max_chunks: int = 200) -> None:
    chunk_files = sorted(CORPUS_DIR.glob("corpus_train_chunk_*.pt"))
    if not chunk_files:
        raise FileNotFoundError(f"No corpus chunks found in {CORPUS_DIR}")

    snapshots: list[dict] = []
    source_chunks: list[str] = []
    candidates = 0
    chunks_used = 0

    for chunk_path in chunk_files[:max_chunks]:
        if len(snapshots) >= TARGET:
            break
        chunk_name = chunk_path.name
        chunk_found = 0
        try:
            for dp in iter_decisions(str(chunk_path)):
                candidates += 1
                if passes_filter(dp):
                    snap = dp.state_before.to_snapshot(bid_value=30)
                    snap["_source_chunk"] = chunk_name
                    snap["_game_idx"] = dp.game_idx
                    snap["_decision_idx"] = dp.decision_idx
                    snap["_action_taken_slot"] = dp.decision.action_taken
                    snap["_e_q"] = dp.decision.e_q.tolist()
                    snap["_legal_mask"] = dp.decision.legal_mask.tolist()
                    snapshots.append(snap)
                    chunk_found += 1
                    if len(snapshots) >= TARGET:
                        break
        except Exception as e:
            print(f"  Warning: {chunk_name} failed: {e}", flush=True)
            continue

        if chunk_found > 0:
            source_chunks.append(chunk_name)
        chunks_used += 1
        print(
            f"  {chunk_name}: {chunk_found} matching | total: {len(snapshots)}",
            flush=True,
        )

    print(f"\nTotal candidates scanned: {candidates}")
    print(f"Snapshots collected: {len(snapshots)}")

    # Validate
    valid_count = 0
    invalid_count = 0
    print("Validating ...", flush=True)
    for i, snap in enumerate(snapshots):
        snap_clean = {k: v for k, v in snap.items() if not k.startswith("_")}
        try:
            GameStateTensor.from_snapshot([snap_clean], device="cpu")
            valid_count += 1
        except Exception as e:
            invalid_count += 1
            if invalid_count <= 3:
                print(f"  snapshot[{i}] failed: {e}")

    valid_snapshots = []
    for snap in snapshots:
        snap_clean = {k: v for k, v in snap.items() if not k.startswith("_")}
        try:
            GameStateTensor.from_snapshot([snap_clean], device="cpu")
            valid_snapshots.append(snap)
        except Exception:
            pass

    out_jsonl = OUTPUT_DIR / "snapshots.jsonl"
    with open(out_jsonl, "w") as f:
        for snap in valid_snapshots:
            f.write(json.dumps(snap) + "\n")

    sha256 = hashlib.sha256(out_jsonl.read_bytes()).hexdigest()

    manifest = {
        "corpus": "void_creation",
        "schema_version": "forge.eq.snapshot.v1",
        "source": "oracle-greedy legacy corpus",
        "n_snapshots": len(valid_snapshots),
        "n_candidates_scanned": candidates,
        "n_chunks_used": chunks_used,
        "source_chunks": source_chunks,
        "filter_definition": FILTER_DEF,
        "validation": {
            "n_valid": valid_count,
            "n_invalid": invalid_count,
            "method": "GameStateTensor.from_snapshot",
        },
        "claim_tested": "ch05-void-creation",
        "output": str(out_jsonl),
        "sha256": sha256,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(valid_snapshots)} snapshots to {out_jsonl}")
    print(f"Manifest: {OUTPUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    build_corpus()
