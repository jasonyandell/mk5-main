"""
Build eighty_four_throwaway snapshot corpus.

The legacy corpus has bid_value=None (treated as bid=30).  There is no
explicit 84 / mark-bid context in GameRecordGPU.

This script probes for 84-eligible hand shapes: bidder holds >= 4 doubles
AND sufficient suit dominance (>= 2 additional trump, or >3 of any off suit).
If found, we emit snapshots with bid_value=84 to document the hand shape.

Target: best-effort up to 200 snapshots; if 0 found, document the blocker.
Tests: (deferred) wave 2.B.2.
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
    trumps_held,
    suits_held,
    BIDDER,
)
from forge.eq.game_tensor import GameStateTensor
from forge.oracle.declarations import PIP_TRUMP_IDS
from forge.oracle.tables import DOMINO_IS_DOUBLE

OUTPUT_DIR = Path(__file__).parent
CORPUS_DIR = Path(PROJECT_ROOT) / "gus/data"
TARGET = 200
MAX_CHUNKS = 50  # limit scan since this is best-effort

FILTER_DEF = {
    "role": "bidder (player 0)",
    "position": "opening lead (trick 0, on lead)",
    "decl_ids": list(PIP_TRUMP_IDS),
    "84_shape": "bidder holds >= 4 doubles in initial hand AND >= 2 additional trump tiles",
    "note": "BLOCKER EXPECTED: legacy corpus has bid_value=None (bid=30); snapshots emitted with bid_value=84 as placeholder shape only",
    "status": "best-effort",
}

MIN_DOUBLES = 4  # per 84/plunge eligibility (typically need 4+ doubles to bid 84)
MIN_EXTRA_TRUMP = 2  # additional trump beyond the doubles already counted


def count_initial_doubles(game, player: int) -> int:
    """Count doubles in player's initial hand."""
    return sum(1 for did in game.hands[player] if DOMINO_IS_DOUBLE[did])


def count_initial_trumps(game, decl_id: int, player: int) -> int:
    """Count trump tiles in player's initial hand for given declaration."""
    from corpus_replayer import get_trump_set
    trump_set = get_trump_set(decl_id)
    return sum(1 for did in game.hands[player] if did in trump_set)


def passes_filter(dp) -> bool:
    """Return True if this is an 84-eligible shape at opening lead."""
    state = dp.state_before
    if dp.player != BIDDER:
        return False
    if dp.trick_number != 0:
        return False
    if state.n_trick_plays() != 0:
        return False  # must be on lead (first play of trick 0)
    if state.decl_id not in PIP_TRUMP_IDS:
        return False

    # Check initial hand shape (all tiles still held at trick 0 lead)
    game = dp.game
    n_doubles = count_initial_doubles(game, BIDDER)
    if n_doubles < MIN_DOUBLES:
        return False

    # Extra trump beyond the doubles
    n_trump = count_initial_trumps(game, state.decl_id, BIDDER)
    n_extra_trump = n_trump - n_doubles  # doubles in trump are already trump
    # Actually all pip-trump doubles count as trump, so:
    # extra non-double trump = n_trump - (doubles that are trump)
    from corpus_replayer import get_trump_set
    trump_set = get_trump_set(state.decl_id)
    n_trump_doubles = sum(1 for did in game.hands[BIDDER]
                         if DOMINO_IS_DOUBLE[did] and did in trump_set)
    n_non_double_trump = n_trump - n_trump_doubles

    if n_non_double_trump < MIN_EXTRA_TRUMP:
        return False

    return True


def build_corpus(max_chunks: int = MAX_CHUNKS) -> None:
    chunk_files = sorted(CORPUS_DIR.glob("corpus_train_chunk_*.pt"))[:max_chunks]
    if not chunk_files:
        raise FileNotFoundError(f"No corpus chunks found in {CORPUS_DIR}")

    snapshots: list[dict] = []
    source_chunks: list[str] = []
    candidates = 0
    chunks_used = 0

    for chunk_path in chunk_files:
        if len(snapshots) >= TARGET:
            break
        chunk_name = chunk_path.name
        chunk_found = 0
        try:
            for dp in iter_decisions(str(chunk_path)):
                candidates += 1
                if passes_filter(dp):
                    # Emit with bid_value=84 as placeholder
                    snap = dp.state_before.to_snapshot(bid_value=84)
                    snap["_source_chunk"] = chunk_name
                    snap["_game_idx"] = dp.game_idx
                    snap["_decision_idx"] = dp.decision_idx
                    snap["_action_taken_slot"] = dp.decision.action_taken
                    snap["_e_q"] = dp.decision.e_q.tolist()
                    snap["_legal_mask"] = dp.decision.legal_mask.tolist()
                    snap["_bid_value_note"] = "placeholder=84 (corpus actual bid_value=None=30)"
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

    # Validate (even placeholder shapes must pass schema)
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

    blocker_note = None
    if len(valid_snapshots) == 0:
        blocker_note = (
            "BLOCKER: No 84-eligible hand shapes found within scanned chunks. "
            "The legacy corpus uses bid_value=None (implicit bid=30) across all games. "
            "True 84-hand analysis requires Wave 2.B.2 corpus generation with "
            "explicit 84-bid auction context."
        )
        print(f"\nBLOCKER: {blocker_note}")
    else:
        print(f"\nFound {len(valid_snapshots)} 84-eligible shapes (bid_value placeholder=84).")
        print("Note: These are SHAPE matches only; actual bid context is bid=30.")

    manifest = {
        "corpus": "eighty_four_throwaway",
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
        "claim_tested": "ch07-eighty-four-throwaway (deferred)",
        "status": "deferred" if len(valid_snapshots) == 0 else "partial",
        "blocker": blocker_note,
        "deferred_to": "Wave 2.B.2",
        "output": str(out_jsonl),
        "sha256": sha256,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(valid_snapshots)} snapshots to {out_jsonl}")
    print(f"Manifest: {OUTPUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    build_corpus()
