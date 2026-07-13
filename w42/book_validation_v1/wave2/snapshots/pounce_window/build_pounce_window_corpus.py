"""
Build pounce_window snapshot corpus (bid=30 slice).

Filter: setter (player 1 or 3) on lead or following, bidder has played an
offsuit tile in the current or immediately previous trick that exposes a count
domino, and that count domino is currently winnable by the setter's side
(i.e., it's still in play or was just played by bidder's side).

Practical implementation:
  - Count domino was played by bidder's team (player 0 or 2) in the last
    completed trick (or current trick if setter is following).
  - Setter is on lead or following.
  - Trick >= 1.

Simpler tractable filter:
  - Decision at trick >= 1.
  - Current player is setter (player 1 or 3).
  - A count domino was played by the bidder's team in the most recently
    completed trick (trick N-1).
  - The current trick has not yet been led by setter, OR setter is following
    and a count domino from bidder's side is in the current trick.

We use a pragmatic proxy: in the last completed trick (if any), the bidder's
side played at least one count domino (>= 5 count points). This captures the
pounce-window position where the setter can play aggressively.

Target: 500 snapshots.
Tests: ch12-setter-pounce (bid=30 slice).
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
    BIDDER,
)
from forge.eq.game_tensor import GameStateTensor
from forge.oracle.tables import DOMINO_COUNT_POINTS

OUTPUT_DIR = Path(__file__).parent
CORPUS_DIR = Path(PROJECT_ROOT) / "gus/data"
TARGET = 500

FILTER_DEF = {
    "role": "setter (player 1 or 3)",
    "position": "on lead or following",
    "min_trick_number": 1,
    "pounce_condition": "bidder's team played a count domino (>= 5 pts) in the last completed trick",
    "note": "Tests ch12-setter-pounce bid=30 slice",
}

SETTER_PLAYERS = {1, 3}
BIDDER_TEAM = {0, 2}
COUNT_THRESHOLD = 5  # minimum count points for a 'count domino'


def _last_trick_plays(state) -> list[tuple[int, int]]:
    """
    Return [(player, domino_id)] for the most recently completed trick.
    Returns [] if no trick has been completed yet.
    """
    history_count = state.history_count
    if history_count < 4:
        return []  # no completed trick yet

    # The most recently completed trick ends at the last multiple of 4
    # within history_count.  Current trick plays are NOT in history yet
    # (they're in trick_plays).  But history records ALL played dominoes
    # including current trick plays.
    # Actually: history records every play including current trick.
    # Completed tricks: floor(history_count / 4) * 4 - 4 .. floor * 4 - 1
    completed_tricks = history_count // 4
    if completed_tricks == 0:
        return []
    last_trick_start = (completed_tricks - 1) * 4
    return [
        (state.history[last_trick_start + i][0], state.history[last_trick_start + i][1])
        for i in range(4)
    ]


def passes_filter(dp) -> bool:
    """Return True if this is a pounce-window shape for the setter."""
    state = dp.state_before

    # Setter only
    if dp.player not in SETTER_PLAYERS:
        return False
    if dp.trick_number < 1:
        return False

    # Check last completed trick for count domino from bidder's team
    last_trick = _last_trick_plays(state)
    if not last_trick:
        return False

    bidder_count_played = any(
        player in BIDDER_TEAM and DOMINO_COUNT_POINTS[domino] >= COUNT_THRESHOLD
        for player, domino in last_trick
    )
    if not bidder_count_played:
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
        "corpus": "pounce_window",
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
        "claim_tested": "ch12-setter-pounce",
        "bid_value_slice": 30,
        "note": "High-bid slice (ch12-setter-pounce-high-bid-off) deferred to Wave 2.B.2",
        "output": str(out_jsonl),
        "sha256": sha256,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(valid_snapshots)} snapshots to {out_jsonl}")
    print(f"Manifest: {OUTPUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    build_corpus()
