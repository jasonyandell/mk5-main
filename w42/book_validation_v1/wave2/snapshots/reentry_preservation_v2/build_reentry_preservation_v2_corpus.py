"""
Build reentry_preservation_v2 snapshot corpus.

Filter: bidder's turn, bidder has exactly 1 trump remaining,
bidder has >= 2 distinct off suits with >= 1 tile each,
decision occurs at trick >= 2 (mid-hand, not opening).

Target: 500 snapshots minimum.
Tests: ch03-reentry-preservation (v2 = oracle-greedy source).
"""
from __future__ import annotations

import glob
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
    distinct_off_suits_held,
    BIDDER,
)
from forge.eq.game_tensor import GameStateTensor
from forge.oracle.declarations import PIP_TRUMP_IDS

OUTPUT_DIR = Path(__file__).parent
CORPUS_DIR = Path(PROJECT_ROOT) / "gus/data"
TARGET = 500

# --------------------------------------------------------------------------
# Filter
# --------------------------------------------------------------------------

FILTER_DEF = {
    "bidder_turn": True,
    "n_trumps_remaining_exact": 1,
    "min_distinct_off_suits": 2,
    "min_trick_number": 2,
    "decl_ids": list(PIP_TRUMP_IDS),  # pip-trump decls only
    "note": "Reentry preservation: 1 trump left, >=2 off suits, trick>=2, bidder on lead",
}


def passes_filter(dp) -> bool:
    """Return True if decision point matches reentry_preservation shape."""
    # Pip-trump decls only (trump concept doesn't apply to doubles/notrump)
    if dp.state_before.decl_id not in PIP_TRUMP_IDS:
        return False
    # Bidder's turn only
    if dp.player != BIDDER:
        return False
    # Trick >= 2 (0-indexed), so at least 2 tricks already played
    if dp.trick_number < 2:
        return False
    # Exactly 1 trump remaining
    trumps = trumps_held(dp.state_before, BIDDER)
    if len(trumps) != 1:
        return False
    # >= 2 distinct off suits with >= 1 tile each
    off_suits = distinct_off_suits_held(dp.state_before, BIDDER)
    if len(off_suits) < 2:
        return False
    return True


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

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
        chunk_candidates = 0
        chunk_found = 0
        try:
            for dp in iter_decisions(str(chunk_path)):
                candidates += 1
                chunk_candidates += 1
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
            f"  {chunk_name}: {chunk_candidates} decisions, {chunk_found} matching "
            f"| total so far: {len(snapshots)}",
            flush=True,
        )

    print(f"\nTotal candidates scanned: {candidates}")
    print(f"Snapshots collected: {len(snapshots)}")

    # Round-trip validation via GameStateTensor.from_snapshot
    valid_count = 0
    invalid_count = 0
    print("Validating snapshots via GameStateTensor.from_snapshot ...", flush=True)
    # Use CPU since we don't need GPU for validation
    for i, snap in enumerate(snapshots):
        snap_clean = {k: v for k, v in snap.items() if not k.startswith("_")}
        try:
            GameStateTensor.from_snapshot([snap_clean], device="cpu")
            valid_count += 1
        except Exception as e:
            invalid_count += 1
            if invalid_count <= 3:
                print(f"  snapshot[{i}] failed: {e}")
    print(f"Validation: {valid_count} valid, {invalid_count} invalid out of {len(snapshots)}")

    # Keep only valid snapshots
    valid_snapshots = []
    for snap in snapshots:
        snap_clean = {k: v for k, v in snap.items() if not k.startswith("_")}
        try:
            GameStateTensor.from_snapshot([snap_clean], device="cpu")
            valid_snapshots.append(snap)
        except Exception:
            pass

    # Write jsonl
    out_jsonl = OUTPUT_DIR / "snapshots.jsonl"
    with open(out_jsonl, "w") as f:
        for snap in valid_snapshots:
            f.write(json.dumps(snap) + "\n")

    sha256 = hashlib.sha256(out_jsonl.read_bytes()).hexdigest()

    manifest = {
        "corpus": "reentry_preservation_v2",
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
        "claim_tested": "ch03-reentry-preservation",
        "output": str(out_jsonl),
        "sha256": sha256,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(valid_snapshots)} snapshots to {out_jsonl}")
    print(f"SHA256: {sha256}")
    print(f"Manifest: {OUTPUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    build_corpus()
