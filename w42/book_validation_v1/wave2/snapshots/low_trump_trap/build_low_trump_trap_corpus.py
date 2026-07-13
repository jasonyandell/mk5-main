"""
Build low_trump_trap snapshot corpus.

Filter: bidder is following (NOT on lead), the trick was led in a non-trump
suit the bidder is void in OR the led suit is trump (forced trump play
context), the bidder holds the dominant trump (highest unplayed trump) AND
at least one other low trump. Decision must be at trick >= 1.

"Dominant trump" = the highest-ranking unplayed trump.
"Low trump" = any other trump the bidder holds (not dominant).

Rationale: This captures the shape where the bidder must decide whether to
play the dominant trump or a lower trump, testing the book claim about not
stranding low trumps.

Target: 300 snapshots.
Tests: ch04-low-trump-trap-against-count-dump.
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
    dominant_trump,
    BIDDER,
)
from forge.eq.game_tensor import GameStateTensor
from forge.oracle.declarations import PIP_TRUMP_IDS
from forge.oracle.tables import led_suit_for_lead_domino

OUTPUT_DIR = Path(__file__).parent
CORPUS_DIR = Path(PROJECT_ROOT) / "gus/data"
TARGET = 300

FILTER_DEF = {
    "role": "bidder (player 0)",
    "position": "following (trick already started)",
    "min_trick_number": 1,
    "decl_ids": list(PIP_TRUMP_IDS),
    "condition": "bidder is void in led suit OR led suit is trump, bidder holds dominant trump AND >= 1 other trump",
    "note": "Tests ch04: low trump trap - should bidder throw dominant trump?",
}


def passes_filter(dp) -> bool:
    """Return True if this is a low-trump-trap shape for the bidder."""
    state = dp.state_before

    # Bidder only, following (not on lead)
    if dp.player != BIDDER:
        return False
    n_trick = state.n_trick_plays()
    if n_trick == 0:  # on lead, not following
        return False
    if dp.trick_number < 1:
        return False

    # Pip-trump decls only
    if state.decl_id not in PIP_TRUMP_IDS:
        return False

    # Determine led suit from trick history
    trick_leader = state.leader
    # The lead domino is the first play of the current trick
    # It's stored in state.trick_lead_domino
    lead_domino = state.trick_lead_domino
    if lead_domino < 0:
        return False
    led_suit = led_suit_for_lead_domino(lead_domino, state.decl_id)

    # Bidder's suit holdings
    bidder_suits = suits_held(state, BIDDER)

    # Check: bidder void in led suit (led_suit != 7) OR led suit is trump
    if led_suit == 7:
        # Trump was led — forced to follow trump or void
        pass  # condition met
    elif led_suit not in bidder_suits or not bidder_suits[led_suit]:
        # Bidder void in led suit — can play anything
        pass  # condition met
    else:
        return False  # bidder can follow led non-trump suit

    # Bidder must hold dominant trump AND at least 1 other trump
    bidder_trumps = trumps_held(state, BIDDER)
    if len(bidder_trumps) < 2:
        return False

    dom = dominant_trump(state)
    if dom < 0:
        return False
    if dom not in bidder_trumps:
        return False  # bidder doesn't hold dominant trump

    # At least one more trump (low trump)
    other_trumps = [t for t in bidder_trumps if t != dom]
    if not other_trumps:
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
        "corpus": "low_trump_trap",
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
        "claim_tested": "ch04-low-trump-trap-against-count-dump",
        "output": str(out_jsonl),
        "sha256": sha256,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(valid_snapshots)} snapshots to {out_jsonl}")
    print(f"Manifest: {OUTPUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    build_corpus()
