"""
Unified driver: mine all Wave 2.A.2 snapshot corpora from the legacy oracle-greedy corpus.

Runs five extraction jobs in sequence:
  1. reentry_preservation_v2  (500 target)
  2. void_creation            (500 target)
  3. low_trump_trap           (300 target)
  4. pounce_window            (500 target)
  5. eighty_four_throwaway    (200 target; best-effort / expected blocker)

Filters are declarative (defined in each build_*.py).

Usage:
    python w42/book_validation_v1/wave2/snapshots/mine_legacy_snapshots.py

Output: one snapshots.jsonl + manifest.json + README.md per corpus under
    w42/book_validation_v1/wave2/snapshots/<name>/
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

PROJECT_ROOT = "/Users/jason/code/mk5-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SNAPSHOTS_DIR = Path(PROJECT_ROOT) / "w42/book_validation_v1/wave2/snapshots"

CORPORA = [
    ("reentry_preservation_v2",  "build_reentry_preservation_v2_corpus"),
    ("void_creation",            "build_void_creation_corpus"),
    ("low_trump_trap",           "build_low_trump_trap_corpus"),
    ("pounce_window",            "build_pounce_window_corpus"),
    ("eighty_four_throwaway",    "build_eighty_four_throwaway_corpus"),
]


def run_corpus(corpus_name: str, module_name: str) -> dict:
    """Import and run one corpus builder; return summary."""
    script_path = SNAPSHOTS_DIR / corpus_name / f"{module_name}.py"
    if not script_path.exists():
        return {"corpus": corpus_name, "status": "ERROR", "reason": f"{script_path} not found"}

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"Running: {corpus_name}", flush=True)
    print(f"{'='*60}", flush=True)
    mod.build_corpus()
    elapsed = time.time() - t0

    # Read manifest for summary
    manifest_path = SNAPSHOTS_DIR / corpus_name / "manifest.json"
    import json
    manifest = json.loads(manifest_path.read_text())

    return {
        "corpus": corpus_name,
        "status": "ok",
        "n_snapshots": manifest.get("n_snapshots", 0),
        "n_candidates_scanned": manifest.get("n_candidates_scanned", 0),
        "elapsed_s": round(elapsed, 1),
        "blocker": manifest.get("blocker"),
    }


def main() -> None:
    print("Wave 2.A.2 — Oracle-Greedy Legacy Snapshot Mining", flush=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(f"Corpora: {[c for c,_ in CORPORA]}", flush=True)

    summaries = []
    for corpus_name, module_name in CORPORA:
        try:
            result = run_corpus(corpus_name, module_name)
        except Exception as e:
            result = {"corpus": corpus_name, "status": "ERROR", "reason": str(e)}
        summaries.append(result)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for s in summaries:
        if s["status"] == "ERROR":
            print(f"  {s['corpus']}: ERROR — {s.get('reason', '?')}")
        else:
            blocker = f" [BLOCKER: {s['blocker'][:60]}...]" if s.get("blocker") else ""
            print(
                f"  {s['corpus']}: {s['n_snapshots']} snapshots "
                f"({s['n_candidates_scanned']} candidates, {s['elapsed_s']}s){blocker}"
            )


if __name__ == "__main__":
    main()
