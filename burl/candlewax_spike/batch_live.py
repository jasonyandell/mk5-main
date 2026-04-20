"""Batch wrapper around live_runner — sequential runs across N decisions,
both modes, all into ``scratch/candlewax_spike/live/{idx}_{mode}/``.

Writes ``scratch/candlewax_spike/live/batch_summary.jsonl`` (one line per
completed run) so the dashboard can list everything without reading each
events.jsonl.

Serves the ``scratch/candlewax_spike/`` directory so the dashboard and
per-run viewer load in the browser.

Run:
    PYTHONPATH=. .venv/bin/python -u -m burl.candlewax_spike.batch_live \
        --decisions 0-9 --modes plain multimodal --port 8769
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

from burl.candlewax_spike.live_runner import (
    DEFAULT_MODEL,
    LIVE_ROOT,
    VIEWER_DST,
    VIEWER_SRC,
    _serve_in_thread,
    run_live,
)


INDEX_SRC = Path(__file__).resolve().parent / "index.html"
INDEX_DST = Path("scratch/candlewax_spike/index.html")

# Overridable via --live-dir. Kept as a module global so _append_summary can
# resolve it; set by main() before the batch starts.
_ACTIVE_ROOT: Path = LIVE_ROOT


def _parse_range(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def _install_pages() -> None:
    LIVE_ROOT.mkdir(parents=True, exist_ok=True)
    if VIEWER_SRC.exists():
        shutil.copy(VIEWER_SRC, VIEWER_DST)
    if INDEX_SRC.exists():
        shutil.copy(INDEX_SRC, INDEX_DST)


def _append_summary(line: dict) -> None:
    path = _ACTIVE_ROOT / "batch_summary.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(line) + "\n")


async def _run_one(idx: int, mode: str, *, model: str, thinking: int | None,
                   max_turns: int, budget: float) -> dict:
    print(f"\n[batch] ▶ d{idx:03d} mode={mode}", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    try:
        summary = await run_live(
            idx=idx, mode=mode, model=model,
            max_turns=max_turns, max_budget_usd=budget,
            thinking_tokens=thinking,
            live_root=_ACTIVE_ROOT,
        )
        ok = True
        err = None
    except Exception as e:
        summary = {"idx": idx, "mode": mode, "run_id": f"{idx:03d}_{mode}",
                   "final_play": -1, "events_path": ""}
        ok = False
        err = f"{type(e).__name__}: {e}"
        print(f"[batch] d{idx:03d} mode={mode} FAILED: {err}",
              file=sys.stderr, flush=True)
    elapsed = time.perf_counter() - t0

    record = {
        "idx": idx, "mode": mode, "run_id": summary["run_id"],
        "final_play": summary["final_play"],
        "elapsed_s": round(elapsed, 1),
        "ok": ok, "error": err,
        "ts": time.strftime("%H:%M:%S"),
    }
    _append_summary(record)
    print(f"[batch] ✓ d{idx:03d} mode={mode} play={summary['final_play']} "
          f"in {elapsed:.1f}s ok={ok}", file=sys.stderr, flush=True)
    return record


async def batch_main(
    *, decisions: list[int], modes: list[str], model: str,
    thinking: int | None, max_turns: int, budget: float,
) -> None:
    # Clear prior summary so the dashboard starts fresh for this batch.
    summary_path = _ACTIVE_ROOT / "batch_summary.jsonl"
    if summary_path.exists():
        summary_path.unlink()

    total = len(decisions) * len(modes)
    print(f"[batch] starting — {total} runs "
          f"({len(decisions)} decisions × {len(modes)} modes)",
          file=sys.stderr, flush=True)

    completed = 0
    for idx in decisions:
        for mode in modes:
            await _run_one(idx, mode, model=model, thinking=thinking,
                           max_turns=max_turns, budget=budget)
            completed += 1
            print(f"[batch] progress {completed}/{total}",
                  file=sys.stderr, flush=True)
    print(f"[batch] DONE — {completed}/{total} runs",
          file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=_parse_range, default="0-9",
                        help="Range like 0-9 or comma list 0,2,5.")
    parser.add_argument("--modes", nargs="+",
                        default=["plain", "multimodal"],
                        choices=["plain", "multimodal"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking", type=int, default=4000)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--budget", type=float, default=0.50)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--live-dir", type=Path, default=None,
                        help="Output root (e.g. scratch/candlewax_spike/live_haiku). "
                             "Defaults to scratch/candlewax_spike/live.")
    args = parser.parse_args()

    global _ACTIVE_ROOT
    if args.live_dir is not None:
        _ACTIVE_ROOT = Path(args.live_dir)

    _install_pages()

    if args.serve:
        _serve_in_thread(Path("scratch/candlewax_spike"), args.port)
        time.sleep(0.2)
        print(f"[batch] dashboard: http://127.0.0.1:{args.port}/index.html",
              file=sys.stderr, flush=True)

    asyncio.run(batch_main(
        decisions=args.decisions,
        modes=args.modes,
        model=args.model,
        thinking=args.thinking if args.thinking > 0 else None,
        max_turns=args.max_turns,
        budget=args.budget,
    ))

    if args.serve:
        print("[batch] server still running — Ctrl-C to stop.",
              file=sys.stderr, flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
