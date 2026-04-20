"""Candlewax A/B: Opus-plain vs Opus-multimodal on the trick-6 decision set.

Reads the snapshot manifest ``scratch/candlewax_spike/snapshots/index.jsonl``,
runs both conditions per decision, writes per-decision trace jsonl files, and
a summary table at the end.

Run:
    # Pilot (3 decisions, both conditions)
    python -u -m burl.candlewax_spike.run_ab --n 3

    # Larger
    python -u -m burl.candlewax_spike.run_ab --n 20 --cap 10.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from burl.candlewax_spike.agent import (
    CandlewaxRunResult,
    DEFAULT_MODEL,
    run_decision_candlewax,
)
from burl.candlewax_spike.render import _domino_label
from burl.eval.decision_dataset import _replay_state


SNAPSHOT_DIR = Path("scratch/candlewax_spike/snapshots")
MANIFEST = SNAPSHOT_DIR / "index.jsonl"
TRACE_DIR = Path("scratch/candlewax_spike/traces")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"{MANIFEST} not found — run `python -m burl.candlewax_spike.snapshot` first."
        )
    rows: list[dict] = []
    with MANIFEST.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _state_from_entry(entry: dict):
    history = [(int(p), int(d)) for p, d in entry.get("play_history", [])]
    if not history:
        # Older snapshot manifests may not embed play_history; reload dataset.
        return None
    return _replay_state(
        seed=int(entry["seed"]),
        decl_id=int(entry["declaration"]),
        play_history=history,
        bidder=int(entry.get("bidder", 0)),
    )


def _save_trace(out: Path, result: CandlewaxRunResult, entry: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(json.dumps({
            "event": "meta",
            "mode": result.mode,
            "model": DEFAULT_MODEL,
            "final_play": result.final_play,
            "cost_usd": result.cost_usd,
            "num_turns": result.num_turns,
            "duration_ms": result.duration_ms,
            "usage": result.usage,
            "is_error": result.is_error,
            "decision": {
                "idx": entry["idx"],
                "seed": entry["seed"],
                "declaration": entry["declaration"],
                "narrator_seat": entry["narrator_seat"],
                "legal_plays": entry["legal_plays"],
                "bot_play": entry["bot_play"],
                "bot_eq": entry["bot_eq"],
                "eq_gap": entry["eq_gap"],
                "per_play_eq": entry["per_play_eq"],
                "per_play_summary": entry.get("per_play_summary"),
            },
        }) + "\n")
        for event in result.events:
            f.write(json.dumps(event) + "\n")


def _grade_play(final_play: int, entry: dict) -> dict:
    """Compare Opus's play against the bot baseline."""
    per_play_eq = {int(k): float(v) for k, v in entry["per_play_eq"].items()}
    bot_play = int(entry["bot_play"])
    bot_eq = float(entry["bot_eq"])
    if final_play < 0:
        return {
            "final_play": -1,
            "committed": False,
            "legal_final": False,
            "matches_bot": False,
            "opus_eq": None,
            "eq_delta_vs_bot": None,
            "k1_pass": False,
        }
    legal = final_play in set(int(p) for p in entry["legal_plays"])
    opus_eq = per_play_eq.get(final_play) if legal else None
    delta = (opus_eq - bot_eq) if opus_eq is not None else None
    k1 = (delta is not None) and (delta >= -1e-6)
    return {
        "final_play": final_play,
        "final_play_label": _domino_label(final_play),
        "committed": True,
        "legal_final": legal,
        "matches_bot": (final_play == bot_play),
        "opus_eq": opus_eq,
        "eq_delta_vs_bot": delta,
        "k1_pass": k1,
    }


async def _run_one(
    entry: dict,
    mode: str,
    *,
    model: str,
    max_budget_usd: float,
    thinking_tokens: int | None,
) -> tuple[CandlewaxRunResult, dict]:
    state = _state_from_entry(entry)
    if state is None:
        # Manifest lacks play_history — we stored a separate dataset dump
        # alongside. Load it.
        raise RuntimeError(
            f"entry idx={entry['idx']} has no play_history; re-run snapshot with "
            "include_history=True or pass --dataset explicitly."
        )
    image_path = Path(entry["png"]) if mode == "multimodal" else None
    if image_path is not None and not image_path.is_absolute():
        image_path = Path.cwd() / image_path
    result = await run_decision_candlewax(
        state,
        legal_plays=[int(p) for p in entry["legal_plays"]],
        mode=mode,
        image_path=image_path,
        model=model,
        max_budget_usd=max_budget_usd,
        decision_idx=int(entry["idx"]),
        thinking_tokens=thinking_tokens,
    )
    grade = _grade_play(result.final_play, entry)
    return result, grade


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3,
                        help="Decisions to run (from the top of the snapshot manifest).")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--modes", nargs="+",
                        default=["plain", "multimodal"],
                        choices=["plain", "multimodal"])
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--cap", type=float, default=4.0,
                        help="Total spend cap across all runs (USD).")
    parser.add_argument("--per-decision-budget", type=float, default=0.50,
                        help="Per-call budget passed to the Agent SDK.")
    parser.add_argument("--thinking-tokens", type=int, default=4000,
                        help="Extended-thinking budget. 0 disables thinking.")
    args = parser.parse_args()

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    entries = manifest[args.start : args.start + args.n]
    log(f"[ab] n={len(entries)} modes={args.modes} model={args.model}")
    log(f"[ab] spend cap ${args.cap:.2f}; per-call ${args.per_decision_budget:.2f}")

    thinking = args.thinking_tokens if args.thinking_tokens > 0 else None

    summary_rows: list[dict] = []
    total_cost = 0.0

    for entry in entries:
        idx = int(entry["idx"])
        for mode in args.modes:
            if total_cost >= args.cap:
                log(f"[ab] spend cap reached (${total_cost:.3f} >= ${args.cap:.2f}); stopping.")
                break
            log(f"\n[ab] d{idx:03d} mode={mode} seed={entry['seed']} "
                f"decl={entry['declaration']} legal={entry['legal_plays']} "
                f"bot_play={entry['bot_play']} eq_gap={entry['eq_gap']:.2f}")
            t0 = time.perf_counter()
            try:
                result, grade = await _run_one(
                    entry, mode,
                    model=args.model,
                    max_budget_usd=args.per_decision_budget,
                    thinking_tokens=thinking,
                )
            except Exception as e:
                log(f"[ab] d{idx:03d} mode={mode} FAILED: {e}")
                continue
            elapsed = time.perf_counter() - t0
            total_cost += result.cost_usd

            trace_path = TRACE_DIR / f"d{idx:03d}_{mode}.jsonl"
            _save_trace(trace_path, result, entry)

            summary_rows.append({
                "idx": idx,
                "mode": mode,
                "seed": entry["seed"],
                "declaration": entry["declaration"],
                **grade,
                "cost_usd": result.cost_usd,
                "num_turns": result.num_turns,
                "duration_s": round(elapsed, 1),
                "is_error": result.is_error,
                "trace": str(trace_path),
            })

            log(
                f"[ab] d{idx:03d} mode={mode}  committed={grade['committed']}  "
                f"play={grade.get('final_play_label', '—')}  "
                f"bot_match={grade['matches_bot']}  "
                f"ΔE[Q]={grade['eq_delta_vs_bot']}  "
                f"K1={grade['k1_pass']}  "
                f"cost=${result.cost_usd:.3f}  elapsed={elapsed:.1f}s"
            )

    # Write summary table + markdown comparison.
    summary_path = TRACE_DIR / "summary.jsonl"
    with summary_path.open("w") as f:
        for r in summary_rows:
            f.write(json.dumps(r) + "\n")

    # Build comparison rows: pair (idx, plain) with (idx, multimodal) if both ran.
    by_idx_mode = {(r["idx"], r["mode"]): r for r in summary_rows}
    idxs = sorted({r["idx"] for r in summary_rows})
    cmp_path = TRACE_DIR / "comparison.md"
    lines = [
        "# Candlewax A/B pilot",
        "",
        f"Model: `{args.model}`  ·  total_cost=${total_cost:.3f}",
        "",
        "| idx | seed | decl | bot | plain | ΔE[Q]p | K1p | multimodal | ΔE[Q]m | K1m |",
        "|-----|------|------|-----|-------|--------|-----|------------|--------|-----|",
    ]
    for idx in idxs:
        p = by_idx_mode.get((idx, "plain"))
        m = by_idx_mode.get((idx, "multimodal"))
        ref = p or m
        bot_label = _domino_label(int(ref["seed"]))  # placeholder; fixed below
        # Fetch the bot label from the manifest.
        entry = next(e for e in manifest if int(e["idx"]) == idx)
        bot_label = _domino_label(int(entry["bot_play"]))
        p_play = p.get("final_play_label", "—") if p else "—"
        m_play = m.get("final_play_label", "—") if m else "—"
        p_delta = f"{p['eq_delta_vs_bot']:+.2f}" if p and p["eq_delta_vs_bot"] is not None else "—"
        m_delta = f"{m['eq_delta_vs_bot']:+.2f}" if m and m["eq_delta_vs_bot"] is not None else "—"
        p_k1 = "✓" if p and p["k1_pass"] else "·"
        m_k1 = "✓" if m and m["k1_pass"] else "·"
        lines.append(
            f"| {idx} | {ref['seed']} | {ref['declaration']} | {bot_label} | "
            f"{p_play} | {p_delta} | {p_k1} | {m_play} | {m_delta} | {m_k1} |"
        )
    cmp_path.write_text("\n".join(lines) + "\n")
    log(f"\n[ab] summary: {summary_path}")
    log(f"[ab] comparison: {cmp_path}")
    log(f"[ab] total cost: ${total_cost:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
