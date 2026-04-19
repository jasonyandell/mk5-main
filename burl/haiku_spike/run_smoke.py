"""T3 Haiku reference-trace smoke.

Runs the first 3 decisions of `burl/eval/data/move3_decisions.jsonl` through
the Claude Agent SDK + Haiku 4.5, writes per-decision JSONL traces to
`scratch/burl_p5_iter2_prep/haiku_traces/smoke_d{0,1,2}.jsonl`, and prints a
cost summary.

Run:
    python -u -m burl.haiku_spike.run_smoke

Hard stops at $0.25 total cost across the 3 decisions.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from burl.eval.decision_dataset import _replay_state
from burl.haiku_spike.agent import run_decision_haiku


DATASET = Path("burl/eval/data/move3_decisions.jsonl")
OUT_DIR = Path("scratch/burl_p5_iter2_prep/haiku_traces")
SPEND_CAP_USD = 0.25
N_DECISIONS = 3
PER_DECISION_BUDGET_USD = 0.09  # 3 * 0.09 = 0.27, but hard-cap enforced below


def _load_first_n(path: Path, n: int) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def _state_from_record(record: dict):
    history = [(int(p), int(d)) for p, d in record["play_history"]]
    return _replay_state(
        seed=int(record["seed"]),
        decl_id=int(record["declaration"]),
        play_history=history,
        bidder=int(record.get("bidder", 0)),
    )


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = _load_first_n(DATASET, N_DECISIONS)
    assert len(records) == N_DECISIONS, (
        f"expected {N_DECISIONS} decisions, got {len(records)} from {DATASET}"
    )

    print(f"[smoke] loaded {len(records)} decisions from {DATASET}")
    print(f"[smoke] writing traces to {OUT_DIR}")
    print(f"[smoke] hard cost cap: ${SPEND_CAP_USD:.2f} total")

    total_cost = 0.0
    summary_rows: list[dict] = []

    for i, record in enumerate(records):
        state = _state_from_record(record)
        legal_plays = record.get("legal_plays", [])
        bot_play = record.get("bot_play", -1)
        bot_eq = record.get("bot_eq", 0.0)
        per_play_eq = record.get("per_play_eq", {})
        eq_gap = record.get("eq_gap", 0.0)

        print(
            f"\n[smoke] d{i} seed={record['seed']} decl={record['declaration']} "
            f"narrator={record['narrator_seat']} legal={legal_plays} "
            f"bot_play={bot_play} eq_gap={eq_gap:.2f}"
        )

        remaining_budget = max(0.0, SPEND_CAP_USD - total_cost)
        per_budget = min(PER_DECISION_BUDGET_USD, remaining_budget)
        if per_budget <= 0.0:
            print(f"[smoke] cost cap ${SPEND_CAP_USD:.2f} reached; skipping remaining "
                  f"decisions.")
            break

        t0 = time.perf_counter()
        try:
            result = await run_decision_haiku(
                state,
                max_turns=10,
                max_budget_usd=per_budget,
                decision_idx=i,
            )
        except Exception as e:
            print(f"[smoke] d{i} FAILED: {type(e).__name__}: {e}")
            summary_rows.append({
                "decision_idx": i,
                "seed": int(record["seed"]),
                "declaration": int(record["declaration"]),
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        wall_s = time.perf_counter() - t0

        total_cost += result.cost_usd

        trace_path = OUT_DIR / f"smoke_d{i}.jsonl"
        with trace_path.open("w") as f:
            header = {
                "event": "header",
                "decision_idx": i,
                "seed": int(record["seed"]),
                "declaration": int(record["declaration"]),
                "narrator_seat": int(record["narrator_seat"]),
                "bidder": int(record.get("bidder", 0)),
                "trick_idx": int(record.get("trick_idx", 0)),
                "play_history": record["play_history"],
                "legal_plays": legal_plays,
                "per_play_eq": per_play_eq,
                "bot_play": bot_play,
                "bot_eq": bot_eq,
                "eq_gap": eq_gap,
                "wall_seconds": round(wall_s, 2),
                "model": "claude-haiku-4-5-20251001",
            }
            f.write(json.dumps(header) + "\n")
            for ev in result.events:
                f.write(json.dumps(ev, default=_default) + "\n")

        tool_calls = [e for e in result.events if e["event"] == "tool_call"]
        tool_names = [tc["tool"] for tc in tool_calls]
        chose_eq = result.final_play
        chose_eq_value = per_play_eq.get(str(chose_eq)) if chose_eq >= 0 else None

        matched_bot = (chose_eq == bot_play) if chose_eq >= 0 else False

        print(
            f"[smoke] d{i} done: final_play={chose_eq} "
            f"(bot={bot_play}, match={matched_bot}) "
            f"tools={len(tool_calls)} turns={result.num_turns} "
            f"cost=${result.cost_usd:.4f} wall={wall_s:.1f}s"
        )
        if tool_names:
            from collections import Counter
            hist = Counter(tool_names)
            print(f"        tool histogram: {dict(hist)}")

        summary_rows.append({
            "decision_idx": i,
            "seed": int(record["seed"]),
            "declaration": int(record["declaration"]),
            "narrator_seat": int(record["narrator_seat"]),
            "legal_plays": legal_plays,
            "bot_play": bot_play,
            "bot_eq": bot_eq,
            "eq_gap": eq_gap,
            "haiku_play": chose_eq,
            "haiku_eq": chose_eq_value,
            "haiku_matched_bot": matched_bot,
            "n_tool_calls": len(tool_calls),
            "tool_histogram": dict(Counter(tool_names)) if tool_names else {},
            "num_turns": result.num_turns,
            "cost_usd": round(float(result.cost_usd), 6),
            "duration_ms": result.duration_ms,
            "wall_seconds": round(wall_s, 2),
            "is_error": bool(result.is_error),
            "trace_path": str(trace_path),
        })

        if total_cost >= SPEND_CAP_USD:
            print(f"[smoke] cost cap ${SPEND_CAP_USD:.2f} hit after d{i}; "
                  f"stopping before remaining decisions.")
            break

    summary_path = OUT_DIR / "smoke_summary.json"
    summary = {
        "dataset": str(DATASET),
        "n_requested": N_DECISIONS,
        "n_completed": sum(1 for r in summary_rows if "error" not in r),
        "total_cost_usd": round(float(total_cost), 6),
        "rows": summary_rows,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[smoke] total cost across run: ${total_cost:.4f}")
    print(f"[smoke] summary: {summary_path}")


def _default(obj):
    try:
        return dict(obj) if hasattr(obj, "__dict__") else repr(obj)
    except Exception:
        return repr(obj)


if __name__ == "__main__":
    asyncio.run(main())
