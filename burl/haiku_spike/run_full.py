"""T8 full reference-trace run — 30 decisions via Haiku 4.5.

Subset (team-lead's proposal, accepted):
  - First 10 rows of `burl/eval/data/move3_decisions.jsonl`
      (iter-0 / iter-1 / spike v2 / Layer 1 held-out eval set)
  - First 20 rows of `burl/eval/data/move4_decisions_n50.jsonl`
      (Move 4 STaR-rollout set)

10 eval decisions + 20 STaR decisions = 30. This straddles both comparison
surfaces so the reference traces anchor (a) the held-out grader numbers and
(b) the STaR iteration rollouts.

Hard spend cap: $1.20 (20% buffer on the $0.87 projection from T3 smoke).
Per-decision budget: $0.09.

Run:
    python -u -m burl.haiku_spike.run_full
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from pathlib import Path

from burl.eval.decision_dataset import _replay_state
from burl.haiku_spike.agent import run_decision_haiku


MOVE3 = Path("burl/eval/data/move3_decisions.jsonl")
MOVE4 = Path("burl/eval/data/move4_decisions_n50.jsonl")
OUT_DIR = Path("scratch/burl_p5_iter2_prep/haiku_traces")
SPEND_CAP_USD = 1.20
PER_DECISION_BUDGET_USD = 0.09
N_FROM_MOVE3 = 10
N_FROM_MOVE4 = 20


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


def _build_subset() -> list[tuple[str, dict]]:
    move3 = _load_first_n(MOVE3, N_FROM_MOVE3)
    move4 = _load_first_n(MOVE4, N_FROM_MOVE4)
    assert len(move3) == N_FROM_MOVE3, f"move3 loader returned {len(move3)} != {N_FROM_MOVE3}"
    assert len(move4) == N_FROM_MOVE4, f"move4 loader returned {len(move4)} != {N_FROM_MOVE4}"
    tagged: list[tuple[str, dict]] = []
    for r in move3:
        tagged.append(("move3", r))
    for r in move4:
        tagged.append(("move4", r))
    return tagged


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subset = _build_subset()
    print(f"[full] subset: {N_FROM_MOVE3} from {MOVE3.name} + "
          f"{N_FROM_MOVE4} from {MOVE4.name} = {len(subset)} decisions")
    print(f"[full] writing to {OUT_DIR}")
    print(f"[full] hard cost cap: ${SPEND_CAP_USD:.2f} total; "
          f"per-decision budget ${PER_DECISION_BUDGET_USD:.2f}")

    total_cost = 0.0
    total_wall = 0.0
    rows: list[dict] = []

    for i, (source, record) in enumerate(subset):
        state = _state_from_record(record)
        legal_plays = record.get("legal_plays", [])
        bot_play = record.get("bot_play", -1)
        bot_eq = record.get("bot_eq", 0.0)
        per_play_eq = record.get("per_play_eq", {})
        eq_gap = record.get("eq_gap", 0.0)

        remaining = max(0.0, SPEND_CAP_USD - total_cost)
        per_budget = min(PER_DECISION_BUDGET_USD, remaining)
        if per_budget <= 0.01:
            print(f"[full] cost cap ${SPEND_CAP_USD:.2f} reached after d{i-1}; "
                  f"stopping early.")
            break

        print(
            f"\n[full] d{i:02d} ({source}) seed={record['seed']} decl={record['declaration']} "
            f"narrator={record['narrator_seat']} legal={legal_plays} "
            f"bot_play={bot_play} eq_gap={eq_gap:.2f}  budget=${per_budget:.3f}"
        )

        t0 = time.perf_counter()
        try:
            result = await run_decision_haiku(
                state,
                max_turns=10,
                max_budget_usd=per_budget,
                decision_idx=i,
            )
        except Exception as e:
            print(f"[full] d{i:02d} FAILED: {type(e).__name__}: {e}")
            rows.append({
                "decision_idx": i, "source": source,
                "seed": int(record["seed"]),
                "declaration": int(record["declaration"]),
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        wall_s = time.perf_counter() - t0
        total_cost += result.cost_usd
        total_wall += wall_s

        trace_path = OUT_DIR / f"full_d{i:02d}.jsonl"
        with trace_path.open("w") as f:
            header = {
                "event": "header",
                "decision_idx": i,
                "source": source,
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
        matched_bot = (result.final_play == bot_play)
        haiku_eq = per_play_eq.get(str(result.final_play)) if result.final_play >= 0 else None

        print(
            f"[full] d{i:02d} done: final_play={result.final_play} "
            f"(bot={bot_play}, match={matched_bot}) "
            f"tools={len(tool_calls)} turns={result.num_turns} "
            f"cost=${result.cost_usd:.4f} (cum=${total_cost:.4f}) wall={wall_s:.1f}s"
        )
        if tool_names:
            hist = Counter(tool_names)
            print(f"        tool histogram: {dict(hist)}")

        rows.append({
            "decision_idx": i,
            "source": source,
            "seed": int(record["seed"]),
            "declaration": int(record["declaration"]),
            "narrator_seat": int(record["narrator_seat"]),
            "legal_plays": legal_plays,
            "bot_play": bot_play,
            "bot_eq": bot_eq,
            "eq_gap": eq_gap,
            "haiku_play": result.final_play,
            "haiku_eq": haiku_eq,
            "haiku_matched_bot": matched_bot,
            "eq_delta_vs_bot": (
                float(haiku_eq) - float(bot_eq) if haiku_eq is not None else None
            ),
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
            print(f"[full] cost cap ${SPEND_CAP_USD:.2f} hit after d{i:02d}; "
                  f"stopping before remaining decisions.")
            break

    summary_path = OUT_DIR / "full_summary.json"
    summary = _aggregate(rows, total_cost, total_wall)
    summary["source_splits"] = {"move3": N_FROM_MOVE3, "move4": N_FROM_MOVE4}
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[full] total cost: ${total_cost:.4f}  "
          f"total wall: {total_wall:.1f}s  "
          f"n_completed: {summary['n_completed']}/{len(subset)}")
    print(f"[full] summary: {summary_path}")
    if summary["n_completed"]:
        print(f"[full] bot_match_rate: {summary['bot_match_rate']:.2%}  "
              f"mean_eq_delta: {summary['mean_eq_delta']:+.3f}  "
              f"p_eq_geq_bot-eps: {summary['p_eq_geq_bot_minus_eps']:.2%}")
        print(f"[full] mean_tools/decision: {summary['mean_tool_calls']:.1f}  "
              f"mean_turns: {summary['mean_turns']:.1f}")
        print(f"[full] tool usage totals: {summary['tool_histogram_total']}")


def _aggregate(rows: list[dict], total_cost: float, total_wall: float) -> dict:
    completed = [r for r in rows if "error" not in r and r.get("haiku_play", -1) >= 0]
    if not completed:
        return {
            "n_attempted": len(rows), "n_completed": 0,
            "total_cost_usd": round(total_cost, 6),
            "total_wall_seconds": round(total_wall, 2),
            "rows": rows,
        }
    n = len(completed)
    matches = sum(1 for r in completed if r["haiku_matched_bot"])
    eq_deltas = [r["eq_delta_vs_bot"] for r in completed if r["eq_delta_vs_bot"] is not None]
    tool_counts = [r["n_tool_calls"] for r in completed]
    turns = [r["num_turns"] for r in completed]

    hist_total: Counter = Counter()
    for r in completed:
        for k, v in r["tool_histogram"].items():
            hist_total[k] += v

    eps = 0.5  # "within 0.5 E[Q] of bot" counts as parity — below dataset gap floor
    p_eq_geq = sum(
        1 for d in eq_deltas if d is not None and d >= -eps
    ) / max(1, len(eq_deltas))

    return {
        "n_attempted": len(rows),
        "n_completed": n,
        "total_cost_usd": round(total_cost, 6),
        "total_wall_seconds": round(total_wall, 2),
        "bot_match_rate": round(matches / n, 4),
        "mean_eq_delta": round(sum(eq_deltas) / max(1, len(eq_deltas)), 4),
        "p_eq_geq_bot_minus_eps": round(p_eq_geq, 4),
        "mean_tool_calls": round(sum(tool_counts) / n, 2),
        "mean_turns": round(sum(turns) / n, 2),
        "tool_histogram_total": dict(hist_total),
        "by_source": _by_source_breakdown(completed),
        "rows": rows,
    }


def _by_source_breakdown(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for src in ("move3", "move4"):
        sub = [r for r in rows if r["source"] == src]
        if not sub:
            continue
        matches = sum(1 for r in sub if r["haiku_matched_bot"])
        eq_deltas = [r["eq_delta_vs_bot"] for r in sub if r["eq_delta_vs_bot"] is not None]
        hist: Counter = Counter()
        for r in sub:
            for k, v in r["tool_histogram"].items():
                hist[k] += v
        out[src] = {
            "n": len(sub),
            "bot_match_rate": round(matches / len(sub), 4),
            "mean_eq_delta": round(
                sum(eq_deltas) / max(1, len(eq_deltas)), 4,
            ) if eq_deltas else None,
            "tool_histogram_total": dict(hist),
        }
    return out


def _default(obj):
    try:
        return dict(obj) if hasattr(obj, "__dict__") else repr(obj)
    except Exception:
        return repr(obj)


if __name__ == "__main__":
    asyncio.run(main())
