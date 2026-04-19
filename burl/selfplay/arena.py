"""4-Opus self-play arena — one full Texas 42 hand, fresh SDK session per turn.

Run:
    python -u -m burl.selfplay.arena --seed 900010

The seed's declaration is looked up from ``burl/eval/data/move4_decisions_n50.jsonl``
unless ``--declaration`` is provided. Game state starts at trick 0 (no plays);
bidding is skipped — we simply honor the seed's declaration as trump.

Every turn spawns a fresh Claude Agent SDK session at ``claude-opus-4-7`` with
the iter-1 prompt shape (``enable_primer=True``, ``enable_rules_tools=False``)
via ``burl.haiku_spike.agent.run_decision_haiku``. No session reuse between
seats — avoids shared-context concerns.

Outputs (per game):
- ``scratch/burl_p5_iter2_prep/arena_traces/seed_{SEED}.jsonl`` — structured
  per-turn log (header + one JSON per event).
- ``scratch/burl_p5_iter2_prep/arena_traces/seed_{SEED}.md`` — human
  narration: trick-by-trick play + short rationale snippet + winner/points.

Pure exploration. Hard cost cap stops the game if total spend blows past
``--max-total-budget-usd``.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from forge.zeb.game import apply_action, is_terminal

from burl.eval.decision_dataset import _slot_for, _zeb_state_from_deal
from burl.haiku_spike.agent import run_decision_haiku
from burl.tools import engine as engine_tools


DEFAULT_DATASET = Path("burl/eval/data/move4_decisions_n50.jsonl")
DEFAULT_OUT_DIR = Path("scratch/burl_p5_iter2_prep/arena_traces")
DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_SEED = 900010

# Budget: task target ~$7/game at $0.25/decision × 28. Leave headroom per
# decision without making any single chatty turn blow the whole budget.
DEFAULT_PER_DECISION_BUDGET_USD = 0.40
DEFAULT_TOTAL_BUDGET_USD = 10.0
DEFAULT_MAX_TURNS = 10


def _domino_label(domino_id: int) -> str:
    """Return ``a|b`` where a is the higher pip."""
    from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW

    return f"{int(DOMINO_HIGH[domino_id])}|{int(DOMINO_LOW[domino_id])}"


def _fmt_dom(domino_id: int) -> str:
    return f"{domino_id}({_domino_label(domino_id)})"


def _lookup_declaration(seed: int, dataset: Path) -> int | None:
    """Return the first declaration seen for ``seed`` in the move4 dataset."""
    if not dataset.exists():
        return None
    with dataset.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if int(rec.get("seed", -1)) == seed:
                return int(rec["declaration"])
    return None


def _first_assistant_text(events: list[dict]) -> str:
    """Pick the first non-empty assistant_text block (agent's own narration)."""
    for e in events:
        if e.get("event") == "assistant_text":
            txt = str(e.get("content", "")).strip()
            if txt:
                return txt
    return ""


def _rationale_snippet(events: list[dict], max_len: int = 220) -> str:
    txt = _first_assistant_text(events)
    if not txt:
        return "(no narration)"
    first_line = txt.splitlines()[0].strip()
    if len(first_line) <= max_len:
        return first_line
    return first_line[: max_len - 1] + "…"


def _tool_histogram(events: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in events:
        if e.get("event") == "tool_call":
            name = str(e.get("tool", ""))
            # Strip the MCP prefix for readability.
            if name.startswith("mcp__burl__"):
                name = name[len("mcp__burl__") :]
            counts[name] = counts.get(name, 0) + 1
    return counts


@dataclasses.dataclass
class ArenaTurn:
    turn_idx: int
    seat: int
    trick_idx: int
    position_in_trick: int
    hand_before: list[int]
    legal_before: list[int]
    trick_so_far: list[int]
    final_play: int
    rationale: str
    tool_histogram: dict[str, int]
    num_turns: int
    cost_usd: float
    duration_ms: int
    is_error: bool


def _legal_domino_ids(state) -> list[int]:
    """Legal plays as domino_ids for the current player, via engine tools."""
    hand = state.hands[(state.trick_leader + len(state.current_trick)) % 4]
    legal = []
    for d in hand:
        if d in state.played:
            continue
        ok, _ = engine_tools.is_legal(state, int(d))
        if ok:
            legal.append(int(d))
    return legal


async def run_arena_game(
    *,
    seed: int,
    decl_id: int,
    bidder: int,
    model: str,
    max_turns: int,
    per_decision_budget_usd: float,
    total_budget_usd: float,
    out_dir: Path,
) -> dict[str, Any]:
    """Run one full 28-turn Texas 42 hand; return a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"seed_{seed}.jsonl"
    md_path = out_dir / f"seed_{seed}.md"

    state = _zeb_state_from_deal(seed, decl_id, bidder=bidder)
    decl_name = engine_tools.trump_declared(state)

    hands_at_start = [[int(d) for d in h] for h in state.hands]

    md: list[str] = [
        f"# Arena game — seed {seed}",
        "",
        f"- Model: `{model}`",
        f"- Declaration (trump): `{decl_name}` (decl_id {decl_id})",
        f"- Bidder: seat {bidder}  (team {bidder % 2})",
        f"- Per-decision budget cap: ${per_decision_budget_usd:.2f}",
        f"- Total budget cap: ${total_budget_usd:.2f}",
        "",
        "## Initial hands",
        "",
    ]
    for seat in range(4):
        md.append(
            f"- Seat {seat} (team {seat % 2}): "
            + ", ".join(_fmt_dom(d) for d in hands_at_start[seat])
        )
    md.append("")

    turns: list[ArenaTurn] = []
    total_cost = 0.0
    stopped_reason = "completed"
    t_game_start = time.perf_counter()

    with jsonl_path.open("w") as jf:
        jf.write(
            json.dumps(
                {
                    "event": "header",
                    "seed": seed,
                    "declaration": decl_id,
                    "declaration_name": decl_name,
                    "bidder": bidder,
                    "model": model,
                    "max_turns_per_decision": max_turns,
                    "per_decision_budget_usd": per_decision_budget_usd,
                    "total_budget_usd": total_budget_usd,
                    "hands_at_start": hands_at_start,
                    "bid": int(state.bid_state.high_bid),
                }
            )
            + "\n"
        )
        jf.flush()

        turn_idx = 0
        current_trick_md: list[str] = []
        trick_idx = 0
        points_before_trick = tuple(state.team_points)

        while not is_terminal(state):
            turn_idx += 1
            seat = (state.trick_leader + len(state.current_trick)) % 4
            position_in_trick = len(state.current_trick) + 1
            hand_before = [int(d) for d in state.hands[seat] if d not in state.played]
            legal_before = _legal_domino_ids(state)
            trick_so_far = [int(d) for d in state.current_trick]

            if position_in_trick == 1:
                trick_idx = len(state.play_history) // 4 + 1
                md.append(f"## Trick {trick_idx}")
                md.append("")
                md.append(f"- Leader: seat {seat}")
                points_before_trick = tuple(state.team_points)
                current_trick_md = []

            remaining_budget = max(0.0, total_budget_usd - total_cost)
            per_budget = min(per_decision_budget_usd, remaining_budget)
            if per_budget <= 0.0:
                stopped_reason = "total_budget_exhausted"
                break

            print(
                f"[arena] turn {turn_idx:02d}/28 trick {trick_idx} "
                f"pos {position_in_trick}/4 seat {seat} "
                f"hand={hand_before} legal={legal_before} "
                f"spent=${total_cost:.2f}",
                flush=True,
            )

            t0 = time.perf_counter()
            try:
                result = await run_decision_haiku(
                    state,
                    model=model,
                    max_turns=max_turns,
                    max_budget_usd=per_budget,
                    decision_idx=turn_idx,
                )
            except Exception as e:
                stopped_reason = f"sdk_error: {type(e).__name__}: {e}"
                print(f"[arena] FAILED at turn {turn_idx}: {stopped_reason}", flush=True)
                jf.write(
                    json.dumps(
                        {
                            "event": "turn_error",
                            "turn_idx": turn_idx,
                            "seat": seat,
                            "error": stopped_reason,
                        }
                    )
                    + "\n"
                )
                break
            wall_s = time.perf_counter() - t0
            total_cost += float(result.cost_usd)

            rationale = _rationale_snippet(result.events)
            tool_hist = _tool_histogram(result.events)

            final_play = int(result.final_play) if result.final_play >= 0 else -1

            if final_play < 0:
                stopped_reason = f"no_commit at turn {turn_idx} seat {seat}"
                print(f"[arena] {stopped_reason}", flush=True)
                jf.write(
                    json.dumps(
                        {
                            "event": "turn_no_commit",
                            "turn_idx": turn_idx,
                            "seat": seat,
                            "cost_usd": float(result.cost_usd),
                            "events": result.events,
                        }
                    )
                    + "\n"
                )
                break

            # Apply the play.
            slot = _slot_for(state.hands[seat], final_play)
            prev_points = tuple(state.team_points)
            prev_trick_len_before_apply = len(state.current_trick)
            state = apply_action(state, slot)

            turn_record = ArenaTurn(
                turn_idx=turn_idx,
                seat=seat,
                trick_idx=trick_idx,
                position_in_trick=position_in_trick,
                hand_before=hand_before,
                legal_before=legal_before,
                trick_so_far=trick_so_far,
                final_play=final_play,
                rationale=rationale,
                tool_histogram=tool_hist,
                num_turns=int(result.num_turns),
                cost_usd=float(result.cost_usd),
                duration_ms=int(result.duration_ms),
                is_error=bool(result.is_error),
            )
            turns.append(turn_record)

            jf.write(
                json.dumps(
                    {
                        "event": "turn",
                        "turn_idx": turn_idx,
                        "seat": seat,
                        "trick_idx": trick_idx,
                        "position_in_trick": position_in_trick,
                        "hand_before": hand_before,
                        "legal_before": legal_before,
                        "trick_so_far": trick_so_far,
                        "final_play": final_play,
                        "rationale": rationale,
                        "tool_histogram": tool_hist,
                        "num_turns": result.num_turns,
                        "cost_usd": float(result.cost_usd),
                        "duration_ms": result.duration_ms,
                        "wall_seconds": round(wall_s, 2),
                        "is_error": bool(result.is_error),
                        "events": result.events,
                    },
                    default=_default,
                )
                + "\n"
            )
            jf.flush()

            current_trick_md.append(
                f"  - seat {seat} plays {_fmt_dom(final_play)} — "
                f"tools={tool_hist or '{}'}, turns={result.num_turns}, "
                f"${result.cost_usd:.3f}"
            )
            if rationale and rationale != "(no narration)":
                current_trick_md.append(f"    > _{rationale}_")

            # If that play completed a trick, the state transitioned: the new
            # ``trick_leader`` is the winner, ``current_trick`` is empty, and
            # ``team_points`` is updated.
            if prev_trick_len_before_apply + 1 == 4:
                winner_seat = state.trick_leader
                points_this_trick = (
                    (state.team_points[0] - prev_points[0])
                    + (state.team_points[1] - prev_points[1])
                )
                md.extend(current_trick_md)
                md.append("")
                md.append(
                    f"- **Trick {trick_idx} winner: seat {winner_seat} "
                    f"(team {winner_seat % 2})** — {points_this_trick} pts. "
                    f"Running score: team0 {state.team_points[0]}, "
                    f"team1 {state.team_points[1]}"
                )
                md.append("")
                current_trick_md = []

        # If the loop exited mid-trick (budget/error/no_commit), flush what we
        # have so the MD reflects partial play.
        if current_trick_md:
            md.extend(current_trick_md)
            md.append("")
            md.append(f"- _(trick {trick_idx} did not complete: {stopped_reason})_")
            md.append("")

    wall_game = time.perf_counter() - t_game_start

    terminal = is_terminal(state)
    team0, team1 = state.team_points
    bid = int(state.bid_state.high_bid)
    bidder_team = state.bidder % 2
    bidder_points = team0 if bidder_team == 0 else team1
    made_bid = terminal and bidder_points >= bid

    md.append("## Game result")
    md.append("")
    if terminal:
        md.append(f"- Final: team0 {team0}, team1 {team1}")
        md.append(
            f"- Bidder team (team {bidder_team}) bid {bid}, earned {bidder_points} — "
            f"{'MADE bid' if made_bid else 'set'}"
        )
    else:
        md.append(f"- **Game did not complete: {stopped_reason}**")
        md.append(f"- Partial score: team0 {team0}, team1 {team1}")
    md.append("")
    md.append(f"- Total cost: ${total_cost:.3f}")
    md.append(f"- Wall time: {wall_game:.1f}s  ({len(turns)} turns played)")
    md.append("")

    # Aggregate tool histogram.
    agg: dict[str, int] = {}
    for t in turns:
        for k, v in t.tool_histogram.items():
            agg[k] = agg.get(k, 0) + v
    if agg:
        md.append("## Aggregate tool histogram")
        md.append("")
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
            md.append(f"- `{k}`: {v}")
        md.append("")

    md_path.write_text("\n".join(md))

    # Append the summary footer event to JSONL.
    with jsonl_path.open("a") as jf:
        jf.write(
            json.dumps(
                {
                    "event": "game_result",
                    "terminal": terminal,
                    "team_points": [team0, team1],
                    "bidder": int(state.bidder),
                    "bidder_team": int(bidder_team),
                    "bid": bid,
                    "made_bid": bool(made_bid) if terminal else None,
                    "n_turns_played": len(turns),
                    "total_cost_usd": round(float(total_cost), 6),
                    "wall_seconds": round(wall_game, 2),
                    "stopped_reason": stopped_reason,
                    "aggregate_tool_histogram": agg,
                }
            )
            + "\n"
        )

    return {
        "seed": seed,
        "declaration": decl_id,
        "declaration_name": decl_name,
        "terminal": terminal,
        "team_points": [team0, team1],
        "bidder_team": bidder_team,
        "bid": bid,
        "made_bid": bool(made_bid) if terminal else None,
        "total_cost_usd": round(float(total_cost), 6),
        "n_turns_played": len(turns),
        "wall_seconds": round(wall_game, 2),
        "stopped_reason": stopped_reason,
        "jsonl_path": str(jsonl_path),
        "md_path": str(md_path),
    }


def _default(obj: Any) -> Any:
    try:
        return dict(obj) if hasattr(obj, "__dict__") else repr(obj)
    except Exception:
        return repr(obj)


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--declaration",
        type=int,
        default=None,
        help="decl_id 0..9; default looks up first entry for --seed in the move4 dataset",
    )
    ap.add_argument("--bidder", type=int, default=0)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    ap.add_argument(
        "--per-decision-budget-usd",
        type=float,
        default=DEFAULT_PER_DECISION_BUDGET_USD,
    )
    ap.add_argument(
        "--total-budget-usd", type=float, default=DEFAULT_TOTAL_BUDGET_USD
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Used only to look up a default declaration for --seed",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return ap


async def _amain() -> None:
    args = _build_argparser().parse_args()

    decl = args.declaration
    if decl is None:
        decl = _lookup_declaration(int(args.seed), args.dataset)
        if decl is None:
            print(
                f"[arena] no declaration found for seed {args.seed} in "
                f"{args.dataset}; defaulting to decl_id=0"
            )
            decl = 0

    print(
        f"[arena] seed={args.seed} decl_id={decl} bidder={args.bidder} "
        f"model={args.model} per_budget=${args.per_decision_budget_usd:.2f} "
        f"total_budget=${args.total_budget_usd:.2f}"
    )

    summary = await run_arena_game(
        seed=int(args.seed),
        decl_id=int(decl),
        bidder=int(args.bidder),
        model=str(args.model),
        max_turns=int(args.max_turns),
        per_decision_budget_usd=float(args.per_decision_budget_usd),
        total_budget_usd=float(args.total_budget_usd),
        out_dir=Path(args.out_dir),
    )

    print("\n[arena] game complete")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(_amain())
