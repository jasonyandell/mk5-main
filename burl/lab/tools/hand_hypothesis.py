"""Targeted hidden-hand hypothesis tool for Burl.

``simulate_hand_impact`` is the ergonomic bridge Burl asked for in the
microscope: ask one concrete hidden-hand question, get its plausibility from
Gus's belief head, and quantify the conditional outcome shift for a candidate
play.
"""

from __future__ import annotations

import re
from typing import Any

from burl.lab.core.tool import ToolResult, ToolSpec

_DOMINO_LABELS = tuple(
    f"{high}-{low}" for high in range(7) for low in range(high + 1)
)


_REL_TO_OFFSET = {
    "left_opp": 1,
    "left": 1,
    "l": 1,
    "partner": 2,
    "p": 2,
    "right_opp": 3,
    "right": 3,
    "r": 3,
}
_OFFSET_TO_REL = {
    0: "self",
    1: "left_opp",
    2: "partner",
    3: "right_opp",
}
_REL_DISPLAY = {
    "self": "SELF",
    "left_opp": "LEFT OPP",
    "partner": "PARTNER",
    "right_opp": "RIGHT OPP",
}


def _label(domino_id: int) -> str:
    return f"{int(domino_id)}({_DOMINO_LABELS[int(domino_id)]})"


def _extract_args(args: dict[str, Any]) -> tuple[int, str | int, int]:
    """Accept the clean schema plus Burl's invented example aliases."""
    play_raw = args.get("play_id", args.get("play"))
    seat_raw = args.get("seat", args.get("opponent_focus", args.get("player")))
    holds_raw = args.get("holds", args.get("domino_id"))

    hypothetical = args.get("hypothetical_hand_to_test")
    if holds_raw is None and isinstance(hypothetical, dict):
        holds_raw = hypothetical.get(
            "holds", hypothetical.get("domino_id", hypothetical.get("hand_id"))
        )

    if play_raw is None:
        raise ValueError("missing play_id")
    if seat_raw is None:
        raise ValueError("missing seat")
    if holds_raw is None:
        raise ValueError("missing holds")

    return int(play_raw), seat_raw, int(holds_raw)


def _parse_seat(raw: str | int, me_abs: int) -> int:
    if isinstance(raw, int):
        seat = raw
    else:
        text = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
        if text in _REL_TO_OFFSET:
            seat = (int(me_abs) + _REL_TO_OFFSET[text]) % 4
        else:
            match = re.search(r"(?:seat_?|^)([0-3])$", text)
            if not match:
                match = re.search(r"([0-3])", text)
            if match is None:
                raise ValueError(
                    "seat must be left_opp, partner, right_opp, or an absolute seat 0-3"
                )
            seat = int(match.group(1))
    if seat < 0 or seat > 3:
        raise ValueError("seat must be 0, 1, 2, or 3")
    return seat


def _relative_seat(abs_seat: int, me_abs: int) -> str:
    return _OFFSET_TO_REL[(int(abs_seat) - int(me_abs)) % 4]


def _belief_probability(
    ctx: Any, abs_seat: int, holds: int
) -> tuple[float | None, dict[str, Any]]:
    """Return Gus marginal P(abs_seat holds domino), without peeking at hidden hands."""
    rel = _relative_seat(abs_seat, ctx.me_abs)
    known = _known_probability(ctx, abs_seat, holds)
    if known is not None:
        return known, {"source": "public_state", "relative_seat": rel}

    from burl.wax_museum.tools import tool_belief_trajectory

    payload = tool_belief_trajectory(ctx)
    structured = dict(payload.get("structured", {}))
    posterior = structured.get("posterior_by_domino", [])
    key = f"p_{rel}"
    for row in posterior:
        if int(row.get("domino_id", -1)) == int(holds):
            prob = row.get(key)
            return (
                float(prob) if prob is not None else None,
                {
                    "source": "gus_belief_trajectory",
                    "relative_seat": rel,
                    "posterior_row": row,
                },
            )
    return None, {
        "source": "gus_belief_trajectory",
        "relative_seat": rel,
        "posterior_row": None,
    }


def _known_probability(ctx: Any, abs_seat: int, holds: int) -> float | None:
    state = ctx.game_state
    played = {int(d) for d in getattr(state, "played", set())}
    if int(holds) in played:
        return 0.0

    # The current player's hand is public to Burl. Opponent hands in the state
    # object are not public; do not inspect them here.
    me_abs = int(ctx.me_abs)
    if int(abs_seat) == me_abs:
        my_hand = {int(d) for d in state.hands[me_abs] if int(d) not in played}
        return 1.0 if int(holds) in my_hand else 0.0
    my_hand = {int(d) for d in state.hands[me_abs] if int(d) not in played}
    if int(holds) in my_hand:
        return 0.0
    return None


def _summarize_dist(dist: Any) -> dict[str, Any]:
    return {
        "mean": round(float(dist.mean), 2),
        "p_make": round(float(dist.p_make), 3),
        "stdev": round(float(dist.stdev), 2),
        "shape": str(dist.distribution_shape),
        "n_samples": int(dist.n_samples),
    }


def _prob_phrase(prob: float | None) -> str:
    if prob is None:
        return "unknown"
    if prob >= 0.60:
        tag = "strong"
    elif prob >= 0.40:
        tag = "medium"
    elif prob >= 0.25:
        tag = "soft"
    else:
        tag = "low"
    return f"{prob * 100:.0f}% ({tag})"


def _render_prose(
    *,
    play: int,
    abs_seat: int,
    rel: str,
    holds: int,
    probability: float | None,
    baseline: Any,
    conditional: Any,
) -> str:
    base_mean = float(baseline.mean)
    cond_mean = float(conditional.mean)
    shift = cond_mean - base_mean
    weighted = probability * shift if probability is not None else None
    direction = "helps" if shift > 1.0 else "hurts" if shift < -1.0 else "barely moves"

    lines: list[str] = []
    lines.append(
        f"HAND HYPOTHESIS: {_REL_DISPLAY[rel]} (seat {abs_seat}) holds {_label(holds)}"
    )
    lines.append("")
    lines.append(f"Plausibility: {_prob_phrase(probability)}")
    lines.append("")
    lines.append(f"Impact if true for play {_label(play)}:")
    lines.append(
        f"  baseline:    Q = {base_mean:+.1f}, p_make = {float(baseline.p_make):.2f}, "
        f"shape = {baseline.distribution_shape}"
    )
    lines.append(
        f"  conditional: Q = {cond_mean:+.1f}, p_make = {float(conditional.p_make):.2f}, "
        f"shape = {conditional.distribution_shape}"
    )
    lines.append(f"  shift:       {shift:+.1f} Q — this hypothesis {direction} this play")
    if weighted is not None:
        lines.append(f"  probability-weighted shift: {weighted:+.1f} Q")
    lines.append("")
    lines.append(
        "Use this as a targeted follow-up to play_brief catalysts: it does not say "
        "what to play by itself; compare the same hypothesis across candidate plays "
        "or test the named upside/downside catalysts."
    )
    return "\n".join(lines)


def _error(message: str, *, args: dict[str, Any]) -> ToolResult:
    return ToolResult(
        evidence={
            "prose": f"ERROR: simulate_hand_impact failed: {message}",
            "structured": {"error": message, "args": args},
        },
        next_tools=(),
    )


def _impl(ctx: Any, args: dict[str, Any]) -> ToolResult:
    try:
        play, seat_raw, holds = _extract_args(args)
        if play < 0 or play >= 28 or holds < 0 or holds >= 28:
            raise ValueError("play_id and holds must be domino_ids 0-27")
        abs_seat = _parse_seat(seat_raw, ctx.me_abs)
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), args=args)

    rel = _relative_seat(abs_seat, ctx.me_abs)
    if rel == "self":
        return _error(
            "simulate_hand_impact is for opponent/partner hidden-hand hypotheses, not self",
            args=args,
        )

    try:
        probability, probability_detail = _belief_probability(ctx, abs_seat, holds)
    except Exception as exc:  # noqa: BLE001
        probability = None
        probability_detail = {
            "source": "belief_error",
            "error": str(exc),
            "relative_seat": rel,
        }

    try:
        from burl.tools.eq_distribution import ConditionUnreachable, conditional_outcome

        baseline = ctx.get_or_build(play).dist
        conditional = conditional_outcome(
            ctx.game_state,
            play=play,
            assumption={"player": abs_seat, "holds": holds},
            n_samples=20,
            max_sampling_tries=100,
            oracle=ctx.oracle,
        )
    except ConditionUnreachable as exc:
        return _error(f"hypothesis unreachable: {exc}", args=args)
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), args=args)

    baseline_summary = _summarize_dist(baseline)
    conditional_summary = _summarize_dist(conditional)
    shift = conditional_summary["mean"] - baseline_summary["mean"]
    structured = {
        "play_id": play,
        "hypothesis": {
            "seat_abs": abs_seat,
            "seat_relative": rel,
            "holds": holds,
            "holds_label": _DOMINO_LABELS[holds],
            "probability": None if probability is None else round(float(probability), 4),
            "probability_detail": probability_detail,
        },
        "baseline": baseline_summary,
        "conditional": conditional_summary,
        "shift_in_mean": round(float(shift), 2),
        "probability_weighted_shift": (
            None if probability is None else round(float(probability) * float(shift), 2)
        ),
    }
    prose = _render_prose(
        play=play,
        abs_seat=abs_seat,
        rel=rel,
        holds=holds,
        probability=probability,
        baseline=baseline,
        conditional=conditional,
    )
    return ToolResult(evidence={"prose": prose, "structured": structured}, next_tools=())


SIMULATE_HAND_IMPACT = ToolSpec(
    name="simulate_hand_impact",
    description=(
        "Targeted hidden-hand hypothesis test. Given a candidate play and a concrete "
        "seat-holds-domino hypothesis, returns Gus's marginal probability for that "
        "hypothesis and the conditional outcome shift if it is true."
    ),
    params={
        "type": "object",
        "properties": {
            "play_id": {
                "type": "integer",
                "description": "domino_id of the candidate play to test under the hypothesis",
            },
            "seat": {
                "type": "string",
                "description": "left_opp, partner, right_opp, or absolute seat like seat2",
            },
            "holds": {
                "type": "integer",
                "description": "domino_id the named seat is hypothesized to hold",
            },
        },
        "required": ["play_id", "seat", "holds"],
        "additionalProperties": False,
    },
    example="simulate_hand_impact(play_id=19, seat='left_opp', holds=12)",
    protocol_role="candidate_eval",
    protocol_phrase=(
        "Call `simulate_hand_impact(play_id=X, seat='left_opp|partner|right_opp', holds=Y)` "
        "when `play_brief` names a catalyst or you need to test a specific hidden-hand hypothesis."
    ),
    impl=_impl,
    requires_context=True,
)


__all__ = ["SIMULATE_HAND_IMPACT"]
