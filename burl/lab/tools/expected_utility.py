"""Integrated expected-utility ranking tool for Burl.

``calculate_expected_utility`` is the high-level optimization tool Burl asked
for after using ``simulate_hand_impact``: instead of testing one catalyst at a
time, rank the legal candidates by their integrated outcome distribution.
"""

from __future__ import annotations

import math
from typing import Any

from burl.lab.core.tool import ToolResult, ToolSpec

_DOMINO_LABELS = tuple(
    f"{high}-{low}" for high in range(7) for low in range(high + 1)
)


def _label(domino_id: int) -> str:
    return f"{int(domino_id)}({_DOMINO_LABELS[int(domino_id)]})"


def _extract_plays(ctx: Any, args: dict[str, Any]) -> list[int]:
    raw = args.get("plays", args.get("candidate_plays"))
    if raw is None:
        from burl.chat.server.tools_library.legal_plays import tool as legal_tool

        payload = legal_tool(ctx)
        structured = dict(payload.get("structured", {}))
        raw = structured.get("legal_plays", [])
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",") if part.strip()]
    if not isinstance(raw, list):
        raise ValueError("plays must be a list of domino_ids, or omitted to use legal plays")
    plays = [int(play) for play in raw]
    if not plays:
        raise ValueError("no candidate plays available")
    for play in plays:
        if play < 0 or play >= 28:
            raise ValueError("candidate plays must be domino_ids 0-27")
    return plays


def _mode_label(center: float) -> str:
    if center >= 15:
        return "BIG WIN"
    if center >= 5:
        return "WIN"
    if center >= -5:
        return "NEAR-BREAKEVEN"
    if center >= -15:
        return "LOSS"
    return "DISASTER"


def _seat_word(seat_label: str, me_abs: int) -> str:
    return {
        "partner": f"PARTNER (seat {(me_abs + 2) % 4})",
        "left_opp": f"LEFT OPP (seat {(me_abs + 1) % 4})",
        "right_opp": f"RIGHT OPP (seat {(me_abs + 3) % 4})",
        "self": "MYSELF",
    }.get(seat_label, seat_label.upper())


def _dominant_information_source(dist: Any, me_abs: int) -> dict[str, Any]:
    spikes = list(dist.spike_drivers or [])
    best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for spike in spikes:
        mode_mass = float(spike.get("mode_mass", 0.0))
        for catalyst in spike.get("catalysts", []) or []:
            lift_raw = catalyst.get("lift", 1.0)
            lift = 99.0 if lift_raw == "inf" or lift_raw == float("inf") else float(lift_raw)
            freq = float(catalyst.get("freq_in_spike", 0.0))
            score = mode_mass * max(1.0, lift) * max(0.01, freq)
            if best is None or score > best[0]:
                best = (score, spike, catalyst)
    if best is None:
        return {
            "kind": "unimodal_or_diffuse",
            "text": "no single dominant catalyst",
        }
    _score, spike, catalyst = best
    dom = int(catalyst["domino"])
    seat = str(catalyst["seat"])
    center = float(spike.get("mode_center", 0.0))
    return {
        "kind": "catalyst",
        "seat": seat,
        "seat_display": _seat_word(seat, me_abs),
        "domino": dom,
        "domino_label": _DOMINO_LABELS[dom],
        "mode_center": round(center, 2),
        "mode_label": _mode_label(center),
        "mode_mass": round(float(spike.get("mode_mass", 0.0)), 3),
        "freq_in_spike": catalyst.get("freq_in_spike"),
        "lift": catalyst.get("lift"),
        "text": (
            f"{_seat_word(seat, me_abs)} holds {_label(dom)} "
            f"({ _mode_label(center) } mode Q={center:+.0f})"
        ),
    }


def _summarize_play(ctx: Any, play: int) -> dict[str, Any]:
    cache = ctx.get_or_build(play)
    dist = cache.dist
    n = max(1, int(dist.n_samples))
    mean = float(dist.mean)
    stdev = float(dist.stdev)
    se = stdev / math.sqrt(n)
    ci95 = (mean - 1.96 * se, mean + 1.96 * se)
    modes = [
        {
            "center": round(float(mode.get("center", 0.0)), 2),
            "mass": round(float(mode.get("mass", 0.0)), 3),
            "label": _mode_label(float(mode.get("center", 0.0))),
        }
        for mode in (dist.modes or [])
    ]
    return {
        "play": play,
        "label": _DOMINO_LABELS[play],
        "mean": round(mean, 2),
        "p_make": round(float(dist.p_make), 3),
        "stdev": round(stdev, 2),
        "ci95_mean": [round(ci95[0], 2), round(ci95[1], 2)],
        "n_samples": n,
        "shape": str(dist.distribution_shape),
        "modes": modes,
        "dominant_information_source": _dominant_information_source(dist, int(ctx.me_abs)),
    }


def _render_prose(rows: list[dict[str, Any]]) -> str:
    ranked = sorted(rows, key=lambda row: float(row["mean"]), reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    lines: list[str] = []
    lines.append("EXPECTED UTILITY RANKING")
    lines.append("")
    lines.append("Ranked by integrated mean Q over the current outcome distribution.")
    lines.append("")
    lines.append("| rank | play | E[Q] | p_make/set | 95% CI(mean) | shape | dominant information source |")
    lines.append("|---:|---|---:|---:|---|---|---|")
    for i, row in enumerate(ranked, 1):
        ci = row["ci95_mean"]
        lines.append(
            f"| {i} | {_label(row['play'])} | {float(row['mean']):+.1f} | "
            f"{float(row['p_make']):.2f} | [{float(ci[0]):+.1f}, {float(ci[1]):+.1f}] | "
            f"{row['shape']} | {row['dominant_information_source']['text']} |"
        )
    lines.append("")
    if second is not None:
        gap = float(best["mean"]) - float(second["mean"])
        lines.append(
            f"Recommendation by E[Q]: {_label(best['play'])} leads "
            f"{_label(second['play'])} by {gap:+.1f} Q."
        )
    else:
        lines.append(f"Only one candidate: {_label(best['play'])}.")
    lines.append("")
    lines.append(
        "This is the high-level EV synthesis. If the CI is wide or the top two plays "
        "are close, use play_brief or simulate_hand_impact on the named dominant "
        "information sources before committing."
    )
    return "\n".join(lines)


def _error(message: str, *, args: dict[str, Any]) -> ToolResult:
    return ToolResult(
        evidence={
            "prose": f"ERROR: calculate_expected_utility failed: {message}",
            "structured": {"error": message, "args": args},
        },
        next_tools=(),
    )


def _impl(ctx: Any, args: dict[str, Any]) -> ToolResult:
    try:
        plays = _extract_plays(ctx, args)
        rows = [_summarize_play(ctx, play) for play in plays]
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), args=args)
    ranked = sorted(rows, key=lambda row: float(row["mean"]), reverse=True)
    structured = {
        "candidate_plays": plays,
        "ranking": ranked,
        "recommended_play": ranked[0]["play"],
        "recommendation_basis": "max_mean_q",
    }
    return ToolResult(
        evidence={"prose": _render_prose(rows), "structured": structured},
        next_tools=(),
    )


CALCULATE_EXPECTED_UTILITY = ToolSpec(
    name="calculate_expected_utility",
    description=(
        "Calculates integrated expected utility for candidate plays by evaluating "
        "each play's outcome distribution over the current hidden-hand world set. "
        "Outputs a ranked list by mean Q, with p_make, uncertainty, and dominant "
        "information sources."
    ),
    params={
        "type": "object",
        "properties": {
            "plays": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "optional candidate domino_ids; omit to use current legal plays",
            },
        },
        "additionalProperties": False,
    },
    example="calculate_expected_utility(plays=[19, 25])",
    protocol_role="candidate_eval",
    protocol_phrase=(
        "Call `calculate_expected_utility()` to rank the current legal plays by "
        "integrated expected utility, or `calculate_expected_utility(plays=[...])` "
        "to rank a specified candidate set."
    ),
    impl=_impl,
    requires_context=True,
)


__all__ = ["CALCULATE_EXPECTED_UTILITY"]
