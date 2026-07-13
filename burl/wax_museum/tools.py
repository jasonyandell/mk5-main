"""HATEOAS tool implementations for wax_museum.

Thin wrappers over ``burl.tools.eq_distribution`` and
``burl.tools.meta_tools``. Each wrapper pares the underlying payload down to
what the model needs at its current state and appends a ``next_actions``
advertisement listing the tools the model CAN reach next.

Caching: spike_drivers for an (play) candidate are computed once per decision
inside ``WaxContext`` so that ``probe_best_case(play)`` and
``probe_worst_case(play)`` can look up the pre-selected catalyst without
recomputing the full distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from burl.harness.agent_runner import _DOMINO_LABELS
from burl.tools.belief_trajectory import belief_trajectory as call_belief_trajectory
from burl.tools.eq_distribution import (
    ConditionUnreachable,
    OutcomeDistribution,
    conditional_outcome,
    eq_outcome_distribution,
    load_eq_oracle,
)
from burl.tools.meta_tools import what_would_change_my_mind
from burl.wax_museum.schemas import (
    GateState,
    next_actions_after_explore,
    next_actions_after_probe,
    next_actions_unchanged,
)
from burl.wax_museum.snapshot import (
    render_full_board_snapshot,
    render_full_board_structured,
)


def _label(dom: int) -> str:
    """Render a domino as `id(pip-pip)`, matching the hand rendering in the
    system prompt (e.g. `21(6-0)`). Zero translation cost for the model."""
    return f"{int(dom)}({_DOMINO_LABELS[int(dom)]})"


def _render_next_actions(actions: list[dict]) -> str:
    if not actions:
        return ""
    lines = ["Next actions:"]
    for a in actions:
        lines.append(f"  - {a['tool']}: {a['when']}")
    return "\n".join(lines)


_SEAT_TO_OFFSET = {"self": 0, "left_opp": 1, "partner": 2, "right_opp": 3}


def _seat_label_to_abs(label: str, me_abs: int) -> int:
    return (me_abs + _SEAT_TO_OFFSET[label]) % 4


def _me_abs(game_state: Any) -> int:
    leader = getattr(game_state, "trick_leader", None)
    if leader is None:
        leader = game_state.leader
    return (int(leader) + len(game_state.current_trick)) % 4


# --------------------------------------------------------------------------- #
# Per-decision state: what plays have been explored, which catalysts to reuse #
# --------------------------------------------------------------------------- #


@dataclass
class PlayCache:
    """Cached exploration for a single candidate play."""

    play: int
    dist: OutcomeDistribution
    # Resolved assumptions derived from spike_drivers (or what_would_change_my_mind
    # on unimodal fallback). None means "no usable catalyst" — the probe tool
    # should return an explanatory error.
    best_case_assumption: dict | None = None   # {"player": abs_seat, "holds": dom}
    worst_case_assumption: dict | None = None
    best_case_rationale: str = ""
    worst_case_rationale: str = ""


@dataclass
class WaxContext:
    """Per-decision tool state — passed through tool_registry as closure state."""

    game_state: Any
    me_abs: int
    oracle: Any = None
    caches: dict[int, PlayCache] = field(default_factory=dict)
    # Tool-call history — the harness uses this to decide menu transitions and
    # detect bail conditions.
    call_log: list[tuple[str, dict]] = field(default_factory=list)

    def get_or_build(self, play: int, n_samples: int = 20) -> PlayCache:
        cached = self.caches.get(play)
        if cached is not None:
            return cached
        dist = eq_outcome_distribution(
            self.game_state,
            play=play,
            n_samples=n_samples,
            oracle=self.oracle,
            suggest_counterfactuals=True,
            include_spike_drivers=True,
        )
        best, best_rat, worst, worst_rat = self._resolve_assumptions(play, dist)
        cache = PlayCache(
            play=play,
            dist=dist,
            best_case_assumption=best,
            worst_case_assumption=worst,
            best_case_rationale=best_rat,
            worst_case_rationale=worst_rat,
        )
        self.caches[play] = cache
        return cache

    def _resolve_assumptions(
        self, play: int, dist: OutcomeDistribution,
    ) -> tuple[dict | None, str, dict | None, str]:
        """Pick the (player, holds) pair for probe_best_case / probe_worst_case.

        For bimodal/multimodal distributions, take the highest- and lowest-center
        modes and grab each's top-lift catalyst. For unimodal, fall back to
        ``what_would_change_my_mind`` and take the most-positive / most-negative
        shift.
        """
        if dist.spike_drivers and dist.distribution_shape != "unimodal":
            # spike_drivers is sorted by mode_center ascending (disaster first).
            worst = dist.spike_drivers[0]
            best = dist.spike_drivers[-1]
            worst_a = self._catalyst_to_assumption(worst, dist.mean)
            best_a = self._catalyst_to_assumption(best, dist.mean)
            return best_a[0], best_a[1], worst_a[0], worst_a[1]

        # Unimodal: use what_would_change_my_mind. Its "player" field is the
        # same relative label ("partner"/"left_opp"/"right_opp") spike_drivers
        # uses, so we convert to abs seat the same way.
        ranking = what_would_change_my_mind(
            self.game_state, play=play, n_samples_per_probe=10, top_k=5,
            oracle=self.oracle,
        )
        assumptions = ranking.get("assumptions", [])
        if not assumptions:
            return None, "no actionable catalyst", None, "no actionable catalyst"
        sorted_by_shift = sorted(assumptions, key=lambda a: a["shift"])
        worst_row = sorted_by_shift[0]
        best_row = sorted_by_shift[-1]
        best_a = {
            "player": _seat_label_to_abs(best_row["player"], self.me_abs),
            "holds": int(best_row["holds"]),
        }
        worst_a = {
            "player": _seat_label_to_abs(worst_row["player"], self.me_abs),
            "holds": int(worst_row["holds"]),
        }
        return (
            best_a,
            f"unimodal; largest upside shift {best_row['shift']:+.2f} Q "
            f"({best_row['player']} holds {best_row['holds']})",
            worst_a,
            f"unimodal; largest downside shift {worst_row['shift']:+.2f} Q "
            f"({worst_row['player']} holds {worst_row['holds']})",
        )

    def _catalyst_to_assumption(
        self, spike: dict, unconditional_mean: float,
    ) -> tuple[dict | None, str]:
        catalysts = spike.get("catalysts", [])
        if not catalysts:
            return None, "mode had no catalyst above lift threshold"
        top = catalysts[0]
        seat_label = top["seat"]  # "partner" | "left_opp" | "right_opp"
        if seat_label == "self":
            return None, "self-catalyst (skipped)"
        abs_seat = _seat_label_to_abs(seat_label, self.me_abs)
        dom = int(top["domino"])
        mode_center = float(spike["mode_center"])
        delta = mode_center - unconditional_mean
        direction = "lifts" if delta > 0 else "drops"
        rationale = (
            f"mode center {mode_center:+.1f} Q ({direction} {abs(delta):.1f} vs "
            f"unconditional {unconditional_mean:+.1f}); catalyst: "
            f"{seat_label} holds {dom} (freq_in_spike={top['freq_in_spike']}, "
            f"lift={top['lift']})"
        )
        return {"player": abs_seat, "holds": dom}, rationale


# --------------------------------------------------------------------------- #
# Tool implementations                                                         #
# --------------------------------------------------------------------------- #


def _summarize_distribution(dist: OutcomeDistribution) -> dict:
    return {
        "mean": round(float(dist.mean), 2),
        "stdev": round(float(dist.stdev), 2),
        "p_make": round(float(dist.p_make), 2),
        "n_samples": int(dist.n_samples),
        "shape": dist.distribution_shape,
        "modes": [
            {"center": round(float(m["center"]), 1), "mass": round(float(m["mass"]), 2)}
            for m in dist.modes
        ],
        "gap_between_modes": round(float(dist.gap_between_modes), 1),
    }


def _summarize_spikes(dist: OutcomeDistribution) -> list[dict]:
    out: list[dict] = []
    for spike in dist.spike_drivers:
        out.append({
            "mode_center": round(float(spike["mode_center"]), 1),
            "mode_mass": round(float(spike["mode_mass"]), 2),
            "n_worlds_in_spike": int(spike["n_worlds_in_spike"]),
            "catalysts": [
                {
                    "seat": c["seat"],
                    "domino": int(c["domino"]),
                    "freq_in_spike": c["freq_in_spike"],
                    "baseline": c["baseline"],
                    "lift": c["lift"] if c["lift"] != float("inf") else "inf",
                }
                for c in spike.get("catalysts", [])
            ],
        })
    return out


# --------------------------------------------------------------------------- #
# Prose renderers — the format the MODEL sees.                                 #
#                                                                              #
# The structured dicts above still travel through events.jsonl for trace       #
# analysis, but the chat-template `content` field is a markdown-ish block      #
# with tables/bullets the model can quote verbatim. JSON tool responses        #
# correlated with hallucinated numeric values in the N=5 run; prose doesn't.  #
# --------------------------------------------------------------------------- #


_CHART_WIDTH = 43  # Q axis from -42 to +42 inclusive → 85 bins, half-width for readability.
_CHART_Q_MIN = -42
_CHART_Q_MAX = +42


def _q_to_col(q: float) -> int:
    """Map a Q value in [-42, +42] onto a column index in [0, _CHART_WIDTH)."""
    frac = (q - _CHART_Q_MIN) / (_CHART_Q_MAX - _CHART_Q_MIN)
    col = int(round(frac * (_CHART_WIDTH - 1)))
    return max(0, min(_CHART_WIDTH - 1, col))


def _render_mode_chart(modes: list[dict]) -> list[str]:
    """Tiny horizontal bar chart: each mode is a bar of width proportional to
    its mass, plotted at its Q center. Designed to be glanceable — 'bimodal
    with a disaster tail' should be readable in one look.
    """
    if not modes:
        return []
    axis = [" "] * _CHART_WIDTH
    marker = [" "] * _CHART_WIDTH
    label_line = [" "] * _CHART_WIDTH
    for m in modes:
        center = float(m["center"])
        mass = float(m["mass"])
        col = _q_to_col(center)
        height = max(1, int(round(mass * 10)))  # 10% mass → 1 char, scales visually
        # Draw a solid block "bar" centered on the Q col.
        width = max(1, int(round(mass * 12)))
        half = width // 2
        for c in range(max(0, col - half), min(_CHART_WIDTH, col + half + 1)):
            axis[c] = "█"
        marker[col] = "┬"
        # Write the center label below the marker, clipping to chart bounds.
        cstr = f"{center:+.0f}"
        start = max(0, min(_CHART_WIDTH - len(cstr), col - len(cstr) // 2))
        for i, ch in enumerate(cstr):
            label_line[start + i] = ch

    axis_str = "".join(axis)
    marker_str = "".join(marker)
    label_str = "".join(label_line)
    # Tick row spanning Q=-42 .. 0 .. +42.
    tick_row = list(" " * _CHART_WIDTH)
    for q, tag in [(-42, "-42"), (0, "0"), (42, "+42")]:
        col = _q_to_col(q)
        start = max(0, min(_CHART_WIDTH - len(tag), col - len(tag) // 2))
        for i, ch in enumerate(tag):
            tick_row[start + i] = ch
    return [
        axis_str,
        marker_str,
        label_str,
        "-" * _CHART_WIDTH,
        "".join(tick_row),
    ]


def _seat_descriptor(seat_label: str) -> str:
    """Human-friendly seat name for if/then narration."""
    return {
        "partner": "partner",
        "left_opp": "left opp",
        "right_opp": "right opp",
        "self": "I",
    }.get(seat_label, seat_label)


def _render_ifthens(dist: OutcomeDistribution, unconditional_mean: float) -> list[str]:
    """For each spike, render an IF/THEN line naming the top catalyst and the
    resulting Q. Ordered by mode center (worst → best) so the model sees the
    downside first.
    """
    lines: list[str] = []
    for spike in dist.spike_drivers:
        center = float(spike["mode_center"])
        mass = float(spike["mode_mass"])
        cats = spike.get("catalysts", [])
        if not cats:
            continue
        top = cats[0]
        seat = _seat_descriptor(top["seat"])
        dom_label = _label(top["domino"])
        delta = center - unconditional_mean
        if delta >= 10:
            effect = f"wins by {delta:.0f}"
        elif delta <= -10:
            effect = f"loses by {abs(delta):.0f}"
        elif abs(delta) < 3:
            effect = "close to breakeven"
        else:
            direction = "up" if delta > 0 else "down"
            effect = f"moves {direction} {abs(delta):.0f}"
        lines.append(
            f"  IF {seat} holds {dom_label}:  Q → {center:+.0f}   "
            f"({mass:.0%} of worlds, {effect})"
        )
    return lines


def _pivot_synthesis(dist: OutcomeDistribution) -> str | None:
    """If every spike's top catalyst points at the same domino, that domino
    is the pivot of the decision — call it out explicitly. This is the
    oracle-authored insight line the model can quote faithfully.
    """
    if not dist.spike_drivers:
        return None
    tops: list[tuple[str, int]] = []
    for spike in dist.spike_drivers:
        cats = spike.get("catalysts", [])
        if not cats:
            return None
        c = cats[0]
        tops.append((c["seat"], int(c["domino"])))
    # Same domino across every spike?
    doms = {t[1] for t in tops}
    if len(doms) == 1:
        dom = next(iter(doms))
        return f"The pivot is {_label(dom)}. This trick turns on who holds it."
    return None


def _render_explore_prose(cache: PlayCache) -> str:
    dist = cache.dist
    lines: list[str] = []
    lines.append(
        f"PLAY: {_label(cache.play)}   "
        f"unconditional Q = {dist.mean:+.1f}   p_make = {dist.p_make:.2f}"
    )
    if dist.spike_drivers:
        lines.append("")
        lines.append("Outcome shape (Q axis, bars at each mode):")
        lines.append("")
        for row in _render_mode_chart(dist.modes):
            lines.append("  " + row)
    lines.append("")
    ifthens = _render_ifthens(dist, float(dist.mean))
    if ifthens:
        lines.append("Scenarios:")
        lines.extend(ifthens)
        lines.append("")
    pivot = _pivot_synthesis(dist)
    if pivot:
        lines.append(pivot)
    else:
        # Fall back to the older best-case / worst-case rationale lines.
        if cache.best_case_assumption is not None:
            lines.append(f"Best case:  {cache.best_case_rationale}")
        if cache.worst_case_assumption is not None:
            lines.append(f"Worst case: {cache.worst_case_rationale}")
    lines.append("")
    lines.append(f"(shape: {dist.distribution_shape}, stdev {dist.stdev:.1f}, "
                 f"N={dist.n_samples} samples)")
    return "\n".join(lines)


def _render_shift_chart(uncond_mean: float, cond_mean: float) -> list[str]:
    """Two-point number line showing the mean shift from uncond → cond."""
    u_col = _q_to_col(uncond_mean)
    c_col = _q_to_col(cond_mean)
    row_bars = [" "] * _CHART_WIDTH
    row_labels = [" "] * _CHART_WIDTH
    row_bars[u_col] = "○"
    row_bars[c_col] = "●"
    for tag, col in (
        (f"{uncond_mean:+.0f} (before)", u_col),
        (f"{cond_mean:+.0f} (after)", c_col),
    ):
        start = max(0, min(_CHART_WIDTH - len(tag), col - len(tag) // 2))
        # Avoid label collision: don't overwrite occupied chars.
        for i, ch in enumerate(tag):
            if row_labels[start + i] == " ":
                row_labels[start + i] = ch
    # Draw a connecting bar between the two points.
    lo, hi = (u_col, c_col) if u_col <= c_col else (c_col, u_col)
    for c in range(lo + 1, hi):
        if row_bars[c] == " ":
            row_bars[c] = "─"
    tick_row = list(" " * _CHART_WIDTH)
    for q, tag in [(-42, "-42"), (0, "0"), (42, "+42")]:
        col = _q_to_col(q)
        start = max(0, min(_CHART_WIDTH - len(tag), col - len(tag) // 2))
        for i, ch in enumerate(tag):
            tick_row[start + i] = ch
    return [
        "".join(row_bars),
        "".join(row_labels),
        "-" * _CHART_WIDTH,
        "".join(tick_row),
    ]


def _render_probe_prose(
    play: int, side: str, assumption: dict, rationale: str,
    cond: OutcomeDistribution, unconditional_mean: float, shift: float,
) -> str:
    seat_abs = int(assumption["player"])
    dom = int(assumption["holds"])
    lines: list[str] = []
    lines.append(
        f"PROBE ({side}): play {_label(play)}   "
        f"assume seat {seat_abs} holds {_label(dom)}"
    )
    lines.append("")
    lines.append("Mean shifts (○ = before, ● = after):")
    lines.append("")
    for row in _render_shift_chart(unconditional_mean, float(cond.mean)):
        lines.append("  " + row)
    lines.append("")
    if shift > 1.0:
        narrative = f"This assumption LIFTS the outcome by {shift:+.1f} Q."
    elif shift < -1.0:
        narrative = f"This assumption DROPS the outcome by {abs(shift):.1f} Q."
    else:
        narrative = f"This assumption barely moves the outcome ({shift:+.1f} Q)."
    lines.append(narrative)
    lines.append("")
    lines.append(
        f"Under the assumption: Q → {cond.mean:+.1f}, p_make = {cond.p_make:.2f}, "
        f"shape = {cond.distribution_shape}."
    )
    return "\n".join(lines)


def _render_prose_error(msg: str) -> str:
    return f"ERROR: {msg}"


def _wrap(prose: str, structured: Any, next_actions: list[dict]) -> dict:
    """Canonical tool-response shape.

    The harness feeds `prose` to the model (as `role="tool"` content) and logs
    `structured` + `next_actions` into events.jsonl for trace analysis.
    """
    full_prose = prose
    tail = _render_next_actions(next_actions)
    if tail:
        full_prose = f"{prose}\n\n{tail}"
    return {
        "prose": full_prose,
        "structured": structured,
        "next_actions": next_actions,
    }


def tool_explore_game(ctx: WaxContext, play: int) -> dict:
    play = int(play)
    cache = ctx.get_or_build(play)
    dist = cache.dist
    has_bimodal = dist.distribution_shape != "unimodal" and bool(dist.spike_drivers)
    structured = {
        "play": play,
        "summary": _summarize_distribution(dist),
        "spikes": _summarize_spikes(dist),
        "best_case_available": cache.best_case_assumption is not None,
        "worst_case_available": cache.worst_case_assumption is not None,
        "best_case_rationale": cache.best_case_rationale,
        "worst_case_rationale": cache.worst_case_rationale,
    }
    return _wrap(
        prose=_render_explore_prose(cache),
        structured=structured,
        next_actions=next_actions_after_explore(play, has_bimodal=has_bimodal),
    )


def _probe(ctx: WaxContext, play: int, side: str) -> dict:
    play = int(play)
    cache = ctx.caches.get(play)
    if cache is None:
        return _wrap(
            prose=_render_prose_error(
                f"probe_{side} requires explore_game(play={play}) first — "
                "no cached distribution for this play."
            ),
            structured={"error": "needs_explore_first"},
            next_actions=next_actions_unchanged(GateState.INITIAL),
        )
    assumption = (
        cache.best_case_assumption if side == "best_case" else cache.worst_case_assumption
    )
    rationale = (
        cache.best_case_rationale if side == "best_case" else cache.worst_case_rationale
    )
    if assumption is None:
        return _wrap(
            prose=_render_prose_error(
                f"no usable {side} catalyst for play={play} ({rationale})"
            ),
            structured={"error": "no_catalyst", "rationale": rationale},
            next_actions=next_actions_after_explore(
                play, has_bimodal=bool(cache.dist.spike_drivers),
            ),
        )
    try:
        cond = conditional_outcome(
            ctx.game_state,
            play=play,
            assumption=assumption,
            n_samples=20,
            oracle=ctx.oracle,
        )
    except ConditionUnreachable as e:
        return _wrap(
            prose=_render_prose_error(f"assumption unreachable: {e}"),
            structured={"error": "unreachable", "detail": str(e)},
            next_actions=next_actions_after_explore(
                play, has_bimodal=bool(cache.dist.spike_drivers),
            ),
        )
    unconditional_mean = float(cache.dist.mean)
    conditional_mean = float(cond.mean)
    shift = conditional_mean - unconditional_mean
    structured = {
        "play": play,
        "side": side,
        "assumption": assumption,
        "rationale": rationale,
        "conditional": _summarize_distribution(cond),
        "unconditional_mean": round(unconditional_mean, 2),
        "shift_in_mean": round(shift, 2),
        "direction": "upside" if shift > 0 else "downside" if shift < 0 else "neutral",
    }
    return _wrap(
        prose=_render_probe_prose(
            play, side, assumption, rationale, cond, unconditional_mean, shift,
        ),
        structured=structured,
        next_actions=next_actions_after_probe(play),
    )


def tool_probe_best_case(ctx: WaxContext, play: int) -> dict:
    return _probe(ctx, play, side="best_case")


def tool_probe_worst_case(ctx: WaxContext, play: int) -> dict:
    return _probe(ctx, play, side="worst_case")


# --------------------------------------------------------------------------- #
# Rules — compact static answers. Grounded in the trimmed primer text so the  #
# model gets consistent vocabulary.                                            #
# --------------------------------------------------------------------------- #


_RULE_ANSWERS: dict[str, str] = {
    "trump_ordering": (
        "Under a pip-suit trump (e.g. fives), every domino containing that pip "
        "is a trump, ranked high-to-low by the OTHER pip. The double of the "
        "trump suit is highest. Example: fives trump → 5-5 (highest), 6-5, "
        "5-4, 5-3, 5-2, 5-1, 5-0 (lowest). Under doubles-as-trump, only the 7 "
        "doubles are trump (6-6 highest, 0-0 lowest). Under notrump, no suit "
        "has trump power."
    ),
    "trick_winner": (
        "If any trumps are played in the trick, the highest trump wins. "
        "Otherwise, the highest domino of the LED suit wins. Off-suit dominoes "
        "can never win a trick unless they are trump."
    ),
    "contract_math": (
        "A hand totals 42 points (35 count pips + 7 trick points). The bidder's "
        "team must capture at least their bid; otherwise they are 'set' and "
        "defense scores the bid value. Count dominoes: 5-5 and 6-4 are worth "
        "10 each; 5-0, 4-1, 3-2 are worth 5 each (35 total)."
    ),
    "void_rules": (
        "You MUST follow the led suit if you hold any domino of it. If you are "
        "void in the led suit, you may play any domino (including a trump — "
        "trumping the trick). Partners' public void history can be inferred "
        "from the play log; use void_audit() via engine tools if needed."
    ),
}


def tool_belief_trajectory(ctx: WaxContext) -> dict:
    """Read Gus's calibrated belief head for the current decision.

    Returns the wrapped payload: the ``prose`` field is what the model sees in
    the tool-response envelope; ``structured`` carries the full dict for trace
    analysis (belief per unseen domino, top shifts since last call on this
    decision, V, attention top tokens, policy top-K). ``format="both"`` on the
    underlying tool gives us both shapes in a single call.

    Does not advance the gate state — re-advertise the current state's
    actions via ``next_actions_unchanged``.
    """
    try:
        payload = call_belief_trajectory(
            ctx.game_state,
            top_k_shifts=5,
            include_policy=True,
            format="both",
        )
    except Exception as e:
        return _wrap(
            prose=_render_prose_error(f"belief_trajectory failed: {e}"),
            structured={"error": str(e)},
            next_actions=[],
        )
    # payload is a dict with a "prose" key + all structured fields.
    prose = payload["prose"]
    structured = {k: v for k, v in payload.items() if k != "prose"}
    return _wrap(
        prose=prose,
        structured=structured,
        next_actions=[],   # re-advertised by the harness via next_actions_unchanged
    )


def tool_ask_rule(ctx: WaxContext, topic: str) -> dict:
    answer = _RULE_ANSWERS.get(topic)
    if answer is None:
        return _wrap(
            prose=_render_prose_error(
                f"unknown topic {topic!r}; valid: {sorted(_RULE_ANSWERS)}"
            ),
            structured={"error": "unknown_topic"},
            next_actions=[],
        )
    # State does not advance — advertise current state's actions via the harness.
    prose = f"RULE ({topic}):\n\n{answer}"
    return _wrap(
        prose=prose,
        structured={"topic": topic, "answer": answer},
        next_actions=[],
    )


# --------------------------------------------------------------------------- #
# Registry factory                                                             #
# --------------------------------------------------------------------------- #


def tool_full_board_snapshot(ctx: WaxContext) -> dict:
    """Single instant picture of the table — pure rule-based read of state.

    No oracle/Gus calls; sub-millisecond. Surfaces seats, trump ranking,
    remaining hand, current trick, score + bid math, count-domino ledger,
    and completed-trick history. Does NOT include opponent posteriors —
    the prose tells the model to call ``belief_trajectory()`` for that.
    """
    prose = render_full_board_snapshot(ctx.game_state, ctx.me_abs)
    structured = render_full_board_structured(ctx.game_state, ctx.me_abs)
    return _wrap(prose=prose, structured=structured, next_actions=[])


def build_registry(ctx: WaxContext) -> dict[str, Any]:
    snap = lambda **kw: tool_full_board_snapshot(ctx, **kw)  # noqa: E731
    return {
        "explore_game": lambda **kw: tool_explore_game(ctx, **kw),
        "probe_best_case": lambda **kw: tool_probe_best_case(ctx, **kw),
        "probe_worst_case": lambda **kw: tool_probe_worst_case(ctx, **kw),
        "ask_rule": lambda **kw: tool_ask_rule(ctx, **kw),
        "belief_trajectory": lambda **kw: tool_belief_trajectory(ctx, **kw),
        "full_board_snapshot": snap,
        # Burl proposed both names — register the alias too.
        "game_state_snapshot": snap,
    }
