"""Phase categorization + dynamic chain composition for tool calls.

Categorizes every tool by its decision phase and provides two services
that the runner uses on every dispatch:

  1. ``refuse_illegal_score(ctx, tool, args)`` — phase-3 (score) calls
     with a ``play`` arg outside the legal set return a structured refusal
     payload pointing back at ``legal_plays``. The oracle is never asked
     to evaluate a counterfactual that violates follow-suit. Motivated by
     harvest_batched_20260426_031338 decision #7 in the t42-dyjn quick
     pass: ``explore_game(play=27)`` returned a plausible Q-distribution
     for an illegal 6-6, the model probed it, the model committed it.

  2. ``compose_next(ctx, just_ran)`` — appends a "Next:" affordance line
     to the dispatched prose. HATEOAS for LLMs: the tool response carries
     the link to the next tool. Fights the [[burl-tool-wishlist]] adoption
     asymmetry without protocol-text patching or adapter retraining.

Tool→phase mapping is hardcoded for base tools. Improvised tools default
to ``read``; their ``DESCRIPTION`` may include ``PHASE: <name>`` to
override.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

PHASES = ("read", "enumerate", "belief", "score", "meta", "commit")

TOOL_PHASE: dict[str, str] = {
    "state_brief": "read",
    "board_snapshot": "read",
    "full_board_snapshot": "read",
    "game_state_snapshot": "read",
    "legal_plays": "enumerate",
    "belief_trajectory": "belief",
    "play_brief": "score",
    "explore_game": "score",
    "probe_best_case": "score",
    "probe_worst_case": "score",
    "ask_rule": "meta",
    "commit_play": "commit",
}

_PHASE_TAG_RE = re.compile(r"^\s*PHASE\s*:\s*(\w+)\s*$", re.MULTILINE)


def phase_of(tool_name: str, description: str = "") -> str:
    if tool_name in TOOL_PHASE:
        return TOOL_PHASE[tool_name]
    m = _PHASE_TAG_RE.search(description)
    if m and m.group(1) in PHASES:
        return m.group(1)
    return "read"


def all_tools_in_phase(phase: str) -> list[str]:
    from . import improvised_tools

    out = [n for n, p in TOOL_PHASE.items() if p == phase]
    for t in improvised_tools.list_all():
        if t.name in TOOL_PHASE:
            continue
        if phase_of(t.name, t.description) == phase:
            out.append(t.name)
    return sorted(set(out))


def legal_set(ctx) -> set[int]:
    """Set of legal play IDs for ctx's current state. Mirrors legal_plays.py."""
    from forge.oracle.tables import can_follow, led_suit_for_lead_domino

    gs = ctx.game_state
    me = int(ctx.me_abs)
    decl_id = int(gs.decl_id)
    played = gs.played
    my_hand = sorted(d for d in gs.hands[me] if d not in played)
    cur = tuple(int(x) for x in gs.current_trick)
    if not cur:
        return set(my_hand)
    lead_dom = cur[0]
    led_suit = led_suit_for_lead_domino(lead_dom, decl_id)
    followers = [d for d in my_hand if can_follow(d, led_suit, decl_id)]
    return set(followers) if followers else set(my_hand)


def refuse_illegal_score(ctx, tool_name: str, args: dict) -> dict | None:
    """Phase-3 guard. Returns a refusal payload, or None to proceed."""
    if phase_of(tool_name) != "score":
        return None
    if "play" not in args:
        return None
    legal = legal_set(ctx)
    play = int(args["play"])
    if play in legal:
        return None
    legal_sorted = sorted(legal)
    prose = (
        f"REFUSED: play {play} is not in your legal set {legal_sorted}.\n"
        f"This tool will not score a hypothetical that violates follow-suit rules.\n"
        f"\nNext: call legal_plays() to confirm your options, then re-call "
        f"{tool_name}(play=X) with X in {legal_sorted}."
    )
    return {
        "prose": prose,
        "structured": {
            "refused": True,
            "reason": "illegal_play_arg",
            "requested_play": play,
            "legal_plays": legal_sorted,
            "tool": tool_name,
        },
    }


def compose_next(ctx, just_ran: str) -> str:
    """HATEOAS affordance line for whatever just dispatched. Stateless."""
    phase = phase_of(just_ran)
    if phase in ("commit", "meta"):
        return ""

    legal = sorted(legal_set(ctx))
    forced = len(legal) == 1

    if phase == "read":
        if forced:
            return (
                f"\nNext: you have one legal play ({legal[0]}). "
                f"Consider commit_play(domino_id={legal[0]})."
            )
        enum_tools = all_tools_in_phase("enumerate")
        primary = enum_tools[0] if enum_tools else "legal_plays"
        return (
            f"\nNext: typically {primary}() to confirm your legal options "
            f"before scoring."
        )

    if phase == "enumerate":
        if forced:
            return (
                f"\nNext: you have one legal play ({legal[0]}). "
                f"Consider commit_play(domino_id={legal[0]})."
            )
        belief_tools = all_tools_in_phase("belief")
        if belief_tools:
            return (
                f"\nNext: typically {belief_tools[0]}() to read opponents' "
                f"likely hands before scoring candidates."
            )
        score_tools = all_tools_in_phase("score")
        primary = (
            "play_brief" if "play_brief" in score_tools
            else (score_tools[0] if score_tools else "explore_game")
        )
        return (
            f"\nNext: score your candidates with {primary}(play=X) for "
            f"X in {legal}."
        )

    if phase == "belief":
        score_tools = all_tools_in_phase("score")
        primary = "play_brief" if "play_brief" in score_tools else "explore_game"
        return (
            f"\nNext: score your legal candidates {legal} with "
            f"{primary}(play=X) — one call per candidate."
        )

    if phase == "score":
        return (
            f"\nNext: score remaining candidates from {legal} with the same tool, "
            f"or commit_play(domino_id=X) once you've decided."
        )

    return ""
