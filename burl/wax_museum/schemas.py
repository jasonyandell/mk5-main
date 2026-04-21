"""Tool JSON schemas + turn-indexed menu state machine for wax_museum.

The hard gate: Gemma 4 sees only the tools the current state permits. The menu
is the single lever — Practicalities 1/3/4 say we bend the environment, not the
weights. Turn transitions are deterministic and cheap.

States (per decision):

    INITIAL           → only explore_game is callable.
    AFTER_EXPLORE     → explore_game + probe_best_case + probe_worst_case + ask_rule.
    AFTER_PROBE       → above + commit_play. Terminal is commit_play.

ask_rule is a free side-call — it does not advance the state, only the probes do.
"""

from __future__ import annotations

from enum import Enum
from typing import TypedDict


# --------------------------------------------------------------------------- #
# Schemas — JSON-Schema dicts the chat template renders into Gemma's native    #
# <|tool>...<tool|> block.                                                     #
# --------------------------------------------------------------------------- #


def _schema(
    name: str,
    description: str,
    properties: dict,
    required: list[str] | None = None,
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


EXPLORE_GAME = _schema(
    "explore_game",
    (
        "Examine a candidate play. Runs the outcome distribution and returns "
        "the distribution shape (unimodal/bimodal/multimodal), each mode's "
        "center + mass, and the catalyst dominoes that drive the best-case "
        "and worst-case spikes. This is the only tool available on the first "
        "turn — use it to form a plan before probing deeper."
    ),
    {
        "play": {
            "type": "integer",
            "description": "domino_id (0..27) from your hand to examine.",
        }
    },
    required=["play"],
)

PROBE_BEST_CASE = _schema(
    "probe_best_case",
    (
        "Condition on the catalyst that drives the UPSIDE spike for `play`. "
        "Answers: 'if the best-case belief is true, what does this play "
        "actually look like?' You must have called explore_game(play) first. "
        "Returns the conditional outcome distribution (mean, p_make, shape) "
        "restricted to worlds where the upside catalyst holds."
    ),
    {
        "play": {"type": "integer", "description": "same domino_id you passed to explore_game"}
    },
    required=["play"],
)

PROBE_WORST_CASE = _schema(
    "probe_worst_case",
    (
        "Condition on the catalyst that drives the DOWNSIDE spike for `play`. "
        "Answers: 'if the disaster branch is real, how bad does this play "
        "get?' You must have called explore_game(play) first. Returns the "
        "conditional outcome distribution restricted to worlds where the "
        "downside catalyst holds."
    ),
    {
        "play": {"type": "integer", "description": "same domino_id you passed to explore_game"}
    },
    required=["play"],
)

ASK_RULE = _schema(
    "ask_rule",
    (
        "Look up a compact rule answer. Free side-call; does not advance the "
        "turn state. Topics: 'trump_ordering' (how trumps rank under this "
        "declaration), 'trick_winner' (who wins a trick), 'contract_math' "
        "(bid vs points captured), 'void_rules' (when can I play off-suit)."
    ),
    {
        "topic": {
            "type": "string",
            "enum": ["trump_ordering", "trick_winner", "contract_math", "void_rules"],
            "description": "Which rule to look up.",
        }
    },
    required=["topic"],
)

COMMIT_PLAY = _schema(
    "commit_play",
    (
        "Commit your final play and end the decision. Only available after "
        "you have called at least one probe (best_case or worst_case). The "
        "engine validates legality; illegal plays return control to you for "
        "another turn."
    ),
    {
        "domino_id": {
            "type": "integer",
            "description": "0..27, must be in your hand.",
        }
    },
    required=["domino_id"],
)


# --------------------------------------------------------------------------- #
# State machine                                                                #
# --------------------------------------------------------------------------- #


class GateState(Enum):
    INITIAL = "initial"
    AFTER_EXPLORE = "after_explore"
    AFTER_PROBE = "after_probe"


_MENUS: dict[GateState, list[dict]] = {
    GateState.INITIAL: [EXPLORE_GAME],
    GateState.AFTER_EXPLORE: [EXPLORE_GAME, PROBE_BEST_CASE, PROBE_WORST_CASE, ASK_RULE],
    GateState.AFTER_PROBE: [EXPLORE_GAME, PROBE_BEST_CASE, PROBE_WORST_CASE, ASK_RULE, COMMIT_PLAY],
}


def menu_for(state: GateState) -> list[dict]:
    """Return the tool schema list visible in this state."""
    return list(_MENUS[state])


def menu_names(state: GateState) -> list[str]:
    return [s["function"]["name"] for s in _MENUS[state]]


class NextAction(TypedDict):
    tool: str
    when: str


def advance(state: GateState, tool_called: str) -> GateState:
    """Return the new state after `tool_called` was invoked.

    Rules:
      - explore_game from INITIAL → AFTER_EXPLORE.
      - probe_* from AFTER_EXPLORE → AFTER_PROBE.
      - ask_rule never advances.
      - explore_game from AFTER_EXPLORE or AFTER_PROBE: re-exploring a different
        candidate is allowed; state does not regress (probes once earned stay
        earned).
      - commit_play is terminal; the harness exits before we'd advance state.
    """
    if tool_called == "ask_rule":
        return state
    if tool_called == "explore_game":
        return GateState.AFTER_EXPLORE if state == GateState.INITIAL else state
    if tool_called in ("probe_best_case", "probe_worst_case"):
        if state == GateState.INITIAL:
            return state  # still locked — probe needs explore first
        return GateState.AFTER_PROBE
    return state


# --------------------------------------------------------------------------- #
# Advertised "next_actions" helpers. Each tool response includes a prose block #
# naming the tools the model CAN reach next — the model reads this alongside  #
# the schema list rendered by the chat template.                               #
# --------------------------------------------------------------------------- #


def next_actions_after_explore(play: int, has_bimodal: bool) -> list[NextAction]:
    upside_when = (
        "Condition on the upside catalyst; check whether the best-case branch survives."
    )
    downside_when = (
        "Condition on the downside catalyst; see how bad the disaster branch really is."
    )
    if not has_bimodal:
        # Unimodal fallback — probes run against what_would_change_my_mind catalysts.
        upside_when = (
            "Probe the assumption that would most lift this play's E[Q]."
        )
        downside_when = (
            "Probe the assumption that would most drop this play's E[Q]."
        )
    return [
        {"tool": "probe_best_case", "when": upside_when + f" Call: probe_best_case(play={play})."},
        {"tool": "probe_worst_case", "when": downside_when + f" Call: probe_worst_case(play={play})."},
        {"tool": "explore_game", "when": "Switch to a different candidate play."},
        {"tool": "ask_rule", "when": "Look up a 42 rule. Free; does not advance state."},
    ]


def next_actions_after_probe(play: int) -> list[NextAction]:
    return [
        {"tool": "commit_play", "when": f"Commit your final play. Call: commit_play(domino_id=<int>)."},
        {"tool": "probe_best_case", "when": f"Check the other branch for play={play}."},
        {"tool": "probe_worst_case", "when": f"Check the other branch for play={play}."},
        {"tool": "explore_game", "when": "Examine a different candidate play before committing."},
        {"tool": "ask_rule", "when": "Look up a 42 rule."},
    ]


def next_actions_unchanged(state: GateState, play: int | None = None) -> list[NextAction]:
    """For ask_rule — re-advertise the current state's actions."""
    if state == GateState.INITIAL:
        return [{"tool": "explore_game", "when": "Examine a candidate play (pick any domino in your hand)."}]
    if state == GateState.AFTER_EXPLORE:
        assert play is not None
        return next_actions_after_explore(play, has_bimodal=True)
    return next_actions_after_probe(play if play is not None else -1)


# --------------------------------------------------------------------------- #
# Smoke                                                                        #
# --------------------------------------------------------------------------- #


def _smoke() -> None:
    s = GateState.INITIAL
    assert menu_names(s) == ["explore_game"], menu_names(s)
    s = advance(s, "explore_game")
    assert s == GateState.AFTER_EXPLORE
    assert set(menu_names(s)) == {"explore_game", "probe_best_case", "probe_worst_case", "ask_rule"}
    # probe advances
    s2 = advance(s, "probe_worst_case")
    assert s2 == GateState.AFTER_PROBE
    assert "commit_play" in menu_names(s2)
    # ask_rule does not advance
    s3 = advance(s, "ask_rule")
    assert s3 == GateState.AFTER_EXPLORE
    # probe from INITIAL is a no-op (gate enforced)
    assert advance(GateState.INITIAL, "probe_best_case") == GateState.INITIAL
    print("[wax_museum.schemas] smoke OK")
    print(f"  INITIAL menu: {menu_names(GateState.INITIAL)}")
    print(f"  AFTER_EXPLORE menu: {menu_names(GateState.AFTER_EXPLORE)}")
    print(f"  AFTER_PROBE menu: {menu_names(GateState.AFTER_PROBE)}")


if __name__ == "__main__":
    _smoke()
