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

BELIEF_TRAJECTORY = _schema(
    "belief_trajectory",
    (
        "Read Gus's calibrated belief over opponent hands for the current "
        "decision. Returns per-domino posterior (probability each unseen "
        "domino sits with left/partner/right opponent), shifts since the "
        "previous call in this decision, Gus's value estimate V, and the "
        "final-layer CLS attention top-3 tokens. Calibrated at the Bayes "
        "ceiling on the eval corpus (39.18% top-1 on 3-way seat classification). "
        "Free side-call; does not advance the turn state. Example: "
        "belief_trajectory()."
    ),
    {},  # no arguments — belief is read from the current game state
    required=[],
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
    # belief_trajectory is a free side-call (like ask_rule): does not advance
    # the gate state, and is visible from every state so we can see whether
    # Gemma reaches for it on turn 1 vs later in the loop.
    GateState.INITIAL: [EXPLORE_GAME, BELIEF_TRAJECTORY],
    GateState.AFTER_EXPLORE: [
        EXPLORE_GAME, PROBE_BEST_CASE, PROBE_WORST_CASE, ASK_RULE, BELIEF_TRAJECTORY,
    ],
    GateState.AFTER_PROBE: [
        EXPLORE_GAME, PROBE_BEST_CASE, PROBE_WORST_CASE, ASK_RULE,
        BELIEF_TRAJECTORY, COMMIT_PLAY,
    ],
}


def menu_for(state: GateState) -> list[dict]:
    """Return the tool schema list visible in this state."""
    return list(_MENUS[state])


def menu_names(state: GateState) -> list[str]:
    return [s["function"]["name"] for s in _MENUS[state]]


# --------------------------------------------------------------------------- #
# Per-turn max-tokens policy (Phase 1, Lever 1)                                #
#                                                                              #
# Generation happens in waves through ``mlx_lm.batch_generate``, which accepts #
# a per-prompt ``max_tokens: List[int]`` and stops each stream the moment it   #
# either emits EOS or hits its own budget. Capping the budget per gate state   #
# does not change the model's outputs in any turn that finishes before the    #
# cap — it only trims the runaway tails. The 2048 floor is preserved for the  #
# final commit turn (state == AFTER_PROBE) per                                 #
# wiki/decisions/max-tokens-2048-floor.md.                                     #
# --------------------------------------------------------------------------- #


_TURN_AWARE_BUDGETS: dict[GateState, int] = {
    # Empirical: the existing 2048 floor ([[max-tokens-2048-floor]]) is
    # already a 4× cut from the bench's 8192 default. Tighter per-state
    # caps were tried on the perf-subset-5 (after the team serialized
    # benches to remove GPU contention) and consistently flipped
    # clean-commits to forced-commits because Gemma's wax_museum turn-1
    # reasoning is consistently 500-700 tokens at temp=0.6 / similar at
    # temp=0:
    #
    #   Spec               wall    K1 vs clean  forced-commits  output-changing?
    #   256 / 512 / 2048   30.7s   3/5 (60%)    2/5 (gi=36,104) yes (gi=104 6 → 19)
    #   768 / 768 / 2048   90.3s*  3/5 (60%)    2/5             yes
    #   1024/1024/2048    124.7s*  3/5 (60%)    1/5             yes
    #   2048/2048/2048     45.5s   5/5 (100%)   0/5             no
    #
    #   * earlier numbers were taken under GPU contention with scribe-A
    #     and over-state the wall; they remain in the ledger for the
    #     contention paper-trail.
    #
    # 2048-flat is the Phase 1 lever. The data structure is the lever;
    # the per-state knob stays so a future scribe can re-tune once the
    # corpus has more headroom.
    GateState.INITIAL: 2048,
    GateState.AFTER_EXPLORE: 2048,
    GateState.AFTER_PROBE: 2048,
}


def max_tokens_for_state(state: GateState) -> int:
    """Per-turn max generation token budget under the ``turn-aware`` policy.

    Lands at flat-2048 across all gate states on the perf-subset-5 because
    tighter caps changed the multi-turn trajectory (clean-commits flipped
    to forced-commits). See ``_TURN_AWARE_BUDGETS`` table for the
    measurements.
    """
    return _TURN_AWARE_BUDGETS[state]


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
    if tool_called == "belief_trajectory":
        return state  # free side-call; never advances
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
        {"tool": "belief_trajectory", "when": "Read Gus's calibrated posterior over opponent hands. Free; does not advance state."},
    ]


def next_actions_after_probe(play: int) -> list[NextAction]:
    return [
        {"tool": "commit_play", "when": f"Commit your final play. Call: commit_play(domino_id=<int>)."},
        {"tool": "probe_best_case", "when": f"Check the other branch for play={play}."},
        {"tool": "probe_worst_case", "when": f"Check the other branch for play={play}."},
        {"tool": "explore_game", "when": "Examine a different candidate play before committing."},
        {"tool": "ask_rule", "when": "Look up a 42 rule."},
        {"tool": "belief_trajectory", "when": "Read Gus's calibrated posterior over opponent hands."},
    ]


def next_actions_unchanged(state: GateState, play: int | None = None) -> list[NextAction]:
    """For ask_rule / belief_trajectory — re-advertise the current state's actions."""
    if state == GateState.INITIAL:
        return [
            {"tool": "explore_game", "when": "Examine a candidate play (pick any domino in your hand)."},
            {"tool": "belief_trajectory", "when": "Read Gus's calibrated posterior over opponent hands. Free; does not advance state."},
        ]
    if state == GateState.AFTER_EXPLORE:
        assert play is not None
        return next_actions_after_explore(play, has_bimodal=True)
    return next_actions_after_probe(play if play is not None else -1)


# --------------------------------------------------------------------------- #
# Smoke                                                                        #
# --------------------------------------------------------------------------- #


def _smoke() -> None:
    s = GateState.INITIAL
    assert set(menu_names(s)) == {"explore_game", "belief_trajectory"}, menu_names(s)
    s = advance(s, "explore_game")
    assert s == GateState.AFTER_EXPLORE
    assert set(menu_names(s)) == {
        "explore_game", "probe_best_case", "probe_worst_case",
        "ask_rule", "belief_trajectory",
    }
    # probe advances
    s2 = advance(s, "probe_worst_case")
    assert s2 == GateState.AFTER_PROBE
    assert "commit_play" in menu_names(s2)
    assert "belief_trajectory" in menu_names(s2)
    # ask_rule does not advance
    s3 = advance(s, "ask_rule")
    assert s3 == GateState.AFTER_EXPLORE
    # belief_trajectory does not advance, from any state
    assert advance(GateState.INITIAL, "belief_trajectory") == GateState.INITIAL
    assert advance(GateState.AFTER_EXPLORE, "belief_trajectory") == GateState.AFTER_EXPLORE
    assert advance(GateState.AFTER_PROBE, "belief_trajectory") == GateState.AFTER_PROBE
    # probe from INITIAL is a no-op (gate enforced)
    assert advance(GateState.INITIAL, "probe_best_case") == GateState.INITIAL
    print("[wax_museum.schemas] smoke OK")
    print(f"  INITIAL menu: {menu_names(GateState.INITIAL)}")
    print(f"  AFTER_EXPLORE menu: {menu_names(GateState.AFTER_EXPLORE)}")
    print(f"  AFTER_PROBE menu: {menu_names(GateState.AFTER_PROBE)}")


if __name__ == "__main__":
    _smoke()
