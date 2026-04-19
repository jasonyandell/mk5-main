"""Flag-routing tests for the native agent runner's rules-as-tools switch.

Covers:
- registry composition in both flag states
- JSON-schema composition in both flag states
- system-prompt composition in both flag states (preamble vs trimmed primer)
- end-to-end stub-model decision with flag=True calls a rules tool

All local CPU; no Modal, no network, no training.
"""

from __future__ import annotations

import random

import pytest

from forge.zeb.game import apply_action, legal_actions, new_game

from burl.harness.agent_runner import build_tool_registry
from burl.harness.agent_runner_native import (
    _RULES_AS_TOOLS_PREAMBLE,
    _RULES_TOOL_SCHEMAS,
    _TRIMMED_PRIMER,
    TOOL_SCHEMAS,
    build_tool_schemas,
    render_native_messages,
    run_decision_native,
)
from burl.harness.agent_runner import _current_player, _visible_history


_RULES_TOOL_NAMES = {
    "count_dominoes_remaining",
    "trick_winner_if",
    "what_beats_what",
    "contract_progress",
}

_BASELINE_TOOL_NAMES = {
    "is_legal",
    "is_trump",
    "unseen",
    "void_audit",
    "trump_declared",
    "eq_outcome_distribution",
    "conditional_outcome",
}


# --------------------------------------------------------------------------- #
# Registry composition                                                         #
# --------------------------------------------------------------------------- #


def test_registry_excludes_rules_tools_by_default():
    registry = build_tool_registry()
    assert set(registry) == _BASELINE_TOOL_NAMES
    assert not (_RULES_TOOL_NAMES & set(registry)), (
        "rules tools must not appear when flag defaults to False"
    )


def test_registry_excludes_rules_tools_when_flag_false():
    registry = build_tool_registry(enable_rules_tools=False)
    assert set(registry) == _BASELINE_TOOL_NAMES


def test_registry_includes_rules_tools_when_flag_true():
    registry = build_tool_registry(enable_rules_tools=True)
    assert _RULES_TOOL_NAMES.issubset(set(registry))
    assert _BASELINE_TOOL_NAMES.issubset(set(registry))
    assert len(registry) == len(_BASELINE_TOOL_NAMES) + len(_RULES_TOOL_NAMES)


def test_rules_tools_callable_from_registry():
    """Flag=True tools must be invokable against a real game_state."""
    registry = build_tool_registry(enable_rules_tools=True)
    state = new_game(seed=2026, skip_bidding=True)

    out = registry["count_dominoes_remaining"](state)
    assert out["loose_count_points"] == 35

    out = registry["contract_progress"](state)
    assert out["status"] == "contract_in_play"

    # trick_winner_if with me leading a fresh trick.
    me = state.trick_leader
    d = state.hands[me][0]
    out = registry["trick_winner_if"](state, domino_id=int(d))
    assert out["i_am_leading"] is True

    # what_beats_what requires an explicit lead when no trick is in play.
    hand = state.hands[me]
    if len(hand) >= 2:
        out = registry["what_beats_what"](
            state,
            domino_a=int(hand[0]),
            domino_b=int(hand[1]),
            lead_domino=int(hand[0]),
        )
        assert out["winner"] in ("a", "b", "neither")


# --------------------------------------------------------------------------- #
# Schema composition                                                           #
# --------------------------------------------------------------------------- #


def test_schemas_default_match_baseline_tool_schemas():
    schemas = build_tool_schemas()
    assert schemas == list(TOOL_SCHEMAS)
    names = {s["function"]["name"] for s in schemas}
    assert not (_RULES_TOOL_NAMES & names)


def test_schemas_flag_true_appends_rules_schemas():
    schemas = build_tool_schemas(enable_rules_tools=True)
    names = [s["function"]["name"] for s in schemas]
    assert len(names) == len(set(names)), "duplicate schema names"
    assert _RULES_TOOL_NAMES.issubset(set(names))
    # Baseline schemas still present and in the original position.
    baseline_names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert names[: len(baseline_names)] == baseline_names


def test_rules_schemas_have_required_shape():
    """Each rules schema must mirror the {type:function, function:{name,
    description, parameters:{type:object, properties:{}, ...}}} shape the
    chat template expects."""
    for schema in _RULES_TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and len(fn["description"]) > 20
        params = fn["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)


# --------------------------------------------------------------------------- #
# Prompt composition                                                           #
# --------------------------------------------------------------------------- #


@pytest.fixture
def decision_state():
    state = new_game(seed=2026, skip_bidding=True)
    # Advance a couple of plays so the user prompt exercises a partial trick.
    rng = random.Random(0)
    for _ in range(2):
        slots = legal_actions(state)
        if not slots:
            break
        state = apply_action(state, rng.choice(slots))
    return state


def test_prompt_uses_trimmed_primer_by_default(decision_state):
    me = _current_player(decision_state)
    hand = [d for d in decision_state.hands[me] if d not in decision_state.played]
    history = _visible_history(decision_state)
    system, _user = render_native_messages(decision_state, hand, history)

    # Trimmed primer markers must be present; rules-as-tools preamble markers
    # must not.
    assert "How trump works" in system
    assert "Following suit" in system
    assert "rule-answering tools" not in system
    assert "what_beats_what" not in system
    # Trimmed primer verbatim token.
    assert _TRIMMED_PRIMER.splitlines()[0] in system


def test_prompt_swaps_to_rules_preamble_when_flag_true(decision_state):
    me = _current_player(decision_state)
    hand = [d for d in decision_state.hands[me] if d not in decision_state.played]
    history = _visible_history(decision_state)
    system, _user = render_native_messages(
        decision_state, hand, history, enable_rules_tools=True,
    )

    # Rules-as-tools preamble markers present; trimmed primer markers absent.
    assert "rule-answering tools" in system
    assert "what_beats_what" in system
    assert "trick_winner_if" in system
    assert "count_dominoes_remaining" in system
    assert "contract_progress" in system
    assert "How trump works" not in system
    assert "Following suit" not in system
    # 42 framing must still be there regardless of flag.
    assert "Texas 42 framing" in system
    # Verbatim preamble present.
    assert _RULES_AS_TOOLS_PREAMBLE.splitlines()[0] in system


def test_prompt_42_framing_unchanged_by_flag(decision_state):
    me = _current_player(decision_state)
    hand = [d for d in decision_state.hands[me] if d not in decision_state.played]
    history = _visible_history(decision_state)
    sys_off, _ = render_native_messages(decision_state, hand, history)
    sys_on, _ = render_native_messages(
        decision_state, hand, history, enable_rules_tools=True,
    )

    # Slice out the 42-framing section; it should be identical in both.
    marker = "=== Texas 42 framing ==="
    assert marker in sys_off and marker in sys_on
    assert sys_off.split(marker, 1)[1] == sys_on.split(marker, 1)[1]


def test_preamble_is_compact():
    """Guard against future drift blowing past the design-doc byte target."""
    # 645-byte target; allow some headroom for future wording tweaks.
    assert len(_RULES_AS_TOOLS_PREAMBLE.encode("utf-8")) < 1000


# --------------------------------------------------------------------------- #
# Primer-off mode (spike-v2 / iter-3-v2 shape)                                 #
# --------------------------------------------------------------------------- #


def test_prompt_primer_off_drops_both_primer_and_rules_preamble(decision_state):
    """enable_primer=False: system message has 42-framing but neither the
    trimmed primer nor the rules-as-tools preamble."""
    me = _current_player(decision_state)
    hand = [d for d in decision_state.hands[me] if d not in decision_state.played]
    history = _visible_history(decision_state)
    system, _user = render_native_messages(
        decision_state, hand, history, enable_primer=False,
    )

    # Primer / rules preamble markers both absent.
    assert "How trump works" not in system
    assert "Following suit" not in system
    assert "rule-answering tools" not in system
    assert "what_beats_what" not in system
    assert _TRIMMED_PRIMER.splitlines()[0] not in system
    assert _RULES_AS_TOOLS_PREAMBLE.splitlines()[0] not in system
    # 42-framing still present.
    assert "Texas 42 framing" in system


def test_three_mode_matrix_renders_distinct_system_strings(decision_state):
    """default / rules-on / primer-off produce three distinct system strings,
    but share the same 42-framing tail."""
    me = _current_player(decision_state)
    hand = [d for d in decision_state.hands[me] if d not in decision_state.played]
    history = _visible_history(decision_state)

    sys_default, _ = render_native_messages(decision_state, hand, history)
    sys_rules, _ = render_native_messages(
        decision_state, hand, history, enable_rules_tools=True,
    )
    sys_no_primer, _ = render_native_messages(
        decision_state, hand, history, enable_primer=False,
    )

    # Three distinct shapes.
    assert sys_default != sys_rules
    assert sys_default != sys_no_primer
    assert sys_rules != sys_no_primer

    # Ordering of length: no_primer < default, no_primer < rules.
    assert len(sys_no_primer) < len(sys_default)
    assert len(sys_no_primer) < len(sys_rules)

    # 42-framing tail is byte-identical across all three.
    marker = "=== Texas 42 framing ==="
    tail_default = sys_default.split(marker, 1)[1]
    tail_rules = sys_rules.split(marker, 1)[1]
    tail_no_primer = sys_no_primer.split(marker, 1)[1]
    assert tail_default == tail_rules == tail_no_primer


def test_primer_off_with_rules_tools_true_raises(decision_state):
    """The incoherent combo is rejected loudly rather than silently
    rendering a shape no-one designed."""
    me = _current_player(decision_state)
    hand = [d for d in decision_state.hands[me] if d not in decision_state.played]
    history = _visible_history(decision_state)
    with pytest.raises(ValueError, match="enable_primer=True"):
        render_native_messages(
            decision_state, hand, history,
            enable_primer=False, enable_rules_tools=True,
        )


def test_run_decision_native_threads_enable_primer(decision_state):
    """run_decision_native must propagate enable_primer=False so that the
    stub model sees a system message with no primer text."""
    state = new_game(seed=2026, skip_bidding=True)
    me = _current_player(state)
    hand = [d for d in state.hands[me] if d not in state.played]
    target = int(hand[0])

    captured_system: list[str] = []
    script = iter([
        f'<|tool_call>{{"name":"commit_play","arguments":{{"domino_id":{target}}}}}<tool_call|>',
    ])

    def stub(messages: list[dict], tools: list[dict]) -> str:
        for m in messages:
            if m.get("role") == "system":
                captured_system.append(m["content"])
                break
        return next(script)

    trace = run_decision_native(
        state, stub, max_turns=4, max_retries=2, enable_primer=False,
    )

    assert trace.final_play == target
    assert captured_system, "stub never saw a system message"
    system = captured_system[0]
    assert "How trump works" not in system
    assert "rule-answering tools" not in system
    assert "Texas 42 framing" in system


# --------------------------------------------------------------------------- #
# End-to-end: stub model drives a decision with flag=True                      #
# --------------------------------------------------------------------------- #


def test_run_decision_native_with_flag_true_dispatches_rules_tool():
    """Scripted stub model: call contract_progress, then commit_play with a
    legal play. Asserts the rules tool resolved and the commit landed."""
    state = new_game(seed=2026, skip_bidding=True)
    # Stay at trick 1 leading — simplest geometry; any hand domino is legal.
    me = _current_player(state)
    hand = [d for d in state.hands[me] if d not in state.played]
    target = int(hand[0])

    script = iter([
        '<|tool_call>{"name":"contract_progress","arguments":{}}<tool_call|>',
        f'<|tool_call>{{"name":"commit_play","arguments":{{"domino_id":{target}}}}}<tool_call|>',
    ])

    seen_schema_names: list[str] = []

    def stub(messages: list[dict], tools: list[dict]) -> str:
        # Record the tool menu so we can assert flag propagation.
        seen_schema_names.extend(t["function"]["name"] for t in tools)
        return next(script)

    trace = run_decision_native(
        state,
        stub,
        max_turns=4,
        max_retries=2,
        enable_rules_tools=True,
    )

    assert trace.final_play == target
    # The stub saw schemas on at least one call; the rules tools must be there.
    assert _RULES_TOOL_NAMES.issubset(set(seen_schema_names))
    # Tool call history should include the rules tool we scripted.
    tool_names = [tc.tool_name for turn in trace.turns for tc in turn.tool_calls]
    assert "contract_progress" in tool_names


def test_run_decision_native_flag_false_does_not_expose_rules_tools():
    """Opposite direction: flag=False means the stub never sees rules schemas."""
    state = new_game(seed=2026, skip_bidding=True)
    me = _current_player(state)
    hand = [d for d in state.hands[me] if d not in state.played]
    target = int(hand[0])

    script = iter([
        f'<|tool_call>{{"name":"commit_play","arguments":{{"domino_id":{target}}}}}<tool_call|>',
    ])

    seen_schema_names: list[str] = []

    def stub(messages: list[dict], tools: list[dict]) -> str:
        seen_schema_names.extend(t["function"]["name"] for t in tools)
        return next(script)

    trace = run_decision_native(
        state,
        stub,
        max_turns=4,
        max_retries=2,
        enable_rules_tools=False,
    )

    assert trace.final_play == target
    assert not (_RULES_TOOL_NAMES & set(seen_schema_names)), (
        f"rules schemas leaked when flag=False: {set(seen_schema_names)}"
    )
