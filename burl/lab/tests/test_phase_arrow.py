"""Focused tests for the low-level phase Trace contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from burl.lab.core.arrow import Trace
from burl.lab.core.transcript import (
    AdvertisedSet,
    EngineCommit,
    Stamp,
    State,
    SystemSet,
    ToolAdded,
    UserChoice,
    UserText,
)
from burl.lab.phases.in_run import IN_RUN
from burl.lab.phases.post_turn import POST_TURN
from burl.lab.phases import pre_game as pre_game_module
from burl.lab.phases.pre_game import DEFAULT_BASE_SYSTEM, PRE_GAME


def _stamp(seq: int = 0) -> Stamp:
    return Stamp(t_wall_ms=seq, t_mono_ns=seq * 1000)


def _state(
    tmp_path: Path,
    *,
    phase: str = "pre_game",
    active_tools: tuple[str, ...] = (),
    advertised: tuple[str, ...] = (),
) -> State:
    return State(
        session_dir=tmp_path,
        phase=phase,
        messages=(),
        active_tools=active_tools,
        advertised=advertised,
        segments=(),
        cum_tok_in=0,
        cum_tok_out=0,
        started_mono_ns=0,
        started_wall_ns=0,
    )


def test_trace_composes_events_and_optional_output() -> None:
    first = UserText(stamp=_stamp(1), text="first")
    second = UserText(stamp=_stamp(2), text="second")

    trace = Trace.emit(first).append(second).map(lambda value: value)
    assert trace.events == (first, second)
    assert trace.output is None

    chained = Trace(events=(first,), output="next").then(
        lambda phase: Trace(events=(second,), output=f"{phase}:done")
    )
    assert chained.events == (first, second)
    assert chained.output == "next:done"


@pytest.mark.asyncio
async def test_pre_game_returns_config_moves_without_writing(tmp_path: Path) -> None:
    state = _state(tmp_path, active_tools=("belief_trajectory",), advertised=())
    move = UserChoice(
        stamp=_stamp(1),
        option_name="set_advertised",
        args={"names": ["belief_trajectory", "explore_game"]},
    )

    trace = await PRE_GAME.handle(state, move)

    assert [type(event) for event in trace.events] == [ToolAdded, AdvertisedSet]
    assert trace.events[0].name == "explore_game"
    assert trace.events[1].names == ["belief_trajectory", "explore_game"]
    assert trace.output is None
    assert not (tmp_path / "events.jsonl").exists()


@pytest.mark.asyncio
async def test_pre_game_load_decision_fills_builder_without_starting_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pre_game_module,
        "_load_decision_prompts",
        lambda harvest, idx: (
            f"system {harvest}:{idx}",
            f"loaded {harvest}:{idx}",
        ),
    )
    state = _state(tmp_path)
    move = UserChoice(
        stamp=_stamp(1),
        option_name="load_decision",
        args={"harvest": "h", "idx": 7},
    )

    trace = await PRE_GAME.handle(state, move)

    assert len(trace.events) == 2
    assert isinstance(trace.events[0], SystemSet)
    assert trace.events[0].text == "system h:7"
    assert isinstance(trace.events[1], UserText)
    assert trace.events[1].text == "loaded h:7"
    assert trace.output is None
    assert not (tmp_path / "events.jsonl").exists()


@pytest.mark.asyncio
async def test_pre_game_send_seeded_decision_uses_existing_system_and_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pre_game_module,
        "_load_seeded_decision_prompts",
        lambda harvest, seed: (
            f"harvested system {harvest}:{seed}",
            f"seed prompt {harvest}:{seed}",
        ),
    )
    state = State(
        session_dir=tmp_path,
        phase="pre_game",
        messages=({"role": "system", "content": "custom system"},),
        active_tools=(),
        advertised=(),
        segments=(),
        cum_tok_in=0,
        cum_tok_out=0,
        started_mono_ns=0,
        started_wall_ns=0,
    )
    move = UserChoice(
        stamp=_stamp(1),
        option_name="send_seeded_decision",
        args={"harvest": "h", "seed": 42},
    )

    trace = await PRE_GAME.handle(state, move)

    assert trace.events == (
        UserText(stamp=trace.events[0].stamp, text="seed prompt h:42"),
    )
    assert trace.output == "in_run"
    assert not (tmp_path / "events.jsonl").exists()


@pytest.mark.asyncio
async def test_pre_game_send_seeded_decision_seeds_default_system_if_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pre_game_module,
        "_load_seeded_decision_prompts",
        lambda harvest, seed: ("ignored system", f"seed prompt {seed}"),
    )
    state = _state(tmp_path)
    move = UserChoice(
        stamp=_stamp(1),
        option_name="send_seeded_decision",
        args={"harvest": "h", "seed": 7},
    )

    trace = await PRE_GAME.handle(state, move)

    assert [type(event) for event in trace.events] == [SystemSet, UserText]
    assert trace.events[0].text == DEFAULT_BASE_SYSTEM
    assert trace.events[1].text == "seed prompt 7"
    assert trace.output == "in_run"


def test_load_decision_strips_legacy_tool_protocol() -> None:
    raw = (
        "You are Burl.\n\n"
        "# Current decision\n\n"
        "state facts\n"
        "\n# Decision protocol (wax_museum)\n\n"
        "old protocol<|tool>declaration:explore_game{}<tool|>"
    )

    assert pre_game_module._strip_legacy_tool_protocol(raw) == (
        "You are Burl.\n\n# Current decision\n\nstate facts"
    )


@pytest.mark.asyncio
async def test_pre_game_set_system_returns_system_set(tmp_path: Path) -> None:
    state = _state(tmp_path)
    move = UserChoice(
        stamp=_stamp(1),
        option_name="set_system",
        args={"text": "You are Burl."},
    )

    trace = await PRE_GAME.handle(state, move)

    assert trace.events == (SystemSet(stamp=trace.events[0].stamp, text="You are Burl."),)
    assert trace.output is None


@pytest.mark.asyncio
async def test_pre_game_generate_system_seeds_default_prompt(tmp_path: Path) -> None:
    state = _state(tmp_path)
    move = UserChoice(
        stamp=_stamp(1),
        option_name="generate_system",
        args={},
    )

    trace = await PRE_GAME.handle(state, move)

    assert trace.events == (
        SystemSet(stamp=trace.events[0].stamp, text=DEFAULT_BASE_SYSTEM),
    )
    assert trace.output is None


@pytest.mark.asyncio
async def test_pre_game_start_run_without_user_text_stays_in_builder(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    move = UserChoice(
        stamp=_stamp(1),
        option_name="start_run",
        args={},
    )

    trace = await PRE_GAME.handle(state, move)

    assert trace.events == ()
    assert trace.output is None


@pytest.mark.asyncio
async def test_pre_game_start_run_with_user_text_transitions(tmp_path: Path) -> None:
    state = State(
        session_dir=tmp_path,
        phase="pre_game",
        messages=({"role": "user", "content": "loaded decision"},),
        active_tools=(),
        advertised=(),
        segments=(),
        cum_tok_in=0,
        cum_tok_out=0,
        started_mono_ns=0,
        started_wall_ns=0,
    )
    move = UserChoice(
        stamp=_stamp(1),
        option_name="start_run",
        args={},
    )

    trace = await PRE_GAME.handle(state, move)

    assert trace.events == (
        SystemSet(stamp=trace.events[0].stamp, text=DEFAULT_BASE_SYSTEM),
    )
    assert trace.output == "in_run"


@pytest.mark.asyncio
async def test_pre_game_ask_gemma_adds_user_text_and_starts(tmp_path: Path) -> None:
    state = _state(tmp_path)
    move = UserChoice(
        stamp=_stamp(1),
        option_name="ask_gemma",
        args={"text": "look at this setup"},
    )

    trace = await PRE_GAME.handle(state, move)

    assert [type(event) for event in trace.events] == [SystemSet, UserText]
    assert trace.events[0].text == DEFAULT_BASE_SYSTEM
    assert trace.events[1].text == "look at this setup"
    assert trace.output == "in_run"


@pytest.mark.asyncio
async def test_in_run_returns_effect_moves_and_transitions(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        phase="in_run",
        active_tools=("belief_trajectory", "explore_game"),
        advertised=("belief_trajectory",),
    )

    interject = await IN_RUN.handle(
        state,
        UserChoice(
            stamp=_stamp(1),
            option_name="interject",
            args={"text": "hold on"},
        ),
    )
    assert interject.events == (
        UserText(stamp=interject.events[0].stamp, text="hold on"),
    )
    assert interject.output is None

    select_tool = await IN_RUN.handle(
        state,
        UserChoice(
            stamp=_stamp(2),
            option_name="select_tool",
            args={"name": "explore_game"},
        ),
    )
    assert select_tool.events == (
        AdvertisedSet(
            stamp=select_tool.events[0].stamp,
            names=["belief_trajectory", "explore_game"],
        ),
    )
    assert select_tool.output is None

    abort = await IN_RUN.handle(
        state, UserChoice(stamp=_stamp(3), option_name="abort", args={})
    )
    assert abort.events == ()
    assert abort.output == "pre_game"

    commit = await IN_RUN.handle(
        state, EngineCommit(stamp=_stamp(4), final={"domino_id": 14})
    )
    assert commit.events == ()
    assert commit.output == "post_turn"
    assert not (tmp_path / "events.jsonl").exists()


@pytest.mark.asyncio
async def test_post_turn_start_new_session_outputs_pre_game(tmp_path: Path) -> None:
    state = _state(tmp_path, phase="post_turn")
    move = UserChoice(
        stamp=_stamp(1),
        option_name="start_new_session",
        args={},
    )

    trace = await POST_TURN.handle(state, move)

    assert trace.events == ()
    assert trace.output == "pre_game"
