"""Drive loop test with a canned fake engine.

The fake engine yields canned event sequences. We assert:

1. drive() yields engine events verbatim — EngineStart, EngineToken,
   EngineToolCall, EngineError, EngineDone — plus the synthesised
   ToolResult on tool dispatch.
2. The ToolResult Move surfaces names from ``next_tools`` (HATEOAS) and
   the surfaced specs are added to the Registry but NOT auto-advertised.
3. After a commit-role tool dispatches, drive synthesises EngineCommit
   (engine never yields it per SPEC #28) and exits.
4. EngineError passes through verbatim, then drive exits on the
   following EngineDone(reason="aborted").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from burl.lab.core.drive import drive
from burl.lab.core.tool import Registry, ToolResult as ImplToolResult, ToolSpec
from burl.lab.core.transcript import (
    EngineCommit,
    EngineDone,
    EngineError,
    EngineStart,
    EngineToken,
    EngineToolCall,
    Stamp,
    State,
    ToolResult,
)


def _stamp(seq: int = 1) -> Stamp:
    return Stamp(t_wall_ms=seq, t_mono_ns=seq * 1000)


class FakeEngine:
    """Yields canned event sequences across multiple step() invocations."""

    def __init__(self, scripts: list[list]):
        self.scripts = list(scripts)
        self.calls = 0
        self.last_messages: list[list[dict]] = []
        self.last_tools: list[list[ToolSpec]] = []

    async def step(  # noqa: D401
        self,
        messages: list[dict],
        tools: list[ToolSpec],
        max_tokens: int = 2048,  # noqa: ARG002
    ) -> AsyncIterator[Any]:
        self.last_messages.append(list(messages))
        self.last_tools.append(list(tools))
        if self.calls >= len(self.scripts):
            return
        script = self.scripts[self.calls]
        self.calls += 1
        for ev in script:
            yield ev


def _spec(name: str, role: str, fn: Any = None) -> ToolSpec:
    def default_impl(ctx: Any, args: dict) -> ImplToolResult:
        return ImplToolResult(evidence={"prose": "noop", "structured": {}})

    return ToolSpec(
        name=name,
        description="",
        params={"type": "object", "properties": {}},
        example=f"{name}()",
        protocol_role=role,  # type: ignore[arg-type]
        protocol_phrase="",
        impl=fn or default_impl,
    )


def _empty_state(advertised: tuple[str, ...] = ()) -> State:
    return State(
        session_dir=Path("/tmp/test-drive"),
        phase="in_run",
        messages=({"role": "system", "content": "S"},),
        active_tools=advertised,
        advertised=advertised,
        segments=(),
        cum_tok_in=0,
        cum_tok_out=0,
        started_mono_ns=0,
        started_wall_ns=0,
    )


@pytest.mark.asyncio
async def test_drive_dispatches_tool_call_and_loops_until_done():
    surfaced_spec = _spec("surfaced_followup", "candidate_eval")
    captured: dict = {}

    def read_state_impl(ctx: Any, args: dict) -> ImplToolResult:
        captured["called"] = True
        captured["args"] = dict(args)
        return ImplToolResult(
            evidence={"prose": "hand: 5-3 4-2", "structured": {"hand": "5-3 4-2"}},
            next_tools=(surfaced_spec,),
        )

    registry = Registry()
    registry.add(_spec("read_state", "first_read", fn=read_state_impl))

    engine = FakeEngine(
        [
            # First step: the engine yields a tool_call.
            [
                EngineStart(
                    stamp=_stamp(1), messages_hash="aaa", n_messages=2, n_tools=1
                ),
                EngineToken(stamp=_stamp(2), text="thinking..."),
                EngineToolCall(
                    stamp=_stamp(3),
                    name="read_state",
                    args={},
                    call_id="call-1",
                ),
                EngineDone(stamp=_stamp(4), reason="tool_dispatch"),
            ],
            # Second step (re-entered after dispatch): final answer.
            [
                EngineStart(
                    stamp=_stamp(5), messages_hash="bbb", n_messages=3, n_tools=2
                ),
                EngineToken(stamp=_stamp(6), text="ok i see the hand."),
                EngineDone(stamp=_stamp(7), reason="done"),
            ],
        ]
    )

    state = _empty_state(advertised=("read_state",))

    moves: list = []
    async for mv in drive(state, registry, engine):
        moves.append(mv)

    # Engine ran twice (re-entered after tool dispatch).
    assert engine.calls == 2

    # Tool was actually invoked.
    assert captured.get("called") is True

    # Verify Move sequence types.
    types = [type(mv).__name__ for mv in moves]
    assert types == [
        "EngineStart",
        "EngineToken",
        "EngineToolCall",
        "EngineDone",
        "ToolResult",
        "EngineStart",
        "EngineToken",
        "EngineDone",
    ]

    # The synthesised ToolResult carries HATEOAS surface.
    tr: ToolResult = next(m for m in moves if isinstance(m, ToolResult))
    assert tr.name == "read_state"
    assert tr.call_id == "call-1"
    assert tr.evidence["prose"] == "hand: 5-3 4-2"
    assert tr.next_tools == ["surfaced_followup"]

    # HATEOAS: surfaced spec is registered ...
    assert registry.find("surfaced_followup") is not None
    # ... but the user-facing State that was passed in did NOT have its
    # advertised set mutated (drive is non-mutating w.r.t. State).
    assert "surfaced_followup" not in state.advertised


@pytest.mark.asyncio
async def test_drive_terminates_cleanly_without_tool_call():
    """Engine yields tokens + EngineDone(reason="done"), no tool_call."""
    registry = Registry()
    engine = FakeEngine(
        [
            [
                EngineStart(
                    stamp=_stamp(1), messages_hash="x", n_messages=1, n_tools=0
                ),
                EngineToken(stamp=_stamp(2), text="hi"),
                EngineDone(stamp=_stamp(3), reason="done"),
            ]
        ]
    )

    state = _empty_state()
    moves = [mv async for mv in drive(state, registry, engine)]
    assert engine.calls == 1
    assert [type(m).__name__ for m in moves] == [
        "EngineStart",
        "EngineToken",
        "EngineDone",
    ]


@pytest.mark.asyncio
async def test_drive_handles_unknown_tool_with_synthetic_result():
    """When the engine asks for a tool the registry doesn't know, drive
    synthesises an error ToolResult so the model sees something."""
    registry = Registry()
    engine = FakeEngine(
        [
            [
                EngineStart(
                    stamp=_stamp(1), messages_hash="x", n_messages=1, n_tools=0
                ),
                EngineToolCall(
                    stamp=_stamp(2),
                    name="not_registered",
                    args={},
                    call_id="call-z",
                ),
                EngineDone(stamp=_stamp(3), reason="tool_dispatch"),
            ],
            [
                EngineStart(
                    stamp=_stamp(4), messages_hash="y", n_messages=2, n_tools=0
                ),
                EngineDone(stamp=_stamp(5), reason="done"),
            ],
        ]
    )
    state = _empty_state()
    moves = [mv async for mv in drive(state, registry, engine)]
    tr = next(m for m in moves if isinstance(m, ToolResult))
    assert tr.name == "not_registered"
    assert "not registered" in tr.evidence["prose"].lower()
    assert tr.next_tools == []


@pytest.mark.asyncio
async def test_drive_synthesizes_engine_commit_for_commit_role_tool():
    """A commit-role tool dispatch produces a harness-synthesised
    EngineCommit, and drive exits without re-entering the engine."""

    captured: dict = {}

    def commit_play_impl(ctx: Any, args: dict) -> ImplToolResult:
        captured["args"] = dict(args)
        return ImplToolResult(
            evidence={
                "prose": f"COMMIT: domino_id={args.get('domino_id')} recorded.",
                "structured": {"committed": args.get("domino_id")},
            },
            next_tools=(),
            next_phase="post_turn",
        )

    registry = Registry()
    registry.add(_spec("commit_play", "commit", fn=commit_play_impl))

    engine = FakeEngine(
        [
            [
                EngineStart(
                    stamp=_stamp(1), messages_hash="aaa", n_messages=2, n_tools=1
                ),
                EngineToken(stamp=_stamp(2), text="committing"),
                EngineToolCall(
                    stamp=_stamp(3),
                    name="commit_play",
                    args={"domino_id": 14},
                    call_id="call-c1",
                ),
                EngineDone(stamp=_stamp(4), reason="tool_dispatch"),
            ],
            # If drive incorrectly re-enters the engine, this script would
            # be consumed and engine.calls would be 2. We assert it stays at 1.
            [
                EngineStart(
                    stamp=_stamp(5), messages_hash="bbb", n_messages=3, n_tools=1
                ),
                EngineDone(stamp=_stamp(6), reason="done"),
            ],
        ]
    )

    state = _empty_state(advertised=("commit_play",))
    moves = [mv async for mv in drive(state, registry, engine)]

    # Drive must NOT re-enter the engine after a commit-role dispatch.
    assert engine.calls == 1

    # The impl saw the args.
    assert captured["args"] == {"domino_id": 14}

    # Move sequence: engine events + ToolResult + harness EngineCommit.
    # PhaseExit/PhaseEnter are NOT minted here — drive is engine-shaped;
    # the server calls `in_run.handle(state, EngineCommit)` after drive
    # returns and journals the transition itself.
    type_names = [type(m).__name__ for m in moves]
    assert type_names == [
        "EngineStart",
        "EngineToken",
        "EngineToolCall",
        "EngineDone",
        "ToolResult",
        "EngineCommit",
    ]

    commit = moves[-1]
    assert isinstance(commit, EngineCommit)
    # `final` carries the args from the original tool_call.
    assert commit.final == {"domino_id": 14}


@pytest.mark.asyncio
async def test_drive_passes_engine_error_through_then_exits():
    """EngineError yields verbatim; drive exits on the trailing
    EngineDone(reason="aborted")."""
    registry = Registry()
    engine = FakeEngine(
        [
            [
                EngineStart(
                    stamp=_stamp(1), messages_hash="x", n_messages=1, n_tools=0
                ),
                EngineToken(stamp=_stamp(2), text="partial..."),
                EngineError(
                    stamp=_stamp(3),
                    message="MLX raised: oh no",
                    traceback="Traceback (most recent call last):\n...",
                    during=None,
                ),
                EngineDone(stamp=_stamp(4), reason="aborted"),
            ]
        ]
    )

    state = _empty_state()
    moves = [mv async for mv in drive(state, registry, engine)]
    assert engine.calls == 1

    type_names = [type(m).__name__ for m in moves]
    assert type_names == [
        "EngineStart",
        "EngineToken",
        "EngineError",
        "EngineDone",
    ]

    err = next(m for m in moves if isinstance(m, EngineError))
    assert err.message == "MLX raised: oh no"
    assert err.during is None
