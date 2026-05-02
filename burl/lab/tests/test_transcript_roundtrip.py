"""Round-trip + fold sanity for burl/lab/core/transcript.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from burl.lab.core.transcript import (
    AdvertisedSet,
    EngineCommit,
    EngineDone,
    EngineError,
    EngineStart,
    EngineToken,
    EngineToolCall,
    PhaseEnter,
    PhaseExit,
    Stamp,
    SystemSet,
    ToolAdded,
    ToolRemoved,
    ToolResult,
    UserChoice,
    UserText,
    append,
    fold,
    replay,
)


def _stamp(t_ms: int = 0, mono: int = 0, *, tok_in: int = 0, tok_out: int = 0,
           cum_in: int = 0, cum_out: int = 0) -> Stamp:
    return Stamp(
        t_wall_ms=t_ms,
        t_mono_ns=mono,
        tok_in=tok_in,
        tok_out=tok_out,
        tok_cum_in=cum_in,
        tok_cum_out=cum_out,
    )


def _representative_moves() -> list:
    return [
        PhaseEnter(stamp=_stamp(0, 1_000), phase="pre_game"),
        UserChoice(
            stamp=_stamp(10, 2_000),
            option_name="start_run",
            args={"hand": "66 55 44 33 22 11 00"},
        ),
        PhaseExit(stamp=_stamp(20, 3_000), phase="pre_game"),
        PhaseEnter(stamp=_stamp(21, 3_100), phase="in_run"),
        UserText(stamp=_stamp(30, 4_000), text="Ready when you are."),
        EngineStart(
            stamp=_stamp(40, 5_000, cum_in=128, cum_out=0),
            messages_hash="abc123",
            n_messages=3,
            n_tools=2,
        ),
        EngineToken(stamp=_stamp(50, 6_000, tok_out=4, cum_in=128, cum_out=4), text="Looking "),
        EngineToken(stamp=_stamp(55, 6_500, tok_out=3, cum_in=128, cum_out=7), text="at hand."),
        EngineToolCall(
            stamp=_stamp(60, 7_000),
            name="belief_trajectory",
            args={"play": 14},
            call_id="call-001",
        ),
        EngineDone(stamp=_stamp(61, 7_100), reason="tool_dispatch"),
        ToolResult(
            stamp=_stamp(70, 8_000),
            name="belief_trajectory",
            call_id="call-001",
            evidence={"prose": "EV +0.42", "structured": {"ev": 0.42}},
            next_tools=["explore_game"],
        ),
        EngineStart(
            stamp=_stamp(80, 9_000, cum_in=200, cum_out=7),
            messages_hash="def456",
            n_messages=5,
            n_tools=3,
        ),
        EngineToken(stamp=_stamp(85, 9_500, tok_out=2, cum_in=200, cum_out=9), text="Commit."),
        EngineCommit(stamp=_stamp(90, 10_000), final={"play": 14}),
        EngineDone(stamp=_stamp(91, 10_100), reason="done"),
    ]


def test_append_replay_roundtrip(tmp_path: Path) -> None:
    moves = _representative_moves()
    for m in moves:
        append(tmp_path, m)

    recovered = list(replay(tmp_path))
    assert len(recovered) == len(moves)
    for original, restored in zip(moves, recovered):
        assert original == restored, f"mismatch on {type(original).__name__}: {original} vs {restored}"


def test_fold_invariants(tmp_path: Path) -> None:
    moves = _representative_moves()
    state = fold(moves, session_dir=tmp_path)

    assert state.session_dir == tmp_path
    assert state.phase == "in_run"

    # cum tokens take the max we saw on any stamp
    assert state.cum_tok_in == 200
    assert state.cum_tok_out == 9

    # session-start anchors come from the first move's stamp
    assert state.started_wall_ns == 0
    assert state.started_mono_ns == 1_000

    # messages: UserText (no UserChoice — UserChoice is a segment, not a chat msg)
    # plus the tool-result message from ToolResult
    user_msgs = [m for m in state.messages if m["role"] == "user"]
    tool_msgs = [m for m in state.messages if m["role"] == "tool"]
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "Ready when you are."
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["name"] == "belief_trajectory"
    assert tool_msgs[0]["call_id"] == "call-001"
    assert tool_msgs[0]["content"] == "EV +0.42"

    # HATEOAS: next_tools name landed in active + advertised
    assert "explore_game" in state.active_tools
    assert "explore_game" in state.advertised

    # tool_call segment has evidence inlined after ToolResult
    tool_call_segs = [s for s in state.segments if s.get("kind") == "tool_call"]
    assert len(tool_call_segs) == 1
    assert tool_call_segs[0]["evidence"] == {"prose": "EV +0.42", "structured": {"ev": 0.42}}
    assert tool_call_segs[0]["next_tools"] == ["explore_game"]

    # assistant_text segments accumulate EngineToken chunks
    assistant_segs = [s for s in state.segments if s.get("kind") == "assistant_text"]
    # two EngineStart events => two assistant_text segments
    assert len(assistant_segs) == 2
    assert assistant_segs[0]["text"] == "Looking at hand."
    assert assistant_segs[1]["text"] == "Commit."

    # commit + engine_done segments present
    assert any(s.get("kind") == "commit" and s.get("final") == {"play": 14} for s in state.segments)
    done_segs = [s for s in state.segments if s.get("kind") == "engine_done"]
    assert [s["reason"] for s in done_segs] == ["tool_dispatch", "done"]


def test_engine_error_roundtrip_and_fold(tmp_path: Path) -> None:
    """EngineError carries a message + optional traceback/during; pairs with EngineDone(aborted)."""
    moves = [
        PhaseEnter(stamp=_stamp(0, 1_000), phase="in_run"),
        EngineStart(
            stamp=_stamp(10, 2_000),
            messages_hash="x",
            n_messages=1,
            n_tools=0,
        ),
        EngineToken(stamp=_stamp(20, 3_000, tok_out=1, cum_out=1), text="hi"),
        EngineError(
            stamp=_stamp(30, 4_000),
            message="MLX out of memory",
            traceback="Traceback (most recent call last):\n  ...\nRuntimeError: oom",
            during=None,
        ),
        EngineDone(stamp=_stamp(31, 4_100), reason="aborted"),
    ]
    for m in moves:
        append(tmp_path, m)

    recovered = list(replay(tmp_path))
    assert recovered == moves

    state = fold(moves, session_dir=tmp_path)
    err_segs = [s for s in state.segments if s.get("kind") == "engine_error"]
    assert len(err_segs) == 1
    assert err_segs[0]["message"] == "MLX out of memory"
    assert err_segs[0]["traceback"].startswith("Traceback")
    assert err_segs[0]["during"] is None


def test_engine_error_during_tool_dispatch(tmp_path: Path) -> None:
    """EngineError can carry the call_id of an in-flight tool dispatch."""
    err = EngineError(
        stamp=_stamp(0, 0),
        message="tool args validation failed",
        traceback=None,
        during="call-7",
    )
    append(tmp_path, err)
    [recovered] = list(replay(tmp_path))
    assert recovered == err
    assert recovered.during == "call-7"
    assert recovered.traceback is None


def test_replay_empty_dir_yields_nothing(tmp_path: Path) -> None:
    assert list(replay(tmp_path)) == []


def test_unknown_kind_rejected(tmp_path: Path) -> None:
    """An events.jsonl line with an unknown `kind` should raise on replay."""
    (tmp_path / "events.jsonl").write_text('{"kind":"NotAThing","stamp":{"t_wall_ms":0,"t_mono_ns":0}}\n')
    with pytest.raises(ValueError, match="unknown Move kind"):
        list(replay(tmp_path))


# --------------------------------------------------------------------------- #
# Config Moves                                                                #
# --------------------------------------------------------------------------- #


def test_system_set_prepends_when_absent(tmp_path: Path) -> None:
    moves = [
        SystemSet(stamp=_stamp(0, 1), text="You are Burl, a Texas 42 player."),
        UserText(stamp=_stamp(1, 2), text="hi"),
    ]
    for m in moves:
        append(tmp_path, m)

    assert list(replay(tmp_path)) == moves
    state = fold(moves, session_dir=tmp_path)
    assert state.messages[0] == {
        "role": "system",
        "content": "You are Burl, a Texas 42 player.",
    }
    assert state.messages[1] == {"role": "user", "content": "hi"}


def test_system_set_replaces_when_present(tmp_path: Path) -> None:
    """Re-issuing SystemSet replaces the existing system message in place."""
    moves = [
        SystemSet(stamp=_stamp(0, 1), text="initial system"),
        UserText(stamp=_stamp(1, 2), text="hi"),
        SystemSet(stamp=_stamp(2, 3), text="updated system"),
    ]
    state = fold(moves, session_dir=tmp_path)
    assert state.messages[0] == {"role": "system", "content": "updated system"}
    # user message stays at index 1; no duplicate system message
    sys_msgs = [m for m in state.messages if m.get("role") == "system"]
    assert len(sys_msgs) == 1


def test_advertised_set_replaces_outright(tmp_path: Path) -> None:
    """AdvertisedSet replaces state.advertised — does not merge."""
    moves = [
        ToolAdded(stamp=_stamp(0, 1), name="explore_game"),
        ToolAdded(stamp=_stamp(1, 2), name="belief_trajectory"),
        ToolAdded(stamp=_stamp(2, 3), name="commit_play"),
        AdvertisedSet(stamp=_stamp(3, 4), names=["belief_trajectory", "commit_play"]),
    ]
    for m in moves:
        append(tmp_path, m)

    assert list(replay(tmp_path)) == moves
    state = fold(moves, session_dir=tmp_path)
    assert set(state.active_tools) == {"explore_game", "belief_trajectory", "commit_play"}
    assert list(state.advertised) == ["belief_trajectory", "commit_play"]

    # Subsequent AdvertisedSet replaces, doesn't append
    later = [
        *moves,
        AdvertisedSet(stamp=_stamp(4, 5), names=["explore_game"]),
    ]
    state2 = fold(later, session_dir=tmp_path)
    assert list(state2.advertised) == ["explore_game"]


def test_tool_added_idempotent(tmp_path: Path) -> None:
    moves = [
        ToolAdded(stamp=_stamp(0, 1), name="explore_game"),
        ToolAdded(stamp=_stamp(1, 2), name="explore_game"),
    ]
    state = fold(moves, session_dir=tmp_path)
    assert list(state.active_tools) == ["explore_game"]


def test_tool_removed_drops_from_active_and_advertised(tmp_path: Path) -> None:
    moves = [
        ToolAdded(stamp=_stamp(0, 1), name="explore_game"),
        ToolAdded(stamp=_stamp(1, 2), name="commit_play"),
        AdvertisedSet(stamp=_stamp(2, 3), names=["explore_game", "commit_play"]),
        ToolRemoved(stamp=_stamp(3, 4), name="explore_game"),
    ]
    for m in moves:
        append(tmp_path, m)

    assert list(replay(tmp_path)) == moves
    state = fold(moves, session_dir=tmp_path)
    assert "explore_game" not in state.active_tools
    assert "explore_game" not in state.advertised
    assert "commit_play" in state.active_tools
    assert "commit_play" in state.advertised


def test_tool_removed_unknown_is_noop(tmp_path: Path) -> None:
    """Removing a tool that isn't active is a no-op (idempotent on absence)."""
    moves = [ToolRemoved(stamp=_stamp(0, 1), name="never_added")]
    state = fold(moves, session_dir=tmp_path)
    assert list(state.active_tools) == []
    assert list(state.advertised) == []


def test_state_journal_only_no_state_json_needed(tmp_path: Path) -> None:
    """End-to-end pre_game-style sequence: journal alone is enough to recover.

    This is the regression test for the design wart that motivated config Moves:
    runtime had been writing state.json because system/advertised/active edits
    weren't journaled. With the new Moves, fold(replay(...)) is total.
    """
    moves = [
        PhaseEnter(stamp=_stamp(0, 1), phase="pre_game"),
        SystemSet(stamp=_stamp(1, 2), text="You are Burl."),
        ToolAdded(stamp=_stamp(2, 3), name="belief_trajectory"),
        ToolAdded(stamp=_stamp(3, 4), name="explore_game"),
        ToolAdded(stamp=_stamp(4, 5), name="commit_play"),
        AdvertisedSet(stamp=_stamp(5, 6), names=["belief_trajectory", "explore_game"]),
        PhaseExit(stamp=_stamp(6, 7), phase="pre_game"),
        PhaseEnter(stamp=_stamp(7, 8), phase="in_run"),
    ]
    for m in moves:
        append(tmp_path, m)

    # Round-trip
    assert list(replay(tmp_path)) == moves

    # Fold the *replayed* events (not the in-memory ones) — same as a fresh
    # process resuming from disk would do.
    state = fold(replay(tmp_path), session_dir=tmp_path)
    assert state.phase == "in_run"
    assert state.messages[0] == {"role": "system", "content": "You are Burl."}
    assert set(state.active_tools) == {"belief_trajectory", "explore_game", "commit_play"}
    assert list(state.advertised) == ["belief_trajectory", "explore_game"]
