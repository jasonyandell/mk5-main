"""Tests for ToolSpecs mined from the burl-chat improvised-tool library."""

from __future__ import annotations

from types import SimpleNamespace

from jsonschema import Draft7Validator

from burl.lab.core.tool import ToolResult
from burl.lab.tools.chat_mined import (
    BOARD_SNAPSHOT,
    CHAT_MINED_TOOLS,
    LEGAL_PLAYS,
    PLAY_BRIEF,
    STATE_BRIEF,
)


def test_chat_mined_tool_specs_have_distinct_protocol_roles() -> None:
    assert [spec.name for spec in CHAT_MINED_TOOLS] == [
        "state_brief",
        "board_snapshot",
        "legal_plays",
        "play_brief",
    ]
    assert STATE_BRIEF.protocol_role == "first_read"
    assert BOARD_SNAPSHOT.protocol_role == "first_read"
    assert LEGAL_PLAYS.protocol_role == "diagnostic"
    assert PLAY_BRIEF.protocol_role == "candidate_eval"
    assert "state_brief()" in STATE_BRIEF.protocol_phrase
    assert "legal_plays()" in LEGAL_PLAYS.protocol_phrase
    assert "play_brief(play=X)" in PLAY_BRIEF.protocol_phrase


def test_chat_mined_schemas_are_valid_json_schema() -> None:
    for spec in CHAT_MINED_TOOLS:
        Draft7Validator.check_schema(spec.params)


def test_no_arg_wrapper_returns_tool_result(monkeypatch) -> None:
    from burl.chat.server.tools_library import board_snapshot

    def fake_tool(ctx, **kwargs):
        return {
            "prose": f"ctx={ctx.name}",
            "structured": {"kwargs": kwargs},
        }

    monkeypatch.setattr(board_snapshot, "tool", fake_tool)

    result = BOARD_SNAPSHOT.impl(SimpleNamespace(name="wax"), {})

    assert isinstance(result, ToolResult)
    assert result.evidence == {"prose": "ctx=wax", "structured": {"kwargs": {}}}
    assert result.next_tools == ()


def test_play_brief_wrapper_passes_play_as_integer(monkeypatch) -> None:
    from burl.chat.server.tools_library import play_brief

    seen = {}

    def fake_tool(ctx, play, **kwargs):
        seen["ctx"] = ctx
        seen["play"] = play
        seen["kwargs"] = kwargs
        return {"prose": "ok", "structured": {"play": play}}

    monkeypatch.setattr(play_brief, "tool", fake_tool)
    ctx = SimpleNamespace()

    result = PLAY_BRIEF.impl(ctx, {"play": "14"})

    assert seen == {"ctx": ctx, "play": 14, "kwargs": {}}
    assert result.evidence == {"prose": "ok", "structured": {"play": 14}}
