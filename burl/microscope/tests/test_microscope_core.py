from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from burl.lab.core.tool import Registry, ToolResult, ToolSpec
from burl.lab.core.transcript import EngineDone, EngineStart, EngineToken, EngineToolCall, Stamp
from burl.microscope.core import Case, MicroscopeSession, load_recipe


class FakeEngine:
    async def step(self, messages: list[dict], tools: list[ToolSpec], max_tokens: int = 2048):
        del messages, tools, max_tokens
        stamp = Stamp(t_wall_ms=0, t_mono_ns=0)
        yield EngineStart(stamp=stamp, messages_hash="fake", n_messages=2, n_tools=1)
        yield EngineToken(stamp=stamp, text="I should inspect the case. ")
        yield EngineToolCall(stamp=stamp, name="probe", args={"x": 7}, call_id="call-1")
        yield EngineDone(stamp=stamp, reason="tool_dispatch")


def test_recipe_supports_tool_overrides_and_response_templates(tmp_path: Path) -> None:
    recipe_dir = tmp_path / "recipes" / "r1"
    (recipe_dir / "tool_responses").mkdir(parents=True)
    (recipe_dir / "system.md").write_text("System {{x}}", encoding="utf-8")
    (recipe_dir / "play.md").write_text("Play {{x}}", encoding="utf-8")
    (recipe_dir / "params.json").write_text('{"max_tokens": 123}', encoding="utf-8")
    (recipe_dir / "tools.json").write_text(
        json.dumps([
            {
                "name": "probe",
                "description": "overridden description",
                "protocol_phrase": "Call `probe(x=7)` when testing.",
            }
        ]),
        encoding="utf-8",
    )
    (recipe_dir / "tool_responses" / "probe.md").write_text(
        "PROSE={{prose}}\nJSON={{structured_json_compact}}",
        encoding="utf-8",
    )

    recipe = load_recipe("r1", root=tmp_path / "recipes")
    assert recipe.max_tokens == 123

    reg = Registry()
    reg.add(
        ToolSpec(
            name="probe",
            description="base description",
            params={"type": "object", "properties": {}},
            example="probe()",
            protocol_role="diagnostic",
            protocol_phrase="base phrase",
            impl=lambda _ctx, _args: ToolResult(evidence={"prose": "ok", "structured": {"a": 1}}),
        )
    )
    specs = recipe.tool_specs(reg)
    assert specs[0].description == "overridden description"
    assert specs[0].protocol_phrase == "Call `probe(x=7)` when testing."
    assert recipe.render_tool_response("probe", {"prose": "ok", "structured": {"a": 1}}) == 'PROSE=ok\nJSON={"a":1}'


def test_session_step_dispatches_one_tool_and_records_tool_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BURL_MICROSCOPE_SESSION_ROOT", str(tmp_path / "sessions"))

    recipe_dir = tmp_path / "recipes" / "r1"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "system.md").write_text("You are test Burl.", encoding="utf-8")
    (recipe_dir / "play.md").write_text("Board: {{board_snapshot}}", encoding="utf-8")
    (recipe_dir / "tools.json").write_text('["probe"]', encoding="utf-8")

    recipe = load_recipe("r1", root=tmp_path / "recipes")
    reg = Registry()
    reg.add(
        ToolSpec(
            name="probe",
            description="probe tool",
            params={"type": "object", "properties": {"x": {"type": "integer"}}},
            example="probe(x=7)",
            protocol_role="diagnostic",
            protocol_phrase="Call `probe(x=7)` when testing.",
            impl=lambda _ctx, args: ToolResult(
                evidence={"prose": f"probe saw {args['x']}", "structured": {"x": args["x"]}}
            ),
        )
    )
    case = Case(
        harvest="h",
        lookup_key="global_idx",
        lookup_value=1,
        row={"bucket": "TEST", "oracle_play": 7, "burl_play": 3},
        meta={"legal_plays": [7], "bot_play": 7},
        harvested_user_prompt="original",
        board_snapshot="snapshot text",
        ctx=object(),
    )
    session = MicroscopeSession(case=case, recipe=recipe, registry=reg, sid="s1")

    result = asyncio.run(session.step(FakeEngine()))

    assert result.tool_call == {"name": "probe", "args": {"x": 7}, "call_id": "call-1"}
    assert result.tool_result is not None
    assert result.tool_result["response"] == "probe saw 7"
    assert session.messages[-1] == {
        "role": "assistant",
        "content": "I should inspect the case.",
        "tool_calls": [
            {"type": "function", "function": {"name": "probe", "arguments": {"x": 7}}}
        ],
        "tool_responses": [{"name": "probe", "response": "probe saw 7"}],
    }
    assert (tmp_path / "sessions" / "s1" / "events.jsonl").exists()
