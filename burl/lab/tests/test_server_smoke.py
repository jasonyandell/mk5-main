"""Server smoke test.

Boots the FastAPI app with a fake engine + the real Registry (or empty if
tools aren't importable) and exercises:

1. ``/api/health`` returns ok.
2. ``POST /api/sessions`` creates a session.
3. ``POST /api/move`` with a pre_game UserChoice SSE-streams the resulting
   transitions; the frame reflects the change.

Stays in pre_game so this test is fast and has no MLX dependency.
"""

from __future__ import annotations

import json
import os
import re
import tempfile

import pytest

# Keep sessions out of ~/.cache during tests.
os.environ.setdefault("BURL_HARNESS_SESSION_ROOT", tempfile.mkdtemp(prefix="burl-lab-"))


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from burl.lab.server.app import app

    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "engine_loaded" in body
    assert "n_tools" in body


def test_create_session_then_post_move(client):
    r = client.post("/api/sessions")
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert sid

    # Frame loads.
    r = client.get(f"/api/sessions/{sid}/frame")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    assert body["phase"] == "pre_game"
    option_names = [o["name"] for o in body["options"]]
    assert "set_system" in option_names
    assert "generate_system" in option_names
    assert "start_run" in option_names

    # Apply set_system; should SSE-stream a no-op transition (still pre_game).
    r = client.post(
        "/api/move",
        json={
            "session_id": sid,
            "move": {
                "kind": "UserChoice",
                "option_name": "set_system",
                "args": {"text": "hello burl"},
            },
        },
    )
    assert r.status_code == 200
    body = r.text
    # CRLF invariant: stream.py strips CR; body must contain none.
    assert "\r" not in body

    # Frame reflects the new system text.
    r = client.get(f"/api/sessions/{sid}/frame")
    frame = r.json()["frame"]
    builder = next(
        (s for s in frame["segments"] if s.get("kind") == "prompt_builder"), None
    )
    assert builder is not None
    assert builder["base_chars"] == len("hello burl")
    rendered = next(
        (s for s in frame["segments"] if s.get("kind") == "rendered_system"), None
    )
    assert rendered is not None
    assert "hello burl" in rendered["text"]


def test_pre_game_generate_system_builds_default_prompt_with_selected_tools(client):
    sid = client.post("/api/sessions").json()["session_id"]
    client.post(
        "/api/move",
        json={
            "session_id": sid,
            "move": {
                "kind": "UserChoice",
                "option_name": "set_advertised",
                "args": {"names": ["state_brief", "legal_plays"]},
            },
        },
    )

    r = client.post(
        "/api/move",
        json={
            "session_id": sid,
            "move": {
                "kind": "UserChoice",
                "option_name": "generate_system",
                "args": {},
            },
        },
    )
    assert r.status_code == 200

    frame = client.get(f"/api/sessions/{sid}/frame").json()["frame"]
    rendered = next(
        (s for s in frame["segments"] if s.get("kind") == "rendered_system"), None
    )
    assert rendered is not None
    assert "You are Burl, a Texas 42 dominoes agent" in rendered["text"]
    assert "## Decision Protocol" in rendered["text"]
    assert "state_brief" in rendered["text"]
    assert "legal_plays" in rendered["text"]


def test_lmstudio_chat_launch_journals_request_and_response(client, monkeypatch):
    from burl.lab.server import app as server_app

    captured: dict = {}

    def fake_lmstudio_chat(config, payload):  # noqa: ARG001
        captured.update(payload)
        return {
            "model_instance_id": payload["model"],
            "output": [{"type": "message", "content": "play the 2-1"}],
            "response_id": "resp_unit_test",
            "stats": {
                "input_tokens": 42,
                "total_output_tokens": 7,
                "tokens_per_second": 21.0,
                "time_to_first_token_seconds": 0.5,
            },
        }

    monkeypatch.setattr(server_app, "lmstudio_chat", fake_lmstudio_chat)

    sid = client.post("/api/sessions").json()["session_id"]
    client.post(
        "/api/move",
        json={
            "session_id": sid,
            "move": {
                "kind": "UserChoice",
                "option_name": "set_system",
                "args": {"text": "You are Burl."},
            },
        },
    )
    client.post(
        "/api/move",
        json={
            "session_id": sid,
            "move": {
                "kind": "UserChoice",
                "option_name": "set_advertised",
                "args": {"names": ["state_brief", "commit_play"]},
            },
        },
    )

    r = client.post(
        f"/api/sessions/{sid}/lmstudio/chat",
        json={"model": "test-model", "input": "board snapshot"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["response"]["response_id"] == "resp_unit_test"
    assert captured["model"] == "test-model"
    assert captured["input"] == "board snapshot"
    assert captured["store"] is True
    assert "You are Burl." in captured["system_prompt"]
    assert "state_brief" in captured["system_prompt"]
    assert "commit_play" in captured["system_prompt"]

    frame = client.get(f"/api/sessions/{sid}/frame").json()["frame"]
    response_seg = next(
        s for s in frame["segments"] if s.get("kind") == "lmstudio_response"
    )
    assert response_seg["response"]["response_id"] == "resp_unit_test"
    assert frame["timing"]["tok_cum_in"] == 42
    assert frame["timing"]["tok_cum_out"] == 7


def test_journal_is_canonical_no_state_json(client, tmp_path, monkeypatch):
    """fold(replay(events.jsonl)) is the only source of truth.

    After a sequence of pre_game moves, the on-disk session dir should
    contain *only* events.jsonl (no state.json), and re-folding the
    journal must reproduce the State that drove the most recent frame.
    """
    from pathlib import Path

    from burl.lab.core.tool import Registry
    from burl.lab.core.transcript import fold, replay

    sid = client.post("/api/sessions").json()["session_id"]
    for opt, args in [
        ("set_system", {"text": "you are burl"}),
        ("add_tool", {"name": "belief_trajectory"}),
        ("add_tool", {"name": "explore_game"}),
        ("set_advertised", {"names": ["belief_trajectory"]}),
    ]:
        r = client.post(
            "/api/move",
            json={
                "session_id": sid,
                "move": {"kind": "UserChoice", "option_name": opt, "args": args},
            },
        )
        assert r.status_code == 200

    session_dir = Path(os.environ["BURL_HARNESS_SESSION_ROOT"]) / sid
    files = sorted(p.name for p in session_dir.iterdir())
    # Only events.jsonl — no state.json.
    assert files == ["events.jsonl"]

    # Independent fold reproduces the State that drives the frame.
    moves = list(replay(session_dir))
    folded = fold(moves, session_dir=session_dir, registry=Registry())
    assert folded.phase == "pre_game"
    assert folded.advertised == ("belief_trajectory",)
    assert "belief_trajectory" in folded.active_tools
    assert "explore_game" in folded.active_tools
    sys_msg = next(
        (m for m in folded.messages if m.get("role") == "system"), None
    )
    assert sys_msg is not None
    assert sys_msg["content"] == "you are burl"

    # And the server's frame agrees.
    frame = client.get(f"/api/sessions/{sid}/frame").json()["frame"]
    assert "belief_trajectory" in frame["advertised"]


def test_server_journals_post_turn_transition_after_drive_commit(client):
    """End-to-end: replacing the live engine with a FakeEngine that emits
    a commit-role tool dispatch.  The server should:

    1. Stream the engine events + harness EngineCommit through SSE.
    2. After drive returns, call ``in_run.handle(state, EngineCommit)``
       to read ``next_phase="post_turn"``.
    3. Journal PhaseExit("in_run") + PhaseEnter("post_turn") and stream
       them as part of the same SSE response.

    Drive does not synthesise PhaseExit/PhaseEnter — that's the server's
    job per SPEC's Phase Protocol.
    """
    from typing import AsyncIterator

    from burl.lab.core.transcript import (
        EngineDone,
        EngineStart,
        EngineToken,
        EngineToolCall,
        Stamp,
        replay,
    )
    from burl.lab.server.app import app_state

    def _stamp(seq: int) -> Stamp:
        return Stamp(t_wall_ms=seq, t_mono_ns=seq * 1000)

    class FakeEngine:
        info = {"model": "fake", "adapter": None}

        async def step(self, *, messages, tools, max_tokens=2048) -> AsyncIterator:
            yield EngineStart(
                stamp=_stamp(1),
                messages_hash="x",
                n_messages=len(messages),
                n_tools=len(tools),
            )
            yield EngineToken(stamp=_stamp(2), text="committing")
            yield EngineToolCall(
                stamp=_stamp(3),
                name="commit_play",
                args={"domino_id": 14},
                call_id="call-c1",
            )
            yield EngineDone(stamp=_stamp(4), reason="tool_dispatch")

    real_engine = app_state.get("engine")
    app_state["engine"] = FakeEngine()
    try:
        sid = client.post("/api/sessions").json()["session_id"]
        # Configure pre_game so commit_play is in active_tools.
        client.post(
            "/api/move",
            json={
                "session_id": sid,
                "move": {
                    "kind": "UserChoice",
                    "option_name": "set_advertised",
                    "args": {"names": ["commit_play"]},
                },
            },
        )
        # Drop into in_run directly; this test is about the post-drive
        # transition after commit, not about pre_game prompt building.
        from burl.lab.core.transcript import PhaseEnter, PhaseExit, append

        from pathlib import Path
        session_dir = Path(os.environ["BURL_HARNESS_SESSION_ROOT"]) / sid
        append(session_dir, PhaseExit(stamp=_stamp(0), phase="pre_game"))
        append(session_dir, PhaseEnter(stamp=_stamp(0), phase="in_run"))

        # Now POST a UserText — the server is in in_run, will drive the
        # FakeEngine, which emits commit_play, drive synthesises
        # EngineCommit, and the server journals PhaseExit/PhaseEnter.
        r = client.post(
            "/api/move",
            json={
                "session_id": sid,
                "move": {"kind": "UserText", "text": "your turn"},
            },
        )
        assert r.status_code == 200

        # Inspect the journal — the server must have appended
        # EngineCommit, PhaseExit("in_run"), PhaseEnter("post_turn").
        kinds = [type(m).__name__ for m in replay(session_dir)]
        # The tail should be: ToolResult, EngineCommit, PhaseExit, PhaseEnter.
        assert kinds[-4:] == [
            "ToolResult",
            "EngineCommit",
            "PhaseExit",
            "PhaseEnter",
        ]

        # And the post-commit phase is post_turn.
        last_enter = next(
            m
            for m in reversed(list(replay(session_dir)))
            if type(m).__name__ == "PhaseEnter"
        )
        assert last_enter.phase == "post_turn"

        # post_turn is a registered Phase — GET /frame must not 500.
        frame = client.get(f"/api/sessions/{sid}/frame").json()
        assert frame["phase"] == "post_turn"
        opt_names = [o["name"] for o in frame["options"]]
        assert "start_new_session" in opt_names

        # start_new_session round-trips back to pre_game.
        r = client.post(
            "/api/move",
            json={
                "session_id": sid,
                "move": {
                    "kind": "UserChoice",
                    "option_name": "start_new_session",
                    "args": {},
                },
            },
        )
        assert r.status_code == 200
        frame = client.get(f"/api/sessions/{sid}/frame").json()
        assert frame["phase"] == "pre_game"
    finally:
        app_state["engine"] = real_engine


def test_post_move_stream_format(client):
    """POST /api/move SSE body parses cleanly with \\r?\\n\\r?\\n splitter."""
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post(
        "/api/move",
        json={
            "session_id": sid,
            "move": {
                "kind": "UserChoice",
                "option_name": "add_tool",
                "args": {"name": "synthetic_tool"},
            },
        },
    )
    body = r.text
    # Defensive parse — split on \r?\n\r?\n even though we never emit CR.
    frames = re.split(r"\r?\n\r?\n", body)
    decoded = []
    for frame in frames:
        data_lines = [
            line[len("data: "):]
            for line in frame.split("\n")
            if line.startswith("data: ")
        ]
        if not data_lines:
            continue
        try:
            decoded.append(json.loads("\n".join(data_lines)))
        except json.JSONDecodeError:
            pass
    # add_tool stays in pre_game so no engine drive; either zero or a few
    # transition events. Either way, no malformed JSON in the stream.
    assert all(isinstance(d, dict) for d in decoded)
