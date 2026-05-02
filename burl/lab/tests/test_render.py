"""Snapshot test for render.render_system.

Stable string == stable adoption. Three mock ToolSpecs across roles get
their protocol_phrase grouped + composed deterministically. If this snap
moves, the model's view of the world moved — own that change.
"""

from __future__ import annotations

from pathlib import Path

from burl.lab.core.render import render_messages, render_system
from burl.lab.core.tool import Registry, ToolResult, ToolSpec
from burl.lab.core.transcript import State


def _spec(name: str, role: str, phrase: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="",
        params={"type": "object", "properties": {}},
        example=f"{name}()",
        protocol_role=role,  # type: ignore[arg-type]
        protocol_phrase=phrase,
        impl=lambda ctx, args: ToolResult(evidence={"prose": "", "structured": {}}),
    )


def test_render_system_empty_advertised_returns_base():
    out = render_system("You are Burl.", [])
    assert out == "You are Burl."


def test_render_system_three_roles_stable_snapshot():
    specs = [
        _spec("read_state", "first_read", "Read the current game state."),
        _spec("simulate", "candidate_eval", "Try a candidate play and see what happens."),
        _spec("commit_play", "commit", "Commit to a play; this ends your turn."),
    ]
    out = render_system("You are Burl.", specs)
    expected = (
        "You are Burl.\n"
        "\n"
        "## Decision Protocol\n"
        "\n"
        "You have access to the following tools. Each tool plays a specific "
        "role in the decision loop; use the protocol phrase to decide when to "
        "call it.\n"
        "\n"
        "### first_read\n"
        "\n"
        "- **read_state** — Read the current game state.\n"
        "\n"
        "### candidate_eval\n"
        "\n"
        "- **simulate** — Try a candidate play and see what happens.\n"
        "\n"
        "### commit\n"
        "\n"
        "- **commit_play** — Commit to a play; this ends your turn.\n"
    )
    assert out == expected


def test_render_system_unknown_role_appended_after_known_roles():
    specs = [
        _spec("custom", "diagnostic", "Inspect."),
        _spec("read_state", "first_read", "Read."),
    ]
    out = render_system("base", specs)
    # first_read is in _ROLE_ORDER ahead of diagnostic, so it renders first.
    first_idx = out.index("### first_read")
    diag_idx = out.index("### diagnostic")
    assert first_idx < diag_idx


def test_render_system_missing_phrase_renders_name_only():
    specs = [_spec("bare", "first_read", "")]
    out = render_system("b", specs)
    assert "- **bare**" in out
    assert "- **bare** —" not in out


def test_render_messages_re_renders_system_each_call():
    """Adding a new advertised tool changes the system message on the very
    next render_messages call — no rebuild step required."""
    spec_a = _spec("a", "first_read", "Phrase for A.")
    spec_b = _spec("b", "candidate_eval", "Phrase for B.")
    reg = Registry()
    reg.add(spec_a)
    reg.add(spec_b)

    state1 = State(
        session_dir=Path("/tmp/x"),
        phase="pre_game",
        messages=({"role": "system", "content": "S"},),
        active_tools=("a",),
        advertised=("a",),
        segments=(),
        cum_tok_in=0,
        cum_tok_out=0,
        started_mono_ns=0,
        started_wall_ns=0,
    )
    msgs1 = render_messages(state1, reg)
    sys1 = msgs1[0]["content"]
    assert "Phrase for A." in sys1
    assert "Phrase for B." not in sys1

    state2 = State(
        session_dir=state1.session_dir,
        phase=state1.phase,
        messages=state1.messages,
        active_tools=("a", "b"),
        advertised=("a", "b"),
        segments=(),
        cum_tok_in=0,
        cum_tok_out=0,
        started_mono_ns=0,
        started_wall_ns=0,
    )
    msgs2 = render_messages(state2, reg)
    sys2 = msgs2[0]["content"]
    assert "Phrase for A." in sys2
    assert "Phrase for B." in sys2
