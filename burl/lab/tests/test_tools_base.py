"""Tests for the three base ToolSpecs.

Verifies:
  * each spec has populated identity + protocol fields
  * ``params`` is a valid JSON Schema (Draft 7)
  * ``commit_play.impl`` returns a ToolResult with ``next_phase="post_turn"``
  * ``belief_trajectory.impl`` and ``explore_game.impl`` return prose evidence
    when their wax_museum legacy callable is mocked

Skips the spec-import gracefully if ``burl.lab.core.tool`` is not yet
landed by the types peer.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

# Skip the whole module if core.tool isn't ready yet — lets the types agent
# land independently without breaking CI.
pytest.importorskip("burl.lab.core.tool")

from jsonschema import Draft7Validator  # noqa: E402

from burl.lab.tools.base import (  # noqa: E402
    BELIEF_TRAJECTORY,
    COMMIT_PLAY,
    EXPLORE_GAME,
)


ALL_SPECS = [BELIEF_TRAJECTORY, EXPLORE_GAME, COMMIT_PLAY]


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
def test_spec_identity_fields(spec):
    assert spec.name
    assert spec.description
    assert spec.example
    assert spec.protocol_role in {"first_read", "candidate_eval", "commit"}
    assert spec.protocol_phrase
    assert callable(spec.impl)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
def test_spec_params_is_valid_json_schema(spec):
    Draft7Validator.check_schema(spec.params)


def test_protocol_roles_match_canonical_assignment():
    assert BELIEF_TRAJECTORY.protocol_role == "first_read"
    assert EXPLORE_GAME.protocol_role == "candidate_eval"
    assert COMMIT_PLAY.protocol_role == "commit"


def test_explore_game_params_require_play_integer():
    schema = EXPLORE_GAME.params
    assert "play" in schema["properties"]
    assert schema["properties"]["play"]["type"] == "integer"
    assert "play" in schema["required"]
    # Validator rejects missing/wrong-typed play.
    v = Draft7Validator(schema)
    assert not v.is_valid({})
    assert not v.is_valid({"play": "14"})
    assert v.is_valid({"play": 14})


def test_commit_play_params_require_domino_id_integer():
    schema = COMMIT_PLAY.params
    assert schema["properties"]["domino_id"]["type"] == "integer"
    v = Draft7Validator(schema)
    assert not v.is_valid({})
    assert v.is_valid({"domino_id": 7})


def test_belief_trajectory_params_is_empty_object():
    schema = BELIEF_TRAJECTORY.params
    assert schema["type"] == "object"
    assert schema["properties"] == {}
    assert schema.get("additionalProperties") is False


# --------------------------------------------------------------------------- #
# Impl behavior — use mocks so we don't need real game state / Gus weights.   #
# --------------------------------------------------------------------------- #


def test_commit_play_impl_records_and_signals_post_turn():
    ctx = SimpleNamespace(final_play=None)
    result = COMMIT_PLAY.impl(ctx, {"domino_id": 14})
    assert result.next_phase == "post_turn"
    assert ctx.final_play == 14
    assert "14" in result.evidence["prose"]
    assert result.evidence["structured"]["committed"] == 14


def test_commit_play_impl_works_without_final_play_attribute():
    """ctx without final_play should not crash — phases manage state."""
    ctx = SimpleNamespace()
    result = COMMIT_PLAY.impl(ctx, {"domino_id": 3})
    assert result.next_phase == "post_turn"
    assert result.evidence["structured"]["committed"] == 3


def test_belief_trajectory_impl_wraps_legacy_payload():
    fake_payload = {
        "prose": "BELIEF: opponents likely void in 5s.",
        "structured": {"per_seat": {"L": [], "P": [], "R": []}},
    }
    with patch(
        "burl.wax_museum.tools.tool_belief_trajectory",
        return_value=fake_payload,
    ):
        result = BELIEF_TRAJECTORY.impl(SimpleNamespace(), {})
    assert result.evidence["prose"] == fake_payload["prose"]
    assert result.evidence["structured"] == fake_payload["structured"]
    assert result.next_phase is None


def test_explore_game_impl_wraps_legacy_payload():
    fake_payload = {
        "prose": "PLAY: 14(5-2)  unconditional Q = +3.5",
        "structured": {"play": 14, "summary": {"mean": 3.5}},
    }
    with patch(
        "burl.wax_museum.tools.tool_explore_game",
        return_value=fake_payload,
    ) as m:
        result = EXPLORE_GAME.impl(SimpleNamespace(), {"play": 14})
    m.assert_called_once()
    # Verify play was forwarded to the legacy callable.
    assert m.call_args.kwargs.get("play") == 14 or 14 in m.call_args.args
    assert "14" in result.evidence["prose"]
    assert result.evidence["structured"]["play"] == 14
    assert result.next_phase is None


# --------------------------------------------------------------------------- #
# Slow / heavy path — only runs when explicitly opted in. Real Gus + oracle.  #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_belief_trajectory_real_gus_smoke():
    """End-to-end smoke against the real Gus belief head — needs weights."""
    pytest.importorskip("burl.tools.belief_trajectory")
    pytest.skip(
        "Real-Gus path requires a constructed game_state + WaxContext; "
        "covered by harness integration tests."
    )
