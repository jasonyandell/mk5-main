"""Tests for reconstructing tool context from the session journal."""

from __future__ import annotations

import json
from pathlib import Path

from burl.lab.core.transcript import EngineCommit, Stamp, State, UserChoice, append
from burl.lab.server import ctx as ctx_module


def _stamp(seq: int = 1) -> Stamp:
    return Stamp(t_wall_ms=seq, t_mono_ns=seq * 1000)


def _write_decision(root: Path, harvest: str, *, global_idx: int, seed: int) -> None:
    harvest_dir = root / harvest
    decision_dir = harvest_dir / "D_required_first" / f"decision_{global_idx}"
    decision_dir.mkdir(parents=True)
    with (harvest_dir / "corpus_index.jsonl").open("w") as f:
        f.write(
            json.dumps(
                {
                    "global_idx": global_idx,
                    "seed": seed,
                    "pi_play": 25,
                    "qmean_play": 25,
                    "burl_play": 18,
                    "oracle_play": 25,
                    "bucket": "BURL_BREAKS_CONSENSUS",
                    "forced_commit": False,
                    "transcript_path": (
                        f"D_required_first/decision_{global_idx}/transcript.live"
                    ),
                }
            )
            + "\n"
        )
    with (decision_dir / "events.jsonl").open("w") as f:
        f.write(
            json.dumps(
                {
                    "kind": "meta",
                    "seed": seed,
                    "declaration": 1,
                    "narrator_seat": 2,
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "kind": "prompt_user",
                    "content": "visible history: seat0:14(foo)",
                }
            )
            + "\n"
        )


def test_build_ctx_for_session_uses_send_seeded_decision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_decision(tmp_path, "h", global_idx=3, seed=42)
    monkeypatch.setenv("BURL_HARNESS_HARVEST_ROOT", str(tmp_path))

    seen: dict = {}

    def fake_build(meta: dict, prompt_user: str) -> dict:
        seen["meta"] = dict(meta)
        seen["prompt_user"] = prompt_user
        return {"ctx": "ok"}

    monkeypatch.setattr(ctx_module, "_build_wax_ctx", fake_build)

    session_dir = tmp_path / "session"
    append(
        session_dir,
        UserChoice(
            stamp=_stamp(),
            option_name="send_seeded_decision",
            args={"harvest": "h", "seed": 42},
        ),
    )

    assert ctx_module.build_ctx_for_session(session_dir) == {"ctx": "ok"}
    assert seen["meta"]["seed"] == 42
    assert seen["prompt_user"] == "visible history: seat0:14(foo)"


def test_build_board_snapshot_prompt_uses_seeded_ctx(monkeypatch) -> None:
    from burl.chat.server.tools_library import board_snapshot

    monkeypatch.setattr(
        ctx_module,
        "build_ctx_for_decision",
        lambda harvest, value, key="global_idx": {
            "harvest": harvest,
            "value": value,
            "key": key,
        },
    )
    monkeypatch.setattr(
        board_snapshot,
        "tool",
        lambda ctx: {"prose": f"snapshot {ctx['harvest']} {ctx['key']}={ctx['value']}"},
    )

    assert ctx_module.build_board_snapshot_prompt("h", 42) == "snapshot h seed=42"


def test_build_session_outcome_records_tool_selection_and_legality(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from burl.chat.server.tools_library import legal_plays

    _write_decision(tmp_path, "h", global_idx=3, seed=42)
    monkeypatch.setenv("BURL_HARNESS_HARVEST_ROOT", str(tmp_path))
    monkeypatch.setattr(
        legal_plays,
        "tool",
        lambda ctx: {
            "structured": {
                "legal_plays": [19, 25],
                "illegal_plays": [18],
            },
        },
    )
    session_dir = tmp_path / "session"
    append(
        session_dir,
        UserChoice(
            stamp=_stamp(),
            option_name="send_seeded_decision",
            args={"harvest": "h", "seed": 42},
        ),
    )
    state = State(
        session_dir=session_dir,
        phase="in_run",
        messages=(),
        active_tools=("board_snapshot", "play_brief", "commit_play"),
        advertised=("board_snapshot", "play_brief", "commit_play"),
        segments=(),
        cum_tok_in=0,
        cum_tok_out=0,
        started_mono_ns=0,
        started_wall_ns=0,
    )
    commit = EngineCommit(stamp=_stamp(2), final={"domino_id": 25})

    outcome = ctx_module.build_session_outcome(
        session_dir,
        state,
        commit,
        ctx={"fake": "ctx"},
    )

    assert outcome is not None
    assert outcome["selected_tools"] == ["board_snapshot", "play_brief", "commit_play"]
    assert outcome["final_domino_id"] == 25
    assert outcome["legal"] is True
    assert outcome["legal_plays"] == [19, 25]
    assert outcome["references"] == {
        "pi_play": 25,
        "qmean_play": 25,
        "burl_play": 18,
        "oracle_play": 25,
    }
    assert outcome["consensus_play"] == 25
    assert outcome["matches"] == {
        "pi": True,
        "qmean": True,
        "burl": False,
        "oracle": True,
        "consensus": True,
    }
