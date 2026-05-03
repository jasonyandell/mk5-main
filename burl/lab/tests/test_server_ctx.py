"""Tests for reconstructing tool context from the session journal."""

from __future__ import annotations

import json
from pathlib import Path

from burl.lab.core.transcript import Stamp, UserChoice, append
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
