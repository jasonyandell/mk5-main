"""Build a WaxContext for a session from its journal.

The tool impls (``belief_trajectory``, ``explore_game``, ``commit_play``)
all take a ``ctx`` first argument; for the base tools that ctx must be
a ``burl.wax_museum.tools.WaxContext`` with a real game state + E[Q]
oracle + ``me_abs`` seat. The smoke driver
(`burl/lab/smoke/real_mlx_smoke.py`) does this construction in-process
before calling ``drive(..., ctx=ctx)``.

The HTTP server lacks an obvious phase to do this in: ``pre_game`` only
knows ``(harvest, idx)`` from the user's ``load_decision`` choice, and
``in_run`` is purely transition-shaped. So we cache the ctx in
``app_state["ctx_cache"]`` keyed by ``session_id``, lazily building it
on the first drive step by replaying the journal to recover the most
recent ``load_decision`` UserChoice and reconstructing a WaxContext
exactly the way the smoke driver does.

This is a temporary seam — see bead t42-xhnl follow-up. The clean
home for ctx construction is a phase responsibility, not server-side
journal sniffing. We keep the shape of the smoke driver verbatim so
the fix later is a move, not a rewrite.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from burl.lab.core.transcript import UserChoice, replay

log = logging.getLogger(__name__)

DEFAULT_HARVEST_ROOT = Path("scratch/belief_trajectory_rollout")

_VISIBLE_HISTORY_RE = re.compile(r"seat(\d+):(\d+)\(")
_VISIBLE_LINE_RE = re.compile(r"visible history:\s*(.+)")


def _harvest_root() -> Path:
    raw = os.environ.get("BURL_HARNESS_HARVEST_ROOT")
    if raw:
        return Path(raw)
    return DEFAULT_HARVEST_ROOT


def _parse_play_history(prompt_user: str) -> list[tuple[int, int]]:
    m = _VISIBLE_LINE_RE.search(prompt_user)
    if not m:
        return []
    body = m.group(1).strip()
    if body.startswith("("):
        return []
    return [(int(s), int(d)) for s, d in _VISIBLE_HISTORY_RE.findall(body)]


def _resolve_decision(
    harvest: str,
    value: int,
    *,
    key: str = "global_idx",
) -> tuple[dict, str]:
    """Return (meta, prompt_user) for one corpus row — same recipe as smoke."""
    root = _harvest_root()
    idx_path = root / harvest / "corpus_index.jsonl"
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)

    matched: dict | None = None
    with idx_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get(key) == value:
                matched = row
                break
    if matched is None:
        raise KeyError(f"{key} {value} not found in {harvest}")

    transcript_rel = matched.get("transcript_path", "")
    dec_dir = root / harvest / transcript_rel.split("/transcript")[0]
    events_path = dec_dir / "events.jsonl"

    meta: dict | None = None
    user_content: str = ""
    with events_path.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = e.get("kind")
            if kind == "meta":
                meta = e
            elif kind == "prompt_user":
                user_content = e.get("content", "")
            if meta and user_content:
                break
    if meta is None or not user_content:
        raise RuntimeError(f"missing meta or prompt_user in {events_path}")
    return meta, user_content


def _build_wax_ctx(meta: dict, prompt_user: str) -> Any:
    """Reconstruct WaxContext — same recipe as smoke."""
    from burl.eval.decision_dataset import _replay_state
    from burl.tools.eq_distribution import load_eq_oracle
    from burl.wax_museum.tools import WaxContext

    play_history = _parse_play_history(prompt_user)
    state = _replay_state(
        int(meta["seed"]),
        int(meta["declaration"]),
        play_history,
    )
    oracle = load_eq_oracle()
    return WaxContext(
        game_state=state,
        me_abs=int(meta["narrator_seat"]),
        oracle=oracle,
    )


def build_ctx_for_session(session_dir: Path) -> Any | None:
    """Replay the journal, find the most recent decision-loading UserChoice,
    and build a WaxContext from it. Returns None if no decision-loading choice
    is in the journal.
    """
    last_load: tuple[str, str, int] | None = None
    for mv in replay(session_dir):
        if isinstance(mv, UserChoice) and mv.option_name == "load_decision":
            harvest = str(mv.args.get("harvest", ""))
            idx = int(mv.args.get("idx", 0))
            if harvest:
                last_load = (harvest, "global_idx", idx)
        elif (
            isinstance(mv, UserChoice)
            and mv.option_name == "send_seeded_decision"
        ):
            harvest = str(mv.args.get("harvest", ""))
            seed = int(mv.args.get("seed", mv.args.get("idx", 0)))
            if harvest:
                last_load = (harvest, "seed", seed)
    if last_load is None:
        log.info("[lab.ctx] no decision context in journal at %s", session_dir)
        return None
    harvest, key, value = last_load
    log.info("[lab.ctx] building ctx for harvest=%s %s=%s", harvest, key, value)
    meta, prompt_user = _resolve_decision(harvest, value, key=key)
    return _build_wax_ctx(meta, prompt_user)


__all__ = ["build_ctx_for_session"]
