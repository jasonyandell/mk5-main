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

from burl.lab.core.transcript import EngineCommit, State, UserChoice, replay

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
) -> tuple[dict, dict, str]:
    """Return (index row, meta, prompt_user) — same recipe as smoke."""
    matched = _find_decision_row(harvest, key=key, value=value)
    meta, prompt_user = _read_decision_events(harvest, matched)
    return matched, meta, prompt_user


def _find_decision_row(harvest: str, *, key: str, value: int) -> dict:
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
    return matched


def _read_decision_events(harvest: str, row: dict) -> tuple[dict, str]:
    root = _harvest_root()
    transcript_rel = row.get("transcript_path", "")
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


def build_ctx_for_decision(
    harvest: str,
    value: int,
    *,
    key: str = "global_idx",
) -> Any:
    """Build a WaxContext directly from a harvest index row."""
    _row, meta, prompt_user = _resolve_decision(harvest, value, key=key)
    return _build_wax_ctx(meta, prompt_user)


def build_board_snapshot_prompt(harvest: str, seed: int) -> str:
    """Render the board snapshot for a seeded decision as the user prompt."""
    from burl.chat.server.tools_library.board_snapshot import tool

    ctx = build_ctx_for_decision(harvest, seed, key="seed")
    payload = tool(ctx)
    return str(payload.get("prose", ""))


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
    _row, meta, prompt_user = _resolve_decision(harvest, value, key=key)
    return _build_wax_ctx(meta, prompt_user)


def _last_decision_choice(session_dir: Path) -> tuple[str, str, int] | None:
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
    return last_load


def build_session_outcome(
    session_dir: Path,
    state: State,
    commit: EngineCommit,
    ctx: Any,
) -> dict | None:
    """Summarize a completed decision for later tool-selection mining."""
    last_load = _last_decision_choice(session_dir)
    if last_load is None or ctx is None:
        return None

    harvest, key, value = last_load
    row, _meta, _prompt_user = _resolve_decision(harvest, value, key=key)
    final = commit.final if isinstance(commit.final, dict) else {}
    final_domino = final.get("domino_id")
    if final_domino is None:
        return None
    final_domino = int(final_domino)

    from burl.chat.server.tools_library.legal_plays import tool as legal_tool

    legal_payload = legal_tool(ctx)
    legal_structured = dict(legal_payload.get("structured", {}))
    legal_plays = [int(d) for d in legal_structured.get("legal_plays", [])]
    illegal_plays = [int(d) for d in legal_structured.get("illegal_plays", [])]

    references = {
        "pi_play": _maybe_int(row.get("pi_play")),
        "qmean_play": _maybe_int(row.get("qmean_play")),
        "burl_play": _maybe_int(row.get("burl_play")),
        "oracle_play": _maybe_int(row.get("oracle_play")),
    }
    consensus_play = (
        references["pi_play"]
        if references["pi_play"] is not None
        and references["pi_play"] == references["qmean_play"]
        else None
    )

    return {
        "harvest": harvest,
        "lookup": {"key": key, "value": value},
        "seed": _maybe_int(row.get("seed")),
        "global_idx": _maybe_int(row.get("global_idx")),
        "bucket": row.get("bucket"),
        "forced_commit": bool(row.get("forced_commit", False)),
        "selected_tools": list(state.advertised),
        "active_tools": list(state.active_tools),
        "final_domino_id": final_domino,
        "legal": final_domino in legal_plays,
        "legal_plays": legal_plays,
        "illegal_plays": illegal_plays,
        "references": references,
        "consensus_play": consensus_play,
        "matches": {
            "pi": final_domino == references["pi_play"],
            "qmean": final_domino == references["qmean_play"],
            "burl": final_domino == references["burl_play"],
            "oracle": final_domino == references["oracle_play"],
            "consensus": (
                final_domino == consensus_play
                if consensus_play is not None
                else None
            ),
        },
    }


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "build_board_snapshot_prompt",
    "build_ctx_for_decision",
    "build_ctx_for_session",
    "build_session_outcome",
]
