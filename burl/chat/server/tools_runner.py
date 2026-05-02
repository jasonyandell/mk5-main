"""Live tool execution against the actual game state of a harvest decision.

Reuses ``burl.wax_museum.tools.build_registry`` so the prose format is
bit-identical to what Burl saw during training. Loads the oracle lazily +
caches it; per-call cost is dominated by the underlying tool (Gus forward
pass for belief_trajectory; oracle samples for explore/probe).

Tools dispatched here MATCH the harvest's tool surface exactly:
    explore_game, probe_best_case, probe_worst_case, ask_rule,
    belief_trajectory

Game-state reconstruction
-------------------------
Each harvest decision has an ``events.jsonl`` whose first line is a ``meta``
event carrying ``(seed, declaration, narrator_seat)`` and whose ``prompt_user``
event includes a ``visible history:`` line with every play observed so far in
``seat{N}:{domino_id}(p-p)`` form. We parse that into a play_history and
replay the state with ``burl.eval.decision_dataset._replay_state`` — the same
primitive ``gus_eval_bridge`` uses internally. This bypasses the held-out
``corpus_eval_20.pt`` index entirely (training-distribution harvests use
seed≈4100, eval corpus uses seed≥900000 — global_idx is not portable).
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


_VISIBLE_HISTORY_RE = re.compile(r"seat(\d+):(\d+)\(")
_VISIBLE_LINE_RE = re.compile(r"visible history:\s*(.+)")


def _ensure_bridge_on_path() -> None:
    """gus_eval_bridge is only used for ``belief_trajectory``'s Gus forward
    pass plumbing, which lives in scratch/. Add it lazily."""
    import sys

    p = str(
        Path("/Users/jason/code/mk5-main/scratch/belief_trajectory_rollout/diagnostic")
    )
    if p not in sys.path:
        sys.path.insert(0, p)


@lru_cache(maxsize=1)
def _oracle() -> Any:
    """E[Q] oracle for explore_game / probe_*. Heavy load (~seconds)."""
    from burl.tools.eq_distribution import load_eq_oracle

    log.info("[tools-runner] loading E[Q] oracle (one-time)")
    return load_eq_oracle()


def _parse_play_history(prompt_user: str) -> list[tuple[int, int]]:
    """Extract ``(seat, domino_id)`` tuples in play order from the user prompt.

    The visible-history line includes both completed tricks AND the current
    trick's plays so far — exactly what ``_replay_state`` expects.
    """
    m = _VISIBLE_LINE_RE.search(prompt_user)
    if not m:
        return []
    body = m.group(1).strip()
    if body.startswith("("):  # "(no plays yet)"
        return []
    return [(int(s), int(d)) for s, d in _VISIBLE_HISTORY_RE.findall(body)]


def _decision_dir(harvest: str, global_idx: int) -> Path:
    """Resolve the on-disk decision dir for ``(harvest, global_idx)``."""
    from . import decisions as dec_mod

    idx_path = dec_mod.SCRATCH_ROOT / harvest / "corpus_index.jsonl"
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    with idx_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("global_idx") == global_idx:
                rel = d["transcript_path"].split("/transcript")[0]
                return dec_mod.SCRATCH_ROOT / harvest / rel
    raise KeyError(f"global_idx {global_idx} not found in {harvest}")


# Per-decision WaxContext cache: avoid rebuilding the state for every tool
# call within the same chat session, AND preserve PlayCache across calls so
# `probe_best_case(play=14)` after `explore_game(play=14)` Just Works.
_CTX_CACHE: dict[tuple[str, int], Any] = {}


def _build_ctx(harvest: str, global_idx: int):
    key = (harvest, int(global_idx))
    if key in _CTX_CACHE:
        return _CTX_CACHE[key]

    from burl.eval.decision_dataset import _replay_state
    from burl.wax_museum.tools import WaxContext

    dec_dir = _decision_dir(harvest, int(global_idx))
    events_path = dec_dir / "events.jsonl"
    meta: dict | None = None
    user_prompt: str | None = None
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
                user_prompt = e["content"]
            if meta is not None and user_prompt is not None:
                break
    if meta is None or user_prompt is None:
        raise RuntimeError(
            f"missing meta or prompt_user in {events_path}"
        )

    play_history = _parse_play_history(user_prompt)
    state = _replay_state(
        int(meta["seed"]),
        int(meta["declaration"]),
        play_history,
    )

    log.info(
        "[tools-runner] built ctx harvest=%s gi=%s seed=%s decl=%s seat=%s "
        "history_len=%d",
        harvest, global_idx, meta["seed"], meta["declaration"],
        meta["narrator_seat"], len(play_history),
    )

    ctx = WaxContext(
        game_state=state,
        me_abs=int(meta["narrator_seat"]),
        oracle=_oracle(),
    )
    _CTX_CACHE[key] = ctx
    return ctx


def reset_ctx(harvest: str | None = None, global_idx: int | None = None) -> None:
    """Drop cached WaxContext (and its PlayCache). With no args, clear all."""
    if harvest is None and global_idx is None:
        _CTX_CACHE.clear()
        return
    drop = [
        k for k in list(_CTX_CACHE)
        if (harvest is None or k[0] == harvest)
        and (global_idx is None or k[1] == int(global_idx))
    ]
    for k in drop:
        _CTX_CACHE.pop(k, None)


_BASE_TOOLS = (
    "full_board_snapshot",
    "game_state_snapshot",
    "belief_trajectory",
    "explore_game",
    "probe_best_case",
    "probe_worst_case",
    "ask_rule",
)


def available_tools() -> list[str]:
    from . import improvised_tools

    return [*_BASE_TOOLS, *(t.name for t in improvised_tools.list_all())]


def run_tool(harvest: str, global_idx: int, tool: str, args: dict) -> dict:
    """Run ``tool(args)`` against the live game state of harvest decision
    ``(harvest, global_idx)``. Returns ``{"prose": str, "structured": ...}``.

    The improvised registry is checked first so a hot-registered tool of the
    same name shadows the static one (deliberate — that's the whole point
    of the experiment loop).
    """
    from . import improvised_tools
    from burl.wax_museum.tools import build_registry

    ctx = _build_ctx(harvest, int(global_idx))
    args = args or {}
    if "play" in args and not isinstance(args["play"], int):
        args["play"] = int(args["play"])

    improv = improvised_tools.get(tool)
    if improv is not None:
        payload = improv.impl(ctx, **args)
        return {
            "prose": payload.get("prose", ""),
            "structured": payload.get("structured"),
        }

    registry = build_registry(ctx)
    if tool not in registry:
        raise ValueError(
            f"unknown tool {tool!r}; known: {list(registry.keys())}"
        )
    payload = registry[tool](**args)
    return {
        "prose": payload.get("prose", ""),
        "structured": payload.get("structured"),
    }
