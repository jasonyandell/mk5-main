"""``pre_game`` phase: configure system prompt + advertised tool set.

In this phase the user is *building the prompt the model will see*. The
render shows the current system prompt + a table of every ToolSpec in the
registry (name, role, protocol_phrase) with checkboxes for which are
advertised. The user can also load a decision from a harvest dir, which
seeds the first user message and transitions to ``in_run``.

Phase handlers journal their effects as Moves directly to ``events.jsonl``
via ``transcript.append``, then re-fold to produce the next State.  The
journal is the source of truth — there is no in-memory side state.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from burl.lab.core.render import render_system
from burl.lab.core.tool import Registry
from burl.lab.core.transcript import (
    AdvertisedSet,
    Frame,
    Move,
    Option,
    State,
    SystemSet,
    ToolAdded,
    ToolRemoved,
    UserText,
    UserChoice,
    append,
    fold,
    now_stamp,
    replay,
)

DEFAULT_HARVEST_ROOT = "scratch/belief_trajectory_rollout"


def _harvest_root() -> Path:
    return Path(os.environ.get("BURL_HARNESS_HARVEST_ROOT", DEFAULT_HARVEST_ROOT))


class _PreGamePhase:
    name = "pre_game"

    def render(self, state: State, registry: Registry | None = None) -> Frame:
        registry = registry or _default_registry()
        all_specs = registry.active()
        advertised_set = set(state.advertised)
        rows = [
            {
                "kind": "tool_row",
                "name": s.name,
                "role": s.protocol_role,
                "phrase": s.protocol_phrase,
                "advertised": s.name in advertised_set,
            }
            for s in all_specs
        ]
        base_text = _system_text(state)
        rendered = render_system(
            base_text, [s for s in all_specs if s.name in advertised_set]
        )
        segments: list[dict] = list(state.segments) + [
            {"kind": "rendered_system", "text": rendered},
            *rows,
        ]
        timing = _timing(state)
        return Frame(
            phase=self.name,
            segments=segments,
            active_tools=list(state.active_tools),
            advertised=list(state.advertised),
            timing=timing,
        )

    def options(self, state: State, registry: Registry | None = None) -> list[Option]:
        return [
            Option(
                name="set_system",
                label="Set system prompt",
                args_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
            Option(
                name="add_tool",
                label="Advertise tool",
                args_schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            ),
            Option(
                name="remove_tool",
                label="Unadvertise tool",
                args_schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            ),
            Option(
                name="set_advertised",
                label="Set advertised set",
                args_schema={
                    "type": "object",
                    "properties": {
                        "names": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["names"],
                },
            ),
            Option(
                name="load_decision",
                label="Load harvested decision",
                args_schema={
                    "type": "object",
                    "properties": {
                        "harvest": {"type": "string"},
                        "idx": {"type": "integer"},
                    },
                    "required": ["harvest", "idx"],
                },
            ),
        ]

    async def handle(
        self,
        state: State,
        move: Move,
        registry: Registry | None = None,
    ) -> tuple[State, str | None]:
        """Journal config Moves to events.jsonl, then re-fold.

        The handler does NOT mutate `state` directly — it appends config
        Moves (SystemSet, AdvertisedSet, ToolAdded, ToolRemoved, UserText)
        to the journal and re-folds.  This keeps `fold(replay(...))`
        total: the journal IS the snapshot.
        """
        if not isinstance(move, UserChoice):
            return state, None

        opt = move.option_name
        args = dict(move.args)
        registry = registry or _default_registry()
        new_moves: list[Move] = []

        if opt == "set_system":
            text = str(args.get("text", ""))
            new_moves.append(SystemSet(stamp=now_stamp(state), text=text))

        elif opt == "add_tool":
            name = str(args.get("name", ""))
            if not name:
                return state, None
            if name not in state.active_tools:
                new_moves.append(ToolAdded(stamp=now_stamp(state), name=name))
            new_advertised = list(state.advertised)
            if name not in new_advertised:
                new_advertised.append(name)
            new_moves.append(
                AdvertisedSet(stamp=now_stamp(state), names=new_advertised)
            )

        elif opt == "remove_tool":
            name = str(args.get("name", ""))
            new_moves.append(ToolRemoved(stamp=now_stamp(state), name=name))

        elif opt == "set_advertised":
            names = [str(n) for n in args.get("names", []) or []]
            # Ensure each name is in active_tools first.
            for n in names:
                if n not in state.active_tools:
                    new_moves.append(ToolAdded(stamp=now_stamp(state), name=n))
            new_moves.append(AdvertisedSet(stamp=now_stamp(state), names=names))

        elif opt == "load_decision":
            harvest = str(args.get("harvest", ""))
            idx = int(args.get("idx", 0))
            user_content = _load_user_content(harvest, idx)
            new_moves.append(UserText(stamp=now_stamp(state), text=user_content))

        else:
            return state, None

        for mv in new_moves:
            append(state.session_dir, mv)
        new_state = fold(
            list(replay(state.session_dir)),
            session_dir=state.session_dir,
            registry=registry,
        )

        next_phase = "in_run" if opt == "load_decision" else None
        return new_state, next_phase


# ---- helpers ---------------------------------------------------------- #


def _system_text(state: State) -> str:
    for msg in state.messages:
        if msg.get("role") == "system":
            return str(msg.get("content", ""))
    return ""


def _timing(state: State) -> dict:
    return {
        "wall_ms": 0,
        "tok_cum_in": state.cum_tok_in,
        "tok_cum_out": state.cum_tok_out,
        "tok_per_s": 0.0,
    }


def _default_registry() -> Registry:
    """Empty fallback for renderers that don't pass one in."""
    return Registry()


def _load_user_content(harvest: str, idx: int) -> str:
    """Read events.jsonl for one decision and return its user content.

    Layout (from the chat-server reference):
      ``$BURL_HARNESS_HARVEST_ROOT/<harvest>/corpus_index.jsonl`` —
        one decision per line; we look up by ``global_idx``.
      ``<row.transcript_path>`` (relative to ``<harvest>/``) points into
        the decision's events dir; replace ``/transcript`` to find
        ``events.jsonl``.
    """
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
            if row.get("global_idx") == idx:
                matched = row
                break
    if matched is None:
        raise KeyError(f"global_idx {idx} not found in {harvest}")

    transcript_rel = matched.get("transcript_path", "")
    dec_dir = root / harvest / transcript_rel.split("/transcript")[0]
    events_path = dec_dir / "events.jsonl"

    with events_path.open() as f:
        for raw in f:
            try:
                e = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if e.get("kind") == "prompt_user":
                return str(e.get("content", ""))
    return ""


PRE_GAME = _PreGamePhase()


__all__ = ["PRE_GAME"]
