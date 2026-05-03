"""``pre_game`` phase: configure system prompt + advertised tool set.

In this phase the user is *building the prompt the model will see*. The
render shows the current system prompt + a table of every ToolSpec in the
registry (name, role, protocol_phrase) with checkboxes for which are
advertised. The user can also load a decision from a harvest dir, which
fills the builder with the harvested system/user prompts. Starting a run is
an explicit follow-up choice.

Phase handlers return effects as Moves in a ``Trace``. The server journals
those Moves; there is no in-memory side state.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from burl.lab.core.arrow import Trace
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
    now_stamp,
)

DEFAULT_HARVEST_ROOT = "scratch/belief_trajectory_rollout"
DEFAULT_BASE_SYSTEM = (
    "You are Burl, a Texas 42 dominoes agent. Pick the next play. "
    "You have tools that describe the game state; call them as needed. "
    "The state tools answer WHAT IS the state; they never tell you WHAT TO DO. "
    "Reason with what they return. When you are ready to commit, call the "
    "`commit_play` tool with the integer domino_id from your hand. That ends "
    "the decision."
)


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
            {
                "kind": "prompt_builder",
                "base_chars": len(base_text),
                "rendered_chars": len(rendered),
                "advertised_count": len(advertised_set),
                "tool_count": len(all_specs),
                "has_user_message": any(
                    msg.get("role") == "user" for msg in state.messages
                ),
            },
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
                name="generate_system",
                label="Generate system prompt",
                args_schema={"type": "object", "properties": {}},
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
                label="Load harvested decision into builder",
                args_schema={
                    "type": "object",
                    "properties": {
                        "harvest": {"type": "string"},
                        "idx": {"type": "integer"},
                    },
                    "required": ["harvest", "idx"],
                },
            ),
            Option(
                name="send_seeded_decision",
                label="Send seeded decision to Gemma",
                args_schema={
                    "type": "object",
                    "properties": {
                        "harvest": {"type": "string"},
                        "seed": {"type": "integer"},
                    },
                    "required": ["harvest", "seed"],
                },
            ),
            Option(
                name="ask_gemma",
                label="Ask Gemma now",
                args_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
            Option(
                name="start_run",
                label="Start run",
                args_schema={"type": "object", "properties": {}},
            ),
        ]

    async def handle(
        self,
        state: State,
        move: Move,
        registry: Registry | None = None,
    ) -> Trace[str]:
        """Return config Moves to journal; do not mutate or write State."""
        if not isinstance(move, UserChoice):
            return Trace()

        opt = move.option_name
        args = dict(move.args)
        new_moves: list[Move] = []

        if opt == "set_system":
            text = str(args.get("text", ""))
            new_moves.append(SystemSet(stamp=now_stamp(state), text=text))

        elif opt == "generate_system":
            text = _system_text(state) or DEFAULT_BASE_SYSTEM
            new_moves.append(SystemSet(stamp=now_stamp(state), text=text))

        elif opt == "add_tool":
            name = str(args.get("name", ""))
            if not name:
                return Trace()
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
            system_content, user_content = _load_decision_prompts(harvest, idx)
            if system_content:
                new_moves.append(
                    SystemSet(stamp=now_stamp(state), text=system_content)
                )
            new_moves.append(UserText(stamp=now_stamp(state), text=user_content))

        elif opt == "send_seeded_decision":
            harvest = str(args.get("harvest", ""))
            seed = int(args.get("seed", args.get("idx", 0)))
            _system_content, user_content = _load_seeded_decision_prompts(
                harvest, seed
            )
            if not _system_text(state):
                new_moves.append(
                    SystemSet(stamp=now_stamp(state), text=DEFAULT_BASE_SYSTEM)
                )
            new_moves.append(UserText(stamp=now_stamp(state), text=user_content))

        elif opt == "ask_gemma":
            text = str(args.get("text", ""))
            if text:
                if not _system_text(state):
                    new_moves.append(
                        SystemSet(stamp=now_stamp(state), text=DEFAULT_BASE_SYSTEM)
                    )
                new_moves.append(UserText(stamp=now_stamp(state), text=text))

        elif opt == "start_run":
            if not _has_user_text(state):
                return Trace()
            if not _system_text(state):
                new_moves.append(
                    SystemSet(stamp=now_stamp(state), text=DEFAULT_BASE_SYSTEM)
                )

        else:
            return Trace()

        next_phase = (
            "in_run"
            if opt in {"ask_gemma", "send_seeded_decision", "start_run"}
            else None
        )
        return Trace(events=tuple(new_moves), output=next_phase)


# ---- helpers ---------------------------------------------------------- #


def _system_text(state: State) -> str:
    for msg in state.messages:
        if msg.get("role") == "system":
            return str(msg.get("content", ""))
    return ""


def _has_user_text(state: State) -> bool:
    return any(msg.get("role") == "user" for msg in state.messages)


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


def _load_decision_prompts(harvest: str, idx: int) -> tuple[str, str]:
    """Read events.jsonl for one decision and return (system, user) prompts.

    Layout (from the chat-server reference):
      ``$BURL_HARNESS_HARVEST_ROOT/<harvest>/corpus_index.jsonl`` —
        one decision per line; we look up by ``global_idx``.
      ``<row.transcript_path>`` (relative to ``<harvest>/``) points into
        the decision's events dir; replace ``/transcript`` to find
        ``events.jsonl``.

    The harvested ``prompt_system`` includes wax_museum's hand-edited
    Decision protocol plus raw ``<|tool>declaration`` blobs. burl-lab owns
    that surface via ``ToolSpec.protocol_phrase`` and ``render_system()``, so
    the imported base prompt stops before the legacy protocol section.
    """
    matched = _find_decision_row(harvest, "global_idx", idx)
    return _load_prompts_for_row(harvest, matched)


def _load_seeded_decision_prompts(harvest: str, seed: int) -> tuple[str, str]:
    """Return the board snapshot as the seeded decision's user prompt."""
    from burl.lab.server.ctx import build_board_snapshot_prompt

    return "", build_board_snapshot_prompt(harvest, seed)


def _find_decision_row(harvest: str, key: str, value: int) -> dict:
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


def _load_prompts_for_row(harvest: str, matched: dict) -> tuple[str, str]:
    root = _harvest_root()
    transcript_rel = matched.get("transcript_path", "")
    dec_dir = root / harvest / transcript_rel.split("/transcript")[0]
    events_path = dec_dir / "events.jsonl"

    system_content = ""
    user_content = ""
    with events_path.open() as f:
        for raw in f:
            try:
                e = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if e.get("kind") == "prompt_user":
                user_content = str(e.get("content", ""))
            elif e.get("kind") == "prompt_system":
                system_content = _strip_legacy_tool_protocol(str(e.get("content", "")))
            if system_content and user_content:
                break
    return system_content, user_content


def _strip_legacy_tool_protocol(system_content: str) -> str:
    """Drop the old wax_museum protocol/tool-declaration tail from a prompt."""
    marker = "\n# Decision protocol (wax_museum)"
    head, _sep, _tail = system_content.partition(marker)
    return head.rstrip()


PRE_GAME = _PreGamePhase()


__all__ = ["PRE_GAME", "DEFAULT_BASE_SYSTEM"]
