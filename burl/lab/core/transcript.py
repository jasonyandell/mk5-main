"""Event-sourced transcript: Stamp, Move union, Frame, Option, State + I/O.

A session is `events.jsonl` on disk. State is recovered via `fold(replay(...))`.
Every Move is a frozen dataclass with a `kind` discriminator, a `Stamp`, and a
flat payload. See SPEC.md for the wire format and yield-order contract.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Union

if TYPE_CHECKING:
    from .tool import Registry


# --------------------------------------------------------------------------- #
# Stamp                                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Stamp:
    t_wall_ms: int
    t_mono_ns: int
    tok_in: int = 0
    tok_out: int = 0
    tok_cum_in: int = 0
    tok_cum_out: int = 0
    ms_ttft: int | None = None
    ms_decode: int | None = None
    tok_per_s: float | None = None


# --------------------------------------------------------------------------- #
# Move kinds                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PhaseEnter:
    stamp: Stamp
    phase: str
    kind: str = "PhaseEnter"


@dataclass(frozen=True)
class PhaseExit:
    stamp: Stamp
    phase: str
    kind: str = "PhaseExit"


@dataclass(frozen=True)
class UserText:
    stamp: Stamp
    text: str
    kind: str = "UserText"


@dataclass(frozen=True)
class UserChoice:
    stamp: Stamp
    option_name: str
    args: dict
    kind: str = "UserChoice"


@dataclass(frozen=True)
class EngineStart:
    stamp: Stamp
    messages_hash: str
    n_messages: int
    n_tools: int
    kind: str = "EngineStart"


@dataclass(frozen=True)
class EngineToken:
    stamp: Stamp
    text: str
    kind: str = "EngineToken"


@dataclass(frozen=True)
class EngineToolCall:
    stamp: Stamp
    name: str
    args: dict
    call_id: str
    kind: str = "EngineToolCall"


@dataclass(frozen=True)
class ToolResult:
    """Journaled tool result. Mirrors `tool.ToolResult` (the impl return value)
    but stores `next_tools` as a list of names — full specs live in the registry.
    """

    stamp: Stamp
    name: str
    call_id: str
    evidence: dict
    next_tools: list[str] = field(default_factory=list)
    kind: str = "ToolResult"


@dataclass(frozen=True)
class EngineCommit:
    stamp: Stamp
    final: Any
    kind: str = "EngineCommit"


@dataclass(frozen=True)
class EngineError:
    """Engine raised mid-step. Yielded immediately before `EngineDone(reason="aborted")`.

    `traceback` may be None if the error site has no traceback (e.g. a guard
    that aborts cleanly). `during` is the call_id of an in-flight tool call if
    the error happened during dispatch — otherwise None.
    """

    stamp: Stamp
    message: str
    traceback: str | None = None
    during: str | None = None
    kind: str = "EngineError"


@dataclass(frozen=True)
class EngineDone:
    stamp: Stamp
    reason: Literal["done", "budget", "aborted", "tool_dispatch"]
    kind: str = "EngineDone"


# --- Config Moves ---------------------------------------------------------- #
# Used by phases (especially pre_game) to journal session configuration. Their
# existence is what lets us drop state.json: `state = fold(replay(events))` is
# total — there is no other source of truth.


@dataclass(frozen=True)
class SystemSet:
    """Set the verbatim system message at the head of `state.messages`.

    Replaces any existing `{"role": "system", ...}` message at index 0; if no
    system message exists, prepends one. Idempotent: re-running the same
    SystemSet produces the same state.
    """

    stamp: Stamp
    text: str
    kind: str = "SystemSet"


@dataclass(frozen=True)
class AdvertisedSet:
    """Replace `state.advertised` outright with this list of tool names.

    Validation (advertised ⊆ active_tools) is the phase's responsibility at
    handle time; fold trusts the journaled value.
    """

    stamp: Stamp
    names: list[str] = field(default_factory=list)
    kind: str = "AdvertisedSet"


@dataclass(frozen=True)
class ToolAdded:
    """Add a tool name to `state.active_tools` (idempotent)."""

    stamp: Stamp
    name: str
    kind: str = "ToolAdded"


@dataclass(frozen=True)
class ToolRemoved:
    """Remove a tool name from `state.active_tools` and `state.advertised`."""

    stamp: Stamp
    name: str
    kind: str = "ToolRemoved"


Move = Union[
    PhaseEnter,
    PhaseExit,
    UserText,
    UserChoice,
    EngineStart,
    EngineToken,
    EngineToolCall,
    ToolResult,
    EngineCommit,
    EngineError,
    EngineDone,
    SystemSet,
    AdvertisedSet,
    ToolAdded,
    ToolRemoved,
]


_MOVE_BY_KIND: dict[str, type] = {
    "PhaseEnter": PhaseEnter,
    "PhaseExit": PhaseExit,
    "UserText": UserText,
    "UserChoice": UserChoice,
    "EngineStart": EngineStart,
    "EngineToken": EngineToken,
    "EngineToolCall": EngineToolCall,
    "ToolResult": ToolResult,
    "EngineCommit": EngineCommit,
    "EngineError": EngineError,
    "EngineDone": EngineDone,
    "SystemSet": SystemSet,
    "AdvertisedSet": AdvertisedSet,
    "ToolAdded": ToolAdded,
    "ToolRemoved": ToolRemoved,
}


# --------------------------------------------------------------------------- #
# View types                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Option:
    name: str
    label: str
    args_schema: dict


@dataclass(frozen=True)
class Frame:
    phase: str
    segments: list[dict]
    active_tools: list[str]
    advertised: list[str]
    timing: dict


# --------------------------------------------------------------------------- #
# State                                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class State:
    session_dir: Path
    phase: str
    messages: tuple[dict, ...]
    active_tools: tuple[str, ...]
    advertised: tuple[str, ...]
    segments: tuple[dict, ...]
    cum_tok_in: int
    cum_tok_out: int
    started_mono_ns: int
    started_wall_ns: int


# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #

EVENTS_FILENAME = "events.jsonl"


def _move_to_json(move: Move) -> str:
    payload = asdict(move)
    return json.dumps(payload, separators=(",", ":"), default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


def append(session_dir: Path, move: Move) -> None:
    """Append one move as a JSON line to `session_dir/events.jsonl`."""
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / EVENTS_FILENAME
    line = _move_to_json(move)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def replay(session_dir: Path) -> Iterator[Move]:
    """Yield Move dataclasses by reading events.jsonl line-by-line."""
    path = session_dir / EVENTS_FILENAME
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            yield _move_from_dict(obj)


def _move_from_dict(obj: dict) -> Move:
    kind = obj.get("kind")
    cls = _MOVE_BY_KIND.get(kind)
    if cls is None:
        raise ValueError(f"unknown Move kind: {kind!r}")
    stamp = Stamp(**obj["stamp"])
    accepted = {f.name for f in fields(cls)} - {"stamp", "kind"}
    payload = {k: v for k, v in obj.items() if k in accepted}
    return cls(stamp=stamp, **payload)


# --------------------------------------------------------------------------- #
# fold                                                                        #
# --------------------------------------------------------------------------- #


def fold(
    moves: Iterable[Move],
    session_dir: Path,
    registry: "Registry | None" = None,
) -> State:
    """Rebuild State from a sequence of Moves.

    `registry` is consulted for HATEOAS resolution: when a `ToolResult` carries
    `next_tools` names, the registry is what makes those names callable. We do
    not require it here (fold is purely about state), but downstream callers
    will pass it.
    """
    phase = ""
    messages: list[dict] = []
    active: list[str] = []
    advertised: list[str] = []
    segments: list[dict] = []
    cum_in = 0
    cum_out = 0
    started_mono = 0
    started_wall = 0
    open_tool_segments: dict[str, int] = {}  # call_id -> segment index

    first = True
    for m in moves:
        if first:
            started_mono = m.stamp.t_mono_ns
            started_wall = m.stamp.t_wall_ms
            first = False

        if m.stamp.tok_cum_in:
            cum_in = max(cum_in, m.stamp.tok_cum_in)
        if m.stamp.tok_cum_out:
            cum_out = max(cum_out, m.stamp.tok_cum_out)

        k = m.kind
        if k == "PhaseEnter":
            phase = m.phase  # type: ignore[attr-defined]
            segments.append({"kind": "phase_enter", "phase": phase})
        elif k == "PhaseExit":
            segments.append({"kind": "phase_exit", "phase": m.phase})  # type: ignore[attr-defined]
        elif k == "UserText":
            messages.append({"role": "user", "content": m.text})  # type: ignore[attr-defined]
            segments.append({"kind": "user_text", "text": m.text})  # type: ignore[attr-defined]
        elif k == "UserChoice":
            segments.append(
                {
                    "kind": "user_choice",
                    "option_name": m.option_name,  # type: ignore[attr-defined]
                    "args": dict(m.args),  # type: ignore[attr-defined]
                }
            )
        elif k == "EngineStart":
            segments.append({"kind": "assistant_text", "text": ""})
        elif k == "EngineToken":
            if segments and segments[-1].get("kind") == "assistant_text":
                segments[-1]["text"] += m.text  # type: ignore[attr-defined]
            else:
                segments.append({"kind": "assistant_text", "text": m.text})  # type: ignore[attr-defined]
        elif k == "EngineToolCall":
            seg = {
                "kind": "tool_call",
                "name": m.name,  # type: ignore[attr-defined]
                "args": dict(m.args),  # type: ignore[attr-defined]
                "call_id": m.call_id,  # type: ignore[attr-defined]
                "evidence": None,
            }
            segments.append(seg)
            open_tool_segments[m.call_id] = len(segments) - 1  # type: ignore[attr-defined]
        elif k == "ToolResult":
            idx = open_tool_segments.pop(m.call_id, None)  # type: ignore[attr-defined]
            evidence = dict(m.evidence)  # type: ignore[attr-defined]
            next_names = list(m.next_tools)  # type: ignore[attr-defined]
            if idx is not None:
                segments[idx]["evidence"] = evidence
                segments[idx]["next_tools"] = next_names
            else:
                segments.append(
                    {
                        "kind": "tool_result",
                        "name": m.name,  # type: ignore[attr-defined]
                        "call_id": m.call_id,  # type: ignore[attr-defined]
                        "evidence": evidence,
                        "next_tools": next_names,
                    }
                )
            for name in next_names:
                if name not in active:
                    active.append(name)
                if name not in advertised:
                    advertised.append(name)
            messages.append(
                {
                    "role": "tool",
                    "name": m.name,  # type: ignore[attr-defined]
                    "call_id": m.call_id,  # type: ignore[attr-defined]
                    "content": evidence.get("prose", ""),
                }
            )
        elif k == "EngineCommit":
            segments.append({"kind": "commit", "final": m.final})  # type: ignore[attr-defined]
        elif k == "EngineError":
            segments.append(
                {
                    "kind": "engine_error",
                    "message": m.message,  # type: ignore[attr-defined]
                    "traceback": m.traceback,  # type: ignore[attr-defined]
                    "during": m.during,  # type: ignore[attr-defined]
                }
            )
        elif k == "EngineDone":
            segments.append({"kind": "engine_done", "reason": m.reason})  # type: ignore[attr-defined]
        elif k == "SystemSet":
            text = m.text  # type: ignore[attr-defined]
            if messages and messages[0].get("role") == "system":
                messages[0] = {"role": "system", "content": text}
            else:
                messages.insert(0, {"role": "system", "content": text})
            segments.append({"kind": "system_set", "text": text})
        elif k == "AdvertisedSet":
            advertised = list(m.names)  # type: ignore[attr-defined]
            segments.append({"kind": "advertised_set", "names": list(advertised)})
        elif k == "ToolAdded":
            name = m.name  # type: ignore[attr-defined]
            if name not in active:
                active.append(name)
            segments.append({"kind": "tool_added", "name": name})
        elif k == "ToolRemoved":
            name = m.name  # type: ignore[attr-defined]
            if name in active:
                active.remove(name)
            if name in advertised:
                advertised.remove(name)
            segments.append({"kind": "tool_removed", "name": name})

    return State(
        session_dir=session_dir,
        phase=phase,
        messages=tuple(messages),
        active_tools=tuple(active),
        advertised=tuple(advertised),
        segments=tuple(segments),
        cum_tok_in=cum_in,
        cum_tok_out=cum_out,
        started_mono_ns=started_mono,
        started_wall_ns=started_wall,
    )


# --------------------------------------------------------------------------- #
# Stamp helper                                                                #
# --------------------------------------------------------------------------- #


def now_stamp(state: State, **kw: Any) -> Stamp:
    """Build a Stamp anchored to `state`'s session start.

    Cumulative tokens default to the state's running totals; pass `tok_in` /
    `tok_out` for the *delta* this step (cumulatives auto-increment unless you
    override `tok_cum_in` / `tok_cum_out` explicitly).
    """
    mono_ns = time.monotonic_ns()
    wall_ns = time.time_ns()
    if state.started_mono_ns:
        # mono delta in ms; wall_ms is the stable session-relative clock
        t_wall_ms = (wall_ns - state.started_wall_ns) // 1_000_000
        t_mono_ns = mono_ns - state.started_mono_ns
    else:
        t_wall_ms = 0
        t_mono_ns = 0

    tok_in = int(kw.pop("tok_in", 0))
    tok_out = int(kw.pop("tok_out", 0))
    tok_cum_in = int(kw.pop("tok_cum_in", state.cum_tok_in + tok_in))
    tok_cum_out = int(kw.pop("tok_cum_out", state.cum_tok_out + tok_out))

    return Stamp(
        t_wall_ms=int(kw.pop("t_wall_ms", t_wall_ms)),
        t_mono_ns=int(kw.pop("t_mono_ns", t_mono_ns)),
        tok_in=tok_in,
        tok_out=tok_out,
        tok_cum_in=tok_cum_in,
        tok_cum_out=tok_cum_out,
        ms_ttft=kw.pop("ms_ttft", None),
        ms_decode=kw.pop("ms_decode", None),
        tok_per_s=kw.pop("tok_per_s", None),
    )


__all__ = [
    "Stamp",
    "PhaseEnter",
    "PhaseExit",
    "UserText",
    "UserChoice",
    "EngineStart",
    "EngineToken",
    "EngineToolCall",
    "ToolResult",
    "EngineCommit",
    "EngineError",
    "EngineDone",
    "SystemSet",
    "AdvertisedSet",
    "ToolAdded",
    "ToolRemoved",
    "Move",
    "Option",
    "Frame",
    "State",
    "EVENTS_FILENAME",
    "append",
    "replay",
    "fold",
    "now_stamp",
]
