"""Pure rendering of the system prompt and chat-message projection.

This module is the **single source of truth for prompt assembly**. Whatever
the engine sees is what `render_system` + `render_messages` produced — no
side channels, no implicit appends elsewhere.

Two pure functions:

- `render_system(base_text, advertised)` composes the final system text by
  appending a Decision Protocol section. The Decision Protocol is built
  from each advertised ToolSpec's ``protocol_phrase`` grouped by
  ``protocol_role``. Stable order ⇒ stable string ⇒ stable adoption.

- `render_messages(state, registry)` projects the State into the
  OpenAI-style chat-message list the engine consumes. The system message
  is **re-rendered every call**, so toolset changes propagate without any
  rebuild step.

No I/O. No mutation. No engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from typing import get_args

from .tool import ProtocolRole, Registry, ToolSpec

if TYPE_CHECKING:  # pragma: no cover
    from .transcript import State


# Stable role ordering for the Decision Protocol section. Roles outside
# this list are appended at the end in first-seen order, so any HATEOAS-
# surfaced tool with a novel role doesn't break the prompt.
_ROLE_ORDER: tuple[str, ...] = (
    "first_read",
    "candidate_eval",
    "diagnostic",
    "commit",
)

# Assert at import time that _ROLE_ORDER stays in lockstep with the
# canonical ProtocolRole literal in core/tool.py. If a peer adds a role
# without updating us, the import fails fast instead of silently
# bucketing the new role into the trailing "first-seen" tail.
assert set(_ROLE_ORDER) == set(get_args(ProtocolRole)), (
    f"_ROLE_ORDER {_ROLE_ORDER} drifted from ProtocolRole {get_args(ProtocolRole)}"
)

_PROTOCOL_HEADER = "## Decision Protocol"
_PROTOCOL_PREAMBLE = (
    "You have access to the following tools. Each tool plays a specific "
    "role in the decision loop; use the protocol phrase to decide when to "
    "call it."
)


def render_system(base_text: str, advertised: Iterable[ToolSpec]) -> str:
    """Return the final system prompt string.

    The Decision Protocol section is **appended** to ``base_text``. With
    zero advertised tools the section is omitted entirely (a bare
    pre-game prompt with no tools renders as just the base text).
    """
    base = (base_text or "").rstrip()
    advertised = list(advertised)
    if not advertised:
        return base

    grouped: dict[str, list[ToolSpec]] = {}
    for spec in advertised:
        grouped.setdefault(spec.protocol_role, []).append(spec)

    seen: list[str] = [r for r in _ROLE_ORDER if r in grouped]
    for r in grouped:
        if r not in seen:
            seen.append(r)

    lines: list[str] = ["", _PROTOCOL_HEADER, "", _PROTOCOL_PREAMBLE, ""]
    for role in seen:
        lines.append(f"### {role}")
        lines.append("")
        for spec in grouped[role]:
            phrase = (spec.protocol_phrase or "").strip()
            if phrase:
                lines.append(f"- **{spec.name}** — {phrase}")
            else:
                lines.append(f"- **{spec.name}**")
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()

    return base + "\n" + "\n".join(lines) + "\n"


def render_messages(state: "State", registry: Registry) -> list[dict]:
    """Project a State into the OpenAI-style chat message list.

    The system message is re-rendered every call (so changing the
    advertised set takes effect on the next engine invocation without
    any explicit rebuild). All other messages come straight from
    ``state.messages`` — those have already been folded from the
    canonical event sequence.
    """
    advertised = _resolve_advertised(state, registry)
    base_text = _state_system_text(state)
    out: list[dict] = [
        {"role": "system", "content": render_system(base_text, advertised)},
    ]
    for msg in state.messages:
        out.append(dict(msg))
    return out


def _resolve_advertised(state: "State", registry: Registry) -> list[ToolSpec]:
    advertised: list[ToolSpec] = []
    for name in state.advertised:
        spec = registry.find(name)
        if spec is not None:
            advertised.append(spec)
    return advertised


def _state_system_text(state: "State") -> str:
    """Pull the base system text from the State.

    State has no dedicated ``system_text`` field (the system message is
    derived). We look in ``messages`` for an existing ``role="system"``
    entry and use its content as the base text; otherwise empty string.
    """
    for msg in state.messages:
        if msg.get("role") == "system":
            return str(msg.get("content", ""))
    return ""


__all__ = ["render_system", "render_messages"]
