"""Tool spine: ToolSpec, ToolResult (the impl-return value), Registry.

The system prompt is RENDERED from the active ToolSpec set — never hand-edited.
Tools advertise follow-up tools via `ToolResult.next_tools` (HATEOAS): the
runtime registers them and surfaces their names to the model as user-selectable
options for subsequent steps.

`Ctx` (the first arg to `impl`) is whatever runtime context tools need at call
time. We currently alias it to `Any` to avoid a hard import of
`burl.wax_museum.tools.WaxContext`. When runtime+tools are in, we'll either
(a) lift WaxContext into burl.lab.core, or (b) alias `Ctx = WaxContext` here
behind a `TYPE_CHECKING` guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


ProtocolRole = Literal["first_read", "candidate_eval", "diagnostic", "commit"]


@dataclass(frozen=True)
class ToolResult:
    """What `ToolSpec.impl(ctx, args)` returns.

    `evidence` is the substantive payload — `{"prose": str, "structured": Any}`.
    `next_tools` carries full ToolSpecs (HATEOAS); the runtime registers them
    and adds their names to `state.advertised`. The journaled `ToolResult` move
    in transcript.py stores only the names — full specs live in the Registry.
    """

    evidence: dict
    next_tools: tuple["ToolSpec", ...] = ()
    next_phase: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    """Declarative description of a tool. The system prompt renders from these."""

    name: str
    description: str
    params: dict                 # JSON schema (draft 7)
    example: str                 # literal call string, e.g. 'explore_game(play=14)'
    protocol_role: ProtocolRole
    protocol_phrase: str         # the literal sentence the system prompt should include
    impl: Callable[[Any, dict], ToolResult]
    requires_context: bool = False


class Registry:
    """In-memory ToolSpec registry. Insertion-ordered."""

    def __init__(self) -> None:
        self._by_name: dict[str, ToolSpec] = {}

    def add(self, spec: ToolSpec) -> None:
        self._by_name[spec.name] = spec

    def remove(self, name: str) -> None:
        self._by_name.pop(name, None)

    def find(self, name: str) -> ToolSpec | None:
        return self._by_name.get(name)

    def active(self) -> list[ToolSpec]:
        return list(self._by_name.values())

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)


__all__ = ["ProtocolRole", "ToolResult", "ToolSpec", "Registry"]
