"""Small human-in-the-loop Burl experiment core.

This is intentionally simpler than ``burl.lab``: a session is one case, one
recipe, one Gemma conversation, and an append-only JSONL trace. The user can
step the model one turn at a time, inspect tool calls/results, edit recipe
files, and rerun the same case.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from burl.lab.core.render import render_system
from burl.lab.core.transcript import EngineDone, EngineStart, EngineToken, EngineToolCall
from burl.lab.core.tool import Registry, ToolSpec
from burl.lab.tools import (
    BELIEF_TRAJECTORY,
    BOARD_SNAPSHOT,
    COMMIT_PLAY,
    EXPLORE_GAME,
    LEGAL_PLAYS,
    PLAY_BRIEF,
    SIMULATE_HAND_IMPACT,
    STATE_BRIEF,
)

DEFAULT_HARVEST_ROOT = Path("scratch/belief_trajectory_rollout")
DEFAULT_SESSION_ROOT = Path("scratch/burl-microscope/sessions")
DEFAULT_RECIPE_ROOT = Path("burl/microscope/recipes")
DEFAULT_HARVEST = "harvest_batched_20260425_072910"


# --------------------------------------------------------------------------- #
# Registry + recipes
# --------------------------------------------------------------------------- #


def load_registry() -> Registry:
    reg = Registry()
    for spec in (
        BELIEF_TRAJECTORY,
        STATE_BRIEF,
        BOARD_SNAPSHOT,
        LEGAL_PLAYS,
        PLAY_BRIEF,
        SIMULATE_HAND_IMPACT,
        EXPLORE_GAME,
        COMMIT_PLAY,
    ):
        reg.add(spec)
    return reg


@dataclass(frozen=True)
class ToolOverride:
    name: str
    description: str | None = None
    protocol_phrase: str | None = None
    protocol_role: str | None = None


@dataclass(frozen=True)
class Recipe:
    name: str
    path: Path
    system_template: str
    play_template: str
    tool_overrides: tuple[ToolOverride, ...]
    params: dict[str, Any]

    @property
    def max_tokens(self) -> int:
        return int(self.params.get("max_tokens", 2048))

    def tool_specs(self, registry: Registry) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for override in self.tool_overrides:
            base = registry.find(override.name)
            if base is None:
                raise KeyError(f"recipe {self.name!r} references unknown tool {override.name!r}")
            spec = base
            if (
                override.description is not None
                or override.protocol_phrase is not None
                or override.protocol_role is not None
            ):
                spec = replace(
                    base,
                    description=override.description or base.description,
                    protocol_phrase=override.protocol_phrase or base.protocol_phrase,
                    protocol_role=override.protocol_role or base.protocol_role,
                )
            specs.append(spec)
        return specs

    def render_tool_response(self, tool_name: str, evidence: dict[str, Any]) -> str:
        template_path = self.path / "tool_responses" / f"{tool_name}.md"
        prose = str(evidence.get("prose", ""))
        structured = evidence.get("structured", {})
        if not template_path.exists():
            return prose
        template = template_path.read_text(encoding="utf-8")
        return _render_template(
            template,
            {
                "prose": prose,
                "structured_json": json.dumps(structured, indent=2, sort_keys=True, default=str),
                "structured_json_compact": json.dumps(structured, sort_keys=True, separators=(",", ":"), default=str),
            },
        )


def recipe_root() -> Path:
    return Path(os.environ.get("BURL_MICROSCOPE_RECIPE_ROOT", str(DEFAULT_RECIPE_ROOT)))


def list_recipes(root: Path | None = None) -> list[str]:
    root = root or recipe_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "system.md").exists())


def load_recipe(name: str, root: Path | None = None) -> Recipe:
    root = root or recipe_root()
    path = root / name
    if not path.exists():
        raise FileNotFoundError(path)
    system_template = (path / "system.md").read_text(encoding="utf-8")
    play_template = (path / "play.md").read_text(encoding="utf-8")
    tool_overrides = _load_tool_overrides(path / "tools.json")
    params_path = path / "params.json"
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
    return Recipe(
        name=name,
        path=path,
        system_template=system_template,
        play_template=play_template,
        tool_overrides=tuple(tool_overrides),
        params=params,
    )


def _load_tool_overrides(path: Path) -> list[ToolOverride]:
    if not path.exists():
        return [
            ToolOverride("belief_trajectory"),
            ToolOverride("explore_game"),
            ToolOverride("commit_play"),
        ]
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[ToolOverride] = []
    for item in raw:
        if isinstance(item, str):
            out.append(ToolOverride(name=item))
        elif isinstance(item, dict):
            out.append(
                ToolOverride(
                    name=str(item["name"]),
                    description=item.get("description"),
                    protocol_phrase=item.get("protocol_phrase"),
                    protocol_role=item.get("protocol_role"),
                )
            )
        else:
            raise TypeError(f"Unsupported tool entry in {path}: {item!r}")
    return out


# --------------------------------------------------------------------------- #
# Harvest cases
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Case:
    harvest: str
    lookup_key: str
    lookup_value: int
    row: dict[str, Any]
    meta: dict[str, Any]
    harvested_user_prompt: str
    board_snapshot: str
    ctx: Any

    @property
    def label(self) -> str:
        return f"{self.harvest} {self.lookup_key}={self.lookup_value}"


def harvest_root() -> Path:
    return Path(os.environ.get("BURL_MICROSCOPE_HARVEST_ROOT", str(DEFAULT_HARVEST_ROOT)))


def session_root() -> Path:
    return Path(os.environ.get("BURL_MICROSCOPE_SESSION_ROOT", str(DEFAULT_SESSION_ROOT)))


def load_case(
    harvest: str = DEFAULT_HARVEST,
    *,
    idx: int | None = None,
    seed: int | None = None,
) -> Case:
    if idx is None and seed is None:
        idx = 1
    key = "seed" if seed is not None else "global_idx"
    value = int(seed if seed is not None else idx)
    row = _find_decision_row(harvest, key=key, value=value)
    meta, harvested_user = _read_decision_events(harvest, row)
    ctx = _build_wax_ctx(meta, harvested_user)
    board_snapshot = _render_board_snapshot(ctx)
    return Case(
        harvest=harvest,
        lookup_key=key,
        lookup_value=value,
        row=row,
        meta=meta,
        harvested_user_prompt=harvested_user,
        board_snapshot=board_snapshot,
        ctx=ctx,
    )


def _find_decision_row(harvest: str, *, key: str, value: int) -> dict[str, Any]:
    idx_path = harvest_root() / harvest / "corpus_index.jsonl"
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    with idx_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get(key) == value:
                return row
    raise KeyError(f"{key} {value} not found in {harvest}")


def _read_decision_events(harvest: str, row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    transcript_rel = str(row.get("transcript_path", ""))
    dec_dir = harvest_root() / harvest / transcript_rel.split("/transcript")[0]
    events_path = dec_dir / "events.jsonl"
    meta: dict[str, Any] | None = None
    user_content = ""
    with events_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") == "meta":
                meta = event
            elif event.get("kind") == "prompt_user":
                user_content = str(event.get("content", ""))
            if meta and user_content:
                break
    if meta is None or not user_content:
        raise RuntimeError(f"missing meta or prompt_user in {events_path}")
    return meta, user_content


def _build_wax_ctx(meta: dict[str, Any], prompt_user: str) -> Any:
    # Reuse the already-vetted Burl Lab context builder. It loads the E[Q]
    # oracle and reconstructs public state from the harvested prompt history.
    from burl.lab.server.ctx import _build_wax_ctx as build_wax_ctx

    return build_wax_ctx(meta, prompt_user)


def _render_board_snapshot(ctx: Any) -> str:
    from burl.chat.server.tools_library.board_snapshot import tool

    payload = tool(ctx)
    return str(payload.get("prose", ""))


# --------------------------------------------------------------------------- #
# Microscope session
# --------------------------------------------------------------------------- #


@dataclass
class StepResult:
    session_id: str
    events: list[dict[str, Any]]
    assistant_text: str
    tool_call: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    committed: bool
    outcome: dict[str, Any] | None


class MicroscopeSession:
    def __init__(
        self,
        *,
        case: Case,
        recipe: Recipe,
        registry: Registry,
        sid: str | None = None,
    ) -> None:
        self.sid = sid or uuid.uuid4().hex[:12]
        self.case = case
        self.recipe = recipe
        self.registry = registry
        self.created_at = time.time()
        self.dir = session_root() / self.sid
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tool_specs = recipe.tool_specs(registry)
        self.messages = self._initial_messages()
        self.committed = False
        self.final_domino_id: int | None = None
        self._write_event(
            "session_start",
            session_id=self.sid,
            case=self.case_summary(),
            recipe=self.recipe_summary(),
            rendered_system=self.messages[0]["content"],
            user_prompt=self.messages[1]["content"],
        )

    def case_summary(self) -> dict[str, Any]:
        row = self.case.row
        meta = self.case.meta
        return {
            "harvest": self.case.harvest,
            "lookup_key": self.case.lookup_key,
            "lookup_value": self.case.lookup_value,
            "bucket": row.get("bucket"),
            "seed": row.get("seed", meta.get("seed")),
            "global_idx": row.get("global_idx"),
            "legal_plays": meta.get("legal_plays"),
            "oracle_play": row.get("oracle_play", meta.get("bot_play")),
            "burl_play": row.get("burl_play"),
            "pi_play": row.get("pi_play"),
            "qmean_play": row.get("qmean_play"),
            "forced_commit": row.get("forced_commit"),
        }

    def recipe_summary(self) -> dict[str, Any]:
        return {
            "name": self.recipe.name,
            "path": str(self.recipe.path),
            "tools": [spec.name for spec in self.tool_specs],
            "params": dict(self.recipe.params),
        }

    def prompt_summary(self) -> dict[str, str]:
        return {
            "system": self.messages[0]["content"],
            "user": self.messages[1]["content"],
        }

    def set_tools(self, names: Iterable[str]) -> None:
        overrides = [ToolOverride(str(name)) for name in names]
        self.recipe = replace(self.recipe, tool_overrides=tuple(overrides))
        self.tool_specs = self.recipe.tool_specs(self.registry)
        # Keep the same base system template, but re-render the protocol from
        # the new active tools. This changes future turns without rewriting
        # the already logged initial prompt.
        self.messages[0] = {
            "role": "system",
            "content": render_system(self.recipe.system_template, self.tool_specs),
        }
        self._write_event("tools_set", tools=[spec.name for spec in self.tool_specs])

    async def step(self, engine: Any, *, user_text: str = "", max_tokens: int | None = None) -> StepResult:
        if user_text:
            self.messages.append({"role": "user", "content": user_text})
            self._write_event("user", text=user_text)

        assistant_chunks: list[str] = []
        events: list[dict[str, Any]] = []
        pending_tool: dict[str, Any] | None = None
        done_reason = ""
        token_cap = int(max_tokens or self.recipe.max_tokens)

        async for move in engine.step(
            messages=list(self.messages),
            tools=list(self.tool_specs),
            max_tokens=token_cap,
        ):
            if isinstance(move, EngineStart):
                event = {
                    "kind": "engine_start",
                    "n_messages": move.n_messages,
                    "n_tools": move.n_tools,
                    "tok_in": move.stamp.tok_in,
                }
            elif isinstance(move, EngineToken):
                assistant_chunks.append(move.text)
                event = {"kind": "assistant_delta", "text": move.text}
            elif isinstance(move, EngineToolCall):
                pending_tool = {
                    "name": move.name,
                    "args": dict(move.args),
                    "call_id": move.call_id,
                }
                event = {"kind": "tool_call", **pending_tool}
            elif isinstance(move, EngineDone):
                done_reason = move.reason
                event = {
                    "kind": "engine_done",
                    "reason": move.reason,
                    "tok_out": move.stamp.tok_cum_out,
                    "ms_decode": move.stamp.ms_decode,
                    "tok_per_s": move.stamp.tok_per_s,
                }
            else:
                event = {"kind": type(move).__name__, "repr": repr(move)}
            events.append(event)
            self._write_event(**event)

        assistant_text = "".join(assistant_chunks)
        tool_result: dict[str, Any] | None = None
        if pending_tool is not None and done_reason == "tool_dispatch":
            tool_result = self._dispatch_tool(pending_tool)
            assistant_turn = self._assistant_turn_with_tool_result(
                assistant_text,
                pending_tool,
                tool_result["response"],
            )
            self.messages.append(assistant_turn)
            self._write_event(
                "assistant",
                text=assistant_text,
                thought=assistant_turn.get("content", ""),
                tool_call=pending_tool,
                tool_response=tool_result["response"],
                done_reason=done_reason,
            )
            events.append({"kind": "tool_result", **tool_result})
            self._write_event("tool_result", **tool_result)
        elif assistant_text or pending_tool:
            self.messages.append({"role": "assistant", "content": assistant_text})
            self._write_event("assistant", text=assistant_text, done_reason=done_reason)

        outcome = self.outcome() if self.committed else None
        if outcome is not None:
            self._write_event("outcome", outcome=outcome)

        return StepResult(
            session_id=self.sid,
            events=events,
            assistant_text=assistant_text,
            tool_call=pending_tool,
            tool_result=tool_result,
            committed=self.committed,
            outcome=outcome,
        )

    async def auto(self, engine: Any, *, user_text: str = "", max_steps: int = 8) -> list[StepResult]:
        results: list[StepResult] = []
        next_user = user_text
        for _ in range(max_steps):
            result = await self.step(engine, user_text=next_user)
            results.append(result)
            next_user = ""
            if result.committed:
                break
            # If the model did not call a tool, pause for human input.
            if result.tool_call is None:
                break
        return results

    def outcome(self) -> dict[str, Any] | None:
        if self.final_domino_id is None:
            return None
        legal_payload = self._run_tool_by_name("legal_plays", {})
        legal_structured = dict(legal_payload.get("structured", {}))
        legal_plays = [int(x) for x in legal_structured.get("legal_plays", [])]
        row = self.case.row
        references = {
            "pi_play": _maybe_int(row.get("pi_play")),
            "qmean_play": _maybe_int(row.get("qmean_play")),
            "burl_play": _maybe_int(row.get("burl_play")),
            "oracle_play": _maybe_int(row.get("oracle_play", self.case.meta.get("bot_play"))),
        }
        return {
            "final_domino_id": self.final_domino_id,
            "legal": self.final_domino_id in legal_plays,
            "legal_plays": legal_plays,
            "references": references,
            "matches": {
                name.removesuffix("_play"): (
                    self.final_domino_id == value if value is not None else None
                )
                for name, value in references.items()
            },
        }

    def _initial_messages(self) -> list[dict[str, str]]:
        variables = self._template_variables()
        system_base = _render_template(self.recipe.system_template, variables)
        user_prompt = _render_template(self.recipe.play_template, variables)
        return [
            {"role": "system", "content": render_system(system_base, self.tool_specs)},
            {"role": "user", "content": user_prompt},
        ]

    def _template_variables(self) -> dict[str, Any]:
        row = self.case.row
        meta = self.case.meta
        return {
            "harvest": self.case.harvest,
            "lookup_key": self.case.lookup_key,
            "lookup_value": self.case.lookup_value,
            "bucket": row.get("bucket", ""),
            "seed": row.get("seed", meta.get("seed", "")),
            "global_idx": row.get("global_idx", ""),
            "oracle_play": row.get("oracle_play", meta.get("bot_play", "")),
            "burl_play": row.get("burl_play", ""),
            "pi_play": row.get("pi_play", ""),
            "qmean_play": row.get("qmean_play", ""),
            "legal_plays": json.dumps(meta.get("legal_plays", [])),
            "harvest_user": self.case.harvested_user_prompt,
            "board_snapshot": self.case.board_snapshot,
        }

    def _dispatch_tool(self, call: dict[str, Any]) -> dict[str, Any]:
        name = str(call["name"])
        args = dict(call.get("args") or {})
        spec = next((s for s in self.tool_specs if s.name == name), None) or self.registry.find(name)
        if spec is None:
            evidence = {
                "prose": f"ERROR: unknown tool {name!r}",
                "structured": {"error": "unknown_tool", "name": name},
            }
        else:
            try:
                impl_result = spec.impl(self.case.ctx, args)
                evidence = dict(impl_result.evidence)
            except Exception as exc:  # noqa: BLE001
                evidence = {
                    "prose": f"ERROR: tool {name!r} raised: {exc}",
                    "structured": {"error": "tool_raised", "exception": str(exc)},
                }

        if name == "commit_play":
            raw = args.get("domino_id", args.get("play"))
            try:
                self.final_domino_id = int(raw)
                self.committed = True
            except (TypeError, ValueError):
                evidence = {
                    "prose": f"ERROR: malformed commit_play args: {args}",
                    "structured": {"error": "bad_commit_args", "args": args},
                }

        response_text = self.recipe.render_tool_response(name, evidence)
        return {
            "name": name,
            "args": args,
            "call_id": str(call.get("call_id", "")),
            "response": response_text,
            "evidence": evidence,
        }

    def _assistant_turn_with_tool_result(
        self,
        assistant_text: str,
        call: dict[str, Any],
        response_text: str,
    ) -> dict[str, Any]:
        """Gemma-native assistant turn carrying tool call + response.

        Gemma 4's chat template silently drops separate role='tool' messages.
        The native shape it was trained on stores both `tool_calls` and
        `tool_responses` on the assistant turn. Keeping this shape here is the
        whole point of routing through the Burl runtime instead of Pi's generic
        tool loop.
        """
        try:
            from burl.harness.tool_loop_native import parse_native_completion

            thought, _calls, _commit = parse_native_completion(assistant_text)
        except Exception:  # noqa: BLE001
            thought = assistant_text
        name = str(call["name"])
        args = dict(call.get("args") or {})
        return {
            "role": "assistant",
            "content": thought,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
            ],
            "tool_responses": [{"name": name, "response": response_text}],
        }

    def _run_tool_by_name(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        spec = self.registry.find(name)
        if spec is None:
            return {"prose": f"ERROR: unknown tool {name}", "structured": {}}
        result = spec.impl(self.case.ctx, args)
        return dict(result.evidence)

    def _write_event(self, kind: str | None = None, **payload: Any) -> None:
        if kind is None:
            kind = str(payload.pop("kind", "event"))
        event = {"ts": time.time(), "kind": kind, **payload}
        path = self.dir / "events.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str, separators=(",", ":")) + "\n")


def _render_template(template: str, variables: dict[str, Any]) -> str:
    out = template
    for key, value in variables.items():
        out = out.replace("{{" + key + "}}", str(value))
    return out


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _jsonable_step(result: StepResult) -> dict[str, Any]:
    return {
        "session_id": result.session_id,
        "events": result.events,
        "assistant_text": result.assistant_text,
        "tool_call": result.tool_call,
        "tool_result": result.tool_result,
        "committed": result.committed,
        "outcome": result.outcome,
    }


__all__ = [
    "DEFAULT_HARVEST",
    "Case",
    "MicroscopeSession",
    "Recipe",
    "StepResult",
    "_jsonable_step",
    "list_recipes",
    "load_case",
    "load_recipe",
    "load_registry",
]
