"""The harness drive loop.

`drive(state, registry, engine)` is the **only** place where engine output
becomes durable Moves. It does three things and nothing else:

1. Project State → messages via ``render_messages`` and call ``engine.step``.
2. For each engine event, yield it verbatim (PhaseEnter, EngineStart,
   EngineToken, EngineToolCall, EngineDone) so the caller can journal it.
3. On a tool call, dispatch through the Registry. Yield a synthetic
   ``ToolResult`` Move carrying the evidence + the **names** of any
   ``next_tools`` the impl returned. Those next_tools are also registered
   with the Registry but **not auto-advertised** — the user is the gate
   (HATEOAS).

The caller appends each yielded Move to ``events.jsonl`` and re-folds the
State before the next ``drive`` call. ``drive`` itself doesn't write to
disk.

Termination conditions, in order:
  * ``EngineDone(reason="tool_dispatch")`` → dispatch the tool, then either
    synthesise ``EngineCommit`` (commit-role tool, exit) or re-enter
    ``engine.step`` with the augmented State.
  * ``EngineDone`` with any other reason (``done`` / ``budget`` /
    ``aborted``) → return (terminal).
  * ``EngineError`` passes through verbatim; the trailing
    ``EngineDone(reason="aborted")`` then exits the loop.

``EngineCommit`` is harness-synthesised, never engine-emitted (per SPEC).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, AsyncIterator

from .render import render_messages
from .tool import Registry, ToolResult as ImplToolResult, ToolSpec
from .transcript import (
    EngineCommit,
    EngineDone,
    EngineError,
    EngineStart,
    EngineToken,
    EngineToolCall,
    Move,
    Stamp,
    State,
    ToolResult,
    fold,
    replay,
)

log = logging.getLogger(__name__)

# The engine signals "paused; harness should dispatch a tool" by yielding
# EngineDone(reason="tool_dispatch"). Any other terminal reason exits drive.
_TOOL_DISPATCH_REASON = "tool_dispatch"


async def drive(
    state: State,
    registry: Registry,
    engine: Any,
    *,
    max_tokens: int = 2048,
    ctx: Any = None,
) -> AsyncIterator[Move]:
    """Stream Moves from one or more engine.step() invocations.

    The drive loop re-enters ``engine.step`` after every successful tool
    dispatch. State is *re-folded* between loop iterations from the moves
    we have already yielded plus any prior State events — keeping the loop
    pure with respect to the journal contract.

    Args:
      state:    Current State (typically ``fold(replay(session_dir))``).
      registry: Tool Registry. New tools surfaced by ``ToolResult.next_tools``
                are added here.
      engine:   Any object with ``async def step(messages, tools, max_tokens)``
                yielding the contracted event sequence.
      max_tokens: Per-step cap.
      ctx:      Opaque value forwarded to each ``ToolSpec.impl(ctx, args)``.
                Tools may write to it (e.g. ``commit_play`` sets ``final_play``).
    """
    current_state = state
    accumulated: list[Move] = []

    while True:
        messages = render_messages(current_state, registry)
        tools = _active_tools(current_state, registry)

        pending_tool: tuple[str, dict, str] | None = None
        done_reason: str | None = None

        async for ev in engine.step(messages=messages, tools=tools, max_tokens=max_tokens):
            # Every engine event passes through verbatim — including
            # EngineStart, EngineToken, EngineToolCall, EngineError. The
            # caller journals what we yield; we don't filter.
            yield ev
            accumulated.append(ev)

            if isinstance(ev, EngineToolCall):
                pending_tool = (ev.name, dict(ev.args), ev.call_id)
            elif isinstance(ev, EngineError):
                # Engine raised mid-step; the EngineDone(reason="aborted")
                # follows. We let the loop see EngineDone and exit.
                pass
            elif isinstance(ev, EngineDone):
                done_reason = ev.reason
                break

        if pending_tool is not None and done_reason == _TOOL_DISPATCH_REASON:
            tr_move = await _dispatch_tool(
                pending_tool=pending_tool,
                registry=registry,
                ctx=ctx,
                current_state=current_state,
            )
            yield tr_move
            accumulated.append(tr_move)

            # Commit synthesis: a commit-role tool ends the decision.
            # Per SPEC #28 the engine no longer yields EngineCommit — the
            # harness mints one here from the original tool_call args.
            #
            # Drive does NOT mint PhaseExit/PhaseEnter.  Phase transitions
            # are owned by the server: after drive returns, the server
            # calls ``in_run.handle(state, EngineCommit)`` to read
            # ``next_phase``, then journals PhaseExit/PhaseEnter itself.
            # This keeps drive engine-shaped (no phase semantics) and the
            # server transition-shaped (no engine semantics).
            committed_spec = registry.find(pending_tool[0])
            if committed_spec is not None and committed_spec.protocol_role == "commit":
                commit_move = EngineCommit(
                    stamp=_stamp_for(current_state),
                    final=dict(pending_tool[1]),
                )
                yield commit_move
                accumulated.append(commit_move)
                return

            # Re-fold State from the on-disk journal so the next
            # engine.step sees the tool result *plus* all prior session
            # moves (system text, advertised tools, user prompt, prior
            # tool calls).  Folding ``accumulated`` alone loses everything
            # that was journaled before drive() was invoked.
            current_state = fold(
                list(replay(state.session_dir)),
                session_dir=state.session_dir,
                registry=registry,
            )
            continue

        # Any other terminal condition → exit the drive loop. EngineError,
        # EngineDone(done|budget|aborted), and clean iteration end all land here.
        return


def _active_tools(state: State, registry: Registry) -> list[ToolSpec]:
    """Resolve the active tool set from State + Registry.

    State carries names; Registry resolves them to ToolSpec objects with
    callable impls. A name in State that isn't in the Registry is dropped
    silently — defensive but not silent: we log it.
    """
    out: list[ToolSpec] = []
    for name in state.active_tools:
        spec = registry.find(name)
        if spec is None:
            log.warning("[drive] active_tools name %r not in registry", name)
            continue
        out.append(spec)
    return out


async def _dispatch_tool(
    *,
    pending_tool: tuple[str, dict, str],
    registry: Registry,
    ctx: Any,
    current_state: State,
) -> ToolResult:
    """Run the tool's impl and build the ToolResult Move.

    Tool impls are sync (per the SPEC: ``Callable[[Any, dict], ToolResult]``).
    We don't await them. On failure we synthesize a ToolResult whose
    ``evidence.prose`` describes the error so the model sees something
    instead of silently looping.
    """
    name, args, call_id = pending_tool
    spec = registry.find(name)
    if spec is None:
        evidence = {
            "prose": f"ERROR: tool {name!r} is not registered.",
            "structured": {"error": "unknown_tool", "name": name},
        }
        return ToolResult(
            stamp=_stamp_for(current_state),
            name=name,
            call_id=call_id,
            evidence=evidence,
            next_tools=[],
        )

    try:
        result: ImplToolResult = spec.impl(ctx, dict(args))
    except Exception as exc:  # noqa: BLE001
        log.exception("[drive] tool %r impl raised", name)
        evidence = {
            "prose": f"ERROR: tool {name!r} raised: {exc}",
            "structured": {"error": "impl_raised", "exception": str(exc)},
        }
        return ToolResult(
            stamp=_stamp_for(current_state),
            name=name,
            call_id=call_id,
            evidence=evidence,
            next_tools=[],
        )

    next_names: list[str] = []
    for surfaced in result.next_tools:
        registry.add(surfaced)
        next_names.append(surfaced.name)

    return ToolResult(
        stamp=_stamp_for(current_state),
        name=name,
        call_id=call_id,
        evidence=dict(result.evidence),
        next_tools=next_names,
    )


def _stamp_for(state: State) -> Stamp:
    """Build a Stamp for a runtime-synthesised Move.

    We use the same anchor as the State's session start. Token counters
    are zero — the tool dispatch costs nothing model-side.
    """
    if state.started_mono_ns:
        wall_ns = time.time_ns()
        mono_ns = time.monotonic_ns()
        t_wall_ms = (wall_ns - state.started_wall_ns) // 1_000_000
        t_mono_ns = mono_ns - state.started_mono_ns
    else:
        t_wall_ms = 0
        t_mono_ns = 0
    return Stamp(
        t_wall_ms=int(t_wall_ms),
        t_mono_ns=int(t_mono_ns),
        tok_in=0,
        tok_out=0,
        tok_cum_in=state.cum_tok_in,
        tok_cum_out=state.cum_tok_out,
    )


def new_call_id() -> str:
    """Helper for engines that don't already mint a call_id."""
    return uuid.uuid4().hex[:16]


__all__ = ["drive", "new_call_id"]
