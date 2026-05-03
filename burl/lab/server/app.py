"""FastAPI app for the burl.lab harness.

Endpoints:

- ``GET /api/health`` — engine + adapter info.
- ``GET /api/sessions`` — list session dirs under the harness root.
- ``POST /api/sessions`` — create a new session (returns ``session_id``).
- ``GET /api/sessions/{id}/frame`` — current Frame (replay + fold + render).
- ``POST /api/move`` — SSE stream of Moves. Body:
  ``{"session_id": str, "move": {"kind": ..., ...}}``.

State is reconstructed via ``fold(replay(session_dir))`` — the journal IS
the snapshot. Phase handlers return trace Moves; this server interprets them:
it appends the incoming user Move, appends returned trace events, synthesises
``PhaseExit``/``PhaseEnter`` when the phase changes, and optionally drives the
engine.

Sessions live at ``$BURL_HARNESS_SESSION_ROOT`` (default
``~/.cache/burl-harness/<session_id>/``). Port 8002 by default to avoid
colliding with burl/chat (8001).
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from burl.lab.core.drive import drive
from burl.lab.core.tool import Registry
from burl.lab.core.transcript import (
    EVENTS_FILENAME,
    PhaseEnter,
    PhaseExit,
    Stamp,
    State,
    UserChoice,
    UserText,
    append,
    fold,
    now_stamp,
    replay,
)
from burl.lab.phases import PHASES

from .ctx import build_ctx_for_session
from .stream import sse_from_async_iter

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# --------------------------------------------------------------------------- #
# Session paths
# --------------------------------------------------------------------------- #


def _session_root() -> Path:
    raw = os.environ.get("BURL_HARNESS_SESSION_ROOT")
    if raw:
        return Path(raw)
    return Path.home() / ".cache" / "burl-harness"


def _session_dir(session_id: str) -> Path:
    if not session_id or "/" in session_id or ".." in session_id:
        raise HTTPException(status_code=400, detail="invalid session_id")
    return _session_root() / session_id


# --------------------------------------------------------------------------- #
# Bootstrapping
# --------------------------------------------------------------------------- #


def _load_registry() -> Registry:
    """Load the base ToolSpecs into a Registry. The ``tools`` package
    re-exports the three baseline tools; if it's not importable we still
    return an empty Registry so the server boots."""
    reg = Registry()
    try:
        from burl.lab.tools import (
            BELIEF_TRAJECTORY,
            CHAT_MINED_TOOLS,
            COMMIT_PLAY,
            EXPLORE_GAME,
        )

        reg.add(BELIEF_TRAJECTORY)
        reg.add(EXPLORE_GAME)
        reg.add(COMMIT_PLAY)
        for spec in CHAT_MINED_TOOLS:
            reg.add(spec)
    except ImportError as exc:
        log.info("[lab] tools not available: %s", exc)
    return reg


def _load_engine() -> Any:
    """Try to instantiate the real MlxEngine. If it can't load (missing
    weights, no MLX, etc.), return None — endpoints that need it will
    return a clear error rather than the server failing to boot."""
    try:
        from burl.lab.core.engine import MlxEngine

        return MlxEngine()
    except Exception as exc:  # noqa: BLE001
        log.info("[lab] engine not available: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# State load
# --------------------------------------------------------------------------- #


def _empty_state(session_dir: Path) -> State:
    return State(
        session_dir=session_dir,
        phase="pre_game",
        messages=(),
        active_tools=(),
        advertised=(),
        segments=(),
        cum_tok_in=0,
        cum_tok_out=0,
        started_mono_ns=0,
        started_wall_ns=0,
    )


def _load_state(session_id: str, registry: Registry) -> State:
    """State = fold(replay(session_dir)). One source of truth — the journal."""
    session_dir = _session_dir(session_id)
    moves = list(replay(session_dir))
    if not moves:
        return _empty_state(session_dir)
    return fold(moves, session_dir=session_dir, registry=registry)


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    log.info("[lab] starting...")
    app_state["registry"] = _load_registry()
    app_state["engine"] = _load_engine()
    log.info(
        "[lab] ready: registry=%d tool(s), engine=%s",
        len(app_state["registry"]),
        "ready" if app_state["engine"] is not None else "absent",
    )
    yield


app_state: dict = {}
app = FastAPI(title="burl.lab", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/api/health")
async def health() -> dict:
    engine = app_state.get("engine")
    info = getattr(engine, "info", None) if engine is not None else None
    if not isinstance(info, dict):
        info = {}
    registry: Registry = app_state.get("registry") or Registry()
    return {
        "ok": True,
        "engine_loaded": engine is not None,
        "model": info.get("model"),
        "adapter": info.get("adapter"),
        "n_tools": len(registry),
    }


@app.get("/api/sessions")
async def list_sessions() -> dict:
    root = _session_root()
    if not root.exists():
        return {"sessions": []}
    out: list[dict] = []
    for path in sorted(root.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        out.append(
            {
                "id": path.name,
                "has_events": (path / EVENTS_FILENAME).exists(),
            }
        )
    return {"sessions": out}


@app.post("/api/sessions")
async def create_session() -> dict:
    sid = uuid.uuid4().hex[:12]
    session_dir = _session_dir(sid)
    session_dir.mkdir(parents=True, exist_ok=True)
    # Seed the journal with PhaseEnter("pre_game") so fold(replay(...))
    # reproduces the starting phase. No state.json — journal is canonical.
    enter = PhaseEnter(stamp=Stamp(t_wall_ms=0, t_mono_ns=0), phase="pre_game")
    append(session_dir, enter)
    return {"session_id": sid}


@app.get("/api/sessions/{session_id}/frame")
async def get_frame(session_id: str) -> dict:
    registry: Registry = app_state.get("registry") or Registry()
    state = _load_state(session_id, registry)
    phase_name = state.phase or "pre_game"
    phase = PHASES.get(phase_name)
    if phase is None:
        raise HTTPException(status_code=500, detail=f"unknown phase: {phase_name}")
    frame = phase.render(state, registry)  # type: ignore[call-arg]
    options = phase.options(state, registry)  # type: ignore[call-arg]
    return {
        "session_id": session_id,
        "phase": phase_name,
        "frame": _to_jsonable(frame),
        "options": [_to_jsonable(o) for o in options],
    }


class MoveRequest(BaseModel):
    session_id: str
    move: dict


@app.post("/api/move")
async def post_move(req: MoveRequest):
    """Apply a move and SSE-stream resulting Moves.

    Flow:
      1. Load State by folding the existing journal.
      2. Decode the incoming move and append it to the journal.
      3. Run the current phase's ``handle`` — it returns journalable
         trace events plus an optional next phase.
      4. Journal the returned trace events; if the phase changed, journal
         PhaseExit/PhaseEnter.
      5. If we're now in_run with an engine available, drive the engine
         (drive() journals its own engine + tool Moves).  After drive
         returns, call the phase's ``handle`` again with the **last
         emitted Move** — typically ``EngineCommit`` — so the phase can
         signal a follow-up transition (e.g. in_run → post_turn).  Any
         transition the phase returns is journaled as PhaseExit/PhaseEnter.
    """
    sid = req.session_id
    registry: Registry = app_state.get("registry") or Registry()
    state = _load_state(sid, registry)

    incoming = _decode_move(req.move, state)
    append(state.session_dir, incoming)
    state = _load_state(sid, registry)

    phase = PHASES.get(state.phase or "pre_game")
    if phase is None:
        raise HTTPException(status_code=500, detail=f"unknown phase: {state.phase}")

    trace = await phase.handle(state, incoming, registry)  # type: ignore[call-arg]
    for mv in trace.events:
        append(state.session_dir, mv)
    new_state = _load_state(sid, registry)
    next_phase_name = trace.output

    pre_drive_moves: list = list(trace.events)
    if next_phase_name and next_phase_name != state.phase:
        pre_drive_moves.extend(_journal_phase_transition(
            state.session_dir, state.phase, next_phase_name, new_state
        ))
        new_state = _load_state(sid, registry)

    engine = app_state.get("engine")
    should_drive = new_state.phase == "in_run" and engine is not None

    # Per-session ctx cache — base tools require a real WaxContext.
    # Built lazily on first drive by replaying the journal for the most
    # recent load_decision UserChoice. See server/ctx.py and bead t42-xhnl
    # follow-up: ctx construction belongs in a phase, not server here.
    ctx: Any = None
    if should_drive:
        ctx_cache: dict = app_state.setdefault("ctx_cache", {})
        ctx = ctx_cache.get(sid)
        if ctx is None:
            try:
                ctx = build_ctx_for_session(state.session_dir)
            except Exception:  # noqa: BLE001
                log.exception("[lab] ctx build failed for session %s", sid)
                ctx = None
            if ctx is not None:
                ctx_cache[sid] = ctx

    async def gen() -> AsyncIterator[Any]:
        for mv in pre_drive_moves:
            yield mv

        if not should_drive:
            return

        last_emitted: Any = None
        try:
            async for mv in drive(new_state, registry, engine, ctx=ctx):
                append(state.session_dir, mv)
                last_emitted = mv
                yield mv
        except Exception as exc:  # noqa: BLE001
            log.exception("[lab] drive failed")
            yield {"kind": "Error", "error": str(exc)}
            return

        # After drive returns, ask the phase whether the terminal Move
        # implies a transition (e.g. EngineCommit → post_turn).  The
        # server, not drive, owns PhaseExit/PhaseEnter.
        if last_emitted is None:
            return
        post_state = _load_state(sid, registry)
        post_phase = PHASES.get(post_state.phase or "")
        if post_phase is None:
            return
        post_trace = await post_phase.handle(post_state, last_emitted, registry)  # type: ignore[call-arg]
        for mv in post_trace.events:
            append(state.session_dir, mv)
            yield mv
        if post_trace.events:
            post_state = _load_state(sid, registry)
        post_next = post_trace.output
        if post_next and post_next != post_state.phase:
            for mv in _journal_phase_transition(
                state.session_dir, post_state.phase, post_next, post_state
            ):
                yield mv

    return EventSourceResponse(sse_from_async_iter(gen()), sep="\n")


def _journal_phase_transition(
    session_dir: Path,
    from_phase: str,
    to_phase: str,
    anchor_state: State,
) -> list:
    """Append PhaseExit(from)/PhaseEnter(to) and return them for SSE."""
    exit_move = PhaseExit(stamp=now_stamp(anchor_state), phase=from_phase)
    enter_move = PhaseEnter(stamp=now_stamp(anchor_state), phase=to_phase)
    append(session_dir, exit_move)
    append(session_dir, enter_move)
    return [exit_move, enter_move]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _decode_move(payload: dict, state: State) -> Any:
    """Build a Move dataclass from a wire-format dict.

    Most user-driven moves arrive as ``UserText`` or ``UserChoice``. The
    Stamp is anchored to the current State's session start.
    """
    kind = payload.get("kind")
    stamp = now_stamp(state)
    if kind == "UserText":
        return UserText(stamp=stamp, text=str(payload.get("text", "")))
    if kind == "UserChoice":
        return UserChoice(
            stamp=stamp,
            option_name=str(payload.get("option_name", "")),
            args=dict(payload.get("args") or {}),
        )
    # Permissive: treat unknown kinds as a UserChoice so callers can pass
    # ``{"option_name": ..., "args": {...}}`` without specifying ``kind``.
    return UserChoice(
        stamp=stamp,
        option_name=str(payload.get("option_name", "")),
        args=dict(payload.get("args") or {}),
    )


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


__all__ = ["app"]
