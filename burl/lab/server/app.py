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

import asyncio
import logging
import os
import subprocess
import sys
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
from burl.lab.core.render import render_system
from burl.lab.core.tool import Registry
from burl.lab.core.transcript import (
    EngineCommit,
    EVENTS_FILENAME,
    LmStudioChatError,
    LmStudioChatRequest,
    LmStudioChatResponse,
    PhaseEnter,
    PhaseExit,
    SessionOutcome,
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
from burl.lab.phases.pre_game import DEFAULT_BASE_SYSTEM

from .ctx import (
    build_board_snapshot_prompt,
    build_ctx_for_decision,
    build_ctx_for_session,
    build_session_outcome,
)
from .lmstudio import LmStudioClientError, LmStudioConfig, act as lmstudio_act
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
            SIMULATE_HAND_IMPACT,
        )

        reg.add(BELIEF_TRAJECTORY)
        reg.add(EXPLORE_GAME)
        reg.add(COMMIT_PLAY)
        for spec in CHAT_MINED_TOOLS:
            reg.add(spec)
        reg.add(SIMULATE_HAND_IMPACT)
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


class LmStudioChatRequestBody(BaseModel):
    model: str | None = None
    input: str | None = None
    system_prompt: str | None = None
    previous_response_id: str | None = None
    harvest: str | None = None
    seed: int | None = None
    store: bool = True
    temperature: float | None = None
    max_output_tokens: int | None = None


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

        if isinstance(last_emitted, EngineCommit):
            outcome = build_session_outcome(
                state.session_dir,
                post_state,
                last_emitted,
                ctx,
            )
            if outcome is not None:
                outcome_move = SessionOutcome(
                    stamp=now_stamp(post_state),
                    summary=outcome,
                )
                append(state.session_dir, outcome_move)
                yield outcome_move
                post_state = _load_state(sid, registry)
        post_next = post_trace.output
        if post_next and post_next != post_state.phase:
            for mv in _journal_phase_transition(
                state.session_dir, post_state.phase, post_next, post_state
            ):
                yield mv

    return EventSourceResponse(sse_from_async_iter(gen()), sep="\n")


@app.post("/api/sessions/{session_id}/lmstudio/chat")
async def launch_lmstudio_chat(session_id: str, req: LmStudioChatRequestBody) -> dict:
    """Launch or continue an LM Studio stateful chat from the current lab session."""

    registry: Registry = app_state.get("registry") or Registry()
    state = _load_state(session_id, registry)
    payload = _build_lmstudio_payload(state, registry, req)

    visible_request = dict(payload)
    request_move = LmStudioChatRequest(
        stamp=now_stamp(state),
        request=visible_request,
    )
    append(state.session_dir, request_move)
    state_after_request = _load_state(session_id, registry)

    app_launch = await asyncio.to_thread(_open_lmstudio_app)
    config = _lmstudio_config()
    ctx = _lmstudio_ctx(req, state)
    tools = _lmstudio_tools(state, registry, ctx=ctx)
    try:
        response = await asyncio.to_thread(
            lmstudio_act,
            config,
            payload,
            tools,
            ctx=ctx,
        )
    except LmStudioClientError as exc:
        detail = {
            **exc.detail(),
            "app_launch": app_launch,
            "hint": _lmstudio_hint(config),
        }
        err_move = LmStudioChatError(
            stamp=now_stamp(state_after_request),
            message=str(exc),
            detail=detail,
        )
        append(state.session_dir, err_move)
        raise HTTPException(
            status_code=502,
            detail={"message": str(exc), **detail},
        ) from exc

    response_move = LmStudioChatResponse(
        stamp=_lmstudio_response_stamp(state_after_request, response),
        response=response,
    )
    append(state.session_dir, response_move)
    return {
        "session_id": session_id,
        "request": visible_request,
        "response": response,
    }


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


def _build_lmstudio_payload(
    state: State,
    registry: Registry,
    req: LmStudioChatRequestBody,
) -> dict[str, Any]:
    model = (
        (req.model or "").strip()
        or os.environ.get("LMSTUDIO_MODEL", "").strip()
        or "ibm/granite-4-micro"
    )
    system_prompt = req.system_prompt
    if system_prompt is None:
        system_prompt = _rendered_system_for_state(state, registry)

    input_text = req.input
    if input_text is None and req.harvest and req.seed is not None:
        input_text = build_board_snapshot_prompt(req.harvest, req.seed)
    if input_text is None:
        input_text = _latest_user_text(state)
    if not input_text:
        raise HTTPException(
            status_code=400,
            detail="LM Studio chat needs input text or harvest+seed.",
        )

    payload: dict[str, Any] = {
        "model": model,
        "input": input_text,
        "system_prompt": system_prompt,
        "store": req.store,
    }
    if req.previous_response_id:
        payload["previous_response_id"] = req.previous_response_id
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.max_output_tokens is not None:
        payload["max_output_tokens"] = req.max_output_tokens
    return payload


def _lmstudio_ctx(req: LmStudioChatRequestBody, state: State) -> Any:
    if req.harvest and req.seed is not None:
        return build_ctx_for_decision(req.harvest, req.seed, key="seed")
    return build_ctx_for_session(state.session_dir)


def _lmstudio_tools(
    state: State,
    registry: Registry,
    *,
    ctx: Any,
) -> list:
    specs = []
    for name in state.advertised:
        spec = registry.find(name)
        if spec is None:
            continue
        if spec.requires_context and ctx is None:
            continue
        specs.append(spec)
    return specs


def _rendered_system_for_state(state: State, registry: Registry) -> str:
    base_text = _state_system_text(state) or DEFAULT_BASE_SYSTEM
    advertised = []
    for name in state.advertised:
        spec = registry.find(name)
        if spec is not None:
            advertised.append(spec)
    return render_system(base_text, advertised)


def _state_system_text(state: State) -> str:
    for msg in state.messages:
        if msg.get("role") == "system":
            return str(msg.get("content", ""))
    return ""


def _latest_user_text(state: State) -> str:
    for msg in reversed(state.messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _lmstudio_config() -> LmStudioConfig:
    return LmStudioConfig(
        base_url=os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234"),
        api_token=os.environ.get("LM_API_TOKEN") or None,
        timeout_s=float(os.environ.get("LMSTUDIO_TIMEOUT_S", "300")),
    )


def _open_lmstudio_app() -> dict[str, Any]:
    """Best-effort focus/open for the LM Studio desktop app.

    This does not guarantee LM Studio's Developer API server is enabled. LM
    Studio owns that toggle, so the UI still reports a clear next step if the
    subsequent API request cannot connect.
    """

    if sys.platform != "darwin":
        return {"attempted": False, "ok": False, "reason": "unsupported_platform"}
    try:
        completed = subprocess.run(
            ["open", "-a", "LM Studio"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # noqa: BLE001
        return {"attempted": True, "ok": False, "reason": str(exc)}
    return {
        "attempted": True,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }


def _lmstudio_hint(config: LmStudioConfig) -> str:
    return (
        "LM Studio was opened/focused if macOS could find it. In LM Studio, "
        f"make sure the local API server is available at {config.base_url}, "
        "install lmstudio-python in this environment if needed, then run the "
        "SDK agent again."
    )


def _lmstudio_response_stamp(state: State, response: dict[str, Any]) -> Stamp:
    stats = response.get("stats") if isinstance(response.get("stats"), dict) else {}
    input_tokens = int(stats.get("input_tokens") or 0)
    output_tokens = int(stats.get("total_output_tokens") or 0)
    ttft_s = stats.get("time_to_first_token_seconds")
    ttft_ms = int(float(ttft_s) * 1000) if ttft_s is not None else None
    tok_per_s = stats.get("tokens_per_second")
    tok_per_s_f = float(tok_per_s) if tok_per_s is not None else None
    return now_stamp(
        state,
        tok_in=input_tokens,
        tok_out=output_tokens,
        ms_ttft=ttft_ms,
        tok_per_s=tok_per_s_f,
    )


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


__all__ = ["app"]
