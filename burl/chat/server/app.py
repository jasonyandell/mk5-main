"""FastAPI app for burl/chat workbench.

Phase 0: ``/api/health`` + ``/api/chat`` (SSE). One in-process MLX-LM model.
No sessions, no tools registry, no decision loading yet — that's Phase 1.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from . import decisions, improvised_tools, tools_runner
from .inference import InferenceEngine

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 0.6
    enable_thinking: bool = False
    stop_at_tool_call: bool = True


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    log.info("[burl-chat] loading model...")
    app_state["engine"] = InferenceEngine()
    log.info("[burl-chat] ready: %s", app_state["engine"].info)
    yield


app_state: dict = {}
app = FastAPI(title="burl-chat", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    engine = app_state.get("engine")
    if engine is None:
        return {"ok": False, "reason": "engine not loaded"}
    return {"ok": True, **engine.info}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    engine: InferenceEngine = app_state["engine"]

    async def event_stream():
        async for event in engine.stream(
            messages=[m.model_dump() for m in req.messages],
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            enable_thinking=req.enable_thinking,
            stop_at_tool_call=req.stop_at_tool_call,
        ):
            yield {"data": json.dumps(event)}

    return EventSourceResponse(event_stream())


@app.get("/api/harvests")
async def harvests() -> list[dict]:
    return decisions.list_harvests()


@app.get("/api/harvests/{harvest}/decisions")
async def harvest_decisions(
    harvest: str,
    bucket: str | None = None,
    limit: int = 40,
    offset: int = 0,
) -> list[dict]:
    return decisions.list_decisions(
        harvest, bucket=bucket, limit=limit, offset=offset,
    )


@app.get("/api/harvests/{harvest}/decisions/{global_idx}")
async def harvest_decision(harvest: str, global_idx: int) -> dict:
    try:
        return decisions.load_decision(harvest, global_idx)
    except KeyError as e:
        return {"error": str(e)}
    except FileNotFoundError as e:
        return {"error": f"missing: {e}"}


class ToolRunRequest(BaseModel):
    harvest: str
    global_idx: int
    tool: str
    args: dict = {}


@app.get("/api/tools")
async def list_tools() -> dict:
    return {"tools": tools_runner.available_tools()}


class ImprovisedToolRequest(BaseModel):
    name: str
    description: str
    python_src: str


@app.get("/api/improvised_tools")
async def list_improvised() -> dict:
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "declaration": improvised_tools.declaration(t),
            }
            for t in improvised_tools.list_all()
        ],
    }


@app.post("/api/improvised_tools")
async def register_improvised(req: ImprovisedToolRequest) -> dict:
    """Hot-register a tool from Python source. Replaces if name exists."""
    try:
        t = improvised_tools.register(
            name=req.name,
            description=req.description,
            python_src=req.python_src,
        )
        return {
            "ok": True,
            "name": t.name,
            "description": t.description,
            "declaration": improvised_tools.declaration(t),
        }
    except Exception as e:  # noqa: BLE001
        log.exception("[burl-chat] improvised register failed")
        return {"ok": False, "error": str(e)}


@app.delete("/api/improvised_tools/{name}")
async def delete_improvised(name: str) -> dict:
    return {"ok": improvised_tools.unregister(name)}


# ---------------------------------------------------------------------- #
# Chat-state share: a one-slot ring the frontend pushes into so the MCP  #
# server can let Claude read what Burl just said without copy-paste.     #
# Lives in process memory; cleared on restart, not persisted.            #
# ---------------------------------------------------------------------- #
_LAST_SHARED_CHAT: dict | None = None


class ChatShareRequest(BaseModel):
    decision: dict | None = None
    segments: list[dict] = []
    note: str | None = None


@app.post("/api/chat/share")
async def share_chat(req: ChatShareRequest) -> dict:
    """Frontend pushes its current segments here so the MCP server can
    surface them to Claude. Latest-only — overwrites the previous snapshot.
    """
    import time
    global _LAST_SHARED_CHAT
    _LAST_SHARED_CHAT = {
        "shared": True,
        "shared_at": time.time(),
        "decision": req.decision,
        "segments": req.segments,
        "note": req.note,
    }
    return {"ok": True, "n_segments": len(req.segments)}


@app.get("/api/chat/shared")
async def get_shared_chat() -> dict:
    if _LAST_SHARED_CHAT is None:
        return {"shared": False, "reason": "no chat shared yet"}
    return _LAST_SHARED_CHAT


@app.post("/api/tools/run")
async def run_tool(req: ToolRunRequest) -> dict:
    """Live-run a wax_museum tool against the actual game state for the
    given harvest decision. Same prose format Burl saw during training.
    """
    import asyncio

    try:
        # Tool work is CPU/GPU-heavy; offload to the MLX executor's pool so
        # we don't block the event loop. Reuse the engine's executor since
        # belief_trajectory hits the Gus model (which is on the same device).
        engine: InferenceEngine = app_state["engine"]
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            engine._executor,
            tools_runner.run_tool,
            req.harvest, req.global_idx, req.tool, req.args,
        )
        return {"ok": True, **result}
    except Exception as e:  # noqa: BLE001
        log.exception("[burl-chat] tool run failed")
        return {"ok": False, "error": str(e), "tool": req.tool, "args": req.args}
