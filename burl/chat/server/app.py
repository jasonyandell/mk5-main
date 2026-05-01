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

from . import decisions
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
