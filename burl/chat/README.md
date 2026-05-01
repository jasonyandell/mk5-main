# burl/chat — interactive workbench for Burl

Standalone REPL + web UI for talking with a loaded Burl adapter mid-game-state, with
live tool reload, conversation rewind, and full prompt-byte audit.

This is a workbench, not a product. Use it to investigate what the model thinks,
not to ship to anyone.

## Architecture

One Python process owns both the model (in-process via `burl.modal.gemma_local`)
and the API. Svelte 5 + Vite frontend talks to it over HTTP/SSE. No vLLM, no
separate inference server.

```
┌────────────────────────────────────────────────┐
│ burl/chat/server/  (one Python process)        │
│   FastAPI ── inference.py ── GemmaLocalNative  │
│      │             │              (MLX-LM)     │
│      │             ▼                           │
│      │        tools registry  (Phase 1)        │
│      ▼                                         │
│   SSE tokens                                   │
└────────────────────────────────────────────────┘
                ▲
                │  HTTP/SSE on :8001
                ▼
        burl/chat/web/  (Vite :5173)
```

## Phase 0 — pipe works

- `/api/health` reports the loaded model + adapter
- `/api/chat` (SSE) streams tokens from a real generate call
- Browser shows one chat pane, sends a message, watches it stream

## Run it

One-time setup:

```bash
# from repo root
uv pip install -r burl/requirements-mlx.txt
uv pip install -r burl/chat/requirements.txt

cd burl/chat/web && npm install
```

Day-to-day (two terminals):

```bash
# terminal 1 — API + model
cd burl/chat
uvicorn server.app:app --reload --port 8001

# terminal 2 — frontend
cd burl/chat/web
npm run dev
```

Open http://localhost:5173.

## Configuration

Environment variables read at server startup:

| Var | Default | Notes |
|---|---|---|
| `BURL_CHAT_MODEL_REPO` | `mlx-community/gemma-4-e2b-it-bf16` | Base model |
| `BURL_CHAT_ADAPTER_PATH` | (none) | Local adapter dir; falls back to base if unset |

Pull the iter3-rules adapter to a local path with `huggingface-cli download
jasonyandell/gemma-4-e2b-texas42-burl-iter3-rules` and point
`BURL_CHAT_ADAPTER_PATH` at the resulting directory.
