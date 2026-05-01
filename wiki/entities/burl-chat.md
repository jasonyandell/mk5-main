---
title: burl-chat — Interactive workbench for talking with Burl
kind: entity
first_seen: cba521d
last_updated: cba521d
status: active
---

## What it is

`burl/chat/` is a standalone interactive workbench for talking with [[burl]] mid-session. It loads a wax_museum decision from any [[burl-2000-harvest]]-class harvest as the conversation prefix, lets the user chat with the model about that decision, and renders Burl's reasoning in typed segments (thoughts, tool calls, tool results, commits) instead of as a single opaque assistant turn. Three-pane web UI: harvest/bucket/decision picker (left), conversation (center), composer (bottom).

Lives entirely under `burl/chat/`. Has its own `requirements.txt`, its own `web/package.json`, and its own dev workflow. Imports from `burl/` (specifically `burl.modal.gemma_local`) but is otherwise independent of the main game web app under `src/`.

## Architecture

One Python process owns both the model (in-process via [[mlx-lm]] through `burl.modal.gemma_local.GemmaLocalNative`) and the API. Svelte 5 + Vite frontend talks to it over HTTP/SSE. **No vLLM, no separate inference server** — vLLM was abandoned twice on this project (see [[star-harness]] history) and never reintroduced.

```
┌────────────────────────────────────────────────┐
│ burl/chat/server/  (one Python process)        │
│   FastAPI ── inference.py ── GemmaLocalNative  │
│      │             │              (MLX-LM)     │
│      ▼                                         │
│   SSE tokens                                   │
└────────────────────────────────────────────────┘
                ▲
                │  HTTP/SSE on :8001
                ▼
        burl/chat/web/  (Vite :5173, proxies /api)
```

### Files

| Path | Role |
|---|---|
| `server/app.py` | FastAPI app: `/api/health`, `/api/chat` (SSE), `/api/harvests/*` |
| `server/inference.py` | Single-thread executor wrapping `GemmaLocalNative`; `on_chunk` → asyncio.Queue → SSE bridge |
| `server/decisions.py` | Reads harvest `corpus_index.jsonl` + per-decision `events.jsonl`; reconstructs typed segments |
| `web/src/App.svelte` | Three-pane UI; segment-typed conversation rendering; live thinking indicator |
| `web/src/lib/parse.ts` | Streaming parser for `<\|channel>thought ... <channel\|>` and `<\|tool_call>...<tool_call\|>` markers |
| `web/src/lib/api.ts` | Fetch + SSE consumer (CRLF-aware framing) |

## Three load-bearing implementation details

**1. Single-thread MLX executor.** MLX's default GPU stream is bound to the first thread that touches it. Loading the model on the FastAPI startup thread and then trying to generate from `loop.run_in_executor(None, ...)` fails immediately with `There is no Stream(gpu, 0) in current thread.` Fix: a `ThreadPoolExecutor(max_workers=1)` that loads the model AND runs every generate call. Both happen on the same thread; the default stream stays valid.

**2. CRLF SSE frames.** sse-starlette emits frames separated by `\r\n\r\n` (literal carriage returns) per the SSE spec. The browser's `TextDecoder` returns the bytes verbatim. A `buf.split("\n\n")` consumer never finds a frame boundary and yields nothing — the response body grows on the wire (Network tab shows 41 kB delivered) but the streaming generator never yields a single event. Fix: strip `\r` before splitting, OR split on `\r?\n\r?\n`. Methodological note: every diagnostic via `curl` on a terminal hides CR, so the bug is invisible until you `repr()` the raw bytes.

**3. Svelte 5 + parser reactivity.** Svelte 5's `$state` proxy doesn't see in-place mutations to objects already in the array. The streaming parser cannot do `last.content += text` and expect the UI to update — the parser holds a reference to a now-proxied object, and writing through the unproxied reference doesn't trigger reactivity. Fix: replace the segment at the array index with a fresh `{...last, content: last.content + text}` object on every append. Same shape, new reference, Svelte re-renders.

## Decision-loader

Reads any `harvest_batched_*` directory under `scratch/belief_trajectory_rollout/` (configurable via `BURL_CHAT_HARVEST_ROOT`). Lists by bucket — `ALL_AGREE_CORRECT`, `BURL_BREAKS_CONSENSUS`, `BURL_INDEPENDENT_RIGHT`, `BURL_INDEPENDENT_WRONG`, `BURL_PARROTS_PI_WRONG`, `BURL_DRIFTS_FROM_PI`, `FORCED_COMMIT`, `ILLEGAL`, `BOTH_FIX`, `QMEAN_ALONE_FIXES`, `BURL_ALONE_FIXES`, `ALL_AGREE_WRONG` — color-coded by category (green for agreement-correct, orange for breaks-consensus, red for wrong, blue for fixes, gray for meta).

Per decision: pulls `events.jsonl`, groups events by turn, emits typed segments: `system`, `user`, `thinking` (per turn), `tool_call` (per turn, parsed `tool` + `args`), `tool_result` (per turn, with content), `assistant_text` (post-strip), `commit` (final play). Strips orphan `<\|channel>thought` / `<channel\|>` / `<turn\|>` markers from `assistant_text`.

## Chat-mode primer

After loading a decision, the workbench injects a synthetic chat-cadence assistant turn after the `commit` segment: *"Yeah, I committed N. The decision is done — ask me anything about it and I'll talk it through with you. No more tool calls."* This breaks the play-decision rhythm via in-context recency. See [[chat-mode-primer]].

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `BURL_CHAT_MODEL_REPO` | `mlx-community/gemma-4-e2b-it-bf16` | Base model |
| `BURL_CHAT_ADAPTER_PATH` | (none) | Local adapter dir; falls back to base if unset |
| `BURL_CHAT_HARVEST_ROOT` | `scratch/belief_trajectory_rollout` | Where to look for `harvest_batched_*` dirs |

## Why this exists

[[gus]] was conceived from the start to be the belief brain that informs the LLM at decision time. The natural product slot above that is **post-commit conversation with the player** — talking with Burl after a hand the way you'd talk with a teammate. Closest published precedent: chapters 2–8 of Roberson's *Winning 42*, which is a worked-example dialogue about bidding hands. See [[post-commit-q-and-a]] for the research framing.

The workbench is the place to figure out what the corpus for an eventual "post-commit Burl" adapter should look like, before paying the cost of generating one. The spike that birthed it ([[burl-chat-spike]]) already produced its first finding: stacked play-decision adapters lock the model into commit_play even with explicit "do not output a tool call" + chat primer; base Gemma + same prefix engages in real prose Q&A. See [[play-adapter-lock-in]].
