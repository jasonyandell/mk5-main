---
title: burl-chat — Interactive workbench for talking with Burl
kind: entity
first_seen: 2026-04-30
last_updated: 2026-05-02
status: superseded
---

> **Superseded.** [[burl-lab]] (greenfield event-sourced platform with first-class ToolSpec + Phase machine) replaced burl-chat as the active experimentation surface. Last commit `18a5b94` (2026-05-03); the whole [[burl]] line has had no commits since 2026-05-07. This page is kept as the working reference for the plumbing burl-lab ported forward — do not delete or restructure.

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
| `server/app.py` | FastAPI app: `/api/health`, `/api/chat` (SSE), `/api/harvests/*`, `/api/improvised_tools` |
| `server/inference.py` | Single-thread executor wrapping `GemmaLocalNative`; `on_chunk` → asyncio.Queue → SSE bridge |
| `server/decisions.py` | Reads harvest `corpus_index.jsonl` + per-decision `events.jsonl`; reconstructs typed segments |
| `server/improvised_tools.py` | Hot-register registry; compiles + persists tools to `tools_library/`; rehydrates on import. See [[improvised-tools]]. |
| `server/tools_runner.py` | Looks up an incoming tool call in the improvised registry first, then base tools |
| `server/tools_library/` | Persisted tool source: `<name>.py` files with a `DESCRIPTION = "..."` constant + `def tool(ctx, **kwargs)` |
| `mcp_server.py` | FastMCP server bridging Claude (the assistant authoring tools) to the chat server; exposes `read_chat_state`, `register_improvised_tool`, `run_tool`, etc. |
| `web/src/App.svelte` | Three-pane UI; segment-typed conversation rendering; live thinking indicator; tool-library popover with per-tool checkboxes + delete buttons |
| `web/src/lib/parse.ts` | Streaming parser for `<\|channel>thought ... <channel\|>` and `<\|tool_call>...<tool_call\|>` markers |
| `web/src/lib/api.ts` | Fetch + SSE consumer (CRLF-aware framing) |
| `burl/wax_museum/snapshot.py` | `render_full_board_snapshot` / `render_full_board_structured` — pure rule-based state-rendering helpers reused by improvised tools (`board_snapshot` wraps these directly) |

## Three load-bearing implementation details

**1. Single-thread MLX executor.** MLX's default GPU stream is bound to the first thread that touches it. Loading the model on the FastAPI startup thread and then trying to generate from `loop.run_in_executor(None, ...)` fails immediately with `There is no Stream(gpu, 0) in current thread.` Fix: a `ThreadPoolExecutor(max_workers=1)` that loads the model AND runs every generate call. Both happen on the same thread; the default stream stays valid.

The executor pattern is **necessary but insufficient** for full MLX threading correctness. `mlx_lm.generate` declares `generation_stream` at module scope, so whichever thread imports the submodule owns the stream — and FastAPI lifespan imports run on the event-loop thread, not the executor thread. burl-chat's GemmaLocalNative path happens to import on the executor thread by construction, leaving the bug latent here; [[burl-lab]] surfaced it explicitly and ships the rebind via `sys.modules["mlx_lm.generate"]` after `load()`. Full diagnosis (and the submodule-shadowing trap that makes the naive `from mlx_lm import generate; generate.generation_stream = ...` silently no-op) lives on [[mlx-lm]] under "Upstream bug: module-level generation_stream."

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

## Improvised-tool registry

A hot-register layer on top of the base wax_museum tools. Claude (acting as the user's tool-author teammate via the [[improvised-tools]] MCP bridge) reads the live chat state, designs a tool against Burl's request, and POSTs Python source to `/api/improvised_tools`. The server compiles it, registers it under a `<name> → ImprovisedTool` map, and writes it to `tools_library/<name>.py` for future hydration. Tools become callable on the next turn with no restart.

The web UI presents the registry as a popover in the header: each tool shows up with a checkbox + a delete (×). The user picks a subset, hits **advertise**, and the workbench appends a synthetic user turn carrying the wax_museum-format declarations and re-streams the model. The same tools can be re-advertised across decisions; selection state survives registry refreshes (newly registered tools auto-select on first appearance, but unchecks stick).

This is the surface where the [[burl-tool-wishlist]] gets answered. Burl articulates a need; Claude implements it; the user evaluates whether it shifts Burl's reasoning.

## Rerun-fresh

Two modes for working a decision:

1. **Join-at-end (default).** Click a decision in the sidebar; the workbench replays the harvest's full typed-segment trace and appends the chat-mode primer. The user converses with Burl about the finished play. This is the post-commit-Q&A path.
2. **Rerun-fresh.** Click the **rerun fresh** button. The workbench rebuilds `segments` as `[harvested_system_with_appended_improvised_tool_declarations, harvested_first_user_message]` and triggers `send("")`. Burl plays the decision from turn 1 with the user's selected improvised tools available from the system prompt onward — no chat-mode primer, no harvested trace to anchor on. The workbench's existing tool dispatch (which checks the improvised registry first) handles the new tools transparently.

Rerun-fresh is the surface where the wishlist tools earn or lose their keep. Mechanism: pick a decision the original Burl handled poorly (high-regret bucket like `BURL_BREAKS_CONSENSUS` or any illegal-commit case), check the tool subset, click rerun. Compare regret of the new commit against the harvested baseline. If the same tool produces lower regret across multiple seeds, it's a candidate for promotion to `burl/wax_museum/tools.py`; if the protocol's literal tool references override its discovery, the system prompt is the next intervention layer.

## Why this exists

[[gus]] was conceived from the start to be the belief brain that informs the LLM at decision time. The natural product slot above that is **post-commit conversation with the player** — talking with Burl after a hand the way you'd talk with a teammate. Closest published precedent: chapters 2–8 of Roberson's *Winning 42*, which is a worked-example dialogue about bidding hands. See [[post-commit-q-and-a]] for the research framing.

The workbench is the place to figure out what the corpus for an eventual "post-commit Burl" adapter should look like, before paying the cost of generating one. The spike that birthed it ([[burl-chat-spike]]) already produced its first finding: stacked play-decision adapters lock the model into commit_play even with explicit "do not output a tool call" + chat primer; base Gemma + same prefix engages in real prose Q&A. See [[play-adapter-lock-in]].
