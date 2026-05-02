---
title: improvised-tools — Hot-register tool registry inside burl-chat
kind: entity
first_seen: 2026-05-01
last_updated: 2026-05-01
status: active
---

## What it is

A registry inside [[burl-chat]] that lets a tool author (Claude, sitting next to the user) compose new wax_museum-style tools mid-conversation, register them with the running chat server, and push their declarations into Burl's context — all without restarting the model or editing `burl/wax_museum/tools.py`. Built for the experiment loop: Burl articulates what tool would help (the [[burl-tool-wishlist]]), Claude implements it, the user advertises it on the next turn and watches whether Burl's reasoning shifts.

Lives at `burl/chat/server/improvised_tools.py` plus `burl/chat/mcp_server.py` for the Claude-facing bridge.

## Architecture

```
Claude (in another window)
   │  MCP: register_improvised_tool(name, description, python_src)
   ▼
burl/chat/mcp_server.py    (FastMCP, proxies to chat HTTP API)
   │  POST /api/improvised_tools
   ▼
burl/chat/server/app.py
   │  improvised_tools.register(...)
   ▼
burl/chat/server/improvised_tools.py
   │  ┌── compile + exec → callable
   │  ├── _REGISTRY[name] = ImprovisedTool(...)
   │  └── _persist(t) → tools_library/<name>.py
```

On import, the module rehydrates the registry from `tools_library/` so tools survive server restarts. The library path defaults to `burl/chat/server/tools_library/` and can be overridden with `BURL_CHAT_TOOLS_LIBRARY`.

## Tool contract

Every tool source must define:

```python
def tool(ctx, **kwargs) -> dict:
    # ctx is burl.wax_museum.tools.WaxContext
    # ctx.game_state — ZebGameState (hands, played, play_history,
    #   decl_id, bid_state, team_points, current_trick, trick_leader)
    # ctx.me_abs    — narrator's absolute seat (0-3)
    # ctx.oracle    — E[Q] oracle (use sparingly, ~seconds per call)
    return {"prose": str, "structured": Any}
```

Imports are unrestricted — any module under `burl.*` or `forge.*` is fair game. The runner in `tools_runner.py` checks the improvised registry **first**, so a hot-registered tool of the same name as a base tool wins the dispatch (useful for trying alternative renderings without removing the original).

## Persisted file format

Each registered tool is written to disk as `<name>.py`:

```python
DESCRIPTION = "One-read state synthesis: my hand, trump hierarchy ..."

from burl.wax_museum.snapshot import (
    render_full_board_snapshot,
    render_full_board_structured,
)

def tool(ctx, **kwargs):
    ...
```

The loader strips the leading `DESCRIPTION = ...\n\n` block so re-registration of a hydrated tool doesn't stack DESCRIPTION lines. The files are intentionally human-readable and check-in-able — the eventual library curation path is to promote good tools into the repo.

## API surface

HTTP (chat server):

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/improvised_tools` | List registered tools with their wax_museum declaration strings |
| POST | `/api/improvised_tools` | Register or replace a tool (compiles + persists) |
| DELETE | `/api/improvised_tools/{name}` | Unregister + delete the file |

MCP (Claude-side, via `burl/chat/mcp_server.py`):

| Tool | Purpose |
|---|---|
| `read_chat_state` | Latest snapshot the user pushed via "share with Claude" — segments, decision meta, harvest |
| `inspect_decision(harvest, global_idx)` | Full decision payload for context while authoring |
| `register_improvised_tool(name, description, python_src)` | Compile + register |
| `run_tool(harvest, global_idx, name, args)` | Smoke-test against live game state before advertising |
| `unregister_improvised_tool(name)` | Remove from registry and disk |
| `list_improvised_tools` / `list_tools` | Inspect registry / full tool surface visible to Burl |

## Library so far (2026-05-01)

| Tool | Genesis | Addresses |
|---|---|---|
| `board_snapshot` | Burl asked for "Comprehensive Game State Visualizer" on decision #0 | Wraps `render_full_board_snapshot` — paragraph-form state synthesis |
| `legal_plays` | Burl committed an illegal 6-6 on decision #7 (must-follow ones) and asked for a "Strategic Synthesis Engine" | Names led suit + lists legal vs illegal plays from current hand. State, not strategy. |
| `state_brief` | Burl said decision #0's input was "slightly confusing" and sketched a labeled-section format | Renders state in `[GAME STATE]` / `[CONTEXT & GOAL]` / `[PROTOCOL]` bullets per Burl's own spec |
| `play_brief` | Burl asked for `explore_game` output reshaped with a headline + risk label + objective link | Same outcome distribution, rendered as `HEADLINE` (variance + p_make) → modes sorted by mass with `[BIG WIN]`/`[WIN]`/`[NEAR-BREAKEVEN]`/`[LOSS]`/`[DISASTER]` labels + catalysts → risk profile. Reuses `WaxContext.get_or_build()` cache so it costs zero extra oracle calls if you've already explored that play. |

Each tool addresses a specific failure mode or articulated need from a real Burl transcript. See [[burl-tool-wishlist]] for the framing.

### Adoption asymmetry

Tools land in Burl's rotation in proportion to whether the system prompt's **Decision protocol** section names them. `state_brief` is consulted reliably in rerun-fresh sessions because the workbench's `state_brief` declaration plus its self-described "first read on any decision" line is enough to clear Burl's "what does the protocol expect first" check. `play_brief` sat unused in the same rerun even though its declaration was in scope — Burl followed the protocol's literal `explore_game(play=X)` instruction. Lesson: adding a tool to the registry makes it *callable*; getting it *called* requires the protocol text in the system prompt to mention it (or to rename the existing tool to point at the new one). Adapter co-training would also do it, but for the spike loop, protocol text is the cheap lever.

## Doctrine

The wax_museum system prompt is explicit: *"State tools answer WHAT IS the state — they never tell you WHAT TO DO."* Improvised tools must respect this boundary. When Burl asks for a "synthesis engine that picks the best play for me," the right move is to interpret the underlying need (state comprehension, legal-move clarity, format readability) and build a tool that surfaces facts, not picks. `legal_plays` is the worked example — Burl asked for a play-picker, got a legal-move enumerator instead, because the picker would violate the contract that makes the corpus trainable later.

## Bug fixes that landed during the build

- **`advertise tools` was inert** — the button only mutated the local `segments` array; the model never re-streamed. Fix: `await send("")` after appending the synthetic user turn. Doctrine: any UI button that appends a segment intended to elicit a response must explicitly trigger the continuation.
- **`$effect` re-checked unchecked tools** — the auto-select logic re-added every known tool to the selection set on every render. Fix: a separate `seenToolNames` set so tools auto-select only on first appearance; subsequent unchecks stick.

## Related

- [[burl-chat]] — host workbench
- [[burl-tool-wishlist]] — what Burl's meta-requests reveal about its reasoning gaps
- [[play-adapter-lock-in]] — why Burl articulates needs in the form of "plan tool calls" instead of plain prose
- [[post-commit-q-and-a]] — the eventual research direction the library is feeding
