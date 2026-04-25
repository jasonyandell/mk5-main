---
title: wax_museum (hard-gated HATEOAS harness)
kind: entity
first_seen: 54f7776
last_updated: 1bf1885
status: active
---

## What it is

wax_museum is a minimal [[burl]] harness under `burl/wax_museum/` that hard-gates the tool
surface using HATEOAS (Hypermedia As The Engine Of Application State). At each gate state
the harness tells the model which tools are legal for that turn, forcing it to reason about
outcome distributions rather than committing from framing alone. (commit message @ 54f7776)

## Gate sequence

```
explore_game  →  probe_*  →  commit_play
```

The model cannot skip to `commit_play` without first traversing the probe states.

## Extension hooks (1bf1885)

Three opt-in extension points let sweep runners vary the belief-trajectory uptake protocol
without forking the harness:

| Hook | Purpose | Used by |
|---|---|---|
| `system_prompt_transform` | Edit rendered system block after default gate instructions | Variant D (required_first) — replaces protocol with "Turn 1: call belief first" |
| `preload_tool_calls` | Execute tools before model's first turn; inject as synthetic assistant tool_calls/tool_responses pair | Variant B — model wakes with [[belief-trajectory]] already in context |
| `menu_override` | Swap default `menu_for` lookup per-turn | Variant E (strip belief) and F (belief_only) |

Used for Phase 1 six-variant belief-trajectory sweep. (commit message @ 1bf1885)

## [[belief-trajectory]] integration

[[belief-trajectory]] is wired as a free (non-advancing) side-call in every gate state.
`_append_gate_instructions` names it explicitly in the protocol text so schema and prompt
stay aligned. `on_tool_result` callback on `run_decision_waxed` streams full tool-response
payloads into live logs. (commit message @ d858781)

## Plugin architecture

Harness takes `parse_completion` + `tool_response_style` plugins so [[gemma-4-e2b]] (native
`assistant.tool_responses` shape) and Qwen (OpenAI-style `role="tool"`) share the same
infrastructure. Introduced alongside the chat-template bug fix in 54f7776.
