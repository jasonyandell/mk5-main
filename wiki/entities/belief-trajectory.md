---
title: belief_trajectory tool
kind: entity
first_seen: d858781
last_updated: 1bf1885
status: active
---

## What it is

`belief_trajectory` is a [[burl]] tool at `burl/tools/belief_trajectory.py` (761 LOC) that
exposes [[gus]]'s calibrated belief head (`v3_consistency_10000g` adapter) as per-domino
posterior + shift-since-last + V (state value) + CLS attention. It is [[burl]]'s production
belief primitive, replacing the parked [[zeb]]. (commit message @ d858781)

## Output

`format="both"` returns:
- **Structured dict**: programmatic callers, per-domino posteriors, KL-ranked shifts
- **LLM-legible prose**: STRONG/MEDIUM/WEAK tiers, KL-ranked shifts, annotated attention
  tokens

## Bridge to Gus

`burl/tools/_gus_adapter.py` bridges Burl's duck-typed `game_state` to Gus's
seat-symmetric tokenizer via a `_ShimDecision` record walk. Gus adapter:
`v3_consistency_10000g`. (commit message @ d858781)

## Integration with wax_museum

Wired into [[wax-museum]] as a free (non-advancing) side-call in every gate state.
`_append_gate_instructions` names `belief_trajectory` explicitly in protocol text to avoid
schema/prompt mismatch. `on_tool_result` callback on `run_decision_waxed` allows streaming
full tool-response payloads into live logs. (commit message @ d858781)

## Extension hooks

Three hooks in [[wax-museum]] (1bf1885) control how `belief_trajectory` enters the model's
context: via `system_prompt_transform`, `preload_tool_calls` (model wakes with belief
already injected), or `menu_override` (gate can strip or enforce belief-only).
