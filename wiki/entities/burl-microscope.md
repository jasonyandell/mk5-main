---
title: burl-microscope — Human-in-the-loop Burl recipe workbench
kind: entity
first_seen: local-2026-05-06
last_updated: local-2026-05-06
status: active
---

## What it is

`burl/microscope/` is a small human-in-the-loop experiment loop for [[burl]]. It
loads one harvested decision, one editable recipe, and one [[gemma-4-e2b]]
conversation. The user steps the model turn-by-turn, inspects tool calls/results,
edits recipe files, and reruns the same case. It is intentionally simpler than
[[burl-lab]]: no phase state machine, no training corpus goal, no production-agent
claim.

## Recipe shape

A recipe is a directory under `burl/microscope/recipes/<name>/`:

| File | Role |
|---|---|
| `system.md` | Base Burl system prompt. |
| `play.md` | User/play prompt template with simple `{{var}}` substitutions. |
| `tools.json` | Active tools plus optional description/protocol overrides. |
| `params.json` | Model repo, adapter path, max-token cap. |
| `tool_responses/<tool>.md` | Optional renderer for a tool's response text. |

The rendered system prompt still uses first-class `ToolSpec` values from
[[burl-lab]]; recipe overrides let the experiment change tool descriptions,
protocol phrases, and protocol roles without editing the Python tool
implementation.

## Pi client

A project-local Pi extension at `.pi/extensions/burl-microscope.ts` registers
`/burl` commands and can route ordinary typed input to Burl while "Burl mode" is
on. Pi is the terminal client; the microscope server owns Gemma inference,
Gemma-native `tool_calls`/`tool_responses`, tool execution, and JSONL traces.

## First smoke result

On `harvest_batched_20260425_072910`, `global_idx=1` (`BURL_BREAKS_CONSENSUS`):

| Recipe | Tool path | Final | Match |
|---|---|---:|---|
| `baseline` | `belief_trajectory → board_snapshot → explore_game(21) → play_brief(21) → play_brief(25) → play_brief(25) → commit_play(25)` | 25 | original Burl, not oracle |
| `legal-brief` | `legal_plays → play_brief(19) → play_brief(25) → belief_trajectory → commit_play(19)` | 19 | oracle / pi / qmean |

This is the microscope's motivating use case: the same base model and case can
flip from the harvested Burl mistake to the oracle play through a recipe-level
prompt/tool-protocol change. The result is a single-case smoke, not a batch
claim.

## Boundary

Use [[burl-microscope]] for prompt/tool/tool-response exploration with the user in
the loop. Use [[burl-lab]] when the experiment needs event-sourced phase machinery,
HATEOAS tool advertisement, LM Studio SDK lanes, or production-like session
journaling.

## Related

[[burl]] · [[burl-lab]] · [[burl-chat]] · [[wax-museum]] · [[gemma-tool-response-shape]] · [[burl-2000-harvest]]
