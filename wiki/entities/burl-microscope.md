---
title: burl-microscope — Human-in-the-loop Burl recipe workbench
kind: entity
first_seen: local-2026-05-06
last_updated: local-2026-05-07
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

`snapshot-hypothesis` is the current garage-tinker variant: it keeps the clean
`board_snapshot()`-first prompt and active tools `board_snapshot`, `legal_plays`,
`play_brief`, `belief_trajectory`, `simulate_hand_impact`, and `commit_play`.

## Pi client

A project-local Pi extension at `.pi/extensions/burl-microscope.ts` registers
`/burl` commands and can route ordinary typed input to Burl while "Burl mode" is
on. Pi is the terminal client; the microscope server owns Gemma inference,
Gemma-native `tool_calls`/`tool_responses`, tool execution, and JSONL traces.

## First smoke result and correction

On `harvest_batched_20260425_072910`, `global_idx=1` (`BURL_BREAKS_CONSENSUS`),
the initial smoke compared `baseline` and `legal-brief`. The first `legal-brief`
version committed oracle/consensus play `19`, but its prompt exposed
`oracle/reference play: 19` and `original Burl play: 25`. That result is now treated
as a reference-leakage smoke, not evidence that the protocol alone fixed the case.

After stripping oracle/original-Burl references and using a minimal prompt shaped as
`board_snapshot()` output plus decide text, both `snapshot-first` and fair
`legal-brief` chose `25` on this case. The interactive session where the user talked
with Burl and asked for `board_snapshot()` also reached `19`, but that conversation
still contained the original prompt's reference leak. The frontier lesson is
methodological: `board_snapshot()` is a good first-read surface, but this case is not
yet a clean prompt-only win.

## Burl-requested hypothesis tool

A later interactive session asked Burl what information was missing. Burl named a
"Hypothetical Hand Simulation" / "Opponent Hand Query" tool and proposed
`simulate_hand_impact`: given a candidate play and a concrete hidden-hand hypothesis,
report how likely the hypothesis is and how the play's outcome distribution changes if
it is true.

That request is now first-class as `simulate_hand_impact(play_id=X, seat=..., holds=Y)`.
The tool combines [[belief-trajectory]]'s marginal probability for the queried
seat/domino with [[wax-museum]]'s conditional E[Q] machinery, rendering the answer as
"plausibility + baseline Q + conditional Q + shift" rather than another broad
posterior table. It is meant to follow `play_brief` catalyst lines, not replace
candidate evaluation.

## Boundary

Use [[burl-microscope]] for prompt/tool/tool-response exploration with the user in
the loop. Use [[burl-lab]] when the experiment needs event-sourced phase machinery,
HATEOAS tool advertisement, LM Studio SDK lanes, or production-like session
journaling.

## Related

[[burl]] · [[burl-lab]] · [[burl-chat]] · [[wax-museum]] · [[gemma-tool-response-shape]] · [[burl-2000-harvest]]
