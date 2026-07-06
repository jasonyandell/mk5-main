---
title: conditional_outcome structurally unused (confounded — see Reframe)
kind: topic
first_seen: 35c75ff
last_updated: 54f7776
status: reframed
---

## Reframe (54f7776)

The "zero usage across 145+ decisions" observation had a trivial mechanical cause that went undetected until 54f7776: **Gemma 4's chat template silently drops `role="tool"` messages.**

`burl/harness/tool_loop_native.py` was packing tool responses into `role="tool"` messages. Gemma 4's Jinja template wraps the entire render loop in `{%- if message['role'] != 'tool' -%}`, so those messages were silently excluded from the rendered prompt. The model never saw any tool responses — not just `conditional_outcome`'s output, but all tool responses from every tool, across every Burl rollout before this fix (54f7776).

**What this invalidates:**

- `conditional_outcome = 0/145` is trivially explained: the model never saw any tool response, so it could not follow up with a composed call. Not "structurally unused" — structurally invisible.
- The three A/B environment-shape experiments (JSON → prose → ASCII+if/then → "probe pivot" environment) ran under this confound. Their null results may not reflect genuine model behavior with visible tool responses.
- All measured bot-match numbers for Burl adapters were produced with invisible tool responses. What those adapters actually learned is unknown.
- The design-signal conclusions ("remove `conditional_outcome` from the harness," "simplify tool surface") are premature. Until a full eval runs with the fix applied, it is not known whether the model would use `conditional_outcome` when it can actually see tool responses.

**Meta-lesson (from PRACTICALITIES §9):** audit the *rendered prompt*, not the messages dict, before concluding "the model can't do X."

**After the fix (N=5, 54f7776):** base Gemma 4 E2B achieves 5/5 bot-match, faithfully quotes numeric output from tool responses, and uses the pivot value verbatim. This is qualitatively different behavior from the pre-fix era.

## Original observation (pre-54f7776)

What was observed prior to the fix is recorded here as historical fact. The observation was real; the interpretation was wrong.

`conditional_outcome` — the tool that combines [[zeb]] belief estimates with E[Q] framework outcome simulation — showed zero calls across 145+ decisions (35c75ff, 39aafaf, dbadb5f).

| Context | Model | Decisions | conditional_outcome calls |
|---|---|---|---|
| Single-decision runner (spike) | Haiku 4.5 | ~10 | 0 |
| Single-decision runner (smoke) | Opus 4.7 | ~3 | 0 |
| Full-game arena, seed 900010 | Haiku 4.5 | 28 | 0 |
| Full-game arena, seed 900010 | Opus 4.7 | 28 | 0 |
| **Total** | | **145+** | **0** |

The prior interpretation was that this reflected model behavior (Zeb distrust, planning overhead). The correct interpretation is that the model received no tool responses at all during these sessions.

## What conditional_outcome does

`conditional_outcome(play, assume)` is a composed tool: (1) ask Zeb for `P(assume | visible)`, then (2) ask the E[Q] framework for the outcome conditional on `assume`, return the slice. It is the only tool in Burl's surface that combines belief + outcome simulation — the counterfactual reasoning primitive the architecture was originally designed around (burl/OVERVIEW.md @ 8d26e0d).

## Open question (post-fix)

With tool responses now visible, does the model reach for `conditional_outcome` zero-shot? This question is reopened. The first post-fix N=5 result (5/5 bot-match) is encouraging but does not address conditional_outcome usage specifically. Requires a fresh N≥50 eval (54f7776).

## Links

[[burl]] [[zeb]] [[decisions/zeb-parked-eq-primitive]] [[tool-orchestration]] [[selfplay-arena]] [[expected-q-value]]
