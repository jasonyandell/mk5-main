---
title: "Gemma Tool Responses: assistant.tool_responses, not role='tool'"
kind: decision
first_seen: 54f7776
last_updated: 54f7776
status: active
---

## Decision

For [[gemma-4-e2b]], tool responses must be attached to the assistant message as `assistant.tool_responses=[{name, response}]` — Gemma's native shape. Do NOT use separate `role="tool"` messages. For Qwen and OpenAI-style models, keep the `role="tool"` pattern. The harness takes `parse_completion` + `tool_response_style` plugins so both paths share infrastructure.

## Why

> Gemma 4's Jinja chat template wraps the whole render loop in `{%- if message['role'] != 'tool' -%}` — every message with `role="tool"` is silently dropped.

— commit 54f7776

Every Burl rollout before this commit ran with tool outputs invisible to the model. The harness sent tool responses as `role="tool"` messages; Gemma's chat template silently discarded all of them.

## What this invalidates / reframes

This is the largest single retroactive confound in the Burl replay:

- **`conditional_outcome=0/145` structural finding** ([[conditional-outcome-structural-nonuse]]) — trivially explained by invisibility. The model never saw the distribution outputs it was supposed to reason with.
- **"Environment-shape ceiling" from three A/B runs** (JSON → prose → ASCII+if/then+pivot) — all three were conducted with tools invisible. The A/B was measuring Gemma's behavior under zero tool input, not under varying tool formats.
- **Every pre-fix adapter's measured bot-match** — all confounded. iter-0 through iter-3-rules, the base-model K1 baseline, the spike v2 88.9% — none of these had working tool responses.

Post-fix validation: base Gemma 4 E2B 5/5 bot-match on N=5 held-out trick-6 decisions, faithful numeric quoting, pivot quoted verbatim from tool response. See [[experiments/chat-template-fix-validation]].

## Generalizable principle

Audit the *rendered prompt*, not the messages dict, before concluding "the model can't do X." Silent-drop via chat template is the quietest failure mode — parallel to [[decisions/sft-max-seq-length]]'s silent truncation and [[decisions/sft-completion-only-loss]]'s gradient waste. Three known TRL/template traps now documented for Burl.

## Related pages

[[burl]] · [[gemma-4-e2b]] · [[tool-orchestration]] · [[decisions/sft-max-seq-length]] · [[decisions/sft-completion-only-loss]] · [[conditional-outcome-structural-nonuse]] · [[wax-museum]] · [[experiments/chat-template-fix-validation]] · [[sources/54f7776]]
