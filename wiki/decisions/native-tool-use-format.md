---
title: Native Tool-Use Format (not XML)
kind: decision
first_seen: 2026-04-19
last_updated: 2026-04-19
status: superseded
---

## Decision

Burl's harness emits tool calls in Gemma's native `<|tool_call>` format (driven by the chat template's `tools=[...]`), not in XML `<tool>`/`<commit>` tags. `commit_play(domino_id)` is a native tool, not an XML tag.

## Why

[[gemma-4-e2b]] was post-trained on its native tool-use grammar. When asked to emit XML, it diverges from its training:

- Only calls `is_legal`; ignores `eq_outcome_distribution`, `void_audit`, `trump_declared`, `unseen`, `conditional_outcome`
- Hallucinates a fake `play` tool in 80% of trials
- Expresses commits as `<|tool_call>` envelopes even when XML is expected

When allowed to use native format (`tools=[...]` via chat template), the full tool surface activates: `eq_outcome_distribution` called 15×, `trump_declared` called 9×, zero hallucinations. Bot-match moves from 60% to 88.9% on the same 10 decisions.

See [[burl-move4-native-spike]] for the full comparison.

## Harness changes

- `burl/modal/gemma_serve_native.py` — `skip_special_tokens=False` so `<|tool_call>` reaches the parser
- `burl/harness/agent_runner_native.py` — passes `tools=[...]` via chat template, includes `commit_play` schema
- `burl/harness/tool_loop_native.py` — `commit_play` sieve, native `<|tool_call>` parser

XML path (`gemma_serve.py`, `agent_runner.py`, `tool_loop.py`) is untouched — strictly additive.

## Retry budget

Native format emits exactly one `<|tool_call>` per turn. XML could pack multiple `<tool>` tags per completion. `max_retries=3` (budgeting 4 total iterations) is insufficient for native. Move 4 production default: **7**.

## Generalizable principle

The harness should bend to the model's post-training. Forcing a model to emit a synthetic format it wasn't trained for wastes its strongest capabilities. Find the model's natural grammar and meet it there.

OVERVIEW principle (added at [[3781dce]]): *"Go with the model's grain; catch it doing right. Small models have their own instincts... STaR trains the model's own best behavior back into itself — their words, their corrections, their self-checks."*

## Related pages

[[burl]] · [[gemma-4-e2b]] · [[tool-orchestration]] · [[burl-move4-native-spike]] · [[burl-move3-base]] · [[3781dce]]

## Status

Dormant since move-4 (mid-April 2026), superseded along with the rest of [[burl]] by
[[champion]] / [[jud]]'s pure-NN direction — no native tool-calling harness is used
there.
