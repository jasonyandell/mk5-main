---
title: Haiku 4.5 (Burl reference-trace generator)
kind: entity
first_seen: 2026-04-19
last_updated: 2026-07-13
status: complete
---

## What it is

Haiku 4.5 is Anthropic's small model used in [[burl]]'s iter-2 prep as a reference-trace
generator. It is not Burl's base model; it is a teacher signal — it produces target
[[topics/tool-orchestration]] traces that Gemma 4 then distills from. (commit messages @
1f13f92, b5d05de)

## Role in Burl

Haiku 4.5 runs against the same tool surface as Burl ([[engine]] wrappers, `eq_distribution`,
`commit_play`) via the Anthropic Agent SDK with in-process MCP tool handlers. Its traces
serve as a reference ceiling for iter-3+ distillation experiments — the "what Gemma should
aim for" anchor.

## N=30 reference run results (b5d05de)

- **29/30 completed** (one 10-turn timeout)
- **72.4% bot-match** overall
- **Cost**: $0.78 (35% under $1.20 cap)
- **Tools per decision**: ~7 distinct tools (vs Gemma iter-1's ~3)
- **Zero retries**: Haiku self-checks `is_legal` before `commit_play`

(commit message @ b5d05de)

## Key findings from the reference run

1. **`conditional_outcome` usage: zero.** Across all 30 decisions, Haiku never called it
   zero-shot. If Burl is to use this tool, STaR must synthesize explicit demos — it will not
   emerge spontaneously.
2. **The 4 big-gap misses share a pattern**: Haiku skipped `eq_outcome_distribution` and
   relied on prose-only reasoning. A prompt nudge ("probe both candidates with E[Q] on close
   calls") would likely recover 2-4 of these.
3. **Tool-use breadth is the distillation target**: 7 tools/decision vs Gemma iter-1's 3,
   zero retries. Haiku's "leads with prose, batches parallel engine-fact probes, calls
   `eq_outcome_distribution` selectively on top 1-2 candidates, self-checks before commit"
   is the pattern worth imprinting.

(commit message @ b5d05de)

## SDK pitfall (documented at 1f13f92)

The Anthropic Agent SDK falls back to non-streaming `--print` mode when `prompt` is a
plain string, which silently breaks in-process MCP RPC. Fix: pass prompt as an async
iterable of stream-json user messages.

## Status

The reference-trace role concluded with the N=30 run above; no distillation from
these traces ever ran before the [[burl]] line went dormant (2026-05-07). See
[[burl-line]].
