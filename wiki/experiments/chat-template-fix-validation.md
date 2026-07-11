---
title: "Chat-Template Fix Validation (Burl N=5)"
kind: experiment
first_seen: 54f7776
last_updated: 54f7776
status: superseded
---

## Summary

5 held-out trick-6 decisions run with the tool-response shape fix applied ([[decisions/gemma-tool-response-shape]]). Base [[gemma-4-e2b]] scores 5/5 bot-match with tool responses now visible. Demonstrates what the model can do when the harness actually works.

([burl/wax_museum/ @ 54f7776](../sources/54f7776.md))

## Setup

- **Fix applied:** tool responses packed as `assistant.tool_responses=[{name, response}]` instead of `role="tool"` messages
- **Models tested:** base [[gemma-4-e2b]] (no adapter); Qwen 3.6-35B-A3B-4bit
- **Eval set:** N=5 held-out trick-6 decisions

## Results

| Model | Bot-match | Notes |
|---|---|---|
| Base Gemma 4 E2B | **5/5** | Faithful numeric quoting; pivot quoted verbatim from tool response |
| Qwen 3.6-35B-A3B-4bit | — | 6K-token `<think>` blocks deriving 42 rules from scratch; different behavior |

## Significance

5/5 bot-match from base Gemma with working tools is the clearest evidence that the [[burl]] premise holds — a 2B model can reason effectively about Texas 42 when it can actually see tool outputs. Every prior measurement ([[experiments/burl-move3-base]], [[experiments/burl-move4-native-spike]], all STaR iterations) was confounded by invisible tool responses.

The [[iter3-rules-adapter]]'s 90% result remains impressive but now needs reframing: it was achieving near-90% under zero tool-input conditions, not under full tool orchestration. What LoRA learned under the confound is unclear — pattern-match on prompt shape, or genuine tool-orchestration reasoning?

## Open question at this frontier — never answered

Does iter-3-rules adapter behavior change under the fix? Re-running with working tool responses would decompose "LoRA learned tool orchestration" from "LoRA learned to pattern-match without tool input."

**This question was never resolved.** No re-run appears anywhere in the [[burl]]
family (checked across all evidence chunks). [[iter3-rules-adapter]]
(`last_updated: dbadb5f`, predating this page's `54f7776` fix) was never revised to
carry this caveat until the era-6 audit (2026-07-06) added it directly. The project's
single headline validated result — 90% bot-match — was measured with the model's
tool outputs silently invisible to it, and the project never re-ran it after fixing
that bug 17 commits (one day) later.

## Related pages

[[decisions/gemma-tool-response-shape]] · [[wax-museum]] · [[burl]] · [[gemma-4-e2b]] · [[iter3-rules-adapter]] · [[experiments/burl-move4-native-spike]] · [[sources/54f7776]]
