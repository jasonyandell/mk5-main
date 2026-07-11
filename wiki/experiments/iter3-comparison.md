---
title: "iter-3: Three-Mode Enable_Primer Comparison"
kind: experiment
first_seen: dbadb5f
last_updated: dbadb5f
status: active
---

## Summary

Three STaR adapters trained with the same corpus infrastructure but different prompt shapes via the three-mode `enable_primer` flag matrix. iter-3-rules (rules-as-tools, primer off) is the clear winner: 90% bot-match, 0 retry-exhausted, 100% first-legal.

([SPIKE_REPORT.md @ dbadb5f](../sources/dbadb5f.md))

## Variants

| Variant | Prompt shape | Adapter |
|---|---|---|
| iter-2 | trimmed primer + 42-framing | [[burl-iter1-adapter]] successor |
| iter-3-v2 | no primer, 42-framing only (spike v2 shape) | separate adapter, 18-row corpus (no wiki page) |
| iter-3-rules | rules-as-tools preamble + 42-framing | [[iter3-rules-adapter]] (winner) |

## Key results

**iter-3-v2 (no primer):** 32% retry-exhausted. The "primer-load-bearing" finding from [[experiments/burl-iter1-mixed]] confirmed at scale — dropping the primer without a replacement structural scaffold causes commit discipline to collapse, even after SFT.

**iter-3-rules (rules-as-tools):** 90% bot-match, 0 retry-exhausted, 100% first-legal. **Winner.** `trick_winner_if` usage went UP after SFT — the adapter learned to reach for on-demand rule answers rather than relying on prose memorization. This validates the core [[burl]] premise: tools replace memorization.

## Significance

iter-3-rules is the Pareto-dominant adapter at this frontier: better commit discipline than iter-1, better bot-match than any prior adapter, and tool-use breadth that increases rather than decreases post-SFT. See [[decisions/primer-tradeoff]] — the resolution is to replace the primer entirely with engine-authoritative tools rather than trimming or keeping it.

## Related pages

[[iter3-rules-adapter]] · [[rules-as-tools]] · [[burl]] · [[decisions/primer-tradeoff]] · [[decisions/commit-discipline]] · [[experiments/burl-iter1-mixed]] · [[sources/dbadb5f]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; 2 corrections applied in place and independently re-verified.

- "0 retry-exhausted" is the N=10 eval headline; at rollout level iter-3-rules had 1/50 (2%) retry-exhausted (SPIKE_REPORT.md Phase 7 table).
- iter-4-thoughts was byte-identical to iter-3-rules on N=10, bounding the value of the thoughts channel at this scale — see [[iter4-null-preserve-thoughts]].
