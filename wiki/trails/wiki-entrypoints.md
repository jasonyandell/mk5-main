---
title: Wiki Entrypoints
kind: trail
first_seen: local-2026-05-03
last_updated: bc4eb386
status: active
---

This trail is the lightweight route map for agents. It exists so the wiki can be
consulted first without loading the full [[index]] on every question.

## Frontier hubs

Start with one of these when the user names a major workstream:

| hub | use when |
|---|---|
| [[texas-42]] | the game, rules, state, and project-wide frame |
| [[engine]] | TypeScript rules, legality, state transitions, and UI/game behavior |
| [[forge]] | oracle, E[Q], solver, model-training substrate, and generated-game data |
| [[lem]] | rules comprehension, Stage 0 adapters, STaR, rationalization, Qwen/Gemma curriculum history |
| [[burl]] | tool-using play, wax_museum, burl-lab, chat, STaR traces, and post-commit Q&A |
| [[gus]] | oracle distillation, belief/value/policy heads, LAMIR, regret eval, strategy probes |
| [[w42]] | Winning 42 book validation, strategy detectors, claim ledger, report-shaped research |
| [[book-strategy-player]] | multi-step book-plan architecture, algebraic spec, strategy-selector data, and W42 planning-aware validation |
| [[jud]] | the unified belief-conditioned core direction; bidding+play as one act |
| [[champion]] | the jud v1 capstone build — zero-adapter, greedy 1-ply value play, oracle-free at runtime |
| [[partnership-wall-research]] | the post-Jud evidence map, measurement cleanup, partnership-value target, causal microgames, and architecture gates |

**The measured baseline since June 2026** is the jud v1 / Zeb-protocol
capstone (`afd4802`, `622c1a1`, `1143ebb`, `bdebb82`, `3ac03de`): start at
[[jud]] or [[champion]] for the graded policy. **The research frontier since
2026-07-11** is [[partnership-wall-research]]: start there when the question is
why E[Q] n=10 remains undefeated at pure play or what earns the next build.

## Current trails

- [[lem-to-burl-handoff]] — why the project pivoted from LEM's weight-memory
  curriculum to Burl's tool-orchestration bet.
- [[w42-book-validation]] — how the Winning 42 book harvest, chapter pages,
  claim ledger, phase sweeps, utility-lens work, Lens v1, and
  planning-aware frontier fit together.
- [[partnership-wall-research]] — how Forge, E[Q], Gus, Burl, W42, Champion,
  Jud, and the book constrain the smallest next experiments without selecting
  CFR, a larger network, an LLM, a symbolic library, or Jud v2 in advance.

## Query shortcuts

Use `rg` when a page name, experiment slug, bead id, or exact phrase is already
known. Load [[index]] only when the question is broad or the likely page name is
unknown.

Useful patterns:

- `rg -n "phrase" wiki`
- `rg -n "bead-id|experiment-slug" wiki`
- `rg --files wiki | rg "slug-fragment"`
- `grep "^## \\[" wiki/log.md | tail`

## Large leaf clusters

The largest clusters are intentionally not all headline material. Prefer a hub
or trail first, then descend into leaves only as needed:

- [[w42]], [[w42-book-validation]], and [[book-strategy-player]] route the W42 /
  Winning 42 leaf pile.
- [[burl]] routes Burl experiments, wax_museum, burl-lab, and perf work.
- [[gus]] routes Gus model/eval/interpretability pages.
- [[lem]] and [[lem-to-burl-handoff]] route LEM's Stage 0 / Stage 1 history.

## Maintenance rule

When a query required several leaf pages to answer, file the route back into a
trail or hub. A useful answer that vanishes into chat is a missed wiki update.
