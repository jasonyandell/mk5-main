---
title: Burl — Tool-using Texas 42 agent
kind: entity
first_seen: 2026-04-18
last_updated: 2026-07-13
status: superseded
---

## What it is

Burl is an agentic tool-using [[texas-42]] player on [[gemma-4-e2b]] — a sibling
project to [[lem]], not a successor. The name is a common Texas name from the 1930s;
it is not an acronym. (burl/OVERVIEW.md @ 8d26e0d)

The line ran hard for ~3 weeks (2026-04-18 → 2026-05-07: harness spikes, primer
ablations, iter-0 through iter-5 STaR, a 2000-decision harvest, a perf sprint, and
three interactive instruments), then went dormant. Zero commits have touched `burl/`
since 2026-05-07 (`465d1af`). [[jud]] superseded the line with a
pure-NN architecture around 2026-06-09.

**Read [[burl-line]] for the full arc in order.** The LEM→Burl pivot is
[[lem-to-burl-handoff]].

## The bet

LEM taught the game in weights (86% comprehension on [[qwen3-1.7b]], 55% bot-match
plateau on open play). Burl inverts the shape:

- The [[engine]] is the authority on rules and visible state. Burl asks it.
- A belief source answers hidden state — [[zeb]] at first (parked at Move 1 after
  [[zeb-calibration-eval]]: hidden-only accuracy ~39%, not the advertised 72%;
  see [[zeb-parked-eq-primitive]]), then the E[Q] N=10 outcome PDF, finally
  [[belief-trajectory]] ([[gus]]'s belief head).
- Burl's job is **reasoning with what the tools return** and **committing a play**.

> "Small models are much better at asking the right questions and synthesizing tool
> responses than at memorizing many facts." (burl/OVERVIEW.md @ 8d26e0d)

The full tool surface, doctrine, and LEM contrast live on [[tool-orchestration]].
Two invariants: tools answer "what IS the state?", never "what SHOULD you do?"
(evaluative tools are forbidden so distillation can't short-circuit), and legality
is by construction via the engine retry loop, not a trained behavior. Eval targets:
legal rate, retry count, bot-match — later upgraded to oracle regret
([[regret-eval]]).

## Headline findings

- **Speak the model's grain.** Native `<|tool_call>` format over XML: 60% → 88.9%
  bot-match on base Gemma, tool breadth real, zero hallucinated tools
  ([[burl-move4-native-spike]], [[native-tool-use-format]]).
- **The primer trade.** Rules primer buys 42 vocabulary but suppresses distribution
  tools and bakes its pathology into adapters ([[burl-phase1-primer]],
  [[burl-iter0-eval]], [[primer-tradeoff]]); dropping it collapses commit
  discipline ([[burl-iter1-mixed]], [[commit-discipline]]).
- **Rules-as-tools won the prompt-shape matrix.** [[iter3-rules-adapter]] hit 90%
  bot-match with rules-tool usage *rising* after SFT ([[iter3-comparison]],
  [[rules-as-tools]]) — but see the confound below.
- **THE CONFOUND.** Gemma 4's chat template silently dropped `role="tool"`
  messages; every rollout from Move 4 through iter-5 ran with tool outputs
  invisible to the model ([[gemma-tool-response-shape]]). Post-fix, base Gemma
  scored 5/5 ([[chat-template-fix-validation]]). The 90% was never re-measured;
  [[conditional-outcome-structural-nonuse]] (0 calls / 145+ decisions) is trivially
  explained by it.
- **Preserve-thoughts is a phase change, not a knob.** Without `--preserve-thoughts`
  a Burl LoRA trains reasoning *out* of the model (0% thought emission); with it,
  95.9%, and −39% oracle regret vs base on the paired eval ([[burl-star-run3]],
  [[preserve-thoughts]], [[sft-max-seq-length]]).
- **Filter-only STaR plateaus.** The self-sharpening loop (harvest → SFT →
  re-harvest) lands back at base play quality at this scale; changing the loss
  target was recommended and never ran ([[burl-harvest-2]]).
- **The verifier wall.** Reasoning-coherence verification — not tool legibility —
  is the bottleneck for LLM-as-reasoner; the candlewax spikes pivoted the project
  off that path ([[iter5-e2-candlewax-null]], [[candlewax-spike]],
  [[reasoning-coherence-verification]], [[candlewax]]).
- **The lock-in family.** A play adapter cannot be talked out of `commit_play`
  ([[play-adapter-lock-in]]); reflection prompts get re-routed into the tool ritual
  ([[burl-reflection-deafness]]); Burl's own tool requests are nonetheless accurate
  diagnoses ([[burl-tool-wishlist]]).

## Adapters

Lineage page: [[burl-adapter-line]] (the chain's full story + comparative table).

| Adapter | Result | Page |
|---|---|---|
| [[burl-iter0-adapter]] | 60% — learned Layer-1's pathology | [[burl-iter0-eval]] |
| [[burl-iter1-adapter]] | 80% on 5 completions, 5/10 exhausted | [[burl-iter1-mixed]] |
| [[iter3-rules-adapter]] | 90% — headline, confounded, never re-run | [[iter3-comparison]] |
| e1-rank16 | 70% — first real preserve-thoughts adapter | [[iter5-e1-rank-sweep]] |
| run-3b / run-3c / run-4 | scratch-only; preserve-thoughts A/B + plateau | [[burl-star-run3]], [[burl-harvest-2]] |

## Instruments and infrastructure

| Thing | What it is |
|---|---|
| [[wax-museum]] | Hard-gated HATEOAS harness; Phase A guards; hosted the harvests |
| [[burl-2000-harvest]] / [[burl-harvest-2]] | The 2000-decision corpora (strict pool, bucket taxonomy) |
| [[burl-chat]] | Talk-to-Burl workbench ([[burl-chat-spike]], [[improvised-tools]]) |
| [[burl-lab]] | Event-sourced experimentation platform; never reached parity |
| [[burl-microscope]] | One-case, one-recipe prompt/tool workbench |
| [[burl-selfplay-arena]] | 4-Claude full-game orchestrator ([[opus-vs-haiku-arena]]) |
| [[haiku-4-5]] | Reference-trace generator (72.4% ceiling, 7 tools/decision) |
| [[batch-throughput-bench]] | MLX batching ceiling (1334 tok/s at batch=128) |
| [[perf-on-the-table]] | Perf-sprint rollup: levers, retraction, Q4 quant win |

Sampler caveat: [[world-sampler-mrv-audit]] (2026-07-11) falsified
`WorldSamplerMRV`'s validity guarantee (invalid worlds at probability 1/3 on the
audit fixture) and shipped a uniform replacement; every historical Burl eval number
predates the repair.

## Frontier status

Dormant. The open questions the line never answered — tool-less transfer,
`conditional_outcome` leakage, belief capacity, the 90% re-run — are enumerated in
[[burl-line]] §12, along with where the value went: the Zeb-negative reconfirmed by
[[w42-jud-v1|jud v1]], the [[candlewax]] consumption question feeding [[jud]],
the [[preserve-thoughts]] recipe lessons joining the [[star]] canon,
and an ideated [[book-strategy-player]] Model A revival slot.
