---
title: The Burl line — tool-using LLM player, spike to dormancy
kind: trail
first_seen: 2026-07-13
last_updated: 2026-07-13
status: complete
---

This trail walks the whole [[burl]] arc: three weeks of hard running (2026-04-18 →
2026-05-07), then dormancy. It is the reading order for the family's experiments,
instruments, and findings. The LEM→Burl pivot itself is a separate trail,
[[lem-to-burl-handoff]].

## 1. Origins (2026-04-18)

Burl is a sibling to [[lem]], not a successor: where LEM taught [[texas-42]] in model
weights and plateaued at 55% bot-match on open play, Burl bets that a small model is
better at asking questions than memorizing facts. The [[engine]] answers rules and
visible state; a belief source answers hidden state; Burl reasons between tool calls
and commits a play. The philosophy, tool surface, and LEM contrast live on
[[tool-orchestration]].

Naming footnote (Names doctrine): the pre-repo, conversation-only names for this
conceptual space (first used Christmas 2025) were "harl" (a GRPO-trained
strategy/goal engine) and "llem" (a narrator model explaining harl's choices).
Neither ever entered git, disk, or beads. `burl` and [[lem]] are the repo's actual
christened successors; no rename or redirect is needed.

## 2. Moves 1–4 — finding the model's grain (commits d9baf3b–3781dce)

Move 1 shipped the tool surface and a ReAct harness (`tool_loop.py`, `retry.py`,
`trace.py`: think/act/observe with an XML parser, illegal-retry, stable JSON trace
schema) ([[d9baf3b]]). Its key pivot: [[zeb]] parked after
[[zeb-calibration-eval]] (hidden-only accuracy ~39%, not the advertised 72%); the
E[Q] N=10 outcome PDF replaced it as belief primitive
([[zeb-parked-eq-primitive]]).

- [[burl-move3-base]] — base Gemma on the XML harness: 100% legal, 60% bot-match,
  but only `is_legal` ever called; the premise survives, the format is wrong.
- [[burl-move4-native-spike]] — switch to Gemma's native `<|tool_call>` format:
  88.9% bot-match (+28.9pp), real tool breadth, zero hallucinated tools. Codified
  as [[native-tool-use-format]] and the principle "go with the model's grain; catch
  it doing right."

## 3. The primer arc — iter-0 and iter-1 (b8116b5–09b841e)

- [[burl-phase1-primer]] — the 1549-word rules primer buys 42 vocabulary
  (0 → 5–11 mentions/trace) at the cost of 18.9pp bot-match and distribution-tool
  suppression. The trade is deliberate: STaR needs the vocabulary
  ([[primer-tradeoff]]).
- [[burl-phase2-starcorpus]] — first STaR corpus, N=50, 54% K1; 23/23 hinted
  rationalizations converge first pass — Gemma as formatter, not second-chance
  reasoner.
- [[burl-iter0-eval]] — [[burl-iter0-adapter]] regresses to 60%: the corpus taught
  the adapter "Layer-1 Gemma," pathologies included.
- [[burl-iter1-mixed]] — trimmed primer: 80% bot-match on the 5 completions but
  5/10 retry-exhausted; the primer was a load-bearing commit-discipline scaffold
  ([[commit-discipline]]). [[burl-iter1-adapter]] is the artifact.

## 4. iter-2/iter-3 infrastructure (f164796–c2aa3a7)

Four workstreams landed between adapters; their receipts live here.

- **EQ-gate STaR rejection sampling** ([[f164796]], 761587c): replaces
  "reveal answer, rationalize" with a non-leaking nudge gate. Outcomes classified
  `converged_first_try` / `self_corrected` / `forced_flip` / `stubborn` /
  `exhausted`; only `self_corrected` traces enter the SFT corpus.
- **LS-Mixture verbosity blender** ([[3414507]]): shortens thought blocks
  at `target_short_ratio=0.33` (preview corpus 118 rows = 79 long + 39 short; short
  mean 767 chars vs long 5078). The training launcher ([[eebcae5]]) then
  discovered Gemma's chat template strips `<|channel>thought` blocks before
  tokenization — so the blend's real benefit was coverage and regularization, not
  thought-content training. First tremor of what became [[preserve-thoughts]].
- **Rules-as-tools scaffold** — four engine-authoritative tools replace the primer
  with a ~645-byte preamble. See [[rules-as-tools]].
- **[[haiku-4-5]] reference traces** — Anthropic's Haiku 4.5 on the same tool
  surface: 72.4% bot-match, ~7 tools/decision vs Gemma's ~3; reference ceiling for
  distillation, not training data.

iter-3 prep codified the prompt-shape space as a three-mode `enable_primer` matrix
(trimmed primer / rules-as-tools / no primer), with
`enable_primer=False ∧ enable_rules_tools=True` rejected as incoherent —
rules-as-tools IS a primer ([[c698091]]). Async semaphore-gated concurrency
landed a 3.92× local rollout speedup at N=8, concurrency=4 (c2aa3a7).

## 5. iter-3 winner, iter-4 null, the arena (35c75ff–dbadb5f)

- [[iter3-comparison]] — [[iter3-rules-adapter]] wins: 90% bot-match, 0
  retry-exhausted; rules-tool usage *rose* after SFT. The family's headline number
  (measured under the §7 confound, never re-run).
- [[iter4-null-preserve-thoughts]] — preserve-thoughts A/B comes back
  byte-identical; later explained as `max_seq_length` truncation, superseded by
  [[iter5-e1-rank-sweep]] and [[burl-star-run3]].
- [[burl-selfplay-arena]] — 4-Claude full-game orchestrator;
  [[opus-vs-haiku-arena]] — Opus salvages 7–35 where Haiku is shut out 0–42, with
  sharply better tool economy (`trump_declared` 1× vs 24×). A fix worth keeping:
  Opus 4.7 emits parallel `tool_use` blocks that raced the shared MCP tool cache;
  an `asyncio.Lock` around `_handle_sdk_mcp_request` unwedged it
  ([[2830be0]]).
- [[conditional-outcome-structural-nonuse]] — 0 calls across 145+ decisions on
  every model tested; later trivially explained by the §7 confound.

## 6. Local MLX, truncation, iter-5, candlewax (6fea6ab–aeafe22)

The pipeline moved onto the M5 Max via [[mlx-lm]], which surfaced TRL's silent
`max_seq_length=1024` truncation ([[sft-max-seq-length]]) — no prior adapter had
trained on complete thought-to-tool-call traces.

- [[iter5-e1-rank-sweep]] — rank-16 is the sweet spot (70.0%, first real
  preserve-thoughts signal); rank-64/128 collapse catastrophically on a 26-row
  corpus.
- [[iter5-e2-candlewax-null]] — bimodality made legible at the tool surface
  ([[candlewax]] fields) changes nothing in behavior; the blocker is policy, not
  legibility.
- [[batch-throughput-bench]] — `mlx_lm.batch_generate` hits 1334 tok/s at
  batch=128; operationalized at 2.3× wall on N=16.
- [[candlewax-spike]] / [[candlewax-spike-e2e]] — multimodal Qwen-VL end-to-end
  spike; v7 adapter +15% bot-match, but STaR plateaus without a verifier.
  [[reasoning-coherence-verification]] is named the bottleneck and the line pivots
  away from LLM-as-reasoner.

The era closed with a PRACTICALITIES.md split logging eight portable lessons
(native tool-call format, dual-use primer, model-invented idioms,
`conditional_outcome` zero-shot invisibility, `max_seq_length` truncation, rank-16
sweet spot, the 43→1334 tok/s ceiling, M5-Max-as-multiplier) ([[aeafe22]]).

## 7. The confound (54f7776)

Gemma 4's chat template silently drops `role="tool"` messages: **every rollout from
Move 4 through iter-5 ran with tool outputs invisible to the model**
([[gemma-tool-response-shape]]). [[chat-template-fix-validation]] — base Gemma goes
5/5 with the fix applied; every prior adapter number needs re-reading, and the
iter-3-rules 90% was never re-measured. The fix shipped inside [[wax-museum]], the
hard-gated HATEOAS harness that also brought [[belief-trajectory]] — [[gus]]'s
belief head as Burl's production belief source. Meta-lesson: audit the rendered
prompt, not the messages dict.

## 8. The harvest era and the STaR runs (063fcac–2026-04-26)

- [[burl-2000-harvest]] — 2000 decisions on `D_required_first`; the v1 run's
  1024-token truncation contaminated 11.4% of decisions and moved 43% between
  buckets while aggregate parity looked fine ([[max-tokens-2048-floor]],
  [[batched-harvest-resilience]]). v2: 1062-row strict pool, zero illegal commits.
- [[burl-star-run3]] — the north-star preserve-thoughts result: run-3b (flag off)
  emits thoughts on 0% of decisions; run-3c (flag on) on 95.9%, and beats base by
  −39% oracle regret on the paired n=180 ([[regret-eval]]). Run-3 proper died three
  times without writing an adapter — the failure that produced
  [[resumable-checkpointing]].
- [[burl-harvest-2]] — the self-sharpening test: run-3c-in-the-loop harvest + run-4
  filter-only SFT lands back at base play quality while repairing
  [[commit-discipline-collapse]]. Filter-only STaR does not compound at this scale;
  [[r1-rationalization]] on `BURL_BREAKS_CONSENSUS` was the recommended next loss
  target and never ran.

## 9. The perf sprint (2026-04-26 → 2026-04-28)

[[perf-on-the-table]] is the campaign rollup: calibration (~20× under raw
inference), six levers, and their outcomes.

- [[burl-perf-phase0]] — the frozen 5-decision bench and its noise floor.
- [[burl-perf-phase2]] — prefix-cache negative; the continuous-batching "1.8–2.1×"
  was later retracted to a statistical tie (contention artifact).
- [[burl-perf-phase3]] — speculative decoding ruled out three ways; Q4 PLE-safe
  quant confirmed at a 45–56% memory cut.

The repeatable procedure is [[perf-sprint]] (with [[perf-sprint-history]]); the
sprint stalled after sprint 2 and never resumed.

## 10. The instruments (2026-04-30 → 2026-05-07)

- [[burl-chat]] — talk-to-Burl workbench; [[burl-chat-spike]] produced the
  findings: [[chat-mode-primer]] is load-bearing, [[play-adapter-lock-in]] is total,
  and the improvised-tool loop works.
- [[improvised-tools]] — hot-register tool registry; [[burl-tool-wishlist]] — Burl's
  meta-asks map correctly onto its real failure modes.
- [[burl-reflection-deafness]] — "why was that wrong?" gets re-routed into the tool
  ritual; the [[post-commit-q-and-a]] corpus must include reflection turns
  explicitly.
- [[burl-lab]] — event-sourced experimentation platform (logged-arrow phases,
  rendered protocol text, HATEOAS tool advertisement); never reached parity with
  burl-chat before dormancy.
- [[burl-microscope]] — one-case, one-recipe workbench; hosted Burl's own tool
  requests (`simulate_hand_impact`, `calculate_expected_utility`).

## 11. The sampler audit (2026-07-11)

[[world-sampler-mrv-audit]] revisited the long-untracked "~6.8 Q sampler bias" note
from [[batch-throughput-bench]]: the original comparison was confounded, but
`WorldSamplerMRV` really could emit invalid worlds (exact probability 1/3 on the
audit fixture); the surviving `uniform-completion-dp-v1` replacement repaired it.
Every historical Burl eval number predates the repair.

## 12. Dormancy — why it paused, where the value went

Zero commits have touched `burl/` since 2026-05-07 (`465d1af`). The frontier moved
on 2026-06-09 to [[jud]] — pure-NN bid/play nets — rather than
answering the line's open questions:

- Does tool-mediated reasoning transfer to tool-less reasoning (ablation)?
- Does `conditional_outcome` leak evaluative signal through the back door?
- Is 3M-parameter belief capacity enough, or does Burl need more?
- Does the iter-3-rules 90% survive a re-run with visible tool responses?
- Does harvest-v2's residual −2.4pp on `BURL_BREAKS_CONSENSUS` (vs sequential 560)
  reflect batched-mode policy drift or noise? (A paired McNemar on the 560 overlap
  would settle it.)

Where the value went: the Zeb-negative was independently reconfirmed by
[[w42-jud-v1|jud v1]] (E[Q] n=10 "champion against every learned challenger since
Zeb"); the [[candlewax]] consumption question — sharpened by this line's verifier
wall — became the framing for [[jud]]; the [[preserve-thoughts]]
and recipe lessons joined the [[star]] canon; the harvests, adapters, and
instruments remain on disk under `scratch/` and `burl/`.

One ideated (never built) revival slot: [[book-strategy-player]]'s Model A — a
strategy-selector over ~15–30 named book strategies ("singleton_lead_to_void",
"trump_pulling") instead of raw dominoes. Small structured action space, natural
narrative rationales, interpretable failures, clean STaR ground truth from recorded
(state, applicable_strategies, chosen, outcome) rows. Speculative until the
book-strategy framework lands.
