# Log — archive

Rolled-out entries from `log.md` (2026-04-09 → 2026-07-10, entries 1–156). Same format, still greppable: `grep "^## \[" log-archive.md`. Newest entries live in [[log]].

---


## [2026-04-09 | a8bccfa | narration generator, rules primer, first Gemma contact]

First meaningful commit in `lem/`. Establishes the project's intent, scaffolding, and first evidence that a phone-class model can say anything coherent about Texas 42.

**Touched pages:** [[entities/lem]] [[entities/gemma-4-e2b]] [[entities/texas-42]] [[entities/forge]] [[topics/star]] [[topics/backwards-curriculum]] [[topics/rules-adapter]] [[topics/narration]] [[topics/k1-grading]] [[topics/r1-rationalization]] [[topics/expected-q-value]] [[topics/lora-unsloth]] [[experiments/first-gemma-contact]] [[sources/a8bccfa]]

**Added:** 14 pages — 4 entities, 8 topics, 1 experiment, 1 source digest.

**Frontier established:**
- LEM's north star is E[Q] delta vs. the bot it replaced, rising over training iterations.
- Two-stage plan: Stage 0 rules adapter → Stage 1 STaR on trick-6 decisions, with a backwards curriculum extending to earlier tricks.
- Base model: Gemma 4 E2B, Apache 2.0, 2.3B effective params, native thinking channel.
- First-contact result: identified trump, led suit, void status, and score math correctly; confused initial hand with remaining hand; misclassified 6-4 as trump under fives-trump. The errors motivate Stage 0 precisely.
- Rules primer commits to 55 testable facts — self-verifying ground truth for the Stage 0 Q&A corpus.

**Questions opened:**
- Does the rules primer stay in the system prompt forever or get distilled into weights?
- Are R1 rationalizations enough, or will later stages need DPO-style preference learning?
- What are the right LoRA hyperparameters for Gemma 4 E2B?
- What are the variance filter thresholds (`δ`, `σ_max`) for Stage 1 filtering?

---

## [2026-04-10 | 24ae55a | Stage 0 built, trained, and the v1 lessons captured (6bb8a40..24ae55a)]

Four commits completing Stage 0: infra, PEFT fix, training run, second contact, and OVERVIEW update.

**Touched pages:** [[entities/lem]] [[entities/gemma-4-e2b]] [[entities/modal]] [[entities/stage-0-adapter]] [[topics/rules-adapter]] [[topics/lora-unsloth]] [[topics/learned-by-playing]] [[experiments/stage-0-v1-training]] [[experiments/second-gemma-contact]] [[sources/6bb8a40]] [[sources/9571a7b]] [[sources/df73c8d]] [[sources/24ae55a]]

**Added:** 9 pages — 2 entities, 1 topic, 2 experiments, 4 source digests.

**Updated:**
- [[entities/lem]] — Progress section added, Stage 1 plan recorded.
- [[entities/gemma-4-e2b]] — Gemma4ClippableLinear PEFT incompatibility documented; bf16 requirement and L4 VRAM budget noted.
- [[topics/rules-adapter]] — Q&A corpus made concrete (3500 ex × 7 cats); primer 55/55 verified; "what transferred / what didn't" section added.
- [[topics/lora-unsloth]] — Full reproducible Stage 0 recipe added.

**Frontier shift:**
- Stage 0 v1 is complete end-to-end on Modal L4; adapter published to HuggingFace as `jasonyandell/gemma-4-e2b-texas42-stage0`.
- Hand-tracking transfers from Q&A to narration; trump membership does not — [[topics/learned-by-playing]] coined to name this gap.
- Training recipe now considered solved: ClippableLinear → nn.Linear patch + bf16 + gradient checkpointing + eval disabled + 1 epoch.

**Questions opened:** none.

---

## [2026-05-04 | local | burl-lab LM Studio SDK agent lane]

[[burl-lab]] gained a first integration slice for LM Studio's Python SDK
agent loop. The lab owns the replayable session journal, callbacks, and
tool execution; LM Studio provides local inference through `lmstudio-python`.

**Touched pages:** [[burl-lab]] [[index]] [[log]]

**Updated:**
- [[burl-lab]] now documents the LM Studio SDK lane: composed rendered system
  prompt plus seeded `board_snapshot()` user prompt into `model.act(...)`.
- The Move table records `LmStudioChatRequest`, `LmStudioChatResponse`, and
  `LmStudioChatError` as journaled wrapper events around SDK `.act()` runs,
  including prediction fragments and tool-call summaries.
- [[index]] now routes readers to burl-lab's LM Studio SDK agent lane
  alongside event-sourced sessions, rendered ToolSpec prompts, and HATEOAS
  tool advertisement.

**Frontier shift:** LM Studio integration starts as a server-side SDK agent
surface, not an MCP façade and not a raw REST stateful-chat wrapper. That
preserves the clean responsibility split: LM Studio supplies inference, while
burl-lab owns the experiment recipe, Python tool functions, callbacks, and
journaled tool-call lineage in `events.jsonl`.

**Questions opened:** none.

**Questions partially resolved:**
- LoRA hyperparameters for Stage 0: 1 epoch sufficient, bf16 required, rank/LR from Unsloth recipe. Stage 1 hyperparameters still open.

---

## [2026-04-10 | b99c64d | batch narration generator + A100 training + eval seeds]

Single commit. Small ingest: 5 page touches, first decision page added.

**Touched pages:** [[entities/lem]] [[entities/modal]] [[topics/narration]] [[decisions/eval-seed-holdout]] [[sources/b99c64d]]

**Added:** 2 pages — 1 decision, 1 source digest.

**Updated:**
- [[entities/lem]] — Stage 1 infrastructure subsection: batch generator stats, eval seed holdout, A100 switch.
- [[entities/modal]] — GPU selection section: L4 vs A100 rationale added.
- [[topics/narration]] — Batch mode section (200 seeds → 3148 examples in 8.5 min); Datasets produced section.

**Frontier shift:**
- Batch narration pipeline is operational; first concrete training-dataset scale numbers exist (3148 examples / 200 seeds).
- Eval seed holdout (900000–909999) is now a permanent rule — first formal decision page in the wiki.
- Stage 1 training moves from L4 to A100.

**Questions opened:** none.

## [2026-04-10 | 6e71df9 | STaR harness built; first smoke test passes; base-model K1 baseline measured (7538016..6e71df9)]

Four commits. STaR moves from plan to implementation; first K1 numbers land.

**Touched pages:** [[entities/lem]] [[entities/gemma-4-e2b]] [[entities/star-harness]] [[topics/star]] [[topics/k1-grading]] [[topics/r1-rationalization]] [[experiments/star-harness-5ex-smoke]] [[experiments/base-model-k1-baseline]] [[sources/7538016]] [[sources/f578bfa]] [[sources/8c5fbca]] [[sources/6e71df9]]

**Added:** 7 pages — 1 entity, 2 experiments, 4 source digests.

**Updated:**
- [[entities/lem]] — Stage 1 kickoff recorded; baseline finding noted.
- [[entities/gemma-4-e2b]] — K1 baseline (60% on 10 trick-6 decisions) added.
- [[topics/star]] — Implementation section added: 6-step flow, three runners, first measurements.
- [[topics/k1-grading]] — Equivalence clarified; first pass-rate measurements added.
- [[topics/r1-rationalization]] — Implementation note: R1 invoked on fail or illegal; vLLM batch mode.

**Frontier shift:**
- STaR is now implementation, not plan. Three runners: Modal harness, local llama.cpp debug runner, single-GPU vLLM loop (~4 min/iter on B200, ~8 min on H100).
- Base Gemma (no adapter) passes K1 at 60% on 10 trick-6 decisions. K1 ≡ argmax-tied action; many trick-6 decisions have obvious answers, so improvement headroom is smaller than Q&A gaps suggested.
- Smoke test (5 examples): 1 pass, 2 rationalized, 2 illegal-rationalized. 40% illegal rate matches hand-tracking errors from second-gemma-contact.

**Questions opened:**
- What is the actual K1 ceiling for trick-6 decisions? If many are near-unanimous-argmax, the ceiling may be well below 100%.

**Questions resolved:** none.

---

## [2026-04-10 | fb47ab3 | discard illegal traces instead of rationalizing them]

Single commit. Policy shift: R1 scope narrowed, illegal_rate promoted to first-class metric.

**Touched pages:** [[entities/star-harness]] [[topics/r1-rationalization]] [[topics/star]] [[decisions/discard-illegal-traces]] [[sources/fb47ab3]]

**Added:** 2 pages — 1 decision, 1 source digest.

**Updated:**
- [[entities/star-harness]] — inlined grading fns (Modal mount fix); three-branch grading table; illegal_rate wandb metric added.
- [[topics/r1-rationalization]] — scope narrowed to legal-fails only; supersedes ingest 4 claim that R1 runs on illegals.
- [[topics/star]] — 7-step flow (was 6); explicit discard step added between grade and rationalize.

**Frontier shift:**
- R1 no longer runs on illegal or parse-fail traces. Those are discarded: "reasoning about impossible states is poison."
- Illegality rate is now a first-class wandb metric and a free diagnostic for rules comprehension (high = more rules work needed; low = strategy focus appropriate).
- Insight credited to ClaudeAI in the source commit.

**Questions opened:** none.
**Questions resolved:** none.

---

## [2026-04-10 | 576b694 | vLLM abandoned for HF batch generate; Stage 1 iteration 0 runs end-to-end (68a0416..576b694)]

Four commits. vLLM removed; first complete STaR iteration finishes on H100.

**Touched pages:** [[entities/lem]] [[entities/star-harness]] [[topics/star]] [[experiments/star-iter-0]] [[sources/68a0416]] [[sources/8724e93]] [[sources/d913932]] [[sources/576b694]]

**Added:** 5 pages — 1 experiment, 4 source digests.

**Updated:**
- [[entities/lem]] — Stage 1 iter-0 subsection; compute setup; vLLM abandonment lesson recorded.
- [[entities/star-harness]] — star_loop.py rewritten: vLLM out, HF generate in; ClippableLinear patch made unconditional.
- [[topics/star]] — Stage 1 iteration 0 section added; vLLM-out note in Implementation.

**Frontier shift:**
- vLLM abandoned in favour of HF `model.generate()` batch inference due to Gemma 4 transformers incompatibility. May reverse on future hardware.
- First full STaR iteration complete end-to-end: 30% K1 pass, 40% illegal, 15 min on H100, ~$1.
- Compute recipe solidified: H100 for full loop, A100 for training, CPU+llama.cpp for debug.

**Questions opened:**
- Will vLLM return once Gemma 4's transformers incompatibility is resolved, or is HF `model.generate()` the permanent simpler choice?

**Questions resolved:** none.

---

## [2026-04-10 | 26f5ddf | vLLM round 2 on B200 abandoned; SDPA + torch.compile + batching is the winning recipe (2c2b851..26f5ddf)]

Two commits. Second vLLM attempt, second revert. Performance recipe now settled.

**Touched pages:** [[entities/modal]] [[entities/gemma-4-e2b]] [[entities/star-harness]] [[topics/star]] [[sources/2c2b851]] [[sources/26f5ddf]]

**Added:** 2 pages — 2 source digests.

**Updated:**
- [[entities/modal]] — B200 added to GPU selection table.
- [[entities/gemma-4-e2b]] — vLLM 0.19.0 LoRA incompatibility with Gemma4ForConditionalGeneration's multimodal weight layout documented.
- [[entities/star-harness]] — Performance recipe section; second vLLM attempt and revert recorded.
- [[topics/star]] — Performance recipe note; second vLLM abandonment noted in Implementation.

**Frontier shift:**
- vLLM abandoned a second time. Root cause pinned: Gemma4ForConditionalGeneration's multimodal weight layout is incompatible with vLLM 0.19.0's LoRA path. Not a transient issue.
- Settled performance recipe: HF generate + SDPA + torch.compile(reduce-overhead) + left-padded batching → 120 tok/s on B200, 151s per 5-example iteration.
- Model loaded once per loop iteration, reused for both inference and R1 rationalization, stopped only for LoRA training.

**Questions opened:** none.

**Questions resolved:**
- "Will vLLM return once Gemma 4's transformers incompatibility is resolved?" — Resolved `26f5ddf`. The incompatibility is structural (multimodal weight layout); HF generate + SDPA is the committed stable path.

---

## [2026-04-11 | 5946c94 | scratchpad validation introduced; hit 64.5% invalid on iter-0; reverted to simple K1 (380f3fa..5946c94)]

Six commits in 14 minutes. First `retired` page in the wiki.

**Touched pages:** [[topics/scratchpad-validation]] [[topics/star]] [[topics/k1-grading]] [[entities/star-harness]] [[experiments/scratchpad-v2-iter0]] [[sources/380f3fa]] [[sources/c88582c]] [[sources/34775ca]] [[sources/b12fcec]] [[sources/78ba940]] [[sources/5946c94]]

**Added:** 8 pages — 1 topic (retired), 1 experiment (retired), 6 source digests.

**Updated:**
- [[topics/star]] — scratchpad attempt section added; revert note; random-subset sampling retained.
- [[topics/k1-grading]] — stricter grading attempted and shelved section added.
- [[entities/star-harness]] — scratchpad code retained but inactive; subset sampling retained.

**Frontier shift:**
- Scratchpad validation tried and reverted in 14 minutes. 64.5% invalid traces on 5-example run.
- Lesson: format-bootstrapping must precede fact-validation. A model cannot be asked to produce new structured output AND satisfy factual constraints on it in a single training step.
- Retained from the attempt: random-subset sampling per iteration for trace diversity.
- Simple K1 remains the active grading path; scratchpad code parked behind flags for later.

**Questions opened:**
- When will the model have learned the scratchpad format well enough to enable fact-validation? What mechanism will teach the format first?

**Questions resolved:** none.

---

## [2026-04-11 | efad16e | combined 7409-example dataset + 10 STaR iterations complete (30% → 36-42% plateau) (ff0d0d2..efad16e)]

Two commits. Stage 1 first-phase milestone reached.

**Touched pages:** [[entities/lem]] [[entities/gemma-4-e2b]] [[entities/star-harness]] [[topics/star]] [[topics/k1-grading]] [[topics/learned-by-playing]] [[experiments/star-10-iterations]] [[sources/ff0d0d2]] [[sources/efad16e]]

**Added:** 3 pages — 1 experiment, 2 source digests.

**Updated:**
- [[entities/lem]] — Stage 1 10-iter subsection; 10 adapters on HF; ~$15 total compute cost.
- [[entities/gemma-4-e2b]] — KV-sharing architecture quirk documented: layers 15-34 have no k/v projections; resolves "missing keys" LoRA warning.
- [[entities/star-harness]] — chained Modal runs documented; B200 cost breakdown (~$0.26/iter).
- [[topics/star]] — 10-iter results table; plateau characterization (36-42%); chaining note.
- [[topics/k1-grading]] — pass-rate progression across 10 iterations added.
- [[topics/learned-by-playing]] — "First evidence" section: 12-pt improvement over baseline, plateau established.

**Frontier shift:**
- 10 STaR iterations complete: K1 pass rate 30% → 42% peak, settling in 36-42% band. Stage 1 first phase plateaued.
- Dataset grew from 3148 → 7409 examples at iter 5, correlated with first 42% reading.
- B200 settled as compute platform: ~$0.26/iter, ~$15 for all 10.
- KV-sharing architecture: Gemma 4 E2B layers 15-34 have no k/v projections; the "LoRA missing keys" warning is expected and benign.
- Candidate next steps logged in OVERVIEW: more data diversity, scratchpad-validation once format is bootstrapped, or larger base model.

**Questions opened:**
- How much of the 30→42% gain is strategy vs rules internalization? (illegal_rate per-iter not tracked in this run.)
- Can the plateau be broken by (a) more data diversity, (b) scratchpad-validation once format is bootstrapped, or (c) a larger base model?

**Questions resolved:**
- "LoRA adapter missing keys for layers 15-34" — Resolved `efad16e`. KV-sharing architecture of Gemma 4 E2B: those layers have no k/v projections by design. Adapter is complete.

---

## [2026-04-11 | 908773a | 15 iterations complete, plateau at 38-42%, 11k narrations generated]

Single commit. Stage 1 first-phase plateau confirmed and named.

**Touched pages:** [[entities/lem]] [[topics/star]] [[topics/k1-grading]] [[topics/learned-by-playing]] [[experiments/star-10-iterations]] [[sources/908773a]]

**Added:** 1 page — 1 source digest.

**Updated:**
- [[entities/lem]] — Stage 1 extended to 15 iters; total cost $25; 15 adapters on HF; pool 11,672; plateau-ceiling hypothesis stated.
- [[topics/star]] — results section extended through iter 14; plateau confirmed; ceiling-hypothesis paragraph added.
- [[topics/k1-grading]] — Ceiling hypothesis section: ~40% is likely the K1-without-fact-verification ceiling.
- [[topics/learned-by-playing]] — Refined claim: learning-by-playing saturates without a fact-verification signal.
- [[experiments/star-10-iterations]] — Title updated to "STaR Stage 1: 15 Iterations"; 5 new table rows; plateau-confirmed section; open directions updated.

**Frontier shift:**
- Plateau at 38-41% confirmed across 15 iterations. No upward trend in the final 5.
- Ceiling hypothesis named: K1 grading without fact-verification is the limiting factor. The model may learn wrong game-facts that coincidentally produce correct argmax plays, making further K1 gains impossible without structural change.
- 15 adapters on HuggingFace; narration pool at 11,672 examples; total cost ~$25.

**Questions opened:** none (existing "plateau-breaking" question from ingest 9 is refined by this data).
**Questions resolved:** none.

---

## [2026-04-11 | 7f1994e | public state block after every trick — played dominoes, count status, hand]

Single commit. Narration gains v3 format; a design philosophy is codified as a decision.

**Touched pages:** [[entities/lem]] [[topics/narration]] [[decisions/public-state-block]] [[sources/7f1994e]]

**Added:** 2 pages — 1 decision, 1 source digest.

**Updated:**
- [[entities/lem]] — Narration v3 subsection added under Progress.
- [[topics/narration]] — Post-trick public state block section: dominoes played, count status, remaining hand; ~60 tok/trick, ~300 extra per prompt.

**Frontier shift:**
- Narration v3: a structured public-state block is appended after each trick. State visible at the real table belongs to the narrator, not the model.
- Design philosophy codified: model capacity is reserved for strategy, not bookkeeping. This is the 3rd decision page in the wiki.

**Questions opened:** none.

**Questions resolved:**
- "How much trick-state to restate each trick?" — Resolved `7f1994e` (implicit question from a8bccfa open-voice notes; never formally logged). Answer: full public-state block after every trick (~60 tok), covering dominoes played, count, and remaining hand.

---

## [2026-04-11 | 43009a4 | Stage 0 v2 Kerry curriculum + adapter eval (f8cdbe7..43009a4)]

Two commits. Stage 0 gets a v2; third-contact eval shows meaningful progress.

**Touched pages:** [[entities/lem]] [[entities/gemma-4-e2b]] [[entities/kerry-adapter]] [[topics/rules-adapter]] [[topics/kerry-curriculum]] [[experiments/third-gemma-contact]] [[sources/f8cdbe7]] [[sources/43009a4]]

**Added:** 5 pages — 2 entities, 1 topic, 1 experiment, 2 source digests.

**Updated:**
- [[entities/lem]] — Stage 0 v2 Kerry subsection added under Progress.
- [[entities/gemma-4-e2b]] — third-contact bullet added.
- [[topics/rules-adapter]] — Stage 0 v2 Kerry section; eval progression table added.

**Frontier shift:**
- Stage 0 v2 uses Kerry Newberry's Learner's Guide curriculum (15k examples, A/B/C/D categories). Trained in 150 steps on B200.
- Third-contact eval vs second: trump non-membership NOW CORRECT for 6-2/6-1. Trump membership error narrowed to single stubborn case (6-4 under fives). Strategic reasoning dramatically deeper.
- Kerry adapter is the active Stage 0 base; original stage-0-adapter is now a legacy checkpoint.

**Questions opened:**
- Why does 6-4 stay stubbornly misidentified as trump under fives? Hypothesis: model conflates 6-4's count-domino status (10 points, contains a 4) with trump membership.

**Questions resolved:** none.

---

## [2026-04-11 | 8c1bb14 | Stage 0 progression Kerry → v3: plateau was curriculum-bound, v3 STaR peaks 48% (a2498e4..8c1bb14)]

Three commits, ~90 min. Plateau reframed; new ceiling record set.

**Touched pages:** [[entities/lem]] [[entities/kerry-adapter]] [[entities/gemma-4-e2b]] [[entities/v3-adapter]] [[topics/rules-adapter]] [[topics/star]] [[topics/trump-drilling]] [[topics/learned-by-playing]] [[experiments/stage-0-progression-star]] [[sources/a2498e4]] [[sources/601f622]] [[sources/8c1bb14]]

**Added:** 6 pages — 1 entity, 1 topic, 1 experiment, 3 source digests.

**Updated:**
- [[entities/lem]] — Kerry + v3 progression; plateau reframed as curriculum-bound.
- [[entities/kerry-adapter]] — superseded-by-v3 note added.
- [[entities/gemma-4-e2b]] — v3-adapter note under third-contact.
- [[topics/rules-adapter]] — v3 section + progression table (v1/Kerry/v3).
- [[topics/star]] — Kerry STaR + v3 STaR section; revised ceiling framing.
- [[topics/learned-by-playing]] — Revised claim: drilling raises the floor; playing improves above it.

**Frontier shift:**
- Stage 0 progression confirmed: v1 avg/peak/illegal = 37/42/33; Kerry = 43/46/12; v3 = 44/48/13. Each curriculum round raises the floor.
- Ingest-10 K1-ceiling hypothesis weakened: Stage-0-quality is the dominant constraint, not K1 grading itself.
- Best adapter: v3 star-iter2 at 48% K1 pass rate.
- kerry-adapter superseded by v3-adapter as active Stage 0 base.

**Questions opened:** none.

**Questions resolved:**
- "Can the Stage 1 plateau be broken by more data diversity, scratchpad-validation, or a larger base model?" — PARTIALLY RESOLVED @ `8c1bb14`. The "better Stage 0 curriculum" path (not on the original list) breaks the plateau: v3 peaks at 48% vs v1's 42%. Options (a) and (c) from the original question remain untested.

---

## [2026-04-13 | 2f11f32 | Stage 0 v4 game-context Q&A — model knows Texas 42 (67% overall, is_trump 100%) (4729dad..2f11f32)]

Four commits. Major eval milestone; two eval bugs fixed; new decision page.

**Touched pages:** [[entities/lem]] [[entities/gemma-4-e2b]] [[entities/v4-adapter]] [[topics/rules-adapter]] [[topics/game-context-qa]] [[decisions/flexible-grader]] [[experiments/stage-0-v4-comprehension-eval]] [[sources/4729dad]] [[sources/1d3e1b7]] [[sources/3c33e86]] [[sources/2f11f32]]

**Added:** 6 pages — 1 entity, 1 topic, 1 experiment, 1 decision, 4 source digests.

**Updated:**
- [[entities/lem]] — Stage 0 v4 subsection added.
- [[entities/gemma-4-e2b]] — v4 eval note; "no longer hallucinates Bridge" recorded.
- [[topics/rules-adapter]] — v4 section + progression note.

**Frontier shift:**
- Stage 0 v4 pivots from flashcard Q&A to game-context Q&A (5 types from real game records, ~170 tok prompts). Thinking mode DISABLED — eliminates "this is Bridge" hallucination.
- Two eval bugs found and fixed: Gemma 4 uses `<turn|>` (token 106) for EOS; left-pad slicing must use `input_ids.shape[1]`.
- Flexible grader decision: extract facts from free-form responses rather than rigid pattern match. Moved legal_moves eval 0% → 70% on the same model responses.
- Held-out eval (100 examples): 67% overall, is_trump 100%, where_is 90%, legal_moves 70%, count_status 60%, what_beats 15%.
- v3-adapter superseded; v4-adapter is the active Stage 0 base.

**Questions opened:**
- Is what_beats (15%) under-trained, or is ranking fundamentally harder than membership for this model?

**Questions resolved:**
- "Why does 6-4 stay stubbornly misidentified as trump under fives?" — Resolved `3c33e86`. v4 eval shows is_trump at 100% on held-out set; the 6-4 error is gone.

---

## [2026-04-16 | 3465e29 | Stage 0 v5 — pivot base to Qwen 3 1.7B (100% vs Gemma 60%)]

Single commit. Major pivot: base model changes for the first time.

**Touched pages:** [[entities/lem]] [[entities/gemma-4-e2b]] [[entities/qwen3-1.7b]] [[entities/v5-adapter]] [[topics/rules-adapter]] [[topics/game-context-qa]] [[decisions/base-model-pivot-qwen]] [[sources/3465e29]]

**Added:** 4 pages — 2 entities, 1 decision, 1 source digest.

**Updated:**
- [[entities/lem]] — Stage 0 v5 Qwen pivot subsection added.
- [[entities/gemma-4-e2b]] — status flipped to `retired-as-base`; architectural reasons for retirement noted.
- [[topics/rules-adapter]] — v5 Qwen section + 5-version progression table.
- [[topics/game-context-qa]] — corpus portability note: game-context Q&A transfers to Qwen without modification.

**Frontier shift:**
- Base model pivots from Gemma 4 E2B → Qwen 3 1.7B. Comprehension: 100% vs Gemma's 60% on same eval_v5.
- Throughput: 36K tok/s on B200, ~19 min/training run.
- t42-hv08 (B200 underutilization) resolved: was a model-architecture problem, not a GPU tuning problem.
- Gemma pipeline retained as legacy; all new work targets Qwen.
- Full adapter superseded chain: stage-0-adapter → kerry-adapter → v3-adapter → v4-adapter → v5-adapter (active).

**Questions opened:** none.

**Questions resolved:**
- "Is fact-verification the only way forward past the plateau?" (implicit question from ingest 10 ceiling hypothesis) — Resolved `3465e29`. Better base model + better curriculum lifts the ceiling without fact-verification. Scratchpad validation remains an option but is not proven necessary.

---

## [2026-04-17 | be7efc4 | LEM finale — v9 (14 categories + verifier), v10 (joint rationalization), 14B capacity, SFT mask fix (b857299..be7efc4)]

Three commits. **Last LEM ingest.** Replay moves to Burl after this entry.

**Touched pages:** [[entities/lem]] [[entities/v5-adapter]] [[entities/v9-adapter]] [[entities/v10-adapter]] [[entities/qwen3-1.7b]] [[entities/qwen3-14b]] [[topics/rules-adapter]] [[topics/game-context-qa]] [[topics/r1-rationalization]] [[topics/rationalization-verifier]] [[topics/single-fact-enumeration]] [[experiments/stage-0-v9-14categories]] [[experiments/qwen-14b-capacity]] [[experiments/v10-maskfix-breakthrough]] [[decisions/sft-completion-only-loss]] [[sources/b857299]] [[sources/0c7392f]] [[sources/be7efc4]]

**Added:** 11 pages — 3 entities, 2 topics, 3 experiments, 1 decision, 3 source digests.

**Updated:**
- [[entities/lem]] — v7-v10 + 14B + maskfix subsection; LEM end-state recorded.
- [[entities/v5-adapter]] — superseded-by-v9 note.
- [[entities/qwen3-1.7b]] — full adapter lineage recorded.
- [[topics/rules-adapter]] — v7-v10 + maskfix section; complete 10-version progression table.
- [[topics/game-context-qa]] — expanded to 14 categories in v9.
- [[topics/r1-rationalization]] — rationalization-SFT bootstrap via v10 joint training.

**Frontier shift (LEM end-state):**
- v9: 14 Q&A categories + rationalization verifier (6 engine checks).
- v10: joint-trained comprehension + upweighted rationalizations → 55/100 bot-match, 96/100 legal.
- 14B capacity experiment: 97/100 rationalization; visibility_audit 0% on both 1.7B and 14B — structural gap, not capacity.
- Mask fix: TRL default SFTConfig diluted answer gradient ~9× with memorized prompt tokens; prompt/completion format auto-enables completion-only loss. Bot-match unchanged at 55/100 → not a gradient problem.
- Best adapter: v10-maskfix at 86% comprehension (= 14B v9 quality at 1/3 cost).
- Next lever is capacity or STaR iteration, not more SFT.

**Questions opened:**
- Does 14B's 97/100 rationalization survive the mask fix?
- What breaks the 55/100 bot-match ceiling? (Capacity? Actual STaR iteration?)

**Questions resolved:**
- "When will the model have learned the scratchpad format well enough to enable fact-validation?" — Resolved `0c7392f`. The rationalization format was bootstrapped via v10 joint training — a different mechanism than scratchpad validation, but the same underlying principle (format-bootstrap before fact-checking). Scratchpad per se remains shelved; the principle succeeded.

---
*LEM replay complete. 16 ingests, 2026-04-09 → 2026-04-17.*

---

## [2026-04-18 | 8d26e0d | Burl introduction — sibling project to LEM, vocabulary cleanup]

Single commit. **First Burl ingest.** Wiki gains a second project and its first trail page.

**Touched pages:** [[entities/forge]] [[entities/gemma-4-e2b]] [[entities/burl]] [[entities/zeb]] [[entities/engine]] [[topics/tool-orchestration]] [[trails/lem-to-burl-handoff]] [[sources/8d26e0d]]

**Added:** 6 pages — 3 entities, 1 topic, 1 trail, 1 source digest.

**Updated:**
- [[entities/forge]] — three-way vocabulary split codified: solver / E[Q] framework / E[Q] bot.
- [[entities/gemma-4-e2b]] — active-in-burl note added: retired as LEM base but active in Burl for agentic reasons.

**Frontier shift:**
- Burl is a sibling project to LEM: tool-using Texas 42 agent with a different base, training shape, eval, and product slot.
- Tool orchestration is Burl's philosophical core: engine has authority on rules, Zeb on beliefs, Burl reasons between and commits.
- Zeb (3.3M-param belief model) is now a first-class wiki entity.
- Engine (TS src/core/) is now a first-class wiki entity — authoritative on rules/legality/state.
- Gemma 4 E2B is simultaneously retired-as-lem-base and active-in-burl: same model, two different use cases.
- Forge vocabulary is now precise: solver ≠ E[Q] framework ≠ E[Q] bot.
- First trail page: [[trails/lem-to-burl-handoff]] makes the project transition navigable.

**Questions opened:**
- Will tool-use let a 2B-class model play competently without the comprehension curriculum LEM required?
- Is retry-on-illegal cheap enough in practice to be Burl's error-correction strategy?
- Does the model know WHEN to call which tool?

**Questions resolved:** none.

---

## [2026-04-19 | 3781dce | Burl Moves 1-4 — harness, Zeb parked, Move 3 base 70% K1, Move 4 R3 spike 88.9% (d9baf3b..3781dce)]

Three commits. Burl premise survives first contact; native format unlocks full tool surface.

**Touched pages:** [[entities/burl]] [[entities/zeb]] [[entities/gemma-4-e2b]] [[topics/tool-orchestration]] [[experiments/zeb-calibration-eval]] [[experiments/burl-move3-base]] [[experiments/burl-move4-native-spike]] [[decisions/zeb-parked-eq-primitive]] [[decisions/native-tool-use-format]] [[sources/d9baf3b]] [[sources/4b3ba3d]] [[sources/3781dce]]

**Added:** 8 pages — 3 experiments, 2 decisions, 3 source digests.

**Updated:**
- [[entities/burl]] — Moves 1-4 shipped section.
- [[entities/zeb]] — Parked-for-Burl section; advertised 72% top-1 was inflated (hidden-only 39%).
- [[entities/gemma-4-e2b]] — Burl-side findings: Move 3 failure modes, Move 4 results.
- [[topics/tool-orchestration]] — First results section; full tool surface table with parked markings.

**Frontier shift:**
- Burl premise confirmed at first contact: base 2B Gemma plays legally at 70% K1 zero-shot via tool-use (Move 3).
- Native tool-use format unlocks 88.9% K1 + 88.9% bot-match on same 10-decision eval (Move 4). Tool breadth real: eq_outcome_distribution called 15×, trump_declared 9×.
- Zeb parked: advertised 72% top-1 was whole-game; hidden-only accuracy is 39%. E[Q] N=10 PDF replaces Zeb as Burl belief primitive.
- Harness bends to model grammar: native format is Gemma's post-trained tool-use style. XML-only → is_legal; native → full surface.
- Lesson: always audit what denominator an ML headline uses.

**Questions opened:** none.

**Questions resolved:**
- "Will tool-use let a 2B-class model play competently without the comprehension curriculum?" — PARTIALLY RESOLVED `3781dce`. Yes at 70% K1 base / 88.9% native on 10-decision eval. Needs larger eval.
- "Is retry-on-illegal cheap enough in practice?" — RESOLVED `4b3ba3d`. Move 3 had 0 retries. Yes.
- "Does the model know WHEN to call which tool?" — RESOLVED (split) `3781dce`. No for XML (only calls is_legal); yes for native (uses full surface).

---

## [2026-04-19 | 789e14d | Burl Phase 1-4 iter-0 pipeline — regressed 10pp, learned something (b8116b5..789e14d)]

Four commits. First full Burl STaR iteration end-to-end. Regression diagnosed; B4 fix identified.

**Touched pages:** [[entities/burl]] [[entities/gemma-4-e2b]] [[entities/burl-iter0-adapter]] [[topics/tool-orchestration]] [[topics/star]] [[experiments/burl-phase1-primer]] [[experiments/burl-phase2-starcorpus]] [[experiments/burl-iter0-eval]] [[decisions/primer-tradeoff]] [[sources/b8116b5]] [[sources/fd6032b]] [[sources/0168210]] [[sources/789e14d]]

**Added:** 9 pages — 1 entity, 3 experiments, 1 decision, 4 source digests.

**Updated:**
- [[entities/burl]] — Moves 1-4 shipped; Phases 1-4 + iter-0 added.
- [[entities/gemma-4-e2b]] — Phase 1-4 findings on Burl side.
- [[topics/tool-orchestration]] — Updated with iter-0 results and regression context.
- [[topics/star]] — Burl STaR pipeline added alongside LEM section.

**Frontier shift:**
- First Burl STaR iteration complete end-to-end. Result: 60% bot-match — regressed 10pp from Layer-1's 70% and 29pp from the native spike's 88.9%.
- Diagnosis: iter-0 adapter baked the eq-shy pathology from Layer-1 corpus into weights. The primer taught rules but also trained the model to over-rely on E[Q] rather than commit.
- vLLM-LoRA blocker resolved via hf_overrides (Gemma4ForCausalLM). Pipeline unblocked.
- Primer trade-off codified: full primer → rules knowledge + eq-shy; trimmed primer → B4's planned fix.

**Questions opened:** none.
**Questions resolved:** none.

---

## [2026-04-19 | 09b841e | iter-1 — trimmed primer + SFT, mixed result]

Single commit. Trimmed primer partially fixes eq-shy but exposes a new failure mode.

**Touched pages:** [[entities/burl]] [[entities/burl-iter1-adapter]] [[experiments/burl-iter1-mixed]] [[decisions/commit-discipline]] [[sources/09b841e]]

**Added:** 4 pages — 1 entity, 1 experiment, 1 decision, 1 source digest.

**Updated:**
- [[entities/burl]] — iter-1 subsection added.

**Frontier shift:**
- Trimmed primer (500 words vs 1549): reduces eq-shy, amplifies reasoning depth — but drops commit discipline. 5/10 decisions retry-exhausted (model loops rather than plays).
- On the 5 completed decisions: 80% bot-match, -0.76 eq_delta — spike-v2 quality level.
- Key lesson: commit discipline is load-bearing in the primer. Dropping the primer entirely removes it. The path forward is tuning the primer to preserve commit discipline while shedding eq-shy.
- Session cost: $1.37 iter-1 / $3.22 total.

**Questions opened:** none.
**Questions resolved:** none.

---

## [2026-04-19 | b5d05de | iter-2 prep cluster — four parallel workstreams staged (f164796..b5d05de)]

Eight commits. Infrastructure sprint; no adapter trained. Four workstreams scaffolded for iter-2/iter-3.

**Touched pages:** [[entities/burl]] [[entities/haiku-4-5]] [[topics/eq-gate-star]] [[topics/ls-mixture]] [[topics/rules-as-tools]] [[topics/reference-trace-distillation]] [[sources/f164796]] [[sources/3414507]] [[sources/b3a27e2]] [[sources/1f13f92]] [[sources/761587c]] [[sources/80704f0]] [[sources/eebcae5]] [[sources/b5d05de]]

**Added:** 13 pages — 1 entity, 4 topics, 8 source digests.

**Updated:**
- [[entities/burl]] — iter-2 prep subsection; four staged workstreams noted.

**Frontier shift:**
- Four parallel workstreams staged for iter-2/iter-3, none yet run end-to-end:
  1. [[topics/eq-gate-star]] — gate STaR keep on E[Q] delta, not just K1 match
  2. [[topics/ls-mixture]] — mix legal-but-suboptimal traces into SFT to train commit discipline
  3. [[topics/rules-as-tools]] — expose rules engine as callable tools rather than baking into primer
  4. [[topics/reference-trace-distillation]] — distill Haiku 4.5 reference traces into Gemma via SFT
- [[entities/haiku-4-5]] introduced as reference-trace teacher. N=30 run: 72.4% bot-match. Zero usage of conditional_outcome — interesting constraint for distillation corpus design.
- No adapter trained this cluster.

**Questions opened:** none.
**Questions resolved:** none.

---

## [2026-04-20 | c2aa3a7 | iter-3 prep — enable_primer three-mode flag, rules_tools threaded, async concurrency (c698091..c2aa3a7)]

Five commits. Second consecutive infrastructure sprint; no adapter trained.

**Touched pages:** [[entities/burl]] [[sources/c698091]] [[sources/abb1b3d]] [[sources/65c749c]] [[sources/faefca7]] [[sources/c2aa3a7]]

**Added:** 5 pages — 5 source digests.

**Updated:**
- [[entities/burl]] — iter-3 prep subsection added.

**Frontier shift:**
- `enable_primer` three-mode flag: off / trim / full. Enables iter-3 to run controlled experiments across all three primer conditions in a single harness.
- `enable_rules_tools` threaded end-to-end. Rules-as-tools workstream wired and ready.
- Async STaR rollout concurrency prepped. Iter-3 run itself comes in B7.

**Questions opened:** none.
**Questions resolved:** none.

---

## [2026-04-19 | dbadb5f | iter-3 winner + selfplay arena + iter-4 null + Opus lock (35c75ff..dbadb5f)]

Five commits, ~90 min. Major milestone: iter-3-rules is the winning Burl adapter.

**Touched pages:** [[entities/burl]] [[entities/gemma-4-e2b]] [[entities/iter3-rules-adapter]] [[entities/selfplay-arena]] [[topics/rules-as-tools]] [[topics/preserve-thoughts]] [[topics/conditional-outcome-structural-nonuse]] [[experiments/iter3-comparison]] [[experiments/iter4-null-preserve-thoughts]] [[experiments/opus-vs-haiku-arena]] [[sources/35c75ff]] [[sources/2830be0]] [[sources/20f4fa2]] [[sources/39aafaf]] [[sources/dbadb5f]]

**Added:** 12 pages — 2 entities, 2 topics, 3 experiments, 5 source digests.

**Updated:**
- [[entities/burl]] — iter-3 winner + iter-4 null + arena + Opus lock subsections.
- [[entities/gemma-4-e2b]] — iter-3-rules bullet on Burl side.
- [[topics/rules-as-tools]] — Validated section: 90% bot-match confirms tools-replace-memorization.

**Frontier shift:**
- **iter-3-rules is the winning Burl adapter: 90% bot-match, 0 retry-exhausted, 100% first-legal.** Rules-as-tools + no primer. trick_winner_if usage UP after SFT — validates the tools-replace-memorization hypothesis.
- iter-4 preserve_thoughts: null result. Byte-identical A/B output. Hypothesis: LoRA rank 16 is capacity-saturated; preserve_thoughts signal has nowhere to go.
- Selfplay arena ships. First match: Opus 7 / Haiku 0 on seed 900010. Opus uses 1× trump_declared vs Haiku 24× — dramatic tool-economy delta.
- conditional_outcome structurally unused: 145+ decisions across 4 models, 0 calls. Design signal that the tool may not be needed.
- Cumulative cost: $20.50 / $40 budget.

**Questions opened:** none.

**Questions resolved:**
- "Does the primer tradeoff resolve?" — YES: rules-as-tools + no primer → 90% bot-match. Full primer → 70%. The tradeoff resolves firmly toward removing the primer when tools cover rules.
- "Will native tool-use + rules-as-tools hit target?" — YES on 10-decision eval at 90%.

---

## [2026-04-19 | edf86e9 | MLX-LM local path + SFTConfig max_seq_length=4096 fix (6fea6ab..edf86e9)]

Two commits. Local inference path added; iter-4 null diagnosis corrected.

**Touched pages:** [[entities/burl]] [[entities/mlx-lm]] [[topics/preserve-thoughts]] [[experiments/iter4-null-preserve-thoughts]] [[decisions/sft-max-seq-length]] [[sources/6fea6ab]] [[sources/edf86e9]]

**Added:** 3 pages — 1 entity, 1 decision, 2 source digests.

**Updated:**
- [[entities/burl]] — MLX-LM local path + SFT truncation reframe subsections.
- [[topics/preserve-thoughts]] — reframed: null was truncation artifact, not LoRA capacity limit.
- [[experiments/iter4-null-preserve-thoughts]] — status retired/reframed; LoRA capacity hypothesis superseded.

**Frontier shift:**
- Burl pipeline can now run locally on Apple Silicon via mlx-lm: 1.86× wall vs Modal, ~10 GB at rank 4.
- iter-4 preserve_thoughts null reframed: TRL default max_seq_length=1024 clipped thought-bearing rows (median 2054, max 4210). The A/B was byte-identical because both branches had zero thought content in training. Parallel to LEM's sft-completion-only-loss trap — TRL defaults are a recurring gradient sink.
- LoRA rank-16 capacity-saturated hypothesis from B7 is no longer privileged. Iter-5 will test cleanly with max_seq_length=4096.

**Questions opened:** none.
**Questions resolved:** none.

---

## [2026-04-20 | 0545342 | candlewax/iter-5 cluster — reasoning-coherence bottleneck named, MLX batch 16× (1efb9c5..0545342)]

Eight commits. Three parallel threads: iter-5 evals, MLX perf, candlewax spike.

**Touched pages:** [[entities/burl]] [[entities/mlx-lm]] [[entities/candlewax-spike]] [[topics/tool-orchestration]] [[topics/candlewax]] [[topics/reasoning-coherence-verification]] [[experiments/iter5-e1-rank-sweep]] [[experiments/iter5-e2-candlewax-null]] [[experiments/batch-throughput-bench]] [[experiments/candlewax-spike-e2e]] [[sources/1efb9c5]] [[sources/ceca203]] [[sources/7321952]] [[sources/ed3cfc3]] [[sources/b0952a2]] [[sources/6a97d55]] [[sources/aeafe22]] [[sources/0545342]]

**Added:** 14 pages — 2 entities, 2 topics, 4 experiments, 8 source digests.

**Updated:**
- [[entities/burl]] — iter-5 E1/E2 + candlewax subsections; reasoning-coherence bottleneck.
- [[entities/mlx-lm]] — batch ceiling updated: 43 → 1334 tok/s (16×) after batching fix.
- [[topics/tool-orchestration]] — reasoning-coherence verification as next architectural lever.

**Frontier shift:**
- iter-5 E1 (rank-16, preserve_thoughts, truncation fix): 70% bot-match, -2.83 eq_delta. First clean preserve_thoughts adapter; still below iter-3-rules 90%.
- iter-5 E2 (candlewax-aware null): bimodality at tool surface doesn't change behavior. Candlewax not yet integrated into training loop.
- **Reasoning-coherence verification identified as the bottleneck.** Model produces syntactically valid traces that are semantically incoherent; K1 grading cannot filter these.
- MLX batch ceiling: 43 → 1334 tok/s (16×). Local iteration now competitive with Modal for small runs.
- **Candlewax spike E2E**: Qwen 3.6-35B-A3B via mlx-vlm reasoning-coherence verifier works end-to-end. Pivots away from LLM-as-reasoner approach.
- PRACTICALITIES.md split; 8 practicalities captured.

**Questions opened:** none.
**Questions resolved:** none.

---

## [2026-04-23 | 1bf1885 | Burl finale — chat-template fix, wax_museum, belief_trajectory, Gus first appearance (54f7776..1bf1885)]

Three commits (2026-04-20 → 2026-04-23). **Last Burl ingest.** Replay moves to Gus after this entry.

**Touched pages:** [[entities/burl]] [[entities/zeb]] [[entities/gus]] [[entities/wax-museum]] [[entities/belief-trajectory]] [[topics/conditional-outcome-structural-nonuse]] [[decisions/gemma-tool-response-shape]] [[experiments/chat-template-fix-validation]] [[sources/54f7776]] [[sources/d858781]] [[sources/1bf1885]]

**Added:** 7 pages — 3 entities, 1 decision, 1 experiment, 3 source digests.

**Updated:**
- [[entities/burl]] — chat-template confound + wax_museum + belief_trajectory + Gus first appearance subsections.
- [[entities/zeb]] — Gus supersedes Zeb as Burl's belief source.
- [[topics/conditional-outcome-structural-nonuse]] — reframed: 0/145 calls was environment confound (tool responses invisible); question now is whether model uses conditional_outcome when actually visible.

**Frontier shift (Burl end-state):**
- **Chat-template confound discovered and fixed**: Gemma 4's Jinja template silently dropped `role="tool"` messages. Every pre-fix Burl rollout (conditional_outcome=0/145, three A/B runs, all adapter bot-match scores) had invisible tool responses. Post-fix base Gemma: 5/5 bot-match on N=5 held-out with faithful numeric quoting.
- This reframes the entire Burl replay: all bot-match numbers below 100% may reflect environment failures, not model failures.
- **Three TRL/template traps now documented**: sft-completion-only-loss (LEM), sft-max-seq-length (Burl iter-4), gemma-tool-response-shape (Burl B10).
- wax_museum harness ships: hard-gated HATEOAS with three extension hooks (system_prompt_transform, preload_tool_calls, menu_override).
- belief_trajectory tool wires Gus's calibrated belief head into Burl. Gus supersedes Zeb.
- Gus (stub) first appears in the wiki — handoff to Gus replay session.

**Questions opened:**
- Does the model use conditional_outcome when tool responses are actually visible (post chat-template fix)?

**Questions resolved:** none.

---
*Burl replay complete. 10 ingests (B1–B10), 2026-04-18 → 2026-04-23.*

---

## [2026-04-23 | 31e10ef | Gus kickoff — joint-world tensor + BUILD_PLAN, 5-head LAMIR-ready (42a7535..31e10ef)]

Two commits. **First Gus ingest.** Third sibling project introduced.

**Touched pages:** [[entities/gus]] [[entities/joint-world-tensor]] [[topics/student-distillation]] [[topics/lamir1]] [[experiments/gus-joint-world-tire-kick]] [[sources/42a7535]] [[sources/31e10ef]]

**Added:** 6 pages — 1 entity, 2 topics, 1 experiment, 2 source digests.

**Updated:**
- [[entities/gus]] — expanded from Burl-era stub; full project description, 5-head architecture, BUILD_PLAN staging.

**Frontier shift:**
- Gus is the third sibling project alongside LEM and Burl. Skips the reasoning channel entirely; distills [[forge]]'s E[Q] oracle via supervised multi-head transformer.
- 5-head output targets: belief / V / π_me / Q / π_opp (π_opp deferred). All targets are LAMIR-ready.
- [[entities/joint-world-tensor]] enables look-ahead without oracle calls — packed game-state representation.
- BUILD_PLAN commits to v0 → v1 → v2 training staging.
- Entities index now has three project sections: LEM / Burl / Gus.

**Questions opened:** none.
**Questions resolved:** none.

---

## [2026-04-20/21 | 3c02d10 | Gus student scaffolding — v0 MLP → v1 transformer → 4-head → 1000g v2 (c04bda3..3c02d10)]

Five commits (2026-04-20/21). Gus v0 through v2 student training fully exercised.

**Touched pages:** [[entities/gus]] [[topics/dense-q-supervision]] [[experiments/gus-v0-v1-belief]] [[experiments/gus-4head-baseline]] [[experiments/gus-v2-voids-1000g]] [[sources/c04bda3]] [[sources/8dbf7f3]] [[sources/da21f52]] [[sources/2e4f586]] [[sources/3c02d10]]

**Added:** 9 pages — 1 topic, 3 experiments, 5 source digests.

**Updated:**
- [[entities/gus]] — v0→v1→v2 Progress subsection added with full metric tables.

**Frontier shift:**
- **v0 MLP (c04bda3)**: 183-dim flat features, ~200K params, 100g corpus. Peak eval 34.6% (chance 33.3%), train 100% — severe overfit. Architecture confirmed; data is the bottleneck.
- **v1 transformer (8dbf7f3)**: replaces bag-of-masks with attention over tokenized play sequences (33 tokens, 5 channels). Enables void inference. 100g: eval 37.5%, train 74%. Late-game belief hits 75% at decision 26 — the aggregated 37.5% is dragged down by the mandatory chance floor at decision 0. Ceiling is data, not architecture.
- **4-head student (da21f52)**: adds V + π_me + world-conditioned Q heads. Dense Q supervision (~3400× gradient signal per decision vs belief alone) regularizes the shared encoder. 100g: π_me 57.9% with no overfit (train and eval track within 1-2 pts). Architecture proven.
- **v2 + 1000g (2e4f586, 3c02d10)**: glob-expand fix enables chunked corpus loading. VoidsEncoder adds [24]-dim void indicator. 1000g result: π_me 66.1% → 65.2% (flat), belief 37.2% → 38.6% (+1.4pp). Transformer already inferred voids attentionally — explicit features are marginal. Real next lever: more data + bigger model.
- [[topics/dense-q-supervision]] established: Q supervision is simultaneously a training-time regularizer and an inference-time LAMIR primitive.

**Questions opened:** none.
**Questions resolved:** none.

---

## [2026-04-21 | fdcd654 | Gus eval harness + LAMIR primitive + 2000g scaling ceiling (5a4c9b9..fdcd654)]

Six commits (2026-04-21). Eval infrastructure, inference-mode comparison, data scaling to 2000g, and ceiling confirmation.

**Touched pages:** [[entities/gus]] [[topics/pimc]] [[topics/regret-eval]] [[experiments/gus-lamir-primitive-eval]] [[experiments/gus-scaling-ladder]] [[sources/5a4c9b9]] [[sources/2a09050]]

**Added:** 7 pages — 2 topics, 2 experiments, 2 source digests, 1 entity update. (Note: sources 0472125, a50c9ef, 5cdec8a, fdcd654 not yet written by scribe at time of indexing.)

**Updated:**
- [[entities/gus]] — scaling ladder + eval metrics subsection; inference-mode comparison; regret metric; decision-hardness analyzer; open directions.
- [[topics/dense-q-supervision]] — regret-eval note appended.

**Frontier shift:**
- **LAMIR primitive eval (5a4c9b9)**: direct π_me (65.4%) beats PIMC-Q K=1 (62.1%) and PIMC-belief K=50 (61.8%). π_me is already the marginalized policy — single-step look-ahead adds variance, not information. PIMC occasionally wins on specific decisions (dec 14: 45/55/65) — hints of complementary signal for ensembles.
- **LAMIR reframing**: single-step PIMC is not the path. LAMIR's value requires multi-step look-ahead with mid-tree belief updates, which needs the π_opp head. Deferred.
- **Regret metric introduced (2a09050)**: `regret = oracle_best_eq − student_chosen_eq`. 35% bot-mismatches are overwhelmingly near-tie alternatives (≤0.5 Q-pt). Mean regret 2.16 Q-pts on ±42 scale = ~2.5% of Q-range lost per decision.
- **2000g scaling ladder (0472125, 5cdec8a, fdcd654)**: best adapter `v2_voids_big_2000g` (3.4M params, d=256, 6L, 60 epochs): 67.3% bot-match, **1.60 Q-pt regret**. 120-epoch re-run confirms ceiling (1.64 — tied). Data dominates capacity: 3.4M→7.4M on same 2000g data is flat.
- **Decision-hardness finding (a50c9ef)**: high-regret decisions correlate with high oracle E[Q] spread. Decision 0: spread 13.2 Q-pts, student regret 4.0 — well below random baseline of 6.6.
- **Architecture/data ceiling confirmed at 2000g × 3-7M params.** Next levers: 5000g–10000g scaling, π_opp head, multi-step LAMIR.

**Questions opened:** none.
**Questions resolved:** none.

---

## [2026-04-21 | b007cf3 | Gus 3000g new best + arena pilot + GEN_FLEET + v3 consistency (286eb23..b007cf3)]

Five commits (2026-04-21). Data scaling continues, game-level eval surfaces V/π decoupling, v3 remedy designed.

**Touched pages:** [[entities/gus]] [[entities/gen-fleet]] [[topics/regret-eval]] [[topics/v-pi-decoupling]] [[topics/consistency-regularizer]] [[experiments/gus-scaling-ladder]] [[experiments/gus-arena-pilot]] [[sources/286eb23]] [[sources/f0139a3]] [[sources/a8bc35a]] [[sources/1a2f67f]] [[sources/b007cf3]]

**Added:** 8 pages — 1 entity, 2 topics, 1 experiment, 5 source digests.

**Updated:**
- [[entities/gus]] — 3000g row, arena results, V/π decoupling, consistency regularizer, GEN_FLEET plan subsections.
- [[topics/regret-eval]] — bimodal distribution finding added: 73% perfect / 6% blunder tail.
- [[experiments/gus-scaling-ladder]] — 3000g row added; best updated to v2_voids_3000g_big at 1.39 Q-pt regret.

**Frontier shift:**
- **3000g NEW BEST (286eb23)**: v2_voids_3000g_big at 1.39 Q-pt mean regret (−13% vs 2000g's 1.60). Data scaling curve continues; not yet plateaued. Note: 7.4M XL at 3000g gave highest composite score but worse regret (1.54) — composite and regret can disagree.
- **PRACTICALITIES receipts (f0139a3)**: 10 receipts including receipt #5 — regret is bimodal: 73% of decisions have 0 regret, 6% blunder tail drives all of mean regret. Mean regret scalar is misleading; the action is in closing the blunder tail.
- **GEN_FLEET plan (a8bc35a)**: Vast.ai distributed corpus generation across 8 workers, ~$15/10k games. Prerequisite: `--n-decl-per-seed` flag (oracle uses 10 decls/seed; Gus uses 1 → 100× fewer distinct states). Phased plan, not yet launched.
- **Arena pilot (1a2f67f)**: first game-level eval. `v2_voids_3000g_big` at seat 0 vs E[Q] bot: 10/20 contracts made (50%) vs all-bot baseline 16/20 (80%), −8.4 bidder points/hand. 1.39 Q-pt regret compounds to ~30pp game-level gap. Blunder forensics surfaced V/π head decoupling: V_head outputs +26 on a decision where π_me picks a −0.4 play.
- **V/π decoupling identified (1a2f67f)**: V and π heads converge independently via shared encoder but without direct coupling loss. V "knows" the position is good while π acts badly.
- **v3 consistency regularizer (b007cf3)**: `L_consistency = (V_head.detach() − Σ softmax(π_me)·Q)²`. Gradient flows only into π_me; warmup 10 epochs. Smoke-tested; full 3000g run not yet complete at this frontier.

**Questions opened:** none.
**Questions resolved:** none.

---

## [2026-04-21 | a09ef43 | Gus blunder detector + detect-and-route PoC (f90682c..a09ef43)]

Eight commits (2026-04-21). Parallel-agent session: blunder analysis, ensemble ceiling, deployable detector, router end-to-end validation.

**Touched pages:** [[entities/gus]] [[topics/blunder-detector]] [[topics/detect-and-route]] [[topics/router-reality-check]] [[experiments/gus-blunder-detector]] [[experiments/gus-router-pilot]] [[experiments/gus-scaling-ladder]] [[sources/f90682c]] [[sources/5373223]] [[sources/109f9e1]] [[sources/eba5103]] [[sources/a09ef43]]

**Added:** 8 pages — 3 topics, 2 experiments, 5 source digests. (Note: scribe used eba5103 and 109f9e1 as additional commits beyond the 6 in team-lead manifest; all on disk.)

**Updated:**
- [[entities/gus]] — blunder detector + detect-and-route subsections added.
- [[experiments/gus-scaling-ladder]] — updated (referenced from new pages; content unchanged).

**Frontier shift:**
- **Ensemble analysis (f90682c)**: naive ensembling (majority vote, softmax avg, V-weighted) boosts bot-match to 71.3% but hurts regret (1.43-1.55 vs 1.39 best). Oracle-per-decision ceiling: 0.36 regret (74% reduction) — 58% of decisions have adapter disagreement. The correct architecture is a router, not an averaging ensemble.
- **Oracle-feature blunder detector (f90682c)**: GBM on oracle E[Q] features, AUC 0.926. Dominant features: oracle_spread + oracle_eq_std (77% importance). At 25% flag rate: 99% recall. Business case: 1.23 → 0.46 regret at 20% flag with oracle replacement.
- **Student-feature blunder detector (5373223)**: deployable version using 28 student-only features. AUC 0.839 / PR-AUC 0.15. Top feature: `pi_peak` (0.19) — "when π is uncertain, call for help." Business case at 20% flag: 1.13 → 0.49 regret (57% reduction). Key limit: Q_head spread is a weak blunder proxy (trained on 1 world per forward pass).
- **PRACTICALITIES receipts 11-13 (109f9e1)**: arena 1.39 Q-pt → 18pp win-rate gap (receipt 11); ensemble hurts (receipt 12); detect-and-route at AUC 0.839 / 0.49 regret (receipt 13). "Detect → route → fallback" named as deployable Gus v1.0 shape.
- **Detect-and-route end-to-end (eba5103)**: student-feature GBM + three fallback policies on 560 held-out decisions. Oracle argmax: 1.39 → 0.56/0.49 regret at 20/25% flag — works as projected. PIMC-Q-K50: 1.39 → 1.47 (WORSE) — fixes 7 blunders, introduces 6 new ones; Q_head too noisy. Next-best-adapter: 1.39 → 1.55 (WORSE) — weaker adapters fail on the same hard decisions.
- **Receipt 14 / router reality-check (a09ef43)**: oracle-fallback path is deployable now. No-oracle path requires Q_head multi-world variance regularization (during training) or K=50+ worlds at inference. Router benefit concentrates on mid-game (decisions 0-12, especially 4, 8, 10); end-game (24-27) never flagged.

**Questions opened:** none.
**Questions resolved:** none.

---

## [2026-04-21 | f138069 | Gus shine analysis + belief calibration gap + lazy dataset (695f2ef..f138069)]

Six commits (2026-04-21). Continued parallel-agent session: explanation sketcher, shine forensics, belief calibration ablation, lazy streaming loader. (Note: source digests b4040c5, 41fdb3c, 31f0ec3, 245918d and experiments/gus-v3-consistency not yet written by scribe; will be captured next ingest. topics/consistency-regularizer full-run update also pending.)

**Touched pages:** [[topics/shine-analysis]] [[topics/belief-propagation-gap]] [[topics/lazy-iterable-dataset]] [[experiments/gus-shine-analysis]] [[experiments/gus-belief-calibration-diagnostic]] [[entities/gus]] [[sources/695f2ef]] [[sources/7a9c720]] [[sources/137a8e7]] [[sources/f138069]]

**Added:** 9 pages — 3 topics, 2 experiments, 4 source digests.

**Updated:**
- [[entities/gus]] — shine analysis, belief propagation gap, lazy IterableDataset, explanation sketcher subsections added.

**Frontier shift:**
- **Explanation sketcher (695f2ef)**: template-based NL rationalization over existing head outputs — no new ML. Surfaces V/π disagreement, policy uncertainty, forced-move patterns in human-readable blockquotes in the visualizer. Correctly mechanical on dead-ties. ML rationale head identified as the storyteller upgrade path.
- **Shine analysis (7a9c720)**: mirror of blunder forensics. 410/560 decisions are perfect (regret < 0.1). Composition: 59% dead-ties, 15.6% moderate, **25.4% sharp-and-perfect** (104 decisions with spread ≥ 5 where student chose correctly — real inference, not luck). Zero-inference routing heuristic: trust student when `legal_count ≤ 2 OR decision_idx ≥ 22` — covers 80% of decisions at ≤ 2% blunder rate, 4× reduction in detector workload.
- **Belief calibration diagnostic (137a8e7)**: frozen-trunk fine-tune of belief head with distribution target. KL to world-marginal improves 21% (0.078 → 0.062). But downstream: PIMC-belief K=50 regresses 1pp, blunder detector PR-AUC regresses. Takeaway: calibration-only fine-tune on frozen ecosystem creates distribution shift Q_head was not prepared for. Co-training {belief, world_encoder, Q_head} together is the prerequisite. Not promoted; kept in scratch/.
- **Lazy IterableDataset (f138069)**: `JointWorldFullIterable` streams chunks through shuffle buffer; memory bounded at ~3.4 GB regardless of corpus size. Unlocks 10k+ game training. `train_v3_consistency` OOM'd on eager load of full 10k corpus; this fixes it.

**Questions opened:** none.
**Questions resolved:** none.

**G6-supplement (all landed):** Source digests b4040c5, 41fdb3c, a14200f, 31f0ec3, 245918d all confirmed on disk and indexed.

Updated pages from G6-supplement:
- [[topics/consistency-regularizer]] — v3 10k full-run results appended (31f0ec3): **v3_consistency_10000g at 0.551 regret** (first Gus adapter under 1.0). v2_voids_10000g at 0.818; v3 is −33% vs v2 at same 10k scale. Consistency loss scales better than plain distillation — gap widens from ~tied at 3k to decisive at 10k. Decision: consistency loss rides forward into LAMIR-1 training.
- [[entities/gen-fleet]] — pre-launch fix list added (a14200f): 5 fixes required before diverse-seed fleet runs. Lazy dataset: done (f138069). Oracle bid=30 bias, schema v2 (adds bid_value + per-seat oracle softmax for π_opp targets), length-cache sidecar, resume-safe HF upload: not yet shipped.
- [[experiments/gus-scaling-ladder]] — last_updated bumped to 31f0ec3; 10k rows live in topics/consistency-regularizer.

---

## [2026-04-21 | 245918d | Gus v3-10k full results + GEN_FLEET pre-launch fixes + interpretability probes (a14200f..245918d)]

Five commits (2026-04-21 evening). v3 confirmed best; fleet prerequisites captured; model internalization verified.

**Touched pages:** [[entities/gus]] [[entities/gen-fleet]] [[topics/consistency-regularizer]] [[topics/probe-analysis]] [[topics/qmae-plateau]] [[experiments/gus-v3-consistency-full-run]] [[experiments/gus-probe]] [[experiments/gus-scaling-ladder]] [[sources/b4040c5]] [[sources/41fdb3c]] [[sources/a14200f]] [[sources/31f0ec3]] [[sources/245918d]]

**Added:** 7 pages — 2 topics, 2 experiments, 5 source digests. (Note: source digests for these commits were delivered with G6 pages; folded into this entry.)

**Updated:**
- [[entities/gus]] — v3-10k results, qMAE plateau, probe findings, GEN_FLEET pre-launch fix list subsections.
- [[entities/gen-fleet]] — pre-launch fix list (5 fixes; 1 done, 4 pending).
- [[topics/consistency-regularizer]] — v3 10k full-run results: 0.551 regret, new best.
- [[experiments/gus-scaling-ladder]] — last_updated bumped.

**Frontier shift:**
- **v3-10k full result (b4040c5, 31f0ec3)**: v3_consistency_10000g at **0.551 regret**, 76.07% bot-match. v2_voids_10000g at 0.818 / 73.21%. Consistency loss beats plain distillation by −33% at 10k. Total regret reduction v2-3k → v3-10k: −60% (data scaling −41%, consistency loss additional −33%). **First Gus adapter under 1.0 Q-pt regret.**
- **PRACTICALITIES §§16-17 (b4040c5)**: §16 — lazy IterableDataset fixed OOM. §17 — oracle utility is p_make (cliff-shaped), not E[Q]; `U = E[Q] + C·p_make` was wrong (picks guaranteed-loss over sliver-of-hope). Correct: pure p_make argmax. bid=30 bias and schema v2 tracked in GEN_FLEET.
- **qMAE plateau §18 (41fdb3c)**: qMAE improved only 7% from 3k→10k while regret dropped 59% and V-MAE 20%. Structural cause: Q_head trains on one world per forward pass, no cross-world consistency reward. Fix paths: multi-world variance regularization or joint co-training. LAMIR-1 not dependent on Q_head → not blocking.
- **GEN_FLEET pre-launch fixes (a14200f)**: 5 fixes before diverse-seed fleet launch. Lazy dataset: done. Oracle bid=30 bias, schema v2 (bid_value + per-seat softmax for π_opp), length-cache sidecar, resume-safe upload: all pending.
- **Interpretability probes §19 (245918d)**: six-probe receipt on v3-10k against nightmare hand (seed 900000, blanks, 5/7 trumps held by opponents). Key results: (1) embedding structure — doubles/counts/high-pip cluster distinctly; (2) attention — DECL → MINE → action commit across layers; (3) counterfactual V deltas match oracle within **0.5 Q-pts**; (4) 6-6 impact is trumpness-gated (+17-22 trump, ~0 fours, −28 displacing boss); (5) 0-0 location worth 26 Q-pt swing. Correction: initial "strategy-fusion" diagnosis retracted — bilateral-swap asymmetry. **Conclusion: Gus has internalized real game structure, not argmax lookup.** Counterfactual tool promotable to gus/eval/.

**Questions opened:** none.
**Questions resolved:** none.

---

## [2026-04-22 | fb03970 | LAMIR-1 rollout scaffolding — 4 modes + 2 bug fixes (581bf1f..fb03970)]

Six commits (2026-04-22). LAMIR-1 harness built, first runs show look-ahead hurts, root cause isolated to V_head distribution shift.

**Touched pages:** [[entities/gus]] [[topics/lamir1]] [[experiments/gus-lamir1-pilot]] [[experiments/gus-lamir1-mode-comparison]] [[sources/581bf1f]] [[sources/7d2af99]] [[sources/e4e6862]] [[sources/566bc4d]] [[sources/8544fbe]] [[sources/fb03970]]

**Added:** 8 pages — 2 experiments, 6 source digests.

**Updated:**
- [[entities/gus]] — LAMIR-1 rollout subsection with mode comparison table and bug notes.
- [[topics/lamir1]] — rollout modes section appended; Bug 5 and Fix 2 documented with invariants.

**Frontier shift:**
- **LAMIR-1 first rollout (581bf1f)**: `lamir1.py` scaffolded (536 lines). 1-ply look-ahead using rotation-equivariant π_me as π_opp + V_head leaf. Result: **regret 2.384 vs 0.551 direct** — 4.3× worse. Damage at trick_pos 0-2; trick_pos 3 (direct fallback) unchanged. Argmax π_me as π_opp creates adversarial leaf states V_head correctly scores as worse, causing action ordering to diverge from E[Q].
- **v-bootstrap mode + per-trick-pos table (7d2af99)**: adds depth-1 V_head with no opponent simulation. Result: regret 2.777 / bot-match 59.46%. trick_pos 0 (leads): 7.41 regret / 22% match; trick_pos 3 (= direct): 0.43 / 82%. **Key finding: v-bootstrap is as broken as full rollout at trick_pos 0 — opponent simulation is not the cause.** V_head itself has distribution shift at the immediately-post-play state.
- **Bug 5 fix (e4e6862)**: opp token building was using real deal hands for slot→domino lookup while π_me selected against world hands — incoherent `(tokens, world_assign)` pairs. Fixed by precomputing per-world 4-player `_world_game_hands` throughout `lamir1_decision`. Silent bug (no crash, plausible-looking output).
- **Fix 2 (566bc4d)**: V_head is in the leaf current_player's team frame; when leaf player is on opponent team, value must be negated before argmax. One-line fix; significant correctness impact. Silent bug.
- **q-bootstrap mode (8544fbe)**: depth-1 Q_head (world-conditioned) instead of V_head; no opponent simulation. Tests whether world-aware Q_head closes the distribution-shift gap vs world-blind V_head.
- **lamir1-qleaf mode (fb03970)**: full opp simulation + Q_head at trick-winner leaf. Completes the four-mode harness (direct / v-bootstrap / q-bootstrap / lamir1-qleaf). Empirical results for q-bootstrap and lamir1-qleaf in next ingest.

**Questions opened:** none.
**Questions resolved:** none.

---

## [2026-04-22 | b42669a | LAMIR-1 ceiling — π_opp trained, all 8 modes lose to direct (93859a0..b42669a)]

Seven commits (2026-04-22). π_opp head trained; full 8-mode LAMIR-1 ladder completed; ceiling confirmed; pivot options documented.

**Touched pages:** [[entities/gus]] [[topics/lamir1]] [[topics/pi-opp-head]] [[topics/lamir1-ceiling]] [[experiments/gus-pi-opp-training]] [[experiments/gus-lamir1-piopp]] [[experiments/gus-lamir1-mode-comparison]] [[sources/93859a0]] [[sources/dcd9365]] [[sources/1a1a324]] [[sources/b4e8ecd]] [[sources/2c380a6]] [[sources/8106f01]] [[sources/b42669a]]

**Added:** 9 pages — 2 topics, 2 experiments, 7 source digests.

**Updated:**
- [[entities/gus]] — π_opp head, full 8-mode ladder verdict, Bug 6, root cause, pivot options.
- [[topics/lamir1]] — lamir1-piopp mode + Bug 6 documented with training invariant.
- [[experiments/gus-lamir1-mode-comparison]] — updated with q-bootstrap and lamir1-qleaf results.

**Frontier shift:**
- **Schema v2 dataset loader (dcd9365)**: exposes `oracle_softmax_per_seat [4,7]`, `legal_mask_per_seat [4,7]`, `voids_per_seat [4,24]`. Backwards-compatible with v1 corpora.
- **π_opp training (93859a0, b4e8ecd)**: `PiOppHead` (1,879 params) on frozen v3 trunk. Legal-masked CE against oracle softmax per seat. NaN bug: `0 × (−∞) = NaN` per IEEE 754 when illegal slot log_probs multiplied by zero target — fixed by zeroing illegal log_probs before dot-product. Result: **68.6% oracle top-1** vs ~55% for rotated π_me proxy.
- **lamir1-piopp mode (1a1a324)**: uses trained PiOppHead for opp steps instead of rotated π_me; Q_head at trick-winner leaf. Result: regret 2.268 / bot-match 62.10% — worse than lamir1-qleaf (2.006). Better opp simulation makes rollout slightly worse, confirming opp quality is not the bottleneck.
- **Bug 6 + Fix 6 (2c380a6)**: post-rollout world_assign was stale — dominoes played during rollout still marked present. Fix: record and zero depleted domino rows in `world_assign_leaf`. The fix made every mode slightly worse, revealing the training invariant: `dataset_seq_world` preserves the original deal layout throughout; Q_head expects full initial-deal assignment, not depleted.
- **LAMIR-1 ceiling (8106f01, b42669a)**: full 8-mode ladder on 560 held-out decisions. Direct π_me (0.551) beats all look-ahead variants. Best look-ahead: q-bootstrap at 0.679 (+23%). Full rollout modes 2.006–2.777. Root cause: Q_head trained on initial-deal world assignments is OOD at depleted post-rollout leaf states. Deeper cause: Kubíček & Lisý require T×T multi-valued-states value function; scalar V/Q distillation noise overwhelms any leaf signal across rollout steps.
- **Four §20 pivot options**: (1) accept depth-1 ceiling — ship q-bootstrap as secondary mode; (2) train look-ahead-compatible V-head on opp-sampled targets; (3) implement LAMIR faithfully (T×T, months); (4) Bridge-AI/PPO self-play using π_opp + Q_head as raw materials.

**Questions opened:** none.
**Questions resolved:** none.

---

## [2026-04-22 | 94d8646 | Gus FINAL — aug Q_head path closed, belief Bayes ceiling, co-train falsified, §22 future direction (a9fa0c6..94d8646)]

Five commits (2026-04-22). Final Gus ingest. Three investigations closed; one open direction recorded.

**Touched pages:** [[entities/gus]] [[topics/q-head-augmentation]] [[topics/belief-bayes-ceiling]] [[topics/belief-co-train]] [[topics/past-belief-future-direction]] [[topics/lamir1-ceiling]] [[experiments/gus-q-head-augmentation]] [[experiments/gus-belief-co-train]] [[sources/a9fa0c6]] [[sources/5f390fb]] [[sources/548d32a]] [[sources/cf8ff79]] [[sources/94d8646]]

**Added:** 9 pages — 4 topics, 2 experiments, 5 source digests.

**Updated:**
- [[entities/gus]] — aug Q_head postmortem, belief ceiling, co-train falsified, q-bootstrap-belief 0.655, §22 future direction.
- [[topics/lamir1-ceiling]] — updated with q-bootstrap-belief 0.655 as new best look-ahead result.

**Frontier shift:**
- **Path (a) closed — aug Q_head postmortem (a9fa0c6, 5f390fb)**: fine-tune Q_head with random-depletion augmentation (freeze trunk; train q_head + world_encoder). Best eval qMAE: 8.169. lamir1-qleaf + aug Q_head: **2.216 regret** — worse than pre-aug rollouts. Root cause: random zeroing ≠ structured causal depletion. Training data must contain actual rollout-generated depleted states, not randomly zeroed rows. Path (a) confirmed closed; path (b) or end-to-end joint training is the only viable fix.
- **Belief at Bayes ceiling (548d32a)**: `belief_ceiling.py` computes Bayes-optimal top-1 from the oracle's own sampled worlds. **39.184%** on corpus_eval_20.pt. Gus v3 matches within noise (~38-39%). Per-decision: d_idx 0-5 = ~33% (pure prior — hidden info not revealed yet); d_idx 18-25 = ~50-75% (void + exclusion inference sharpens). Top-1 accuracy is a dead lever. **Zeb's 39% "plateau" was always the information ceiling of the game state, not an architecture limit.**
- **Belief co-train falsified (cf8ff79)**: joint co-train belief + world_encoder + Q_head with distribution target. Belief KL: 0.084 → 0.067 (−20%). q-bootstrap regret: 0.685 → 0.718 (+5% worse). Q_head was at a sweet spot for the original belief's output distribution; retraining upsets that equilibrium. "Calibration propagates" hypothesis falsified on this pipeline.
- **q-bootstrap-belief: unexpected best look-ahead (cf8ff79)**: inference mode where worlds are sampled from the belief head rather than read from oracle corpus. On the original adapter: **regret 0.655** vs 0.679 corpus worlds — the **closest any look-ahead variant has reached to the 0.551 direct baseline (gap: 19%)**. Mechanism: oracle adaptive sampling over-concentrates on near-consensus worlds; belief-head softmax sampling is smoother and better aligned with Q_head's training distribution. `sample_worlds.py` (written off-plan, now in use) is the artifact.
- **§22 — past belief (94d8646)**: with belief at the Bayes ceiling, remaining 42-craft is acting well under unresolvable uncertainty. Three analytics computable from q_per_world with no new training: outcome-variance, action-choice fragility, belief-limited high-impact decisions. Research question: π_me commits to one meta-strategy ("mode of the marginal"); humans use mode/signal/hedge/gamble. A richer student could output a meta-strategy distribution. Training data already exists in the oracle's per-world tensor. No code — noted as future direction.

**Questions opened:** none.
**Questions resolved:** none.

---
*Gus replay complete. 10 ingests (G1–G10), 2026-04-20 → 2026-04-22. 50 commits across student scaffolding, scaling, eval infrastructure, LAMIR-1 rollout, and interpretability. Final state: v3_consistency_10000g at 0.551 Q-pt regret; q-bootstrap-belief 0.655 best look-ahead; belief at Bayes ceiling 39.2%; direct π_me is the deployable player.*

---

## [2026-04-25 | 063fcac | Burl 2000-decision harvest — Phase A guards land, v1 truncation bug caught, v2 ships clean]

One landed commit (`063fcac` Phase A guards) plus an in-session research milestone: the 2000-decision Burl batched-MLX harvest that the project has been working toward through the Burl iter-0 → iter-5 → wax_museum chain. Two corpus runs (v1 contaminated, v2 clean), one decision page locked in, one engineering pattern documented.

**Touched pages:** [[entities/burl]] [[entities/wax-museum]] [[topics/star]] [[topics/batched-harvest-resilience]] [[decisions/max-tokens-2048-floor]] [[experiments/burl-2000-harvest]] [[sources/063fcac]]

**Added:** 4 pages — 1 source digest (063fcac), 1 experiment, 1 topic, 1 decision.

**Updated:**
- [[entities/burl]] — Phase A guards section + 2000-decision harvest section + new open question on McNemar parity test.
- [[entities/wax-museum]] — Phase A guards section + 2000-decision harvest section.
- [[topics/star]] — Burl 2000-decision corpus ready section + recipe lessons from prior collapses.

**Frontier shift:**

- **Phase A guards land (063fcac).** Two safety nets on [[wax-museum]]'s `run_decision_waxed`: (1) turn-budget extension on engine reject (max +6 turns, capped at 3 extensions); (2) forced-commit fallback on turn-cap (highest-E[Q] from probed plays → oracle scan → first legal). `max_turns_extensions` and `forced_commit*` fields added to the trace. Pre-Phase-A blunder rerun: 3/29 decisions committed illegally. Post-Phase-A on the v2 2000-decision harvest: **zero illegal commits**, 219 / 2000 forced-commits, 733 budget extensions across the run (max 3 per decision — guards fire routinely, stay bounded).
- **v1 contamination — turn-1 truncation bug.** First batched 2000-decision attempt (`harvest_batched_20260425_031033`, `max_tokens=1024`) SIGKILLed at 1398/2000. Bucket parity vs sequential 560 looked plausible at distribution level (within ±5pp everywhere). Three parallel investigation agents reproduced the actual data: **11.4% of v1 decisions had `belief_called_turns` not starting at turn 1** because the model had exhausted its 1024-token budget mid-thinking on turn 1 and never reached [[belief-trajectory]]. Per-decision audit on the sequential overlap: **43% of decisions had moved between buckets**. Aggregate parity was a false pass — distribution-level errors cancelled. **Methodological lesson: distribution-level parity gates can hide systematic per-decision regressions when the regression has multiple compensating directions.** The lesson generalizes to any future "matches in aggregate" claim about model behavior.
- **max_tokens=2048 locked as the floor (decision).** Sequential's p99 turn was 1770 chars (~600 tok); max 2639 chars (~900 tok). 1024 was below the natural distribution. 2048 covers it with comfortable margin and sequential's max with 9× headroom. 4096+ unnecessary; KV-cache cost not justified by the data. See [[decisions/max-tokens-2048-floor]].
- **v2 clean run (2026-04-25 13:15)**: 5h 46m wall, batch=6, `max_tokens=2048`, `D_required_first` variant, fresh seeds from `gus/data/corpus_train_chunk_0-99.pt`. **Zero quarantine fires** across 333 batches. **0% truncation-at-cap** vs v1's 1.2%. **0/2000 belief-not-starting-turn-1** vs v1's 11.4%. Bucket distribution within ±2pp of sequential 560 everywhere. The −2.4pp `BURL_BREAKS_CONSENSUS` decline (17.3% → 14.9%) is directionally correct (truncation was inflating that bucket); residual delta is a candidate for a paired McNemar test on the 560 overlap.
- **Corpus ready for STaR run-3.** Strict pool: 1062 rows (52.5% of corpus, 3.6× the sequential 294). Non-trivial gold: 202 rows (excluding `ALL_AGREE_CORRECT`), 3.9× the prior 52. Sharpest [[r1-rationalization]] target: 299 rows of `BURL_BREAKS_CONSENSUS` (3.1× the prior 97). See [[experiments/burl-2000-harvest]].
- **Engineering pattern: per-wave OOM resilience layer** ([[topics/batched-harvest-resilience]]). OOM classifier (`metal::malloc`, `Resource limit`, `Resource exhausted`, `broadcast_shapes`, `out of memory`, `MemoryError`) → quarantine.jsonl ledger; SIGKILL recovery via `wave_in_progress.txt` sentinel cleared after each wave; `--rerun-quarantined` retry pass. Smoke-tested with `--inject-oom-at-wave N` (4 decisions resolved on retry). Dead code on the v2 success path — its existence is the entire point. Turns multi-hour batched harvest from "one failure or restart" into "one failure or retry six decisions."

**Questions opened:**

- Does v2's residual −2.4pp gap on `BURL_BREAKS_CONSENSUS` (vs sequential 560) reflect real batched-mode policy drift or sampling noise? A paired McNemar test on the 560 overlap (paired by aligned `(seed, declaration, narrator_seat, legal_plays)` tuples) would settle it; not yet run.

**Questions resolved:** none.

**Side artifact:** `scratch/belief_trajectory_rollout/HARVEST_REVIEW.html` (engineer audience) and `HARVEST_REVIEW_FAMILY.html` (intelligent-but-not-ML-engineer audience), bundled to https://burl-42-review.pages.dev as a one-off Cloudflare Pages publish for the 42-playing family. Both pages are static, regenerable from `corpus_index.jsonl` + `trace_summary.json` files via `build_review.py` / `build_review_family.py`.

**Near-miss (2026-04-25):** Run-3 scout caught a pre-built min-300 corpus that had been silently sourced from `harvest_20260424_133611` (the held-out eval set), not the 2000-decision harvest. Quarantined as `..._FROM_HELD_OUT_EVAL_DO_NOT_TRAIN`; replacement built from `harvest_batched_20260425_072910` (manifest-verified). Rule "trust the manifest, not the prose" added to [[star]] (recipe lesson #4) and [[burl-2000-harvest]] (Footgun caught section). [[AGENTS]] lint sweep now also flags pre-existing-artifact claims for verification.

---

## [2026-04-25 | 06bf5bf | Burl STaR run-3 attempt — three launch failures, val curve captured, no adapter]

Filter-only STaR on the 2000-harvest strict pool launched on 2026-04-25 afternoon. Three attempts, three failures, no adapter on disk. The third attempt's val curve (2.354 → 0.302 over 9 evals before crash at iter 487/1343) is the result that was filed.

**Touched pages:** [[topics/star]] [[experiments/burl-star-run3]] [[index]]

**Added:** none (skeleton already at `02d9096`).

**Updated:**
- [[experiments/burl-star-run3]] — Training, Eval, Verdict, What's next, Pointers all filled. Three failure-mode subsections describe what died and why. Last attempt's quarantined log path called out for forensic re-read.
- [[topics/star]] — recipe lessons extended to **#5 (trust the argparse, not the prose)** generalizing the manifest-vs-prose rule to source-vs-doc, and **#6 (detach the trainer; resumable checkpoints; catch-all snapshot save)** capturing the three orthogonal blockers run-3b must clear.
- `burl/STAR_RUN3_PLAN.md` — §Step 2 launch command rewritten with the three flag-name fixes, the multiplicative `1.02` early-stop fix, the `.venv/bin/python` invocation, and the `nohup setsid` detachment. New §Postmortem section names the three blockers.
- [[index]] — burl-star-run3 line updated with the OOM verdict.

**Frontier shift:**
- The recipe was working when the environment killed it. Val loss dropped 7× (2.354 → 0.302) over 9 evals, no collapse signal, no early-stop trip, no divergence — but no adapter ever materialized because (a) the trainer's exception handler only catches a local `EarlyStopRequested`, (b) `mlx_lm.Trainer`'s `steps_per_save` is set to `10**9` (effectively off), and (c) the in-memory best-checkpoint snapshot is only serialized at end-of-run.
- **Three blockers are now named and gating run-3b**: resumable checkpoints (priority — see project memory `project_resumable_training_priority.md`), generic-exception catch-and-save in `train_mlx`, and reduced peak-memory headroom (try `--max-seq-length 4096 → 2048` first; corpus's p99 is ~600 tok per [[max-tokens-2048-floor]]).
- **Two new recipe-lesson generalizations land on [[star]]**: "trust the argparse, not the prose" extends the manifest-vs-prose rule to source-code-vs-documentation; the new lesson #6 packages detachment + resumability + catch-all-save as a single discipline for any future Burl/forge training.

**Questions opened:**
- Is the MLX OOM at iter 487 reproducible (suggesting a memory leak in the cosine LR scheduler at version 0.31.2), or one-shot (suggesting a peak-memory misalignment with the 4096 max-seq-length default)? Re-launching with the same recipe at `--max-seq-length 2048` answers both.
- Is the val curve's trajectory (0.302 at iter 450, concave-down) on track for a useful adapter at convergence (~iter 1343), or is the apparent improvement a curve-fitting artifact of the small corpus? Only a complete run answers this.

**Questions resolved:** none.

**Quarantined adapter dirs (gitignored under `scratch/`):**
- `run3_20260425_144858_DIED_AT_BASELINE_VAL_PARENT_SHELL_KILLED/`
- `run3_20260425_145914_DIED_BARE_PYTHON_NO_MLX/`
- `run3_20260425_150538_FAILED_MLX_OOM_AT_ITER_487/` (last one's `train.log` has the full val-loss curve and the MLX traceback)

---

## [2026-04-25 | 47f2d85 | Burl STaR run-3b + 3c — preserve-thoughts is load-bearing; recovers thought-block emission 0% → 95%]

Two trained adapters and an in-progress eval. Run-3b (default = preserve-thoughts OFF) and run-3c (preserve-thoughts ON, only delta) form the cleanest A/B in the project to date on whether retaining thought tokens at SFT actually changes inference behavior. Run-3b's adapter learned to skip reasoning entirely; run-3c's adapter emits thoughts on ~95% of decisions and improves on bot-match and |Δ|. The recipe-lesson is filed; the experiment page now reflects the success.

**Touched pages:** [[experiments/burl-star-run3]] [[topics/star]] [[topics/preserve-thoughts]] [[index]]

**Added:** none.

**Updated:**
- [[experiments/burl-star-run3]] — fully rewritten as the run-3 → run-3b → run-3c story. Three failed launch attempts kept as the run-3 history; run-3b and run-3c training tables and adapter pointers added; partial-eval results filed for both; verdict reframed around the 0% → 95% thought-block-emission swing; prediction-vs-reality table appended.
- [[topics/star]] — recipe lesson #7 added: `--preserve-thoughts` is load-bearing, not a tuning knob; defaults flips whether the model thinks at all.
- [[topics/preserve-thoughts]] — `status: re-opened` → `confirmed`; new "Result: confirmed (run-3c, 2026-04-25)" section with run-3b vs run-3c table and within-run-3c thinking-vs-no-thought split.
- [[index]] — burl-star-run3 hook updated; preserve-thoughts hook updated to reflect the confirmation.

**Frontier shift:**
- **Preserve-thoughts is a phase change, not a knob.** Run-3b: 0% thought presence at inference, 60.2% bot-match, |Δ| 2.89 (n=113 partial eval). Run-3c: **95.9% thought presence, 66.6% bot-match, |Δ| 2.11, mean signed Δ −1.98 (n=560 full eval)**. Same corpus, same recipe, same early-stop, same eval. The Gemma 4 chat template runs `strip_thinking()` before tokenization unless the trainer opts out via `--preserve-thoughts`; without that opt-out, the LoRA learns to skip the reasoning channel.
- **A 0.27% LoRA can flip whether the model thinks at all.** Run-3c emits thoughts on 95.9% of decisions vs run-3b's 0%. The within-3c thought-vs-no-thought split tightened as the eval grew (from ~8pp at n=304 to ~1.5pp at n=560 on bot-match) — the early signal that "thinking helps" was real but smaller than first-look. The signal generalizes the [[iter5-e1-rank-sweep]] +3.3pp result at N=26 to a 95-pp swing on thought-block presence at N=560.
- **Eval-as-designed answers "does it think" but not "does it play well."** Bot-match alone undersells the picture: run-3c's 66.6% is mostly the 373 ties; among the 187 disagreements with the bot, the adapter splits **22 wins / 165 losses (7.5:1 lossy)** with mean signed Δ = −1.98. This is the [[regret-eval]] reframe applied to Burl evaluation. Three eval upgrades now gate the next-iteration decision: (a) same-harness same-seeds base-model eval, (b) report mean signed Δ alongside |Δ|, (c) oracle-relative regret. Until those land, "did STaR-iter-3c improve play quality" cannot be answered cleanly.
- **The val-loss penalty for preserve-thoughts is small.** Run-3c best val 0.268 vs run-3b's 0.238 (~13% higher) and 13 more wall-clock minutes — the predicted 0.5–0.9 range was too pessimistic. Predicting thought tokens is harder than predicting tool-call patterns, but not nearly as much harder as expected.
- **Run-3 (the OOM crash) joins the recipe-lesson canon as historical context only.** Resumable checkpointing and generic-exception catch-all save are still unaddressed; run-3b/3c happened to survive without them.

**Questions opened:**
- Why does run-3c skip thinking on ~5% of decisions? Likely candidates: forced-commit fallback rows, multi-turn corrections, or a corpus-side minority pattern where Burl committed without thought blocks.
- Would loss-weighting on thought tokens (up-weight thoughts vs tool-call tail) further improve coverage and match rate, or saturate?

**Questions resolved:**
- "Does preserve-thoughts actually change adapter behavior once truncation is fixed?" — yes, dramatically and at scale (run-3c). The iter-4 byte-identical-weights result remains attributed to the SFTConfig truncation confound.

**Pre-launch prediction artifact (kept for the loop):** `scratch/belief_trajectory_rollout/star/run3c_prediction.md` — the project under-estimated preserve-thoughts in two consistent directions: the val-loss gap was smaller than expected, and the thought-block coverage was much higher than expected. See [[burl-star-run3]] §"Prediction vs reality".

**Live snapshot used by archivist:** `scratch/belief_trajectory_rollout/star/RUN3BC_LIVE_SNAPSHOT.md`. Run-3c eval completed at n=560 (4.1h wall) after the snapshot; final numbers folded in by team-lead before this commit.

---

## [2026-04-26 | fbe798f | Burl STaR run-3b/3c — STaR-shaped post-hoc rescore lands]

Closes the eval-side gap the prior entry flagged: the run-3c eval was good on bot-match and thought-block presence but couldn't answer "does the adapter actually play better Texas 42 than naked-Burl?" The new rescorer (`scratch/belief_trajectory_rollout/star/star_eval_report.py`) joins the existing per-decision eval files to the harvest's `corpus_index_k200.jsonl` and the diagnostic `per_decision_eval_k200.jsonl` to compute STaR-shaped metrics — `k1_pass`, `signed_delta`, `oracle_regret`, re-bucketed adapter-vs-base classification, and per-base-bucket flip matrix — without rerunning inference.

**Touched pages:** [[experiments/burl-star-run3]]

**Frontier shift:**
- **Run-3c is the first preserve-thoughts adapter to beat naked-Burl on oracle regret on the held-out 560.** Mean oracle regret 2.295 (base) → 2.165 (run-3c), a 5.7% relative reduction. The seven Burl-loss buckets shrink 213 → 144 in count (32% relative). On the harness-independent regret axis, the adapter is on the right side of the line.
- **K=1 pass rate (Δ ≥ 0) for run-3c is 70.5% vs run-3b's 61.5%.** Run-3c yields ~395 surviving decisions for harvest-2 carry-forward; run-3b yields ~80. Run-3c is the carry-forward adapter; run-3b is dominated on every STaR-shaped metric (regret 3.02 vs 2.17, signedΔ −2.92 vs −1.98).
- **The cost of run-3c's regret win is forced-commit inflation: 12.3% (base) → 34.1% (adapter).** 122 additional decisions where the harness has to pick for the model after the adapter fails to terminate cleanly. Most of this comes out of `ALL_AGREE_CORRECT` (68 of 242 base AAC decisions go to FORCED_COMMIT under the adapter). The forced-commit pick is usually the highest-E[Q] probed play, so match_oracle stays high — but the decision shape is degraded vs a clean Burl commit.
- **Bucket flip matrix is reproducible.** Saved per-eval as `star_rescore.json` alongside each eval dir, with `match_by_base_bucket`, `bucket_flip_matrix`, and full per-row JSONL. Future runs can drop into the same rescorer without changing the eval harness.
- **FORCED_COMMIT inflation is NOT a `--preserve-thoughts` artifact.** Follow-up diagnosis (`scratch/belief_trajectory_rollout/star/FORCED_COMMIT_DIAGNOSIS_2026-04-26.md`): run-3b (no thoughts) and run-3c (with thoughts) force at the same ~33% rate. The cause is the LoRA fine-tune itself — both adapters lose the "now is the time to commit" discrimination signal because the strict-pool training corpus contained, by construction, only successful Burl commits. 134 of 158 newly-forced decisions hit exactly n_turns=8 (the default cap); 0 bailed; 0 degenerate. The adapter is doing real strategic exploration but never narrowing.

**Questions resolved (this entry):**
- "Did run-3c actually improve play quality, separate from reasoning emission?" — yes, modestly (5.7% relative regret reduction on the same held-out 560). The bucket-shape evidence is consistent: every Burl-loss bucket shrinks under the adapter, and the only bucket that *grows* is FORCED_COMMIT (a known harness-side fallback, not a training-side regression).
- "What's driving the 122 new forced commits?" — turn-cap exhaustion combined with lost commit-discipline signal. Recommendation for harvest-2: bump rollout turn cap from 8 → 12 and smoke 50 decisions; if force rate doesn't drop to ≤20%, augment training corpus with negative-commit-discipline rows before launching the full harvest.

**Questions still open:**
- Are the BURL_BREAKS_CONSENSUS reductions (95 → 66, with 20 flipping all the way to ALL_AGREE_CORRECT) generalizing to held-out, or are they a function of training corpus overlap with this specific 560? Confirming requires a fresh-seeds harvest; that's harvest-2 and run-4.

**Reads-only rescore. No GPU. No retraining. Adapter weights unchanged.**

**Artifacts:**
- New: `scratch/belief_trajectory_rollout/star/star_eval_report.py`
- New: `scratch/belief_trajectory_rollout/star/STAR_EVAL_REPORT_2026-04-26.md`
- New: `scratch/belief_trajectory_rollout/star/eval/run3b_eval_seq560_20260425_164239/star_rescore.json` + `star_rescore_rows.jsonl`
- New: `scratch/belief_trajectory_rollout/star/eval/run3c_eval_seq560_20260425_181016/star_rescore.json` + `star_rescore_rows.jsonl`
- New: `scratch/belief_trajectory_rollout/star/FORCED_COMMIT_DIAGNOSIS_2026-04-26.md` (Q1/Q2/Q3 forensic walk + spot-check transcripts + harvest-2 recommendation)
- Updated: `wiki/experiments/burl-star-run3.md` ("What's next" section replaced; per-bucket regret movement added; FORCED_COMMIT diagnosis subsection added; pointers list extended).
- New: `wiki/topics/commit-discipline-collapse.md` — names the failure mode for the canon. Backlinks to [[burl-star-run3]], [[k1-grading]], [[ls-mixture]], [[commit-discipline]], [[primer-tradeoff]], [[preserve-thoughts]].
- Updated: `wiki/topics/regret-eval.md` — new "Ported to Burl evaluation" section + index hook updated. The gus-side regret framework (oracle_best_eq − student_chosen_eq) now scores Burl adapters as well as gus students. Cross-links to [[commit-discipline-collapse]] (the FORCED_COMMIT decision-shape-not-quality refinement was discovered by the regret reframe).

## [2026-04-25 | add6a2a | Resumable checkpointing + crash-snapshot save lands on `star_mlx.py`]

Closes the priority blocker named in [[burl-star-run3]] §"What's next" #4 + #5 and in [[star]] recipe-lessons §6. The run-3 OOM lost a healthy adapter (val 0.302 @ iter 487) because `train_mlx` only caught `_EarlyStopSignal`. Three changes land:

- `--steps-per-checkpoint N` (default 100): every Nth iter, the best-val snapshot (or current weights if no eval has fired) is atomic-written to `{adapter-out}/checkpoint_iter{N}/adapters.safetensors` and mirrored at `{adapter-out}/adapters.safetensors`, with `checkpoint_state.json` recording iter + best-val tracker.
- `--resume`: reads `checkpoint_state.json`, calls `model.load_weights(...)` after `linear_to_lora_layers` is applied, restores the best-val tracker, sets `iter_offset` so trajectory step axes stay continuous, and trims `total_iters` by the resumed count. LR schedule restarts from zero on the trimmed budget — crash-recovery, not bit-perfect continuation.
- Generic-exception catch in `train_mlx`: any non-`_EarlyStopSignal` exception triggers `_save_crash_snapshot` to `{adapter-out}/best_on_crash/adapters.safetensors` + `crash_info.json` (including `iter_when_crashed`, `exc_type`, `exc_msg`, traceback) before re-raising.

**Touched pages:** [[decisions/resumable-checkpointing]] [[experiments/burl-star-run3]] [[topics/star]] [[index]]

**Added:** [[decisions/resumable-checkpointing]]

**Updated:**
- [[topics/star]] §"Burl 2000-decision corpus ready" recipe-lesson #6 marked resolved with forward link to [[decisions/resumable-checkpointing]].
- [[experiments/burl-star-run3]] §"Training-side follow-ups" #4 (resumable) and #5 (generic-exception) marked resolved.
- [[index]] decisions section gained the new page.

**Tested:** 12 new unit tests in `burl/train/test_star_mlx.py` (33/33 pass). End-to-end: 50-row corpus, killed at iter 30, relaunched with `--resume`; verified prior checkpoints land at iters 10/20/30, `checkpoint_state.json` records `best_val_loss=1.10 @ iter 24` from the killed run, resume picks up with `iter_offset=30` and trims remaining iters. Smoke artifacts under `scratch/resume_smoke/`.

**Frontier shift:** Future Burl/STaR runs (the cluster of pending tasks numbered #5/#6/#7 on the team's task list — harvest-2, run-4 train, run-4 eval) inherit crash-recovery for free. The run-3 attempt-3 metal-OOM at iter 487/1343 would now leave a recoverable adapter on disk in three places (latest periodic checkpoint, `adapters.safetensors` mirror, `best_on_crash/adapters.safetensors`). The cost is ~one disk write per N iters and a tiny CPU spike on the checkpoint hook.

---

## [2026-04-26 | 86334b8 | batched eval port + resumability hardening on eval_adapter_smoke.py]

Eval-side companion to the same-day [[decisions/resumable-checkpointing|trainer resumability]] work. Held-out adapter eval was 4h sequential (run-3c attempt 1, `run3c_eval_seq560_20260425_181016/`); ported to batched lockstep via `GemmaLocalNativeBatched` (proven by [[burl-2000-harvest|harvest-batched]]) with full crash-recovery.

**Touched pages:** [[experiments/burl-star-run3]] [[topics/batched-eval-resilience]] [[index]]

**Added:** [[topics/batched-eval-resilience]]

**Updated:**
- [[experiments/burl-star-run3]] §"Recipe" eval line (now documents `--batch-size 6 --max-tokens 8192 --resume-dir`) and §"Pointers" eval-harness line (full flag list).
- [[index]] topics catalog gained the new page.

**Tested:** Smoke (n=6 batch=6 vs run-3c sequential reference): wall=137.6s, match=2/6 vs reference 3/6 (drift = 1, within spec). Resume sanity (n=12 batch=6, kill@6-of-12): `resume: 6 already done, 6 TODO` — exactly correct, final summary.json has all 12 in gi order. Live n=180 base eval running on hardened code (rescoped from n=560 to fit overnight chain).

**Frontier shift:** Eval cycle goes from "all-or-nothing 4h" to "lose ≤3 min on any failure" — same cadence improvement the [[batched-harvest-resilience]] layer delivered for the harvest pipeline. Combined with [[decisions/resumable-checkpointing]], the train→eval→retag iteration loop is now bracket-resilient end-to-end. The 5h n=560 ETA wouldn't have fit before the user's wake; the resilience layer made the team-lead's mid-flight rescope to n=180 cost-free (kept the 18 already-done decisions via `--resume-dir`).

## [2026-04-26 | fc8f2dd | Resumable trainer hardened: standalone-loadable checkpoints, --resume-from PATH, OOM injection test]

Lifts the resumable trainer to the ULTRA-CRITICAL bar after team-lead flagged
data-loss survivability as the team's top priority and noted the prior add6a2a
commit had a correctness gap.

**The bug add6a2a left:** every periodic checkpoint dir contained
`adapters.safetensors` but no `adapter_config.json`. mlx-lm's
`load_adapters` requires the config (it reads `num_layers` +
`lora_parameters` for the LoRA layer reconstruction), so a crash mid-run
would have left dirs that `mlx_lm.load(adapter_path=...)` couldn't open
without manual config-copy. Found by re-reading the team-lead's
"checkpoints must be REAL — every N steps a complete adapter write that
can be loaded standalone" instruction.

**Touched pages:** [[decisions/resumable-checkpointing]]

**Updated:** policy page rewrites mechanism + tradeoffs + tested
sections to reflect hardening; how-to-use section adds `--resume-from
PATH` recipe for the dir-fork case.

**Concretely:**
- Each periodic write now produces `{adapters.safetensors,
  adapter_config.json}` in BOTH the snapshot dir and the top-level
  mirror, via a shared `_build_adapter_config` skeleton +
  `_enrich_adapter_config` per-write layering.
- `_save_crash_snapshot` does the same for `best_on_crash/` (so the
  crash dir loads standalone too).
- `--resume-from PATH` lands as the explicit "fork from another dir"
  form alongside `--resume`.
- New `test_train_mlx_metal_oom_re_raises_after_persisting`
  monkey-patches `mlx_lm.tuner.trainer.train` to raise the EXACT
  signature from run-3 attempt 3 (`RuntimeError("[metal::malloc]
  Resource limit (...) exceeded")`) and asserts `train_mlx` re-raises
  while leaving `best_on_crash/` loadable via
  `mlx_lm.load(adapter_path=...)`.

**Tested:** 43/43 unit tests pass (was 21, now includes 22 new — +13
for this hardening pass). End-to-end `kill -9` + `--resume` validated
on 50-row corpus: pre-kill best (val 1.163 @ iter 24) preserved across
SIGKILL, resumed iter-1 val (1.163) matches, new best lands at val
0.7955 @ iter 49 mid-resume, final adapter loads standalone.
`--resume-from` dir-fork validated separately. Wall overhead measured
at <0% (within run-to-run noise) for `--steps-per-checkpoint 5` over
30 iters; default of 100 is comfortably under the 5% bar.

**Frontier shift:** Run-4 (#6) and any subsequent training run inherits
crash-survivability that the team-lead emphasized was "a huge part of
the OOM puzzle." The run-3 attempt-3 metal-OOM at iter 487/1343 would
now leave a recoverable adapter on disk in five places: latest
periodic checkpoint dir, prior periodic checkpoint dir, top-level
mirror, `best_on_crash/`, and any explicit `--resume-from` fork.

**Receipts:** `scratch/resume_smoke/TRANSCRIPT.md` documents the full
SIGKILL+resume transcript with the standalone-loadability checks and
overhead measurements.

---

## [2026-04-26 | fbe798f | In-distribution paired n=180 base eval lands — run-3c regret −39% vs naked-Burl]

The eval-side gap that was open all morning closes here: eval-speeder's batched n=180 base-model eval lands cleanly through the same harness as run-3c. Rescorer immediately folds the paired comparison into the report.

**Touched pages:** [[experiments/burl-star-run3]] [[topics/regret-eval]] [[topics/preserve-thoughts]] [[index]]

**Frontier shift:**
- **Run-3c beats naked-Burl on oracle regret by 39% on the cleanest comparison available.** Paired n=180 (gi 0..179, same batched harness): regret 1.92 vs 3.13 (−1.22 absolute, −39% relative). The cross-harness 5.7% number from earlier in the day understated the win by 7×.
- **Per-bucket: the adapter wins on every major bucket including AAC.** AAC −0.36 (cross-harness "AAC tax" was a harness artifact, not the adapter), BIW −6.21, BOTH_FIX −5.73, FORCED_COMMIT −3.15, AAW −1.67, BBC −0.48. Minor losses on small buckets (BPQ +0.73, BAF +0.20, QAF +1.10, BDP +7.08) are on n ≤ 3 each and noise-dominated.
- **`BURL_BREAKS_CONSENSUS` shrinks 45 → 23 (51% reduction)** in the paired n=180; `BURL_INDEPENDENT_WRONG` shrinks 17 → 4 (76% reduction). Adapter shifts these into FORCED_COMMIT (+41) but the FORCED regret is 0.19 vs the original BBC/BIW regrets of 5.24/7.46 — play quality improves dramatically even though decision shape shifts. Consistent with [[commit-discipline-collapse]]'s "decision-shape cost not play-quality cost" framing.
- **The user's "did first STaR improve reasoning?" question now has a clean answer: yes, by 39% on the regret metric.** Load-bearing result for the carry-forward decision; validates the run-3c → harvest-2 → run-4 pipeline plan.

**Questions resolved:**
- "Is the adapter actually a better Texas 42 player than naked-Burl, controlling for harness?" — yes, decisively. The 39% in-distribution paired-regret reduction is much larger than the cross-harness 5.7%; the harness-difference confound was the source of the conservative initial estimate.
- "Does the cross-harness 'AAC tax' (+0.66 regret on already-correct decisions) survive in-distribution?" — no, it reverses to a slight win (−0.36). The tax was a harness artifact, not adapter-induced.

**Artifacts:**
- New: `scratch/belief_trajectory_rollout/star/eval/base_eval_batched_n560_20260426_005754/star_rescore.json` + rows.jsonl (in-distribution n=180 base rescore)
- New: `scratch/belief_trajectory_rollout/star/base_vs_run3c_paired_n180.md` (one-shot paired comparison fragment from `fold_base_into_report.py`)
- Updated: `scratch/belief_trajectory_rollout/star/STAR_EVAL_REPORT_2026-04-26.md` (final paired n=180 triangle replaces the early-signal table; per-bucket regret-delta + bucket distribution shift sections added)
- Updated: `wiki/experiments/burl-star-run3.md` (paired n=180 result inserted in §"STaR-shaped rescore", supersedes the cross-harness pending note)
- Updated: `wiki/topics/regret-eval.md` ("Burl-side reading" section now leads with the paired n=180 result)
- Updated: `wiki/index.md` (burl-star-run3 + regret-eval hooks updated)

## [2026-04-26 | fbe798f | Promote Burl STaR eval/rescore + corpus ops out of scratch]

The load-bearing pure-Python parts of the run-3/run-4 loop now have tracked
homes while the cap-12 harvest continues undisturbed on its original scratch
runner.

**Touched pages:** [[experiments/burl-star-run3]] [[topics/regret-eval]] [[topics/batched-eval-resilience]]

**Frontier shift:**
- `burl/eval/star_metrics.py` promotes the STaR-shaped metric contract and
  oracle-regret rescore path: signed delta, K1 pass, oracle regret, near-tie
  rate, bucket flips, forced-commit counts, and real thought-block rate.
- `burl/eval/star_eval_report.py` is the tracked CLI wrapper. A run-3c smoke
  reproduced the existing regret/bucket metrics and corrected thought-block
  accounting to actual `thinking` events (`537/560 = 95.9%`) rather than
  `belief_trajectory` calls.
- `burl/train/star_corpus.py` promotes the filter-only corpus builder and
  per-turn row conversion; `burl/train/build_star_corpus.py` is the tracked
  CLI wrapper. A smoke rebuild from the 2026-04-25 harvest reproduced the
  known strict-min300 row shape: 2686 train rows + 665 val rows.
- The active harvest runner (`scratch/belief_trajectory_rollout/harvest_batched.py`)
  was not edited. The live cap-12 harvest can still resume against the exact
  scratch script surface it launched with.

**Tested:** `python -m pytest burl/eval/test_star_metrics.py burl/train/test_star_corpus.py -q`
passes (6 tests).

---

## [2026-04-26 | 74464e9 | perf-on-the-table — name the speedup landscape]

Forward-looking topic page after the user's calibration that Gemma 4 E2B is mobile-class (1334 tok/s benched) but our harness sits at ~70 tok/s effective — two orders of magnitude of pure software cost, not model capacity.

**Touched pages:** [[topics/perf-on-the-table]] [[index]]

**Added:** [[topics/perf-on-the-table]]

**Updated:**
- [[index]] topics catalog gained the new page.

**Frontier shift:** Names the six engineering levers (prefix sharing, continuous batching, turn-aware token budgets, speculative decoding, parallel tool calls, quantization) and ranks by ROI. Compounded top-three is ~7–10× on M5 Max alone with no model changes — would land harvest-2 in 30–45 min instead of the 5h overnight slot it took. Not research, just engineering: each lever is a 2–5 day sprint. The page is reference; the work itself is for next sessions. The 12h-iter regime selects against breadth experiments ([[backwards-curriculum]] scout, rank ablations, prompt sweeps) — the 1.5h-iter regime opens that door.

**Why now:** the live n=180 base eval had a 50-min wave dominated by two long-tail decisions (the sync-wave straggler tax) that the user noticed and used as the calibration moment. Capturing the landscape before the iteration cycles forget what slow felt like.

---

## [2026-04-26 | 74464e9 | iter-without-regression — name the milestone]

The user surfaced the framing during the overnight cycle: harvest-2 + run-4 is the **first Burl iteration that landed without a new bug, pathology, or regression**. Iter-0 baked in eq-shy, iter-1 collapsed commit discipline, iter-5 had the rank-128 cliff, the 71-row run loss-collapsed, run-3 burned three launches before run-3b adapter wrote. Run-4 cleared all of these — harvest survived 5h+, trainer survived a probability-of-resume crash, eval ran end-to-end, adapter loaded standalone.

**Touched pages:** [[topics/iter-without-regression]] [[index]]

**Added:** [[topics/iter-without-regression]]

**Updated:**
- [[index]] topics catalog gained the new page (under [[commit-discipline-collapse]]).

**Frontier shift:** Names a milestone that's easy to underclaim ("nothing broke") but is actually the foundation everything else needs. Lists the six infra prerequisites that quietly enabled it ([[batched-harvest-resilience]], [[batched-eval-resilience]], [[resumable-checkpointing]], [[preserve-thoughts]] defaulted ON, [[regret-eval]] ported to Burl, [[commit-discipline-collapse]] named and explained). Forward-implications: r1-rationalization on the 299 base-harvest BBC bucket, FORCED_COMMIT-as-negative corpus enrichment, backwards-curriculum ratchet to trick 5, capacity scaling experiments — all of these become *cleanly testable* against a stable iter-2 base. Calibration paragraph at the end: 4000 decisions vs Zeb's hundreds-of-thousands is two ratchets in on a workstream that needs many more; "no regression" ≠ "no plateau"; the milestone is **stability**, full stop.

**Why now:** the user explicitly flagged the framing during recharge mode ("this is the first Burl iteration that didn't introduce a bug or regression"). Capturing the celebratory-but-bounded read while the prior-iteration regressions are still concrete enough to enumerate.

---

## [2026-04-26 | 74464e9 | burl-harvest-2 + run-4 — first STaR self-sharpening test, plateau confirmed]

The overnight ingest of harvest-2 (run-3c-as-rollout) → run-4 train (filter-only, same recipe as run-3c) → run-4 eval (paired n=180, cap=12, three-way fold against base + run-3c). The first end-to-end iter-1 → iter-2 test for filter-only STaR on Burl.

**Touched pages:** [[experiments/burl-harvest-2]] [[index]]

**Added:** [[experiments/burl-harvest-2]] — harvest+train+eval ingest as one coherent page.

**Updated:**
- [[index]] experiments catalog gained the new page.
- `scratch/belief_trajectory_rollout/star/STAR_EVAL_REPORT_2026-04-26.md` extended with §"Run-4 fold — three-way paired n=180".

**Frontier shift:** Names the **iter-2 plateau** clearly. Filter-only STaR on a same-shape corpus from a sharper rollout policy (run-3c) does not compound — run-4 lands at regret 2.97 vs base's 3.13 (within noise) and regresses ~1.05 vs run-3c (1.92). What it does deliver: clean commit discipline (FC 9.4% vs run-3c's 33.9%), back to base-comparable. The two-axis Pareto run-3c surfaced is real: run-3c trades commit discipline for play quality; run-4 reverses the trade. Neither is strictly dominant. Bucket distribution at the harvest level was essentially identical (strict pool 1062→1075, +13 dec) — the structural shape of which decisions land in which bucket is fixed by the seed pool, not by the rollout policy. The initial ILLEGAL=28.6% harvest-yield regression read was later corrected: both harvests were `--limit 2000` runs against 2800-row chunks, and `tag_corpus.py` filled the unattempted `global_idx=2000..2799` rows as placeholder ILLEGAL. On attempted `global_idx < 2000`, harvest-2 has 2000 trace summaries and 0 ILLEGAL rows. Forward-implication: filter-only iter-N is a flat slope at this scale; the next move is **changing the loss target** ([[r1-rationalization]] on BBC, FORCED_COMMIT-as-negative), not iterating on the same shape.

**Why now:** run-4 eval landed in the closing window of the overnight session (PID 51999, 45.6 min wall, 180/180 batched). The closer directive was explicit: fold three-way, write harvest-2 with corrected framing (not the early "iter-2 wins" overclaim), capture the play-quality plateau honestly. The iter-without-regression milestone holds at the *behavioral* level, but does not hold at the *play-quality* level — naming both publicly closes the loop on the overnight cycle.

---

## [2026-04-27 | fbe798f | perf-sprint playbook promoted to wiki]

Codifies the sprint-1 lessons (the burl-perf overnight session, [[burl-perf-phase0]] → [[burl-perf-phase3]]) into a reusable playbook so future sprints start armed. The first sprint failed at the wrap step because the loop message didn't restate the goal and "wrap and write the digest" felt like discipline rather than rationalization. The playbook closes that off-ramp.

**Touched pages:** [[perf-sprint]] [[perf-sprint-loop]] [[perf-sprint-goal]] [[perf-sprint-levers]] [[perf-sprint-traps]] [[perf-sprint-history]] [[index]] [[AGENTS]]

**Added:**
- [[perf-sprint]] — entry point. Kickoff line, team shape, three explicit wrap conditions, mandatory rules.
- [[perf-sprint-loop]] — the /loop message verbatim. Restates goal every fire; names idle as a bug.
- [[perf-sprint-goal]] — `scratch/PERF_GOAL.md` template. Quality bar + paired protocol + wrap-conditions reference, re-readable in 30s.
- [[perf-sprint-levers]] — 8-lever ROI-ordered ladder. Closed-lever pre-conditions. PLE quant landmine (broken set + safe set).
- [[perf-sprint-traps]] — recipes for the two Phase 4 crashes (`dynamic_roll` broadcast, `assert play is not None`), cross-scribe contention detection, comparison-anchor footguns, the wrap-rationalization warning.
- [[perf-sprint-history]] — append-only post-mortems. Sprint 1 entry as canonical form.

**Updated:**
- [[AGENTS]] — `playbooks/` directory recognized; `playbook` added to the kind enum.
- [[index]] — new "Playbooks" section catalogs the six pages.

**Retired:** none.

**Questions opened:** none. The relevant open questions live in `questions/open.md` already; the playbook references them.

**Frontier shift:** the wiki now has a how-to layer alongside the what/why layers. Playbooks are the first kind that's neither historical (sources/, experiments/) nor reference (entities/, topics/, decisions/) but procedural — read this and execute. The expectation is that sprint 2 starts by reading [[perf-sprint]] and ends by appending to [[perf-sprint-history]].

**Why now:** the user, post-sprint-1 wake-up, explicitly asked for promotion to wiki ("in-wiki is more first-class than project root"). The lessons were freshest now. Codifying late would have lost the texture of the failure modes (wrap rationalization, contention detection signals, the bench's hardcoded misleading log line).

## [2026-04-27 | unstaged | perf-sprint playbook simplified]

**Touched pages:** [[perf-sprint]] [[perf-sprint-loop]] [[perf-sprint-goal]] [[perf-sprint-levers]] [[perf-sprint-traps]] [[perf-sprint-history]] [[perf-on-the-table]] [[index]]

**Why:** the original promotion (fbe798f) over-specified the playbook with sprint-1 trauma laced through every page — wrap-condition predicates, mandatory rules, scribe team shape, "user is recharging" framing in the goal template, the 3.4× noise-floor claim repeated as doctrine, dead backlinks to phase pages that were never written. An implementer reading it at minute zero would hit conflicting instructions ("3.4× is the floor" vs "1.5× variance = contention"), unsatisfiable kickoff steps (read the doc and every doc it links to, half of which don't exist), and mood framing that bakes a single session's circumstances into universal procedure.

**Updated:**
- [[perf-sprint]] — collapsed to the contract: make it faster, verify equivalence with wall/K1/regret, don't give up. Removed team shape, scribe roles, kickoff procedure, wrap-conditions list, mandatory rules. Added scope (Burl-style mlx-lm on Apple Silicon).
- [[perf-sprint-goal]] — stripped recharge framing. EQUIVALENCE BAR replaces QUALITY BAR. "Don't give up" lives here and in the loop, nowhere else.
- [[perf-sprint-loop]] — kept goal-restating + idle-is-a-bug + "don't give up" as the loop's job. Dropped the wrap-conditions recital and the deliberate-features commentary.
- [[perf-sprint-levers]] — replaced dead backlinks (`burl-perf-phase{1,2,3}`, `harvest-cohort-abstraction`) with inline facts. Trimmed the per-lever budget paragraph.
- [[perf-sprint-traps]] — kept mechanical recipes (broadcast_shapes, AssertionError, hardcoded bf16 log line, comparison anchor). Removed the wrap-rationalization mood section. Folded clean-GPU contention floor (±4%, ~170 decode tok/s) into the contention detection signal.
- [[perf-sprint-history]] — sprint-1 trauma absorbed here as one entry: noise-floor misdiagnosis, wrap-rationalization, in-session retractions, what changed in the playbook.
- [[perf-on-the-table]] — entry-point description updated to match the new shape.
- [[index]] — playbook hooks rewritten.

**Frontier shift:** the playbook is now "clear goal + clear guidelines + trust the team" rather than predicate-and-procedure. Failures of past sprints are recorded as history, not as scaffolding on every page. "Don't give up" stays — it's the load-bearing instruction — but it lives in two places, in one voice.

## [2026-04-27 | unstaged | perf-sprint reshaped around Karpathy's autoresearch primitives]

**Touched pages:** [[perf-sprint]] [[perf-sprint-loop]] [[perf-sprint-goal]] [[perf-sprint-levers]] [[perf-sprint-traps]] [[perf-sprint-history]] [[index]]

**Why:** Karpathy's autoresearch (March 2026) collapses ML research to three primitives — one editable asset, one scalar metric, one time-boxed cycle with keep/discard — and gets clarity from aligning the solution to the problem. Same move applies here. The earlier simplification removed mood and dead backlinks but kept "wrap conditions" and "ledger row per lever" as procedural scaffolding. Replacing them with one metric (`wall_s_per_decision` on perf_subset_5, paired) plus a binary equivalence gate plus a TSV ledger structurally closes the wrap-checkbox loophole and makes every iteration's keep/discard decision unambiguous.

**Updated:**
- [[perf-sprint]] — leads with the contract paragraph (metric + gate). Adds editable-surface list (likely candidates: bench harnesses, eval harness, wax_museum, mlx-lm pin, batch flags) — non-strict, "go wild" with git as the safety net. Pins the TSV format (commit, wall_s, k1_match, regret_delta, peak_gb, status, description). The TSV is the digest.
- [[perf-sprint-loop]] — replaced loop message with a LOOP FOREVER block. Metric, gate, keep/discard rule stated together. "Don't give up" lives only here now (was in two places).
- [[perf-sprint-goal]] — collapsed to one paragraph: target, contract metric, equivalence gate values, ledger location.
- [[perf-sprint-levers]] — reframed intro: suggestions, not procedure. Hypothesis fuel for the loop.
- [[perf-sprint-traps]] — reframed intro: when the loop hits one of these, here's the smallest fix.
- [[perf-sprint-history]] — added one line: results.tsv is the truth, prose is texture.

**Design choices:**
- End-to-end metric — `wall_s` includes token generation, tool calls, harness overhead. Optimizing tool dispatch and prefix sharing both count.
- `aggregate_tok_s` deliberately left out of the TSV. Stick to one metric; learn the lay of the land; improvise if needed.
- Editable surface is a list of likely candidates, not strict. Trust the model + git.

**Frontier shift:** wrap conditions disappear structurally — the loop has two exits (metric hits target OR user typed stop). Paired protocol stops being a separate rule (it's step 3 of the loop). The sprint is interruptible — the user wakes up, reads the TSV, picks winners.

## [2026-04-27 | unstaged | perf-sprint context discipline + iteration-agent delegation]

**Touched pages:** [[perf-sprint]] [[perf-sprint-loop]] [[index]]

**Why:** sprint-1's team-of-scribes architecture wasn't actually about parallelism — it was about context isolation. Each scribe had its own fresh context to dump bench output, source reads, and tracebacks; the orchestrator stayed lean because it never saw the noise. The simplified single-threaded playbook lost that benefit and the orchestrator hits context-degradation thresholds 800k+ tokens deep into a long sprint. The fix isn't to bring back the team — it's to delegate each iteration (not each parallel scribe) to a fresh Agent. Single-threaded loop, isolated per-iteration contexts.

**Updated:**
- [[perf-sprint]] — added "Context discipline" section: orchestrator never reads bench output, source dumps, or tracebacks; each iteration delegates to a fresh Agent. Iteration agent contract: one coherent variant per spawn (co-required changes + obvious blocker fixes ship together); no side quests; wiki updates are the one sanctioned side effect; rigid return shape (one TSV row + 2 sentences). Includes a verbatim Agent spawn template.
- [[perf-sprint-loop]] — thinned orchestrator loop. Steps are now pick-variant, spawn-iteration, append-row, slack-update. The modify/run/parse/decide work moved into the iteration agent's context. "You do not read bench output, source dumps, or tracebacks" stated explicitly.

**Design choices:**
- "Coherent variant" not "atomic change." A stack of co-required knobs (e.g. lift batch ceiling + sweep batch + add prefill_batch_size=2) ships as one variant. Karpathy's program.md backs this implicitly — no atomicity rule.
- Wiki updates by iteration agents are deliberately encouraged, not just permitted. The wiki is the cross-iteration learning channel that closes the "slower cadence loses cross-iteration learning" critique of the delegation pattern.
- Return shape rigid (one row + 2 sentences) by contract. Forces summarization during the iteration, not after.

**Frontier shift:** the orchestrator is now structurally bounded in context cost. Each iteration costs ~2k tokens to the orchestrator (TSV row + brief note) regardless of how heavy the iteration was. 100 iterations = ~200k orchestrator context. The bench output, source dumps, and tracebacks live in iteration-agent contexts that die on return.

## [2026-04-27 | unstaged | perf-sprint sanctions web search for fast-moving deps]

**Touched pages:** [[perf-sprint]] [[perf-sprint-levers]] [[perf-sprint-traps]]

**Why:** Gemma 4 E2B is weeks old; mlx-lm ships frequently; spec-decode and continuous-batching state-of-the-art moves weekly. The iteration agent's default reflex is to reason from priors, which are stale by default in this domain. Trap recipes and lever notes also age — a workaround may already be fixed upstream; an "untested" lever may already be the upstream default. The fix is to explicitly sanction web search and tell the agent to reach for upstream changelogs, GitHub issue trackers, model cards, and recent papers when "common knowledge" is weeks old.

**Updated:**
- [[perf-sprint]] — added "Web search is sanctioned and encouraged" bullet to the iteration-agent contract. Spawn template now reminds the agent to load `WebSearch` / `WebFetch` via `ToolSearch` (deferred tools) and names the cases where primary sources beat priors.
- [[perf-sprint-levers]] — one-line note that lever notes age; web-search upstream before assuming "untested" or "could be already-applied" is current.
- [[perf-sprint-traps]] — one-line note that trap recipes age; web-search upstream changelog before spending an iteration on a workaround.

**Frontier shift:** the playbook now treats upstream web sources as a peer of the wiki's own pages. Wiki = synthesized state of the project; web = synthesized state of the deps. Both are primary sources from the iteration agent's perspective.

## [2026-04-27 | unstaged | perf-sprint emphasizes the /loop heartbeat as load-bearing]

**Touched pages:** [[perf-sprint]]

**Why:** observed in the wild — an orchestrator read the playbook, spawned one iteration agent, and stopped. The `/loop 10m` registration was buried in step 3 of "How to work" without flagging it as load-bearing, so the orchestrator skipped it. Without `/loop`, the orchestrator gets one turn and the sprint dies after the first iteration. This is the heartbeat that makes the architecture work.

**Updated:**
- [[perf-sprint]] — added a "Heartbeat" line to the contract paragraph (`/loop 10m` from [[perf-sprint-loop]], registered before the first iteration; load-bearing). Added a follow-up sentence to the kickoff section warning that the orchestrator's first job is to register the heartbeat. Reworded "How to work" step 3 to lead with "**Register the heartbeat**" and explain what registering does and doesn't do.

**Frontier shift:** none — this is a defect fix, not an architecture change. The architecture already required `/loop`; the playbook just didn't say so loudly enough.

## [2026-04-28 | unstaged | perf-sprint adopts orchestrator + backgrounded team-worker shape]

**Touched pages:** [[perf-sprint]] [[perf-sprint-loop]] [[perf-sprint-traps]] [[index]]

**Why:** the previous shape had the orchestrator block synchronously on each `Agent` spawn. While blocked, `/loop` couldn't fire — `/loop` only fires between orchestrator turns. So if a worker hung (mlx-lm wedge, OOM stuck process, bench infinite loop), the orchestrator hung with it and the heartbeat never lit up. The supervision was structurally dead exactly when needed most.

The fix uses Claude Code's team primitives (`TeamCreate`, `SendMessage`, `TaskStop`) to spawn the worker in background. `/loop` fires regardless of worker state and pings via `SendMessage`; if a worker hangs, the orchestrator can `TaskStop` and respawn. This keeps the architecture single-threaded (one worker at a time) but makes supervision real.

The "team" is a harness convenience — the team has exactly one member at a time. It's the addressability scope for `SendMessage` and the lifecycle handle for `TaskStop`, not a real team-of-agents architecture.

**Updated:**
- [[perf-sprint]] — replaced the "Heartbeat" line in the contract with an "Architecture" line that names the team primitive. Reworked "How to work" into a "Kickoff sequence" (load deferred tools → write goal → init TSV → TeamCreate → register /loop → spawn first worker) plus a "Steady state" block (driven by worker returns, not by /loop). Renamed "Context discipline" to "Architecture" and updated the rationale to name `run_in_background` as the mechanism that keeps supervision live. Spawn template now includes `team_name`, `name`, `run_in_background: true`, plus a SendMessage status hook in the prompt body.
- [[perf-sprint-loop]] — loop message rewritten as supervisory. Each fire: SendMessage worker for one-line status, slack-update, re-anchor. If silent for 2+ fires: SendMessage once more, TaskStop on no response, respawn fresh worker with same variant. Same variant wedges twice → crash row, pick different variant.
- [[perf-sprint-traps]] — added a "Stuck worker" section with the SendMessage / TaskStop / respawn recipe.
- [[index]] — playbook hooks updated.

**Design choices:**
- Worker drives iteration cadence; orchestrator spawns the next worker immediately on return, not on `/loop` fire. `/loop` is purely supervisory. Otherwise the loop interval becomes the iteration interval (10 min/iteration) which throws away the throughput.
- Same-variant wedges twice → crash + skip. Don't grind on a wedge variant indefinitely.
- Worker contract gains a SendMessage status hook (one-line response, no output paste) but is otherwise unchanged. Iteration agent contract still single-coherent-variant, no side quests, rigid TSV-row + 2-sentence return.

**Frontier shift:** the architecture now has explicit asynchronous primitives. The worker is non-blocking; the orchestrator can intervene on hangs; the supervision heartbeat is structurally guaranteed to fire.

## [2026-04-28 | unstaged | perf-sprint adds explicit worker-cleanup rule]

**Touched pages:** [[perf-sprint]] [[perf-sprint-loop]]

**Why:** observed during sprint 2 — backgrounded workers don't auto-release on return. Sessions persist until explicitly stopped, so over a multi-hour sprint with dozens of iterations, completed worker sessions accumulate and leak. The user flagged this after seeing the first two workers run.

**Updated:**
- [[perf-sprint]] — added "TaskStop the returned worker by name" to the steady-state on-return bullet, with a parenthetical explaining the leak. Added an end-of-sprint cleanup bullet (`TaskStop` active worker + `TeamDelete` the team). Added `TeamDelete` to the deferred-tool load list at kickoff. Added an Architecture paragraph explaining that workers don't auto-release and the orchestrator owns cleanup.
- [[perf-sprint-loop]] — added a CLEANUP step to the loop body: if any prior worker is in returned/idle state, `TaskStop` by name. Catches the case where the orchestrator missed cleanup between iterations.

**Frontier shift:** none — defect fix. The architecture already required explicit cleanup; the playbook didn't say so.

## [2026-04-28 | unstaged | perf-sprint-levers gains four mlx-lm-side levers]

**Touched pages:** [[perf-sprint-levers]]

**Why:** sprint 2 closed levers #1-#7 on the model+quant+kernel side and surfaced a fresh win on #6 (cohort abstraction MVP, iter 20). The model-side ladder is exhausted on M5 Max with current MLX/Gemma 4 stack; the harness-side ladder hadn't been written down. Per [[perf-on-the-table]] the bench is operating ~20× under raw mlx-lm capability, so the next factor-2 likely lives in the harness↔mlx-lm coupling, not in the model. Adding four levers to make that surface explicit and pickable.

**Updated:**
- [[perf-sprint-levers]] — appended rows #8–#11 to the active ladder. #8 prefill prefix-sharing (re-opens sprint 1's LRU-prompt-cache closure under its stated pre-condition; ~3–5% wall, cheap, low gate-risk). #9 CPU/GPU pipeline overlap audit (research-only first iter to map sync points; iter 10's PhaseTimer was GPU-only). #10 mlx-lm parallel-tool-call coupling (design pass first; constrained by wax_museum gate's one-call-per-turn rhythm). #11 Burl-aware continuous-batching dispatcher (STRUCTURAL flag only — multi-week project that belongs as its own sprint).

**Design choices:**
- Appended cleanly. Existing rows untouched, so worker descriptions that reference levers by number stay valid.
- Each new lever named "PROPOSED" or "STRUCTURAL" + first-iter shape (research-only / design-pass / iter-shippable) so spawned workers can scope their iteration before pulling.
- #11 explicitly marked "flag only — not for spawn picks." The active-ladder header already says "Suggestions, not procedure," but the multi-week scope is worth flagging at the row level.

**Frontier shift:** the playbook now distinguishes model-side and harness-side perf surfaces. Sprint 2 closed the model side; sprint 3 (or an extension of sprint 2 if it keeps running) has a written ladder of harness-side surfaces to climb.

## [2026-04-28 | unstaged | perf-sprint-levers gains tool-dispatch surface (#12, #13)]

**Touched pages:** [[perf-sprint-levers]]

**Why:** iter 10's `PhaseTimer` finding "tool dispatch hypothesis is dead — apply_step is just 4.78%" was correct on the *old* 68.9s sync-wave baseline but **inverts on the new 13.28s/decision continuous-batching floor**. Apply_step is 549ms × 6 turns = ~3.3s/decision regardless of model-side wall (tool dispatch is per-turn, not per-wall-time). At iter 14's floor that's ~25% of wall — the single biggest known overhead in the harness. Memory says tool execution is "sub-millisecond per call (table-cached)" — a ~1000× gap between expectation and measurement that demands diagnosis.

**Updated:**
- [[perf-sprint-levers]] — appended rows #12 and #13 to the active ladder. **#12** is a research-only diagnostic (mirrors iter 10's PhaseTimer pattern): wrap `_apply_step` in `burl/wax_museum/harness.py` with a sub-phase timer splitting into `chat_template_render`, `tool_call_inner`, `state_update`, `serialize_result`, `gate_validation`. Pure instrumentation; gate auto-passes; produces a fingerprint table. **#13** is a conditional fix that branches by what #12 reveals: (a) incremental chat-template caching, (b) gus non-cached path optimization, (c) avoid round-trip serialization, (d) pre-compute frozen-subset gus calls (bench-only, with explicit caveat that this is measurement separation, not production-translatable perf).

**Design choices:**
- Diagnostic + conditional fix is the same pattern that worked for iter 10 → iter 11. Keep it.
- #13(d) is explicitly bench-only with a "do NOT treat as production-translatable win" caveat — distinguishes "make the bench faster" from "make Burl faster" so the orchestrator doesn't game the metric without realizing it.
- The lever rows themselves cite the math: 4.78% of OLD wall = 25% of NEW wall, because tool dispatch is per-turn not per-wall-time. This is the kind of insight that ages without context, so it's encoded in the row text rather than just the log.

**Frontier shift:** none — the architecture and contract are unchanged. This corrects a stale conclusion ("tool dispatch is fine") that was correct on the old baseline and isn't on the new one.

## [2026-04-28 | unstaged | perf-sprint-levers gains five model+harness levers shaped for production scale]

**Touched pages:** [[perf-sprint-levers]]

**Why:** the user articulated production-scale framing ("hundreds of thousands or even a million+ decisions"; "thousands of turns would be a rounding error"). At that scale, ROI ranking changes — single-digit-% harness wins are dwarfed by levers that re-open closed surfaces (gate-passing quant, vocab pruning) because the savings compound across millions of forward passes. The lever ladder needs surfaces that haven't been varied yet so the next wave of iters has hypothesis fuel beyond just "scale cohort #6 wider."

**Updated:**
- [[perf-sprint-levers]] — appended rows #14–#18.
  - **#14 Runtime comparison probe** (research-only, cheapest first): mlx-lm vs llama.cpp vs mlc-llm vs candle on Gemma 4 E2B bf16. Theoretical bandwidth ceiling ~100–200 tok/s/stream; current 133–150 is ~50–60% of that, suggesting 1.5–2× headroom. Single afternoon's work; reshapes the rest of the ladder by establishing the actual ceiling.
  - **#15 KV cache quantization (int8)**: doesn't touch model weights so gi=0 quant-fragility doesn't fire. Cuts KV memory in half (pushes jetsam ceiling), reduces KV-fetch bandwidth (~5–10% wall). Cheap, low gate-risk, untouched surface.
  - **#16 lm_head vocab pruning / Burl-derivative checkpoint**: Burl's emit space is probably <2K unique tokens vs Gemma 4's 262K-vocab. Pruning lm_head → ~15–20% wall reduction. **At 1M+ decision scale, plausibly the single highest-value lever in the playbook** — weeks of M5 Max compute saved per full corpus run. Risks (Burl-only checkpoint, UNK semantics) are explicitly *features* at production scale because Burl is what runs.
  - **#17 Speculative prefill parallelism**: gate state deterministically dictates next-turn tool surface; speculatively prefill turn N+1 while decode N runs. ~3–5% wall stacked with #8.
  - **#18 Mixed-precision quantization with gi=0 protection** (multi-iter investigation): the gi=0 brittleness probably lives in specific layers, not all of them. Per-layer activation analysis → selective quantization preserving the brittle layer(s) at bf16. Re-opens previously-closed Q4/Q8 surface. Higher-risk multi-iter project but at scale, ~30–40% wall reduction with the gate intact dwarfs every other lever.

**Design choices:**
- Sequenced the rows by cost-of-discovery: #14 first (cheapest, most informational), #15 next (cheap win), #16 (structural Burl-derivative; needs distribution analysis first), #17 (harness-side stack with #8), #18 (multi-iter; sequence after #14 confirms runtime ceiling).
- **Production-scale framing baked into row descriptions**, not just the log. At 1M+ decisions, "Burl-only derivative checkpoint" is a feature; "weeks of compute saved" is a real metric. The lever rows make those tradeoffs explicit so future workers see the right ROI calculus.

**Frontier shift:** the playbook now has explicit hypothesis fuel for "what would close the gap to the bf16 ceiling" — runtime swap, model surgery, mixed-precision quant, KV quant, harness speculation. The model-side ladder isn't actually exhausted; it was exhausted *with respect to the surfaces tried*. Five new untouched surfaces are now visible.

## [2026-04-28 | unstaged | back out synchronous wax_museum eager-cache attempt]

**Touched pages:** [[wax-museum]] [[perf-sprint-traps]]

**Why:** mk5-main-7v4 implemented a synchronous whole-lattice registry cache, but the user's intent was ANE/Core ML overlap while Gemma was busy. The implementation did not use ANE, did not overlap with prompt/decode, and precomputed too broadly.

**Updated:**
- [[wax-museum]] — removed the eager tool-cache option section so the active entity page no longer advertises the rejected implementation.
- [[perf-sprint-traps]] — added "Synchronous whole-lattice wax_museum precompute is not ANE overlap" with the clean stub A/B: lazy 3.6s wall vs eager 13.5s wall on `perf_subset_5`, same legal/final behavior, 79 scheduled responses, 10 hits, 69 wasted.

**Side artifact:** `scratch/mk5-main-ane-sidecar-bead-draft.md` captures the replacement bead text because active `bd create` is blocked by the current `.beads` reinitialization state.

**Frontier shift:** whole-lattice pre-turn precompute is rejected. The next attempt must start with an ANE/Core ML proof and a turn-scoped overlapping sidecar, not a blocking registry memoization layer.

## [2026-04-30 | local | Gus strategy tags probe promoted]

**Touched pages:** [[entities/gus]] [[experiments/gus-strategy-tags-probe]]

**Why:** Winning 42 strategy-book work suggested a concrete empirical question: do
human-legible strategy tags help a small Gus-like policy model learn from public state,
or are they just explanation garnish?

**Updated:**
- [[entities/gus]] — added the current-frontier strategy-tag probe result and promoted
  file locations.
- [[experiments/gus-strategy-tags-probe]] — new experiment page with setup, feature
  shape, 100-game, 10k early-decision, and 28k early-decision results.

**Result:** explicit strategy tags contain real signal but do not beat `E[Q] N=10`.
On the 28k early-decision probe, the tiny base model scores 2.012 regret, the same
model with 68 global + 7×32 action-local strategy features scores 1.181 regret, and
`E[Q] N=10` remains far ahead at 0.167 regret.

**Frontier shift:** strategy tags are now a durable Gus experiment, not scratch. The
next question is concept-bucketed: find where tags already help most, especially
count pressure, trump pressure, off-risk/protection, donation windows, pounce windows,
and walker/endgame states.

---

## [2026-04-30 | 4a747f6 | catalog forge/analysis workstream into the wiki (t42-ff42)]

Brings the forge/analysis/ workstream into the wiki as first-class. Synthesis ingest, not a single-commit ingest — the underlying work landed across many commits between 2026-01-06 (`5ffdf58`) and 2026-01-31 (`4a747f6`); the wiki gap was the issue, fixed now via bead `t42-ff42`.

**Touched pages:** [[entities/forge-analysis]] [[topics/oracle-vs-human-play]] [[topics/risk-return-inverse]] [[topics/q0-positional-bias]] [[entities/forge]] [[index]] [[log]]

**Added:** 4 pages — 1 entity (`forge-analysis`), 3 topics (`oracle-vs-human-play`, `risk-return-inverse`, `q0-positional-bias`).

**Updated:**
- [[entities/forge]] — appended a "Standalone analytics workstream" section with backlinks to forge-analysis, oracle-vs-human-play, risk-return-inverse, and q0-positional-bias.
- [[index]] — catalogued the new entity + 3 topics.

**Frontier established:**
- forge/analysis/ is a publication-shaped statistical analysis of the perfect-information oracle — distinct from LEM/Burl/Gus modeling work. ~21 numbered notebook themes, per-section report writeups, executive summary at `forge/analysis/report/00_executive_summary.md`.
- The load-bearing caveat ([[oracle-vs-human-play]]) is its own page so any downstream wiki claim that wants to extrapolate to human play has a one-link reminder of the gap.
- Headline finding extracted as its own topic: [[risk-return-inverse]] (r=−0.38, medium effect, survives FDR + CV).
- The slot-0 Q-bias investigation in `forge/analysis/bias/` (20 probes, proposed shuffle fix not yet validated) lands as [[q0-positional-bias]] — reference for any future model retraining work.

**Questions opened:**
- Does the proposed shuffle fix (`bias/20-proposed-fix-shuffle.md`) actually eliminate the bias when validated? Open until [[gus]] or another model retraining incorporates it.
- Does the inverse risk-return correlation hold in marginalized data and (eventually) human play? Marginalized data is a partial bridge but full validation is untested.

---

## [2026-04-30 | cba521d | burl-chat workbench + post-commit Q&A research direction]

Standalone interactive workbench under `burl/chat/` for talking with Burl about a finished decision. FastAPI + in-process [[mlx-lm]] (no vLLM — abandoned twice on this project, never reintroduced) + Svelte 5 + Vite frontend. Loads any `harvest_batched_*` decision as a typed-segment conversation prefix (system, user, thinking, tool_call, tool_result, assistant_text, commit), color-coded by [[burl-2000-harvest]] bucket. Three implementation gotchas worth their own wiki page: MLX default GPU stream is thread-affine (single-thread executor required), sse-starlette emits `\r\n\r\n` frame separators that browser TextDecoder doesn't strip (split must be CRLF-aware), and Svelte 5 reactivity loses in-place mutations to objects already in `$state` (parser must replace segments by index, not mutate).

First-session findings opened a new research direction: **post-commit Q&A**. Talking with Burl after a hand the way a teammate would. Roberson's *Winning 42* chapters 2-8 are the canonical voice anchor — structurally a worked-example dialogue corpus, owned by the project via the user's family heritage.

**Touched pages:** [[entities/burl-chat]] [[experiments/burl-chat-spike]] [[topics/post-commit-q-and-a]] [[topics/at-risk-points]] [[decisions/chat-mode-primer]] [[decisions/play-adapter-lock-in]] [[entities/burl]] [[index]] [[log]]

**Added:** 6 pages — 1 entity (`burl-chat`), 1 experiment (`burl-chat-spike`), 2 topics (`post-commit-q-and-a`, `at-risk-points`), 2 decisions (`chat-mode-primer`, `play-adapter-lock-in`).

**Updated:**
- [[entities/burl]] — appended a "burl-chat workbench + post-commit Q&A research direction" section linking the new entity, experiment, primer decision, and lock-in decision.
- [[index]] — catalogued 1 entity + 2 topics + 1 experiment + 2 decisions.

**Frontier established:**
- **Adapter lock-in is real.** Stacking Q&A on top of a STaR-distilled play adapter does not work. e1-rank16 + harvested prefix + chat-mode primer + explicit "do not output a tool call" + a non-tool question → model still emits `commit_play({"domino_id":14})` mid-response, re-committing a play already in the prefix. Same prefix without the adapter (base Gemma 4 E2B) → engages cleanly, produces structured prose Q&A. A/B clean. See [[play-adapter-lock-in]].
- **Chat-mode primer is load-bearing.** Synthetic "Yeah, I committed N. Ask me anything" assistant turn injected after `commit_play` flips base Gemma from play-decision mode to chat mode via in-context recency. Without it, even base Gemma falls back to "I am Burl, my next action is to call commit_play." See [[chat-mode-primer]].
- **First product feedback from Burl on its own tools.** Base Gemma + a 9-word prompt produced a structured three-section critique of the eq_outcome_distribution and probe tool surface, suggesting (a) structured summaries before raw histograms, (b) explicit strategic labels, (c) "why" framing matching the [[at-risk-points]] frame. Real backlog item.
- **Voice gap visible.** Model spontaneously used Gus/forge vocabulary (Q axis, mean shifts, catalyst dominoes), not Roberson vocabulary (offs, walkers, double ahead of your off). To get the family-game voice the north star requires, either the tools must surface Roberson framing or a Roberson primer rides the system prompt. Probably both.

**Methodological lesson logged:** SSE consumers must be tested with `repr()` of raw response bytes, not eyeballed terminal output. The CRLF framing bug was invisible to every prior `curl` test because terminals strip CR. Cost: one false success cycle and a rebuild of the streaming consumer.

**Questions opened:**
- Does [[iter3-rules-adapter]] (90% bot-match Burl winner) suffer the same lock-in as e1-rank16, or is it more steerable due to less aggressive distillation? Adapter is on HuggingFace at `jasonyandell/gemma-4-e2b-texas42-burl-iter3-rules` — pull and A/B in [[burl-chat]].
- Does a Roberson primer (Flemmons foreword + chapter 2 paragraphs in the system prompt) shift the model's vocabulary toward the family-game register without retraining?
- Does bucket category (agreement / disagreement / forced / illegal) map to qualitatively different self-critique shapes? Need ~3-4 samples per bucket.
- Would a stronger chat-mode system-prompt override on top of the primer help on adapters that are merely steered, not welded?

---

## [2026-05-01 | unstaged | improvised-tools loop and the meta-layer lock-in finding]

Second working session in [[burl-chat]]. Wired up a hot-register tool layer ([[improvised-tools]]) so Claude (in another window) can read the live chat state, design a wax_museum-shaped tool against Burl's request, register it with one MCP call, and the user advertises it on the next turn. Tools persist to `burl/chat/server/tools_library/` as `<name>.py` files (`DESCRIPTION` constant + `def tool(...)`); the registry rehydrates from disk on import.

Three tools landed in the library across two decisions, each addressing a real failure visible in Burl's own transcript that turn:
- `board_snapshot` — wraps the existing `render_full_board_snapshot` in `burl/wax_museum/snapshot.py`. Built when Burl asked for "Comprehensive Game State Visualizer" on decision #0.
- `legal_plays` — names led suit + lists legal/illegal plays. Built when Burl committed an illegal 27(6-6) on decision #7 (must-follow ones from lead 2(1-1)) and asked for a "Strategic Synthesis Engine that picks for me." The literal request was answered with a state-tool, not a strategy-tool, per the wax_museum doctrine *"state tools answer WHAT IS the state — they never tell you WHAT TO DO."*
- `state_brief` — Burl's own preferred `[GAME STATE]` / `[CONTEXT & GOAL]` / `[PROTOCOL]` labeled-bullet rendering of the same data. Built when Burl said decision #0's input was "slightly confusing" and sketched the format.

UI: per-tool checkbox popover in the workbench header with `all` / `none` / delete (×) controls. Two bug fixes during the build — the `advertise tools` button was inert (only mutated segments without re-streaming; needed `await send("")`); and the auto-select `$effect` clobbered manual unchecks (needed a separate `seenToolNames` set so tools auto-select only on first appearance).

**Touched pages:** [[entities/burl-chat]] [[entities/improvised-tools]] [[topics/burl-tool-wishlist]] [[experiments/burl-chat-spike]] [[decisions/play-adapter-lock-in]] [[index]] [[log]]

**Added:** 2 pages — 1 entity (`improvised-tools`), 1 topic (`burl-tool-wishlist`).

**Updated:**
- [[entities/burl-chat]] — Files table extended with the registry, MCP server, library directory, and `snapshot.py` helpers; added an "Improvised-tool registry" section.
- [[experiments/burl-chat-spike]] — appended a "Session 2 (2026-05-01)" subsection documenting the three meta-asks, the doctrine moment with `legal_plays`, and the two UI bug fixes.
- [[decisions/play-adapter-lock-in]] — appended a "Meta-layer corroboration" section: the lock-in is structural ("the next assistant turn is a tool-call plan"), not just contextual to play states. Even when explicitly invited to chat about itself, Burl produces tool-spec plans in prose.
- [[index]] — catalogued the new entity + topic.

**Frontier shift:**
- The lock-in is at the structural level of "what an assistant turn looks like," not at the contextual level of "what to do at a play state." The post-commit Q&A adapter must train against this structurally — not just bolt a chat primer onto a play distribution.
- The lock-in is also **productive** as long as it is read sideways. Burl can name what would help its reasoning, and what it names is correct; it just names it in tool-spec form. That makes [[burl-tool-wishlist]] a corpus-mining strategy: every (Burl-asks, Claude-implements, demonstrated-improvement) triple is a candidate row for the future post-commit-Q&A training set.
- The improvised-tool layer is the operational answer to "vibe coding buddy" — Claude as the in-the-loop tool author. The library is the on-ramp; promotion to `burl/wax_museum/tools.py` is the destination for tools that earn their keep across decisions.

**Questions opened:**
- Does providing the named tool actually shift Burl's play on the same kind of decision next time? The wishlist is the question; the experiment is the answer. Pending.
- Can the wishlist be harvested across many decisions to bootstrap a real post-commit-Q&A corpus, or does the same ask recur and saturate quickly?
- What's the right doctrine line when Burl asks for a strategy-picker? `legal_plays` was the worked example (build the state-tool that closes the same gap), but harder cases will arrive — e.g., "rank my candidate plays by expected count" is borderline.
- Does swapping in the [[iter3-rules-adapter]] change the *shape* of the wishlist, or does the lock-in produce the same meta-structure across adapters?

---

## [2026-05-01 | unstaged | rerun-fresh + play_brief + first measurable lift on decision-1]

Second wave of the [[improvised-tools]] loop, same calendar day as the [[burl-chat-spike]] session 2. Three follow-on changes.

**`play_brief` tool.** Burl asked for `explore_game` output reshaped with a HEADLINE (variance + p_make), modes sorted by mass with `[BIG WIN]`/`[WIN]`/`[NEAR-BREAKEVEN]`/`[LOSS]`/`[DISASTER]` labels + catalysts, and a risk-profile line. Built on top of `WaxContext.get_or_build()` so it shares the cache with `explore_game` — calling both costs one set of oracle samples, not two. Doctrine intact: labels describe outcomes, not picks.

**Rerun-fresh in the workbench.** New header button. Pick a decision, check the desired improvised-tool subset in the popover, click `rerun fresh`. The workbench replaces `segments` with `[harvested_system + appended_tool_declarations, harvested_first_user_message]` and re-streams from turn 1. Burl plays the decision again with the new tools available from the system prompt onward — no chat-mode primer, no harvested trace to anchor on. The existing tool dispatch (improvised-registry checked first) handles new tools transparently.

**First measurable lift.** Reran `harvest_batched_20260425_072910` decision #1 (`BURL_BREAKS_CONSENSUS`, defense, position 2/4 in trick 1, lead 14(4-4) → led suit 4s). Original harvest played 25(6-4) — burned a 10-point count domino on trick 1, regret 3.53. Rerun-fresh with state_brief / legal_plays / play_brief / board_snapshot all available: Burl called `state_brief` first, immediately enumerated the legal subset (`I have 4(2-1) and 19(5-4) and 25(6-4). I can follow suit.`), and committed `4(2-1)` — a 0-count blank, defensively correct. State-brief's upstream legality clarity is now confirmed across two seats and roles (defense seat-3 trick-2 + defense seat-1 trick-1). Working hypothesis: state-brief lifts mean regret on follow-suit decisions where original-Burl explored an illegal candidate before catching the constraint.

**Two new findings worth their own pages.**

The first: **adoption is not automatic.** `play_brief` was registered, advertised, and never called in the rerun-fresh of decision-1. Burl followed the system prompt's literal `explore_game(play=X)` reference. `state_brief` got picked up because its self-description ("first read on any decision") was strong enough to clear the protocol's check. Documented in [[improvised-tools]] under "Adoption asymmetry" and in [[burl-tool-wishlist]] under "Adoption is not automatic." Cheap lever: tool descriptions that mimic protocol language. More expensive: patching the protocol section of the system prompt at rerun time (not yet implemented).

The second: **[[count-vs-pip-sum-confusion]]**, a separable rules-grounding bug. In the rerun-fresh decision-1 trace Burl wrote *"4(2-1) is a low count domino (2 points). 19(5-4) is medium count (5 points). 25(6-4) is high count (10 points)"* — confusing pip-sum with the official Texas 42 count value. Actual count values: 0/0/10. The two coincide for the five count-carriers (5-5, 6-4, 5-0, 4-1, 3-2) and only those, so the bug is silent ~5/28 of the time and bites the rest. Likely origin: the rules primer learned a "high pips → expensive" continuous proxy instead of the categorical labels. Fix path: improvised `count_ledger` tool → system-prompt patch → adapter retraining.

**Touched pages:** [[entities/burl-chat]] [[entities/improvised-tools]] [[topics/burl-tool-wishlist]] [[topics/count-vs-pip-sum-confusion]] [[experiments/burl-chat-spike]] [[index]] [[log]]

**Added:** 1 page — 1 topic (`count-vs-pip-sum-confusion`).

**Updated:**
- [[entities/burl-chat]] — added a "Rerun-fresh" section describing the two modes (join-at-end vs replay-turn-1).
- [[entities/improvised-tools]] — added `play_brief` to the library table; added "Adoption asymmetry" subsection on why `state_brief` got picked up and `play_brief` did not.
- [[topics/burl-tool-wishlist]] — added the `play_brief` row to the wishlist table; added "Adoption is not automatic" section listing the three intervention layers (tool description, protocol-text patch, adapter co-training).
- [[experiments/burl-chat-spike]] — appended a "Second wave" subsection covering `play_brief`, rerun-fresh, the decision-1 comparison, and the two new findings.
- [[index]] — catalogued the new topic.

**Frontier shift:**
- The wishlist loop now has a measurable lift signal, not just an aesthetic one. State-brief plus rerun-fresh produces strictly better play on a follow-suit decision the original Burl mishandled — without any model retraining. That's the demo template. The next experiment milestone: 5–10 such comparisons across diverse buckets and seats, recorded as a paired-regret table in `experiments/`.
- Adoption asymmetry (`state_brief` used, `play_brief` ignored) reframes the question. The improvised-tools loop is now in two stages — *will Burl call this?* and only then *will it lift his play?*. The first stage is downstream of system-prompt protocol text, not just tool quality.

**Questions opened:**
- Across N=5+ rerun-fresh comparisons, what's the mean regret delta from making `state_brief` available? Is it concentrated in `BURL_BREAKS_CONSENSUS` and `ILLEGAL` buckets, or does it lift `BURL_INDEPENDENT_WRONG` too?
- Will Burl ask for a count-ledger tool on a future decision, or does the count-vs-pip-sum confusion remain invisible to him because his pip-sum heuristic happens to land on the right answer often enough?
- What's the minimum protocol-text patch that gets `play_brief` into rotation? "Call `play_brief(play=X)` to examine a candidate" as a one-line replacement for the existing `explore_game` reference would be the natural test.
- Is there a third workbench mode — *patch-and-rerun*, where the user can edit the system prompt's protocol section before the rerun streams — that closes the adoption gap without code changes? Worth prototyping if the protocol-text lever pans out on the play_brief case.

---

## [2026-05-01 | unstaged | autoFillFromHarvest silent-fallback bug + reflection-deafness]

Third wave on `burl-chat`. One real bug fixed, one structural finding documented.

**Bug fix: `autoFillFromHarvest` silent wrong-args fallback.** When Burl emitted `explore_game(play=X)` for any X that wasn't in the harvest's recorded tool calls, the workbench's stream-end handler pre-populated the manual-review draft with the *first* matching `tool_result` by tool name, regardless of args. For decision #0 the harvest only recorded `explore_game(play=14)`, so every rerun-fresh `explore_game(play=25)` or `(play=20)` got pre-filled with the play=14 prose. The user clicked "feed" without re-reading; Burl reasoned over fabricated outcome distributions thinking he had explored multiple candidates. Three different play arguments returned identical output in one shared trace. Symptom-equivalent to a hard wiring bug, mechanism is a UX-quality lapse: the auto-filled wrong-args response is structurally identical to a real response and only became visible when one trace happened to call `explore_game` with three different args in a row.

Fix in `burl/chat/web/src/App.svelte`:
- `autoFillFromHarvest` returns `null` on args mismatch (caller fetches live or the user supplies manually).
- `explore_game`, `probe_best_case`, `probe_worst_case` joined `AUTO_SERVE_BASE` so they live-dispatch instead of going through the harvest-fallback path. Trade-off: live samples are not byte-identical to what the harvest recorded (N=20 fresh resample), so join-at-end mode no longer reproduces the harvest's exact numbers — but it always returns correct numbers for the actual args. For exact reproduction the user reads the events panel directly; the chat path is now correct-by-args, not faithful-to-harvest.

**Topic added: [[burl-reflection-deafness]].** During the same trace the user injected (via harness free-text feedback routed back as a `commit_play` tool_result): *"that is not the best play. why?"* Burl's response: zero engagement. Re-ran `explore_game`, re-probed, re-committed `14`. Three identical commits in a row. The pedagogical opening was absorbed into "the user wants me to do my decision job again." This is [[play-adapter-lock-in]]'s third symptom (alongside the wishlist's tool-spec response shape and the post-commit primer's load-bearing role). Implication for [[post-commit-q-and-a]]: the eventual training corpus has to include reflection turns explicitly, since the current model cannot produce them organically.

**Touched pages:** [[experiments/burl-chat-spike]] [[topics/burl-reflection-deafness]] [[index]] [[log]]

**Added:** 1 page — 1 topic (`burl-reflection-deafness`).

**Updated:**
- [[experiments/burl-chat-spike]] — appended a "Third wave" subsection covering the autoFill bug, fix, and reflection-deafness observation.
- [[index]] — catalogued the new topic.

**Frontier shift:**
- Past `BURL_BREAKS_CONSENSUS` analyses that *appeared* to show Burl exploring multiple candidates and selecting the best one need re-examination if they came from join-at-end chat sessions where the harvest didn't record the explored args. The autoFill silently produced uniform output. Going forward, `play_brief`'s headline (which always names the actual play in its first line) is the cheap diagnostic — if every `play_brief` in a trace shows the same `PLAY: X(p-p)` regardless of input arg, the bug regression is back.
- Reflection-deafness reframes a chunk of the post-commit-Q&A corpus question. We can't bootstrap reflections from organic chat with the current model; either hand-write from Roberson chapters 2-8 (canonical voice) or distill from a stronger model.

**Questions opened:**
- Does the [[iter3-rules-adapter]] (which we still haven't pulled locally) suffer reflection-deafness too, or is its less-aggressive distillation enough to engage with "why?" prompts?
- Does a system-prompt patch — e.g., *"if the user asks you a question instead of giving you a state, answer the question; do not call tools"* in the protocol section — close the deafness, or does the trained pattern override even that explicit instruction (parallel to the [[chat-mode-primer]] vs adapter lock-in case)?
- Now that `explore_game` outputs are trustworthy in rerun-fresh, is there a measurable regret delta when Burl is rerun on `BURL_BREAKS_CONSENSUS` decisions with the corrected dispatch? Worth a small N=5 batch to confirm before scaling.

---

## [2026-05-01 | f746b93 | w42 rich-tag many-signal probe]

**Touched pages:** [[w42-rich-tag-many-signal-probe]] [[index]] [[log]]
**Added:** 1 experiment page — [[w42-rich-tag-many-signal-probe]].
**Updated:** [[index]] catalogued the new w42 experiment.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- [[w42]] now has a live-W&B rich-tag probe on the same small split as the raw and v0 baselines. The many-signal public feature surface narrowly beats v0 on one seed (1.970 vs 2.000 mean regret) and improves raw-final tail risk, but remains report-only and underpowered.
- The useful signal still concentrates in count/donation and pounce-like proxy windows. True claim movement still waits for multi-seed runs or claim-specific detector/oracle evidence.

---

## [2026-05-02 | local | w42 W&B series logging standard]

**Touched pages:** [[w42-lab-infrastructure]] [[w42-wandb-series-logging-standard]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-wandb-series-logging-standard]].
**Updated:** [[w42-lab-infrastructure]] now links the named-axis series standard; [[index]] catalogued the page.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- [[w42]] now treats W&B as a trajectory notebook. Any long run with epochs, variants, claims/specs, bootstrap checkpoints, chunks, seeds, or eval checkpoints should log multiple points with a meaningful numeric axis.
- The shared W&B helper has `log_series_point(...)`; ablations now log generic `variant/*` metrics on `variant/index`, and setter-defense claim validation logs generic `claim/*` metrics on `claim/index`.
- The live smoke run `xq7q9bar` logged six points across epoch and bootstrap axes at `https://wandb.ai/jasonyandell-forge42/w42/runs/xq7q9bar`.

---

## [2026-05-02 | local | w42 multi-seed larger-eval replication]

**Touched pages:** [[w42-multi-seed-larger-eval-replication]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-multi-seed-larger-eval-replication]].
**Updated:** [[index]] catalogued the replication page.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- [[w42]] now has a five-seed raw/v0/rich replication on 2,800 held-out decisions, with live per-epoch W&B series for all 15 model runs.
- Raw-to-v0 and raw-to-rich gains held on the larger slice; rich-over-v0 is a modest positive signal, not a promotion decision or claim-ledger verdict.
- `E[Q] N=10` remains far stronger on the same eval slice, so the result supports continued feature work rather than final conclusions.

---

## [2026-05-02 | local | w42 initial survey close-out]

**Touched pages:** [[w42]] [[winning42-strategy-measurement]] [[w42-hugging-face-artifact-publishing]] [[w42-final-empirical-strategy-report]] [[w42-next-model-decision]] [[w42-promote-or-retire]] [[index]] [[log]]
**Added:** 4 pages — 2 experiment pages ([[w42-hugging-face-artifact-publishing]], [[w42-final-empirical-strategy-report]]) and 2 decision pages ([[w42-next-model-decision]], [[w42-promote-or-retire]]).
**Updated:** [[w42]] now points future readers at the initial survey synthesis and keep-as-research decision; [[winning42-strategy-measurement]] records that strategy tags help models but tactical advice remains mostly underpowered/context-limited; [[index]] catalogued the new pages.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- The initial w42 survey is closed as a research map, not as a book verdict. Strategy tags are useful model inputs; exact odds/rules/scoring claims have narrow support; tactical claims need direct detectors and paired tests.
- HF publishing is deferred because current artifacts are research reports/checkpoints, not stable public datasets or promoted models.
- w42 stays active as research. The next model step is a targeted v2 direct-detector probe for one tactical regime, preferably setter pounce if current public-state/contract context can support it.

---

## [2026-05-02 | local | promote w42 to top-level workstream]

**Touched pages:** [[w42]] [[w42-promote-or-retire]] [[w42-hugging-face-artifact-publishing]] [[index]] [[log]]
**Added:** top-level `w42/README.md` and package marker.
**Updated:** tracked w42 code, reports, schemas, tables, and small checkpoints moved from `scratch/w42/` to `w42/`; wiki paths now point at the durable project home; local W&B directories remain ignored machine-state.
**Retired:** tracked `scratch/w42/` home.
**Questions opened:** none.

**Frontier shift:**
- w42 is no longer throwaway scratch. It is a durable project-root research workstream like Gus or Burl in repo shape, while still keeping a research boundary around model/checkpoint promotion.
- The promotion is about ownership and continuity, not about declaring the tiny w42 probes production-ready.
- Temporary unrelated notebooks and generated caches can still live in `scratch/`; durable w42 evidence and scripts now belong under `w42/`.

---

## [2026-05-02 | local | E[Q] browser visualizers refreshed]

**Touched pages:** [[eq-browser-visualizers]] [[candlewax]] [[index]] [[log]]
**Added:** 1 experiment page - [[eq-browser-visualizers]].
**Updated:** [[index]] catalogued the E[Q] visualizer runbook; [[candlewax]] links the browser visualizers as the visual companion to the tool-surface PDF fields.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- The E[Q] web visualizers now have a portable local data-export path instead of relying on stale hardcoded `/home/jason/...` script paths.
- Browser Use verified all three local pages: aggregate 3D surface, per-action PDF discs, and game-journey value trajectories.
- The visualizer is a useful hypothesis tool for w42/E[Q] work: it makes near-tie, high-uncertainty, candlewax-shaped decision regions visible before any reranker or policy change is attempted.

---

## [2026-05-02 | a2db3c7 | burl-lab platform spec lands]

**Touched pages:** [[burl-lab]] [[burl-chat]] [[index]] [[log]]
**Added:** 1 entity page — [[burl-lab]] documenting the new deterministic experimentation platform replacing burl-chat.
**Updated:** [[burl-chat]] gains a status note flagging burl-lab as the active surface and itself as the reference predecessor (not yet superseded — that flips when burl-lab reaches parity); [[index]] catalogues the new entity and rewrites burl-chat's hook.
**Retired:** none.
**Questions opened:**
- Will phase markers stay harness-private as the post-commit-Q&A adapter co-trains, or get tokenized once the corpus harvests stabilize?
- Does HATEOAS `next_tools` advertisement actually shift Burl's tool selection, or does the model still defer to the system-prompt protocol-text (the [[improvised-tools]] adoption-asymmetry finding) even when the prior tool result names the next move?

**Frontier shift:**
- burl-chat is no longer the active workbench — it is the reference predecessor. New experimentation lands in `burl/lab/`.
- The architecture turns four spike findings into structural choices: (1) rendered protocol text from first-class ToolSpec replaces hand-edited primer prose; (2) `Stamp` makes timings part of the journal; (3) HATEOAS tool advertisement replaces "Burl plans a tool he doesn't have"; (4) Phase machine gives reflection a dedicated surface instead of relying on play-decision lock-in to crack open.
- The platform was first named `burl/harness/` before the team noticed the collision with the existing agent tool-loop runner package (~20 importers across `wax_museum/`, `haiku_spike/`, `candlewax_spike/`, `eval/run_move4_*`, `burl/chat/server/tools_runner.py`). Renamed to `burl/lab/`; misrouted files were consolidated via `git mv` during the spike. The existing `burl/harness/` package is untouched.
- SPEC.md (`burl/lab/SPEC.md`) is the canonical contract; build order is types → engine → tools → runtime. As of this entry, types + engine + tools have landed (transcript, tool, phase, engine modules + base ToolSpecs); runtime layer (render, drive, hf_sink, phases, server) is in flight.
- Citation note: this entry was originally written before the commit landed; it now cites `a2db3c7` ([[sources/a2db3c7]]) — the platform spine commit (31 files, 4619 insertions) that landed all three same-day burl-lab milestones in one shot.

---

## [2026-05-02 | a2db3c7 | burl-lab server runs end-to-end on fake engine]

**Touched pages:** [[burl-lab]] [[log]] [[questions/open]]
**Added:** none.
**Updated:** [[burl-lab]] gains a Status section: server live on port 18002 (alt of 8002), 11/11 tests pass (`test_transcript_roundtrip`, `test_engine_smoke`, `test_tools_base`, `test_render`, `test_drive_with_fake_engine`, `test_server_smoke`), `/api/health` + `/api/sessions` + `/api/move` wired, three base ToolSpecs registered, `events.jsonl` authoritative. Also documents the interim `state.json` companion as transient — pending `SystemSet` / `AdvertisedSet` Move kinds.
**Retired:** none.
**Questions opened:**
- When do `SystemSet` / `AdvertisedSet` Move kinds land in `core/transcript.py`, and does `state.json` get fully dropped at that point or does any read path linger?

**Frontier shift:**
- burl-lab transitions from server-spec to server-running. The Phase machine + Engine protocol + ToolSpec rendering + HATEOAS advertisement compose end-to-end against a fake engine — the architecture is no longer a paper claim.
- Real MLX engine integration is gated on a thread-affinity bug surfaced during runtime testing. The same single-thread executor pattern from [[burl-chat]] (`ThreadPoolExecutor(max_workers=1)` with model load and generate on the same thread) is the known fix; recovery in progress.
- One transient shape worth tracking: `state.json` is written alongside `events.jsonl` because `fold` does not yet reconstruct config moves. This is interim, not the contract — once the new Move kinds land, the wiki should not ossify the two-sources-of-truth shape.

---

## [2026-05-02 | a2db3c7 | burl-lab journal-canonical + post_turn lands]

**Touched pages:** [[burl-lab]] [[mlx-lm]] [[burl-chat]] [[log]] [[questions/open]] (resolved)
**Added:** none.
**Updated:** [[burl-lab]] retires the **Transient: `state.json` companion** subsection in place; Status now reads "events.jsonl is the only on-disk source of truth" with the journal-canonical test cited (`test_server_smoke.py::test_journal_is_canonical_no_state_json`). Adds a **Phase ownership of transitions** paragraph: drive is engine-shaped (no `PhaseExit`/`PhaseEnter`), server is transition-shaped (owns those Moves whenever `handle()` returns a non-`None` `next_phase`). Phase machine description bumped to ship `pre_game` + `in_run` + `post_turn`; module layout adds `phases/post_turn.py`. Test count 11/11 → **15/15** across the six test files. [[mlx-lm]] gains the **Upstream bug: module-level generation_stream** section (diagnosis + fix + submodule-shadowing trap). [[burl-chat]] notes that its single-thread executor pattern is necessary but **insufficient** for full MLX threading correctness; the executor's import path leaves the bug latent there.
**Retired:** `state.json` companion as a runtime artifact (the journal is canonical); the corresponding **Transient** subsection on [[burl-lab]] (retired in place per the prior log entry's standing instruction).
**Resolved:** the open question "When do `SystemSet` / `AdvertisedSet` Move kinds land in burl-lab's `core/transcript.py`, and does `state.json` get fully dropped?" — answered: `SystemSet`/`AdvertisedSet`/`ToolAdded`/`ToolRemoved` are journaled directly by phase handlers; `state.json` is gone; verified by test.

**Frontier shift:**
- burl-lab is now **journal-canonical**: every read goes through `fold(replay(session_dir))`, no snapshot file exists, and a server-smoke test enforces it. SPEC.md philosophy point 1 ("state is a fold over events") is no longer aspirational.
- The Phase Protocol's `next_phase` return value is the only place transition information lives. The drive loop never knows which phase it is in; the server is the only component that materializes phase boundaries as journal Moves. This split keeps drive engine-shaped and reusable across recorded / fake / real engines.
- The upstream `mlx_lm.generate.generation_stream` bug is fully diagnosed and fixed: rebind via `sys.modules["mlx_lm.generate"]` after `load()` on the executor thread. The submodule-shadowing trap (`from mlx_lm import generate` resolves to the function and silently no-ops) is what made the naive fix invisible; documented on [[mlx-lm]] so future readers do not retread it. burl-chat's import path happened to mask the bug; burl-lab surfaced it explicitly.
- One open question remains for the platform: does HATEOAS `next_tools` advertisement actually shift adoption when a real model is on the other end? Real MLX is wired but the experiment has not run yet.

---

## [2026-05-02 | local | w42 phase-2 research surfaces land]

**Touched pages:** [[w42]] [[w42-phase2-statistics-claims-ledger]] [[w42-phase2-seat-position-strategy-map]] [[w42-phase2-hidden-domino-threat-attribution]] [[w42-phase2-distribution-aware-ev-report]] [[w42-phase2-setter-pounce-direct-label-probe]] [[w42-phase2-84-weapon-preservation-probe]] [[index]] [[log]]
**Added:** 6 experiment pages and six top-level `w42/` artifact directories for phase-2 research.
**Updated:** [[w42]] now links the phase-2 surfaces; [[index]] catalogues the new pages.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- w42 phase 2 now has a concrete research bench rather than a discussion queue: a statistics ledger, seat/position strategy map, hidden-domino threat attribution design, distribution-aware EV report, setter-pounce direct-label probe, and 84 weapon-preservation probe.
- The statistics ledger is the bedrock: originally 62 rows here, now 64 rows after the Gus-corpus tactical deep dive. It separates supported exact/rules/scoring substrates and direct tactical-corpus evidence from underpowered tactical advice. No tactical book claim is promoted simply because its arithmetic substrate is true.
- The E[Q] visualizer insight has been operationalized: phase-2 reports now track threshold mass, tails, quantiles, branch shape, and belief-impact magnitude instead of reducing every decision to scalar mean EV.
- The direct tactical probes remain conservative. Setter pounce and 84 preservation produced label specs, fixtures, required fields, and leakage checks; dynamic rollouts/model probes are explicitly next work, not silently assumed.

---

## [2026-05-02 | local | w42 phase-2 decision table v0]

**Touched pages:** [[w42]] [[w42-next-model-decision]] [[w42-phase2-decision-table]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-phase2-decision-table]]; 1 top-level artifact directory - `w42/phase2_decision_table/`.
**Updated:** [[w42]] lists the decision table as the seventh phase-2 surface; [[w42-next-model-decision]] records it as the first bridge artifact for targeted probes; [[index]] catalogues the new page.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- w42 now has a reusable v0 decision table rather than only separate phase-2 reports. It joins the E[Q] PDF visualizer sample to seat/role context, public trick state, actor hand/action facts, and distribution-aware labels.
- The table has 140 decision-state rows and 346 legal-action rows from 5 games, with 1000 samples per PDF. It reproduces the prior 68 scalar-EV omission decisions and adds detector tags for first-trick belief update, late threshold closure, last-to-act closure, defender damage leads, bidder first leads, setter count pressure, and partner donation.
- The missing columns are explicit: bid amount, bid margin, and hidden-world ownership are not present in the current visualizer JSONL. Hidden-threat holder/domino/impact fields remain blank until a joint-world generation pass lands.
- The table is a schema and slice-analysis artifact, not a training run, W&B run, HF artifact, or claim-ledger verdict.

---

## [2026-05-02 | local | burl/lab first wire run end-to-end]

**Touched pages:** [[burl-lab]] [[burl-chat]] [[log]]

First end-to-end wire run of the burl/lab platform: web → SSE → server :8002 → real `MlxEngine`. Session `d80349b45ad6` committed `domino_id=21 (6-0)` for `harvest_batched_20260425_072910` decision_idx=1 via the `commit_play` envelope, matching the canonical in-process smoke `3dc73cb67bc1`. First live confirmation that the Decision Protocol section is rendered from active `ToolSpec.protocol_phrase` (captured in `scratch/burl-lab-runlogs/rendered-system.txt`), not hand-edited. Full write-up — including the chat-vs-lab structural diff and a sampling-stochastic envelope-miss note from session `8ff6e41fd87e` — at [[burl-lab]] § "First wire run (session d80349b45ad6)".

---

## [2026-05-02 | local | w42 powered branch atlas v1]

**Touched pages:** [[w42]] [[w42-next-model-decision]] [[w42-phase2-hidden-domino-threat-attribution]] [[w42-phase2-distribution-aware-ev-report]] [[w42-powered-branch-atlas-v1]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-powered-branch-atlas-v1]]; 1 artifact directory - `w42/branch_atlas_v1/`.
**Updated:** [[w42]] now lists the powered branch atlas as the eighth phase-2 surface; [[w42-next-model-decision]] records it as the first executable hidden-impact measurement loop; [[w42-phase2-hidden-domino-threat-attribution]] now points to real hidden-threat rows instead of only the schema design.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- w42's branch-aware EV work is no longer only a visualizer/JSONL schema. The project generated two schema-v2 E[Q] games with N=1000 saved joint worlds and built 56 decision rows, 134 legal-action rows, and 1026 hidden-threat attribution rows.
- Hidden-holder impact is now measured by conditioning `q_per_world` on `(hidden_domino, holder)` and comparing mean Q, lower-tail mass, high-shelf mass, and std against the baseline action distribution. The labels remain offline-only and must not become live hidden-truth inputs.
- W&B is now used as a dashboard for report experiments: run `44z1kl9j` logs repeated points over `progress/decisions_processed` rather than only final summary fields.
- The pilot still uses fixed `bid_value=30`, so bid-margin and "bid only enough" claims remain untested until real auction metadata is attached.

---

## [2026-05-02 | local | w42 branch atlas scaled v0]

**Touched pages:** [[w42]] [[w42-next-model-decision]] [[w42-powered-branch-atlas-v1]] [[w42-phase2-hidden-domino-threat-attribution]] [[w42-phase2-distribution-aware-ev-report]] [[w42-branch-atlas-scaled-v0]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-branch-atlas-scaled-v0]]; 1 artifact directory - `w42/branch_atlas_scaled_v0/`.
**Updated:** [[w42]] lists the scaled atlas as a ninth phase-2 surface; [[w42-next-model-decision]] records the bid-aware threshold plumbing; [[w42-powered-branch-atlas-v1]], [[w42-phase2-hidden-domino-threat-attribution]], and [[w42-phase2-distribution-aware-ev-report]] point forward to the scale-up.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- The branch-atlas loop now covers all ten declarations for seed 9430: 280 decisions, 773 legal-action rows, and 5955 hidden-threat rows at N=1000.
- The E[Q] generator now uses recorded `bid_value` for p_make thresholds, and the atlas computes threshold mass / hidden-threat shelf mass from each recorded bid value. Bid 30 behavior is preserved and tested.
- W&B run `7fwi2zwn` logs repeated progress points for the scaled report, including action rows, hidden-threat rows, branch rates, coverage, and wall time.
- The scaled v0 artifact still fixes `bid_value=30`; it validates declaration coverage and bid-aware plumbing, not real auction metadata, bid margin, or "bid only enough" strategy claims.

---

## [2026-05-02 | local | burl/lab logged arrows restored]

**Touched pages:** [[burl-lab]] [[logged-arrows]] [[index]] [[log]]
**Added:** [[logged-arrows]]
**Updated:** [[burl-lab]] records the Trace-based phase protocol; [[index]] catalogues the topic.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- burl-lab now has an explicit logged-arrow algebra for phase handling: `Trace[O] = (events: tuple[Move, ...], output: O | None)`.
- Phase handlers no longer append to disk or re-fold state. They return journalable Moves plus an optional next phase; the server interprets the trace by appending, streaming, folding, and materializing phase transitions.
- The narrower `WaxContext`-in-server cleanup remains a symptom; the deeper invariant is that harness steps compose by returning logs plus optional outputs rather than by smuggling side effects through phase code.

---

## [2026-05-02 | local | w42 Gus corpus tactical claim deep dive]

**Touched pages:** [[w42]] [[w42-gus-corpus-tactical-claim-deep-dive]] [[w42-phase2-setter-pounce-direct-label-probe]] [[w42-setter-defense-claim-validation]] [[w42-partner-support-claim-validation]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-gus-corpus-tactical-claim-deep-dive]]; 1 artifact directory - `w42/gus_corpus_claim_deep_dive/`.
**Updated:** [[w42]] lists the Gus-corpus tactical report as an active phase-2 surface; setter-pounce and partner-support pages point to the new direct evidence.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- w42 can use existing Gus v2 joint-world corpora for some tactical claim tests instead of regenerating seeds. The run processes 10 source files, 28,000 decisions, and 75,079 legal actions at N=200 sampled worlds per decision.
- Setter pounce moved from direct-label spec/proxy evidence to a powered corpus result on pip declarations: pounce-count actions beat same-decision alternatives by about +4.1 Q, and pounce-count actions that set now beat alternatives by about +6.1 Q.
- Reckless count donation into the bidder side is a strong negative control at about -8.4 Q versus non-reckless alternatives. Partner safe donation has a smaller positive paired signal, while unsafe partner count donation is strongly negative.
- W&B run `zm3jdrnj` logs progress per source shard; the report remains a corpus-slice result, not a model-training run, HF artifact, or universal book-claim verdict.

---

## [2026-05-02 | local | burl/lab harvested chat prompt import]

**Touched pages:** [[burl-lab]] [[log]]

`pre_game.load_decision` now imports the harvested [[burl-chat]] `prompt_system` alongside `prompt_user`, strips the legacy `# Decision protocol (wax_museum)` section and raw `<|tool>declaration:` blobs, and journals the cleaned prompt via `SystemSet`. The next engine render appends the lab-owned Decision Protocol from active `ToolSpec.protocol_phrase` values, preserving chat-era Burl grounding without letting the protocol/tool surface drift out of sync.

---

## [2026-05-02 | local | burl/lab mined chat tools]

**Touched pages:** [[burl-lab]] [[log]]

The four useful [[improvised-tools]] from [[burl-chat-spike]] now load into burl-lab as first-class `ToolSpec` wrappers: `state_brief`, `board_snapshot`, `legal_plays`, and `play_brief`. Implementations still delegate to `burl/chat/server/tools_library/`, but burl-lab now owns their schemas, examples, protocol roles, and protocol phrases, so the rendered Decision Protocol can name them directly.

---

## [2026-05-02 | local | w42 claim analysis matrix]

**Touched pages:** [[w42]] [[w42-phase2-statistics-claims-ledger]] [[w42-phase2-claim-analysis-matrix]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-phase2-claim-analysis-matrix]]; 1 artifact directory - `w42/claim_analysis_matrix/`.
**Updated:** [[w42]] lists the claim-analysis matrix as the eleventh phase-2 surface; [[w42-phase2-statistics-claims-ledger]] points to the routing matrix as planning metadata; [[index]] catalogues the new page.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- The 64-row w42 statistics ledger now has a generated test-design matrix with required fields, leakage risks, likely blockers, sample/power needs, target wiki pages, and next beads for every claim.
- Nineteen rows are fixture-only exact or deterministic substrates. Forty-two rows are routed to powered research queues, led by 84 dynamic generation, auction/bid-margin generation, direct detector implementation, bidder sequence counterfactuals, and doubles/no-trump regime generation.
- The matrix preserves the current claim statuses. It is not evidence that additional claims are supported; it is the route map for the next empirical beads.

---

## [2026-05-02 | local | w42 claim analysis harness]

**Touched pages:** [[w42]] [[w42-phase2-claim-analysis-matrix]] [[w42-phase2-claim-analysis-harness]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-phase2-claim-analysis-harness]]; 1 code package - `w42/claim_analysis/`; 1 smoke artifact directory - `w42/claim_analysis_smoke/`.
**Updated:** [[w42]] lists the harness as the twelfth phase-2 surface; [[w42-phase2-claim-analysis-matrix]] points its harness dependency to the new page; [[index]] catalogues the new page.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- The one-off Gus tactical deep-dive loop now has a reusable row-level harness for label metrics, label-vs-nonlabel paired contrasts, bootstrap CIs, examples, manifests, and W&B progress series.
- A branch-atlas smoke over 773 legal action rows produced 36 label metric rows and 14 paired contrast rows, with W&B run `rj3j0jsz` logging eight progress points over rows processed.
- A Gus claim-row smoke reproduced the six exported tactical labels, but its paired contrasts are explicitly limited because the exported JSONL is label-filtered rather than a full legal-action table.

---

## [2026-05-02 | local | w42 tactical claim replication]

**Touched pages:** [[w42]] [[w42-gus-corpus-tactical-claim-deep-dive]] [[w42-tactical-claim-replication]] [[w42-setter-defense-claim-validation]] [[w42-partner-support-claim-validation]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-tactical-claim-replication]]; 1 artifact directory - `w42/tactical_claim_replication/`.
**Updated:** [[w42]] lists the tactical replication as the thirteenth phase-2 surface; the Gus deep-dive, setter-defense, and partner-support pages point to the replication.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- The tactical replication reruns all ten Gus v2 source shards: 28,000 decisions, 75,079 legal-action rows, and 3903 claim-labeled action rows.
- It preserves full legal-action rows for the reusable harness, then adds 166 label-slice rows and 133 paired-contrast slice rows over declaration, seat, trick, current-control, count amount, later-seat proxy, and sets-now context.
- The original narrow conclusions hold: pounce count is about `+4.1` Q, pounce-that-sets-now is about `+6.1`, reckless count is about `-8.4`, safe partner count is about `+0.7`, and unsafe partner count is about `-8.4` on paired same-decision contrasts.
- W&B run `jv9luhgp` logs ten progress points, one per processed source shard. No claim status is broadened beyond the already stated operationalized slice.

---

## [2026-05-02 | local | burl/lab prompt UX clarification]

**Touched pages:** [[burl-lab]] [[log]]

Fresh pre-game sessions still start with no system prompt (`rendered system · 0 chars`) until a `SystemSet` is journaled, but the UI now explains the two paths: manual `Set system prompt`, or `Load harvested decision + prompt` to import harvested `prompt_system` plus `prompt_user`. The tool drawer now counts registry rows in pre-game, so the initial surface reports `0/7 advertised` instead of the misleading `0/0 advertised`.

---

## [2026-05-03 | local | burl/lab prompt builder]

**Touched pages:** [[burl-lab]] [[log]]

Pre-game is now a walkable prompt-builder phase. The user can select the exact advertised tool set, click `Generate system prompt` to seed the default Burl base prompt, inspect the composed rendered system with the selected tools' protocol phrases, then explicitly `Start run` or `Ask Gemma now`. Harvested decision loading now fills the builder with `prompt_system` and `prompt_user` without automatically entering `in_run`, so prompt construction and model generation are separate logged-arrow steps.

---

## [2026-05-03 | local | burl/lab guided wizard]

**Touched pages:** [[burl-lab]] [[log]]

The web surface now presents the prompt-builder path as a four-step wizard: start a new session, build a prompt or load/chat from a harvested decision, select tools with presets or checkboxes, then confirm and ship to Gemma. The wizard drives the same logged-arrow moves (`generate_system`, `set_advertised`, `load_decision`, `ask_gemma`, `start_run`) while keeping the raw option composer behind an advanced console.

---

## [2026-05-03 | local | burl/lab no-context ship guard]

**Touched pages:** [[burl-lab]] [[log]]

The wizard and `pre_game.start_run` now require a user/decision prompt before entering `in_run`. `ToolSpec.requires_context` also keeps game-state tools hidden from the engine until a harvested decision has produced ctx, with a `missing_context` tool result as the last-resort guard. This prevents the footgun where a system prompt plus advertised game-state tools could be shipped with no harvested decision, causing context-bound tools such as `state_brief` to run with `ctx=None`.

---

## [2026-05-03 | local | burl/lab seeded decision lane]

**Touched pages:** [[burl-lab]] [[log]]

The guided wizard now exposes the intended seeded decision flow directly: optional chat, generate the system prompt from the selected tool protocol, select/apply tools, then `send_seeded_decision(harvest, seed)`. The arrow now uses rendered `board_snapshot()` prose as the user message instead of the harvested prompt text, without requiring `legal_plays` in the selected tool set. After a commit, the server journals `SessionOutcome`: selected tools, final domino, legal/illegal status, legal set, and pi/qmean/burl/oracle/consensus comparisons for later tool-selection mining.

---

## [2026-05-03 | local | w42 claim data inventory]

**Touched pages:** [[w42]] [[w42-claim-data-inventory]] [[w42-phase2-claim-analysis-matrix]] [[w42-doubles-no-trump-claim-validation]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-claim-data-inventory]]; 1 artifact directory - `w42/claim_data_inventory/`.
**Updated:** [[w42]] lists the data inventory as the fourteenth phase-2 surface; [[index]] catalogues the new page.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- The W42 recovery now accounts for the large local Gus corpus under `gus/data/`, not only promoted `w42/` artifacts or repo `scratch/`.
- A full one-chunk-at-a-time inventory inspected 113 payloads with no load errors: all 100 legacy seed-modulo chunks, the legacy 100-game file, all 10 v2 train files, and both eval files.
- The legacy corpus provides 10000 inferred games and 280000 decisions, with 1000 games per declaration. It supports broad within-regime play, action, and hidden-threat diagnostics, including Chapter 9 doubles-trump/no-trump tactical mining.
- The same inventory confirms the missing-field blockers: legacy chunks have no observed real `bid_value` or auction margin, and the v2 all-declaration files use fixed `bid_value=30`. Hard bid-only-enough claims and true 84-contract claims still require generated data.

---

## [2026-05-03 | local | w42 doubles/no-trump legacy mining]

**Touched pages:** [[w42]] [[w42-doubles-no-trump-legacy-mining]] [[w42-doubles-no-trump-claim-validation]] [[winning42-ch09-doubles-no-trump]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-doubles-no-trump-legacy-mining]]; 1 artifact directory - `w42/doubles_no_trump_legacy_mining/`.
**Updated:** [[w42]] lists the legacy mining pass as the fifteenth phase-2 surface; [[w42-doubles-no-trump-claim-validation]] points to the new action-row follow-up.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- Chapter 9 is no longer static-only. The large legacy Gus corpus supports within-regime declaration-7 and declaration-9 tactical proxy mining, even though same-hand no-trump-vs-doubles-trump regime choice still needs generated paired declarations.
- The run processes all 100 legacy chunks: 56000 declaration-7/9 decisions, 149415 legal action rows, and 25906 proxy-labeled action rows. W&B run `o96omty7` logs 20 progress points and the report artifact.
- No-trump double control is the strongest signal: early support-double spend proxies beat non-double alternatives by about `+5.76` Q, late double spend by about `+3.03` Q, and defender double weapons by about `+12.80` Q on paired same-decision contrasts.
- Broad doubles-trump low-double sacrifice and `6-5` dual-suit-top proxies do not promote claims by themselves. They need the book's missing gates: planned loss budget, missing higher doubles, off/walker payoff, suit depletion, and support-double preservation state.

---

## [2026-05-03 | local | w42 hidden-threat legacy mining]

**Touched pages:** [[w42]] [[w42-hidden-threat-legacy-mining]] [[w42-phase2-hidden-domino-threat-attribution]] [[w42-phase2-distribution-aware-ev-report]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-hidden-threat-legacy-mining]]; 1 artifact directory - `w42/hidden_threat_legacy_mining/`.
**Updated:** [[w42]] lists the hidden-threat legacy pass as the sixteenth phase-2 surface; hidden-attribution and distribution-aware EV pages point to the full-corpus follow-up.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- The hidden-threat bead `t42-0b4l.4` now has a full legacy-corpus pass: 100 Gus chunks, 280000 decisions, 748305 legal actions, and W&B run `74vfet6o` with 20 progress points.
- Hidden-holder impact is pervasive in this corpus: 85.16% of legal actions have top hidden-impact score at least `5.0`; mean top impact is `17.46`, and max top impact is `85.04`.
- The cleanest mitigation contrast is close-mean hidden downside. Across 4619 same-decision pairs, choosing the lower hidden-downside action changes mean by only `-0.021` Q with CI crossing zero, while reducing lower-tail mass by `-0.033` and hidden-downside score by `-7.49`.
- Broader safest-tail and top-threshold choices expose the tradeoff surface: tail safety can cost mean/threshold mass, and threshold-mass improvements can worsen lower-tail exposure. The result is diagnostic branch-impact evidence, not a central claim-status promotion.

---

## [2026-05-03 | local | w42 dynamic 84 branch lab]

**Touched pages:** [[w42]] [[w42-phase2-84-weapon-preservation-probe]] [[index]] [[log]]
**Added:** 1 dynamic artifact directory - `w42/eighty_four_weapon_preservation_probe/dynamic_branch_lab/`; 1 runner - `w42/eighty_four_weapon_preservation_probe/run_dynamic_branch_lab.py`.
**Updated:** [[w42]] records the generated 84 follow-up; [[w42-phase2-84-weapon-preservation-probe]] now includes the dynamic branch lab and W&B run; [[index]] updates the catalog description.
**Retired:** none.
**Questions opened:** how to inject exact late-hand states, or mine many seeds, so same-suit-pair and last-trick stopper claims are no longer limited by the greedy policy trace.

**Frontier shift:**
- Bead `t42-0b4l.7` now has explicit bid-84 generated data rather than only static shape evidence. Six constructed hands cover laydown control, protected one-off, straight-off named stopper threat, two-off ordering, same-suit pair/protector pressure, and dead-asset release.
- The run uses schema-v2 E[Q] generation with `bid_value=84`: 168 decision states, 584 legal action rows, 4497 hidden-threat rows, and W&B run `f7uzoo7f`.
- Dynamic proxy labels find 10 live double weapon actions, 21 live same-suit pair actions, 4 pair-protector actions, 7 dead-asset release candidates, 13 preserve-vs-spend paired decisions, and 23 offense trump-vs-final-off paired decisions.
- The first preserve/spend contrast is deliberately small and conservative: preserving lower-asset alternatives is `+0.136` mean Q with `+0.005` threshold-mass delta across 13 same-decision pairs. This proves the measurement surface exists, not that the broad book claim is settled.
- No central claim status changes. The next credible step is either explicit late-hand state injection or seed mining at scale for powered 84 endgame tableaux.

---

## [2026-05-03 | local | w42 bid-only-enough margin probe]

**Touched pages:** [[w42]] [[w42-bidding-risk-budget-claim-validation]] [[index]] [[log]]
**Added:** 1 artifact directory - `w42/bid_only_enough_claim_tests/`.
**Updated:** [[w42]] records the Chapter 2 bid-margin follow-up; [[w42-bidding-risk-budget-claim-validation]] now includes the generated counterfactual probe.
**Retired:** none.
**Questions opened:** how to add real or simulated auction histories so bid-only-enough can be tested with bidder seat, score, partner/opponent bids, pass/bid policy, and opponent response.

**Frontier shift:**
- Bead `t42-0b4l.5` now has a partial empirical probe instead of only a missing-data specification.
- The run generates 12 arbitrary hands, chooses two static-best pip declarations per hand, simulates 96 play outcomes for each fixed hand/declaration, and expands those same outcome samples across current-high-bid and actual-bid thresholds.
- The artifact has 24 hand/declaration rows, 1224 bid counterfactual rows, and 1104 positive unnecessary-margin contrasts. Mean delta versus minimum winning bid is `-0.111555` `P(make)` and `-0.223109` one-mark swing; 1040 contrasts worsen, 64 tie, and 0 improve.
- The result supports the same-contract arithmetic direction of "bid only enough," but it does not promote full auction discipline, natural bid buckets, partner bid signals, or opponent response claims.

---

## [2026-05-03 | local | w42 seat-position claim tests]

**Touched pages:** [[w42]] [[w42-phase2-seat-position-strategy-map]] [[index]] [[log]]
**Added:** 1 artifact directory - `w42/seat_position_claim_tests/`.
**Updated:** [[w42]] records the row-level seat/position follow-up; [[w42-phase2-seat-position-strategy-map]] now includes results, artifacts, commands, and caveats.
**Retired:** none.
**Questions opened:** which role/position claims need sequence counterfactuals or model probes because all legal actions in one decision share the same structural seat context.

**Frontier shift:**
- Bead `t42-0b4l.6` now turns the seat-position map into detector labels and tests over the 75,079 full legal-action rows from the tactical replication.
- The run emits 37 label metric rows, 36 role-position slice rows, 8 paired contrasts, 160 paired contrast slices, compact labeled rows, and examples.
- Structural seat/phase/role labels are slice evidence only. Action-local labels pair cleanly: defensive pounce count is `+3.85` Q over 489 pairs, pounce closure is `+6.76` Q over 128 pairs, partner support count is `+0.61` Q over 614 pairs, unsupported partner count into defense is `-8.35` Q over 802 pairs, closure take-trick is `+10.94` Q over 808 pairs, closure take-count is `+4.67` Q over 277 pairs, and closure slough-count is `-6.38` Q over 1803 pairs.
- No central claim status changes. The result is closeable as seat/position row-level evidence and as input to the claim-tag model probe.

---

## [2026-05-03 | local | w42 claim-tag model probe]

**Touched pages:** [[w42]] [[w42-claim-tag-model-probe]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-claim-tag-model-probe]]; 1 artifact directory - `w42/claim_tag_model_probe/`.
**Updated:** [[w42]] records the direct model-probe follow-up; [[index]] catalogues the new page.
**Retired:** none.
**Questions opened:** how to normalize bid-risk, 84/endgame, doubles/no-trump, and public-safe hidden-threat proxies into one future row-model table.

**Frontier shift:**
- Bead `t42-0b4l.9` now has a direct legal-action-row model probe rather than only a proxy rich-tag rerun.
- The train/eval split uses `w42/seat_position_claim_tests/labeled_action_rows.csv`, with seeds `0..79` train and `80..99` eval: 59,893 train actions / 22,400 train decisions and 15,186 eval actions / 5,600 eval decisions.
- The primary run improves from `1.483` public-feature selected mean regret and `61.70%` oracle-best match to `1.319` regret and `63.64%` match with claim tags. Tail regret `>=5` drops from `10.43%` to `8.95%`. W&B run `gn7xxk14` logs the per-variant summary series.
- Family drops are modest but directional: dropping pounce/donation worsens mean regret by `+0.098` versus the full tag model, while dropping seat-position/closure worsens by `+0.053`.
- Bid-risk, 84/endgame, doubles/no-trump, and hidden-threat/distribution are documented as unavailable or eval-only for this training table. No claim-ledger status changes.

---

## [2026-05-03 | local | w42 claim-analysis synthesis]

**Touched pages:** [[w42]] [[w42-claim-analysis-synthesis-report]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-claim-analysis-synthesis-report]]; 1 artifact directory - `w42/claim_analysis_synthesis/`; 1 phase-3 epic - `t42-qtwb`.
**Updated:** [[w42]] records the phase-2 synthesis and phase-3 route; [[index]] catalogues the synthesis report.
**Retired:** none.
**Questions opened:** none.

**Frontier shift:**
- Bead `t42-0b4l.10` synthesizes all phase-2 child results after `.5`, `.6`, and `.9` closed.
- The synthesis table records every major family: claim matrix/harness, tactical pounce/donation, hidden threat, bidding risk, seat/position, 84, doubles/no-trump, model probe, and data inventory.
- No broad central claim-ledger status changes are made in the final synthesis. Narrow supports stay tied to their operationalized pages; broad book claims move to generated counterfactual phase-3 work.
- New epic `t42-qtwb` owns four next beads: auction-aware bid discipline, 84 endgame state injection, sequence/seat counterfactuals, and a joined claim-row model table.

---

## [2026-05-03 | local | w42 phase3 auction bid discipline corpus]

**Touched pages:** [[w42]] [[w42-bidding-risk-budget-claim-validation]] [[w42-phase3-auction-bid-discipline-corpus]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-phase3-auction-bid-discipline-corpus]]; 1 artifact directory - `w42/auction_bid_discipline_claim_tests/`.
**Updated:** [[w42]] records the first `t42-qtwb` follow-up; [[w42-bidding-risk-budget-claim-validation]] now links the phase-3 auction-pressure corpus; [[index]] catalogues the new page.
**Retired:** none.
**Questions opened:** what real auction policy or logged auction source can replace the generated partner/opponent pass-value proxy.

**Frontier shift:**
- Bead `t42-qtwb.1` now has a generated auction-aware corpus rather than only a phase-3 plan.
- The run evaluates 32 generated deals, 384 seat/declaration contract labels, 608 auction contexts, and 16800 P0 bid-action rows, with W&B run `6cup1bat`.
- Bid-only-enough survives the stronger auction-pressure operationalization: across 14976 positive-margin bid rows, 12872 worsen, 2104 tie, and 0 improve versus the minimum winning bid.
- Natural bid buckets are now measurable but not broadly promoted: 68 / 384 contract labels land on the chapter's natural max-profitable threshold buckets.
- Partner bid signal is conditional under the generated proxy. P0 overcalls partner in 77 / 192 partner-high contexts, so the evidence supports a nuanced behavioral signal, not a static "partner bid means bid higher" rule.
- No central claim-ledger status changes. The result strengthens local support for the operational bid-only-enough slice and routes partner-signal claims toward real auction-policy data.

---

## [2026-05-03 | local | w42 phase3 84 seed mining corpus]

**Touched pages:** [[w42]] [[w42-phase2-84-weapon-preservation-probe]] [[w42-phase3-84-seed-mining-corpus]] [[w42-claim-analysis-synthesis-report]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-phase3-84-seed-mining-corpus]]; 1 artifact directory - `w42/eighty_four_seed_mining/`.
**Updated:** [[w42]] records the phase-3 84 seed-mining follow-up; [[w42-phase2-84-weapon-preservation-probe]] now points from the six-fixture dynamic lab to the natural seed corpus; [[w42-claim-analysis-synthesis-report]] records the closed `.2` route.
**Retired:** none.
**Questions opened:** how to turn the mined natural seeds into exact late-hand state injections or larger branch-atlas preserve/spend contrasts.

**Frontier shift:**
- Bead `t42-qtwb.2` closes through the documented seed-mining corpus path rather than arbitrary late-state injection.
- The scan covers 50000 seeds, 4 seats, and 7 pip declarations: 1400000 seat/declaration checks.
- The artifact emits 214229 candidate rows, 43013 unique candidate seeds, 256 recommended rows, and W&B run `f33g4fy1`.
- Mined surfaces include 22176 protected one-off rows, 8330 straight one-off rows, 183722 two-off same-suit rows, 140264 defender live-double rows, 206213 defender same-suit-pair rows, 85753 pair-protector-pressure rows, and one natural all-trump laydown.
- No central claim-ledger status changes. The result removes the hand-built-only blocker and gives phase 3 a natural seed menu for 84 branch-atlas/state-injection work.

---

## [2026-05-03 | local | w42 phase3 sequence seat counterfactuals]

**Touched pages:** [[w42]] [[w42-phase3-sequence-seat-counterfactuals]] [[winning42-ch03-bidder-play]] [[w42-claim-analysis-synthesis-report]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-phase3-sequence-seat-counterfactuals]]; 1 artifact directory - `w42/sequence_seat_counterfactuals/`.
**Updated:** [[w42]] records the `t42-qtwb.3` sequence/seat follow-up; [[winning42-ch03-bidder-play]] records the lead-plan and partner-support nuance; [[w42-claim-analysis-synthesis-report]] records the closed `.3` route; [[index]] catalogues the new page.
**Retired:** none.
**Questions opened:** how to add hand-shape/state-injection gates for trump count, off count, reentry preservation, early-off exceptions, and low-trump command exceptions.

**Frontier shift:**
- Bead `t42-qtwb.3` now has a phase-3 branch-value counterfactual artifact over the 75079 legal-action rows from the tactical replication.
- The run emits 50 label metric rows, 14 paired contrasts, 394 paired contrast slice rows, compact labeled sequence rows, examples, and W&B run `kp0otpdo`.
- Follow-seat control is strongly supported: taking control from an opponent is `+11.564` Q across 1870 paired states, while count into opponent control is `-7.322` Q across 3526 pairs.
- Last-seat closure and setter pounce remain robust: closure take-trick is `+10.936` Q across 808 pairs, closure slough-count is `-6.377` Q across 1803 pairs, setter pounce count is `+3.846` Q across 489 pairs, and reckless setter count to bidder is `-7.829` Q across 2027 pairs.
- Partner count support is real but gated: bidder-side control count is `+0.608` Q across 614 pairs, while partner count into defensive control is `-8.351` Q across 802 pairs.
- Bidder lead sequencing remains context-limited rather than broadly promoted. Called-suit lead versus off-suit lead is `-1.935` Q across 1657 pairs, but called double beats lower called-suit lead by `+5.720` Q across 182 pairs. The next lead-plan test needs explicit trump/off/reentry/live-count gates.

---

## [2026-05-03 | local | w42 phase3 joined claim row model table]

**Touched pages:** [[w42]] [[w42-phase3-joined-claim-row-model-table]] [[w42-claim-tag-model-probe]] [[w42-claim-analysis-synthesis-report]] [[index]] [[log]]
**Added:** 1 experiment page - [[w42-phase3-joined-claim-row-model-table]]; 1 artifact directory - `w42/joined_claim_row_model_table/`.
**Updated:** [[w42]] records the `t42-qtwb.4` joined-table close; [[w42-claim-tag-model-probe]] points to the phase-3 continuation; [[w42-claim-analysis-synthesis-report]] records the `.4` result; [[index]] catalogues the new page.
**Retired:** none.
**Questions opened:** whether hidden public proxies need belief-calibrated features before they should be used by a row model.

**Frontier shift:**
- Bead `t42-qtwb.4` now has a joined public-safe legal-action table over 75079 rows and 28000 decisions.
- The table joins sequence/seat labels, partial auction-risk labels, partial public 84 bidder-structure labels, doubles/no-trump declaration/action labels, and hidden public pressure proxies. Hidden-owner truth, 84 defender assets, and auction partner/opponent pass values remain eval-only.
- On the seed-mod held-out split, public features alone score `1.3598` mean regret and `64.464%` best-mean match. All public claim families improve that to `1.1257` mean regret and `68.107%` best-mean match; tail regret `>=5` falls from `9.536%` to `7.250%`.
- Family drops identify sequence/seat as the main signal: dropping it gives `1.3678` mean regret, worse than the public baseline. Dropping bidding risk gives `1.1461`; dropping public 84 gives `1.1415`; dropping doubles/no-trump is nearly neutral; dropping hidden public proxy slightly improves to `1.1179`.
- W&B run `jvjuld7m` logs the variant series and artifacts. No central claim-ledger status changes.

---

## [2026-05-03 | local | w42 phase4 book-claim test sweep]

**Touched pages:** [[w42]] [[w42-phase4-sequence-handshape-tests]] [[w42-phase4-84-dynamic-seed-tests]] [[w42-phase4-doubles-notrump-regime-tests]] [[w42-phase4-laydown-rule-accounting]] [[w42-phase4-scoring-objective-tests]] [[w42-phase4-claim-completion-board]] [[w42-phase4-bidding-count-exposure-tests]] [[winning42-ch01-in-a-nutshell]] [[winning42-ch02-bidding]] [[winning42-ch03-bidder-play]] [[winning42-ch04-partner-support]] [[winning42-ch05-setter-defense]] [[winning42-ch07-taking-every-trick-84]] [[winning42-ch08-setting-84]] [[winning42-ch09-doubles-no-trump]] [[winning42-ch10-tournament-scoring]] [[winning42-ch12-advanced-bidding-playing]] [[winning42-ch16-statistical-odds]] [[index]] [[log]]
**Added:** 7 experiment pages - [[w42-phase4-sequence-handshape-tests]], [[w42-phase4-84-dynamic-seed-tests]], [[w42-phase4-doubles-notrump-regime-tests]], [[w42-phase4-laydown-rule-accounting]], [[w42-phase4-scoring-objective-tests]], [[w42-phase4-claim-completion-board]], [[w42-phase4-bidding-count-exposure-tests]]; 7 artifact directories under `w42/phase4_*`.
**Updated:** [[w42]] records the phase-4 sweep; chapter pages record the strongest supported/context-limited/blocker changes; [[index]] catalogues the new pages.
**Retired:** none.
**Questions opened:** which remaining blocker rows deserve arbitrary state injection, high-bid contract generation, real auction-policy data, or human/tournament population data.

**Frontier shift:**
- Phase 4 extends `t42-br7n` from a plan into seven concrete claim-test artifacts plus a completion board over all 64 ledger claims.
- The tactical lane sharpens the core folk rule: commanding called doubles beat off leads by `+1.316` Q, while generic non-double called-suit leads lose by `-3.682` Q. Partner closure count donation is strong, earlier donation is weak, partner count-liability leads are bad, and setter pounce/count-calling labels remain strong.
- The 84 lane turns mined natural seeds into reached-state action evidence: preserve expendable versus spend live assets is `+1.946` Q, dead-asset release is `+0.504` Q, and bidder trump-pull before final-off is `+4.295` Q. Final set attribution, full throwaway ladder, score 42-vs-84, and "good player" straight-off rates remain blocked.
- The Chapter 9 lane finally tests same-hand doubles-trump versus no-trump regimes: no-trump beats doubles-trump on most four-plus-double hands, while high/top double control is the near-flat pro-doubles slice.
- The deterministic lane verifies 29 Chapter 1 / laydown assertions with zero failures and turns the Chapter 3 final-deuce warning into a concrete rejected-laydown counterexample.
- The scoring lane supports core Chapter 10 mechanics: marks create early terminal states in `94.5%` of generated hands, erase defender partial points in made ordinary contracts, compress ordinary set severity, and disagree with point-score winners in about `15.5%` of proxy matches.
- The bidding/count-exposure lane closes the completion-board scope gap: three-plus trumps are much stronger in generated contracts, risk <=12 is modestly better, four/five offs are worse, natural max-profitable buckets appear in 66 / 384 rows, partner two-plus-double prior is about `58%`, and double-side protection is explicitly side-limited.
- No broad central ledger mutation is made by the wiki update. Rows are promoted only at the page/evidence level; exact private-state, high-bid, auction-policy, state-injection, and human/tournament population blockers remain explicit.
- [[w42-phase4-final-claim-audit]] independently confirms the closure condition: all 64 ledger rows have evidence and/or explicit bounded blockers, with no no-evidence/no-blocker rows.
- [[w42-book-claim-synthesis-and-ai-directions]] records the post-closure synthesis: confirmed claims, unconfirmed technical blockers, distribution-aware E[Q] alternatives, and model/Burl/Gus experiment directions.

---

## [2026-05-03 | local | w42 book validation v1 wave 1]

**Touched pages:** [[w42]] [[w42-book-claim-synthesis-and-ai-directions]] [[w42-bookval-v1-wave1-distribution-lens-reranker]] [[w42-bookval-v1-wave1-mark-utility-transform]] [[w42-bookval-v1-wave1-hidden-threat-impact-ranker]] [[w42-bookval-v1-wave1-cross-ai-agreement]] [[w42-bookval-v1-wave1-independent-audit]] [[index]] [[log]]
**Added:** 5 experiment pages — wave 1 of the book-validation campaign (distribution-lens reranker, mark-utility transform, hidden-threat impact ranker, cross-AI agreement, independent audit). Artifacts under `w42/book_validation_v1/wave1/<bead>_<slug>/`. Wave 0 baseline + agent rules of engagement under `w42/book_validation_v1/`.
**Updated:** [[w42-book-claim-synthesis-and-ai-directions]] absorbs Wave 1's structural findings and ledger absorptions; `w42/statistics_claims_ledger/claims.csv` and `w42/phase4_claim_completion_board/completion_board.csv` move two Ch 10 rows to context-limited; `w42/phase4_scoring_objective_tests/claim_summary.csv` normalizes a non-vocabulary status string. Status counts after wave 1: supported 23, context-limited 14, underpowered 20, not-yet-tested 5, contradicted 2.
**Retired:** none.
**Questions opened:** can a bid-aware E[Q] generator land in time to unblock Ch 10 mark-multiplier tests; how to refine the three over-firing detectors (`ch05_reckless_count`, `ch03_called_non_double`, `ch05_setter_pressure_regime`) without losing their within-pair contrast value.

**Frontier shift:**
- Tail-aware utilities (CVaR_10, robust_q25) agree with EV at 82-83% on the seed-9430 branch atlas, while p_make / threshold_mass only agree at 59%. The book's "manage tail risk" framing is more aligned with optimal play than its "make first" framing.
- mark_ev is algebraically identical to p_make at bid=30 with one-mark multiplier. Genuine mark-vs-point flips are 3.6% of decisions; 81.5% of nominal flips are surface-flattening artifacts. Bid-aware E[Q] generation is now a hard prerequisite for further mark-objective work.
- Hidden-threat attribution surfaces concrete training targets: trump-count tiles are 100% directionally helpful as load-bearing tiles; 5-5 in twos and 4-4 in no-trump are the highest-impact non-trump load-bearing tiles; right- and left-setter seats are asymmetric in load-bearing-tile category, contradicting the book's symmetric Ch 5 treatment.
- Two Ch 10 ledger rows promoted to context-limited (`ch10-point-system-skill-signal`, `ch10-timed-marks-advancement-objective`) after the independent audit found phase-4 worker evidence the prior audit had not absorbed. The `ch10-tournament-speed-tradeoff` worker artifact's non-vocabulary status string was normalized.
- Three detectors flagged for refinement (not retirement): `ch05_reckless_count` (mean regret 9.18 over 2,300 fires - overfires beyond qualifying window); `ch03_called_non_double` (within-pair contrast valid, action-level endorsement wrong 86% of the time); `ch05_setter_pressure_regime` (regime label, not action label).

---

## [2026-05-03 | local | w42 book validation v1 wave 2 infra]

**Touched pages:** [[w42]] [[w42-book-claim-synthesis-and-ai-directions]] [[w42-bookval-v1-wave2-infra-design]] [[w42-bookval-v1-wave2-reentry-preservation]] [[w42-bookval-v1-wave2-bid-aware-atlas]] [[w42-book-validation-campaign]] [[index]] [[log]]
**Added:** 3 experiment pages — Wave 2 infra design, Wave 2.A reentry preservation probe, Wave 2.B bid-aware E[Q] atlas. Forge core extensions: `GameStateTensor.from_snapshot/to_snapshot`, `generate_eq_from_snapshots`, `--snapshot-file` CLI flag (commit 19fc675). W42-side bid-aware driver and 5,550-row joined action table (commit 0c802d4).
**Updated:** [[w42-book-claim-synthesis-and-ai-directions]] absorbs Wave 2 infra-build findings and the cross-bid `mark_ev` divergence. [[w42-book-validation-campaign]] marks Waves 2.A and 2.B closed.
**Retired:** none.
**Questions opened:** when does the 50-seed × 1000-sample CUDA sweep land for ledger-moving statistical power; should the reentry-preservation corpus be rebuilt on oracle-greedy trajectories before downstream Ch 03 work treats its `underpowered` direction as evidence; does bid=84 strategic interpretation require filtering to ≥4-double hands or accept the engine's permissive enforcement.

**Frontier shift:**
- `forge.eq.GameStateTensor` gains a public mid-game snapshot interface. The forge engine already produced these states via `apply_actions`; what was missing was the entry-point constructor, and the `from_snapshot`/`to_snapshot` round-trip plus 6 new tests prove the schema (`forge.eq.snapshot.v1`) is consistent.
- The first state-injection probe (200 reentry-shape snapshots, bid=30, M5 MPS, 50 samples) returns direction `consume > preserve` with EV delta `-1.62` (CI `[-2.93, -0.31]`), correctly self-classified as `underpowered`. Random-play context bias dominates; oracle-greedy trajectory regeneration is the next prerequisite for promotion.
- Bid-aware E[Q] generation now exists W42-side (forge already supported `--bid-values`). At bid=30 vs `branch_atlas_scaled_v0`, 10/10 decl_id pairs match within sampling noise; aggregate mean-EV is the correct validation contract because two stochastic oracle runs diverge in trajectory after decision 0.
- Cross-bid `mark_ev` divergence breaks the Wave 1.2 algebraic-identity finding for inter-bid comparisons: 56-63% of actual-action mark_evs change at bids 32-42 vs bid=30; 100% change at bid=84. The structural mechanism is that the Q-space threshold for "made" rises with bid (`tq_off = 2·bid - 42`), so even with multiplier=1 the mark utility distribution shifts.
- Wave 2 unblocks Wave 2.C-2.H (six paired-bid / state-injection probes for Ch 02 / Ch 04 / Ch 05 / Ch 08 / Ch 10 / Ch 12). Smoke-scope evidence is sufficient to start; ledger-moving statistical power requires the 50-seed CUDA sweep.
## [2026-04-27 | 1f11d28 | burl-perf-phase0 — measurement harness for the speedup sprint]

The foundation under [[topics/perf-on-the-table]]: a tracked bench that drives the production batched eval path against a frozen 5-decision subset and records per-decision wall, prefill/decode tok-s, peak memory, and a K1-grade-match-pct vs the latest baseline-bf16 ledger row. Phases 1–3 (cheap wins, continuous batching + prefix sharing, speculative decoding + quantization) thread their levers through `--variant <name>` and write a row each.

**Touched pages:** [[experiments/burl-perf-phase0]] [[topics/perf-on-the-table]] [[index]]

**Added:** [[experiments/burl-perf-phase0]] — frontmatter + Subset rationale + Measurements + Determinism notes + Pointers.

**Updated:**
- [[topics/perf-on-the-table]] gained a "Measurement harness" section with the canonical baseline-bf16 numbers (79.5s/5dec, 87 decode tok/s, 11.59 GB peak) and the Phase-1+ noise floor read; bumped `last_updated` to `1f11d28`.
- [[index]] experiments catalog gained the new page.

**Frontier shift:** Names the bench, freezes its inputs (`burl/eval/data/perf_subset_5.jsonl` covers gi=0/36/72/104/136 = trick positions 1/3/5/6/7 across declarations 0..4), publishes the canonical baseline-bf16 row at `1f11d28`, and documents the temp=0.6 noise floor (wall ±0.5%, decode tok/s ±9%, K1 grade match 80–100% on the 5-row subset). The 5-row floor isn't tight enough to confirm sub-10% regret deltas; that's why the bench also accepts `--subset 560` for the phase-exit gate. Promotes `gus_eval_bridge.py` from `scratch/belief_trajectory_rollout/diagnostic/` to `burl/eval/` so tracked benches can resolve `global_idx → BurlDecision` without sourcing from scratch.

**Why now:** three other scribes (B, A, C) are blocked on this row. Phase 0 had to land first because every later finding hangs on the bench's accuracy, and all three downstream scribes need the same frozen subset + ledger schema to write rows comparable across runs.

---

## [2026-04-27 | 29da3d2 | burl-perf-phase2 — continuous batching ships at 1.8–2.1×, prefix-cache closes negative]

Scribe A's closeout: lever 1 (LRUPromptCache prefix sharing) and lever 2 (continuous batching) for [[topics/perf-on-the-table]]. Lever 1 closed at 0× on M5 Max — heterogeneous-cache batched decode pads to the longest cache and chat-template re-rendering breaks key alignment, costing both speed (84 → 45 decode tok/s) and correctness (60% K1 match). Lever 2 lands at **1.8–2.1× wall** on the bench's 5-row temp=0 subset (71 s → 34–40 s) via a `BatchGenerator`-backed dispatcher (`run_bench_continuous` in `burl/eval/bench_decision_latency.py`).

**Touched pages:** [[experiments/burl-perf-phase2]] [[topics/perf-on-the-table]] [[questions/open]] [[index]]

**Added:** [[experiments/burl-perf-phase2]] — full lever-1 negative-result writeup + lever-2 results table + production-harvest migration plan.

**Updated:**
- [[topics/perf-on-the-table]]: levers 1 and 2 in the ranked list now carry their measured outcomes; bumped `last_updated`.
- [[questions/open]]: filed the question of how the [[topics/batched-harvest-resilience]] wave-sentinel + quarantine layer migrates onto a continuous dispatcher.
- [[index]]: added phase-2 experiment.

**Frontier shift:** Names the perf-on-the-table prediction wrong on lever 1 (the "1.5–2× expected" line was based on the single-stream argument that doesn't hold when the batched decode has to pad heterogeneous KV widths), and right on lever 2 (the "3–5×" estimate; the bench measured 1.8–2.1× on the 5-row subset which is the conservative lower-bound — the harvest-level straggler tail savings will be larger). The compounded realistic stack at the wiki's calibration calculus drops from 7–10× to roughly 1.8 × Phase 1 × Phase 3.

**Why now:** Phase 2 had a 90-min budget; lever 1 burned about half of it on the negative-result loop, lever 2 landed cleanly in the second half. Closing in the same session keeps the wiki + ledger consistent before scribe-C's Phase 3 work picks up.

---

## [2026-04-27 | 0310b12 | phase-2 caveat — GPU-contention disclosed, root-cause writeup, dispatcher design]

Team-lead paused new bench runs because cross-scribe GPU contention with scribe-B's parallel mlx-lm batch=5 jobs is consistent with the same-config baseline-t0 walking 71 → 73.7 s and the lever-1 prefix-cache rows hitting 114 s and 137 s.  All Phase-2 wall-time magnitudes are pending re-validation.  Used the pause window for non-GPU work: deeper mlx-lm source audit, Lever-1 root-cause writeup, continuous-batching dispatcher design doc.

**Touched pages:** [[experiments/burl-perf-phase2]] [[topics/perf-on-the-table]] [[topics/continuous-batching-dispatcher-design]] [[index]] [[log]]

**Added:** [[topics/continuous-batching-dispatcher-design]] — submit / pump / close API for a `ContinuousDispatcher` class wrapping mlx-lm's `BatchGenerator`; cohort-based OOM resilience that preserves [[topics/batched-harvest-resilience]]'s wave-sentinel + quarantine semantics under continuous batching; explicit acknowledgement that the current `run_bench_continuous` is inline and the class extraction is a follow-up refactor.

**Updated:**
- [[experiments/burl-perf-phase2]] gained a top-of-page contention caveat and a "Lever 1 — root-cause writeup" section diagnosing the two structural failure modes (`_merge_caches` heterogeneous-pad penalty + chat-template re-rendering vs trie-key alignment) with line-cited sources from `mlx_lm/models/cache.py` and the Gemma 4 `chat_template.jinja`.
- [[topics/perf-on-the-table]] gained a "What mlx-lm 0.31.2 actually exposes" section: `batch_generate` / `BatchGenerator` / `LRUPromptCache` API surface, the constraints that ruled the naive Lever-1 shape out (model-key hashability, `BatchKVCache.merge` padding, `prefill_batch_size=8` broadcast bug, chat-template structured re-extraction), and the "right pattern for prefix reuse" derived from `mlx_lm.server.py`.  Compounded-stack estimate revised from 7–10× down to ~4–6× for the Phase-2 contribution.
- Lever-1 and Lever-2 entries in the ranked list now flag wall magnitudes as contention-suspect while preserving the contention-independent structural claims.

**Frontier shift:** The Phase-2 *direction* (Lever 1 fails, Lever 2 wins) survives the caveat because it rests on structural mlx-lm properties, not measured wall.  The *magnitude* (1.8–2.1× for Lever 2) is parked pending clean re-run.  The compounded realistic stack at the wiki's calibration calculus drops from 7–10× to roughly 1.5 × Phase 1 × Phase 3 — which means Phase 1 and Phase 3 carry more of the speedup load than the original perf-table estimate assumed.

**Why now:** Pausing on GPU lets the wiki absorb what was learned without churning the conclusion when the clean re-run lands.  The structural arguments stay; the numbers will be replaced.

---

## [2026-04-27 | unstaged | burl-perf-phase3 — research-mode spec, two dragons disclosed]

Scribe-C's Phase-3 research-mode landing: full spec page for the speculative-decoding + quantization phase, written during the GPU pause (scribe-B holds the bench).  Two structural findings forced the spec to bend before any wall measurement:

1. **mlx-lm 0.31.2 spec decode is single-stream-only.**  `speculative_generate_step` (`mlx_lm/generate.py:473`) plumbs through `stream_generate` and the CLI/HTTP server, but **not** through `batch_generate` / `BatchGenerator` — the path [[burl-perf-phase2]] just landed.  Confirmed independently via LM Studio's MLX engine raising `SpeculativeDecodingNotSupportedError` even at batch=1 (lmstudio-ai issues #269, #1519).  Consequence: spec decode cannot stack on Phase-2's continuous batching; choosing it means accepting single-stream inference and losing Phase-2's parallelism win.  Anchor numbers: 43 tok/s single-stream vs ~84 tok/s batched — spec decode needs >2× to catch and >4× to win.
2. **There is no Gemma 4 E0.5B.**  `google/gemma-4-E2B-it` is the smallest Gemma 4 release; the team-lead spec assumed an E0.5B based on [[perf-on-the-table]]'s old draft-model line which was speculative.  Viable drafts narrow to Gemma 3 270M IT (vocab 262144 = Gemma 4 E2B's 262144 — passes mlx-lm's `server.py:354` validator, but token id alignment is unverified across the gemma3_text → gemma4 family boundary; a 5-min tokenizer probe gates the variant) and self-speculation (Q4-E2B drafting bf16-E2B) as fallback.

Quant work is also dragon-rich: every `mlx-community/gemma-4-*-{4,8}bit` and the original `unsloth/gemma-4-*-MLX-{4,8}bit` quants produce garbage output because they quantize PLE (Per-Layer Embeddings) layers — Gemma 4's PLE uses ScaledLinear with output multipliers that amplify quant error.  PLE-safe quants are released in `FakeRocket543/gemma-4-e2b-it-MLX-{4bit,8bit,bf16}` and `unsloth/gemma-4-E2B-it-UD-MLX-4bit` (note: Unsloth's *non-UD* MLX-4bit is in the broken set).

**Touched pages:** [[burl-perf-phase3]] [[index]]

**Added:** [[burl-perf-phase3]] — full Phase-3 spec covering eight named variants (`q8-bf16-cont`, `q4-mlx-cont`, `q4-kvq8-cont`, `q6-mxfp-cont`, `spec-stream-bf16`, `spec-stream-q8`, `spec-stream-self`, `phase3-stack-best`), tradeoff matrix template (rows = variants, columns = wall / k1 / regret / mem / complexity), validation bar (≥4/5 K1 + regret Δ ±10% vs Phase-2's continuous baseline; tighter ±5% on Q4 to catch the quality cliff), spec-decode acceptance instrumentation plan (`was_drafted` per token, segmented by thought / tool-call / text), full test plan with hard time-boxes (tokenizer probe → quant smoke → quant benches → spec benches → stack), memory + complexity tradeoffs for the morning digest, and three open questions on mlx-lm's spec-decode internals.

**Updated:**
- [[index]]: added [[burl-perf-phase3]] entry with the dragon summary.

**Frontier shift:** Phase 3's compounded ceiling is *forced to a choice* by Dragon 1 — spec decode and continuous batching cannot stack in mlx-lm 0.31.2, so the headline `phase3-stack-best` row is necessarily a quant-only stack on top of Phase-2's continuous-batching win.  Spec decode is benched as an alternative path (single-stream + draft) and the two paths' compounded numbers will be reported side-by-side to scribe-team-lead.

**Why now:** Research-mode pause is the right time to land the spec — the dragon list is the load-bearing finding (named variants + validation bar are easy without GPU; the dragons require web search + mlx-lm source audit).  The ledger row format and validation gate are settled before any bench so the later wall-time write-up doesn't have to argue with the framing.

---

## [2026-04-27 | 96ebf0b | burl-perf-phase3 — Q4 PLE-safe ships, spec-decode dies on tokenizer probe]

Scribe-C's Phase-3 execution under exclusive GPU.  Headline: **Q4 PLE-safe (`FakeRockert543/gemma-4-e2b-it-MLX-4bit`) is the production pick** — drop-in `--model-repo` swap, no other code changes, 4/5 paired play match vs bf16 at temp=0 (single divergence at gi=0 marginal-decision slot bf16 itself flips across runs), 1.3–2.3× paired wall delta (1.29× on cleaner pair, 2.34× on noisier pair), peak mem 8.93 GB vs 10.80 GB bf16 (−17%).  `phase3-stack-best` row: **28.3 s wall** vs paired baseline 36.4 s, decode 87.7 tok/s.

**Speculative-decoding lever shelved before any GPU bench.**  Tokenizer compat probe (5 plain-text + special-token Burl prompts) showed Gemma 3 270M IT (the only viable smaller draft on the same vocab 262144) collapses Gemma 4's special tokens (`<|tool_call>`, `<|channel>`, `<channel|>`, `<tool_call|>`) into byte-fallback subword sequences.  Burl's outputs are dominated by these tokens (every assistant turn opens with `<|channel>thought` and closes with `<|tool_call>`), so a draft model whose tokenizer can't emit them as single tokens hits acceptance rate ≈ 0 on the highest-acceptance regions — exactly the regions spec decode would *most* want to win.  Combined with mlx-lm 0.31.2's spec-decode-not-supported-on-batched constraint (would surrender Phase-2's parallelism to use), the lever is closed for this harness shape.

Q8 PLE-safe ran *faster* than Q4 in raw decode tok/s (90.8 vs 62.2) but **lost on quality** (3/5 paired play match) and on memory (9.83 GB vs 9.17 GB).  Worse pick than Q4 on both axes — the M5 Max's memory-bandwidth-bound regime favors the smaller-weights variant despite same compute.

**Touched pages:** [[burl-perf-phase3]] [[perf-on-the-table]] [[index]]

**Updated:**
- [[burl-perf-phase3]] flipped `status: spec` → `status: active`, added a tradeoff matrix with real numbers, validation-against-bar table, paired-comparison play-match audit, three execution-time dragon notes (Gemma 3 tokenizer probe, HF username typo `FakeRockert543` not `FakeRocket543`, gi=0 marginal-decision noise widening the K1 gate), bumped `last_updated` to 96ebf0b.
- [[perf-on-the-table]] lever 4 (speculative decoding) marked closed with the three-way kill writeup; lever 6 (quantization) marked confirmed at 1.3–2.3× wall + 25% memory cut, citing `phase3-stack-best` numbers.
- [[index]] hook on [[burl-perf-phase3]] reflects the result.

**Frontier shift:** The compounded perf-table stack is now: Phase-1 turn-aware tokens × Phase-2 continuous batching × Phase-3 Q4 PLE-safe.  Speculative decoding is shelved permanently for this harness shape; the parallelism win and the spec-decode win are mutually exclusive in mlx-lm 0.31.2, and continuous-batching wins on Burl's workload (heterogeneous turn-counts make the straggler-tail savings substantial).  Phase-4 full-560 will pin the absolute compounded multiplier; the 5-row results are *suggestive directional evidence*.

**Why now:** Headline + writeup land in the same session per the wiki "update is a side effect" rule.  The bench rows are durably in `burl/eval/results/perf_ledger.csv`; the per-run JSON detail captures step-stats for downstream analysis; the wiki frontier reflects the current truth.

---

## [2026-04-27 | unstaged | burl-perf-phase3 — Unsloth UD cross-check + PLE landmine doc]

Scribe-C's Phase-3 follow-up under continued exclusive GPU.  Team-lead's "GO" message arrived after the first headline shipped, with two explicit asks: (1) cross-check `unsloth/gemma-4-E2B-it-UD-MLX-4bit` (UD variant ONLY, the non-UD is in the broken-PLE set) head-to-head against the FakeRockert Q4, (2) document the PLE-quant landmine on [[gemma-4-e2b]] with the broken set + safe set.

**Cross-check headline:** Unsloth UD-MLX-4bit is the new production pick.  Identical plays to FakeRockert Q4 in head-to-head paired runs at temp=0 (5/5 same play, 5/5 same delta), 34% smaller peak memory (5.08–6.24 GB vs FakeRockert's 9.17 GB on the same workload), 41% smaller disk (4.2 vs 7.1 GB).  Three Unsloth UD stacked-with-continuous-batching runs landed at peak **5.08 / 6.07 / 6.24 GB** — a **45–56% memory cut** vs bf16's 11.6 GB.  Wall is in the bench's noise floor (paired baselines themselves swing 36 → 80 s on identical config) — wall is suggestive, memory is load-bearing.

**Cohort-size unlock:** at peak 5.08 GB, the M5 Max's 16 GB practical ceiling (3× nominal-peak headroom against chunked-prefill + heterogeneous-cache merge bursts) admits cohort=10 vs current cohort=5 at bf16.  This is the harvest-throughput multiplier the wiki's [[perf-on-the-table]] calibration calculus has been waiting for.  The 5-row subset is too small to exercise it directly; Phase 4 full-560 will validate.

**Defensive `mlx_vlm` audit:** searched `mlx_vlm/*.py` for `speculat*` and `draft_model` references — **no matches**.  Spec decode is dead-on-arrival on Apple Silicon for batched workloads regardless of whether you reach for `mlx_lm.batch_generate` or `mlx_vlm`.  Confirms the Phase-3 dragon-1 finding through a second source.

**Touched pages:** [[burl-perf-phase3]] [[perf-on-the-table]] [[gemma-4-e2b]] [[index]]

**Updated:**
- [[burl-perf-phase3]] tradeoff matrix gained 3 Unsloth UD rows (q4-unsloth-ud-cont, phase3-stack-best v2, v3); ledger snapshot expanded to 11 rows; head-to-head Q4-source comparison added (5/5 identical plays); cohort-size headroom analysis added; defensive audits section added (`mlx_vlm` spec-decode + tokenizer probe per-prompt table); top-of-page summary rewritten to call Unsloth UD as production pick.
- [[gemma-4-e2b]] gained an "MLX quant landscape — PLE landmine + the safe set" section with both the broken set table and the PLE-safe set table (disk + peak GB + paired play match + recommendation).  Plus a Burl bench-rows snippet citing the relevant phase-3 ledger entries.  Bumped `last_updated` to ec46190.
- [[perf-on-the-table]] lever 6 rewritten with Unsloth UD as production pick + the −56% memory finding + cohort=10 unlock.
- [[index]]: hook on phase-3 reflects the new headline + memory cut.

**Frontier shift:** Production Burl inference path is now **`unsloth/gemma-4-E2B-it-UD-MLX-4bit` + Phase-2 continuous batching**.  bf16 stays as the belt-and-suspenders default; Q4 ships when memory or cohort-size is the binding constraint.  The wiki's [[perf-on-the-table]] compounded-stack calculus picks up Phase-3's memory-ceiling unlock as a separate axis from raw wall reduction.

**Why now:** Same-session wiki update per the "update is a side effect" rule.  The Unsloth UD cross-check was an explicit ask in the GO message and produces a strictly better production pick than the first-pass FakeRockert headline.  Documenting the PLE landmine on [[gemma-4-e2b]] is doctrinally important — every future scribe touching Gemma 4 quantization will hit the broken-set repos by default if the wiki doesn't name the safe set.

---

## [2026-05-03 | local | w42 book validation v1 wave 2 probes]

**Touched pages:** [[w42]] [[w42-book-claim-synthesis-and-ai-directions]] [[w42-book-validation-campaign]] [[w42-bookval-v1-wave2-reentry-v2]] [[w42-bookval-v1-wave2-low-trump-trap]] [[w42-bookval-v1-wave2-pounce-window-bid30]] [[w42-bookval-v1-wave2-void-creation]] [[w42-bookval-v1-wave2-void-creation-follow]] [[w42-bookval-v1-wave2-ch02-multistep]] [[w42-bookval-v1-wave2-ch10-action-level]] [[w42-bookval-v1-wave2-pounce-high-bid]] [[winning42-ch02-bidding]] [[winning42-ch03-bidder-play]] [[winning42-ch04-partner-support]] [[winning42-ch05-setter-defense]] [[winning42-ch10-tournament-scoring]] [[winning42-ch12-advanced-bidding-playing]] [[index]] [[log]]
**Added:** 7 wave-2 probe pages plus pounce-high-bid. Wave 2.A.2 oracle-greedy snapshot mining produced 1822 snapshots across 5 corpora; Wave 2.B.2 produced 259618 action rows from a 50-seed bid-aware MPS sweep.
**Updated:** [[w42-book-claim-synthesis-and-ai-directions]] absorbs all wave-2 probe verdicts, the position-dependent reversal, the p_make/EV split, the threshold-q insight, and the demotion methodology lesson. Six chapter pages (ch02, ch03, ch04, ch05, ch10, ch12) record the wave-2 findings on their respective claims. `w42/statistics_claims_ledger/claims.csv` and `w42/phase4_claim_completion_board/completion_board.csv` move ch02-bid-only-enough through context-limited to **supported** and ch12-setter-pounce-high-bid-off from underpowered through context-limited to **contradicted**. `w42/book_validation_v1/AGENTS.md` adds a promotion-guard rule: aggregate proxies do not qualify for ledger promotion.
**Retired:** none.
**Questions opened:** does the p_make/EV objective split apply systematically across all context-limited rows (Wave 3.0 in flight); how to mine 84-eligible snapshots when the legacy corpus is bid=30 and bid-aware atlas does not enforce 4+ doubles (Wave 2.F deferred); should the ledger schema carry per-utility statuses going forward (Wave 3.0 will recommend).

**Frontier shift:**
- ch02-bid-only-enough is the campaign's first non-trivial promotion to `supported`. All 5 adjacent step pairs on n=8000-10000 paired same-hand decisions show monotone overbid penalty, Cohen d 0.16 to 0.47, all 85 of 85 slice cells in book direction, transitive cumulative additive within 0.81%. The audit overclaim risk for this row is resolved at the same-hand bid-margin scope.
- ch05 void-creation splits along position: lead-to-self-void is **contradicted** (EV -2.63 CI [-3.42, -1.84] on n=276), follow-position-discard is **context-limited** in book direction (EV +0.77 CI [+0.12, +1.42] on n=500). The book canonical scenario survives in its actual described shape; the adjacent leading scenario does not. This is the strongest case study for scope precision in book validation.
- ch12-setter-pounce-high-bid-off was promoted to context-limited based on Wave 2.B.2 aggregate Q-delta, then DEMOTED to **contradicted** when Wave 2.E.2 ran the snapshot-level paired contrast on n=1140 high-bid pounce-eligible positions. EV delta -10.42 CI [-11.25, -9.59] across all 4 high bids. The aggregate proxy and the paired action-contrast are fundamentally different objects. New AGENTS.md rule: aggregate proxies do not qualify for promotion.
- ch10-special-bid-mark-multiplier evidence base widened from deterministic transform to action-level demonstration: 71 percent of decisions at bid=42 have a different mark_ev top-1 action than at bid=30. The strategic effect operates through threshold_q recomputation; bid=84 plateaus at bid=42 because both share threshold_q=42. The mm scalar is a positive affine transform that cannot change argmax.
- The campaign's emergent thread is the p_make / EV objective lens split: the book may encode p_make-optimized advice at the contract threshold, with EV-optimal play differing in some sharp-threshold positions. Wave 3.0 (in flight) will formalize this as a per-utility re-classification of all closed probes.
- Status counts after wave 2.E.2: supported 24, context-limited 14, underpowered 19, not-yet-tested 4, contradicted 3.

---

## [2026-05-03 | local | w42 book validation v1 wave 3.0 utility-lens reconciliation]

**Touched pages:** [[w42-book-claim-synthesis-and-ai-directions]] [[w42-book-validation-campaign]] [[w42-bookval-v2-utility-lens-synthesis]] [[log]]
**Added:** Wave 3.0 (`t42-f2ur`) re-processed all 7 closed Wave 2 probes through 5 utility lenses (EV, p_make, mark_ev, CVaR_10, robust_q25). Outputs: `w42/book_validation_v1/wave3/t42-f2ur_utility_lens_synthesis/` with per-probe verdicts, per-claim pivot, proposed schema JSON, reproducibility script, long-form synthesis. Wiki page `w42-bookval-v2-utility-lens-synthesis` records the per-utility verdict matrix.
**Updated:** [[w42-book-claim-synthesis-and-ai-directions]] absorbs the reconciliation — narrows the broad p_make/EV thread to its actual scope (1 of 7 claims, ch05-void-creation-follow), supersedes Wave 2.E.2's "p_make split" framing for high-bid pounce (it's contradicted under all 4 utilities), and records ADOPT-DEFERRED on the per-utility ledger schema. The earlier Wave 2.E synthesis text is softened with a forward-pointer to the Wave 3.0 reconciliation. SESSION_HANDOFF.md updated to reflect closed status, narrowed thread, and revised Wave 4 design direction. AGENTS.md amended with a utility-coverage requirement: every future paired-contrast probe must record all 5 utilities at probe time so the schema can be populated when adopted.
**Retired:** the campaign-wide formulation that "the book may be implicitly p_make-optimized at the contract threshold" — replaced with the narrower "ch05-void-creation-follow is the sole claim where EV-head and p_make-head training signals diverge."
**Questions opened:** does Burl's actual ch05-void-creation-follow behavior match a pure-EV oracle (tests whether single-head is sufficient at this one claim); when do we have 3-5 new probes with full utility coverage to revisit schema adoption.

**Frontier shift:**
- The campaign's **strongest emergent theoretical thread is reduced from "many" to "one."** Wave 3.0 is a textbook case of a meta-analysis correcting a hypothesis projected from a small number of probes. Wave 1.4's 59% p_make/EV agreement is an action-ranking-level disagreement that does NOT propagate to strategic-claim-level verdicts in the closed probe set.
- Multi-objective head architecture (EV head + p_make head + CVaR head) is **worth carrying forward as future work but not yet justified by validated evidence**. The single concrete testable prediction is ch05-void-creation-follow.
- ch12-setter-pounce-high-bid is **unanimously contradicted** under all 4 available utilities (EV -10.42, p_make -0.047, mark_ev -0.047, CVaR_10 +4.40). The earliest "p_make split" framing was an artifact of EV-only reporting in Wave 2.E.2.
- Schema decision **ADOPT-DEFERRED** — defer per-utility status columns until future probes record all 5 utilities by default. AGENTS.md utility-coverage amendment makes this happen automatically.
- Status counts unchanged (Wave 3.0 is read-only meta-analysis): supported 24, context-limited 14, underpowered 19, not-yet-tested 4, contradicted 3.

---

## [2026-05-03 | local | w42 book validation v1 wave 4.0 utility-argmax divergence]

**Touched pages:** [[w42-bookval-v3-utility-argmax-divergence]] [[w42-bookval-v2-utility-lens-synthesis]] [[index]] [[log]]
**Added:** Wave 4.0 (`t42-hmjr`) measures utility-argmax divergence at the policy-action level on the Wave 2 ch05-void-creation-follow corpus (n=500, bid=30, all-setter-to-act). For every snapshot, argmax is computed over ALL legal actions under EV / p_make / mark_ev / CVaR_10 / robust_q25. Outputs: `w42/book_validation_v1/wave4/t42-hmjr_utility_argmax_divergence/` with `per_snapshot_argmax.csv`, `disagreement_matrix.csv` (5×5 with bootstrap CIs), `void_subset_confusion.csv`, `summary.json`, `manifest.json`, `analyze.py`. Wiki page `w42-bookval-v3-utility-argmax-divergence` records the verdict.
**Verdict:** **DISAGREE ≥ 5%** — EV vs p_make argmax disagreement is **41.2% (CI [36.8%, 45.6%])**, an order of magnitude above the 5% gate. Recommend scoping rung-2 (utility-tunable searcher) as the next build. p_make = mark_ev exactly (0/500 disagreement) — empirical confirmation of the bid=30 mm=1 affine identity at the argmax level.
**Frontier shift:**
- Wave 3.0's EV vs p_make split is **NOT** purely a contrast-magnitude artefact; the two utilities pick different actions on ~41% of the corpus. The architecture-decision gate trips clearly.
- A subtle inversion of the Wave 3.0 framing emerged: p_make picks the void slot (38.6%) MORE often than EV does (29.4%). EV more often picks a third action that is neither void nor preserve (37.6%). The canonical "EV→void, p_make→preserve" pattern accounts for only 5.8% (29/500) of snapshots — meaningful, but only ~14% of the total disagreement. Rung-2 design framing should be "EV/p_make systematically pick different slots" rather than "EV likes void, p_make likes preserve."
- This wave is read-only on the ledger (architecture-decision measurement, no claim row moves).
- Status counts unchanged.

---

## [2026-05-03 | local | w42 lens v1 head-to-head — ev wins decisively]

**Touched pages:** [[w42-lens-v1-utility-head-to-head]] [[w42-book-claim-synthesis-and-ai-directions]] [[w42-book-validation-campaign]] [[index]] [[log]]
**Added:** Lens v1 (`t42-4ouu`) — 1-step Q-greedy player wrapped by utility, parallel-hand simulator, round-robin {ev, p_make, cvar_10, robust_q25} × 6 pairings × 1000 hands paired-seed in 7 minutes wall on M5 Max (parallel-hand K=500, N=10 per Q-query, fp32). Outputs: `w42/lens_v1/` with `lens.py`, `parallel_match.py`, `round_robin.py`, `fp_sanity.py`, `analyze.py`, `results/round_robin_n10.csv`, `results/per_hand_margins.csv`, `results/sample_sweep.csv`, `results/fp_sanity.csv`, `results/mark_ev_pmake_sanity.csv`, `manifest.json`, `summary.json`. Wiki page `w42-lens-v1-utility-head-to-head`.
**Verdict:** **EV WINS.** Total ordering ev > robust_q25 ≳ cvar_10 > p_make, all 6 CIs exclude zero, ev beats p_make by +5.42 pts/hand ([+4.03, +6.81]), 59.5% A win rate, 99.5% decisive. Sample-sweep at N ∈ {10, 50, 100} confirms direction is robust to N. fp16 sanity passed (≥99% argmax match all 5 utilities) but round-robin ran fp32 (MPS doesn't autocast).
**Frontier shift:**
- Wave 4.0's "EV is the outlier preferring third-option discards" framing is **inverted** by Lens v1: those third-option picks are point-winning, not noise. Wave 4.0 measured *who picks differently*; Lens v1 measured *who wins*. The answers point in opposite directions on the void-creation slice — the book aligns with the worst-scoring utility (p_make).
- This kills the "build a multi-objective head because the utilities differ" thread as a research agenda. A single fixed-utility EV head is the strongest of the four tested at 1-step lookahead on this corpus. Multi-objective architecture only re-emerges if utility selection is *state-conditioned* (filed: t42-nwuu Lens v2).
- **Production-code follow-up:** `forge.eq.generate.actions.select_actions` is essentially Lens(p_make) — it picks p_make-argmax with EV as tie-break. The current production E[Q] action selector is the WORST of the four utilities tested. Switching to ev-argmax is a one-line change predicted to lift the existing E[Q]-N=100 vs Zeb-Large win rate (currently 55.7% per `forge/zeb/OVERVIEW.md`). Filed as a separate bead.
- The Zeb-era N=10 ≈ N=100 finding (from `forge/zeb/OVERVIEW.md` lines 622-632) empirically holds for Lens-vs-Lens matchups too. N=10 is the right operating point for utility-comparison work.
- Promotion guard respected: no central ledger row moved.

---

## [2026-05-03 | local | w42 lens v1 disaster utility + methodology insight]

**Touched pages:** [[w42-lens-v1-utility-head-to-head]] [[w42-book-claim-synthesis-and-ai-directions]] [[w42-book-validation-campaign]] [[log]]
**Added:** `disaster` utility in `w42/lens_v1/lens.py` — clipped EV that floors all sub-threshold Q samples to −42 then takes expectation under PDF (matches EV above threshold, max-loss-equivalent below). Head-to-head 1000 paired-seed hands, N=10, fp32: disaster vs ev = −1.55 (CI [−3.07, +0.05] grazes zero); disaster vs p_make = +2.74 (CI excludes zero); disaster vs robust_q25 = +0.32 (tied). Implied ordering: ev ≳ disaster ≳ robust_q25 ≳ cvar_10 > p_make. Artifacts: `w42/lens_v1/results/disaster_head_to_head.csv`, `w42/lens_v1/run_disaster.py`. Sanity test on synthetic PDF passes.
**Updated:** [[w42-book-claim-synthesis-and-ai-directions]] gains a "## Methodology insight — the single-decision blind spot" section. The deepest finding of this campaign is structural, not from any single probe: **most book claims are multi-step plans, but most probes are single-decision contrasts.** Single-decision EV can be the locally-best move yet still lose to a planner that sets up future tricks (the book's bread and butter). This reframes Lens v1's "EV wins" result — EV won the *individual moves* contest; the book was never in that contest because the book plays plans. Wave 4.0's 41% argmax disagreement may be "EV picks locally-best; book picks setup move" rather than "EV picks weird things." [[w42-book-validation-campaign]] gets the same insight in summary form + a Wave 5 frontier note.
**Retired:** the implicit assumption that single-decision contrasts are the right test granularity for all book claims. From Wave 5 onward, strategy-shaped claims (multi-trick plans) need planning-aware probes; single-decision probes only for the genuinely single-decision ones.
**Questions opened:** which of the 19 underpowered + 14 context-limited claims are actually multi-step plans we tested at the wrong granularity? What's the right Wave 5 first probe (book-strategy player on 84-throwaway, or general-purpose Lookahead-Lens at K=2)?

**Frontier shift:**
- **No fixed pointwise utility distinguishably beats EV at one-step lookahead.** Disaster confirmed this. EV is by construction the maximum-information summary of the per-action outcome distribution; any other pointwise utility either throws information away (hard cliff, p_make) or imposes a fixed reward shape that EV can already represent (soft cliff = EV minus a bias term, same argmax). To beat EV you need information EV doesn't have — which means planning, not a different summary statistic.
- **The campaign has been operating at the wrong abstraction level for ~80% of the book's claims.** That's not a bug; single-decision probes were the right starting point because they're cheap and they catch local errors. But the bulk of the book's wisdom lives at the plan level. Wave 5+ should be planning-aware for the multi-step claims.
- **Three planning architectures named:** MCTS over forge (heavy, generic), Lookahead-Lens (cheap, generic), book-strategy player (cheap, claim-specific). For the book validation use case, book-strategy player is most direct; for broader model-design questions, Lookahead-Lens is the cheap general-purpose tool.
- **Wave 2.F (84-throwaway, t42-wikw) reactivated** with a sharper motivation: it's the campaign's first concrete multi-step strategy to encode and test. The methodology insight retroactively justifies the design that had been deferred for lack of motivation.

---

## [2026-05-03 | local | book-strategy-player architecture designed (wiki-first)]

**Touched pages:** [[book-strategy-player]] [[zeb]] [[gus]] [[burl]] [[w42-book-claim-synthesis-and-ai-directions]] [[index]] [[log]]
**Added:** new entity page [[book-strategy-player]] capturing the multi-step strategy framework designed in conversation 2026-05-03. Page covers: Strategy protocol (4 lifecycle methods + optional opponent-observation hook), namespaced PlanState with publishable facts, the 5-phase player loop (retire → recognize → arbitrate → apply delta → bail or play), the 5 composition modes (state-conditioned selection, strategy chaining, hierarchical sub-libraries, opponent-aware counter-strategies, cross-hand match memory), the DecisionRecord format, the recording → training pipeline (Models A/B/C), and the phased build plan. Crucially: the book's chapter structure IS exactly mode 1 + mode 3 — chapters partition state space, sub-strategies live within. A faithful book-encoded player is `BookStrategyPlayer(strategies=[Chapter3_BidderPlay(), Chapter5_SetterDefense(), ...], fallback=Lens(ev))`.
**Updated:** Forward-pointer notes added to [[zeb]] (training-data generator + Model C target candidate), [[gus]] (input encoder for Model A — what Gus was originally designed for), and [[burl]] (Model A strategy selector candidate — small structured action space + narrative reasoning is exactly Burl's shape; the use case Burl was built for and we just hadn't found it). [[w42-book-claim-synthesis-and-ai-directions]] gains a "## The architectural payoff" subsection inside the methodology insight section. [[index]] gets the new entity entry.
**Retired:** none.
**Questions opened:** which 5-10 strategies to encode first; whether the framework's training data is sufficient at ~50K labeled decisions or needs an order of magnitude more; which Model A backbone wins (Burl-as-LLM vs small NN); whether the strategy-discovery loop (cluster fallback-invoked decisions) actually surfaces new candidate strategies in practice.

**Frontier shift:**
- The book-strategy framework is **not just a measurement instrument** — it is **also a structured-action-space training data factory**. Recording is a one-line addition to the player loop and unlocks Model A (strategy selector), Model B (plan-success predictor), and Model C (end-to-end policy distillation) with no extra simulation cost.
- **Three parked-or-experimental models are simultaneously rescued by this architecture:** Burl (LLM strategy selector — finally a use case where small action space + narrative reasoning is the natural fit), Gus (input encoder — what it was designed for), Zeb (training-data generator + Model C target). Coherent re-integration that makes all three load-bearing again rather than parked.
- This is the cleanest path past the EV ceiling that the campaign has identified. Other paths are dead (multi-utility heads, soft-cliff utilities) or impractical (generic MCTS, Lookahead-Lens at K=2-3 with the 4-player branching multiplier).
- Wiki-first per project doctrine — bead filing is next, scoped as Phase 1 (framework + 1-3 starter strategies + recording + Lens(ev) head-to-head harness).

---

## [2026-05-03 | local | wiki curation pilot — hubs, trails, and W42 routing]

Pilot curation pass for the steady-state wiki. The goal is not to delete leaf
evidence, but to make the wiki usable without loading the full catalog into
context. The pass adds a frontier-hub / trail / leaf-page navigation role and
uses W42 as the first pilot cluster.

**Touched pages:** [[AGENTS]] [[index]] [[w42]] [[wiki-entrypoints]] [[w42-book-validation]] [[book-strategy-player]] [[w42-lens-v1-utility-head-to-head]] [[w42-bookval-v1-wave1-mark-utility-transform]] [[w42-bookval-v1-wave2-low-trump-trap]] [[w42-bookval-v1-wave2-pounce-high-bid]] [[w42-claim-analysis-synthesis-report]] [[log]]

**Added:**
- [[wiki-entrypoints]] — lightweight route map for agents: frontier hubs, large
  leaf clusters, and `rg` query shortcuts.
- [[w42-book-validation]] — curated path through the W42 / Winning 42 book
  harvest, chapter pages, claim-validation ladder, phase sweep, evidence
  surfaces, and current validation campaign.

**Updated:**
- [[AGENTS]] now names frontier hubs, trails, and leaf pages as explicit
  navigation roles and makes [[index]] a catalog fallback rather than mandatory
  first context for every query.
- [[index]] gained a compact "Start here" section and trail entries for the two
  new route pages.
- [[w42]] was slimmed into a frontier hub; long phase-by-phase evidence moved
  behind [[w42-book-validation]] and existing leaf pages, including the later
  Wave 3/4, Lens v1, and [[book-strategy-player]] planning-aware frontier. The
  refreshed route includes the book-strategy-player algebraic spec as the build
  contract for Phase 1.
- Four W42 link-hygiene fixes landed while validating the trail: the
  mark-utility transform now points to [[w42-book-validation]], the phase-2
  synthesis now points to [[w42-phase2-seat-position-strategy-map]], and two
  Wave 2 pages point to [[w42-book-validation-campaign]] instead of placeholder
  overview/handoff pages.

**Frontier shift:** The wiki remains Karpathy-style and backlink-first, but
large clusters should now be approached through hubs and trails. W42 is the
pilot: the latest Wave 2-4, Lens v1, and [[book-strategy-player]] evidence stays
intact, while the entity page becomes orientation material rather than a
complete ledger.

**Questions opened:** none.

---

## [2026-05-04 | local | book-strategy-player algebra tightened]

Design refinement for [[book-strategy-player]] after reviewing `t42-zrf9` against
the wiki algebra.

**Touched pages:** [[book-strategy-player]] [[index]] [[log]]

**Updated:**
- [[book-strategy-player]] now makes arbitration order-independent even on equal
  priorities by requiring a canonical `strategy.name` tie-break.
- L4 now states the intended invariant directly: active plans are not
  re-recognized. One-shot-per-hand behavior is an explicit tombstone fact, not
  the framework default.
- L6 and the core operation now agree: bail falls back while preserving the
  active plan unless the strategy explicitly emits a retire delta or `plan_done`
  retires it.
- Hierarchical composition is finite and acyclic. A wrapped sub-player remains
  a valid Strategy, but construction should reject cycles rather than implying a
  recursive execution stack.
- Decision records now carry `strategy_path` and coverage buckets. Fallback-only
  decisions are explicitly the future strategy-discovery surface: uncovered,
  high-regret or high-tail-risk regions can later suggest candidate strategies.
- Context-aware strategies are recorded as a future composition pattern through
  published facts or hierarchical sub-libraries, not through cross-strategy
  reads of private plan state.

**Frontier shift:** The Phase 1 build contract is crisper. The immediate build
is still the algebra-preserving framework plus measurement instrument; uncovered
region mining, context wrappers, and richer nested libraries are recorded as
future exploration surfaces rather than detailed near-term plan.

**Questions opened:** none.

---

## [2026-05-04 | local | book-strategy-player commitment semantics refined]

Second design refinement for [[book-strategy-player]] before implementing `t42-zrf9`.
External review exposed where the previous algebra still conflated recognition,
commitment, active execution, local fallback proposals, and causal attribution.

**Touched pages:** [[book-strategy-player]] [[index]] [[log]]

**Updated:**
- [[book-strategy-player]] now splits fresh recognition from active-plan execution.
  Fresh strategies may propose `commit_recognition` deltas, but only the arbitration
  winner's recognition delta is committed to `active_plans`.
- The Strategy protocol now uses `recognizes` instead of `applies_to`; active plans are
  arbitration candidates because they are already committed, not because they newly
  recognize the current state.
- Shared `facts` are explicitly lawful `FactValue` wrappers. Scalar-looking phase,
  target, and context data belong in private plan state unless wrapped in a commutative
  monoid value with canonical conflict behavior.
- `fallback_action` is now a pure local counterfactual proposal, not an exact causal
  effect. Exact contribution requires paired-seed replay or value-model estimation.
- Hierarchical wrappers are weakened from a broad functor-equivalence claim to explicit
  wrapper semantics: expose parent priority, namespace by `strategy_path`, return `None`
  when no inner strategy applies unless configured as a terminal fallback, and reject
  cycles.
- L3, L5, L10, the core operation, and the Phase 1 build order now include duplicate-name
  rejection, non-NaN priorities, illegal-action handling, replay/version fingerprints,
  lossless game-state snapshots, pure fallback dry-runs, and framework-contract tests
  before W42 strategy logic.

**Frontier shift:** `t42-zrf9` remains the right bead, but implementation should start
with the framework contract and fake-strategy invariant tests before encoding the first
book tactic. The wiki contract supersedes older bead wording that used `applies_to`,
committed every fresh recognizer, or described the logged fallback proposal as exact
strategy contribution.

**Questions opened:** none.

---

## [2026-05-04 | local | book-strategy-player amended design organized]

The amended design for [[book-strategy-player]] was folded into the live wiki
contract instead of being appended as another review note.

**Touched pages:** [[book-strategy-player]] [[w42]] [[w42-book-validation]] [[index]] [[log]]

**Updated:**
- [[book-strategy-player]] now reads as the current Phase 1 build-ready contract:
  fresh recognition is separate from active-plan execution, only the arbitration
  winner commits a fresh plan, shared facts must be monoid-valued wrappers, and
  fallback recording must be pure.
- The page now distinguishes bail, retire, blocked, and disrupted states; records
  `fallback_action` as a local counterfactual proposal rather than exact causal
  attribution; and replaces optimistic coverage names with conservative
  pre-replay buckets such as `covered_diff_unattributed`.
- The `DecisionRecord` section now carries replay/version/fingerprint fields,
  lossless `game_state_snapshot` separate from model input tensor, RNG/forge
  seed fields, `strategy_path`, strategy-returned action, status, delta summary,
  and plan completion status.
- The implementation surface now includes Phase 1 property tests T1-T17 and a
  concrete module layout under `w42/book_strategy/`.
- [[w42]], [[w42-book-validation]], and [[index]] now route readers to the amended
  build-ready contract rather than the older "design pending" framing.

**Frontier shift:** The design is no longer just a conceptual route past the
single-decision blind spot. It is organized as the implementation contract for
`t42-zrf9`: framework laws and fake-strategy tests first, then fallback purity
and recording, then the first book tactic and head-to-head measurement.

**Questions opened:** none.

---

## [2026-05-04 | local | book-strategy-player algebra refactored]

The long amended [[book-strategy-player]] design was split into a compact core
algebra plus satellite pages for recording, implementation, and future extension
points.

**Touched pages:** [[book-strategy-player]] [[book-strategy-player-recording]] [[book-strategy-player-phase-1-build]] [[book-strategy-player-extension-points]] [[w42]] [[w42-book-validation]] [[index]] [[log]]

**Added:**
- [[book-strategy-player-recording]] now owns the Writer-side contract:
  pure fallback proposals, replay-heavy `DecisionRecord`, conservative coverage
  buckets, and the warning that local fallback divergence is not causal
  attribution.
- [[book-strategy-player-phase-1-build]] now owns the implementer checklist:
  fake-strategy law tests, pure `Lens("ev")` fallback adapter, recording,
  starter strategies, error policy, module layout, and head-to-head measurement.
- [[book-strategy-player-extension-points]] now parks hierarchy, observation,
  match memory, learned selectors, plan-success prediction, and Gus/Burl/Zeb
  roles outside the Phase 1 algebra.

**Updated:**
- [[book-strategy-player]] now centers the single implementer equation:
  `Env + GameState + PlanState -> Action + PlanState' + DecisionRecord`.
  It frames BookStrategyPlayer as a pure Reader/State/Writer player over a
  finite strategy library, per-hand plan state, lawful shared facts,
  deterministic arbitration, and replay-pure recording.
- The previous "nine algebras / ten laws" presentation is reduced to the
  load-bearing contracts implementers must preserve: empty-library fallback
  identity, unique strategy names, order independence, recognition/commit
  separation, active-plan continuation, finite-priority arbitration, namespace
  hermeticity, fact merge laws, bail/retire orthogonality, and recording purity.
- [[w42]], [[w42-book-validation]], and [[index]] now route readers to the split
  core/spec/checklist structure rather than treating one long page as the whole
  design.

**Frontier shift:** Phase 1 is now easier to implement because the main page is
the algebra and the satellite pages are supporting contracts. Future model and
hierarchy ideas remain preserved, but they no longer compete with the core
framework specification.

**Questions opened:** none.

## [2026-05-06 | local | Burl microscope prompt/tool recipe workbench]

[[burl-microscope]] lands as a lightweight human-in-the-loop workbench for Burl
prompt, tool, and tool-response experiments. It is intentionally smaller than
[[burl-lab]]: one harvested case, one editable recipe, one Gemma conversation,
and one JSONL trace.

**Touched pages:** [[burl]] [[burl-microscope]] [[index]] [[log]]

**Added:**
- [[burl-microscope]] — documents the recipe shape, Pi extension client, and
  first smoke result.

**Updated:**
- [[burl]] now routes lightweight prompt/tool experimentation to
  [[burl-microscope]] and keeps [[burl-lab]] for heavier event-sourced phase
  machinery.
- [[index]] adds [[burl-microscope]] to the Burl entity cluster.

**Frontier shift:** Prompt/tool iteration now has a tight interactive loop:
edit recipe files, reopen the same failed decision from [[burl-2000-harvest]],
step Gemma with native tool calls, and compare final play against the original
Burl/oracle references. First smoke on `global_idx=1` reproduced the original
Burl mistake under `baseline` (`25`) and flipped to oracle/consensus play `19`
under `legal-brief`.

**Questions opened:** none.

## [2026-05-07 | local | Burl microscope board-snapshot prompt correction]

The first post-commit microscope review read the user's live session logs and
corrected the initial `legal-brief` smoke claim.

**Touched pages:** [[burl]] [[burl-microscope]] [[index]] [[questions/open]] [[log]]

**Updated:**
- [[burl-microscope]] now treats the original `legal-brief → 19` smoke as
  reference-leakage-tainted because the prompt exposed `oracle/reference play: 19`.
- [[burl]] now routes the frontier result to the workflow and the strength of
  `board_snapshot()` as a first-read surface, not to a solved recipe.
- [[index]] updates the one-line hook accordingly.

**Frontier shift:** `board_snapshot()` is promoted as the clean default user-prompt
substrate. Baseline and `legal-brief` prompts were stripped down to
`board_snapshot()` output plus decide text / legal-candidate instructions. On the
same `global_idx=1` case, fair no-reference `snapshot-first` and `legal-brief`
runs both committed `25`; the case remains unsolved by prompt shape alone.

**Questions opened:**
- What additional prompt/tool-response framing lets Gemma choose `19` on this case
  without oracle/original-play leakage or human steering? Logged in [[questions/open]].

## [2026-05-07 | local | Burl microscope hand-hypothesis tool]

A live [[burl-microscope]] conversation asked Burl what was missing after
`board_snapshot`, `legal_plays`, `play_brief`, and `belief_trajectory`. Burl named a
"Hypothetical Hand Simulation" / "Opponent Hand Query" and proposed
`simulate_hand_impact`.

**Touched pages:** [[burl]] [[burl-lab]] [[burl-microscope]] [[index]] [[log]]

**Added:**
- `simulate_hand_impact(play_id=X, seat=..., holds=Y)` as a first-class ToolSpec.
  It returns one hidden-hand hypothesis's Gus plausibility plus baseline vs.
  conditional E[Q] shift.
- `snapshot-hypothesis` recipe: clean `board_snapshot()`-first prompt with
  `board_snapshot`, `legal_plays`, `play_brief`, `belief_trajectory`,
  `simulate_hand_impact`, and `commit_play` active.

**Updated:**
- [[burl-microscope]] documents the Burl-requested hypothesis tool and recipe.
- [[burl-lab]] records the registry expansion from seven to eight tools.
- [[burl]] notes that the requested tool is a query-shaped belief/simulation bridge,
  not another broad posterior dump.
- [[index]] updates the Burl microscope hook.

**Frontier shift:** The microscope can now test targeted questions of the form
"if seat S holds domino D, what happens to play P?" directly after a `play_brief`
catalyst line.

**Questions opened:** none.

## [2026-05-07 | local | Burl microscope expected-utility tool]

A follow-up [[burl-microscope]] conversation using `snapshot-hypothesis` asked Burl
what tool should exist after `simulate_hand_impact`. Burl named
`calculate_expected_utility`: a high-level EV synthesis that ranks candidate plays
instead of making the model manually integrate single-hypothesis probes.

**Touched pages:** [[burl]] [[burl-lab]] [[burl-microscope]] [[index]] [[log]]

**Added:**
- `calculate_expected_utility(plays=[...])` as a first-class ToolSpec. Omitting
  `plays` ranks the current legal plays. Output includes mean Q, p_make, sampled
  mean CI, distribution shape, and dominant information source.
- `snapshot-utility` recipe: clean `board_snapshot()`-first prompt with
  `legal_plays`, `calculate_expected_utility`, and the lower-level follow-up tools
  (`play_brief`, `simulate_hand_impact`, `belief_trajectory`) available.

**Updated:**
- [[burl-microscope]] documents the expected-utility tool and recipe.
- [[burl-lab]] records the registry expansion to nine tools.
- [[burl]] notes the sequence of Burl-requested tools from targeted hypothesis to
  integrated EV ranking.
- [[index]] updates the hooks for Burl Lab and Burl microscope.

**Frontier shift:** The live microscope can now test whether Gemma behaves better
when the EV integration step is made explicit as a tool, rather than reconstructed
from separate `play_brief` and `simulate_hand_impact` calls.

**Questions opened:** none.

## [2026-06-09 | local-session | Champion — unified belief-state player direction]

The session reframed the project's player work around one architecture: a
belief-state player that bids and plays full games to 7 marks. Pre-wiki
bidding work (`forge/bidding/` 2026-01, `gus/bidding/` 2026-04) was located
and promoted into the wiki for the first time. The auction was named the
dominant strength gap (auction ≫ belief-weighted worlds > score utility ≫
card-play polish), and the teaching half (champion vs. the W42 detector
battery, Burl narration) was identified as the same object read from the
other side.

**Touched pages:** [[champion]] [[gus]] [[index]] [[log]]

**Added:**
- [[champion]] — decision loop, asset map, pre-wiki bidding inventory,
  self-consistency fixed point, 8-rung build ladder, teaching half.

**Updated:**
- [[gus]] — role in the champion: belief head as the posterior engine,
  auction-conditioned and wired into oracle world sampling.
- [[index]] — champion added to Start here and Shared infrastructure.

**Frontier shift:** beads is retired (2026-06); the champion action ladder
lives in GitHub issues (milestone "Champion") on jasonyandell/mk5-main.
Historical beads remain readable in `.beads/issues.jsonl`. The wiki carries
information; GitHub issues carry action.

**Questions opened:** none.

## [2026-06-12 | a715af4 | Arena — full-game harness landed (champion rung 1)]

**What happened:** The [[champion]] ladder's first rung shipped: `arena/`
plays four seats through real auctions (docs/rules.md §4) and hands to 7
marks, with paired-seed team rotation. GitHub issue #20 closed. First
physics followed within the hour: under identical oracle play (lens:ev,
N=10), a static Roberson risk-budget bidder beats the always-bid-30
baseline 58.9% (113/192 games), mark margin +0.78/game, 95% CI
[+0.28, +1.26] — the auction-first marginal-value ranking got its first
confirmation at the cheapest possible rung. 192 full games run in ~150 s
on MPS.

**Touched pages:** [[arena]] [[champion]] [[index]] [[log]]

**Added:**
- [[arena]] — design (engine split, lockstep batching, oracle never sees
  the bid), shipped bidders/players, first physics, v0 limits.

**Updated:**
- [[champion]] — asset map: full-game arena → done; auction policy → v0
  (static risk-budget in `arena/bidders.py`); ladder rung 1 marked done.
- [[index]] — arena added to Shared infrastructure.

**Frontier shift:** every player in the stack is now comparable at the
level the game is actually played; "best player" is a measurable sentence.
Next rung: auction v0 (#21) — replace the static ceiling with gus/bidding
simulated mark swings; the static bidder is the baseline to beat.

**Questions opened:** none.

## [2026-06-12 | local-session | Champion rungs #21, #27 v1, #23 — bidder, marks-to-7, bid_value plumbing]

**What happened:** Three champion rungs advanced in one session, the first
working through epic [[champion]] (GitHub issue #29) directly. (1) **Auction
v0 (#21):** `champion/bidder.py` — `GusBidder` takes the cheapest
positive-utility legal bid over a `gus/bidding` simulated P(make) table,
declares the trump maximizing P(make) at the contract threshold, prefilters
hopeless hands, evaluates each hand once. Wired into the arena as
`gus[:N[,wp]]`. (2) **Marks-to-7 utility v1 (#27):** `champion/utility.py` —
`race_wp` is the score→WP lookup the issue asks for, built as Pascal's
recursion under a neutral one-mark-per-hand race model; `MarksToSeven`
scores a contract as Δ win-probability. The Pascal identity makes 1-mark
contracts flip sign at p=½ at every score, so conditioning bites exactly on
multi-mark bids (84 needs p>¾ ahead 6-0, p>¼ behind 0-6) — prudence and
desperation emerge rather than being authored. (3) **bid_value plumbing
(#23):** `forge/cli/generate_eq_continuous.py` now threads per-seed
`bid_values` into `generate_eq_games_gpu` and records them per `.pt`, with a
`--bid-value 30|42|84|seed` flag. Proven live: regenerating 8 seeds at
bid=42 vs bid=30 changes 18–24 of 28 decisions per game. Same session fixed
the `estimator.py` 84-threshold bug and the `cefb617` import breakage that
had left the continuous generator unrunnable, and de-skipped a dead
integration test (`forge.eq.generate_gpu` → `forge.eq.generate.pipeline`).

**Touched pages:** [[champion]] [[arena]] [[log]]

**Added:** `champion/` package (`utility.py`, `bidder.py`, tests).

**Updated:**
- [[champion]] — asset map: mark utility → v1 score-conditioned; auction
  policy → v0 two-tier (static + Gus-backed); landmine struck through as
  fixed. Ladder rungs #21, #27, #23 annotated done/v1.
- [[arena]] — Gus bidder section; `BidContext` now carries game score.

**Frontier shift:** the champion has a model-backed auction policy and a
score-aware utility for the first time; generated corpora can finally vary
the bid. Highest-leverage rung remaining is belief-weighted world sampling
(#25). Open: equilibrium-aware pass baseline for `MarksToSeven` (rung #26);
the play-risk hook for marks-to-7; the full 128-game `gus` vs `heuristic`
headline (match running at session end).

**Questions opened:** does `MarksToSeven`'s neutral race model (p=½/hand)
bias bidding vs a model that knows the bidder's own edge? Flagged for the
self-play loop to answer.

---

## [2026-06-12 | 40a32a7 | Champion rungs #22 + #27 v2: bid-strength net; play-risk measured the wrong lever]

A two-track push on the [[champion]] ladder: rung #22 (bid-strength net) landed
by a parallel agent while rung #27 v2 (marks-to-7 v2) was built in the main loop.

**Touched pages:** [[entities/champion]] [[entities/arena]]

**Rung #22 — bid-strength net.** The 2026-01 corpus generator
(`forge/cli/bidding_continuous.py`) was rewired off the retired 817k policy
model onto the Gus simulator (`simulate_all_gus_batch`, all 9 EVAL_DECLS incl
notrump, MPS/auto). A scaled Gus-backed corpus (604 rows) was distilled by
`champion/bid_net.py` into a hand → (9 decl × 13 bid) p_make MLP: **test MAE
0.053, ECE 0.007, 0.012 ms/call** — the sub-millisecond replacement for the
live Gus sim the `GusBidder` pays per hand.

**Rung #27 v2 — marks-to-7 v2.** Two pieces, both correct and unit-tested:
(1) an equilibrium-aware pass baseline on `MarksToSeven` (`pass_q_opp`/
`pass_make_rate`, default off = v1) — a behavioral diff shows it shifts 16.7%
of sampled bids, all toward fighting harder for the auction; (2) a
score-conditioned play-risk hook (`champion/play_risk.py
ScoreConditionedLensPlay` + the new risk-seeking `upside_10` lens + a
`marks`/`marks_to_win` channel on `PlayPolicy.choose`). The play-risk hook was
**measured and it loses**: `scorelens` vs `lens:ev`, identical bidders, 192
games — **72/192 (37.5%), −1.20 marks/game, 95% CI [−1.69, −0.70]**, make-rate
48.9% vs 60.2%.

**Frontier shift:** a direct, CI-excludes-zero confirmation of the marginal-value
ranking (auction ≫ belief ≫ score-utility ≫ **card-play polish**). Within a 42
hand, marks-optimal play ≈ maximize P(make), which is score-independent; the
risk-shaped lenses sacrifice contracts. Play-risk is the wrong lever — the
mechanism is correct, the measurement redirects effort to the auction and
belief. Highest-leverage rung remaining is still **#25 belief-weighted world
sampling**, now with a clean lower-risk plan (importance-weight the
marginalization, `compute_eq_pdf` already takes per-world weights).

**Questions opened:** does marks-optimal *play* (`lens:p_make`, score-blind)
beat the EV default `lens:ev`? Untested — the cheap next play-side experiment.
Does the pass baseline's win-rate edge survive full games (vs the behavioral
diff)? Needs an expensive paired run; deferred.

---

## [2026-06-12 | pending | Champion #25: belief-weighted world sampling — mechanism landed, measured null]

The highest-leverage architectural slot on the [[champion]] ladder, wired and
validated. `champion/play.py BeliefLensPlay` keeps the validity-guaranteed MRV
world sampler and the oracle E[Q] path untouched and changes only the
marginalization: it importance-weights the sampled worlds by the Gus belief
posterior (`champion/belief.py`) instead of averaging them uniformly. A world's
weight is softmax over worlds of Σ log P(seat | tile); `compute_eq_pdf` already
accepted per-world weights and `compute_eq_weighted_mean` was added for the mean.
The belief head's three classes (relative opponents P+1/+2/+3) align exactly with
the MRV sampler's three opponent rows from the same POV — no reindex.

**Touched pages:** [[entities/champion]] [[entities/arena]]

**Result (128 games, identical heuristic bidders):** `belieflens:ev` vs `lens:ev`
is a **null** — 58/128 (45.3%), −0.13 marks/game, 95% CI [−0.76, +0.48]; make-rate
54.9% vs 55.9%. The weights are genuinely active (effective sample size ~5–8 of
10, min ~2), so the mechanism works — but the play-evidence-only belief is too
weak to move play. Exactly the [[belief-bayes-ceiling]] prediction: top-1 belief
sits at the ~39% Bayes ceiling, barely above 33% chance.

**Frontier shift:** the belief→world-sampling slot — the single change the
champion design says will improve bidding, play, and defense together — is now
built, unit-tested (degrades to uniform exactly), and arena-validated. The win is
gated not on the wiring but on belief quality: **rung #24 (auction-conditioned
belief)** is now the clear unlock, and #26 (self-play) feeds it. A bug was caught
and fixed mid-run (the CLI player ran uniform-mode until `BeliefLensPlay` was made
to load the belief model by default — the ESS heartbeat surfaced it).

**Questions opened:** does a sharper belief (lower `tau`) or the
`arena_v3_consistency` adapter move the null, or is 39% top-1 a hard floor until
auction evidence enters the belief input (#24)? The latter is the bet.

---

## [2026-06-12 | pending | Champion #22 follow-up: bid-strength net wired into the policy]

The rung-#22 distilled net plugged back into the auction policy. `GusBidder`
gained an optional `pmake_fn` (behavior-preserving — default `None` is the exact
old simulated path), and `champion.NetPointsEvaluator` loads `champion/bid_net.pt`
to serve a `{decl: {threshold: P(make)}}` table in **0.68 ms/hand** (~1500× faster
than the ~1 s Gus simulation). CLI: `net[:wp[,pass<q>]]`.

**Touched pages:** [[entities/champion]] [[entities/arena]]

**Validation (128 games, seed 0 — directly comparable to the gus 84/128 run):**
`net:wp` vs `heuristic`, lens:ev both — **85/128 (66.4%), +1.29 marks/game, 95%
CI [+0.72, +1.84]**, make-rate 69.3% vs 54.1%. The distillation fully preserved
(slightly exceeded) the live Gus bidder's own +1.09 edge, and the net bidder is
*more* selective (42.8% offense share) and reaches `notrump`/`doubles-trump` the
8-decl sim bidder structurally cannot.

**Frontier shift:** the auction policy is now both strong AND fast. The Gus sim
bidder cost ~49 min for 128 games; the net bidder runs in arena-play time. Fast
full-game auction sweeps (e.g. measuring the #27 v2 pass baseline's win-rate
impact in full games) are now practical.

---

## [2026-06-13 | pending | Champion #28: teaching battery — first champion-vs-the-book receipts]

The pedagogy chain produces its first receipts. [[w42-champion-teaching-battery]]
runs the champion (GusBidder auction + lens:ev play) through the ch04/ch05 tactical
detectors on its OWN trajectories — 256 games, 7,168 decisions, 19,264 action rows,
1,484 labeled — and scores each claim by the paired same-decision contrast
(`w42/claim_analysis/harness.py`), with a half/seed key so contrasts never straddle
the two arena halves.

**Touched pages:** [[entities/champion]] (Teaching half) · new
[[experiments/w42-champion-teaching-battery]]

**Result — champion agrees with the book on 3 of 6 checkable claims:**
- setter pounce-count: **+4.9 pts** [CI +3.3,+6.7], 117 paired — supported-on-slice
- pounce-count before certainty: **+3.9 pts** [+2.2,+5.7], 87 paired — supported
- extra-count-to-set: **+10.0 pts** [+4.6,+15.8], 23 paired — supported
- reckless count to the bidder: **−7.7 pts** [−8.6,−6.8], 550 paired — contradicted (negative control)
- unsafe partner donation: **−9.2 pts** [−11.0,−7.7], 201 paired — contradicted (negative control)
- safe partner donation: +0.05 [−1.1,+1.2] — within-CI (a genuine draw)

The two "contradicted" are claims about BAD plays: the negative deltas confirm the
plays lose, and the champion's low obey rate (~0.32) shows it learned to avoid them —
exactly Roberson's "don't throw count to the bidder." `CHAMPION_SPECS` appended to
the registry (additive); central ledger untouched.

**Frontier shift:** the teaching half is no longer aspirational — the champion's play
can be graded against the book with paired-contrast receipts. Out of scope (honest):
84-endgames (champion rarely bids 84), multi-step ch03/ch10 claims (need
trajectory contrasts), paired-bid auction claims (one bid per auction), and a
production n_worlds=50 run to halve the CIs.

---

## [2026-06-13 | pending | Champion #27 v2 pass baseline — win-rate measured null]

The fast net bidder (rung #22) made the once-impractical experiment cheap: does
crediting the defensive cost of passing actually win games? `net:wp,pass0.4` vs
`net:wp` (identical lens:ev play), 128 games — **61/128 (47.7%), −0.07 marks/game,
95% CI [−0.62, +0.46]** (includes zero). The pass baseline does what it's designed
to — A takes more auctions (54.1% offense share) — but the extra marginal contracts
are made at a lower rate (65.6% vs 69.1%), so volume and quality cancel: q=0.4 sits
near the break-even point. The behavioral change is real and win-rate-neutral; the
*right* q, derived from self-play rather than a hand-set scalar, is rung #26.

---

## [2026-06-13 | pending | Champion #24 + #26 scaffolded + adversarially reviewed]

Resumed the [[champion]] epic to land the heavy-training frontier the last session
documented-but-didn't-build. The handoff's blocker — "bid tokens grow
`tokenize.py`'s vocab and break every adapter" — dissolved on reading the code: the
**auction is a side feature**, not tokens. `gus/model/auction.py
auction_feature_vector` (per-relative-seat bid/pass/winner + winning-bid level + a
declared-trump one-hot) → `BidsEncoder` → added to the pooled state_emb, mirroring
`VoidsEncoder` exactly (`StudentTransformerFullVoidsAuction`). No tokenizer change,
so the #25 belieflens and the gus bidder load and behave identically; `load_gus`
auto-detects via the `--auction` flag. Same "find the lower-risk seam" move as #25's
`compute_eq_weighted_mean`.

**#26 data bridge** (the reason #24 can mean anything): the belief corpus had no real
auction (seed deal + imposed bid), so `arena.cli --emit-snapshots` now dumps each
contracted hand's deal + real per-seat auction, and
`forge.cli.generate_eq_from_snapshots` runs the SAME oracle E[Q] generation on those
deals — with the **declarer leading the first trick** — stamping the auction onto each
`GameRecordGPU`. Proven end-to-end on MPS. Two-track build (model vs bridge) split by
file-conflict structure; a background agent owned the bridge.

**Adversarial review workflow** (5 dimensions → verify) confirmed 9 findings, dismissed
5. The keeper was CRITICAL: the bridge generated every deal with seat 0 leading while
the stamped bidder varied — a train/inference mismatch. Fixed (`from_deals` sets
`leader=bidder`). Also added the declared-trump to the feature (the head's own
"declared fours ⇒ holds fours" rationale was missing) and hardened the gus bidder
against an auction model. Follow-ups #30/#31 filed.

Trained measurement is **GPU-gated** — the box was busy with other training all
session — set up as a train-time A/B (auction vs voids control on the SAME
real-auction corpus) so the delta isolates the auction's contribution. Honest prior:
with the conservative `net:wp` bidder the live signal is mostly winner+suit (bid
magnitude near-degenerate, #31), so expect a small delta — one more honest
measurement in this project's tradition. Scaffold pushed to `origin/forge`
(d84e22e, b0b35d3, b4baec7); one-command training kickoff at
`scratch/champion-run/run_24_pipeline.sh`.

---

## [2026-06-13 | complete | Champion #24 auction-conditioned belief — MEASURED WIN]

The heavy-training frontier delivered its first measured win. Conditioning the [[gus]]
belief head on the completed auction (side feature, no tokenizer change) gives **+2.59pp
held-out belief accuracy** vs an identical voids-only control on the same real-auction
corpus — see [[w42-champion-auction-belief]]. Bulletproofed against the adversarial
review's two valid objections: it **generalizes** (3 independent corpora A/B/C: +2.42,
+2.21, +3.12pp; 11/11 seed deltas positive) and it's **information not capacity** (a
shuffled-auction control with the same BidsEncoder sits at voids level, −0.39pp, while the
real auction is +2.63pp). No leakage (one auction key → 22 distinct belief targets).

The better belief is **marks-neutral** under oracle play (auction belieflens vs lens:ev:
−0.29/game, CI [−0.92,+0.36], belief active at ESS≈4.8) — rung #25's lesson again. So the
thesis **auction ≫ belief ≫ utility ≫ play-polish** refines: #24 confirms the auction→belief
link is real and strong, while belief→marks stays weak when card play is already near-oracle.
#24's value is the belief quality itself — the substrate the self-play loop (#26) compounds.

Process: two-track build (model / data bridge) → pre-train adversarial review (fixed a
CRITICAL declarer-leads corpus bug) → 5-seed parallel measurement (10 trains, all cores) →
4-skeptic refutation workflow → multi-corpus + capacity matrix to answer the survivors. The
machine went from idle to ~10-core-saturated; the whole measurement landed in minutes.

## [2026-06-14 | pending | Champion design review promoted — Fable's recovered reasoning + 2 caveats]

**What happened:** The [[champion]] direction came from one 2026-06-09 design
session (Fable 5) that was compacted out of live context; only its conclusions
survived into [[champion]]. Mined the transcript (`0a708a4e`) and promoted the
five design turns verbatim to [[champion-design-review]], with a graded
predictions ledger (Fable's calls vs what the rungs measured) and the two
load-bearing caveats distillation had sheared off.

**Touched pages:** [[champion-design-review]] (new) · [[champion]] · [[index]] · [[log]]

**The two recovered caveats (each corrects a live over-claim):**
- **Information-blind arena** — both sides are PIMC, so the arena cannot reward
  belief/concealment value via play-marks. Reframes the #25 decisive null from
  "belief→play is dead" to "the harness can't see it here"; #24's belief value
  routes through bidding/defense (#26), shown by belief accuracy not arena marks.
- **Score-conditioning is auction-not-play** — #27's −1.20 negative tested play
  risk; Fable located mark-state value at the auction, which is unrun. (#31's
  dead bid-magnitude channel is a different thing.)

**Why:** distillation shears caveats, and the missing ones were quietly steering
parallel sessions toward "belief is dead / score-conditioning is the wrong lever"
when the real reading is "value is at the auction, and the play harness is blind
to it." Folded both back into [[champion]]'s ladder (#25, #27).

**Meta-pattern recorded:** every rung null/negative (#25, #27) is a *play-phase*
lever; the pattern is the marginal-value ranking confirming itself from the
bottom (`≫ card-play polish`), not the levers being worthless.

**Questions opened:** none (sharpens the auction-side frontier already in #26).

## [2026-06-14 | complete | Champion #26 self-play fixed point — converges to a calibratable over-bidder]

**What happened:** Ran [[champion-design-review|Fable's]] rung #6 to the end. Built the keystone
`champion/belief_bidder.py::BeliefBidder` (hypothetical-completed-auction + belief-weighted oracle
E[Q] → P(make) → score-conditioned util-max; 0.118 s/bid-turn) — the organ that closes the loop,
since the corpus = f(deal, decl, bids, bidder) and only a changing bidder iterates it. Two 4-round
self-play runs ([[w42-champion-selfplay-fixed-point]]).

**Result:** the loop reaches a **stable fixed point** (belief-KL 0.116 → ~0.08 plateau vs a measured
0.045 seed floor — genuine policy↔belief iteration). The fixed point is a **stable over-bidder**:
the belief bidder loses to `net:wp` by ~3.4 marks/game (A wins ~9/80) because the oracle's
double-dummy P(make) exceeds achievable PIMC play (strategy fusion, [[pimc]]). An optimism
correction (`pmake_scale=0.70`, the measured 0.58/0.83 gap) **halves the loss (~−2.2) and doubles
the wins (~20/80)** — confirming the diagnosis — but the PIMC-calibrated `net:wp` stays stronger.
Did NOT grind further scales (belief value is legibility, not marks).

**Measured-finding sidebars:** seed-noise floor of belief-KL ≈ 0.045 nats/slot (calibrates the
convergence threshold); ONNX export of the student is 40 KB with exact parity (enables the [[plunge]]
`onyx` player); 4-chunk arena process-parallelism netted only ~1.18× (MPS-dispatch-bound, not
idle-latency — real lever is batching the bidder across games, deferred).

**Also closed this session:** #30 (unified the gus adapter loaders), #31 (util-max bidder; bid-
magnitude structurally dead in 42 — measured null). New: `gus/eval/eval_belief_kl.py` (belief-KL
convergence metric), `train_v2_voids --out-belief`, deterministic bridge seed.

**Questions opened:** does a PIMC-calibrated (not double-dummy) value backing the belief bidder reach
parity with `net:wp`? — the principled next lever, left for a future rung.

## [2026-06-14 | local | Jud — the unified belief-conditioned core: direction + vocabulary]

**Touched pages:** [[jud]] [[champion]] [[index]]
**Added:** [[jud]] — names the unified belief-conditioned core the [[champion]] points at: one organ
that bids and plays as the *same act* by conditioning search on a learned belief, trained by self-play
over whole games. Fixes the precise vocabulary the project had been conflating — **solve** (exact
perfect-information 42, exhaustive backward induction per deal, `forge/oracle/solve.py`), **oracle**
(its ≈97% distillation; both are perfect-information values and share the brick wall — they cannot take
uncertainty as input), **eq** (the lift past the wall → a *distribution* per action, the honest object,
[[expected-q-value]]), **the blob** (eq's melted per-action distribution), **belief** (a learned
conditioned weighting over worlds, today only a post-hoc reweight, marks-neutral in play at #25 because
the [[arena]] is information-blind), **utility** (the collapse to a scalar; EV settled at
[[w42-lens-v1-utility-head-to-head]]). Core idea: belief belongs *inside* the search, not as a reweight
after it; the same operation at every depth makes bidding and play one act; signaling and
self-consistency follow from belief-in-the-rollout trained on whole games ([[champion-design-review]]).
The solve/oracle is bootstrap and referee, not the thing copied — jud's value target is realized
whole-game (belief-state) outcomes, not perfect-information Q ([[pimc]] strategy fusion). First picture:
a scratch sketch (`scratch/jud_demo/`, uncommitted) un-melts an eq blob by belief weighting (world
ESS 128 → 10.5, a contract's p_make 0.20 → 0.61) — the #25 belief value seen directly in the
distribution rather than through the information-blind arena.
**Updated:** [[champion]] (self-consistency section links the unified-core framing to [[jud]]);
[[index]] (jud added to the entities catalog).
**Retired:** none.
**Questions opened:** how is the value trained belief-native (on realized whole-game outcomes) rather
than distilled from perfect-information Q; what is the minimal first loop that makes a convention appear.
Status: a direction and a vocabulary, captured 2026-06-14 — not built; engineering deferred.

## [2026-06-14 | local | Belief-conditioned self-play — Fable's training approach, rediscovered (clear vs not)]

**Touched pages:** [[belief-conditioned-self-play]] [[jud]] [[index]]
**Added:** [[belief-conditioned-self-play]] — records the training approach behind [[jud]], separating
what Fable specified (sourced, clear) from this session's extension and the open gaps. **Clear (Fable,
verbatim [[champion-design-review]]):** the spine (posterior → belief-sampled worlds → solve/oracle
value → marks-to-7), the [[arena]] as measuring stick, and the self-play loop *as written* — play full
games → **retrain the belief** → re-derive the policy by belief-weighted [[expected-q-value|eq]] search
over the **fixed** oracle → repeat; conventions emerge. So Fable's loop trains the **belief**; the value
stays the perfect-information oracle. **Extension (this session):** belief *inside* the search, and the
**value itself** trained belief-native on realized whole-game outcomes — because a fixed perfect-info
value is why #26 ([[w42-champion-selfplay-fixed-point]]) converged to an over-bidder. In the source,
learned values appear only at the optional summit ("Gus V as leaf values"), not the main loop.
**Explicitly flagged NOT established:** whether the value-native step is Fable's intent (the source
shows a belief-only loop over a fixed value); the training mechanics (losses, targets, credit assignment
across bid + 14 plays, how the belief-updating rollout is computed, what "the policy" is once the value
is learned) — undetermined, the value-native loop has no implementation or objective; and whether the
deep fixed point is reachable / tournament-strong (#26 reached only a shallow one).
**Updated:** [[jud]] (links the training-approach page); [[index]] (topic added).
**Questions opened:** the training mechanics above, and whether the value-native loop's fixed point
exists and is strong — carried as explicit unknowns, not guessed.

## [2026-06-14 | local | Provenance correction — champion-design-review is a recovered summary, not Fable's words]

**Touched pages:** [[belief-conditioned-self-play]] [[jud]]
**Correction:** earlier entries this session leaned on [[champion-design-review]] as a "verbatim
primary source." It is a **recovered summary of Fable's conclusions** compacted from session logs; its
"verbatim" self-claim is not a reliable transcript. Pages demoted accordingly.
[[belief-conditioned-self-play]] gains a **"What is remembered of the coherent vision"** section
distinguishing (a) what the summary records, (b) what the participant remembers Fable emphasizing — a
trained model that changes how the game is played *even during search*, belief-conditioned search, and
"somehow all about bidding," as one elegant whole — and (c) interpretation. Recorded plainly: the
[[champion]] ladder (#20–#28) built **staples**, not this coherent vision; the over-bidder and the
play-side nulls measured the staples, not Fable's design, which has not been built.
**Questions opened:** the "somehow it's all about bidding" why; whether [[champion-design-review]]'s own
"verbatim" self-claim should be corrected on that page (it predates this session).

---

## [2026-06-14 | local | Fresh-eyes review — provenance correction reversed, decoration numbers relabeled, optimism meter built]

A fresh-eyes adversarial review (17-agent workflow + independent verification) of the
[[jud]] / [[belief-conditioned-self-play]] / #26 surface. Three outcomes, all working-tree-local.

**1. The provenance "correction" above (entry of the same day) was itself an over-correction —
reversed.** That entry demoted [[champion-design-review]] to "a recovered summary of Fable's
conclusions, not Fable's own words." It is now disproven: the page's verbatim forward-design section
is **byte-exact** to the original transcript (session `0a708a4e`). Verified this pass with difflib —
the assistant forward-design turn vs the wiki section normalize to **5590 == 5590 chars, similarity
1.0000**, all anchor sentences present in both. The canonical [[champion-design-review]] page was
never demoted in place (it remained correctly verbatim); only its two backlinking pages carried the
wrong "recovered summary / not his words" language. Those are now restored: [[jud]] (idea section +
backlink) and [[belief-conditioned-self-play]] (the "What the design review records" heading + the
sourcing paragraph). The honest framing: the page is **Fable's verbatim forward design + a later-pass
graded ledger synthesized on top** — synthesis above, faithful Fable below. (The [[jud]]-as-extension
line stands unchanged and is *confirmed* by the source: value-native / "changes-the-game-during-search"
is Fable's doubly-hedged optional step-6 summit, not his core; those phrases appear nowhere in the
1195-line transcript.)

**2. Decoration relabeled — prose that over-claimed measurement.** Three #26 numbers had no computing
script and are now labeled as the tuned knobs / asserted values they are, across [[champion]],
[[belief-conditioned-self-play|w42-champion-selfplay-fixed-point]], [[index]], and
`champion/belief_bidder.py`: (a) the "0.58/0.83 measured gap" justifying `pmake_scale=0.70` — never
computed, prose-only in ~5 files (0.83 even collided numerically with `a_offense_share=0.8333`);
(b) the "~0.045 nats/slot seed floor" — a hardcoded literal in the run script tagged "measured" with
no surviving artifact (actual KL never drops below 0.072); (c) "halves the loss" — really ~34%
(−3.4→−2.2). The real convergence evidence (KL plateau 0.072–0.080 + A-vs-B acc-gap collapse Δ−0.043→~0)
and the real loss (CI-excluding-zero every round) stand on their own without the decoration.

**3. The optimism meter — the first real "oracle as ruler" instrument** (`champion/optimism_meter.py`
→ committed `champion/optimism_gap.json` + `.png`). Computes oracle double-dummy P(make) (bid-aware
atlas, N=50) vs realized 4-seat make-rate (forge bidding parquet, N=604), best-declaration, by bid.
Findings: realized make-rate falls **0.52 @ bid30 → 0.11 @ bid42** (a global `pmake_scale` is the
wrong *shape*, not the wrong constant); the prose 0.83/0.58 does **not** reproduce (closest real
analog at bid 30 is oracle 0.64 / realized 0.52, ratio ~0.81); the optimism ratio is bid-dependent
(0.64–0.81 over 30–39). It is a distributional gap (different hand pools) — a first instrument; the
paired refinement (oracle over the exact parquet hands) is the obvious next step.

**Evidence rescued (Step 0).** The load-bearing #26 artifacts lived only in gitignored `scratch/`
(one re-run from loss). Copied to committed `champion/evidence/` (run26cal `kl.log`/`ab.json`/
`RESULTS.txt`; jud_demo `manifest.json` + figures + headline/counter-example `.npz`), and the 4-game
belief-vs-`net:wp` smoke `arena/results/*` committed. See `champion/evidence/README.md`.

**Touched:** [[jud]] [[belief-conditioned-self-play]] [[champion]] [[index]]
[[experiments/w42-champion-selfplay-fixed-point]] + `champion/optimism_meter.py`,
`champion/optimism_gap.{json,png}`, `champion/belief_bidder.py` (comment), `champion/evidence/`,
`arena/results/`.

**Questions opened:** does the oracle-minus-realized gap actually *move* as self-play learns to
signal (the meter is static today), or is it confounded by play-skill? — the test of whether "oracle
as ruler" is a real instrument. The deferred jud engineering (value-native training target + the
belief-conditioned *sampler*, which does not yet exist — today's belief only reweights) remains open.


## [2026-07-05 | local | jud engineering first cut — value-native endorsed, rank-vs-price mechanism]

**Touched pages:** [[jud]] [[belief-conditioned-self-play]] [[rank-vs-price]] [[pimc]] [[champion]] [[index]]

**Added:** [[rank-vs-price]] — the mechanism resolving the "somehow it was all about
bidding" fragment: PIMC's strategy-fusion optimism is a distribution-shape error that
cancels in play (argmax over siblings; rankings survive common-mode inflation) and
lands whole in bidding (tail mass read cardinally against pass/`race_wp`). One stroke
explains the play-side nulls (#25, #27) and the #26 over-bidder; measured legs:
`optimism_gap.json` (realized 0.52 @ 30 → ~0.19 @ 41 vs oracle 0.64 @ 30), the three
play nulls, and `net:wp`'s frozen-realized-calibration dominance.

**Updated:** [[jud]] — "The engineering, first cut" (Fable 5 session, 2026-07-05; new
session, no memory of `0a708a4e` claimed): value-native **endorsed** for the pricing
path, promoted from optional summit to spine. jud v0 = one added head (V_realized:
info-state → distribution over realized hand margin, categorical CE on realized
outcomes from the #26 arena bridge, MC targets, coverage via ε/forced-bid corpora),
bidder prices via tail mass at the hypothetical-auction root through `MarksToSeven`
(auction-side score-conditioning rides along), play stays `lens:ev`, `pmake_scale`
retires. Factorization law (learn the unknown / compute the exact), the referee
instrument (oracle EV − V_realized EV = price of hidden information), three registered
predictions (v0 ≥ `net:wp` parity; V calibration matches the realized curve; the
fixed point stops over-bidding), and the v1/v2 ladder (search leaves; opponents-in-
rollout signaling). [[belief-conditioned-self-play]] — the open "is value-native
Fable's intent" question split: historical intent stays open (likely permanently);
design question closed by the Fable 5 endorsement; training mechanics now first-cut.
[[pimc]] — rank-vs-price section. [[champion]] — self-consistency section points at
the first cut. Provenance line preserved: the value-native extension was the
2026-06-14 session's, and it was right.

**Evidence rescued:** `champion-one-organ-theory-2026-06-14.md` and
`handoff-2026-06-14-jud-vocab-and-fable-words.md` copied from gitignored scratch into
`champion/evidence/` (they back [[jud]]'s conclusions and the provenance chronicle).

**Questions opened:** none new; jud's open engineering narrows to the v1 search shape
and v2 opponent-model mechanics.

## [2026-07-06 | d678598 | arena perf pass — 2.38× games/sec on MPS, byte-identical]

**Touched pages:** [[arena]]

**Updated:** [[arena]] — dispatch/sync reduction on the oracle decision path
(numpy-assembled state tensors, vectorized order-preserving pool construction,
maskless `scatter_add` void aggregation, MRV loop ~45→~20 kernels/step with dead
per-step syncs removed, per-device table caches, memoized `current_player`).
Byte-identical to baseline on CPU and MPS was the correctness gate. Paired MPS
bench: 0.56 → 1.34 games/s (2.38×), reproduced across two A/B pairs; post-merge
production throughput ~1.34 games/s on a pooled 128-game A/B. Key finding: the
arena is dispatch-bound, not compute-bound — the oracle forward dominates CPU
wall (62%) but shrinks on MPS, where per-tick kernel-dispatch/sync overhead
(~1,500–2,000 launches, ~25–45 syncs) becomes the bottleneck. Full profile:
`docs/arena-perf-2026-07-06.md`. A second pass (constant-batch-width refill,
gate relaxed from byte-identity to distribution-level equivalence by user
decision) is in flight as of 2026-07-06.

**Questions opened:** none new.

## [2026-07-06 | 4080e07 | jud v0 built and graded — value-native bidder reaches net:wp parity]

**Touched pages:** [[w42-jud-v0]] [[jud]] [[champion]] [[rank-vs-price]] [[index]] [[log]]

**Added:** [[w42-jud-v0]] — the closing write-up for jud's first buildable slice
(Champion rung #32, GitHub #32). A value-native bidder prices contracts from a head
(`V_realized`/`champion/margin_net.py`) trained on **realized** 4-seat self-play
outcomes instead of the double-dummy oracle; the bidder (`champion/value_bidder.py`,
CLI `margin:wp`) reads tail mass at the hypothetical-auction root through
`MarksToSeven`; play stays `lens:ev`; `pmake_scale` retires. Graded against the three
registered predictions: **P2 calibration PASS** (ECE 0.046, max |Δ| ≤ 0.029 over 13
thresholds, 6× closer to realized than to oracle, sits 0.09–0.13 below double-dummy);
**P1 round-0 parity MISS** (−1.44 [−1.88, −0.95] — a legible over-bidder that wins
points (+5.66/hand) and loses marks, via a winner's-curse-on-*selection* channel plus a
notrump declaration-level artifact, both independent of double-dummy optimism); **P3
self-play loop PASS** (4 rounds carry the margin −1.44 → −0.31 → +0.24 → +0.24 → +0.22,
CI includes zero from round 2; made-rate 49.9% → 60–64%; notrump artifact dies in one
on-policy round, share 47.5% → 1.5%). Canonical same-seed check: −0.07 [−0.66, +0.49],
**statistical parity** with `net:wp` while winning +7 points/hand. Two methodological
findings: coverage anchoring beats single-variable recipe purity (the recipe fork —
dropping net:wp self-play chunks regressed round 1 to −2.18), and the `MarksToSeven`
pass baseline is a denial-bidding lever that makes over-bidding worse (the A2 sign-catch,
credited to the value-bidder subagent). Evidence at `4080e07`
(`champion/evidence/jud_v0/`). Definitive 512-game same-seed A/B: **−0.01/game
[−0.28, +0.25]**, 258/512 — dead parity (`ab_definitive_512_r4_summary.json`).

**Updated:** [[jud]] — honest status flipped from "not built" to "v0 built and graded";
predictions ledger graded in place (P2 pass / P1 miss→loop-recovered / P3 pass); v1
named as the parity-breaking frontier. [[champion]] — self-consistency section carries
the rung #32 outcome; champion's bidder stays `net:wp` for now (`margin:wp` its
value-native equal on marks, superior on legibility). [[rank-vs-price]] — leg 3
(the pricing mechanism's full test) confirmed at parity, with the winner's-curse-on-
selection rider the mechanism did not originally name.

**Frontier shift:** the value-native pricing path is validated — realized-outcome pricing
dissolves the #26 over-bidder without a tuned knob, and reaches the best hand-tuned
baseline. But it converges *at* parity, not past it (offense share plateaus ~60–67% vs
the predicted selective 50–55%). Whether the residual is the [[pimc]] price of hidden
information or further calibration headroom is v1's question — value at the leaves of
shallow belief-state search in play and defense.

**Questions opened:** what breaks the parity plateau (v1 search shape vs opponent
modeling); reading the referee gap (oracle EV − V_realized EV) as a live convergence
instrument rather than a static meter.

---

## [2026-07-06 | 68fda7b + 0bdd4d5 | the plateau probe: data starvation, not structure]

The jud v0→v1 bridge. [[w42-jud-v0]]'s open question 1 — was the parity plateau the
[[pimc]] price of hidden information (structural) or a data-starved tiny MLP? — run as a
registered prediction and answered.

**Touched pages:** [[w42-plateau-probe]] [[w42-jud-v0]] [[jud]] [[champion]] [[rank-vs-price]]

**Added:** [[w42-plateau-probe]] — the registered-prediction write-up (GitHub #33). The
structural reading was registered pre-run as a falsifiable prediction: scaling on-policy
data would NOT break parity. **Falsified.** Rounds 5–8 at 3× data/round (1000 self-play
games/round vs 300) carried `margin:wp` past `net:wp` — head_8 beats it **+0.38
[+0.09, +0.67]** (reserved seed 7000000, 512 games) and **+0.42 [+0.12, +0.72]** (fresh
seed 9000000), both 287/512 (56.1%), with round 7's 256-game A/B independently excluding
zero. **The first learned bidder to beat the hand-tuned champion on marks.** The plateau
was calibration headroom in a data-starved head, not the hidden-information price. A
registered extension (rounds 9–12) confirmed **saturation**: head_12 at +0.21 [−0.08, +0.47]
/ +0.37 [+0.09, +0.65], inside the registered [+0.2, +0.6] band, indistinguishable from
head_8 — the data-scaling curve flattens at **≈ +0.3–0.4 marks/game** at this net capacity.
The page carries the full r0–r12 round table (`champion/evidence/jud_v0/loop_metrics.json`).
The registered prior was wrong, recorded plainly — a falsified prediction run to its
falsifier is the system working.

**Updated:** [[w42-jud-v0]] — addendum + open question 1 marked ANSWERED (data starvation),
pointing to the probe. [[jud]] — honest status flipped from "reaches parity, does not yet
beat" to "past parity, saturating at ≈+0.3–0.4"; v1 re-described as one net for bid + play,
in build. [[champion]] — the bidder claim flipped: `margin:wp`(head_8) is the first learned
bidder to beat `net:wp`, best-measured bidder is `champion/margin_net_r8.pt`; saturation
noted. [[rank-vs-price]] — leg 3 upgraded from "validated at parity" to "validated and then
dominant"; the winner's-curse-on-selection channel is data-limited, not structural.

**Frontier shift:** the value-native pricing path no longer ties the best hand-tuned
baseline — it beats it. The binding constraint at v0's scale was on-policy data volume, not
the [[pimc]] price of hidden information; that reading is refuted at this scale. Data has
run its course at this net capacity, so the next constraint is capacity or mechanism —
jud v1's premise (one net, bid + play; play-history-conditioned V_realized with 1-ply
argmax-EV play replacing E[Q] n=10 at runtime; then the same self-play-loop method).

**Questions opened:** none new — the probe closed [[w42-jud-v0]]'s open question 1.

## [2026-07-06 | 0d82a97 | jud v1 built: one organ, two consumers (6860a75..0d82a97)]

Three commits building jud v1's machinery: play-decision snapshot emission, the unified
JudNet organ, the judplay consumer, and a graded round-0 head.

**Touched pages:** [[entities/jud]] [[sources/0d82a97]]

**Added:** 1 source digest.

**Frontier established:**
- One net now serves bid and play: info-state (own hand + canonical auction + play
  history) → 43-bin realized-points categorical; bid-time = empty-history play-time,
  byte-identical to margin_net's root encoding (tested train/serve both ways).
- The arena emits per-decision corpora compactly: `HandRecord.plays` (28 seat/domino
  pairs), every decision a prefix, offense and defense rows sharing the hand's Monte
  Carlo label. Works in sequential and fast-batching modes.
- Registry: `jud[:wp][,pass<q>][,model=]` bidder (ValueBidder unchanged) and
  `judplay[:model=]` play (greedy depth-1, argmax E[pts], defenders minimize, one
  forward per tick).
- Round 0 graded honestly (champion/evidence/jud_v1/): combined −6.09 vs
  net:wp+lens:ev, bidder −4.08, play −5.53 (defense the biggest channel);
  judplay beats random +1.80; value sharpens with depth (MAE 8.6 → 3.5).

**Questions opened:**
- Overfitting is the binding constraint (memorizes 200K rows in ~1 epoch at lr 1e-3);
  regularization/data scale for loop rounds.
- Root-row dilution (bid roots are 2/56 of samples; worst ECE slice) — up-weight or
  let the loop close it?
- Does the v1 self-play loop dissolve the round-0 over-bid (93% offense share) the
  way v0's did?

## [2026-07-06 | 3ac03de | jud v1 graded: one organ, bid and play (f550205..3ac03de)]

Four commits carrying jud v1 from build to a graded verdict, every rung registered on
GitHub #33 before its measurement: `f550205` (the organ — one net, play-history
snapshots, the `judplay` consumer), `9d30b25` (the loop grades JP1/JP2/JP3), `e596205`
(judsearch — belief-lift worlds, current-trick rollout, V_realized leaves), and
`3ac03de` (the search-ladder grades JS1/JS2/JS3 + the night verdict).

**Touched pages:** [[experiments/w42-jud-v1]] [[entities/jud]] [[entities/champion]] [[topics/rank-vs-price]]

**Added:** [[experiments/w42-jud-v1]] — the full arc as a registered-prediction ledger.

**Updated:**
- [[entities/jud]] — v1 promoted from "machinery built" to "built and graded"; the
  policy-conditional pricing law added to the vocabulary; v1/v2 ladder bullets and the
  honest status rewritten to the graded verdict.
- [[entities/champion]] — current best player stated: `margin:wp`(head_8)+`lens:ev`
  (+0.38/+0.42 over `net:wp+lens:ev`); rung #33 recorded as built-and-graded, not
  displacing the champion.
- [[topics/rank-vs-price]] — the play half measured: greedy value play is a bad ranker,
  search recovers most (not all) of the gap oracle-free, the oracle's rankings stay
  unbeaten.

**Frontier verdict:** the one-organ unification **holds at the auction and is
mechanism-limited at play**. The bidder survives the fold intact (beats v0's own round-0
bidder); greedy 1-ply value play is a bad move-ranker (the loop moves it zero, JP3
falsified); `judsearch` recovers two-thirds of the play gap oracle-free (−3.44 → −1.16,
JS1 PASS +2.28) but not parity, and neither more worlds (JS2 below band) nor a
better-calibrated head (JS3 falsified) closes the rest. The wall is per-move
discrimination — a 470k MLP on hand-level Monte-Carlo labels cannot out-rank E[Q] n=10's
per-move oracle. The stack went −4.37 → −1.43 oracle-free in one night; the current best
player is unchanged. v2's cue is concrete: a bigger leaf on per-move targets (E[Q]
distilled as bootstrap) plus opponents-in-rollout.

**Questions opened:** none new — v1's play wall is named, and v2's target follows from it.

## [2026-07-06 | working-tree | the wall, stated precisely — distill-for-what + candlewax concordance]

**Touched pages:** [[candlewax]] [[jud]] [[w42-lens-v1-utility-head-to-head]]

**Updated:** [[candlewax]] — two new sections. (1) *Concordance*: candlewax ≡ bimodal/
multimodal PDF ≡ jud's "melted blob" ≡ mixed-mode geometry, with the founding-era
provenance (report/11, 2026-01-06: −42→+40 swings, 11% stable hands, 53% within-hand
variance; the 85-bin discs rendered 2026-01-24 — ~3.5 months before the name). (2) *The
wall, stated precisely* — Jason, verbatim: "I saw eq, I said sure I could distill it.
but for what purpose? no idea what to do with distilled melted candlewax." Distillation
was never the wall; CONSUMPTION is. Every era is a successive consumer hypothesis
(LLM-as-reasoner → tool surface → Lens utilities → rank-vs-price → jud), and any
"distill X" proposal must first name the consumer and the licensed collapse.
[[jud]] — the blob entry now names the identity with [[candlewax]]. [[w42-lens-v1]] —
the EV-wins verdict now carries the aggregate-vs-per-decision reconciliation note
against [[candlewax]].

**Context:** first surgical edits from the wiki-overhaul mining (5 scout reports in
scratch/wiki-mine/); the full charter (trails, founding-era backfill, page zero) is
pending approval.

**Questions opened:** none new.

## [2026-07-06 | afd4802 | wiki overhaul: archaeology backfill (eras 1-5) + full staleness audit (era 6)]

The seven-month archaeology, landed. Two moves at once: (1) **backfill** — ~30 new pages
reconstructing the project's pre-wiki history, one page per era-question, every date traced
to a git/bead/conversation timestamp and every named thing classified BUILT / IDEATED /
RENAMED; (2) **audit** — a full pass over the existing wiki that reconciled ~173 pages
against the repo, flipping stale `status: active` claims to their true frontier state and
making every correction reachable from the page that carries the stale claim. New front
door: [[the-wall]] → [[the-wall-biography]] → [[consumption-ledger]]. New anti-rot rules
codified in `AGENTS.md` (status-must-be-falsifiable, corrections-reachable-from-error,
questions-not-goals, names-doctrine, dates-trace-to-artifacts, privacy-firewall,
worth-a-bead-is-not-a-resting-state).

**Touched pages (hubs + spine):** [[the-wall]] [[the-wall-biography]] [[consumption-ledger]]
[[web-game]] [[the-oracle]] [[breakthrough-and-oracle]] [[eq-genesis]] [[strategy-fusion]]
[[argmax-q-ceiling]] [[alphazero-under-imperfect-information]] [[the-gestation]]
[[ideated-not-built]] [[zeb]] [[zeb-fleet-ops]] [[lem]] [[burl]] [[gus]] [[forge]]
[[champion]] [[jud]] [[index]]

**Added (~30 new pages):**
- Era backfill topics: [[pre-ml-ai-attempts]], [[multiplayer-lineage]], [[the-book-enters]],
  [[breakthrough-and-oracle]], [[the-oracle]], [[the-analysis-epic]], [[suit-algebra]],
  [[eq-genesis]], [[strategy-fusion]], [[alphazero-under-imperfect-information]],
  [[argmax-q-ceiling]], [[belief-feeding-policy]], [[eval-matrix-bradley-terry]],
  [[the-gestation]], [[ideated-not-built]], plus the front-door pair [[the-wall]] +
  [[consumption-ledger]].
- Entities: [[web-game]] (era-1 founding substrate), [[zeb-fleet-ops]].
- Experiments: [[full-teacher-eq-experiment]], [[gus-drama-atlas]].
- Decisions: [[qval-over-policy-models]], [[vs-random-eval-is-suspect]], [[grok-not-converge]].
- Source digests: [[sources/claude/era1-web-game-prologue]], [[sources/claude/era2-breakthrough-oracle]],
  [[sources/claude/era3-eq-era]], [[sources/claude/era4-zeb-era]], [[sources/claude/era5-gestation]]
  (privacy-curated claude.ai user turns; Texas-42 content only).
- Trail: [[the-wall-biography]] (seven-month capstone).
- Index: 7 previously-unindexed w42 strategy-tag experiments folded into the catalog.

**Updated:** ~173 pages status-reconciled per the era-6 audit; `index.md` catalog synced
(111 status-suffix corrections + 37 new/backfilled entries + a new "Era backfill" Topics
subsection + a claude.ai-digest Sources subsection + the-wall front-door entry);
`questions/open.md` extended (4 items, below).

**Retired / superseded (frontier flips, not deletions):** LEM and its Stage-0 adapter
chain → `complete`/`superseded`; [[burl]] and its lab/chat/microscope/wax-museum surface →
`superseded`; [[zeb]] → `superseded` (belief work carried by Gus/jud); the Winning-42
per-chapter book cluster and its phase-2/3/4 probes → `complete`/`superseded` as the
campaign closed; the LAMIR/router/blunder-detector no-oracle branch → `retired`/`superseded`;
[[candlewax-spike]] and [[burl-selfplay-arena]] → `retired`. The live frontier
([[gus]], [[w42]], [[forge]], [[champion]], [[jud]], [[engine]]) stays `active`.

**Questions opened (raising `afd4802`):**
- `WorldSamplerMRV` sampler bias (~6.8 Q-pts vs enumeration at trick 6) — parked "worth a
  bead," never filed; **needs a GitHub issue** (anti-rot rule 7).
- The never-applied Lens ev-argmax switch — production `select_actions` is still Lens(p_make),
  the worst of four utilities; the one-line ev-argmax fix is open two months on.
- jud v2 cue — does a bigger leaf on per-move targets + opponents-in-rollout close the play
  gap the v1 hand-level MLP could not?
- The era-5 gestation's IDEATED generation — which unbuilt designs are worth resurrecting?

**Lint:** dead backlinks in the new era pages fixed (`[[the-engine]]`→`[[engine]]`,
`[[era-1-web-game-prologue]]`→`[[web-game]]`, `[[era-2-breakthrough-oracle]]`→[[breakthrough-and-oracle]],
`[[the-wall-stated-precisely]]`/`[[distill-for-what]]`→[[the-wall]]/[[candlewax]], phantom
`[[layer-system]]`/`[[mccfr-excursion]]`/`[[walker]]` de-linked or redirected); two private-memory
filename links purged from [[burl-chat-spike]] and [[post-commit-q-and-a]] (privacy firewall);
qualified body links in the new pages converted to bare. No orphans (every new page has an
inbound content link). Flagged-not-fixed: dead `[[plunge]]`, `[[burl-perf-phase1]]`,
`[[topics/spec-decode-acceptance]]`, `[[explore-game-cache-bug]]` in pre-existing modified
pages (missing-page candidates), and the two distinct "~74%" ceilings (era-3 argmax-vs-oracle
tie-structure vs era-4 Zeb vs-random capacity) that no page cross-claims as identical.

## [2026-07-07 | 5e3f3245 | book second pass: what the first extraction missed]

**Touched pages:** [[w42-book-second-pass]] [[w42-book-validation]] [[w42-bookval-v1-wave2-pounce-high-bid]]
**Added:** [[w42-book-second-pass]] — four parallel readers re-read the full OCR text with the
finished campaign as lens. The first pass extracted the book's tactics and missed its
information theory: the auction decoder (ch 6/12 bid→hand posteriors, the {30,31,35,36} bid
lattice, who-bid asymmetry, match-score-conditioned bidding), the action-choice inference
catalog, the signaling conventions (top-unplayed-trump as protocol, donate-highest code,
dump-to-inform, Plunge as legal one-bit signal), reputation-driven overbidding, the
quantified-prior calibration table, and multi-step plans with author-supplied win rates
(strip-the-protector p.92/94, double-ahead-of-off 53/60/33). Nine ranked follow-up
experiments; raw reader reports preserved at `wiki/sources/book-second-pass-2026-07-07/`.
**Updated:** [[w42-book-validation]] trail (frontier section routes to the second pass);
[[w42-bookval-v1-wave2-pounce-high-bid]] gains caveat 0 — the `contradicted` verdict is a
probable information-regime category error (book's clause is an imperfect-information hedge,
"regardless of whether you know who will win the trick," tested under a perfect-information
oracle).
**Questions opened:** does the pounce contradiction dissolve under a belief/PIMC defender
(bid ≥ 35, bidder ≤ 2 offs)? Do the book's bid→hand posteriors hold empirically, and does
head_8 respect the {30,31,35,36} bid lattice? OCR re-scan needed: book pages 181–182 and
185–186 absent, ch 16 four-trump table truncated, worked-hand diagrams are images.
Filed in `questions/open.md`.
**Curation (2026-07-10, PR #35 landing):** provenance pinned to `5e3f3245`; the
"named untested gaps" claim on [[w42-book-second-pass]] §1 re-attributed from jud's
page to the campaign synthesis ([[w42-book-claim-synthesis-and-ai-directions]]),
where the list actually lives; [[jud]] Links section now routes to the second pass
as the book-sourced experiment queue for the auction-first frontier and v2
opponents-in-rollout; opened questions filed into `questions/open.md`.

## [2026-07-07 | working-tree | experiment-page audit: 162 pages validated against primary artifacts, 43 corrected in place]

**Touched pages:** all 162 `experiments/` pages audited via two-pass fan-out (162 auditors, then adversarial re-review of every corrected page); 43 corrected in place, no audit residue left on pages.
**Updated (highest-weight):** [[gus-lamir1-piopp]] (Bug-6 outcome was inverted — the world_assign fix made regret *worse* 2.268→2.350; root cause is scalar V/Q distillation noise flipping argmax, not Q_head depletion-OOD; pivot options replaced with the real four from MORNING4_STATUS @ b42669a), [[gus-q-head-augmentation]] (conclusion rewritten to the sourced diagnosis), [[batch-throughput-bench]] (baseline is 83 tok/s not 43; prompt-cache reuse *was* tested — negative, see [[burl-perf-phase2]]), [[burl-perf-phase0]] (K1 flip was gi=36, not gi=72), [[gus-belief-co-train]] (2×2 regret table disentangled; §20-vs-§21 source discrepancy noted in place), [[gus-belief-calibration-diagnostic]] (receipt quoted verbatim), [[iter5-e1-rank-sweep]] (2048-vs-1024 truncation-ceiling source conflict noted in place), [[iter3-rules-adapter]] (fix landed 17 commits after, not three).
**Trail:** per-page audit ledgers ("page said X; artifact says Y, evidence path") live in git history at `45a7e358` / `29641ded`; live questions surfaced by the audit were already tracked in `questions/open.md` or on their pages.
**Lint:** zero new dead links; pre-existing dead `[[burl-perf-phase1]]`, `[[topics/spec-decode-acceptance]]`, `[[log]]` (burl-perf-phase3), `[[sources/<sha>]]` placeholder (burl-star-run3) remain flagged from the era-6 audit.

## [2026-07-10 | working-tree | log rotated: changelog-not-chronicle rule, digest + archive]

**Touched pages:** [[log]] [[log-archive]] `AGENTS.md`
**Added:** [[log-archive]] — entries 1–145 (2026-04-09 → 06-14) moved verbatim; log.md keeps a phase digest + last ~10 entries.
**Updated:** AGENTS.md log section — entry budget (~5 pointer lines), mechanical rotation trigger (>15 entries), no claim without a link.


## [2026-07-11 | bc4eb386 | partnership wall — cumulative record to measurement spine]

**Touched pages:** [[partnership-wall-research]] [[partnership-value]] [[partnership-research-gates]] [[the-wall]] [[consumption-ledger]] [[arena]] [[forge]] [[champion]]
**Added:** [[partnership-failure-atlas-v0]] [[world-sampler-mrv-audit]] [[partnership-decision-record-v1]] [[sources/bc4eb386]]
**Measured/built:** five-way 75,079-action join + 114-source seam inventory; legacy MRV malformed/bias mechanisms; rejected uniform-rejection repair; exact completion-count sampler; replay-verified Arena records with C0 policy and leakage fingerprints.
**Frontier:** CUDA/MPS sampler performance, historical exposure, two-block C0 reproduction, forced causal arms, and information-reactive fixed/shuffled partnerships remain open; no successor architecture selected.

## [2026-07-11 | a2bb0437 | result vocabulary — partnership remains untested]

**Touched pages:** [[partnership-wall-research]] [[partnership-failure-atlas-v0]] [[world-sampler-mrv-audit]] [[partnership-research-gates]] [[sources/a2bb0437]]
**Updated:** archive insufficiency is not a partnership null; three no-flip sampler fixtures are a bounded observation; the confounded `~6.8 Q` estimate stays retired; failed uniform rejection is the genuine negative design result.
**Frontier:** no negative or null result about partnership value has been measured.

## [2026-07-11 | 5f314d2b | partnership research review surface]

**Touched pages:** [[partnership-wall-research]] [[sources/5f314d2b]] [[index]]
**Updated:** one review-first table now distinguishes measured/built/open/untested/designed work; four ordered gates route baseline cleanup → causal runner → first partnership discriminator → architecture selection.
**Frontier:** [[partnership-wall-research]] is the single PR-review entrypoint.

## [2026-07-11 | b89ff635 | docs→wiki consolidation: game-of-42 cluster, engine second pass, forge/burl/lem/gus promotion, entrypoints rewritten]

**Touched pages:** [[texas-42]] [[rules-of-42]] [[suit-algebra-spec]] [[play-phase-algebra]] [[engine]] [[engine-architecture]] [[layer-system]] [[multiplayer-pattern]] [[client-implementation]] [[engine-testing-patterns]] [[intermediate-ai]] [[forge]] [[expected-q-value]] [[the-oracle]] [[gus-qmean-router]] [[router-reality-check]] [[engine-adrs]] (+~30 more: hooks, citations, sha-stamps; waves 2ab1a825, d1f1633d, e2171816, 522779c5, b89ff635)
**Added:** the game-of-42 cluster (rules + algebra + play phase), the six-topic engine reference cluster, [[gus-qmean-router]] (the no-oracle router that works), [[engine-adrs]]; `sources/` gains pi-oracle-bidding {question,answer}, mccfr-exploration, and the book-second-pass reader reports (relocated from docs/)
**Updated:** [[router-reality-check]] corrected (replacement hurts, second opinion helps); [[ls-mixture]] mis-expansion fixed (always the arxiv short/long sense); forge foot-guns/folk-wisdom/training-data doctrine promoted into [[forge]] and [[expected-q-value]]; ~12 stale engine-doc claims corrected against current code while writing the cluster
**Retired:** docs/{adrs,archive,research,wiki-mine} and docs core+theory+rules files (rules-tournament.md unmigrated — erroneous), 36 burl/gus session docs, forge/eq/cpu_deprecated/ (no-legacy violation), SPIKE_REPORT.md, MORNING_DIGEST.md; CLAUDE.md/AGENTS.md/README.md rewritten wiki-first (beads → GitHub issues)

## [2026-07-11 | 4123b2d5 | review repairs — MPS sampler defect + prior-sweep completion + rebalance]

**Touched pages:** [[partnership-wall-research]] [[partnership-research-gates]] [[the-wall]] [[world-sampler-mrv-audit]] [[wiki-entrypoints]] [[burl]] [[forge]] [[sources/4123b2d5]]
**Added:** [[sources/4123b2d5]] — MPS int64-gather defect in the shipped sampler repair, fixed with per-device uniformity regressions.
**Updated:** prior sweep completed ([[w42-champion-selfplay-fixed-point]], [[lamir1-ceiling]], [[strategy-fusion]], [[past-belief-future-direction]], [[pi-opp-head]], Plunge/Splash); clairvoyance decomposition registered as gate 2; partnership reframed as one registered direction on [[the-wall]]; `~6.8 Q` resolved-question entry rewritten as a split; source-digest correction shrunk to a one-line qualifier.
**Frontier:** CUDA benchmark, exposure scan, two-block C0 reproduction, and the clairvoyance bound precede the causal runner.

## [2026-07-11 | c7f74f5c | measurement-ready frontier — infrastructure before path selection]

**Touched pages:** [[partnership-wall-research]] [[partnership-value]] [[partnership-research-gates]] [[the-wall]] [[wiki-entrypoints]] [[index]]
**Added:** [[sources/c7f74f5c]] — review correction and domain synthesis behind the cleaned PR frontier.
**Updated:** Q-mean is restored as bounded positive consumer evidence; natural policy legibility is separated from sparse intentional signaling; wall promotion is separated from the additional fixed-vs-shuffled partnership criterion.
**Frontier:** PR 39 delivers trustworthy measurement infrastructure and an evidence ledger; no next experiment, causal microgame, or successor architecture is selected.

## [2026-07-12 | 1a4482fe | convention-aware blueprint search preserved without selection]

**Touched pages:** [[convention-aware-blueprint-search]] [[partnership-wall-research]] [[partnership-value]] [[sources/1a4482fe]] [[index]]
**Added:** the SPARTA-style blueprint proposal and its separate project-record refinement; neither is promoted to a build.
**Questions opened:** can [[w42-book-second-pass|Winning 42]] conventions seed a shared codebook while a learned policy supplies the complete blueprint?
**Frontier:** the design is IDEATED, unbuilt, and unselected; clairvoyance remains a consumer-specific sensitivity probe rather than a universal bound.

## [2026-07-12 | a6590bf6 | book-seeded coordinated initialization promoted into research trail]

**Touched pages:** [[convention-aware-blueprint-search]] [[partnership-wall-research]] [[partnership-research-gates]] [[sources/a6590bf6]] [[index]]
**Updated:** Winning 42 becomes a visible candidate codebook overlay; convention value is a sender x partner-reader x opponent-reader interaction, not a double-dummy rejection label.
**Questions refined:** can the sparse overlay become a complete calibrated blueprint whose partner gain survives opponent decoding and full-match marks?
**Frontier:** the mechanism is preserved above leaf level but remains IDEATED, unbuilt, unmeasured, and unselected.

## [2026-07-12 | d5816915 | blueprint hypothesis framing rebalanced]

**Touched pages:** [[convention-aware-blueprint-search]] [[partnership-wall-research]] [[partnership-research-gates]] [[sources/d5816915]] [[index]]
**Updated:** the surviving structural case now leads; Winning 42 initialization, existing infrastructure, causal attribution, and four-seat inference receive the same weight as the engineering requirements.
**Corrected:** repeated status caveats no longer imply a negative result; no contrary experiment exists.
**Frontier:** blueprint search remains one candidate among several, with durable research-trail visibility and no editorial presumption against it.

## [2026-07-12 | f6b691da | belief-weighted Jud MCTS preserved and synthesized]

**Touched pages:** [[belief-weighted-jud-mcts]] [[jud]] [[partnership-wall-research]] [[partnership-research-gates]] [[sources/f6b691da]] [[index]]
**Added:** belief particles → information-set MCTS → blueprint policy → Jud realized-value leaf, grounded by JudSearch's `+2.28` gain and the JS2 worlds-sweep boundary.
**Separated:** J0-J4 attributes root belief, adaptive depth, information-set updates, and convention value; determinized and information-set MCTS have distinct promotion gates.
**Frontier:** the idea is a surviving search-consumer hypothesis; Zeb and LAMIR are relevant prior evidence but did not test this combination.

## [2026-07-13 | b28fb55a | continuation-frontier research ingested — lanes selected, MCTS backup semantics corrected]

**Touched pages:** [[research-lane-selection]] [[search-literature-transfer]] [[auction-decoder]] [[belief-weighted-jud-mcts]] [[jud]] [[convention-aware-blueprint-search]] [[the-wall]] [[partnership-wall-research]] [[sources/b28fb55a]] [[index]]
**Added:** [[research-lane-selection]] (the gates' step-3 experiment selection), [[search-literature-transfer]], [[auction-decoder]].
**Updated:** [[belief-weighted-jud-mcts]] backup semantics — two legal forms replace partner-max/opponent-min; actor-relative node identity; J3 gated on calibrated likelihoods. [[jud]] v2 per-move targets split into two consumer-distinct signals.
**Frontier:** Stage 0 closure (CUDA bench, exposure scan, two-block P0/C0) precedes lane grading; Lanes A/B primary.

## [2026-07-13 | research-night | Stage 0 closes; Lane A v0 validates; Lane B armed]

**Touched pages:** [[stage-0-closure]] [[world-sampler-mrv-audit]] [[partnership-wall-research]] [[champion]] [[auction-decoder]] [[auction-decoder-v0]] [[jud-target-granularity]] [[search-literature-transfer]] [[index]]
**Added:** [[stage-0-closure]] (all six arena arms in registered bands; CUDA correctness PASS + throughput-prediction MISS: sampler is launch-bound; exposure 2.51% distributional, 20/200-worst argmax flips), [[auction-decoder-v0]] (instrument validated, causal signature clean), [[jud-target-granularity]] (R1–R6 registered before evaluation).
**Updated:** audit + trail + champion pages forward-linked to the closure; exposure question moved to `questions/resolved.md`; literature citations verified against primary sources.
**Capability:** `--teacher-forced` E[Q] labeling (decision k = recorded play step k; 107,244/107,244 decision coordinates covered on the Lane B corpus).

## [2026-07-13 | research-night close | Lane B graded both rounds; night digest filed]

**Touched pages:** [[jud-target-granularity]] [[jud]] [[consumption-ledger]] [[the-wall]] [[partnership-wall-research]] [[dense-q-supervision]] [[research-lane-selection]] [[sources/research-night-2026-07-13]] [[index]]
**Verdict:** per-move targets at v1 capacity are marks-null in both forms (parent-side aux; child-state values); 3× volume moves calibration only; ranking-label agreement does not order play strength. The only registered prediction that hit was the one predicting a null.
**Residual:** capacity×target interaction, on-policy loop data, opponents-in-rollout; a never-significant ~+0.18 search-side trace for CE-lowering leaves.
**Digest:** [[sources/research-night-2026-07-13]] carries the whole night (Stage 0 closure, Lane A validation, Lane B negative, new capabilities).

## [2026-07-13 | wiki-reorg | schema v2, routing trails, status truth, split catalog]

**Schema:** [[AGENTS.md|AGENTS]] amended — 4-value status enum with lifecycle rules, date timestamps, kind decision tree, one-home-per-fact, routing rules, named sources codified; `scripts/wiki_lint.py` enforces mechanically (`--strict` clean at this entry).
**Added:** [[trails/gus-line|gus-line]] [[trails/burl-line|burl-line]] [[trails/champion-ladder|champion-ladder]] [[entities/stage-0-adapter-line|stage-0-adapter-line]] [[entities/burl-adapter-line|burl-adapter-line]] [[decisions/beads-to-gh-issues|beads-to-gh-issues]] [[playbooks/research-night|research-night]]; catalog split into `index-<kind>.md`.
**Updated:** hubs thinned to budget (gus 726→98, burl 586→116, champion/jud/arena/w42); ~120 stale statuses flipped; w42 rollups canonicalized on [[experiments/w42-book-claim-synthesis-and-ai-directions|w42-book-claim-synthesis-and-ai-directions]]; 84-claim + reentry verdicts reconciled; 17 pivot-dead questions moved to resolved.
**Moved:** 16 winning42 chapter digests → `sources/`; 7 era chronicles → `trails/`; 10 adapter receipts → `experiments/`; selfplay-arena → [[entities/burl-selfplay-arena|burl-selfplay-arena]].
**Record:** `docs/wiki-reorg-proposal-2026-07.md` (the adopted proposal; audit evidence in its appendix).

## [2026-07-13 | champion-fold | champion folded into jud — aspirational name retired]

**Decision:** "champion" named the player before it existed; the built thing is [[jud]]. Asset map, decision loop, and auction-dominance analysis moved to [[entities/jud|jud]]; teaching half to [[topics/the-wall|the-wall]] as a declared side benefit; [[entities/champion|champion]] reduced to a superseded pointer.
**Sweep:** 74 live pages retargeted by meaning (player → jud; era/rungs → [[trails/champion-ladder|champion-ladder]]; goal → the-wall). GitHub milestone **Champion** and repo dir `champion/` keep the name (BUILT); CLAUDE.md frontier line updated.

## [2026-07-14 | 45fe646b | count-fate ledger — the consumption object formulated]

**Touched pages:** [[count-fate-ledger]] [[the-wall]] [[past-belief-future-direction]] [[w42-phase2-hidden-domino-threat-attribution]]
**Added:** [[count-fate-ledger]] — hand value as a belief-weighted ledger of count-fate scenarios (IDEATED, conversation 2026-07-13→14, issue #49); guards/walkers as one junk-retention economy per-world E[Q] cannot price ([[strategy-fusion]]).
**Updated:** [[the-wall]] contextual-distribution direction now names its consumption object; threat-attribution grain framed as one factor of row probability.
**Questions opened:** tied-strategy rollouts pricing guard/walker retention (questions/open.md, issue #49).

## [2026-07-15 | conversation | the argument's referees — Phase 2 grading doctrine ratified]

**Touched pages:** [[count-fate-ledger]] [[strategy-fusion]] [[otis]]
**Added:** "The argument's referees" on [[count-fate-ledger]] — the outcome leak named ("would it have worked *more often*", belief-averaged tied grading), dispersion-triaged lesson harvesting, the three-referee split, the claim-vs-cash gap as the loop's convergence metric; "eq is not 42" coda on [[strategy-fusion]].
**Updated:** [[otis]] design commitments route to the doctrine; amendments filed as a comment on issue #55.

## [2026-07-15 | b347897b | otis v0: count-fate ledger built and graded overnight (#49)]

**Touched pages:** [[otis]] [[otis-v0]] [[count-fate-ledger]] [[world-sampler-mrv-audit]] [[jud]] `questions/open.md`
**Added:** [[otis]] [[otis-v0]] — the fate-ledger-native player, all seven registered predictions graded on the branch `worktree-otis-v0` night
**Updated:** [[count-fate-ledger]] (new Measured section; open question narrowed), [[world-sampler-mrv-audit]] (corpus-scale contamination quantified, issue #52), [[jud]] (sibling link)
**Questions opened:** issues #51 (doubles-suit engine representability), #52 (corpus regeneration on repaired sampler), #53 (retention-policy consumer — the remaining half of #49's question)

## [2026-07-16 | conversation | the belief/policy/value algebra promoted; measurement program filed]

**Touched pages:** [[belief-policy-value-algebra]] [[strategy-fusion]] [[count-fate-ledger]] [[index-topics|index]]
**Added:** [[belief-policy-value-algebra]] — tilt form (b = u·e^g; eq is the g≡0 limit), coupling theorem, information identity, eq located as "π deleted twice," exact gap and claim-vs-cash decompositions, the (ε, init) family; conclusions tiered CAN (mathematical) vs MIGHT (conjectures, each paired with its deciding probe).
**Questions opened:** [issue #64](https://github.com/jasonyandell/mk5-main/issues/64) — the M1–M6 measurement program (meaning map, tilt profile, channel bandwidth, accidental-convention detector, realized tiger, field docility).
