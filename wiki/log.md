# Log

Chronological record of ingests. Append-only.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md` gives a clean timeline.

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
**Questions resolved:** none.

---

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
