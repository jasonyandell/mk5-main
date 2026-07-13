---
title: STaR — Self-Taught Reasoner
kind: topic
first_seen: 2026-04-09
last_updated: 2026-04-26
status: superseded
---

## Current frontier

STaR carried forward from [[lem]] into [[burl]] (see "STaR on Burl's trajectories" below)
and was the active training mechanism through the Burl replay's 2000-decision harvest and
[[preserve-thoughts]] work. It is not, however, the path to the strongest player at this
frontier: the [[champion]] / [[gus]] / jud pure-value-net line that superseded both LEM and
Burl has no LLM-agent step at all — no model to STaR-train. STaR remains live tooling for
the legibility/pedagogy track (narrating and explaining play), not for the strength track.
See [[champion]] ("oracle → champion → gus → burl → lem" pedagogy chain) and
[[lem-to-burl-handoff]].

## Overview

STaR (Self-Taught Reasoner) is the training paradigm [[lem]] uses in Stage 1. The procedure: generate a reasoning trace from the model; evaluate it against [[k1-grading]]; if it passes, keep the trace as training data; if it fails with a legal action, reveal the correct action and ask the model to rationalize it ([[r1-rationalization]]), then train on that rationalization instead (lem/OVERVIEW.md @ a8bccfa).

## Role in LEM

STaR is the path from rules-comprehension (achieved by the [[rules-adapter]]) into play-decisions. Stage 1 applies STaR to trick-6 decisions, where the lookahead horizon is one ply and E[Q] labels are cleanest. The [[backwards-curriculum]] extends STaR to earlier tricks in later stages (lem/OVERVIEW.md @ a8bccfa).

## Implementation

The STaR harness exists in three variants (see [[star-harness]]):

- **Modal harness** (`star_harness.py`, A10G) — original batch function; tested end-to-end on 5 examples.
- **Local runner** — runs Gemma 4 E2B via llama.cpp (CPU, Q4_K_M GGUF) with streaming thinking-channel output for interactive debugging. Reuses `parse_play`/`grade_k1` from the harness.
- **Single-GPU loop** (`star_loop.py`) — one [[modal]] function, one GPU, model loaded once. All phases run sequentially on H100.

The inference step uses HF `model.generate()` in batch mode. vLLM has been tried and abandoned twice:

1. **8724e93** — removed due to a version conflict between vLLM's pinned transformers and Gemma 4's tokenizer format. See [[sources/8724e93]].
2. **2c2b851** — re-attempted on B200 with vLLM 0.19.0; abandoned 80 min later in **26f5ddf**. Root cause: `Gemma4ForConditionalGeneration`'s multimodal weight layout is incompatible with LoRA in vLLM 0.19.0 — merged LoRA checkpoints cannot be loaded. See [[sources/2c2b851]] and [[sources/26f5ddf]].

The settled recipe (26f5ddf): HF `model.generate()` with three fixes — `attn_implementation="sdpa"` (PyTorch native flash attention), `torch.compile(model, mode="reduce-overhead")`, and left-padded batching (all prompts in one `generate()` call). The model is loaded once and reused for both the inference pass and the [[r1-rationalization]] pass; stopped only for the LoRA training phase. Smoke test on B200: 120 tok/s, full 5-example iteration in 151s.

## Concrete 7-step flow

1. **Inference** — HF `model.generate()` batch on narration prompts (with optional LoRA adapter).
2. **Parse** — extract the model's chosen action from the thinking-channel output.
3. **K1 grade** — classify each trace as `pass`, `fail`, `illegal`, or `parse_fail`.
4. **Pass** → keep the full response as a training trace.
5. **Illegal or parse_fail** → DISCARD. Traces arriving at impossible states are poison — the reasoning chain is corrupted even if intermediate steps looked reasonable. See [[decisions/discard-illegal-traces]]. (Prior to fb47ab3 these were rationalized; that behavior is superseded.)
6. **Fail** → [[r1-rationalization]]: reveal the correct action, ask the model to justify it, keep that rationalization as a training trace.
7. **LoRA step** on the collected traces → push adapter to HuggingFace → repeat.

(lem/gemma_star/star_harness.py @ 576b694; lem/gemma_star/star_loop.py @ 576b694)

## Diagnostics

`illegal_rate` = (illegal + parse_fail) / total is logged to wandb each iteration. It is a rules-comprehension diagnostic, not a training objective: high (~40%) means Stage 0 needs more work; low (~5%) means the remaining signal is strategy (fb47ab3).

## Stage 1 iteration 0

First full end-to-end STaR iteration (H100, Stage 0 adapter, 10 examples, ~15 min, ~$1):

- 30% pass (3/10), 30% fail (3/10), 40% illegal (4/10)
- 6 training traces generated (3 wins + 3 rationalizations); 4 illegal traces discarded
- LoRA trained in 15s, loss = 0.11
- Adapter pushed: `jasonyandell/gemma-4-e2b-texas42-star-iter0`
- Wandb: `jasonyandell-forge42/lem-star`

The 40% illegal rate confirms the diagnostic prediction: Stage 0 rules knowledge is still incomplete. The rate is expected to drop across iterations as the model internalizes rules through practice (see [[learned-by-playing]]). See [[experiments/star-iter-0]] for the full run writeup (lem/OVERVIEW.md @ 576b694).

## First measurements

Base-model K1 baseline (10 examples, no adapter, local runner, llama.cpp): 60% pass rate — 60% pass, 30% fail, 10% illegal, 0% parse fail. See [[experiments/base-model-k1-baseline]] (f578bfa).

## Data flow

1. Narrate a game from a narrator's seat (see [[narration]]).
2. Truncate the narration at the narrator's trick-6 decision.
3. Collect a thinking trace from the [[rules-adapter]] checkpoint via Gemma's native thinking channel.
4. Grade the trace via [[k1-grading]].
5. Keep passes; rationalize legal fails; discard illegals and parse failures.
6. LoRA step on the kept corpus.
7. Iterate until [[expected-q-value]] delta plateaus.

## 10 iterations result

10 full STaR iterations completed on B200, chained as separate [[modal]] runs (one iteration each) to avoid container staleness. Total cost: ~$15. Wandb: `jasonyandell-forge42/lem-star` (lem/OVERVIEW.md @ efad16e).

| Iter | Pass | Loss | Pool |
|---|---|---|---|
| 0 | 30% | 31.6 | 3148 |
| 1 | 34% | 17.5 | 3148 |
| 2 | 33% | 21.2 | 3148 |
| 3 | 36% | 15.5 | 3148 |
| 4 | 35% | 19.7 | 3148 |
| 5 | **42%** | 12.8 | 7409 |
| 6 | 36% | 12.3 | 7409 |
| 7 | **42%** | 12.9 | 7409 |
| 8 | 38% | 11.4 | 7409 |
| 9 | 36% | 11.8 | 7409 |

Data pool expanded at iter 5 from 3148 (seeds 0–199) to 7409 examples (seeds 200–499 added via ff0d0d2); a different random subset is drawn each iteration for diversity.

Pass rate: 30% → plateau at 36–42%. Best: 42% at iters 5 and 7, reproducible but not durable across iterations. Loss: 31.6 → 11.8. 10 adapters on HuggingFace: `star-iter0` through `star-iter9`.

The plateau at 36–42% average suggests the current signal is saturating. See [[experiments/star-10-iterations]] for the full table (lem/OVERVIEW.md @ efad16e).

**Iterations 10–14 (908773a):** 39%, 41%, 40%, 39%, 38%. Loss stable at ~10. Plateau confirmed — 5 additional iterations on the 7409 pool did not break 42%. Total: 15 adapters on HuggingFace (`star-iter0` through `star-iter14`), total B200 cost ~$25.

Also generated 4263 more narrations (seeds 500–799); total pool now 11,672 examples available for larger-subset iterations.

> "The plateau at ~40% likely reflects the ceiling of K1 grading without fact-verification. The model may be learning wrong game-facts that happen to produce correct plays ~40% of the time but can't go further because the reasoning is polluted." (lem/OVERVIEW.md @ 908773a)

The proposed remediation: bootstrap the scratchpad format via SFT first, then resume [[scratchpad-validation]]. See [[experiments/star-10-iterations]] (the experiment page tracks all 15 iterations).

## Kerry STaR + v3 STaR

**Kerry STaR** (3 iterations, a2498e4, starting from [[kerry-curriculum]] adapter + enriched narrations with state blocks):

| Iter | Pass | Illegal |
|---|---|---|
| 0 | 44% | 13% |
| 1 | 42% | 10.5% |
| 2 | 44% | 13% |

vs. v1 baseline iter-0: 30% pass / 33% illegal. Kerry's curriculum + public state block = floor above the old ceiling. Illegal rate cut from 33% to ~12–13% (a2498e4).

**v3 STaR** (5 iterations, 8c1bb14, starting from [[trump-drilling]] + [[kerry-curriculum]] adapter):

| Iter | Pass | Illegal |
|---|---|---|
| 0 | 44% | ~13% |
| 1 | 42% | ~13% |
| 2 | **48%** | ~13% |
| 3 | 47% | ~13% |
| 4 | 38% | ~13% |

Peak 48% at iter-2 is a new high water mark (previously 42% on v1, 46% on Kerry). Best adapter: `star-iter2`. See [[experiments/stage-0-progression-star]] (8c1bb14).

**Revised ceiling framing:** The "~40% K1-without-fact-verification ceiling" reading from ingest 10 (908773a) was premature. v3 shows Stage-0-quality improvements push through it — the 42% ceiling broke at 48%. The revised hypothesis: K1 has a ceiling that depends on the rules-comprehension floor provided by Stage 0. Better Stage 0 → higher STaR plateau. [[scratchpad-validation]] may still be needed eventually but is not proven necessary yet (8c1bb14).

## STaR on Burl's trajectories

[[burl]] runs STaR on a different trace format than LEM did. LEM traces are free-form reasoning text ending in a domino choice. Burl traces are sequences of `(reasoning, tool_call, tool_response)` turns ending in `commit_play`. The tool-call history is part of the training signal — the adapter learns not just which play won but which sequence of tool-asks and reasoning steps produced it (burl/OVERVIEW.md @ 8d26e0d).

**Phase 2 iter-0 corpus** (fd6032b): 50 decisions rolled out with the Layer-1 prompt (rules primer + 42-aware framing). 27 K1 wins + 23 hinted rationalizations, K1 rate 54%. Total cost $0.91.

**Suspicious rationalization pattern:** All 23/23 rationalizations converged to the ground-truth play on the first hinted re-prompt. 100% convergence suggests Gemma is behaving as a structured formatter given the answer rather than as a second-chance reasoner. These rationalizations may be cheaper to produce but less signal-rich than LEM's [[r1-rationalization]] traces, which involved the model reasoning backward from the correct action (fd6032b).

**Phase 3 LoRA** (0168210): Trained on 50-entry corpus, B200, ~3 min, $0.60. Loss 53 → 4.5, token accuracy 3.5% → 29.5%. Clean descent.

**Phase 4 eval — regression** (789e14d):

| Metric | Spike v2 (base) | Layer 1 | iter-0 |
|---|---|---|---|
| Bot-match | 88.9% | 70% | 60% |
| K1 | 88.9% | 70% | 60% |
| `eq_outcome_distribution` calls / 10 decisions | 15 | 2 | 2 |
| Legal rate | 100% | 100% | 100% |

iter-0 did not improve on Layer 1; it reproduced Layer 1's pathology in the adapter weights. Root cause: Phase 2's K1-win corpus was harvested from the primer+framing base, which was already `eq_outcome_distribution`-shy and `is_legal`-heavy. STaR baked in those habits rather than correcting them (789e14d).

**Implication:** STaR on Burl needs a corpus where the WINNING traces are tool-use-broad. If the rollout policy itself is tool-use-narrow, the K1 wins will be narrow too — and STaR amplifies what wins. Planned fix: strip the primer, keep the framing block, re-harvest with the lighter prompt (789e14d). See [[tool-orchestration]] "Primer trade-off" section.

### Burl 2000-decision corpus ready (2026-04-25, post-063fcac)

The corpus that the planned re-harvest produces is now in hand. See [[burl-2000-harvest]]. 2000 decisions of [[burl]] on the `D_required_first` variant of [[wax-museum]] — light prompt, [[belief-trajectory]] required first, post-chat-template-fix tool responses visible. Bucket-classified against K=200 belief-sampled E[Q] (see [[lamir1]]):

- **Strict pool: 1062 rows** (Burl matched the oracle). Of those, 202 are non-trivial (excluding `ALL_AGREE_CORRECT`, where π already had the right answer).
- **Sharpest [[r1-rationalization]] target: 299 rows** of `BURL_BREAKS_CONSENSUS` — π and Q-mean both agreed on the oracle's answer; Burl alone deviated.
- 219 forced-commit rows (excluded from regret math), 0 illegal.

Recipe lessons from prior Burl STaR runs (iter-0 through iter-5 + the 71-row collapse):

1. **No tiny corpora.** Filter-only on 71 rows collapsed to loss 0.10 in 100 iterations and learned to memorize templated tail content (0/3 eval). 1062 strict-pool rows is the minimum scale that the next attempt should not collapse on.
2. **Strip memorizable rows.** `--min-assistant-chars 300` filter on `build_filtered_corpus.py` strips ~40% of templated short rows uniformly across all buckets — they were row-decomposition noise (bare `<|tool_call>...<tool_call|>` strings around 50 chars), not signal. Zero gold-bucket decisions are lost at the 300 cutoff.
3. **Conservative hyperparams.** rank=8 (not 16), lr=3e-5 (not 1e-4), 1 epoch, val-loss + early-stopping. The val-loss + early-stop wiring already lives in `burl/train/star_mlx.py` from the prior session.
4. **Trust the manifest, not the prose.** Before training on a "pre-existing" corpus, `jq .harvest_dir manifest.json` and confirm the source. Recipe documents go stale; manifests don't lie. See [[burl-2000-harvest]] "Footgun caught (2026-04-25)".
5. **Trust the argparse, not the prose.** The run-3 plan §Step 2 named three flags wrong (`--train-corpus` vs actual `--corpus`; `--out` vs `--adapter-out`; `--early-stop-val-rise 0.02` vs the multiplicative 1.02). Caught against `_build_arg_parser` in `burl/train/star_mlx.py`. Same generalization as #4: documentation drifts; the source is the ground truth.
6. **Detach the trainer; resumable checkpoints; catch-all snapshot save. (Resolved.)** Run-3's working attempt was killed at iter 487/1343 (val 0.302, healthy curve) by an MLX `RuntimeError: [metal::malloc] Resource limit (499000) exceeded` in the cosine LR scheduler. The `train_mlx` exception handler only caught the local `EarlyStopRequested`, so the in-memory best snapshot was lost. Postmortem: [[burl-star-run3]]. **Now landed:** `burl/train/star_mlx.py` writes an on-disk adapter every `--steps-per-checkpoint N` (default 100) iters, accepts `--resume` to pick up from `checkpoint_state.json`, and a generic-exception handler in `train_mlx` serializes the in-memory best to `best_on_crash/adapters.safetensors` before re-raising. macOS detachment uses `nohup ... </dev/null >/dev/null 2>&1 & disown` (no `setsid` on the OS). Policy + recipe: [[resumable-checkpointing]].
7. **`--preserve-thoughts` is load-bearing; it is not a tuning knob.** Run-3b (default = preserve-thoughts OFF) trained successfully — best val 0.238, adapter on disk, eval ran — but emitted a thought block on **0/113** decisions because Gemma 4's chat template ran `strip_thinking()` before tokenization, producing rows where only `(user, tool-calls, commit)` reached the loss. The model dutifully learned the visible target: tool-call patterns without reasoning. Run-3c with `--preserve-thoughts` ON (only delta) recovered thought-block emission to **537/560 = 95.9%** at n=560 with bot-match 66.6% (vs run-3b's 60.2%) and mean |Δ| 2.11 (vs 2.89). This is a phase change on the thought-emission axis, not a knob — without preserve-thoughts a Burl LoRA trains reasoning *out* of the model. Default it ON for any [[burl]] STaR or [[r1-rationalization]] run. **Caveat**: bot-match alone is not sufficient evidence of "playing well" — run-3c's mean signed Δ is −1.98 (the adapter loses 7.5:1 to the bot on the 187 disagreements). The play-quality verdict needs same-harness base-eval + signed-delta + oracle regret; see [[burl-star-run3]] §"What's next" for the eval-side gaps that block the next-iteration decision. See [[preserve-thoughts]] for the full A/B.

The harvest scaffold's per-wave resilience layer (see [[batched-harvest-resilience]]) is what made the 5h 46m run survive without intervention; future iteration on prompt variants or scaled corpora is now a weekly cadence rather than quarterly.

## Scratchpad validation attempt (retired at this frontier)

On 2026-04-11, over a 14-minute window, the project tried a stricter grading strategy: require the model to fill a structured HAND/VOIDS/COUNTS/PLAY scratchpad before choosing a play, and only train on traces where each claim is engine-verified correct. The first iteration produced a 64.5% `invalid` rate and only 5 trainable traces. The model hadn't been exposed to the scratchpad format, so it couldn't produce one reliably — format-bootstrapping must come before fact-validation. The experiment was reverted to simple K1 grading in 78ba940. See [[scratchpad-validation]] for the full account.

At this frontier the project continues with simple K1 + random-subset sampling per iteration, tracking the 34% → 40% pass-rate trend from earlier iterations.

## Links

[[k1-grading]] [[r1-rationalization]] [[lem]] [[burl]] [[tool-orchestration]] [[backwards-curriculum]] [[rules-adapter]] [[narration]] [[expected-q-value]] [[star-harness]] [[modal]] [[experiments/base-model-k1-baseline]] [[experiments/star-iter-0]] [[experiments/star-10-iterations]] [[experiments/stage-0-progression-star]] [[decisions/discard-illegal-traces]] [[decisions/resumable-checkpointing]] [[learned-by-playing]] [[scratchpad-validation]] [[kerry-curriculum]] [[trump-drilling]] [[sources/8724e93]] [[sources/2c2b851]] [[sources/26f5ddf]]
