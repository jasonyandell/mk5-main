---
title: STaR Harness
kind: entity
first_seen: 2026-04-10
last_updated: 2026-04-11
status: complete
---

## What it is

The STaR Harness is the code artifact implementing [[lem]]'s Stage 1 self-taught reasoning
loop. It runs [[gemma-4-e2b]] on narration prompts, grades each action via [[k1-grading]],
rationalizes failures via [[r1-rationalization]], and produces a JSONL training set for the
next LoRA iteration. Three variants exist at this frontier. (commit messages @ 7538016, f578bfa, 8c5fbca)

## Variants

### `star_harness.py` — Modal batch function (7538016)

The primary Stage 1 entry point. Runs on [[modal]] (A10G GPU). Six-step flow:

1. Load Gemma (with optional [[stage-0-adapter]] via `--adapter`)
2. Run inference on narration prompts; emit thinking-channel output
3. Parse the model's action from thinking channel (`parse_play`)
4. Grade K1: `E[Q][gemma] >= E[Q][bot]` (`grade_k1`)
5. On pass: keep full response as a training trace
6. On fail or illegal: rationalize — reveal bot's correct action, prompt model to justify it,
   keep that rationale as the trace
7. Output JSONL training set for the next LoRA step

Smoke tested on 5 examples: 1 K1 pass (20%), 2 failures rationalized, 2 illegals
rationalized. All traces collected. Pipeline works end-to-end.
See [[experiments/star-harness-5ex-smoke]]. (commit message @ 7538016)

### `star_local.py` — local llama.cpp runner (f578bfa)

Runs Gemma 4 E2B locally via llama.cpp (CPU, Q4_K_M GGUF) with streaming thinking-channel
output. Reuses `parse_play` and `grade_k1` from `star_harness.py`. Intended for interactive
debugging, not production iteration. (commit message @ f578bfa)

### `star_loop.py` — single-GPU continuous loop (8c5fbca, updated 68a0416–d913932)

One [[modal]] function, one GPU, model loaded once. Full loop per iteration:

1. HF `model.generate()` batch inference (tokenize all prompts, pad, forward, decode)
2. K1 grading
3. HF batch rationalization of legal failures only (see Grading policy below)
4. LoRA training on winning traces plus rationalizations
5. Push adapter to HuggingFace
6. Repeat

Actual throughput: ~15 min/iteration on H100 (iter 0); 151s for a 5-example iteration on
B200 with the SDPA recipe (see Performance recipe below). Accompanied by `train_star.py`
(standalone STaR training) and `iterate.sh` (manual loop).
(commit messages @ 8c5fbca, 576b694, 26f5ddf)

**Second vLLM attempt and final abandonment (2c2b851 → 26f5ddf)**: vLLM was re-introduced
for B200 (2c2b851, expected 71s/prompt → ~5s/prompt with 16-way concurrency). Reverted 80
minutes later (26f5ddf): `Gemma4ForConditionalGeneration`'s multimodal weight layout is
incompatible with vLLM 0.19.0's LoRA support. Merged adapter checkpoints cannot be loaded.
HF `model.generate()` is retained as the permanent inference backend.

**vLLM removed from H100 path (68a0416–d913932)**: vLLM bundles its own `transformers`, which conflicts
with the `Gemma4ClippableLinear` monkey-patch required by PEFT. After an attempt to make
the patch conditional (68a0416) failed to resolve the clash, vLLM was replaced entirely
with HF `model.generate()` (8724e93), and remaining `SamplingParams` references removed
(d913932). The HF path also eliminates the ~5-minute vLLM engine cold-start per iteration.
The `Gemma4ClippableLinear` patch is now applied unconditionally throughout.

**Module mount note (fb47ab3)**: `parse_play` and `grade_k1` are inlined directly into
`star_loop.py` and `star_harness.py` rather than imported from a shared module. Reason:
the `lem` package was not reliably mounting inside [[modal]] containers at runtime.

## Chained Modal runs (as of efad16e)

Iterations are chained as separate [[modal]] function invocations rather than a single
long-running session. Rationale: avoids container staleness across iterations, limits the
blast radius of any failing iteration to a single run, and makes each adapter independently
inspectable on HuggingFace.

**Throughput on B200 (confirmed over 10 iterations):**
- LoRA training step: ~13s
- Full 5-example iteration: ~2.5 min
- Cost: ~$0.26/iteration; 10 iterations total cost ~$15

(lem/OVERVIEW.md @ efad16e)

## Scratchpad validation (retained, not active)

Scratchpad validation was introduced and reverted within a 14-minute window (380f3fa →
78ba940, 2026-04-11). The code remains in `star_loop.py` for later use.

**What it was**: the model was required to fill a structured scratchpad (HAND / VOIDS /
COUNTS / PLAY sections) before deciding. Each claim was validated against engine ground
truth. New grading categories: `valid_pass` (keep), `valid_fail` (rationalize), `invalid`
(discard — wrong facts), `illegal` (discard), `parse_fail` (discard). Only traces with
correct facts were trained on, preventing the model from learning hallucinated game-facts
that happened to produce correct moves.

**Why it was turned off**: 64.5% of traces were `invalid` on the first iteration — yielding
only 5 trainable traces. The model had never seen the scratchpad format; validating against
a format the model hasn't learned yet produces mostly noise. Relaxing to hand-only
validation (b12fcec) didn't resolve the issue, so the entire validation layer was disabled
(78ba940) and simple K1 grading restored. Insight: format bootstrapping must come before
format validation. See [[topics/scratchpad-validation]]. (commit messages @ 380f3fa, b12fcec, 78ba940)

**Narration v2**: a new narration format with ground-truth fields (HAND / VOIDS / COUNTS)
was generated alongside the scratchpad feature. Current production runs (`iterate.sh`) use
v1 narrations; v2 is available but not active as of 5946c94.

**Per-iteration random-subset sampling** (`--subset N`): introduced alongside scratchpad
validation and retained after the revert. Ensures trace diversity across iterations by
sampling a different random subset of examples each time.

## Performance recipe (B200, as of 26f5ddf)

Three changes unlock 120 tok/s batch inference for [[gemma-4-e2b]] on B200:

1. `attn_implementation="sdpa"` — PyTorch native flash attention; no dependency on vLLM or
   custom kernels.
2. `torch.compile(model, mode="reduce-overhead")` — applied once after model load.
3. **Left-padded batching** — all prompts tokenized and padded to equal length, then passed
   in a single `model.generate()` call.

Result: 120 tok/s; 5-example iteration in 151s. The model is loaded once per Modal function
invocation and reused across both the inference phase and the R1 rationalization phase.
It is stopped only for LoRA training. (commit message @ 26f5ddf)

## Grading policy (as of fb47ab3)

Three-branch outcome for each graded example:

| Grade | Condition | Action |
|---|---|---|
| `pass` | `E[Q][gemma] >= E[Q][bot]` | Keep trace as-is |
| `fail` | Legal move, `E[Q][gemma] < E[Q][bot]` | Rationalize with [[r1-rationalization]] |
| `illegal` / `parse_fail` | Illegal move or unparseable output | **Discard** — see [[decisions/discard-illegal-traces]] |

Illegal and parse-fail traces are discarded rather than rationalized. Rationale: reasoning
chains that arrive at an impossible game state are corrupted throughout, even if intermediate
steps appeared coherent. Training on them would poison the model. (commit message @ fb47ab3)

`illegal_rate` (combined `illegal + parse_fail` / total) is now a tracked wandb metric
alongside `pass_rate`. Interpretation per [[k1-grading]]:
- `illegal_rate` ~40% → Stage 0 needs more rules work
- `illegal_rate` ~5% → model knows rules; focus on strategy

## K1 and R1 definitions

- **[[k1-grading]]**: keep a trace iff `E[Q][gemma] >= E[Q][bot]`. The bot plays E[Q]-greedy,
  so K1 reduces to "Gemma picked an argmax-tied action." Baseline (base model, no adapter,
  10 examples): 60% K1 pass. See [[experiments/base-model-k1-baseline]].
- **[[r1-rationalization]]**: on legal K1 failure, reveal the bot's action and ask Gemma
  to justify why it is correct. Keep that justification as the training trace. Illegal and
  parse-fail traces are discarded, not rationalized (as of fb47ab3).
  Simpler than DPO; proven effective in the original STaR paper.

## End state

The harness was frozen at efad16e (10-iteration run) with no `lem/gemma_star/` commits
after that point. [[burl]] does not reuse this code — it runs [[star]] on a different
trace format (tool-call trajectories, not free-form narration text). See
[[lem-to-burl-handoff]].
