---
title: Gemma 4 E2B
kind: entity
first_seen: a8bccfa
last_updated: dbadb5f
status: retired-as-lem-base
---

## What it is

Gemma 4 E2B (`google/gemma-4-E2B-it`) is the base model chosen for [[lem]]. It is a
phone-class open model with approximately 2.3B effective parameters, a 128K context window,
a native thinking channel, and an Apache 2.0 license. At the start of the LEM experiment
the model has never encountered Texas 42. (lem/OVERVIEW.md @ a8bccfa)

## Why it was chosen

| Property | Value |
|---|---|
| Parameter count | ~2.3B effective |
| Context | 128K tokens |
| Thinking channel | Native (enabled by default in inference) |
| License | Apache 2.0 |
| Inference class | Phone-class (runnable on consumer hardware) |
| Fine-tune tooling | [[lora-unsloth]] has a ready Gemma 4 recipe |

The thinking channel allows LEM's [[star]] harness to collect reasoning traces without
post-hoc extraction. The Apache 2.0 license permits redistribution of fine-tuned adapters.
(lem/OVERVIEW.md @ a8bccfa)

## Inference setup

At this frontier, inference runs on Modal (L4 GPU, fp16, `transformers>=4.52`,
`max_tokens=2048`, `temperature=0.6`). The model weights are cached in a Modal volume.
The processor requires PIL even for text-only inputs because Gemma 4 E2B is multimodal.
(lem/gemma_star/modal_app.py @ a8bccfa)

## First contact

See [[first-gemma-contact]]. In its first run against a narrated [[texas-42]] game (seed 42,
fives trump, Player 3 narrator), the model:

- Correctly identified trump suit, led suit, void status, and score math.
- Confused the initial dealt hand with the remaining hand (dominoes already played were
  still treated as available).
- Misclassified 6-4 as trump under fives-trump (6-4 is a count domino but is only trump
  when fours or sixes are the trump suit, not fives).

Both errors are exactly the gaps [[rules-adapter]] Stage 0 is designed to close.
(commit message @ a8bccfa)

## Training constraints (discovered during Stage 0)

Three non-obvious constraints apply when fine-tuning Gemma 4 E2B with PEFT on L4 (22GB):

1. **`Gemma4ClippableLinear` monkey-patch required.** The model wraps its linear layers in
   `Gemma4ClippableLinear(nn.Module)` which PEFT cannot target. Patching it to inherit from
   `nn.Linear` instead restores LoRA targeting and gradient flow. Without the patch,
   `grad_norm=0` throughout training. Fix sourced from huggingface/peft#3129. The patch must
   be applied in both the training function and the inference loading path.
   (commit message @ 9571a7b)

2. **bf16 required; fp16 breaks.** `GradScaler` (used with fp16) fails when gradient
   checkpointing is enabled. Switch to bf16 eliminates the failure.
   (commit message @ 9571a7b)

3. **Gradient checkpointing mandatory on L4.** The model is ~20GB in bf16, leaving ~2GB for
   activations on a 22GB card. `gradient_checkpointing=True` with `use_reentrant=False` is
   required. Eval forward pass OOMs regardless; disable eval entirely.
   (commit message @ df73c8d)

See [[stage-0-adapter]] for the training outcome and [[experiments/stage-0-v1-training]] for
the full run record. [[lora-unsloth]] is the fine-tune library used.

## K1 baseline (base model, no adapter)

Measured at commit f578bfa via local llama.cpp (Q4_K_M GGUF, CPU) on 10 trick-6 decisions
from eval seeds:

| Outcome | Rate |
|---|---|
| K1 pass (E[Q][gemma] >= E[Q][bot]) | 60% |
| Legal but suboptimal | 30% |
| Illegal move | 10% |
| Parse fail | 0% |

[[k1-grading]] here reduces to "picked an argmax-tied action" because the bot is E[Q]-greedy.
The 60% baseline is surprisingly high, suggesting many trick-6 decisions are near-unanimous
under E[Q] and the model stumbles mainly on hand-state tracking (the 10% illegal rate).
This is the number the [[stage-0-adapter]] + [[star]] iterations must beat.
See [[experiments/base-model-k1-baseline]]. (commit message @ f578bfa)

## Architecture quirk: KV-sharing (layers 15-34)

Gemma 4 E2B shares k_proj/v_proj weights across layers. Layers 15–34 reuse KV states
computed by layers 0–14 and therefore have no k_proj/v_proj parameters of their own.

This explains a PEFT warning seen during [[star-harness]] LoRA adapter merging:

> "Missing keys for `language_model.layers.{15..34}.self_attn.{k,v}_proj`"

The adapter is complete; those weight keys simply do not exist in the base model. When
targeting `q_proj, k_proj, v_proj` for LoRA, the adapter naturally covers only the layers
where those projections exist. No change to training is needed; no action beyond
understanding the warning. (lem/OVERVIEW.md @ efad16e)

## Inference quirks (vLLM incompatibility)

`Gemma4ForConditionalGeneration` uses a multimodal weight layout incompatible with merged
LoRA checkpoints in vLLM 0.19.0. Specifically, vLLM's LoRA support cannot handle the
multimodal weight structure, causing adapter loading to fail. vLLM is therefore not a viable
inference backend for adapter-tuned Gemma 4 E2B at this frontier.

The working alternative is HF `model.generate()` with three optimizations:

1. `attn_implementation="sdpa"` — PyTorch native flash attention
2. `torch.compile(model, mode="reduce-overhead")`
3. Left-padded batching — all prompts in one `generate()` call

This combination achieves 120 tok/s on B200, rendering vLLM unnecessary for LEM's batch
sizes. See [[star-harness]] for the full recipe. (commit message @ 26f5ddf)

## Second contact

See [[experiments/second-gemma-contact]]. After the [[stage-0-adapter]] was applied:
hand tracking was fixed (model correctly reads remaining hand, not initial hand) and the
final move was legal and correct. Trump membership errors (4-4, 6-4 called trump under
fives-trump) persisted. (lem/OVERVIEW.md @ 24ae55a)

## Third contact

See [[experiments/third-gemma-contact]]. After the [[kerry-adapter]] was applied: trump
non-membership correct (model correctly identifies that 6-2 and 6-1 are not trump under
fives); trump membership still partially wrong (6-4 still called trump under fives — same
stubborn error from first contact, now narrowed to one case); strategic reasoning depth
improved substantially (evaluates both candidate plays); final answer legal and correct.
(commit message @ 43009a4)

The [[v3-adapter]] extends Kerry with 5k targeted trump-drill Q&A (`is_trump`,
`list_trumps`, `which_trumps`, `trump_or_follow`, `count_trump`). Over 5 STaR iterations
it achieves a peak of 48% K1 pass — the highest recorded at this frontier.
See [[v3-adapter]]. (commit message @ 8c1bb14)

The [[v4-adapter]] (game-context Q&A, 31,830 examples, flexible grader, thinking disabled):
overall 67% comprehension. `is_trump` 100% — the 6-4-under-fives error that persisted
through all prior adapters is fully resolved. The model no longer hallucinates playing
Bridge. `what_beats` at 15% is the remaining weak spot.
(commit messages @ 3c33e86, 2f11f32)

## Retired as base model (2026-04-16)

Per commit 3465e29, Gemma 4 E2B is no longer the LEM base model. Replaced by [[qwen3-1.7b]].
See [[decisions/base-model-pivot-qwen]].

Architectural reasons for retirement:
- **PLE (parameter-embedding layer)** — architectural overhead absent in Qwen.
- **KV-sharing (layers 15–34)** — layers reuse KV states from 0–14, blocking
  LoRA-in-vLLM support (see Architecture quirk section above).
- **No flash-attention-2 support** — limits throughput on B200.
- **Comprehension eval v5**: 60% vs Qwen 3 1.7B's 100% on the same eval.

All Gemma-based adapters ([[stage-0-adapter]], [[kerry-adapter]], [[v3-adapter]],
[[v4-adapter]]) and the 15 STaR-iter adapters remain on HuggingFace as historical
checkpoints. The lessons learned on Gemma (KV-sharing, ClippableLinear patch, eval bugs,
thinking-mode behavior, curriculum progression) informed the Qwen pivot.
(commit message @ 3465e29)

## Burl-side findings (Moves 3-4, 2026-04-18/19)

**Move 3** (XML harness, base Gemma, 10 held-out decisions, zero fine-tuning): 100% legal,
60% bot-match, 70% K1. Premise survives. See [[experiments/burl-move3-base]].

Failure modes observed:
- Only `is_legal` called; distribution tools (`eq_outcome_distribution`, `conditional_outcome`)
  never reached — tool-use breadth is the Move 4 target.
- Hallucinates a fake `play` tool 80% of the time (harmless; commit still lands, but flags
  prompt ambiguity). Native format fixes this.
- `enable_thinking` is a no-op on Gemma 4's Jinja template; `max_tokens` budget is what
  actually controls thinking-channel consumption.

**Move 4 R3 spike** (native tool-use format, 9/10 decisions completed): 88.9% bot-match,
88.9% K1 (+28.9pp / +18.9pp vs XML). `eq_outcome_distribution` used 15× (was 0), no
hallucinated tools. Zero-shot XML emission was valid; native format is simply the better
path. See [[experiments/burl-move4-native-spike]] and [[decisions/native-tool-use-format]].
(commit messages @ 4b3ba3d, 3781dce)

**Phase 1 primer regression**: adding the 1549-word rules primer caused bot-match to drop
88.9% → 70% — primer occupies attention and suppresses `eq_outcome_distribution` calls
(15 → 2). 42-vocabulary in traces rose from 0 to 5-11 mentions/trace. This is the
substrate trade for STaR. See [[decisions/primer-tradeoff]].

**Phase 4 vLLM-LoRA blocker**: vLLM 0.19 rejects `Gemma4ForConditionalGeneration` for LoRA
inference. Fix: `hf_overrides` to `Gemma4ForCausalLM` at load time. Separate blocker from
LEM's ingest-7 vLLM attempt (which hit multimodal weight layout during adapter merge, not
inference load). (commit message @ 789e14d)

**iter-3-rules best result**: [[iter3-rules-adapter]] on Gemma 4 E2B with
`enable_rules_tools=True`, `enable_primer=off` reaches **90% bot-match, 0
retry-exhausted, 100% first-legal** — the best Burl result to date and an improvement over
spike v2's 88.9%. Validates rules-as-tools + native tool-use format together: `trick_winner_if`
usage increased after SFT, confirming tools replace memorization. (commit message @ dbadb5f)

## Chosen for Burl (different reasons than LEM)

Gemma 4 E2B is retired as LEM's base but is the chosen base for [[burl]]. The reasons are
different:

- **Comprehension matters less for Burl** — tools provide facts; Burl does not need to
  memorize the game. The 60% comprehension ceiling that disqualified Gemma from LEM is
  irrelevant.
- **Training volume is smaller** — tool-using trajectories are thousands, not tens of
  thousands of flashcards. Gemma's training-throughput issues matter less.
- **Gemma 4 was designed for agentic function-calling.** Native tool-use training is baked
  in; thinking mode is actively used during tool loops.
- **Agentic benchmarks favor Gemma 4 at the 2B scale.**

The architectural issues (PLE, KV-sharing, no FA2) that blocked LEM's training recipe
become lower priority when training on smaller trajectory corpora and relying on
inference-time behavior. Gemma 4 E2B is therefore simultaneously retired from LEM and
active in Burl. (burl/OVERVIEW.md @ 8d26e0d)
