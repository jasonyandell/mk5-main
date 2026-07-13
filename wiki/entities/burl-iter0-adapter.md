---
title: Burl iter-0 LoRA Adapter
kind: entity
first_seen: 2026-04-19
last_updated: 2026-04-19
status: superseded
---

## What it is

The Burl iter-0 adapter is the first LoRA fine-tune of [[gemma-4-e2b]] for [[burl]]'s
tool-using play task. Trained on a 50-entry STaR corpus from Phase 2. Private HF repo:
`jasonyandell/gemma-4-e2b-texas42-burl-iter0` (124 MB). (commit messages @ 0168210, 789e14d)

## Training provenance

- **Base model**: [[gemma-4-e2b]] (`google/gemma-4-E2B-it`), bf16, sdpa, ClippableLinear
  patched
- **Corpus**: 50 entries — 27 K1 wins (base Gemma got it right) + 23 hinted rationalizations
  (Gemma restating the ground-truth play via tool-call protocol). See
  [[experiments/burl-phase2-starcorpus]].
- **Method**: [[lora-unsloth]] SFTTrainer, rank 16, alpha 32, all attn+MLP, dropout 0.05
- **Hyperparameters**: 3 epochs, bs 2, grad_accum 4 (eff bs 8), LR 1e-4 cosine, warmup 0.1,
  gradient checkpointing; B200 (~3 min, $0.60)
- **Special tokens preserved**: `<|channel>`, `<|tool_call>`, `<|tool_response>`
- **Loss masking**: full sequence including user turn (same LEM default — could flip to
  `assistant_only_loss=True` for iter-1 per [[decisions/sft-completion-only-loss]])
- **Wandb**: `jasonyandell-forge42/burl-star/runs/vxqlnamb`

Training numbers: loss 53 → 4.5, token accuracy 3.5% → 29.5%, grad norm 45 → 0.88 (clean
descent, no NaN). (commit message @ 0168210)

## Phase 4 eval results (10-decision held-out)

See [[experiments/burl-iter0-eval]].

| Metric | Spike v2 | Layer 1 | **iter-0** |
|---|---|---|---|
| Bot-match | 88.9% | 70% | **60%** |
| K1 | 88.9% | 70% | **60%** |
| Mean E[Q] delta | −1.92 | −3.00 | −3.33 |
| `eq_outcome_distribution` calls | 15 | 2 | 2 |
| Legal rate | 100% | 100% | 100% |

Regressed 10pp from Layer 1 and 29pp from spike v2.

## Why it regressed

1. **Corpus shape mirrored Layer 1's pathology.** The 50 K1 wins were harvested from
   Layer 1 (primer+framing base), which was already eq-shy and `is_legal`-heavy. The
   adapter learned "Layer-1 Gemma" baked into weights — including its failure to call
   distribution tools.
2. **Primer is too long.** ~40K chars/decision of rules text displaced attention from
   decision-making. See [[decisions/primer-tradeoff]].
3. **Hint-and-format rationalization.** The 23 rationalizations converged on first pass —
   Gemma acting as a structured formatter given the answer, not a second-chance reasoner.
   The rationalizations teach the format but not the judgment.
4. **One catastrophic tail** (decision 6: dom 27 vs bot 9, Δ−24.81) accounts for most of
   the mean regression; without it, iter-0 tracks Layer 1.

Syntax is fine — tool-call format works. This is a judgment regression, not a format one.

## Superseded by

Iter-1 will re-harvest corpus on a trimmed-primer prompt, targeting recovery toward spike
v2's 88.9%. Iter-0 demonstrates a real signal: training on the corpus is possible and clean;
the problem is what's in the corpus.
