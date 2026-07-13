---
title: Stage 0 v10 Adapter (joint rationalization + maskfix)
kind: experiment
first_seen: 2026-04-17
last_updated: 2026-07-13
status: complete
---

Receipt in the [[stage-0-adapter-line]] — its terminal adapter.

## What it is

The Stage 0 v10 Adapter is the current best LoRA fine-tune of [[qwen3-1.7b]] at the LEM
replay frontier. It exists in two forms that share the same training corpus but differ in
loss computation. (commit messages @ 0c7392f, be7efc4)

## v10 (original)

HF repo: `jasonyandell/qwen3-1.7b-texas42-stage0-v10`

Jointly trains [[v9-adapter]]'s 14-category comprehension corpus with 331 clean
rationalizations (scouted from v9, filtered by [[rationalization-verifier]], then
upweighted 10× in the training mix). The 331 examples come from 500 decisions scouted on
v9 (66% pass rate through the verifier).

Results:
- Comprehension: **83%** preserved
- Transition test (open-ended prompts): **55/100 bot-match**, 96/100 legal moves
- Visible reasoning, commits to plays
- Rationalization: does not improve over v9's ~68/100 ceiling on 1.7B

(commit message @ 0c7392f)

## v10-maskfix (current best)

HF repo: `jasonyandell/qwen3-1.7b-texas42-stage0-v10-maskfix`

Same 31,307-example corpus and hyperparameters as v10 original, but with completion-only
SFT loss enabled. See [[sft-completion-only-loss]].

**Root cause of fix**: TRL's `SFTConfig` defaults (`assistant_only_loss=False`) compute
loss over the full prompt+answer sequence. The ~50-token answer gradient was diluted ~9×
by ~400 template/prompt tokens the model had already memorized. Fix: switch dataset format
from `messages` → `prompt`/`completion`; TRL 1.2+ auto-enables completion-only loss.

Results vs v10 original:

| Metric | v10 | v10-maskfix |
|---|---|---|
| Comprehension overall | 83% | **86%** (= 14B v9 at 1/3 cost) |
| `intervention_check` | 70% | **88%** (beats 14B) |
| `partner_response` | 48% | **58%** |
| `beaters_in_unseen` | 46% | **56%** |
| Transition bot-match | 55/100 | 55/100 (unchanged) |

The unchanged bot-match confirms the transition gap is not a gradient-allocation problem.
Capacity or STaR iteration is the next lever, not more SFT signal.
(commit message @ be7efc4)

## End-of-LEM-replay state

v10-maskfix is the best LEM adapter at the close of the replay. Neither proposed lever
(capacity — compound 14B + joint training — or STaR on v10/14B) was taken; the 55/100
bot-match ceiling was never tested against them. How the line ended — LEM dormancy at
`be7efc4`, the mechanism pivot to [[burl]], and jud v1's zero-adapter endpoint — is on
[[stage-0-adapter-line]].
