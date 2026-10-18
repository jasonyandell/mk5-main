---
title: "Gus learns exact Silas endgame answers with GRPO"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---

## Question and result

Can textbook RLVR improve [[gus-tiny-gpt]] on exactly checkable 42 endgames,
and does group-relative learning add anything beyond supervised correction?
Yes in this declared finite model. On 512 fresh physical deals, GRPO reduces
exact regret by 2.371 percentage points versus the frozen pretrained policy
(paired 95% interval 1.114–3.677 pp), and by 1.072 pp versus supervised correction
(0.359–1.828 pp). General playing strength is unestablished; pretraining
retention worsens. This is a useful restricted curriculum, not a replacement
for the full-game population experiment. Tracking issue: [#218](https://github.com/jasonyandell/mk5-main/issues/218).

The architecture, clean inputs and output contract come from
[[gus-splat-pretraining-2026-10-18]]. This experiment starts at its step-195000
checkpoint, SHA256 `f261e0f257047565cb481f88c5480e39e219d951173710c3dbf858b4b9cd563c`,
not the cooldown from [[gus-modern-learning-recovery-2026-10-18]].

## The exactly verifiable approximation

[[silas]] enumerates every remaining deal consistent with the acting player's
own tiles and public legal history. A question occurs during the penultimate
trick: the actor has two legal tiles and, after choosing, just one forced tile
remaining. There are at most 90 hidden allocations. Rules, legality, banked
points and complete support come from Silas; Python `Fraction` arithmetic
sums continuation outcomes without Monte Carlo noise or approximate ties.

The root belief is uniform over the legal-consistent fiber. It deliberately
ignores the behavioral likelihood of the preceding history. At the root, an
equal mixture chooses one shared table style for all three other seats:
lowest legal tile, highest legal tile, or uniformly random legal tile. Each
policy uses its own legal information. This is a small declared population,
not a realistic or equilibrium population. Bid 30 and pip declarations only.

For information state I and legal tile a:

`Q(I,a) = sum_world P(world|I) × sum_style (1/3) × P(viewer's team succeeds | world, a, style)`.

The candidate a is fixed before averaging worlds. There is no per-world
maximization; the viewer's later move is forced. Thus this construction
respects [[strategy-fusion-fence]]. Adding a later discretionary viewer choice
would require an information-state policy solver, not a max inside each world.
Root weights have Silas authority `exact`; continuation values have authority
`exact_given(model_id)`. Exact arithmetic certifies this specification, not
unknown human behavior or unrestricted optimal 42 play. This implements a
small tractable instance of the distinctions in [[two-recursions-and-transformers]].

The binary verifier accepts every exact maximizer: reward 1 iff
`Q(I,a) = max_b Q(I,b)`. It rejects other tiles. A one-tile answer is sufficient;
Gus is not being trained to emit a proof or chain of thought. The mathematical
reasoning is currently in the verifier. `lem/observer/rlvr/verifier.py` builds
the rational certificate; training reads that pinned answer key.

## How the textbook pieces fit

1. **Finite decision problem.** A contextual bandit is an RL problem with one
   rewarded decision per question. Restricting the horizon makes the verifier
   exhaustive and the credit assignment clear. This is a curriculum choice;
   it does not silently solve full sequential play. The underlying finite-RL
   vocabulary is that of [Sutton and Barto, contextual bandits, §2.8 in this author draft](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf).
2. **Pretraining.** The existing network already represents hands, contracts
   and public history. Fine-tuning starts with those weights. The original
   seven own tiles and public record remain the entire input; the legal mask
   selects outputs after the forward pass. Hidden worlds, values, answer keys
   and candidate lists are never input tokens.
3. **Verifiable reward.** Like answer checking in mathematical RLVR, the
   checker is executable and independent of a learned reward model. Correctness
   is binary here; exact regret is measured separately so that nearly tied and
   disastrous mistakes do not look identical on the scoreboard. See
   [DeepSeek-R1](https://arxiv.org/html/2501.12948v1) for rule-based outcome rewards.
4. **GRPO.** Sample eight answers to the same question, grade each, subtract
   their mean reward and divide by population standard deviation. All-equal
   groups get zero advantage and are retained. Optimize a clipped ratio of
   new to sampled-old action probabilities; regularize toward a separately
   frozen pretrained reference. This is the one-token case of
   [DeepSeekMath §4.1](https://arxiv.org/html/2402.03300v3). Exact categorical
   `KL(new || reference)` replaces its sampled KL estimator because there are
   only two legal outputs. There is no learned critic.
5. **Stable policy updates.** The old policy generated the answers; the fixed
   reference is the starting prior. Their roles differ. Two updates reuse
   each batch, clipping the probability ratio at 0.8–1.2. Clipping is the
   [PPO surrogate](https://arxiv.org/abs/1707.06347), not a hard guarantee that
   the resulting policy stays within that interval.
6. **Supervised comparator.** The verifier also exposes the complete best set,
   so cross-entropy on its uniform distribution is a natural, cheap baseline.
   It resembles the improvement/distillation loop in
   [Expert Iteration](https://arxiv.org/abs/1705.08439); it is not mislabeled RLVR.
   Highest-regret mining chooses the training question within each deal.
   Development and test use the first admissible question, avoiding that bias.
7. **Measurement.** Split physical deals before examples; exclude previous
   training/reserved deals and feasibility probes. Select on development
   regret, test once, bootstrap whole deals, and measure retention plus a
   small independently replayed transfer smoke. Exercise accuracy, game wins
   and interpretable units answer different questions.
8. **Representation.** A fixed ridge probe asks whether the two exact Q values
   are linearly readable from the final actor residual. Train-only scaling,
   fixed regularization and an untrained twin guard against fitting the test
   or crediting the architecture alone. Inspired by
   [linear probes](https://arxiv.org/abs/1610.01644), this is a diagnostic of
   accessible information, not a causal explanation of a neuron. Unit claims
   still require the `clean-neuron` skill's matched causal tests.

## One real update, without the patois

The bidder/viewer has ones trump and two tiles remaining: **3–0 and 6–6**.
Its original hand is `2–0 3–0 5–4 6–0 6–1 6–5 6–6`; 20 public plays precede
this question. There are 90 legal-consistent hidden allocations. The clean
prefix is preserved verbatim in the evidence's `grpo/worked-update.json`.

| Other seats' style | Q(3–0) | Q(6–6) |
|---|---:|---:|
| Lowest legal | 5/9 | 23/45 |
| Highest legal | 5/9 | 11/15 |
| Uniform legal | 197/360 | 11/18 |
| Equal mixture | **199/360** | **167/270** |

The exact improvement is `71/1080`, about 6.57 percentage points. The preferred
move changes with the other players: lowest-legal favors 3–0, while the mixture
favors 6–6. This makes the behavior-model dependence visible in one hand.

Gus sampled `[3–0, 3–0, 3–0, 6–6, 6–6, 3–0, 3–0, 3–0]`.
The checker returned `[0,0,0,1,1,0,0,0]`: mean 1/4, standard deviation
`sqrt(3)/4`. Successful answers receive advantage `sqrt(3)`; unsuccessful ones
receive `−1/sqrt(3)`. The optimizer increases the successful answer's likelihood
while accounting for the other 63 questions, clipping and KL. After the two
updates, 6–6's probability on this question rose from about 32.9% to 37.5%.
This is a recorded gradient update, not a fabricated illustration.

## Registered recipe and fresh test

2,048 train, 512 development and 512 test deals; all 3,072 distinct. The training
pool selects maximum frozen-baseline regret among eligible questions within a
deal, retaining zero-regret and tied examples. Test has 172 exact ties, so even
uniform legal sampling scores 66.80% correctness. Report regret beside accuracy.

Both arms: 256 rounds × 64 questions, two AdamW updates per round; LR 1e-5,
betas (0.9,0.95), matrix decay 0.1, gradient norm cap 1, KL coefficient 0.02,
seed42. GRPO group8, ratio clip0.2, population-standard-deviation normalization.
SFT uses uniform exact-best targets and the same KL and question batches.
Development evaluated every32 rounds; minimum exact argmax regret selects,
then accuracy, then earlier checkpoint. SFT selected224; GRPO selected256.
No updates followed test. `registration.json` predates training.

| Fresh test, 512 deals | Exact-best accuracy | Rational regret | Team success in the finite model |
|---|---:|---:|---:|
| Uniform legal sampling | 66.80% | 6.344 pp | 56.62% |
| Frozen pretrained tile policy | 70.12% | 5.509 pp | 57.46% |
| Supervised correction | 73.63% | 4.210 pp | 58.76% |
| GRPO | **76.56%** | **3.138 pp** | **59.83%** |

Both trained arms meet the preregistered material-gain criterion: at least
1 pp regret reduction and a paired interval excluding zero. Intervals use
2,000 physical-deal bootstrap samples, seed42. This is one seed and one narrow
finite-model task; the interval does not cover training-seed uncertainty.

## Retention and representation

On a fixed 1,024-view slice of the old pretraining test, tile accuracy changes
50.73% → 48.78% (SFT) → 45.43% (GRPO); NLL 1.319 → 1.368 → 1.512.
Outcome KL changes 0.390 → 0.468 → 0.440. Those outcome weights were frozen,
but their shared backbone was not, so their predictions still changed.
The narrow gains cost general imitation/forecast retention.

Ridge Q-probe test MSE: constant 0.10475, matched untrained twin 0.10474,
pretrained 0.08317, SFT 0.07899, GRPO 0.08006. The learned representation already
contains useful information. Better probe fit does not imply a better policy:
SFT has the lowest probe MSE; GRPO has the lowest decision regret.

## Reproduction, recovery and evidence

Implementation: `lem/observer/rlvr/{verifier,data,train,evaluate}.py`.
Tests independently enumerate hidden allocations and replay all continuations
through the forge referee; they also check seat rotation, hidden-input
invariance, exact reward ties, clipping and real SIGINT recovery.
SIGINT, SIGTERM, a stop file or a cooperative time budget pauses at a completed
round. Atomic checkpoints retain both samplers, all RNG state, AdamW, fixed
recipe/reference identity, selection history and W&B identity. `--resume`
requires the same recipe; completed runs return their original receipts.
Hardware/software changes are not covered by bitwise CPU recovery tests.

[GRPO telemetry](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/2818b60b)
and [SFT telemetry](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/0b96cd1a).
Each arm's optimizer/development/checkpoint segment took about8.3s; outer process
wall12.5/13.3s and retired instructions7.048e10/6.960e10 for GRPO/SFT. Building
all questions took152.4s wall,3.020e12 instructions. Time is descriptive; fixed
question/update budgets govern the comparison.

The predeclared GRPO transfer smoke completed all eight duplicate pairs within
60s per pair: 3/8 game wins against G0 (−1.125 marks/game), 1/8 against Zeb
(−3.375 marks/game). Both use the common bid30 bidder and fixed deterministic
opponent budgets. These small samples neither establish an improvement nor
estimate a stable win rate. Every move and score was independently replayed;
no training/reserved-deal overlap was found.

The next justified experiment is broader lawful curriculum with an explicit
old-task retention objective or constraint, followed by fresh full-game
confirmation. It is a proposal, not a completed or secretly continued run.

## Immutable publication

[Complete exercise/formation/evaluation archive](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/f7bd44b36c8f005d3e34ebea548393b5e325392a/lem/gus-rlvr-silas-2026-10-18)
and [verification receipts](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/3242419683586c7fd80e1954506baea931c3c568/lem/gus-rlvr-silas-2026-10-18/verification). The combined companion packages
contain 142 files / 91,165,523 bytes; every file passed fresh anonymous download
and SHA256 comparison. The 3,072 deal records and inputs were reconstructed,
64 sampled exact certificates recomputed, and all 95 unique smoke deals checked
against training, reserved and feasibility-probe exclusions. Independent referee
tests compare 12 exact trees, including all three styles.

[GRPO model release](https://huggingface.co/jasonyandell/gus/tree/rlvr-grpo-v1-20261018), immutable revision
`0d3210f08de6d039176c6fcdf05a6c13d54e80e8`. Every staged model file passed fresh
anonymous checksum verification; safe-format and inference-checkpoint tensors
and both heads' CPU logits match the selected full checkpoint. Use `head="tile"`.
The SFT best/final and all formation checkpoints remain in the data archive.
The Hub main incumbent is unchanged.

Source implementation: `e8a4566d`. Full `scripts/check`: GREEN, 20 suites passed,
zero failures; CUDA-only Forge unavailable and retired Burl excluded by the
repository's standard gate. Lem has91 passing tests. W&B final metrics were
read back against local history; the installed `gus-training` skill matches
the validated repository files. No training process remains.
