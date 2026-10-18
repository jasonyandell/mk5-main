---
title: "Gus training review: public settlement and the next learning signal"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---

## Scope and conclusion

Independent code review and saved-hand replay after
[[gus-population-depth-2026-10-18]] identify a concrete next ablation for
[[gus-tiny-gpt]]: stop assigning reward advantage to choices after public points
have already settled make/set. This review ran no new training or network
inference. It does not establish that the change will improve strength or that
this noise caused the campaign plateau.

The evidence supports useful learned play at 107k parameters: the large-corpus
pretrained G0 beat its 50k-deal predecessor in
[[lem-observer-first-h2h-2026-10-17]]. The completed depth campaign did not
establish another improvement on fresh confirmation. This distinguishes an
unsuccessful reward recipe from an established capacity ceiling.

## Saved-hand audit

The review replayed all 4,096 training hands in each of G1, G10 and G20. Before
each action, bid-30 settlement means completed-trick banked points satisfy
`bidder_points >= 30` or `defender_points > 12`. It uses public information only.
Discretionary means the learner has more than one legal tile in its own hand.

| Generation | Discretionary learner choices | Already publicly settled | Share |
|---|---:|---:|---:|
| G1 | 17,315 | 4,141 | 23.92% |
| G10 | 17,473 | 4,585 | 26.24% |
| G20 | 17,468 | 4,348 | 24.89% |

These are complete counts for three selected archived generations, not an
estimate over all generations or a strength interval. The audit verifies every
actor and legal play, seven learner decisions per hand, terminal banked points,
and team reward. Public settlement cannot change before the hand ends.

`lem/observer/population.py` completes all 28 plays and assigns the final team
reward to all seven learner actions. The policy update uses that reward minus a
historical scalar baseline. After settlement, the true advantage of every action
for contract return is zero. At the sampling policy, the ordinary score-function
term therefore has zero expectation and can contribute finite-sample noise.
That statement does not imply that clipped, multi-pass PPO remains unbiased
after the policy changes. Entropy regularization is a separate term.

All 160 campaign blocks completed all four passes. Across the 640 saved pass
reports, maximum KL was **0.01223**, below the registered 0.03 guard. The campaign
was not stopped early by that guard; increasing it alone does not address the
observed result.

## Prioritized experiments, not executed

Follow-up: [#215](https://github.com/jasonyandell/mk5-main/issues/215).

1. **Credit assignment at fixed small capacity.** Start from G0 and compare the
   original recipe with reward advantage set to zero only after public
   settlement. Keep entropy treatment, original loss normalization, population,
   update budget and other settings identical. Use several training seeds and
   fresh paired evaluation. Measure gradient variability and actual outcomes.
   A later separate comparison can replace the scalar baseline with a baseline
   trained from the same visible prefix and realized outcomes; it cannot read
   other hands.
2. **Retain useful pretraining.** Compare plain PPO with a predeclared small
   replay loss on lawful pretraining records. Measure sampled-policy and greedy
   play separately, plus own-seat legal-conditioned prediction on discretionary
   choices before settlement. Raw imitation retention alone is not success.
3. **Teach consequences with richer outputs.** The pending splat direction in
   [[lem-nanogpt-foundations]] retains the input tokens and predicts lawful
   per-action consequences. A first make/set probability target avoids treating
   random post-settlement continuation as strategic margin advice. Calibrate
   against the specified teacher field, measure tie-aware teacher regret, then
   test actual games. It does not inherit an optimal-play guarantee.

The next experiments need new selection/confirmation deals; the completed
campaign's confirmation is retired from selection. Preserve the pairwise and
partner matrix. Recent checkpoint count is not evidence of behavioral diversity.

## Reusable method

The versioned [`gus-training` skill](../../.claude/skills/gus-training/SKILL.md)
records the input contract, corpus squeeze, population learning, evaluation,
formation and publication method. Its operations reference covers bounded
workers, resume semantics, telemetry and download verification. An identical
copy is installed at `~/.codex/skills/gus-training` for Codex discovery.
It names the companion **`clean-neuron`** skill and preserves its tools →
vocabulary → data/learning → recursive-generalization ladder, with the
sequence-model scope correction in [[clean-neurons-42]]. Neither the skill nor
this review authorizes an unrequested training campaign.

## Evidence

Source checkout: `b3c80377f86090d9939b215756ae123c18badac6`.
[Public replay audit, script and execution log](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/1845d399021856f6a56d0714a768361df6de01e4/lem/gus-training-review-2026-10-18).
The audit records 184 input-file SHA256s, source hashes, exact counts and all
per-pass diagnostics. Its JSON SHA256 is
`51df88dab85f91dd424d1554de478bc00c87915a1b78bb92b64da849d9c7dc67`.
All four published files passed fresh anonymous download/hash verification.
Original trajectories and completion receipts are in
[[gus-depth-campaign-evidence-2026-10-18]].
