<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

# Exact Silas curriculum and textbook GRPO

Read `wiki/experiments/gus-rlvr-silas-2026-10-18.md` for the registered question,
worked update, results and primary sources. This is a one-answer endgame
curriculum, not the earlier full-game population PPO implementation.

For the selected success-probability objective, read
`wiki/decisions/gus-success-probability-planning.md`. Use `--mode exact --reward
success` for direct verified expected success; `--reward best` is the explicit
binary-correctness control. Search and future-splat auxiliary training are
deferred. Keep the other seats fixed. Lazy continuation commands are in
`references/benchmark.md`; the fixed training and evaluation fields differ.

## Specify what the verifier proves

`lem.observer.rlvr.verifier` enumerates Silas's full legal-consistent remaining
deal fiber and sums exact `Fraction` outcomes under a declared population.
Version1 uses bid30/pip, a two-tile penultimate-trick decision and equal mixtures
of all-other-seats lowest/highest/uniform legal play. Root belief ignores prior
behavior likelihood; this is an explicit approximation. Continuation authority
is `exact_given(model_id)`, not a certificate of unknown opponents' behavior.

The root action is fixed across worlds before averaging. The learner's later
tile is forced. Extending to a second discretionary learner choice requires an
information-state solver: never replace that with per-world maximizing. Keep
the exact-vs-sampled and world-vs-continuation distinctions visible.

Teacher worlds, rational values and best-set labels are outputs/sidecars only.
The neural input remains the original own seven and public contract/history.
Legal output masking is permitted; it is not an input feature. Keep the literal
prefix in each row, reconstruct it independently and pin its tokenizer.

## Run a controlled comparison

- Register source weights, field, data namespaces/exclusions, budgets, dev
  selection, stopping, material gain and untouched test before learning.
- `python -m lem.observer.rlvr.data` constructs resumable SQLite-backed data;
  repeat the identical command after its cooperative budget. Its manifest
  pins the compiled JSONL archive. Highest-baseline-regret mining is train-only.
- `python -m lem.observer.rlvr.train --mode grpo ...` samples groups per
  question, accepts every exact-best action, uses within-group population
  standard deviation and retains zero-advantage groups. Old policy probabilities
  and the separately frozen reference have distinct jobs. Version1 uses PPO
  clipping and exact full-support KL in place of a sampled estimator.
- Run `--mode sft` from the same parent with the same question stream and
  optimizer-step budget. Complete best-set supervision is cheap here; GRPO
  must earn its extra machinery. Do not call imitation of corrected actions RLVR.
- `python -m lem.observer.rlvr.evaluate` reads the fresh test only after both
  arms finish, with paired deal intervals, named-field results, ridge probes,
  a random-initialization control and old-task retention. Never tune on this
  report. Preserve final and best independently.
- A frozen output head can drift because its shared backbone changes. Measure
  both tile prediction and splat retention. The first curriculum improved
  exact answers while degrading old-task prediction; it did not establish a
  stronger general player. Smoke games are adapter/transfer observations.

## Stop, resume and retain the lesson

Wrap heavy commands in `scripts/cpu-slot`; use a cooperative budget below its
outer cap. SIGINT/SIGTERM, `--stop-file` and `--seconds` pause after a complete
round (both inner updates). `latest.pt` retains AdamW, batch and answer sampler
states, Python/NumPy/Torch/device RNG, fixed recipe, source/data hashes, best
selection/history and W&B identity. Resume with identical flags plus `--resume`.
Completed resumes preserve existing receipts. No bitwise promise across devices
or versions; actual signals and bitwise CPU recovery are tested.

Keep the first mixed-reward group in `worked-update.json`: exact Q values,
sampled answers, verifier rewards, normalized advantages and old/new/reference
probabilities. This connects a real move to the gradient update without
inventing a reasoning trace. Preserve formation checkpoints for `clean-neuron`.
Linear probes show readable information; causal unit claims still need that
skill's held-out matched interventions. Publish evidence and exact checkpoint
identities before explaining a result as reproducible.

## Retain the old task and choose the simplest justified objective

The completed three-iteration study lives in
`wiki/experiments/gus-retention-iterations-2026-10-18.md`. Exact expected binary
reward plus rehearsal improved fresh finite-model answers while retaining the
two original losses. GRPO with rehearsal had nearly identical development
regret; one extra block gave no gain. This is one short seed/recipe, not a
universal algorithm ranking or evidence that larger models cannot help.

- Build a compact `python -u -m lem.observer.rlvr.rehearsal` bundle from the
  original splat caches. Its training split comes from original train deals;
  retention selection uses original validation deals. Split physical deals,
  retain all four views, and exclude exact exercises and reserved benchmark
  schedules. The first bundle balances sources; state that distribution rather
  than presenting it as the original pours' frequency distribution.
- Enable `--replay DIR --replay-weight 1 --replay-batch 32` to add original
  tile CE plus splat CE on each optimizer update. The outcome head learns only
  from this supervised task. The independent replay RNG is checkpointed;
  original question/answer streams stay matched across comparison arms.
- If every legal answer's reward is cheaply available, compare `--mode exact`:
  `-sum(pi * is_exact_best) + beta * KL(pi || frozen_parent)`. This directly
  differentiates the finite expected reward; it needs neither a learned critic
  nor action sampling. It differs from GRPO's normalized clipped surrogate.
  The default `--reward best` uses binary best-set labels. With `--reward success`,
  it uses Q directly: probability gaps scale the reward gradient. Both tied
  answers have equal reward and zero reward gradient. Evaluate rational regret,
  greedy success and stochastic policy return as well as correctness. Success
  rewards are restricted to exact mode: two-level within-question GRPO
  standardization would discard their gap. Keep KL and rehearsal scales explicit.
- Keep matched `--mode sft` and `--mode grpo` controls when answering a method
  question. Rehearsal reuses supervised labels; it is not stale-policy replay
  inside the on-policy GRPO objective. Use a fixed parent KL reference and save
  old sampled-policy probabilities separately when GRPO needs them.
- Set `--retention-nll` and `--retention-splat` before the run. These bounds
  apply to loss increases over the initial parent. Include initialization in
  checkpoint selection; minimize exact development regret among feasible
  checkpoints. Report raw accuracy too, since stable losses do not guarantee
  stable top-1. Do not select again after fresh confirmation.
- For a bounded capacity question, `--extra-layers 1` adds one zero-output
  residual block, preserving both initial heads. Compare from the same original
  parent and at the same question/replay/update budget. Require a predeclared
  material development gain within a cost bound. CPU instructions measure host
  work; MPS work and wall time need separate descriptive reporting.
- Reserve a distinct `rlvr.data --namespace` and exclude all previously used
  physical deals. Generate before selection, read parent versus the selected
  model once afterward. The frozen static panel stays a repeated regression
  panel; it cannot become fresh confirmation merely by being renamed.

Use short cooperative segments below the cpu-slot outer cap and a finite arm/
segment count. The study used256 rounds,45s cooperative,90s outer, maximum4
segments/arm; each finished in8–15s. These are demonstrated starting settings,
not mandatory budgets. Resume with identical flags plus `--resume`, using the
checkpoint's pinned source version. New objective/replay/growth settings are
part of the identity, not flags to change in place. Actual SIGTERM plus bitwise
CPU recovery is tested for all three objectives with rehearsal and growth.

Save a real prefix, exact values, legal probabilities and an analytic derivative
for teaching. Distinguish an initial/final comparison from a recorded single
optimizer update. Preserve all formation checkpoints and static native traces
for `clean-neuron`; successful training alone is not a unit explanation.

## The completed probability comparison and full-hand next step

`wiki/experiments/gus-probability-reward-v3-2026-10-18.md` records three paired
seeds, 256 rounds each. Probability rewards did not beat binary best-set rewards
at the registered threshold. Both improved fresh finite-model regret while
retaining the original losses; brief actual games did not establish transfer.
Do not rerun that completed confirmation or interpret the null as evidence
against expected success as the objective. The two rewards agree on the optimal
move set; their finite-capacity gradients and regularization scales differ.

Issue #222 carries the next authorized stage: full trajectories with independently
verified terminal make/set rewards, one learner seat, fixed G0 partner and pinned
Zeb opponents, same joint network and clean encoder. This is a declared full-game
stage, separate from the three-style exact endgame model. Preserve the previous
fields and results rather than silently relabelling them.

When all continuation rewards are no longer cheaply enumerable, grouped sampled
trajectories justify GRPO's estimator. Register the actual groups, normalization,
old/reference probabilities, clipping, KL, replay and update budgets. Compare a
matched rehearsal-only control. Preserve root deal seeds separately from rollout
sampling seeds; repeating a deal must not accidentally repeat the identical
learner random stream. Every seven-decision trajectory has one independently
verified terminal reward. Full hands and verifier results stay audit/output data.

Use short collection/update segments and complete atomic recovery state. Measure
zero-variance groups, legal policy entropy, initial probability-ratio alignment,
retention and actual paired outcomes before changing capacity. Preserve fixed
other-seat identities; initial model, latest checkpoint and selected checkpoint
must stay distinct. No long run or general-player promotion follows from a smoke.
