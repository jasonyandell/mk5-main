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

## The completed probability comparison

`wiki/experiments/gus-probability-reward-v3-2026-10-18.md` records three paired
seeds, 256 rounds each. Probability rewards did not beat binary best-set rewards
at the registered threshold. Both improved fresh finite-model regret while
retaining the original losses; brief actual games did not establish transfer.
Do not rerun that completed confirmation or interpret the null as evidence
against expected success as the objective. The two rewards agree on the optimal
move set; their finite-capacity gradients and regularization scales differ.

## Full-hand terminal-outcome training

`lem.observer.outcomes` implements grouped legal rollouts for the existing joint
model: one learner, fixed G0 partner and pinned Zeb opponents, same clean encoder.
The complete study is `wiki/experiments/gus-full-hand-outcomes-2026-10-18.md`.
This is a declared full-game stage, separate from the exact three-style endgame
model. A verified observed terminal result is not an exact population win rate.

When continuation rewards cannot cheaply be enumerated, sampled trajectories
justify GRPO. Group rewards use their mean and population standard deviation;
all seven learner decisions initially share the trajectory advantage. Retain and measure
zero-variance groups. Record old legal probabilities separately from the frozen
parent distribution. The implementation uses clipped action ratios and exact
full-support categorical KL, with the original tile/splat supervised rehearsal.
The matched rehearsal control has the same optimizer steps and supervised draws,
without the RL or KL term. It also collects trajectories for matched diagnostics.

Declare `recipe.reward_horizon` explicitly: `full` credits all seven decisions;
`unsettled` removes reward credit after public banked points guarantee make/set.
For bid30 this is bidder points >=30 or defender points >12. The mask is an
independently replayed loss sidecar, never a new model input. Preserve all28
plays, the original seven-turn denominator, full-state KL and rehearsal. Dividing
by only active turns would change trajectory weighting and confound the ablation.
This removes zero-mean post-outcome score terms by policy-gradient causality;
it does not repair all-win/all-loss groups or prove a finite clipped update helps.
Old v1 plans must run under their pinned source, not an implicit default horizon.
The controlled comparison is recorded in
`wiki/experiments/gus-public-terminal-credit-2026-10-18.md`.

The JSON plan declares train/development/confirmation physical deals, source,
weights, fixed policies, namespaces, seed, recipe, retention bounds and selection.
`validate_plan` rejects split overlap, rehearsal overlap and changed seating.
Source hashes include the native binary; a changed runtime requires a new plan.
Original deal seeds and independently derived rollout action seeds are distinct.
The unchanged collector internally uses the action key; archived episode `seed`
is restored to the original deal seed, with `action_seed` explicitly retained.
Inputs, legal supports, chosen actions, scores and rewards are rebuilt from every
full replay before training. Exact old/new log probability alignment is checked.

```sh
scripts/cpu-slot --threads 2 --timeout 90 -- scratch/venv/bin/python -u \
  -m lem.observer.outcomes --plan PLAN.json --out RUN --mode grpo \
  --device mps --seconds 45
```

Use `--mode rehearsal` for the control and a different output. Resume the same
plan/output with `--resume`. A completed round, including all inner updates and
scheduled development checks, is the atomic progress boundary. SIGINT/SIGTERM,
a stop file or cooperative deadline completes that boundary. A hard crash
repeats only the uncommitted round from saved state. Latest retains AdamW,
rehearsal RNG, Python/NumPy/Torch/device RNG, source/config hashes, history,
selection and W&B identity. One writer owns the run lock. Completed resumes
verify receipts without changing artifact bytes. Real SIGTERM and simulated
mid-update crash recovery have bitwise CPU tests; MPS and cross-runtime bitwise
training are not promised.

`round-*/` stores trajectories, literal decision arrays, before/after/reference
distributions and separate timing receipts. Formation checkpoints preserve every
round. Before/after distributions span the declared inner updates, not one step.
Include initialization in selection: maximize development greedy success within
retention bounds, break ties by sampled success and then earliest round. Read
fresh confirmation only after selection; never promote from a handful of games.
A final trained specimen can differ from the selected checkpoint, especially
when selection retains initialization. Label both explicitly on Hugging Face.

Use the existing `benchmark.extension.Extension` for lazy actual-game evidence:
key by pinned recipe, physical question and sample stream, use individual CPU
float32 inference, independently replay, atomically commit and hash snapshots.
Changed request order or work slicing must not change the completed bytes.
Seeds alone do not establish that guarantee. The study's published evaluator
preserves each full-hand confirmation as such a record; interrupted runs compute
only missing units. Keep sampled and greedy deployment scores separate.


## Diagnose gradient signal before choosing the next lever

Compare loss gradients on the shared backbone, not just scalar losses or a
whole-network norm dominated by an output head. Reconstruct a saved batch's
pre-update checkpoint and rehearsal RNG, then separate reward, KL, tile and
splat gradients. Check their weighted sum against the recorded pre-clipping
norm and component losses. Cross-device reconstruction needs explicit numeric
tolerances; it is not bitwise replay. Raw gradient alignment does not attribute
AdamW's step, which also depends on moments, clipping and decay.

The public-terminal study's follow-up measured comparable, nearly orthogonal
reward/rehearsal backbone gradients; it did not support assuming rehearsal
swamped reward. Six of32 batches had zero current reward gradient. The next
registered root-batch comparison therefore increased independent deals per
update, holding eight rollouts per deal and64 optimizer steps fixed. Count its
extra rollout/data cost. Increasing roots and increasing draws of the same root
answer different variance/coverage questions; do not silently conflate them.
Use fresh reserved confirmation after selection and retain the rehearsal control.
