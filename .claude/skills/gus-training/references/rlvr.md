<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

# Exact Silas curriculum and textbook GRPO

Read `wiki/experiments/gus-rlvr-silas-2026-10-18.md` for the registered question,
worked update, results and primary sources. This is a one-answer endgame
curriculum, not the earlier full-game population PPO implementation.

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
