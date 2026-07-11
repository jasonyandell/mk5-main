---
title: E[Q] — Expected Q-Value
kind: topic
first_seen: ece6dcf
last_updated: b89ff635
status: active
---

## Overview

E[Q] is a scalar measuring how good a position is for the team taking a given action, averaged over game outcomes. It is produced by the Q-value checkpoint from [[forge]] (lem/OVERVIEW.md @ a8bccfa).

## Founding (era 3, 2026-01-09..31)

E[Q] was born to fix "strategy fusion": the perfect-information oracle it's derived from plays
as if it can see all four hands, because during training it can. See [[strategy-fusion]] for
the full diagnosis-through-founding arc; the mechanical summary is here.

The founding sentence, asked as a proposal and checked against a "karpathy himself" standard
(2026-01-10T23:56:04): "We train on games where every move is chosen by averaging oracle values
over all hidden worlds consistent with what's been publicly played so far." Birth commit
`ece6dcf` (2026-01-10, "E[Q] data generation pipeline with backtracking sampler") creates
`forge/eq/` from nothing — 18 files, 2,860 insertions: `game.py`, `voids.py`, `sampling.py`
(backtracking sampler, MRV heuristic), `oracle.py`, `generate.py`. Modal GPU infra for
shard generation lands the same day (10,000 shards, ~$25).

The commit that defines the mechanism lands the next day: `341dd53` (2026-01-11) fixes
`generate.py` to reconstruct a *hypothetical* initial hand per sampled world
(`remaining + played_by`) instead of using the true deal. That one diff is what "E[Q]" means in
this codebase — a Q-value averaged over sampled worlds consistent with the public history,
never the real deal.

The scope was cut the same night to argmax-only consumption (2026-01-11T00:04:42): "I also
don't care about signaling. yet. I want a damn good solid base player." That cut is the wall's
footing — see [[argmax-q-ceiling]] and [[strategy-fusion]].

**The uniform world prior is the root of E[Q]'s later-diagnosed feel-lessness.** Belief-weighting
of the sampled worlds — assuming the opponent's likely holdings aren't uniform, given what they
did and didn't play — was proposed and deferred twice on principle ("no heuristics... that's
just an example of something that should arise naturally if it is correct to do,"
2026-01-14T03:13:23). Uniform world-sampling is precisely what removes the read: no posterior
over what the opponent likely holds means no concealment, no signal, no person. This is not a
contradiction to resolve after the fact — it was already on the page in week one.

## Build timeline (era 3 highlights)

- **2026-01-17** — posterior-weighted E[Q] (ESS mitigation, adaptive K-window, rejuvenation
  kernel) across a 9-commit chain spanning Jan 17-18; [[decisions/qval-over-policy-models]]
  documented the same week.
- **2026-01-18** — schema v2 (`e_logits`→`e_q_mean`), canonical spec `docs/EQ_STAGE2_TRAINING.md`,
  99.3% cut in GPU→CPU transfer via world-invariance of the actor's own hand (`9098e5d`).
- **2026-01-19** — **12,325x GPU vectorization** (`0db7750`): 60,745ms → 4.9ms end-to-end,
  0.5 → 6,493 games/sec, "57/57 tests pass, bit-for-bit correctness verified." The old
  sequential-Monte-Carlo path was deleted the same day — no legacy fallback kept.
- **2026-01-22..24** — CUDA-only enumeration; CPU pipeline moved to `cpu_deprecated/`, its
  fallback stripped fifteen minutes after landing (the directory itself was deleted from the
  repo in the 2026-07 docs consolidation). `ef199b0` (Jan 24) replaces mean/variance
  storage with a full **85-bin E[Q] histogram** (`e_q_pdf: Tensor[7,85]`), 42.5x the storage —
  the mechanical ancestor of the later name [[candlewax]] (a distinct, later-named object; see
  that page's concordance section for the dating).
- **2026-01-25..26** — the slot-0 positional-bias investigation; see [[q0-positional-bias]].
- **2026-01-27** — adaptive sampling to convergence; the new labels agreed with the old
  1k-sample method only ~65% of the time, meaning the *old* sampling had been wrong a third of
  the time. "I may have said disagree 65ish percent but it's agree only 65ish percent... that's
  a breakthrough."

The theoretical accuracy ceiling for any argmax-Q player is documented separately: see
[[argmax-q-ceiling]].

## Consumption foot-guns

Hard-won rules for anyone consuming E[Q] outputs (forge/eq/README.md @ 233b7dc5):

- **E[Q] is points, never logits — do not softmax.** `e_q_mean` is expected points,
  roughly [−42, +42]. The values are already interpretable (−17.89 means "expect to be
  17.89 points behind"); softmaxing them is a category error.
- **Q-value checkpoints, not policy checkpoints.** Soft-cross-entropy (logit) models
  preserve action ordering — argmax matches — but their magnitudes are arbitrary-scale;
  only `loss_mode='qvalue'` checkpoints (`domino-qval-*.ckpt`) output points directly.
  See [[decisions/qval-over-policy-models]].
- **World sampling is backtracking, never rejection.** Rejection sampling has an
  exponential rejection rate as voids accumulate and no termination guarantee;
  backtracking with the MRV heuristic always finds a solution if one exists — and one
  always does, the real deal being proof. A backtracking failure therefore indicates a
  void-inference bug, not bad luck.
- **Adaptive sampling is the "drunken master" technique.** Sample until max(SEM) over
  legal actions falls below 0.1 points instead of to a fixed count: 8.4× better SEM
  than 1k-fixed (0.077 vs 0.649) in a 100-game head-to-head. Game outcomes are the
  wrong metric for judging label quality — tiny E[Q] differences cascade into different
  game trajectories; what training needs is confident labels across diverse positions.
  Convergence cost varies by phase: early game ~75–95k samples, mid ~55–75k, late ~50k.

Adjacent rule: bidding consumes the oracle via Monte-Carlo P(make) simulation, not the
model's value head — bid thresholds (30, 31, 32, 36, 42) make the target a cliff
landscape MSE regression can't fit, while simulate-and-count handles it naturally
(forge/ORIENTATION.md @ 233b7dc5; the value-head failure itself is on [[the-oracle]]).

## Roles across the project

E[Q] is the project's central cross-era primitive, not a LEM-only mechanism. It has served
a distinct role in each era: [[lem]]'s policy/grading, then [[burl]]'s belief tool
(`tools/eq_distribution.py`), then [[gus]]'s distillation target, and now jud v1's explicit
"wall" — the frontier finding that a 470k MLP trained on hand-level Monte-Carlo labels
cannot out-rank E[Q] n=10's per-move oracle (`wiki/log.md`, 2026-07-06 entry).

Two of its original functions, from the LEM era:

1. **Game generation policy** — games are played E[Q]-greedy with N=10 rollouts. This setting was validated during an earlier experiment (zeb) as not materially worse than N=100 and significantly cheaper (lem/OVERVIEW.md @ a8bccfa).

2. **Ground-truth grader for [[k1-grading]]** — the bot's E[Q] sets the acceptance threshold for [[star]] traces. The model's chosen action must match or exceed the bot's E[Q] to keep the trace (lem/OVERVIEW.md @ a8bccfa).

## Relationship to narration

[[narration]] carries running score information, which reflects trick points accumulated. E[Q] over game outcomes is the deeper signal that determines which actions are worth learning from (lem/OVERVIEW.md @ a8bccfa).

## Links

[[forge]] [[k1-grading]] [[narration]] [[lem]] [[strategy-fusion]] [[argmax-q-ceiling]]
[[q0-positional-bias]] [[candlewax]] [[eq-genesis]] [[decisions/qval-over-policy-models]]
