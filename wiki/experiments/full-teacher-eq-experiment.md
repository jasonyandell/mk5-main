---
title: Full-teacher E[Q] experiment — the era-4 closeout
kind: experiment
first_seen: 6081420
last_updated: 6081420
status: complete
---

## Question

[[zeb]]'s self-play loop has a bootstrapping problem: early policy mistakes compound
because the model trains on its own games. **Can mixing in oracle-guided
[[expected-q-value|E[Q]]] teacher examples — with the teacher allowed to shape the policy
head directly (`policy_weight=1.0`), not just value/belief — break that compounding and
push play strength past whatever self-play alone reaches?** Framed by Jason at kickoff:
*"The selfplay loop has a bootstrapping problem... The E[Q] oracle breaks this by
providing a 'ground truth' signal for what the best action is in any game state"*
(2026-02-16T05:32:47, [[sources/claude/era4-zeb-era|conversation digest]]). The idea
descends from a seed planted eleven days earlier: *"what if we play zeb vs that [e[q]]
and train... until it exploits zeb the best it can, then train zeb on that play"*
(2026-02-05T17:19).

A second, narrower question arose mid-run when the first mix showed no gain: does
cranking the E[Q] share further (95% E[Q] / 5% self-play) unlock the signal, or does more
oracle exposure hit the same wall?

## Setup

- Bead `t42-4xvg`; training session named `lb-v-eq-1920` — a **different, later** run than
  `lb-v-eq-3740` (the Feb-15 large-belief bootstrap documented in
  `docs/zeb-worker-saga.md`, see [[zeb-fleet-ops]]). Do not collapse the two names.
- Infra: `forge/zeb/learner/go-full-teacher.sh`, still on disk. A real bug fixed in-flight:
  `eq_player.py`'s tensor-shape bug where raw Q-value PDFs `[N,7,85]` were used as policy
  targets instead of `p_make` probabilities `[N,7]` (`3e6ad2b`, 2026-02-15).
- Two mix regimes, run sequentially: 75% self-play / 25% oracle-guided first, then 95%
  E[Q] / 5% self-play (2026-02-17T05:23). Fleet: 3 self-play workers (~10 g/s) + 1
  eval-aux worker (RTX 4070 Ti, ~4 g/s), cost ~$0.335/hr.
- 3.3M-param model (the "large-belief" checkpoint).

## Results

Pre-eval loss-curve numbers at cycle 45 of 50 (25% mix, before the first win-rate eval):
`policy_loss` 0.413→0.377, `value_loss` 0.423→0.211, `ea_policy_loss` 1.195→0.836,
`ea_value_loss` 0.581→0.147, `belief_acc` 71.3%→71.7%, `top1_accuracy` 88.3%→89.0%,
`policy_entropy` 0.552→0.377; E[Q] teacher winning 53.2% vs the model's 46.8% in aux games
(conversation `35859811`, W&B summary paste).

Closeout commit **`6081420`** (2026-02-16 20:52, `t42-4xvg`): *"Experiment ran 1059 cycles
across 25% and 95% E[Q] mix regimes. Finding: E[Q] policy signal doesn't improve play
beyond ~74% vs random at 3.3M params — appears to be a capacity ceiling for this
approach."* The diff itself is a single-line `--eval-every` flag addition — the 1059-cycle
and 74% figures live only in the commit message; no separate results artifact (W&B export,
eval-log file) was located to independently re-derive them. **Asserted, unverified beyond
the commit message and Jason's contemporaneous chat reports** — flagged, not silently
trusted.

Jason's own real-time verdict on the 25% mix was a hedged null, not a triumphant negative:
*"I think it didn't work? not catastrophic but not improving either"* (2026-02-16T13:47).
After the 95%-E[Q] rerun: *"I cranked eq to 95 percent and self play to 5 and have been
training that model all day. it didn't improve. I committed and pushed and we are good"*
(2026-02-17T05:23).

The ~74% figure matches a running joke from earlier in the era — *"I bet it's gonna cap at
74 lol .. it's always 74 with this game"* (2026-02-09T03:36) — suggesting ~74% vs random
was already the observed ceiling for prior Zeb variants before this experiment, and the
E[Q]-teacher mix did not move it.

## Why "capacity ceiling" is probably the wrong name

Going 557K → 3.3M params (6×) bought roughly 70.9% → 76.3% vs random on self-play alone —
about five points ([[zeb]]). That is not the shape of a capacity wall; it's the shape of
an information/architecture wall. Calling the full-teacher result a "capacity ceiling"
quietly points future work at the wrong lever (make the model bigger); the era's own
self-play scaling data says bigger alone won't do it.

## What it ruled in/out about the wall

This experiment tested the most direct fix for [[candlewax|the wall]] — feed E[Q]'s
per-action expected-value signal straight into the policy head as a training target — and
found it did not move play strength past ~74% vs random at 3.3M params, at either mix. It
**rules out "just increase the E[Q]-teacher dose"** as a solution to consuming the
oracle's output. It does not resolve whether the ceiling is capacity, architecture, or the
structural fact that E[Q] is a distribution with no "best" to pick — the thing Jason had
already named the day the teacher idea was conceived:

> "there's no such thing as an e[q] optimal policy unfortunately. it's a distribution, an
> often lumpy, often smooth histogram of discrete scores, not something you can just pick
> 'best' for that application." (2026-02-06T04:00)

The experiment collapsed exactly that histogram into a `p_make` target and trained the
policy head to imitate it — it tested a thing Jason already predicted the answer to, not
the harder open question of how to consume the distribution *without* collapsing it. That
harder question is untested here and remains one of the candidates named at closeout: see
[[zeb]] ("What the era left open") and [[belief-feeding-policy]].

## Terminal status

BUILT and formally closed at conclusion — a deliberate ops commit stating a finding, not a
"we stopped because something broke" note. Distinct from, and prior to, the April
`zeb-parked-eq-primitive` decision (Zeb retired as a system and repurposed as [[burl]]'s
belief-only primitive after a separate calibration eval) — do not conflate the two events.

This is era 4's one durable, load-bearing negative result about E[Q] consumption. See
[[candlewax]], [[zeb]], [[eval-matrix-bradley-terry]],
[[sources/claude/era4-zeb-era|conversation digest]].

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- `forge/zeb/learner/go-full-teacher.sh` confirms `--eval-aux-policy-weight` default 1.0
  and the 25% eval-aux mix; the 95% regime was presumably a runtime env override — not
  verifiable from the script, only from chat.
- The bugfixed file is `forge/zeb/eq_player.py` (not under `learner/`).
- Cheap next probe if this era is revisited: train the policy head on the full E[Q]
  histogram via a distributional loss, rather than another p_make-collapse mix sweep.
