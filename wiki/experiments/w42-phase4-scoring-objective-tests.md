---
title: w42 Phase4 Scoring Objective Tests
kind: experiment
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: active
---

## Summary

[[w42]] bead `t42-br7n.3` tests Chapter 10 scoring-objective claims with
deterministic terminal transforms, generated hand traces, match-mode proxies,
and timed advancement simulations under `w42/phase4_scoring_objective_tests/`.

The run generates 4000 hand traces, 640 match simulations, and 160 timed
advancement trials. Early mark-terminal states appear in `94.5%` of generated
hands, saving `4.3805` tricks on average. Among made ordinary contracts,
defender partial points are erased `84.989%` of the time under marks. Ordinary
sets compress 41 distinct point-severity values into the same one-mark outcome.

The generated match proxy disagrees on winner between point-to-250 and marks-to-7
`15.4688%` of the time. Timed synthetic pools choose a different leader by marks
than by points `15%` of the time. These are strong evidence that marks change
the objective surface, but not proof of human tournament policy or skill
development claims.

Claim-ledger impact: deterministic and generated-trace scoring mechanics are
supported; `ch10-point-system-skill-signal` and
`ch10-timed-marks-advancement-objective` remain context-limited because they
need real policy populations, tournament formats, or calibrated timing data.

## Method

| field | value |
|---|---|
| bead | `t42-br7n.3` |
| artifact directory | `w42/phase4_scoring_objective_tests/` |
| runner | `w42/phase4_scoring_objective_tests/run_phase4_scoring_objective_tests.py` |
| validation | `w42/phase4_scoring_objective_tests/validate_outputs.py` |
| generated hands | 4000 |
| match simulations | 640 |
| timed trials | 160 |
| policy pairs | 4 heuristic matchups |

Generated play uses deterministic heuristic policies. Trick count is used as a
tournament-speed proxy; no wall-clock table timing or human bracket logs are
included.

## Claim Results

| claim | status | evidence |
|---|---|---|
| `ch10-score-mode-objective` | supported | point and mark terminal labels diverge over generated hands |
| `ch10-early-terminal-under-marks` | supported | early terminal rate `0.945` |
| `ch10-nonbidder-partial-points-erased` | supported | made ordinary partial erasure rate `0.84989` |
| `ch10-set-severity-compression` | supported | 41 ordinary set severity values collapse to one mark |
| `ch10-special-bid-mark-multiplier` | supported | deterministic 84/126/168 multiplier table covered |
| `ch10-low-bid-score-distortion` | supported | match winner disagreement `0.154688` |
| `ch10-point-system-skill-signal` | context-limited | heuristic policy rows only; needs oracle/human population |
| `ch10-tournament-speed-tradeoff` | supported-for-generated-trace-proxy | marks reduce played tricks by `23.814062` in the proxy |
| `ch10-timed-marks-advancement-objective` | context-limited | synthetic pool disagreement `0.15`; needs real format |

## Links

[[w42]] | [[winning42-ch10-tournament-scoring]] | [[forge]]
