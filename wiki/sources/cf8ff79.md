---
title: "Source: cf8ff79"
kind: source
commit: cf8ff79
date: 2026-04-22
author: Jason Yandell
---

## Commit message

> feat(gus): belief co-train experiment + q-bootstrap-belief mode
>
> §21 open-loop experiment: joint co-train belief + world_encoder + Q_head
> with distribution target. Belief KL dropped 0.084→0.067 (−20%) as designed,
> but downstream q-bootstrap regret got slightly WORSE (0.685→0.718).
> Hypothesis that "co-training propagates calibration" is falsified on our
> setup — Q_head was already sitting at a sweet spot for the original
> belief's output distribution.
>
> Unexpected win from the A/B harness: sampling worlds from the belief head
> at inference (vs reading from oracle corpus) gives regret 0.655 on the
> ORIGINAL adapter. Closest any look-ahead variant has come to the 0.551
> direct baseline (gap 19%).
>
> - gus/train/train_belief_q_joint.py: joint co-train trainer (frozen
>   trunk + π_me + V_head; unfrozen belief + world_encoder + q_head)
> - gus/eval/lamir1.py: q-bootstrap-belief mode added (reuses
>   gus/model/sample_worlds.py)
> - gus/model/sample_worlds.py: finally in use — the file symmetry-checker
>   wrote off-plan turned out to be exactly what this follow-up needed
> - PRACTICALITIES §21 extended with both findings
> - MORNING4_STATUS addendum 2

## Files changed

| File | Change |
|---|---|
| `gus/train/train_belief_q_joint.py` | New — joint co-train trainer |
| `gus/eval/lamir1.py` | Updated — q-bootstrap-belief mode |
| `gus/model/sample_worlds.py` | Now in use — belief world sampler |
| `gus/PRACTICALITIES.md` | §21 extended — co-train + q-bootstrap-belief findings |
| `gus/MORNING4_STATUS.md` | Addendum 2 |

## What this commit establishes

Co-train hypothesis falsified (KL improves, regret worsens). Unexpected win: q-bootstrap-
belief at 0.655 is the best look-ahead result to date. `sample_worlds.py` finds its use.

## Links

[[experiments/gus-belief-co-train]] · [[topics/lamir1]] · [[topics/regret-eval]]
