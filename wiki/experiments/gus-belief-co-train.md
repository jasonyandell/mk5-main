---
title: Gus Belief Co-Train + q-bootstrap-belief Mode (§21)
kind: experiment
first_seen: 548d32a
last_updated: cf8ff79
status: active
---

## Summary

Two findings from §21: (1) belief top-1 is at the Bayes ceiling (39.184%) — the signal
simply isn't there at early decisions; (2) joint co-training belief + world_encoder +
Q_head with distribution target slightly worsens q-bootstrap regret; but sampling worlds
from belief at inference (q-bootstrap-belief) gives 0.655 — the closest any look-ahead
has come to the 0.551 direct baseline. (commit messages @ 548d32a, cf8ff79)

## Belief at Bayes ceiling (§21 — 548d32a)

Computed Bayes-optimal belief top-1 from the oracle's own sampled worlds on `corpus_eval_20.pt`:

**Bayes ceiling: 39.184%**

Gus v3 belief head is at ~38-39% per §6 — within noise of the theoretical maximum.
The cause is not architecture or capacity: at decision_idx 0-5, belief is ~33% (pure
prior) because the hidden information simply hasn't been revealed yet. Top-1 accuracy
is solved. (commit message @ 548d32a)

**Reorientation**: the remaining lever is posterior *shape* (calibration), not top-1
accuracy. §15 showed distribution-target training closes 47% of the KL gap. The open
loop is co-training belief + world_encoder + Q_head jointly so that better calibration
propagates to look-ahead value estimates.

## Co-train experiment (cf8ff79)

**Setup**: frozen trunk + π_me + V_head; unfrozen belief + world_encoder + Q_head;
distribution target for belief (not one-hot truth).

**Script**: `gus/train/train_belief_q_joint.py`

**Results**:

| Metric | Before | After | Delta |
|---|---|---|---|
| Belief KL | 0.084 | 0.067 | −20% |
| q-bootstrap regret | 0.685 | 0.718 | +5% (worse) |

Belief KL improvement is real but downstream q-bootstrap regret worsens slightly.
Q_head was already at a sweet spot calibrated to the original belief's output distribution;
perturbing belief upsets that equilibrium. The "co-training propagates calibration"
hypothesis is falsified on this setup. (commit message @ cf8ff79)

## Unexpected win: q-bootstrap-belief mode

**New inference mode**: sample worlds from the belief head at inference, rather than
reading from the oracle corpus.

| Mode | Regret |
|---|---|
| Direct π_me (baseline) | 0.551 |
| **q-bootstrap-belief (original adapter)** | **0.655** |
| q-bootstrap (corpus worlds) | 0.679 |
| All other look-ahead modes | ≥ 0.718 |

0.655 is the closest any look-ahead variant has reached to the 0.551 direct baseline —
a 19% gap. This mode uses `gus/model/sample_worlds.py` (the symmetry-checker wrote it
off-plan; it turned out to be exactly what this experiment needed). (commit message @ cf8ff79)

## §22 — "What you do past belief"

With belief at the Bayes ceiling, the remaining 42-craft is about acting well given
unresolvable uncertainty. §22 identifies extractable analytics from the existing corpus
(outcome-variance, action-choice fragility, belief-limited high-impact decisions) and the
research question: π_me commits to one meta-strategy ("play for the mode of the marginal")
where humans use several (mode, signal, hedge, gamble). A richer student could output a
meta-strategy distribution; training data already exists in the oracle's per-world tensor.
No code written — noted as future direction. (commit message @ 94d8646)

## Links

[[gus]] · [[topics/regret-eval]] · [[experiments/gus-lamir1-mode-comparison]] · [[experiments/gus-belief-calibration-diagnostic]] · [[joint-world-tensor]]
