---
title: Blunder Detector (student-feature blunder classifier)
kind: topic
first_seen: 2026-04-21
last_updated: 2026-04-21
status: retired
---

## Overview

The blunder detector is a small `GradientBoostingClassifier` that predicts, from decision-level features, whether [[gus]]'s student will blunder (regret > threshold, typically 8 Q-pts) on a given decision. It is trained on labeled decisions from eval runs and used at inference to gate the [[detect-and-route]] fallback policy (f90682c, 5373223).

## Two versions

**v1 — oracle features** (f90682c): trained with oracle E[Q] vectors available. Primarily for analysis.
- ROC-AUC: 0.926 / PR-AUC: 0.29
- Dominant features: `oracle_spread` (max−min legal E[Q]) and `oracle_eq_std` — together 77% of importance.
- At 15% flag rate: 80% recall of blunders. At 25%: 99% recall.

**v2 — student features only** (5373223): trained using only what is available at inference from the student's own outputs. This was the deployable candidate; it was never wired into champion/arena/forge production (`grep -rl blunder_detector` across the repo returns no hits outside `gus/eval/`).
- ROC-AUC: 0.839 / PR-AUC: 0.15
- 28 features: policy (pi_peak, pi_entropy, pi_argmax_margin), V_head scalar, Q_head across K=20 sampled worlds (per-action mean/std/min/max, legal spread, std of means), consistency gaps (V_head vs policy-expected-Q, V_head vs Q_mean_chosen), meta (decision_idx, trick_num, trick_pos, player_rel, legal count, declaration one-hot, voids count), belief (max prob, entropy, high-confidence count).
- **Top feature: pi_peak (0.19)** — when π_me is uncertain, trigger fallback. Intuitive and directly measurable.

## Business case

At 20% flag rate with oracle-argmax replacement:
- Baseline regret 1.13 → 0.49 (57% reduction, approaching the 0.5-1.0 teacher-noise floor).

Key limitation: student's Q_head spread across worlds is a weaker blunder proxy than oracle spread. Q_head was trained on one random world per forward pass, making its spread signal noisy. Two upgrade paths: multi-world variance regularization during Q_head training, or K=50+ worlds at inference (5373223).

## Training data

Labeled decisions from 3000g corpus eval runs (6k train / 3k test split). Label: `regret > 8 Q-pt` = blunder. Files: `gus/eval/blunder_detector.py` (oracle), `gus/eval/blunder_detector_student.py` (student) (f90682c, 5373223).

## Retired

Never wired into champion, arena, or forge. The [[detect-and-route]] architecture it was
built to gate was itself abandoned when the project pivoted to `jud`/[[champion]] (self-play,
no CFR+) rather than a distilled-value look-ahead/fallback stack. Kept as a source-backed
record of the blunder-rate analysis, not a live component.

## Links

[[gus]] [[regret-eval]] [[detect-and-route]] [[v-pi-decoupling]] [[expected-q-value]] [[champion]]
