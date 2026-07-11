---
title: Gus Arena Pilot (game-level eval)
kind: experiment
first_seen: 1a2f67f
last_updated: 1a2f67f
status: active
---

## Summary

First game-level evaluation of [[gus]] student against the E[Q] bot. Decision-level
regret of 1.39 Q-pts compounds to a ~30pp contract-made gap at game level. Blunder
forensics surfaced the [[topics/v-pi-decoupling]] pattern motivating v3. (commit message @ 1a2f67f)

## Setup

- **Adapter**: `v2_voids_3000g_big` (3.4M params, 3000g corpus, best regret 1.39 Q-pts)
- **Student seat**: seat 0
- **Opponents**: E[Q] bot at seats 1, 2, 3
- **Games**: 20 (2 seeds × 10 declarations, seeds 900020-900021)
- **Baseline**: all-E[Q]-bot on same seeds

## Results

| Metric | Student | All-bot baseline |
|---|---|---|
| Contracts made | 10/20 (50%) | 16/20 (80%) |
| Avg bidder points | 21.2 | 29.6 |
| Delta | | −8.4 points/hand |

## Key insight: compounding

Decision-level 1.39 Q-pt mean regret translates to a ~30pp contract-made gap at game
level. Individual blunders on strategically consequential decisions accumulate across 28
plays per hand. The scalar regret metric understates game-level impact. (commit message @ 1a2f67f)

## Blunder forensics

`scratch/BLUNDER_FORENSICS.md` (gitignored) rendered per-trick tables with regret badges
(✓ ▽ ⚠ 🔥) and student brainstate (legal E[Q], π probs, V_head, top-3 belief) per
decision. Key pattern surfaced: on specific blunder decisions, V_head correctly predicts
+26 (best-legal E[Q]) while π_me concentrates mass on a play worth −0.4. The heads are
decoupled — the student "knows" the position is good but acts badly.

This V/π decoupling is the diagnostic that motivates the v3 consistency regularizer.
See [[topics/v-pi-decoupling]] and [[topics/consistency-regularizer]].

## Eval tooling shipped

- `gus/eval/arena.py` — full-game simulation; replaces `select_actions` for student seat
  with neural inference; reports contracts made/set, avg bidder/defender points, baseline comparison
- `gus/eval/play_visualizer.py` — renders a single game as readable markdown with per-trick
  regret badges and per-decision brainstate

## Links

[[gus]] · [[topics/regret-eval]] · [[topics/v-pi-decoupling]] · [[experiments/gus-scaling-ladder]]
