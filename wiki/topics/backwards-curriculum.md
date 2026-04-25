---
title: Backwards Curriculum
kind: topic
first_seen: a8bccfa
last_updated: a8bccfa
status: active
---

## Overview

The backwards curriculum is [[lem]]'s sequencing strategy for training difficulty. Training begins at the end of the game (trick 6, one ply of lookahead) and ratchets backward one ply per stage — trick 5, trick 4, and so on. Each stage adds exactly one ply, making difficulty a counted quantity rather than a hand-wavy one (lem/OVERVIEW.md @ a8bccfa).

## Rationale

At trick 6 there is only one more trick to play; [[expected-q-value]] labels are cleanest and the lookahead horizon is shortest. Starting here gives the model the easiest signal first. As the model plateaus at one stage, the next stage begins, extending the horizon by a single decision. The claim is that this controlled ramp avoids the credit-assignment problems of training on full games from the start (lem/OVERVIEW.md @ a8bccfa).

## Stage structure

| Stage | Decisions covered | Lookahead |
|---|---|---|
| 0 | Rules adapter (no play) | — |
| 1 | Trick-6 narrator decision | 1 ply |
| 2+ | Trick 5, then 4, then 3 … | +1 ply each |

The gate between stages is plateau detection on [[expected-q-value]] delta. Stage 2 does not begin until Stage 1 has clearly plateaued (lem/OVERVIEW.md @ a8bccfa).

## Open questions

- Exact plateau criterion is not defined at this frontier. (?)

## Links

[[lem]] [[star]] [[expected-q-value]]
