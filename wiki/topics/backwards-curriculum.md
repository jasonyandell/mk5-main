---
title: Backwards Curriculum
kind: topic
first_seen: a8bccfa
last_updated: a8bccfa
status: superseded
---

## Overview

The backwards curriculum was [[lem]]'s planned sequencing strategy for training difficulty: begin at the end of the game (trick 6, one ply of lookahead) and ratchet backward one ply per stage — trick 5, trick 4, and so on. Each stage was to add exactly one ply, making difficulty a counted quantity rather than a hand-wavy one (lem/OVERVIEW.md @ a8bccfa).

**Designed, never attempted.** LEM plateaued in Stage 1 (trick-6 decisions only) and pivoted to [[burl]] before ratcheting backward past trick 6 even once. See [[lem-to-burl-handoff]]. Every claim below describes the plan as designed, not a result achieved.

## Rationale

At trick 6 there is only one more trick to play; [[expected-q-value]] labels are cleanest and the lookahead horizon is shortest. Starting here gives the model the easiest signal first. The plan called for widening backward once Stage 1 plateaued, extending the horizon by a single decision each round. The claim was that this controlled ramp would avoid the credit-assignment problems of training on full games from the start — untested, since Stage 1 never ratcheted (lem/OVERVIEW.md @ a8bccfa).

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
