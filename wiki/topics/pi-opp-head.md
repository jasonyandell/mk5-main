---
title: π_opp Head
kind: topic
first_seen: 2026-04-22
last_updated: 2026-07-13
status: retired
---

## What it is

`PiOppHead` is a lightweight adapter head trained to predict what each opponent seat will
play given the current game state. It takes the frozen trunk's `state_emb` plus a 3-way
seat embedding (L-opp / partner / R-opp relative to the decision player) and outputs
logits over the 7 legal domino slots. 1,879 parameters; the trunk is fully frozen during
training (93859a0).

## Why it was needed

[[lamir1]] rollout variants simulate 1-3 opponent plays between the decision and the
leaf. Earlier modes used rotated π_me as a proxy for opponent play — a poor fit because
π_me was trained from the decision player's frame, not a seat-relative opponent frame
(~55% oracle top-1 as π_opp). The dedicated head reaches **68.57% oracle top-1**
(MORNING4_STATUS @ b42669a).

The motivation was that better opponent simulation might fix the lamir1 look-ahead
failure. The result confirmed it did not — `lamir1-piopp` performed slightly worse than
rotated-π_me rollout; the bottleneck is the distilled leaf evaluator, not opp simulation
quality (b42669a).

## Where the results live

- [[gus-pi-opp-training]] — training setup (Schema v2 corpus fields, legal-masked CE
  against `oracle_softmax_per_seat`, 3 pairs per batch item), the IEEE `0 × (−∞) = NaN`
  masking bug and fix, and the 68.57% result. Adapter: `gus/adapters/v3_10k_piopp.pt`.
- [[gus-lamir1-piopp]] — the head in rollout: the full 8-mode ladder where every
  look-ahead variant loses to direct π_me. See [[lamir1-ceiling]].
- [[joint-world-tensor]] — Schema v2 fields the head trains on.

## Retired

The head is a real trained artifact (68.57% oracle top-1) but belongs to the abandoned
[[lamir1]] rollout track. [[jud]] does not use it — the project's decision-time
look-ahead question was closed by taking pivot option 4 (self-play, no CFR+), not by
fixing the rollout this head was built to serve. See [[lamir1-ceiling]] for the full
pivot record.

## Links

[[lamir1]] [[lamir1-ceiling]] [[gus-pi-opp-training]] [[gus-lamir1-piopp]] [[student-distillation]] [[gus]] [[jud]]
