---
title: "Belief probes: does Zeb's field net encode the teacher's exact posterior over who holds what? (Lem, the paper's G1)"
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: complete
---

G1 of the paper's work order ([`lem/paper/README.md`](../../lem/paper/README.md) § 5; [[lem-the-teacher]]).
Pre-registered here before any probe was fit. The code is `lem/probe/belief.py` (CC0; tests in
`lem/tests/test_probe.py`).

## Question

The teacher's belief at a position is exact: for every live tile the viewer does not hold and
every other seat, P(seat holds tile | the viewer's information), under uniform worlds that respect
the voids the seats have shown ([[silas]], `SilasBelief.uniform_resolving` on that tile). Does the
student, the shipped field net, encode it? And does it encode more of it than its own inputs
state linearly?

## Pre-registration

- **Net and sample.** `zeb/net/margin_d250k_h128x64.fnet`; the audit's sample
  ([[lem-field-net-audit-2026-10-15]]: 434,535 held-out positions, pip trumps, seat 0 bids 30).
- **Targets.** For each position, `P[rel, slot]` for the three other seats (left, partner, right)
  and every live hidden tile in the net's canonical slot; undefined elsewhere. The **counting
  prior** `B[rel, slot]` = that seat's remaining hand size over the hidden tiles (voids ignored).
  The probe's target is the deviation `P − B`: what the voids tell the teacher. The entries the
  voids move (|P − B| > 0) are reported separately ("moved").
- **Split.** Deals shuffled once (seed 20261016) and halved; 20,000 fit rows from one half,
  20,000 eval rows from the other, so no deal is on both sides.
- **Probes.** Ridge regression (λ = 1, standardized features, one ridge per output entry over
  the rows where it is defined) from four representations: the 343 raw input bits `x`; the
  first hidden layer `a1` (128); the second `a2` (64); `a1+a2`. Scored on the eval rows by R² over
  all defined entries and over the moved entries, plus mean absolute error against the prior's.
- **Chance.** The same probe fit on row-shuffled targets, scored the same way.
- **Predictions, before measuring** (the desk's priors).
  1. The raw inputs state most of it: R² of `x` on the moved entries ≥ 0.8 (the void bits are
     inputs, and the posterior under uniform worlds is close to linear in them).
  2. `a1` recovers nearly what `x` does (within 0.1 of R²); `a2` recovers less (at least 0.1
     below `a1`): the belief is read off the inputs, not built deeper.
  3. No hidden representation beats `x` by more than 0.02 R²: the student adds no belief
     computation beyond what its inputs state linearly.
  4. Shuffled R² within ±0.02 of 0.
- **What a surprise would mean.** `a2` ≥ `x` on the moved entries would say the net computes
  something about ownership its inputs do not state linearly, and would send G3 (causal tests)
  at those units first.
- **Scope.** Play-only evidence (the record and the shown voids). The auction carries nothing
  in these pours (seat 0 always bids 30); the auction-evidence posterior (`Belief::hand_law`,
  #134) is a later arm, and the field's own posterior (reading plays) another.
- **Cost.** Exact targets were priced at 0.7 ms a position (one 1-tile resolution per live
  hidden tile). The run cost about 50 ms a position, which was not Silas but the probe kit's
  sample reader decompressing a whole column on every attribute read (fixed the same day: a
  cached column, 0.45 ms a position after). The first single-slot run hit its cap before any
  target was written, so the targets were computed in 16 slot-sized parts and the fit run
  separately. The sizes, split and probes are unchanged; this note is the only deviation from
  the sealed text.

## Results

Run 2026-10-16 as registered: 20,000 fit and 20,000 eval rows from disjoint deals; 724,221 defined
eval entries, of which 370,025 (0.511) are moved by the voids. The counting prior's mean absolute
error is 0.0532 over all entries and 0.1041 over the moved ones. Evidence (targets, `belief.json`,
`belief.md`, runner, logs, `SHA256SUMS`): `lem/belief-probe-2026-10-16/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10).

| representation | width | R² all | R² moved | MAE all | MAE moved | R² shuffled (all / moved) |
|---|---:|---:|---:|---:|---:|---:|
| raw inputs `x` | 343 | **0.777** | **0.791** | 0.0260 | 0.0420 | −0.023 / −0.014 |
| first hidden layer `a1` | 128 | 0.198 | 0.241 | 0.0568 | 0.0888 | −0.013 / −0.005 |
| second hidden layer `a2` | 64 | 0.021 | 0.035 | 0.0579 | 0.1016 | −0.008 / −0.003 |
| `a1+a2` | 192 | 0.199 | 0.244 | 0.0572 | 0.0887 | −0.023 / −0.012 |

**The predictions, graded.**

| | predicted | measured | |
|---|---|---|---|
| 1. the inputs state most of it | R² of `x` on moved ≥ 0.8 | 0.791 | just under; the posterior is nearly, not quite, linear in the inputs |
| 2. `a1` within 0.1 of `x`; `a2` ≥ 0.1 below `a1` | yes / yes | `a1` is 0.55 below `x`; `a2` is 0.21 below `a1` | first half wrong, second half right |
| 3. no hidden layer beats `x` by > 0.02 | yes | none comes near | holds |
| 4. shuffled within ±0.02 of 0 | yes | −0.023 to −0.003 | holds (one at the edge) |

**Reading.** The teacher's belief is in the student's inputs (the void bits and hand sizes carry
0.79 of what the voids tell the posterior, linearly) and almost not in its hidden layers: the first
layer keeps a quarter of it, the second nearly none, and the second layer's error against the
prior is the prior's own. The field net plays without a readable belief over who holds what. It
uses the voids (its play changes with them), but it compresses them into what the move needs,
not into P(seat holds tile). For the paper this is the answer key at work: a measurement that
would read as "the net represents the posterior" without the input baseline reads, with it, as
"the posterior is in the inputs and the net discards most of it on the way to a move".

Caveats: linear probes on ReLU activations; a policy net trained on move advantages, not on
beliefs, has no reason to keep the posterior readable; play-only evidence (the auction carries
nothing in these pours); one net, one seed. The next arms: the field's own posterior (reading
plays, `draw=posterior`) as the target; Lem's nets, which have a value head and more width; and a
nonlinear probe as an upper bound, registered before it runs.

## Links

[[lem-the-teacher]] · [[lem-field-net-audit-2026-10-15]] · [[silas]] · [[scenario-player]] ·
[[strategy-fusion-fence]]
