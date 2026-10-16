---
title: "Fidelity to the teacher by concept region: where Zeb's field net disagrees with Walt (Lem, the paper's G7)"
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: active
---

G7 of the paper's work order ([`lem/paper/README.md`](../../lem/paper/README.md); [[lem-the-teacher]]).
Pre-registered here before any regret was grouped. Code: `lem/probe/fidelity.py` (CC0).

## Question

The student's grade is its mean regret under the teacher's plan (the teacher's best make
probability minus that of the student's move; `scenario.field_train` reports 0.35 regret in
margin points and 0.0084 in make probability on the validation rows of the 250k net). Where does
that regret sit? In which of the audit's concept regions, at which trick positions, with how
many legal moves, and at how decisive a position (the teacher's best-minus-second gap)?

## Pre-registration

- **Sample and labels.** The audit's 434,535 held-out rows and its 48 Silas labels
  ([[lem-field-net-audit-2026-10-15]]); the net's pick recomputed from its scores
  (ties to the lowest tile id, as the net plays); the teacher's `q` per legal move read back from
  the five pours (`fidelity-collect`, one `cpu-slot` call per pour, rows checked against the
  sample's actions and Walt's choice).
- **Regret.** `max_a q(a) − q(pick)` in the mover's units (make probability; a defender's values
  flipped); zero is agreement with the teacher's best move.
- **Groups.** Each concept holding and absent; the number of legal moves (2 to 7); the
  teacher's gap in four bands (0, (0, 0.02], (0.02, 0.1], > 0.1). Means with a deal-clustered
  bootstrap interval (2,000 resamples, seed 20261016). Also the share of all regret carried by
  the top decile of positions.
- **Predictions, before measuring** (the desk's priors).
  1. Regret is concentrated: the top decile of positions carries more than half of it.
  2. The three concepts with the highest regret when they hold include `can-take-it` and
     `count-on-trick`; the lowest include `lead` and `forced`-like regions (`forced` itself has
     regret 0 by construction and is excluded from the ranking).
  3. Regret rises with the number of legal moves (monotone from 2 to 7) and is higher in the
     widest gap band than in the narrowest non-zero band.
- **Cost.** Five slot calls to collect `q` (about 90k rows each), one to group. CPU minutes.

## Results

Not yet run.

## Links

[[lem-the-teacher]] · [[lem-field-net-audit-2026-10-15]] · [[scenario-player]] · [[strategy-fusion-fence]]
