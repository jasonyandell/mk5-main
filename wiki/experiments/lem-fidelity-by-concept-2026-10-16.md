---
title: "Fidelity to the teacher by concept region: where Zeb's field net disagrees with Walt (Lem, the paper's G7)"
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: complete
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

Run 2026-10-16 as registered on all 434,535 rows (the teacher's values read back from the five
pours and checked row by row). Mean regret **0.0084 [0.0083, 0.0086]** make probability (the
trainer's own validation grade is 0.0084); the net picks the teacher's best move on 0.754 of
rows; the **top decile of positions carries 0.825 of all regret**. Evidence
(`q/`, `fidelity.json`, `fidelity.md`, runner, logs, `SHA256SUMS`): `lem/fidelity-by-concept-2026-10-16/`
in [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10).

**Where the regret sits** (mean when the concept holds, 95% deal bootstrap; the overall mean is 0.0084).

| highest | regret | | lowest | regret |
|---|---|---|---|---|
| masters-2plus | 0.0106 [0.0095, 0.0117] | | set / made (nothing left to decide) | 0.0000 |
| **lead** | 0.0102 [0.0100, 0.0103] | | holds-boss-of-led | 0.0052 [0.0048, 0.0056] |
| to-make-le-10 | 0.0101 [0.0097, 0.0105] | | safe-trick | 0.0054 [0.0051, 0.0056] |
| is-bidder | 0.0100 [0.0098, 0.0103] | | to-set-le-5 | 0.0055 [0.0053, 0.0058] |
| holds-five | 0.0098 [0.0096, 0.0099] | | ten-on-trick | 0.0062 [0.0059, 0.0065] |
| can-ruff | 0.0096 [0.0093, 0.0099] | | trump-led | 0.0065 [0.0064, 0.0067] |
| | | | last; count-on-trick; winner-cannot-be-beaten | 0.0066 each |

By legal moves: 2 → 0.0051, 3 → 0.0104, 4 → **0.0115**, 5 → 0.0109, 6 → 0.0095, 7 → 0.0093
(a peak at four, not a monotone rise). By the teacher's best-minus-second gap: 0 (the teacher
indifferent at the top, 40.6% of rows) → 0.0008; (0, 0.02] → 0.0094; (0.02, 0.1] → 0.0159;
> 0.1 → **0.0214 [0.0206, 0.0222]**.

**The predictions, graded.**

| | predicted | measured | |
|---|---|---|---|
| 1. the top decile carries > 0.5 of regret | yes | 0.825 | holds |
| 2. highest with can-take-it and count-on-trick; lowest with lead | | can-take-it 0.0086 (the mean), count-on-trick 0.0066 (below it); **lead is among the highest** at 0.0102 | wrong, and backwards on the lead |
| 3. monotone in legal moves; higher at the widest gap than the narrowest non-zero | yes / yes | peak at four, then falls; 0.0214 vs 0.0094 | first half wrong, second half right |

**Reading.** The student is faithful where the position has already told it what to do: the
trick is decided (`safe-trick`, `winner-cannot-be-beaten`), count is on the table, trump was led,
the bid is already made or set, or it holds the boss of the suit led. It strays where the
decision is open: **on lead** (the one position with no trick to read), as the bidder with the
contract close (`to-make-le-10`), holding fives (the five-count tiles are the ones a lead can
give away), and when it can ruff. Regret also grows with how much the teacher cares (the gap),
so the net's errors are not spread as noise over indifferent positions but sit on the decisive
ones. The lead is the field net's weak region; the audit found the lead to be its best-labelled
region (32 clean `lead` poles), so a legible unit is not an accurate one. For Lem's nets the
lead and the near-contract bidder are where fidelity to the teacher should be measured first.

Caveats: regret is to the teacher's finite-budget plan, not to an optimum; make-probability units
under one contract (seat 0 bids 30); the concept groups overlap (a position holds many), so the
table is descriptive, not a decomposition.

## Links

[[lem-the-teacher]] · [[lem-field-net-audit-2026-10-15]] · [[scenario-player]] · [[strategy-fusion-fence]]
