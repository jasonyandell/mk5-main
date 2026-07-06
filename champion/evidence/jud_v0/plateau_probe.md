# jud v0 → v1 bridge: the plateau probe (registered 2026-07-06 ~01:50, pre-run)

## Question (#33, open question 1 from w42-jud-v0)

The loop converged AT net:wp parity (rounds 2–4: +0.24/+0.24/+0.22 on 128-game A/Bs;
definitive 512-game same-seed: −0.01 [−0.28, +0.25]). Is the plateau **structural**
(the PIMC price of hidden information, reachable only by belief-search) or **data
starvation** (the head is a tiny MLP trained on ~12k hands; selection error shrinks
with on-policy data volume)?

## The probe

Extend the cumulative-recipe loop, rounds 5–8, with the data rate ~3×:
- SP_GAMES 300 → 1000 per round (fast batching, merged 2c652f0)
- AB_GAMES 128 → 256 per round (tighter per-round CIs, ±~0.4)
- Recipe, seeds, driver otherwise unchanged; rounds 1–4 cached; canonical seed
  7000000 stays out of training.
- Final measurement: 512-game A/B head_8 vs net:wp at seed 7000000 (same protocol
  as the head_4 definitive number).

## Registered prediction (stated before any round-5 data exists)

**The plateau is structural.** Per the write-up's interpretation (the residual
offense-share stickiness at ~60–67% is priced correctly by the head; the bidder is
no longer punished), scaling on-policy data will NOT break parity:

- P-probe: head_8's 512-game same-seed A/B lands within ±0.30 of zero (i.e., CI
  includes 0, point estimate not significantly positive). Offense share stays in
  the 60–70% band. Notrump share stays < 12% (no artifact revival).
- Falsifier: a point estimate ≥ +0.30 with CI excluding zero would mean calibration
  headroom was the answer and v1's search-leaves premise needs re-weighting toward
  "more data first."

Either outcome sharpens #33: PASS (structural) ⇒ the parity-breaking edge must come
from belief-state search; FAIL ⇒ scale data before building search.

---

# VERDICT (2026-07-06 ~02:00): PREDICTION FALSIFIED — the plateau was data starvation

Rounds 5–8 (1000 SP games/round, 256-game A/Bs): +0.12 [−0.30,+0.52], +0.02 [−0.37,+0.41],
+0.40 [+0.02,+0.80], +0.32 [−0.10,+0.73]. head_8 definitive measurements:

- seed 7000000 (reserved, 512 games): **+0.38 [+0.09, +0.67]**, 287/512 (56.1%)
- seed 9000000 (fresh, 512 games): **+0.42 [+0.12, +0.72]**, 287/512 (56.1%)

The falsifier fired: point estimate ≥ +0.30 with CI excluding zero, twice, plus round 7's
independent 256-game exclusion. **margin:wp(head_8) is the first learned bidder to beat
net:wp on marks.** Offense 64.4%, made 63–64%, notrump ~5%, calibration maxgap 0.029.
3× data/round broke what 4 rounds at 300 games could not. The registered prior (structural
plateau) was WRONG — stated plainly. Consequence for #33: v1 re-weights toward data
scaling before search, exactly as the registered falsifier said it should.

## Extension registered (pre-run): rounds 9–12, the saturation question

Same recipe, 4 more rounds at 1000 SP games. Prediction: **saturation** — the r5–r8
sequence (+0.12, +0.02, +0.40, +0.32) looks like noise around a fixed edge, not growth;
head_12's 512-game same-seed measurements land in [+0.2, +0.6], indistinguishable from
head_8. Falsifier: head_12 ≥ +0.7 (data still the binding constraint → keep scaling) or
≤ +0.1 (r7/r8 were a transient — overfitting to the opponent region).
