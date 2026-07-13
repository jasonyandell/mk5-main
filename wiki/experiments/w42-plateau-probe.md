---
title: The plateau probe — data starvation, not structure
kind: experiment
status: complete
task_id: gh-33
first_seen: 2026-07-06
last_updated: 2026-07-11
---

# w42-plateau-probe

## Summary

[[w42-jud-v0]] left one open question load-bearing for [[jud]] v1: the value-native
bidder's self-play loop dissolved the #26 over-bidder but converged *at* parity with the
hand-tuned `net:wp` baseline (definitive 512-game same-seed: −0.01 [−0.28, +0.25]). Was
that plateau **structural** — the [[pimc]] price of hidden information, reachable only by
belief-state search — or **data starvation**, a tiny MLP head trained on ~12k hands whose
selection error shrinks with on-policy data volume?

The probe ran the question as a registered prediction, and the prediction was wrong. The
plateau was data starvation. Scaling on-policy self-play ~3× per round (rounds 5–8, 1000
games/round vs 300) carried `margin:wp` past parity: **head_8 beats `net:wp` +0.38
[+0.09, +0.67]** at the reserved seed and **+0.42 [+0.12, +0.72]** at a fresh seed, both
287/512 (56.1%). This is the **first learned bidder to beat the hand-tuned champion on
marks**. A registered extension (rounds 9–12) then confirmed saturation: the curve
flattens at **≈ +0.3–0.4 marks/game** at this net capacity. The binding constraint at v0's
scale was data; the next constraint is capacity or mechanism, which is jud v1's cue.

## The registered prediction (pre-run, on GitHub #33)

Stated before any round-5 data existed, per [[w42-jud-v0]]'s interpretation that the
residual ~60–67% offense-share stickiness was priced *correctly* by the head (the bidder
no longer punished for it):

**The plateau is structural.** Scaling on-policy data will NOT break parity.

- Prediction: head_8's 512-game same-seed A/B lands within ±0.30 of zero (CI includes 0,
  point estimate not significantly positive). Offense share stays in the 60–70% band.
  Notrump share stays < 12% (no artifact revival).
- Falsifier: a point estimate ≥ +0.30 with CI excluding zero ⇒ calibration headroom was
  the answer, and v1's search-leaves premise needs re-weighting toward "more data first."

Either outcome sharpens #33: PASS (structural) ⇒ the parity-breaking edge must come from
belief-state search; FAIL ⇒ scale data before building search.

## The probe

Extend the cumulative-recipe loop (`scratch/jud-v0/loop/run_loop.py`), rounds 5–8, with
the data rate ~3×:

- SP_GAMES 300 → 1000 per round (fast batching, merged `2c652f0`).
- AB_GAMES 128 → 256 per round (tighter per-round CIs, ±~0.4).
- Recipe, seeds, driver otherwise unchanged; rounds 1–4 cached; canonical seed 7000000
  stays out of training.
- Final measurement: 512-game A/B head_8 vs `net:wp` at seed 7000000 (same protocol as the
  head_4 definitive number), plus a replication at fresh seed 9000000.

## Result — FALSIFIED: the plateau was data starvation

The falsifier fired: a point estimate ≥ +0.30 with CI excluding zero, **twice** at 512
games, plus round 7's independent 256-game exclusion.

| measurement | margin (A−B) | 95% CI | wins |
|---|---|---|---|
| head_8 @ seed 7000000 (reserved, 512g) | **+0.38** | [+0.09, +0.67] | 287/512 (56.1%) |
| head_8 @ seed 9000000 (fresh, 512g) | **+0.42** | [+0.12, +0.72] | 287/512 (56.1%) |

Offense 64.4%, made 63–64%, notrump ~5%, calibration max gap 0.029 — all inside the
predicted bands *except the one that mattered*: 3× data/round broke what 4 rounds at 300
games could not. **`margin:wp`(head_8) is the first learned bidder to beat `net:wp` on
marks** (`champion/margin_net_r8.pt`, CLI `margin:wp,model=champion/margin_net_r8.pt`).

The registered prior was wrong — stated plainly. A falsified registered prediction, run to
its falsifier, is the system working: the pre-committed threshold made the reversal
unambiguous instead of a matter of interpretation. The consequence for #33 is exactly what
the registered falsifier said it should be — v1 re-weights toward data scaling before
search. The premise that the parity-breaking edge must live in belief-state search
re-weights: at v0's scale, data was the binding constraint, not the hidden-information
price.

## Extension — saturation confirmed at ≈ +0.3–0.4

A second prediction, registered pre-run: rounds 9–12, same recipe at 1000 SP games. The
r5–r8 sequence (+0.12, +0.02, +0.40, +0.32) reads as noise around a fixed edge, not
growth, so **saturation** — head_12's 512-game measurements land in [+0.2, +0.6],
indistinguishable from head_8. Falsifier: head_12 ≥ +0.7 (data still binding → keep
scaling) or ≤ +0.1 (r7/r8 a transient — overfitting the opponent region).

**CONFIRMED.**

| measurement | margin (A−B) | 95% CI |
|---|---|---|
| head_12 @ seed 7000000 (512g) | **+0.21** | [−0.08, +0.47] |
| head_12 @ seed 9000000 (512g) | **+0.37** | [+0.09, +0.65] |

Both inside the registered [+0.2, +0.6] band and statistically indistinguishable from
head_8 (+0.38/+0.42). The data-scaling curve flattens at **≈ +0.3–0.4 marks/game** over
`net:wp` at this net capacity (a tiny MLP) and recipe. head_8 remains the best-measured
head (both its 512-game CIs exclude zero; head_12's reserved-seed CI does not). The next
binding constraint is capacity or mechanism, not rounds.

## The full round table (r0–r12)

Rounds 0–4 are the [[w42-jud-v0]] loop (128-game A/Bs, 300 SP games); rounds 5–12 are the
probe and its extension (256-game A/Bs, 1000 SP games). All `lens:ev` play both sides,
`nopass`, cumulative recipe. Source: `champion/evidence/jud_v0/loop_metrics.json`.

| round | AB margin | 95% CI | offense | made | notrump share | games |
|---|---|---|---|---|---|---|
| 0 | −1.44 | [−1.88, −0.95] | 75.2% | 49.9% | 47.5% | 128 |
| 1 | −0.31 | [−0.90, +0.26] | 70.8% | 55.8% | 1.5% | 128 |
| 2 | +0.24 | [−0.36, +0.84] | 66.9% | 61.3% | 3.2% | 128 |
| 3 | +0.24 | [−0.33, +0.83] | 59.1% | 64.0% | 5.7% | 128 |
| 4 | +0.22 | [−0.29, +0.74] | 66.9% | 60.1% | 7.5% | 128 |
| 5 | +0.12 | [−0.30, +0.52] | 63.9% | 60.8% | 6.7% | 256 |
| 6 | +0.02 | [−0.37, +0.41] | 63.8% | 63.1% | 7.4% | 256 |
| 7 | **+0.40** | **[+0.02, +0.80]** | 58.2% | 64.9% | 6.9% | 256 |
| 8 | +0.32 | [−0.10, +0.73] | 64.4% | 61.7% | 4.7% | 256 |
| 9 | +0.16 | [−0.27, +0.58] | 60.4% | 64.8% | 7.5% | 256 |
| 10 | +0.31 | [−0.13, +0.74] | 64.5% | 63.2% | 7.2% | 256 |
| 11 | +0.31 | [−0.08, +0.71] | 61.8% | 64.9% | 10.2% | 256 |
| 12 | **+0.42** | **[+0.02, +0.81]** | 62.9% | 64.0% | 9.2% | 256 |

Rounds 7 and 12 independently exclude zero on their 256-game A/Bs (bold). Calibration
tightens monotonically across the run (ECE @ p30 0.053 → 0.010; max pred−emp gap 0.035 →
0.012) — the head keeps getting better-calibrated as data accumulates, consistent with the
data-starvation reading.

## Interpretation

1. **Data was the binding constraint, not structure.** The v0 write-up read the plateau as
   possibly the [[pimc]] price of hidden information — a wall belief-search would have to
   climb. It was not. A tiny MLP simply had not seen enough on-policy hands to price the
   selection region correctly; 3× data/round moved it past `net:wp`. The winner's-curse-on-
   selection channel [[w42-jud-v0]] identified is real, and *more on-policy data* is what
   closes it — the same mechanism the loop already used, just fed harder.
2. **The registered prior was wrong, and that is the finding.** Recording a falsified
   prediction plainly is worth more than a confirmed one: the pre-committed falsifier
   (+0.30, CI excludes zero) made the reversal unarguable. The [[jud]] program moves on the
   evidence, not the prior.
3. **Saturation re-locates the frontier.** At this capacity the curve flattens ≈ +0.3–0.4.
   The remaining edge is not in more rounds of the same recipe — it is in capacity or
   mechanism. That is jud v1's premise: ONE net for bid + play (play-history-conditioned
   `V_realized`, 1-ply argmax-EV play replacing E[Q] n=10 at runtime), then the same
   self-play-loop method on the full stack. In build as of this writing.

## Links

- [[w42-jud-v0]] — the v0 write-up this extends; its open question 1 (what breaks the
  parity plateau?) is answered here: data starvation, not structure
- [[jud]] — the unified belief-conditioned core; v0 bidder now beats `net:wp`, v1 in build
- [[champion]] — the ladder; `margin:wp`(head_8) is the first learned bidder to beat `net:wp`
- [[rank-vs-price]] — the mechanism the value-native bidder validates; now beats, not ties
- [[pimc]] — strategy fusion; the structural residual this probe refuted at v0's scale
- [[w42-champion-selfplay-fixed-point]] — #26, the over-bidder the loop dissolves

## Evidence

Primary artifact: `champion/evidence/jud_v0/plateau_probe.md` — the probe's run log and
numbers, colocated with the other jud_v0 reports cited by [[w42-jud-v0]].
