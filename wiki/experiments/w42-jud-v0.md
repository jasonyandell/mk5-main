---
title: Jud v0 — the value-native bidder
kind: experiment
status: complete
task_id: gh-32
first_seen: 2026-07-06
last_updated: 2026-07-13
---

# w42-jud-v0

## Summary

**Question:** [[jud]]'s first buildable slice ([[rank-vs-price]], GitHub #32) — replace the
oracle's double-dummy price at the auction with a value trained on **realized** self-play
outcomes (`V_realized`, the `MarginNet` head), leave play on `lens:ev`, and iterate the
whole thing by self-play. Does a value-native bidder dissolve the [[w42-champion-selfplay-fixed-point|#26]]
over-bidder the principled way — with play-side cash backing every bid-side check by
construction — rather than with a tuned `pmake_scale` knob?

**Answer, graded against the three registered predictions:**

- **P2 (calibration): PASS.** `margin_net` is honestly calibrated to realized 4-seat
  outcomes and reproduces the optimism gap — it prices below the double-dummy oracle
  and on the realized curve, exactly as the design requires.
- **P1 (parity at round 0): MISS, clean.** A single-round head loses to `net:wp` by
  −1.44 marks/game [−1.88, −0.95] — the strongest learned bidder measured against that
  baseline, but not at parity. It *wins points* (+5.66/hand) while *losing marks*: a
  legible over-bidder.
- **P3 (the loop dissolves over-bidding): PASS.** Four rounds of on-policy self-play
  carry the margin −1.44 → −0.31 → +0.24 → +0.24 → +0.22 (CI includes zero from round
  2). At the Step-3 seed the final head sits at −0.07 [−0.66, +0.49], statistical
  parity with `net:wp` while winning +7 points/hand.

The value-native move works. The over-bidding was value–policy inconsistency, and clearing
it by depositing only realized outcomes into the value head dissolves it — but the
equilibrium plateaus *at* parity with the hand-tuned baseline, not past it. That plateau is
the finding v1 inherits.

## Addendum — the plateau was data starvation (2026-07-06, [[w42-plateau-probe]])

Open question 1 below ("what breaks the parity plateau?") is answered. The probe registered
the *structural* reading as a falsifiable prediction — scaling on-policy data would NOT break
parity — and **falsified it**. Running the same loop at 3× data/round (rounds 5–8, 1000
self-play games/round vs 300) carried `margin:wp` past `net:wp`: head_8 beats it **+0.38
[+0.09, +0.67]** (reserved seed) and **+0.42 [+0.12, +0.72]** (fresh seed), the first learned
bidder to beat the hand-tuned champion on marks. A registered extension (rounds 9–12)
confirmed saturation at **≈ +0.3–0.4 marks/game**. The plateau was a tiny MLP starved of
on-policy data, not the [[pimc]] price of hidden information; the winner's-curse-on-selection
channel closes with more of exactly the data the loop already used. See [[w42-plateau-probe]]
for the full round table and the registered-prediction ledger.

## Method

- **`V_realized` — `champion/margin_net.py::MarginNet`.** Info-state (own hand + full
  auction + play-so-far) → a 43-class distribution over the hand's final points margin,
  read as a per-declaration threshold-exceedance curve at the root. Categorical
  cross-entropy on the **realized** outcome of arena self-play hands — Monte Carlo
  targets, no bootstrapping (a 7-trick horizon makes TD pointless). This is *not* the
  rejected realized-make-rate scalar: it replaces the evaluator with the full outcome
  distribution, so EV and tails both survive and utilities collapse only at decision time.
- **The bidder — `champion/value_bidder.py::ValueBidder`.** Prices each candidate contract
  by querying `V_realized` at the hypothetical-completed-auction root (the in-distribution
  trick [[w42-champion-selfplay-fixed-point|belief_bidder]] built), reads tail mass at the
  threshold, and collapses through the existing `MarksToSeven` utility. Min positive-utility
  legal bid (rung #21 convention), argmax-exceedance declaration cached from bid time. The
  oracle E[Q] path leaves the pricing loop entirely; `pmake_scale` retires. **Play stays
  `lens:ev`** — per [[rank-vs-price]], play consumes rankings (near-ceiling), value-native
  is load-bearing only where prices are consumed. Registered as CLI bidder
  `margin[:wp[,pass[<q>]]][,model=…]` in `arena/cli.py`.
- **The corpus.** #26's bridge (`arena.cli --emit-snapshots` → stamped `GameRecordGPU`)
  extended to stamp the realized outcome already recorded in `arena/results/per_hand.csv`.
  Round-0 corpus: 20 chunks, 22,001 hands — 12 `net:wp` self-play (on-policy auctions) + 8
  `random`-vs-`random` (coverage), the two regimes giving bimodal realized-outcome support
  across the whole 0–42 range.
- **The loop — `scratch/jud-v0/loop/run_loop.py`.** Round *n* = `margin:wp`(head *n*)
  self-play arena → retrain `V_realized` on the accumulated corpus → next round. 300 self-play
  games + a 128-game paired A/B vs `net:wp` per round, `lens:ev` play both sides so only the
  bidder differs. Convergence read three ways: A/B mark margin, the reliability curve
  (predicted vs realized exceedance), and the notrump calibration gap.

## Results

### P2 — calibration (the gate) — PASS

`margin_net` on a held-out test split (N=568): ECE on P(pts≥30) = **0.046**; predicted vs
empirical exceedance within **≤0.029** at every one of 13 thresholds (max |Δ| @32). It
reproduces the optimism gap in the right direction — at bids 30/36 its exceedance sits
**0.09–0.13 below** the reliable double-dummy oracle, on the realized curve
(`champion/optimism_gap.json`). MAE of the net-predicted curve vs the realized curve over
bids 30–41 is **0.020**; vs the reliable oracle points, **0.118** — the head is **6× closer
to realized than to oracle**, which is exactly what a price-consumer needs and what the
oracle structurally cannot be ([[rank-vs-price]], the calibration law). Evidence:
`champion/evidence/jud_v0/margin_net_eval.json`, `step2b_report.md`.

### P1 — round-0 parity — MISS

`margin:wp` vs `net:wp`, 128 paired games at seed 7000000, `lens:ev` both sides:

| metric | margin:wp (A) | net:wp (B) |
|---|---|---|
| game wins | **36 / 128 (28.1%)** | 92 / 128 |
| mark margin (A−B) | **−1.44 [−1.88, −0.95]** | — |
| hand point margin (A−B) | **+5.66 / hand** | — |
| auction offense share | **75.2%** | 24.8% |
| made-rate | **49.9%** | 75.4% |

The CI sits entirely on `net:wp`'s side — a clean miss, not a wash. But the shape is the
diagnostic: **margin:wp wins the point margin and loses the mark margin.** It wins three of
four auctions and makes half its contracts; a set forfeits a full mark regardless of how
close the points were, so bidding thin on 75% of hands against a selective opponent hands
`net:wp` a steady drip of set-marks the point margin never sees. It is the strongest learned
bidder measured against `net:wp` (the #26 belief bidder lost −2.2 to −3.4), just not at
parity. Evidence: `step3_report.md`, `ab_round0_summary.json`.

### The mechanism — winner's curse, not miscalibration

The on-policy made-rate (0.499 @ bid 30.9) *matches* the head's held-out calibration — so
this is not gross miscalibration (P2 passed). Two things convert a calibrated head into an
over-bidder under a `p > 0.5` threshold:

1. **Made-rate is selectivity, not calibration.** `MarksToSeven` bids iff p_make > 0.5. A
   head that correctly rates ~0.5 on a large fraction of hands then bids all of them — and a
   p≈0.5 contract is a coin flip for a full mark. `net:wp` survives the same utility only
   because its double-dummy evaluator crosses 0.5 on far fewer hands: it is selective by the
   accident of a conservative evaluator, not by a better rule.
2. **Winner's curse on selection.** margin:wp bids exactly the hands `V_realized` rates
   highest, concentrating on the head's positive estimation errors. Round-0's corpus is
   `net:wp` self-play + random, so margin:wp's own aggressive low-bid distribution is
   off-policy and the head extrapolates optimistically there — the #26 failure mode wearing
   a new coat, optimism sneaking back through *selection* rather than through the oracle's
   double-dummy assumption.

The curse compounds at the **declaration** level. margin:wp declared notrump on 36% of
round-0 contracts. Notrump is *calibrated in-distribution* (the corpus's rare notrump
samples are `net:wp`'s selective strong-hand declarations, realized 0.77) but rare and
strong-hand-only, so the head learns a positive notrump offset; argmax-over-declarations then
**exploits** the offset on weak hands, declaring notrump that realizes 0.44 against a
predicted 0.70 (a +0.259 predicted−empirical gap). This is a winner's curse at the decl
level stacked on the bidding-level one. No knob fixes either — both are off-policy
extrapolation, and only the loop retrains them out.

### P3 — the self-play loop — PASS

Four rounds, cumulative-recipe on-policy self-play, `nopass`:

| round | AB margin | 95% CI | offense | made | notrump share | notrump gap@30 |
|---|---|---|---|---|---|---|
| 0 | −1.44 | [−1.88, −0.95] | 75.2% | 49.9% | 47.5% | +0.259 |
| 1 | −0.31 | [−0.90, +0.26] | 70.8% | 55.8% | 1.5% | −0.035 |
| 2 | +0.24 | [−0.36, +0.84] | 66.9% | 61.3% | 3.2% | +0.013 |
| 3 | +0.24 | [−0.33, +0.83] | 59.1% | 64.0% | 5.7% | −0.037 |
| 4 | +0.22 | [−0.29, +0.74] | 66.9% | 60.1% | 7.5% | +0.017 |

The margin improves monotonically to round 2, then stabilizes; the CI includes zero from
round 2 onward. Over-bidding dissolves on made-rate (49.9% → 60–64%) and margin. The
**notrump artifact dies in a single on-policy round** — share 47.5% → 1.5%, gap +0.259 →
|≤0.04| every round thereafter (the head learns notrump-on-weak-hands realizes ~0.45, not
0.73; A1's sub-prediction confirmed at head *and* policy level).

**Canonical same-seed check** (head_4 vs `net:wp`, seed 7000000 — the exact Step-3 seed):
**−0.07 [−0.66, +0.49]**, 65/128 game wins (50.8%), +7.01 points/hand, offense 69.6%.
Round 0 at this seed was −1.44. **Definitive 512-game A/B at the same seed: −0.01/game
[−0.28, +0.25]**, 258/512 game wins (50.4%), +7.23 points/hand — dead statistical parity
with the best hand-tuned baseline while winning points; the Step-3 shape minus the marks
deficit. Evidence: `loop_metrics.json`, `ab_canonical_r4_summary.json`,
`ab_definitive_512_r4_summary.json`, `step4_report.md`.

P3 is the prediction that lands: auction escalation was value–policy inconsistency, and once
the only deposits into `V_realized` are cleared outcomes, the fixed point is not an
over-bidder.

## The recipe-fork finding

A real methodological result, not bookkeeping. Two loop recipes were in play as instructions
crossed:

- **Original recipe** — cumulative self-play + the fixed round-0 *random* coverage chunks
  13–20 only, **dropping** the `net:wp` self-play chunks (single-variable purity).
- **Cumulative recipe** — all 20 round-0 chunks (incl. `net:wp` self-play) + every round's
  self-play + every prior A/B's snapshots.

From the **same** round-1 self-play data (`sp_r1.json`), the original recipe **regressed** to
−2.18 with notrump amplification (share 36% → 55%), while the cumulative recipe went to
−0.31. The plausible mechanism: dropping the `net:wp` self-play chunks removed the
strong-hand/selective coverage that anchors the head where `net:wp`'s contracts actually
live, so the purer head got worse exactly where it mattered. **Data mixing and coverage
anchoring mattered more than single-variable purity** — the head needs to keep seeing the
region of hand-space the opponent bids, or it drifts there. Rescued original-recipe artifacts
are `champion/evidence/jud_v0/origrecipe_r1_*`.

## The A2 denial-bidding finding

The `MarksToSeven` pass baseline (the #27-v2 pass hook, `pass_q_opp`/`pass_make_rate`) was
probed as a candidate over-bidding fix — and is structurally a *denial-bidding* lever, the
wrong tool. Wiring it to `net:wp`'s measured offense make-rate made over-bidding **worse**,
twice:

- on head_0: −1.44 → **−1.82**, offense 75.2% → 87.5%, made 49.9% → 43.5%.
- on the final head_4: −0.07 → **−0.38**, offense 69.6% → 80.5%, made 59.0% → 51.9%.

Mechanically the hook only lowers `wp_pass` (crediting that passing hands the opponent a
~75%-likely mark), so passing looks worse and the bidder fights *harder* to deny — it can
only push toward more bidding, never less. The pass baseline is a real, correct input; it is
an under-bidding/denial lever and cannot reduce over-bidding, which only the loop fixes. The
code and its measured parameterization stay; the hook stays out of the loop.

**Credit:** this sign-catch — a stated pre-run prior, confirmed on both heads — and the
`MarksToSeven` denial-bidding-equilibrium reasoning belong to the **value-bidder subagent**
(a prior session's agent). It caught that the pass baseline was a genuine input error *and* a
backwards fix before either A/B ran. Evidence: `ab_a2_pass_summary.json`,
`ab_a2_rerun_r4_summary.json`.

## Interpretation

1. **The value-native move is validated at the pricing path.** `V_realized` prices honestly
   (P2), and pricing honestly dissolves the over-bidder that a tuned knob only dampened at
   #26 (P3). [[rank-vs-price]] holds up: the auction was where the stack's one systematic
   error term was read cardinally, and a realized-outcome price redeems the tail masses the
   policy actually cashes. `net:wp` is confirmed as what the design said it was — a frozen,
   belief-blind, score-blind slice of `V_realized` — and the design reaches it.
2. **Optimism returns through selection, not just the evaluator.** The round-0 miss shows the
   winner's curse is a *second* optimism channel independent of double-dummy: even a
   realized-calibrated head over-bids on the hands it happens to over-rate, until on-policy
   retraining sees those hands fail. The loop is not optional polish; it is the mechanism that
   closes the selection channel the way calibration closes the evaluator channel.
3. **Parity, not dominance — and that is the honest result.** The loop reaches `net:wp` and
   stops there. Offense share plateaus at ~60–67%, not the predicted 50–55%; the bidder is
   still more aggressive than the selective baseline, just no longer punished for it. The
   value-native bidder wins the argument on *legibility and principle* (its price is backed
   by realized cash, no hand-tuned knob) and ties on marks.

## Open questions

- **What breaks the parity plateau? — ANSWERED (data starvation, [[w42-plateau-probe]]).**
  The [[pimc]]-price reading was registered as a prediction and falsified: scaling on-policy
  data 3× per round (rounds 5–8) carried head_8 past `net:wp` to +0.38/+0.42 marks/game, and
  rounds 9–12 confirmed saturation at ≈ +0.3–0.4. It was calibration headroom in a
  data-starved head, not the hidden-information price. The remaining edge is capacity or
  mechanism, which v1 inherits.
- **The referee gap as an instrument.** Per position, oracle EV − `V_realized` EV = the price
  of hidden information (`champion/optimism_meter.py` is its static ancestor). v0 built the
  head that makes this gap measurable per-position; it is not yet read as a live convergence
  instrument. That is v1's to narrate.
- **v1 — value at the leaves of shallow belief-state search** in play and especially defense,
  where information-set value concentrates (Student-of-Games shape). This is where the
  parity-breaking edge, if it exists, should live.
- **v2 — opponents-in-rollout belief updates**, so signaling gets priced and conventions
  emerge with the referee gap telling the story.

## Provenance

Value-native was the 2026-06-14 session's extension of Fable's written forward design
([[champion-design-review]]), endorsed for the pricing path by a Fable 5 session 2026-07-05
([[jud]] §"The engineering, first cut"). The [[pimc]]-blindness insight that reframes the
belief→marks nulls is Fable's, preserved across the related pages. The A2 sign-catch and
denial-bidding equilibrium reasoning are the value-bidder subagent's.

## Links

- [[jud]] — the unified belief-conditioned core this is the first buildable slice of
- [[rank-vs-price]] — the mechanism: play consumes rankings, bids consume prices; validated at parity here
- [[champion-ladder]] — the rung record (this is rung #32)
- [[w42-champion-selfplay-fixed-point]] — #26, the over-bidder v0 dissolves; the loop machinery reused
- [[champion-design-review]] — Fable's forward design + the two load-bearing caveats
- [[pimc]] — strategy fusion, the residual the parity plateau may reflect
- [[arena]] · [[gus]] · [[forge]] — the measuring stick, the belief head, the solve/oracle
