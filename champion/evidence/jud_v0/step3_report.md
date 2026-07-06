# jud v0 — Step 3: ValueBidder + A/B vs net:wp

Commit: `92ee436` (branch forge, unpushed). Building on `2e4ea25` (trained margin_net).

## Phase 1 — code + tests (DONE)

New: `champion/value_bidder.py` (`ValueBidder`, `load_margin_net`), `champion/test_value_bidder.py`.
Changed: `arena/cli.py` — registered `margin[:wp[,pass[<q>]]]` in `parse_bidder`.

Design (as speced):
- Prices contracts with the realized-value head via `MarginNet.pmake_table` at the
  hypothetical completed auction; `canonical_auction` (level-blind, later-seat-masked)
  is applied INSIDE pmake_table — never bypassed.
- Min positive-utility legal bid (rung #21 convention, NOT #31 max), declaring the
  argmax-exceedance decl at that threshold, cached from bid time for `declare()`.
- Default utility `MarksToSeven` (mirrors `net:wp`). NO `pmake_scale`. **NO static prefilter**
  (approved by lead): the value head IS the evaluator, so hopeless hands pass on their own
  utility rather than a pre-learned trump/doubles gate; `net:wp` keeps its own prefilter as
  part of what `net:wp` is. The A/B compares each bidder's full shipping policy.

Tests (12, CPU, 0.4s): featurization byte-consistency (decision-time BidContext view ==
corpus `featurize_snapshot` view — hand-made + over a real match); min-positive convention
(monster → bid 30, not 84); argmax-decl; pass-when-hopeless; p=0.5 boundary → pass; declare
cache + forced-open cold path; CLI registry; real-head round trip; full 2-game CPU match.
Gate: `pytest champion/ arena/ -q` → **112 passed**.

## Phase 2 — 128-game paired A/B (DONE)

`margin:wp+lens:ev` (A) vs `net:wp+lens:ev` (B), pooled 128 games, n-samples 10, device mps,
ckpt `domino-large-817k-valuehead-acc97.8-qgap0.07`, base-seed 7000000. **Wall 107s** (single
pooled run, no split needed). Artifacts: `ab_margin_vs_net/{summary.json,per_hand.csv,per_game.csv}`,
`ab_snaps.json` (1438 snapshots).

### Headline — REGISTERED PREDICTION 1: **MISS**

> Prediction 1: `margin:wp` reaches ≥ parity with `net:wp` (CI no longer excludes zero in
> net:wp's favor). **Result: margin:wp LOSES.**

| metric | value |
|---|---|
| A (margin:wp) game wins | **36 / 128 (28.1%)** — halves 32.8% / 23.4% |
| mean mark margin (A−B) | **−1.44 / game** |
| 95% CI | **[−1.88, −0.95]** — excludes zero, entirely negative |
| mean hand point margin (A−B) | **+5.66 / hand** |
| hands/game | 11.2 |

The CI sits fully on net:wp's side, so this is a clean miss, not a wash. **But** −1.44 is a
markedly smaller loss than the belief bidder's −2.2 to −3.4 against the same baseline —
margin:wp is the strongest learned bidder-vs-net:wp measured so far, just not at parity.

### The diagnostic story — margin:wp is an OVER-BIDDER (legible, not a bug)

| | A = margin:wp | B = net:wp |
|---|---|---|
| auction offense share | **75.2%** | 24.8% |
| contracts | 1081 | 357 |
| made-rate | **49.9%** | 75.4% |
| mean bid | 30.9 | 30.8 |

margin:wp wins three of every four auctions and makes only half its contracts; net:wp bids
selectively and makes three quarters. Bid histogram `{30: 426, 31: 792, 32: 205, 33: 15}`: the
792 bids at **31** are margin:wp *stealing* net:wp's 30-opens by raising a tick (30 is always
legal from scratch, so 31 can only be an over-raise). Decl histogram is notrump-heavy
(`notrump 522` ≈ 36% of all contracts) — margin_net's argmax declaration lands on notrump far
more than net:wp does (flagged for Step 4; see below).

**The signature: A wins the point margin (+5.66/hand) but loses the mark margin (−1.44/game).**
margin:wp captures more points on average yet loses, because a *set* forfeits a full mark to
the opponent regardless of how close it was (28/42 and 41/42 both pay the same). Bidding thin
on 75% of hands and getting set half the time hands net:wp a steady drip of set-marks that the
point margin doesn't see.

### Why — and why prediction 2 passing is consistent with this

The on-policy made-rate (0.499 @ bid 30.9) matches margin_net's held-out calibration
(exceedance ≈0.52 @ 30, prediction 2 PASSED). So this is **not** a gross miscalibration — the
head is roughly calibrated at the bid-30 margin. Two things convert a calibrated head into an
over-bidder under a `p > 0.5` threshold:

1. **Thin-thresholding.** `MarksToSeven` bids iff p_make > 0.5. A head that (correctly) rates
   ~0.5 on a huge fraction of hands then bids all of them — but a p≈0.5 contract is a coin flip
   for a full mark, and doing that 75% of the time against a selective opponent loses the marks
   race even at even point-EV. net:wp survives on the SAME utility only because its evaluator
   (double-dummy P(make)) crosses 0.5 on far fewer hands, so it is selective by accident of a
   conservative evaluator, not by a better rule. **Made-rate is selectivity, not calibration.**

2. **On-policy selection / winner's curse.** margin:wp bids exactly the hands margin_net rates
   highest, concentrating on the head's positive estimation errors. The round-0 corpus is
   net:wp self-play + random — margin:wp's own aggressive low-bid distribution is off-policy, so
   the head extrapolates optimistically there. This is the #26 failure mode wearing a new coat:
   optimism sneaking back through *selection*, now via the value head's estimation noise instead
   of the oracle's double-dummy assumption.

Both point straight at **Step 4** (the loop): regenerate the corpus with margin:wp self-play and
retrain the head on its own distribution to correct the on-policy optimism — and reconsider the
0.5 threshold / a positive margin. Prediction 3 ("the self-play fixed point stops over-bidding")
is precisely the open question this result sharpens: v0's single round does **not** stop
over-bidding — if anything margin:wp over-bids harder than net:wp — which is exactly the thing
the loop exists to dissolve.

### Follow-up worth a look in Step 4
- **notrump at 36% of contracts.** margin_net's argmax-exceedance declaration is notrump on a
  plurality of hands. Is notrump genuinely the best realized-value declaration under lens:ev, or
  a corpus artifact (notrump over-represented / a head bias)? Cheap to check from `ab_snaps.json`.
- Threshold/margin sweep for margin:wp (e.g. require p > 0.5 + m) as a quick, loop-free probe of
  how much of the gap is thin-thresholding vs on-policy optimism.
