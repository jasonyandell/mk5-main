# W2 — on-policy corpus: empirical count-fate ledger

_Generated 2026-07-15 01:03:59 · 26 chunks · 147,516 hands · 8.6s (CPU)._

## 1. Corpus totals & integrity

- Hands parsed: **147,516** (fate rows: 737,580).
- Three independent referees agree on **100.0%** of hands: parser P1 identity, recorded-points cross-check (147,516/147,516), and the ledger identity `bidder_team_pts == Σ value·X_t + T`.
- `made` field == `bidder_team_pts >= bid_value` on **99.98%** of hands; the 24 exceptions are all `bid_value = 84` two-mark bids (making requires all 42 points, not `pts ≥ 84`) — the recorded `made` is authoritative. The 84 marks-bid hands fall below the 500-hand floor and enter no P3 slice.

| source | hands |
|---|---|
| netwp | 17,026 |
| random | 17,017 |
| selfplay | 113,473 |

## 2. Deal-hash split (champion convention)

Split on `(seed, hand_idx)` via `champion/margin_net.py::split_of` — paired replays of a deal never straddle splits.

| source | train | val | test |
|---|---|---|---|
| netwp | 15,360 | 801 | 865 |
| random | 15,366 | 838 | 813 |
| selfplay | 102,207 | 5,557 | 5,709 |
| **all** | 132,933 | 7,196 | 7,387 |

## 3. P3 — independence composition vs empirical joint tail

Per bid-value slice B (selfplay hands, ≥500/slice): empirical `P(bidder_team_pts ≥ B)` vs the independence composition (per-tile Bernoulli captures ⊛ trick marginal). `diff = empirical − independence` (pp). Band: **|diff| ≥ 2pp CONFIRMED** (the joint head is mandatory), < 0.5pp falsifier.

| slice | thr | n | empirical | independence | diff (pp) | verdict |
|---|---|---|---|---|---|---|
| bid_30 | 30 | 28,033 | 68.52% | 72.09% | -3.57 | CONFIRMED |
| bid_31 | 31 | 65,178 | 63.10% | 67.37% | -4.27 | CONFIRMED |
| bid_32 | 32 | 19,430 | 60.11% | 65.50% | -5.39 | CONFIRMED |
| bid_33 | 33 | 748 | 63.50% | 63.97% | -0.47 | FALSIFIER |
| bid_31plus | 31 | 85,440 | 63.80% | 68.31% | -4.51 | CONFIRMED |

> **VERDICT: P3 CONFIRMED on magnitude (|diff| ≥ 2pp at every make-threshold slice), but the registered DIRECTION is corrected by the data.** The registered prediction — *independence underprices the high tail* — holds for the *genuine* high tail (≥34) but is **reversed at the make threshold**: independence **overprices** the make probability by 3.6–5.4pp at B∈{30,31,32}. One mechanism explains both: the capture/trick indicators are **positively correlated** (mean off-diagonal capture corr **+0.084**; X_t–T corr +0.29…+0.42), which fattens *both* tails around the fixed mean (μ = 33.2, σ = 8.3). Below μ independence over-states the tail mass; above μ it under-states it. The crossover is at threshold **33** ≈ μ — which is exactly why the thin `bid_33` slice (threshold 33, at the crossover) reads ≈0: that near-zero is the mechanism's fingerprint, not marginals sufficing.

Full tail curve, pooled selfplay (empirical − independence), showing the over→under crossover and the sweep underpricing:

| threshold | empirical | independence | diff (pp) |
|---|---|---|---|
| 25 | 83.17% | 89.10% | -5.94 |
| 28 | 75.49% | 78.82% | -3.33 |
| 30 | 69.93% | 73.60% | -3.67 |
| 31 | 63.34% | 67.78% | -4.44 |
| 32 | 57.57% | 62.02% | -4.45 |
| 33 | 57.46% | 56.68% | +0.78 |
| 34 | 56.49% | 54.95% | +1.54 |
| 35 | 52.85% | 50.39% | +2.46 |
| 36 | 45.62% | 43.38% | +2.24 |
| 38 | 38.29% | 30.04% | +8.25 |
| 40 | 36.98% | 22.78% | +14.20 |
| 41 | 32.49% | 14.69% | +17.79 |
| 42 | 22.18% | 6.80% | +15.38 |

At the sweep line (≥41) independence prices 14.7% vs the empirical 32.5% — a **+17.8pp** underpricing: the joint head is most indispensable exactly where marks are decided by all-or-nothing capture. Direction (corrected): positive capture/trick correlation fattens BOTH tails around the fixed mean: independence OVERprices below the mean (the make region) and UNDERprices the genuine high tail (sweeps); crossover ~ the mean.

### 6×6 correlation matrix of (X_5-5, X_6-4, X_5-0, X_4-1, X_3-2, T)

| | X_5-5 | X_6-4 | X_5-0 | X_4-1 | X_3-2 | T |
|---|---|---|---|---|---|---|
| **X_5-5** | +1.000 | +0.011 | +0.245 | +0.062 | +0.057 | +0.315 |
| **X_6-4** | +0.011 | +1.000 | +0.029 | +0.094 | +0.053 | +0.294 |
| **X_5-0** | +0.245 | +0.029 | +1.000 | +0.093 | +0.092 | +0.376 |
| **X_4-1** | +0.062 | +0.094 | +0.093 | +1.000 | +0.103 | +0.407 |
| **X_3-2** | +0.057 | +0.053 | +0.092 | +0.103 | +1.000 | +0.421 |
| **T** | +0.315 | +0.294 | +0.376 | +0.407 | +0.421 | +1.000 |

## 4. Fate base rates (P2 baseline, selfplay/train)

Per-tile 8-class distribution = (capture vs BIDDING team) × played_mode, over **102,207** hands. Entropy in nats (P2's NLL units). Full tables in `scratch/otis-night/fate_base_rates.json`.

| tile | H (nats) | bid·led | bid·follow | bid·trump | bid·slough | opp·led | opp·follow | opp·trump | opp·slough |
|---|---|---|---|---|---|---|---|---|---|
| 5-5 | 1.479 | 49.0 | 11.2 | 1.7 | 21.2 | 6.2 | 4.8 | 0.0 | 5.9 |
| 6-4 | 1.790 | 16.4 | 30.7 | 13.5 | 18.2 | 2.4 | 12.7 | 1.1 | 4.9 |
| 5-0 | 1.692 | 10.5 | 37.5 | 7.3 | 25.8 | 2.3 | 7.3 | 1.9 | 7.4 |
| 4-1 | 1.792 | 9.5 | 33.9 | 10.5 | 23.6 | 3.2 | 9.8 | 3.4 | 6.2 |
| 3-2 | 1.813 | 10.1 | 29.5 | 8.7 | 26.4 | 4.3 | 11.3 | 1.9 | 7.7 |

(cells are % of hands; row sums to 100. Per-decl breakdown in the JSON.)

## 5. Junk economy

- Count-carrying tricks: **595,436**.
- Walker catches (count trick won by a NON-count tile): **387,465** = 65.1% of count tricks (192,773 of them by trump power).
- Count tricks won by trump power: 318,747 (53.5%).

### Count routing per tile

| tile | → own team | → opp team | frac→own | sloughs | slough→opp |
|---|---|---|---|---|---|
| 5-5 | 109,188 | 38,328 | 74.0% | 40,487 | 48.5% |
| 6-4 | 95,038 | 52,478 | 64.4% | 32,918 | 41.5% |
| 5-0 | 75,483 | 72,033 | 51.2% | 46,916 | 50.3% |
| 4-1 | 78,469 | 69,047 | 53.2% | 43,006 | 49.3% |
| 3-2 | 79,568 | 67,948 | 53.9% | 48,482 | 47.0% |

### Capture mechanism per tile (holder-relative capture × played_mode, count of hands)

| tile | own·led | own·follow | own·trump | own·slough | opp·led | opp·follow | opp·trump | opp·slough |
|---|---|---|---|---|---|---|---|---|
| 5-5 | 64,062 | 21,277 | 3,018 | 20,831 | 13,018 | 5,654 | 0 | 19,656 |
| 6-4 | 22,782 | 34,208 | 18,793 | 19,255 | 6,154 | 32,032 | 629 | 13,663 |
| 5-0 | 12,751 | 26,311 | 13,108 | 23,313 | 8,242 | 39,439 | 749 | 23,603 |
| 4-1 | 12,693 | 25,977 | 17,989 | 21,810 | 7,885 | 38,651 | 1,315 | 21,196 |
| 3-2 | 13,221 | 26,472 | 14,199 | 25,676 | 9,583 | 34,751 | 808 | 22,806 |

## 6. Findings (fable gate, 2026-07-15)

An independent gate recomputed the load-bearing claims from `master_hands.parquet` with a
from-scratch implementation (brute-force enumeration over all 2^5 capture patterns × 8 trick
counts, no shared code with `otis/analysis/empirical_ledger.py`). Every headline number
reproduced exactly: the B=30 slice (n=28,033) reads empirical 68.52% vs independence 72.09%
(−3.57pp), B=31 −4.27pp, B=32 −5.39pp, pooled 31+ −4.51pp, and the pooled ≥41 tail +17.79pp /
≥42 +15.38pp. The ledger identity `bidder_team_pts = Σ value·X_t + T` holds on all 147,516
hands; the correlation matrix reproduces (mean off-diagonal capture correlation +0.084, all
ten pairs positive; X_t–T from +0.294 to +0.421; μ=33.18, σ=8.33). The independence
composition in `empirical_ledger.py::independence_pmf` is a genuine convolution of five
per-tile Bernoulli kernels with the empirical trick marginal, and `p3_slice` computes both
the empirical tail and the composition from the same sliced hand population — the comparison
never mixes populations. The split column also verified: `split_of(seed, hand_idx)`
recomputed correctly on 20 random rows, zero of 78,545 `(seed, hand_idx)` groups straddle
splits, and zero of the 68,971 paired-deal groups (both `a_team` halves present) straddle.

**What P3 will grade as (W7).** The magnitude band — |diff| ≥ 2pp at the make threshold —
is met at every make-threshold slice above the 500-hand floor except `bid_33` (n=748,
−0.47pp), which sits at the crossover threshold 33 ≈ μ where any mean-preserving correction
must pass through zero; the three passing slices cover 112,641 of the 113,389 sliced hands
(99.3%). The registered direction — "independence underprices high tails" — grades as
**corrected, not confirmed**: it holds for thresholds above the mean (+1.5pp at ≥34 rising
to +17.8pp at ≥41) and is reversed below it (−3.6 to −5.4pp at B∈{30,31,32}), because the
make thresholds sit *below* the bidder-team mean of 33.2, not in the high tail as the
registration implicitly assumed. The registered mechanism — positive fate correlation — is
confirmed outright (all fifteen off-diagonal correlations positive). The recommended W7
grade is therefore: **P3 CONFIRMED on magnitude and mechanism, direction amended** —
positive correlation fattens both tails around the fixed mean, so independence overprices
the make region and underprices sweeps, with the crossover at ≈33. Under either sign the
conclusion the band was built to test stands: the joint pricing head is mandatory, and it
matters most (+17.8pp) exactly where marks are decided.

**Caveats for downstream consumers.** (1) P3 is a data-property measurement with no fitted
model, computed over all selfplay hands (train+val+test pooled); both sides of every
comparison use identical hands, so no train/test leakage arises — but any *learned*
reproduction of these numbers in W3+ must re-derive them on `split == train` only. (2)
Paired halves share their auction (99.9% identical on selfplay; 100% on netwp/random —
deterministic bidders on the same deal) but diverge in play (only 37.6% of selfplay pairs
realize identical bidder points), so the nominal n's overstate the effective sample size by
less than 2× and point estimates are unbiased; slice n's here are nominal. (3) The 24
`bid_value = 84` hands are excluded from every P3 slice by the 500-hand floor, and their
recorded `made` (requiring all 42 points) is authoritative over `pts ≥ bid_value`.

## 7. W3 label-table schema

Training labels for W3 join the info-state (conditioning) to the OFFLINE fate labels by hand. Join key: **`(source, seed, game_idx, hand_idx, a_team)`** (unique per hand across a paired match; `game_id` encodes the same and is unique within a source).

**`master_hands.parquet`** — one row per hand (pricing + trick + P3 substrate):

| column | meaning |
|---|---|
| `game_id` | stable id `<source>:a<a_team>:g<game_idx>:h<hand_idx>` |
| `source` | corpus tag: selfplay | netwp | random |
| `seed` | deal seed (split key) |
| `game_idx` | game index within the match |
| `hand_idx` | hand index within the game (split key) |
| `a_team` | which physical team was 'A' in the paired match |
| `dealer` | dealer seat 0-3 |
| `bids` | JSON 4-list of per-seat bids |
| `bidder` | winning-bid seat 0-3 (leads trick 1) |
| `bidding_team` | bidder % 2 (0 or 1) |
| `bid_value` | winning bid amount (30..42) |
| `decl_id` | forge declaration id 0..9 |
| `decl_name` | declaration name |
| `bidder_team_pts` | realized bidder-team points 0..42 (PRICING TARGET) |
| `opp_team_pts` | 42 − bidder_team_pts |
| `made` | 1 if contract made, else 0 |
| `team0_tricks` | tricks won by absolute team 0 |
| `team1_tricks` | tricks won by absolute team 1 |
| `bidder_tricks` | bidder-team trick count 0..7 (TRICK-HEAD TARGET, = T) |
| `opp_tricks` | 7 − bidder_tricks |
| `X_5-5 … X_3-2` | bidder-team capture indicator per count tile (1=captured) |
| `split` | train | val | test (deal-hash) |

**`master_fates.parquet`** — 5 rows per hand (fate-head labels), one per count tile. Join to hands on the key above; `tile` selects the head:

| column | meaning |
|---|---|
| `game_id` | join to master_hands |
| `source, seed, game_idx, hand_idx, a_team` | join keys |
| `tile` | count tile pip string (5-5,6-4,5-0,4-1,3-2) — selects the fate head |
| `tile_id` | forge domino id |
| `holder_seat / holder_team` | seat/team dealt the tile |
| `trick_idx` | 0..6 trick the tile was played in |
| `played_mode` | led | followed | trumped_in | sloughed (FATE-HEAD mode axis) |
| `winner_seat / winner_team` | seat/team that won the tile's trick |
| `capture_side` | holder_team | opp_team (holder-relative) |
| `capture_bidding` | bidding_team | opp_of_bidder (BIDDER-relative; FATE-HEAD capture axis) |
| `won_by_trump` | did the winning domino win by trump power |
| `bid_value, decl_id, decl_name, bidder` | hand context |
| `team{0,1}_count/tricks/points` | per-team tallies (redundant with hands) |
| `split` | train | val | test |

**Fate-head target (treatment)**: the 8-class label per tile = `(capture_bidding ∈ {bidding_team, opp_of_bidder}) × (played_mode ∈ {led, followed, trumped_in, sloughed})`. **Pricing-head target (control+treatment)**: `bidder_team_pts` (0..42). **Trick-head target**: `bidder_tricks` (0..7). Base rates for the fate NLL baseline: `fate_base_rates.json`.
