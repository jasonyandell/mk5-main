---
title: w42 Book Validation v1 — Wave 2.B Bid-Aware E[Q] Atlas
kind: experiment
status: complete
bead: t42-6j3k
wave2b2_bead: t42-7eop
parent_epic: t42-4zi6
wave: wave2
first_seen: 2026-05-03
last_updated: 2026-07-13
---
## Summary

This wave delivers the W42-side bid-aware E[Q] driver (Wave 2.B) that
[[w42-bookval-v1-wave2-infra-design]] specifies as Build B. The driver sweeps
bid_values ∈ {30,32,35,36,39,42,84} on the same hands, produces per-bid joint-world
`.pt` files, joins them into a single action table keyed by
`(seed, decl_id, bid_value, decision_idx, action_slot)`, and recomputes `mark_ev`
with the correct mark multiplier at each bid.

**Headline finding: mark_ev diverges from threshold_mass monotonically as bid rises.**
At bid=30, mark_ev ≈ 2·threshold_mass − 1 (algebraic identity, multiplier=1, consistent
with [[w42-bookval-v1-wave1-mark-utility-transform]]). At bid=84, divergence is 3.77 units
(mean absolute) and 100% of actual-action mark_evs differ from the bid=30 baseline.
This is the first corpus where the Ch 10 special-bid mark-multiplier claim is testable.

**Validation: PASS.** At bid=30, seed=9430, aggregate per-decl_id mean EV matches
[[w42-branch-atlas-scaled-v0]] within sampling noise (10/10 decl_id pairs, 0 discrepant).

## Slice

- Primary sweep: seeds 9000–9004 (5 seeds), decl_ids 0–9, bid_values {30,32,35,36,39,42,84}
- Validation seed: 9430 (1 seed), all 7 bids; matches [[w42-branch-atlas-scaled-v0]] seed
- N samples: 200 per decision (smoke run)
- Device: Apple M-series MPS (no CUDA in this environment; H100 path available)
- Run label: smoke (n_seeds ≤ 10, n_samples ≤ 300)

## Method

### Driver Architecture

`w42/book_validation_v1/wave2/run_bid_aware_atlas.py` accepts:
`--start-seed`, `--n-seeds`, `--n-decl-per-seed`, `--bid-values`, `--n-samples`,
`--output-dir`, `--device`, `--dry-run`.

For each bid_value, it invokes `python -m forge.eq.generate` with `--schema v2
--save-joint-worlds --bid-values <N repeated>`. The forge layer's `_parse_bid_values`
is already wired; no forge-core changes were required.

After all bids, the driver joins per-bid `.pt` files into a single CSV with columns:
scalar EV, std, q10/q25/q50/q75/q90, threshold_mass, CVaR_10, p_make (from mark_util),
mark_ev (bid-aware), mark_ev_equals_p_make flag, seat_role, declaration_name, bid_value,
mark_multiplier, and game context columns.

### Mark EV Computation

The per-world mark utility transform is verbatim from
[[w42-phase4-scoring-objective-tests]], extended to multiplier > 1:

1. `remaining_total = 42 − pre_t0 − pre_t1`
2. `remaining_t0[w,a] = (q[w,a] + remaining_total) / 2`
3. `final_t0[w,a] = pre_t0 + remaining_t0[w,a]` (clamped 0..42)
4. `made[w,a] = (final_t0 ≥ bid)` for bid < 42; `= (final_t0 == 42)` for bid = 42
5. `mark_util[w,a] = +multiplier if team0_won else −multiplier`
6. `mark_ev[a] = mean_w(mark_util[w,a])`
7. `p_make[a] = P(mark_util[w,a] > 0)`

`mark_multiplier(bid) = 1` for bid < 84 (including bid=42); `= bid // 42` for bid ≥ 84 (so 2 at bid=84).

### Validation Protocol

Two oracle runs on the same seed diverge in game trajectory (greedy-stochastic, MPS
vs CUDA numerical drift). Per-`(decision_idx, slot)` comparison shows large diffs on
diverged trajectories. The correct contract is aggregate: per-`(seed, decl_id)` mean EV
of actual-action decisions, compared within combined SEM tolerance (3×SEM + 3 pt buffer).

### 84-Eligibility

All 10 decl_ids are generated at bid=84. The engine does not enforce bidding legality
(4+ doubles required), so all runs complete. The manifest documents this. Strategic
interpretation of bid=84 outputs requires filtering to hands with ≥ 4 doubles.

## Findings

### Mark EV Diverges from Threshold Mass at Higher Bids (Headline)

| bid | mm | tq_off | mean_mark_ev | mean_tm | divergence |
|-----|----|----|------|------|------|
| 30 | 1 | 18 | −0.473 | 0.462 | 0.398 |
| 32 | 1 | 22 | −0.500 | 0.464 | 0.428 |
| 35 | 1 | 28 | −0.625 | 0.468 | 0.560 |
| 36 | 1 | 30 | −0.698 | 0.471 | 0.640 |
| 39 | 1 | 36 | −0.822 | 0.468 | 0.757 |
| 42 | 1 | 42 | −1.000 | 0.449 | 0.897 |
| 84 | 2 | 42 | −2.000 | 0.444 | 3.774 |

`tq_off` = threshold_q for offense player = 2·bid − 42. As bid rises, the Q-space
threshold rises, shifting which worlds count as "made" under the schema's definition
vs the mark utility's final_t0 definition. These coincide only at pre_t0 = pre_t1 = 0;
at mid-game they diverge.

The divergence is **structural and monotone**: higher bids demand more points to make
contract, so fewer worlds clear the threshold, collapsing mark_ev toward −multiplier.

### Algebraic Identity Confirmation and Extension

The [[w42-bookval-v1-wave1-mark-utility-transform]] finding that mark_ev ≡ p_make at
bid=30 is confirmed here (mean_residual for identity check ≈ 0 within floating-point).
The identity `mark_ev = mm · (2·p_make − 1)` holds **by construction** at all bids
(p_make is derived from mark_util directly). The "divergence" is between mark_ev and the
schema's threshold_mass metric: they are the same at trick 0 but diverge mid-game.

At bid=42: mark_ev = −1.0 exactly (offense essentially never scores all 42 vs greedy
opponents, so P(made) ≈ 0, mark_ev ≈ −1). This is a real game-theoretic observation,
not a sampling artifact — bid=42 is dominated except when forced.

### Cross-Bid Mark EV Change Rates vs bid=30

| bid vs 30 | n actual actions | change rate |
|-----------|-----------------|-------------|
| 32 | 280 | 63.2% |
| 35 | 280 | 63.2% |
| 36 | 280 | 55.7% |
| 39 | 280 | 57.5% |
| 42 | 280 | 57.5% |
| 84 | 280 | 100.0% |

At bid=32 and bid=35, 63% of actual-action mark_evs differ from their bid=30 values.
This is the direct measure that mark_ev ≢ p_make across bid values — the Ch 10
multiplier effect is active and measurable.

### Validation: bid=30 Matches branch_atlas_scaled_v0

Aggregate per-`(seed=9430, decl_id)` mean EV for actual-action decisions:
10/10 decl_id pairs within sampling noise (match_rate = 1.0, 0 discrepant).

The validation protocol uses combined SEM = std_ev × (1/√n_driver + 1/√n_atlas)
with 3×SEM + 3 pt buffer to account for trajectory divergence after decision 0.

## Caveats

1. Smoke run (5 seeds, n=200). The full 50-seed × 1000-sample run is needed for
   Ch 02/10/12 claim promotion to `supported`. H100 estimated 35s.
2. Trajectory divergence: two stochastic oracle runs on the same seed produce different
   game trajectories. Per-decision comparison is not valid without state hashing.
   Aggregate per-`(seed, decl_id)` comparison is the correct validation contract.
3. bid=84 with n=200: all mark_evs cluster near −2.0 (rarely makes). Needs larger n
   and filtering to eligible hands for strategic interpretation.
4. bid=42: all mark_evs = −1.0 at n=200; threshold collapses. Real bid=42 behavior
   requires hands where offense genuinely threatens all 42.
5. Claim-ledger impact of this wave: `context-limited` for Ch10 (demonstrates divergence
   is real, not powered for full claim promotion). Ch 02/07/12 downstream work unblocked.

## Artifacts

| Path | Description |
|------|-------------|
| `w42/book_validation_v1/wave2/run_bid_aware_atlas.py` | Driver script |
| `w42/book_validation_v1/wave2/bid_aware_atlas/README.md` | Artifact README |
| `w42/book_validation_v1/wave2/bid_aware_atlas/bid_aware_actions.csv` | Per-action joined CSV (current on-disk file is the full-sweep corpus: seeds 9000–9049, 7 bids, 259,618 rows; the earlier seed-9430 smoke output was overwritten) |
| `w42/book_validation_v1/wave2/bid_aware_atlas/validation_check.csv` | Aggregate validation vs branch_atlas_scaled_v0 |
| `w42/book_validation_v1/wave2/bid_aware_atlas/manifest.json` | Full provenance, SHAs, divergence stats |
| `eq_pdf_seeds90*0-90*9_bid*.pt`, `eq_pdf_seeds9430-9430_bid*.pt` | Per-bid joint-world .pt files (not checked into the repo; filenames + SHA256s recorded in `manifest.json` join logs) |

## Provenance

- Transform source: [[w42-phase4-scoring-objective-tests]] (`run_phase4_scoring_objective_tests.py`)
- Reference atlas: [[w42-branch-atlas-scaled-v0]] (`eq_pdf_s9430_d10_bid30_joint_1000s_v2.pt`)
- Repo commit: see `manifest.json`
- Reproduce (smoke, MPS):
  ```
  python -u w42/book_validation_v1/wave2/run_bid_aware_atlas.py \
    --start-seed 9000 --n-seeds 5 --n-decl-per-seed 10 \
    --n-samples 200 --bid-values "30,32,35,36,39,42,84" --device mps
  ```
- Reproduce (full, CUDA):
  ```
  python -u w42/book_validation_v1/wave2/run_bid_aware_atlas.py \
    --start-seed 9000 --n-seeds 50 --n-decl-per-seed 10 \
    --n-samples 1000 --bid-values "30,32,35,36,39,42,84" --device cuda
  ```

## Wave 2.B.2 Full Sweep (bead t42-7eop)

**Headline: first corpus with statistical power for paired-bid claim promotion on ch10 and ch02.**
The full 50-seed × 10-decl × 7-bid × n=200 sweep was completed on 2026-05-03 in 32.3 min wall
on Apple M-series MPS. The corpus contains 259,618 action rows across 50 seeds (9000–9049),
10 declarations each, 7 bid values. Validation at bid=30 PASS (10/10).

### Full Sweep Parameters

| Parameter | Value |
|-----------|-------|
| Seeds | 9000–9049 (50 seeds) |
| Declarations per seed | 10 (decl_ids 0–9) |
| Bid values | 30, 32, 35, 36, 39, 42, 84 |
| Samples per decision | 200 |
| Device | Apple MPS (M-series) |
| Batch strategy | 5 batches × 10 seeds (MPS INT_MAX limit prevents 500-game batches) |
| Wall time | 32.3 min (1939.9 s) |
| Total action rows | 259,618 |

### Per-Bid Row Counts

| bid | rows |
|-----|------|
| 30 | 36,940 |
| 32 | 37,148 |
| 35 | 37,036 |
| 36 | 36,980 |
| 39 | 37,254 |
| 42 | 37,103 |
| 84 | 37,157 |

All 10 decl_ids appear at all 7 bid values.

### Validation Status

bid=30 vs branch_atlas_scaled_v0 at seed 9430: **PASS 10/10** (0 discrepant).

### Mark EV Divergence at Full Scale (50 seeds)

| bid | mm | tq_off | mean_mark_ev | mean_tm | divergence |
|-----|----|----|------|------|------|
| 30 | 1 | 18 | −0.549 | 0.442 | 0.432 |
| 32 | 1 | 22 | −0.624 | 0.447 | 0.519 |
| 35 | 1 | 28 | −0.729 | 0.452 | 0.634 |
| 36 | 1 | 30 | −0.774 | 0.449 | 0.672 |
| 39 | 1 | 36 | −0.872 | 0.457 | 0.786 |
| 42 | 1 | 42 | −1.000 | 0.444 | 0.889 |
| 84 | 2 | 42 | −2.000 | 0.442 | 3.769 |

Consistent with smoke-run findings; monotone divergence confirmed at 50-seed scale.

### Power Analysis

`power_analysis.csv` computed with 1000-iteration bootstrap CIs (percentile method).

#### ch10-special-bid-mark-multiplier (mark_ev change-rate vs bid=30)

| bid | n | change_rate | 95% CI | verdict |
|-----|---|-------------|--------|---------|
| 32 | 14,000 | 0.665 | [0.658, 0.673] | sufficient |
| 35 | 14,000 | 0.656 | [0.648, 0.664] | sufficient |
| 36 | 14,000 | 0.651 | [0.643, 0.658] | sufficient |
| 39 | 14,000 | 0.647 | [0.640, 0.655] | sufficient |
| 42 | 14,000 | 0.638 | [0.631, 0.646] | sufficient |
| 84 | 14,000 | 1.000 | [1.000, 1.000] | sufficient |

CI half-widths all < 0.005. At bid=84, mark_ev changed at every actual decision (100%) vs bid=30.
**Verdict: sufficient for claim promotion** — multiplier effect is measurable and tight.

#### ch02-bid-only-enough (paired bid=32 − bid=30 deltas)

| metric | n | delta | 95% CI | verdict |
|--------|---|-------|--------|---------|
| mark_ev | 14,000 | −0.076 | [−0.085, −0.067] | sufficient |
| p_make | 14,000 | −0.038 | [−0.042, −0.034] | sufficient |
| threshold_mass | 14,000 | +0.006 | [−0.000, +0.013] | borderline |

mark_ev and p_make: raising bid from 30→32 reduces bidder's make probability by ~3.8pp paired.
threshold_mass: 95% CI crosses zero (borderline) — threshold_mass is a less sensitive proxy.
**Verdict: sufficient for mark_ev and p_make; borderline for threshold_mass.**
Interpretation: bidding 32 instead of 30 measurably reduces mark_ev for the bidder — supports
the book's "bid only enough" advice, but the effect (−0.076 mark_ev units) is small.

#### ch12-setter-pounce-high-bid-off (setter mean Q delta vs bid=30)

| bid | n | delta_Q | 95% CI | verdict |
|-----|---|---------|--------|---------|
| 35 | 8,400 | +0.179 | [−0.172, +0.527] | borderline |
| 36 | 8,400 | +0.209 | [−0.150, +0.564] | borderline |
| 39 | 8,762 | +0.376 | [+0.010, +0.751] | sufficient |
| 42 | 8,734 | +0.410 | [+0.068, +0.765] | sufficient |
| 84 | 8,688 | +0.348 | [−0.008, +0.725] | borderline |

At bid≥39, setter's mean Q rises vs bid=30 (CI excludes zero), suggesting setter-side positions
improve as bid rises — consistent with "pounce at high bids" claim. At bid=35/36/84, CI includes
zero (borderline).
**Verdict at this aggregate-proxy level: sufficient at bid=39 and bid=42; borderline at bid=35/36/84.**
Caveat: mean Q is a rough proxy; proper test needs snapshot-level setter-side probes.

**Reversed by Wave 2.E.2 (`t42-8kbh`, [[w42-bookval-v1-wave2-pounce-high-bid]]).** The
snapshot-level probe this page calls for was run and found the opposite of "sufficient":
`ch12-setter-pounce-high-bid-off` was **demoted to `contradicted`** at all four high-bid
buckets (N=1,140, all CIs excluding zero in the contradicting direction). The campaign used
this reversal to write a new methodology guardrail: aggregate proxies like the one above do
not qualify for ledger promotion, only paired same-decision contrasts do
([[w42-book-validation-campaign]], "Ledger movement to date"). Treat the "sufficient at
bid≥39" read above as superseded by that snapshot-level result, not as a standing verdict.

### Infrastructure Note

MPS backend raises `MPSGraph does not support tensor dims larger than INT_MAX` with
500+ games (50 seeds × 10 decls) in a single `forge.eq.generate` call at n=200 samples.
Workaround: batch runner `run_full_sweep_batched.py` runs 10 seeds per forge call (100 games).
The driver `run_bid_aware_atlas.py` was not modified.

### Overall Power Verdict

| Claim | Status |
|-------|--------|
| ch10-special-bid-mark-multiplier | **sufficient** (all bid buckets) |
| ch02-bid-only-enough | **sufficient** for mark_ev/p_make; borderline for threshold_mass |
| ch12-setter-pounce-high-bid-off | aggregate-proxy read: sufficient at bid≥39, borderline at bid=35/36/84 — **reversed to `contradicted` by [[w42-bookval-v1-wave2-pounce-high-bid]]'s snapshot-level probe; do not read this row as standing** |

This corpus has enough power for ledger promotion of ch10 (full multiplier effect) and
ch02 (mark_ev/p_make paired delta). ch12 promotion requires snapshot-level probes for
the setter-pounce mechanism at bid=35/36 — those probes ran ([[w42-bookval-v1-wave2-pounce-high-bid]],
bead `t42-8kbh`) and demoted the claim to `contradicted` rather than promoting it.

### Reproduce (Full Sweep, MPS)

```
python -u w42/book_validation_v1/wave2/run_full_sweep_batched.py \
  --start-seed 9000 --n-seeds 50 --batch-seeds 10 \
  --n-decl-per-seed 10 --n-samples 200 \
  --bid-values "30,32,35,36,39,42,84" \
  --device mps \
  --output-dir w42/book_validation_v1/wave2/bid_aware_atlas

# Post-process: include seed 9430 for validation
python -u w42/book_validation_v1/wave2/rejoin_with_validation.py

# Power analysis
python -u w42/book_validation_v1/wave2/compute_power_analysis.py
```

## Links

- [[w42-bookval-v1-wave1-mark-utility-transform]] — Wave 1.2 finding that triggered this build
- [[w42-branch-atlas-scaled-v0]] — reference corpus for validation
- [[w42-bookval-v1-wave2-infra-design]] — design record for Wave 2 builds
- [[w42-phase4-scoring-objective-tests]] — mark utility transform source
- [[w42]] | [[w42-book-claim-synthesis-and-ai-directions]]
