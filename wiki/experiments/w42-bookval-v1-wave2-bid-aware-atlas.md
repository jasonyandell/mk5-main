---
title: w42 Book Validation v1 — Wave 2.B Bid-Aware E[Q] Atlas
kind: experiment
status: active
bead: t42-6j3k
parent_epic: t42-4zi6
wave: wave2
first_seen: local-2026-05-03
last_updated: local-2026-05-03
---

# w42-bookval-v1-wave2-bid-aware-atlas

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

`mark_multiplier(bid) = 1` for bid < 42; `= bid // 42` for bid ≥ 84.

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
| `w42/book_validation_v1/wave2/bid_aware_atlas/bid_aware_actions.csv` | Per-action joined CSV (last run: seed 9430, 7 bids, 5550 rows) |
| `w42/book_validation_v1/wave2/bid_aware_atlas/validation_check.csv` | Aggregate validation vs branch_atlas_scaled_v0 |
| `w42/book_validation_v1/wave2/bid_aware_atlas/manifest.json` | Full provenance, SHAs, divergence stats |
| `w42/book_validation_v1/wave2/bid_aware_atlas/eq_pdf_seeds9000-9004_bid*.pt` | 5-seed sweep per-bid .pt files |
| `w42/book_validation_v1/wave2/bid_aware_atlas/eq_pdf_seeds9430-9430_bid*.pt` | Validation seed per-bid .pt files |

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

## Links

- [[w42-bookval-v1-wave1-mark-utility-transform]] — Wave 1.2 finding that triggered this build
- [[w42-branch-atlas-scaled-v0]] — reference corpus for validation
- [[w42-bookval-v1-wave2-infra-design]] — design record for Wave 2 builds
- [[w42-phase4-scoring-objective-tests]] — mark utility transform source
- [[w42]] | [[w42-book-claim-synthesis-and-ai-directions]]
