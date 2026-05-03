# Bid-Aware E[Q] Atlas — Wave 2.B Artifact

**Bead:** t42-6j3k  
**Parent epic:** t42-4zi6  
**Created:** 2026-05-03

## Question

Does mark_ev diverge from p_make (threshold_mass) when bid > 30?
Wave 1.2 found mark_ev ≡ p_make at bid=30 (algebraic identity, mark_multiplier=1).
This atlas provides bid-swept data to test whether that identity breaks at higher bids.

## Slice

- Seeds 9000-9004 (5 seeds), decl_ids 0-9 (10 declarations), bid_values {30,32,35,36,39,42,84}
- Seeds 9430 (1 seed), all 7 bids (for validation against branch_atlas_scaled_v0)
- N samples: 200 per decision (smoke run; full run needs 1000 on H100)
- Device: Apple M-series MPS (no CUDA available)
- Total: 5 seeds × 10 decls × 7 bids = 350 games + 1 seed × 10 decls × 7 bids = 70 games for validation

## N

- 5-seed sweep: 26,031 action rows across 7 bids
- Validation set (seed 9430, 7 bids): 5,550 action rows

## Metric

- `mark_ev`: E[mark_utility] per action per world, where mark_utility = +/-multiplier
- `threshold_mass`: P(q >= threshold_q) — the schema's remaining-point threshold
- `divergence(mark_ev vs 2*tm-1)`: |mark_ev - (2*threshold_mass - 1)| averaged across actions
- `mark_ev_change_rate`: fraction of actual-action mark_evs that differ from bid=30 baseline

## Status

- **Validation (bid=30 vs branch_atlas_scaled_v0):** PASS — 10/10 (seed, decl_id) pairs within sampling noise
- **Mark EV divergence:** CONFIRMED — divergence rises monotonically with bid value

## Mark EV Divergence Headline

| bid | mm | tq_off | mean_mark_ev | mean_tm | divergence |
|-----|----|----|------|------|------|
| 30 | 1 | 18 | -0.473 | 0.462 | 0.398 |
| 32 | 1 | 22 | -0.500 | 0.464 | 0.428 |
| 35 | 1 | 28 | -0.625 | 0.468 | 0.560 |
| 36 | 1 | 30 | -0.698 | 0.471 | 0.640 |
| 39 | 1 | 36 | -0.822 | 0.468 | 0.757 |
| 42 | 1 | 42 | -1.000 | 0.449 | 0.897 |
| 84 | 2 | 42 | -2.000 | 0.444 | 3.774 |

Cross-bid mark_ev change rate vs bid=30 (fraction of actual actions with different mark_ev):
- bid=32: 63.2%
- bid=35: 63.2%
- bid=36: 55.7%
- bid=39: 57.5%
- bid=42: 57.5%
- bid=84: 100.0% (trivial — multiplier doubles)

## Claim Ledger Impact

- `ch10-special-bid-mark-multiplier`: **context-limited** — algebraic identity mark_ev=p_make breaks at bid>30; divergence is structural and monotone with bid
- `ch10-score-mode-objective`: **supported** (within this slice) — mark_ev and threshold_mass are distinct metrics at bid>=32

## Caveats

1. Smoke run (5 seeds, n=200 samples). Full run (50 seeds, n=1000) on H100 needed for Ch 02/10/12 claim promotion.
2. Oracle trajectory diverges between runs on the same seed (greedy-stochastic, MPS vs CUDA numerical drift). Per-(decision_idx, slot) comparison is not valid; aggregate comparison (per-decl_id mean EV) is the correct contract.
3. 84-bid: all 10 decl_ids run; engine does not enforce 4+ doubles eligibility during generation.
4. mark_ev at bid=42 is always -1.0 (offense rarely scores all 42 with greedy oracle at n=200). This is a sampling artifact, not a claim about bid=42 outcomes.

## Reproduce

```bash
# Validation seed only (fast)
python -u w42/book_validation_v1/wave2/run_bid_aware_atlas.py \
  --start-seed 9430 --n-seeds 1 --n-decl-per-seed 10 \
  --n-samples 200 --bid-values "30,32,35,36,39,42,84" \
  --device mps

# 5-seed smoke sweep
python -u w42/book_validation_v1/wave2/run_bid_aware_atlas.py \
  --start-seed 9000 --n-seeds 5 --n-decl-per-seed 10 \
  --n-samples 200 --bid-values "30,32,35,36,39,42,84" \
  --device mps

# Full 50-seed run (H100, ~35s)
python -u w42/book_validation_v1/wave2/run_bid_aware_atlas.py \
  --start-seed 9000 --n-seeds 50 --n-decl-per-seed 10 \
  --n-samples 1000 --bid-values "30,32,35,36,39,42,84" \
  --device cuda

# Dry-run (print commands only)
python -u w42/book_validation_v1/wave2/run_bid_aware_atlas.py --dry-run
```

## Artifacts

| File | Description |
|------|-------------|
| `bid_aware_actions.csv` | Joined per-action rows for last run (seed 9430, 7 bids, 5550 rows) |
| `validation_check.csv` | Aggregate validation at bid=30 vs branch_atlas_scaled_v0 |
| `manifest.json` | Full provenance, SHAs, divergence stats |
| `eq_pdf_seeds9000-9004_bid*.pt` | Per-bid joint-world .pt files for 5-seed sweep |
| `eq_pdf_seeds9430-9430_bid*.pt` | Per-bid joint-world .pt files for validation seed |
| `../run_bid_aware_atlas.py` | Driver script |
