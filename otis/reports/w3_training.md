# W3 — Otis v0 arm training + P2 calibration

Both arms trained on the full self-play train split, same seed, one-flag discipline.
Exported bidder checkpoints load STRICT through `champion.value_bidder.load_margin_net`
and are bit-exact on the pricing path. P2 fate-calibration band **PASSES** on both the
NLL and top-1 sub-bands with cluster-bootstrap CIs excluding zero.

Run: 2026-07-15, CPU only (`OMP_NUM_THREADS=4`), device=cpu, source=selfplay, seed=42.

## Hyperparameters (identical both arms; mirrored from `champion.margin_net.train`)

| Param | Value |
|---|---|
| Featurization | 91-dim info-state (declarer hand + canonical auction), imported verbatim |
| Trunk | 91 → 256 → 256 (shape/order-identical to `MarginNet`) |
| Pricing head | → 43 bins (our-team total points 0..42) — the only path licensed to price |
| Optimizer | Adam, lr 1e-3 |
| Batch size | 256 |
| Max epochs | 60, early-stop patience 8 on **val pricing CE** |
| Split | 90/5/5 deal-hash (train 102,207 / val 5,557 / test 5,709) |
| Seed | 42 (identical trunk+pricing init + batch order across arms — RNG parity) |

### Fixed auxiliary loss weights (treatment only — registered, never tuned)

```
total = L_price + 0.5·mean(fate CE over 5 tiles) + 0.25·(trick CE) + 0.1·(mean-consistency)
```

- W_FATE = 0.5, W_TRICK = 0.25, W_CONSISTENCY = 0.1
- Fate heads: 5 tiles (5-5, 6-4, 5-0, 4-1, 3-2) × 8-class (captured_by × played_mode)
- Trick head: our-team trick count 0..7
- Consistency: `|E[pricing pdf] − (Σ_t value_t·P(my_team captures t) + E[tricks])|`

## Training curves (summary)

| Arm | Epochs run | Early stop | Best val pricing CE | Wall |
|---|---|---|---|---|
| control   | 15 | e15 (patience 8, best e7)  | **2.6456** | 6.4 s |
| treatment | 25 | e25 (patience 8, best e17) | **2.6407** | 19.7 s |

Full per-epoch curves: `scratch/otis-night/w3_train_control.log`, `scratch/otis-night/w3_train_treatment.log`.

- Control val CE floors early (~e5–e7) around 2.646, then overfits gently.
- Treatment starts higher on `total` (it carries the aux terms) but its **pricing** val CE
  floors at the same level (2.6407) — the auxiliary organs impose **no pricing tax** on the
  consumer path. Pricing matched (control 2.6456 vs treatment 2.6407; treatment marginally
  lower), consistent with P4's "no tax" expectation.

## Export / consumer-path integrity

`otis.export_bidder` lifts `net.{0,2,4}` (trunk + pricing) into a vanilla
`{"model_state","feature_dim"}` MarginNet checkpoint; STRICT load through
`champion.value_bidder.load_margin_net`.

| Arm | strict_load | bit_exact | max_abs_diff (100 rows) |
|---|---|---|---|
| control   | True | True | 0.000e+00 |
| treatment | True | True | 0.000e+00 |

Bidder ckpts: `otis/models/otis_v0_control.pt`, `otis/models/otis_v0_treatment.pt`.
Raw OtisNet ckpts (W4 loop + debugging): `scratch/otis-night/otis_raw_control.pt`,
`scratch/otis-night/otis_raw_treatment.pt` (also mirrored as `otis/models/*_net.pt`).

## Pricing reliability (val, both arms)

| Arm | val NLL (=CE) | ECE P(pts≥30) |
|---|---|---|
| control   | 2.6456 | 0.0187 |
| treatment | 2.6407 | 0.0226 |

Both well-calibrated on the bid-relevant P(pts≥30) exceedance.

## P2 — fate calibration (held-out TEST, treatment model)

Base rates recomputed on **train only**. Cluster bootstrap over deal groups
`(seed, hand_idx)` — 3,053 clusters, 1000 resamples. Test N = 5,709 rows.

| Metric | Point | 95% CI | Band | Verdict |
|---|---|---|---|---|
| Fate NLL improvement vs base rate | **0.4775 nats** | [0.4669, 0.4862] | ≥ 0.15 nats + CI excl 0 | **PASS** |
| Top-1 accuracy delta | **+17.02 pp** | [16.21, 17.79] | ≥ 8 pp + CI excl 0 | **PASS** |

Overall: model NLL 1.2332 vs base 1.7107; model top-1 0.530 vs base 0.360.
Falsifier (≤ 0.03 nats) not triggered. Formal grading is W7.

### Per-tile fate NLL improvement (nats) + top-1 delta (pp)

| Tile | model NLL | base NLL | Δ NLL (nats) | model top-1 | base top-1 | Δ top-1 (pp) |
|---|---|---|---|---|---|---|
| 5-5 | 1.099 | 1.455 | 0.356 | 0.590 | 0.500 | +9.00 |
| 6-4 | 1.241 | 1.777 | 0.536 | 0.514 | 0.314 | +19.99 |
| 5-0 | 1.249 | 1.726 | 0.477 | 0.524 | 0.351 | +17.34 |
| 4-1 | 1.250 | 1.785 | 0.535 | 0.532 | 0.341 | +19.16 |
| 3-2 | 1.326 | 1.810 | 0.484 | 0.491 | 0.295 | +19.60 |

Every tile beats its base rate on both NLL and top-1. The 5-5 tile shows the smallest
lift — its base rate is already near-informative (base top-1 0.500), so there is less
headroom; the off-double count tiles (6-4/4-1) show the largest gains.

## Read

Otis v0's fate heads carry real, held-out information beyond the marginal — the
count-fate ledger decomposition is learnable from info-state alone, comfortably inside
its registered P2 band, with the pricing consumer path untaxed. Downstream: P4 arena
bid-swap gate (GPU) is not run here.

## Artifacts

- Bidders: `otis/models/otis_v0_control.pt`, `otis/models/otis_v0_treatment.pt`
- Raw nets: `scratch/otis-night/otis_raw_{control,treatment}.pt` (+ `otis/models/*_net.pt`)
- Calibration: `scratch/otis-night/w3_calibration.json`, `otis/reports/otis_v0_calibration.json`
- Curves: `scratch/otis-night/w3_train_{control,treatment}.log`
- Tests: `.venv/bin/python -m pytest otis/ -q` → 59 passed, no skips
