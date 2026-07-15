# W3 — otis v0 net, trainer, calibration, bidder export

_CPU only. Corpus: on-policy `selfplay` split (train 102,207 · val 5,557 · test
5,709), joined snapshot info-states → W2 offline fate labels on
`(source, seed, game_idx, hand_idx, a_team)` (w2_corpus.md §7). Seed 42._

## Arms (one flag)

- **control** = trunk (256→256) + pricing head (→43) — bit-exactly the jud v0
  `champion.margin_net.MarginNet` shape; featurization imported verbatim.
- **treatment** = control + five 8-class fate heads + one 8-class trick head off
  the same trunk + a mean-consistency penalty.

RNG parity verified: under one `torch.manual_seed`, both arms draw identical
trunk+pricing initial weights (heads constructed after the shared params) and
identical DataLoader batch order (seeded generator). Same corpus, capacity, seed.

## Registered loss weights (FIXED — no tuning, one-flag honesty)

```
total = L_price + 0.5·mean(fate CE) + 0.25·(trick CE) + 0.1·(mean-consistency)
        W_FATE = 0.5   W_TRICK = 0.25   W_CONSISTENCY = 0.1
```

- `L_price` — CrossEntropy over 43 bins on `bidder_team_pts` (identical to control).
- fate CE — mean over the five tile heads, 8-class `(capture_bidding × played_mode)`.
- trick CE — 8-class over `bidder_tricks` (0..7).
- mean-consistency — `mean |E[pricing pts] − (Σ_t value_t·P(my team captures t) + E[tricks])|`,
  where `P(my team captures t)` sums the four `bidding_team|*` fate classes (my-team
  perspective = MarginDataset's `bidder_team_pts` y perspective).

Hyperparameters (mirrored from `margin_net.train`, identical both arms): Adam
lr 1e-3, batch 256, 60 epochs, patience-8 early-stop on **val pricing CE**.

## Results

| arm | best val pricing CE | epochs (early-stop) | wall (CPU) |
|---|---|---|---|
| control | **2.6456** | 15 | 6.3 s |
| treatment | **2.6407** | 25 | 19.6 s |

Pricing head is **matched** — adding the decomposition organs did not tax the
consumer path (treatment val NLL 2.6407 vs control 2.6456; instrument-only, in
line with P4's registered "tie"). The fate/trick heads are pure additions.

## P2 — fate calibration (held-out test, cluster bootstrap over deals, 1000 resamples)

Base rates recomputed on **train only**; NLL improvement = base NLL − model NLL.
Cluster = deal `(seed, hand_idx)` (paired `a_team` halves share an auction, per
the fable W2b gate). Band: NLL ≥ 0.15 nats CI-excludes-0 AND top-1 ≥ 8 pp.

| metric | point | 95% CI | band | verdict |
|---|---|---|---|---|
| fate NLL improvement (nats) | **0.4775** | [0.4669, 0.4862] | ≥0.15, excl 0 | **PASS** |
| top-1 accuracy delta (pp) | **17.02** | [16.21, 17.79] | ≥8, excl 0 | **PASS** |

Per-tile NLL improvement (nats) / top-1 delta (pp): 5-5 0.356 / 9.0 · 6-4 0.536 /
20.0 · 5-0 0.477 / 17.3 · 4-1 0.535 / 19.2 · 3-2 0.484 / 19.6. **Every tile beats
its base rate; P2 PASSES with margin** — the count-fate ledger is a learnable
object from info-states alone.

Pricing-head val reliability (both arms): control ECE(P(pts≥30)) and treatment
reliability tables in `otis/reports/otis_v0_calibration.json`.

## Export — trunk+pricing → vanilla MarginNet checkpoint

`otis/export_bidder.py` lifts `net.{0,2,4}` and writes
`otis/models/otis_v0_<arm>.pt` as `{"model_state", "feature_dim"}`.
`champion.value_bidder.load_margin_net` loads it **STRICT**; pricing logits
verified **bit-exact** on 100 rows for both arms (`max_abs_diff = 0.000e+00`).

| artifact | path |
|---|---|
| control net (OtisNet) | `otis/models/otis_v0_control_net.pt` |
| treatment net (OtisNet) | `otis/models/otis_v0_treatment_net.pt` |
| control bidder (MarginNet) | `otis/models/otis_v0_control.pt` |
| treatment bidder (MarginNet) | `otis/models/otis_v0_treatment.pt` |
| calibration eval | `otis/reports/otis_v0_calibration.json` |

## Code + tests

`otis/model.py` · `otis/data.py` · `otis/train.py` · `otis/eval_calibration.py` ·
`otis/export_bidder.py` · `otis/tests/test_model.py`. Full otis suite: **59 passed**,
no skips. test_model covers export bit-exactness (both arms), fate-class derivation
(3 hand-derived + real rows), consistency-penalty math (3 synthetic pdfs), and
control==MarginNet shape parity.

## Notes / concerns

- Corpus scope = `selfplay` only (matches `fate_base_rates.json` and the on-policy
  design); `--source all|netwp|random` are supported and re-derive base rates on
  the matching train split. netwp/random are not folded into v0 training.
- Row convention: one declarer-POV row per hand (MarginDataset's y perspective).
  Unlike MarginDataset, no cross-`a_team` dedup — paired halves are distinct
  outcome rows sharing an info-state; both arms consume the identical set, so the
  arm comparison stays fair. Documented divergence, not silent.
- P4 (paired bid-swap arena treatment-vs-control) is downstream of this build and
  not run here; the pricing-parity + P2-pass results are the instrument evidence
  the arena gate will sit on top of.
