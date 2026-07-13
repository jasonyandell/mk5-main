---
title: "W42 Book Validation v1 — Wave 1 Distribution Lens Reranker"
kind: experiment
status: complete
first_seen: 2026-05-03
bead: t42-ybo6
parent_epic: t42-4zi6
last_updated: 2026-07-13
---
## Summary

Scalar EV disagrees with at least one risk-sensitive utility lens in **64.9% of
decisions** (137/211) on the branch_atlas_scaled_v0 corpus. CVaR_10 (tail-risk
mean) and robust_q25 (25th percentile) are the most divergent from EV, each
selecting a different top-1 action roughly 25-30% of the time. Despite the
disagreements, EV's preferred action never loses in expected-value units — meaning
the divergence is about risk framing, not expected-value dominance. The no-trump
regime shows the highest EV-lying rate (90%), followed by large_lower_tail and
multi_peak_pdf distribution shape tags (67-69%).

## Slice

- **Primary:** branch_atlas_scaled_v0 — seed 9430, 10 declarations (blanks through
  no-trump), bid=30 fixed, 1000 sampled worlds per decision via oracle softmax
- **Supplemental:** branch_atlas_v1 — seeds 9420-9421, 2 declarations, bid=30,
  1000 worlds
- **N decisions:** 211 | **N legal-action rows:** 907
- **Paired:** yes — same decision, same world sample, different utility lens

## Method

1. Load per-world Q tensors from `eq_pdf_*_joint_1000s_v2.pt`; confirm
   `q_per_world.mean() == e_q` (verified exact match).
2. For each legal action compute 8 utility scalars from the 1000-world sample:
   `ev`, `p_make`, `p_set`, `threshold_mass_low`, `threshold_mass_high`,
   `cvar_10`, `mark_ev`, `robust_q25`.
3. For each decision rank actions by each utility; identify the top-1 action per
   utility.
4. Record pairwise top-1 disagreement rates (8x8 matrix).
5. For each decision compute `n_disagree_with_ev` (count of utilities preferring a
   different action than EV) and `ev_gap_vs_<utility>` (EV-point cost of choosing
   the alternative-preferred action).
6. Cross-tab disagreements against `distribution_shape_tags`,
   `strategy_context_tags`, and `matched_position_detectors` from
   `decision_actions.csv`.
7. Rank all 211 decisions by disagreement count; export top 200 for human review.

### Utility definitions

| Utility | Formula | Lens perspective |
|---------|---------|-----------------|
| ev | mean(q_per_world) | expected count-point swing |
| p_make | P[Q >= 18] | probability of making bid (30) |
| p_set | P[Q < 18] | probability of being set |
| threshold_mass_low | P[Q <= -18] | danger-zone mass |
| threshold_mass_high | P[Q >= 18] | = p_make |
| cvar_10 | mean of lowest-10% Q worlds | tail conditional expectation |
| mark_ev | E[team0_marks] with Ch10 transform | tournament-mark objective |
| robust_q25 | Q.quantile(0.25) | 25th-percentile robustness |

**Degeneracy note:** For ordinary bid=30, `p_make == threshold_mass_high == mark_ev`
(all use the Q >= 18 cut). This is correct by construction — verified against the
deterministic_terminal_transform from [[w42-phase4-scoring-objective-tests]].

## Findings

### Disagreement matrix (top-1 action)

EV vs utility pair top-1 disagreement rates:

| vs utility | disagree rate |
|-----------|--------------|
| cvar_10 | 25.6% |
| robust_q25 | 29.9% |
| threshold_mass_low | 36.0% |
| p_make / p_set / mark_ev / threshold_mass_high | 38.9% (all equal) |

EV vs all-other combined: **64.9%** (at least one utility disagrees).

Full 8x8 matrix in `utility_disagreement_matrix.csv`.

### EV-gap analysis

When an alternative utility picks a different action, EV's preferred choice is
never strictly dominated in expected value:

| utility | mean EV gap | n_alt_dominates_ev |
|---------|-------------|-------------------|
| cvar_10 | +1.4 pts | 0 |
| robust_q25 | +1.3 pts | 0 |
| threshold_mass_low | +1.4 pts | 0 |
| p_make et al. | +1.1 pts | 0 |

**Interpretation:** EV's top action retains ~1.1-1.4 count-point EV advantage over
the alternative-utility-preferred action. The divergence reflects risk framing, not
EV dominance — the alternative choice is systematically safer (lower tail exposure)
at a small EV cost.

### Per-detector EV-lying rates (n_decisions >= 5)

| Detector tag | n_decisions | ev_lying_rate | cvar_disagree | robust_q25_disagree |
|---|---|---|---|---|
| no_trump_regime | 20 | 90.0% | 40.0% | 35.0% |
| setter_count_pressure | 5 | 80.0% | 40.0% | 20.0% |
| large_lower_tail | 108 | 69.4% | 34.3% | 39.8% |
| multi_peak_pdf | 189 | 66.7% | 27.0% | 32.3% |
| late_trick_threshold_closure | 77 | 66.2% | 23.4% | 31.2% |
| last_to_act | 53 | 64.2% | 32.1% | 33.9% |
| high_variance_close_mean | 101 | 61.4% | 24.8% | 30.7% |
| doubles_trump_regime | 18 | 61.1% | 38.9% | 38.9% |
| bidder_opening_lead | 12 | 58.3% | 16.7% | 0.0% |
| first_trick_public_belief_update | 182 | 68.1% | 27.5% | 32.9% |

### Top 5 most-disagreed decisions (all 7 utilities disagree)

1. `v0:r7:d21` — bidder_partner, doubles-trump-regime, late hand — near-zero EV
   context where tie in p_make yields different action across lenses.
2. `v0:r9:d4` — bidder, no-trump regime — EV picks 6-5, CVaR picks 6-6 (higher
   worst-case floor), EV delta is +0.016 pts.
3. `v0:r9:d15` — right_setter, no-trump, last_to_act — EV and tail lenses strongly
   split on all 4 legal actions.
4. `v0:r2:d12` — bidder, no tagged context — multi-peak PDF with high variance.
5. `v0:r9:d11` — right_setter, no-trump, last_to_act — similar to decision 15.

## Caveats

- Single base seed (9430) and fixed bid=30. No bid variation, no multi-seed
  generalization. Sample size is 211 decisions across 12 game records.
- Claim-family tags sourced from branch atlas internal CSVs (different corpus than
  [[w42-phase3-joined-claim-row-model-table]] seeds 0-99). No direct row join.
- `mark_ev == p_make` for ordinary bid. Multi-mark bids (84+) not present in corpus.
- `n_alt_dominates_ev = 0` does not test cross-world distributional bias — only
  within-sample dominance.
- Detector rates for tags with < 20 decisions are high-variance.

## Artifacts

Under `w42/book_validation_v1/wave1/t42-ybo6_distribution_lens_reranker/`:

| File | Description |
|------|-------------|
| `action_utility_scalars.csv` | 907 rows, all utility scalars + ranks + claim tags |
| `utility_disagreement_matrix.csv` | 8×8 top-1 disagreement rate matrix |
| `detector_explained_disagreements.csv` | 31 detector tags with EV-lying and per-utility rates |
| `top_disagreements.csv` | Top 200 most-disagreed decisions for human review |
| `summary.json` | Machine-readable headline numbers |
| `manifest.json` | Provenance, input SHAs, exact command |
| `README.md` | Agent contract fields (question, slice, N, metric, status, caveats) |
| `run_distribution_lens_reranker.py` | Reproducibility script |

## Provenance

| Input | SHA256 |
|-------|--------|
| eq_pdf_s9430_d10_bid30_joint_1000s_v2.pt | c0d124b0... |
| decision_actions.csv (v0) | 112cc962... |
| eq_pdf_s9420-9421_joint_1000s_v2.pt | c6e2a700... |
| decision_actions.csv (v1) | 41d1d73d... |

Exact command:
```
python -u w42/book_validation_v1/wave1/t42-ybo6_distribution_lens_reranker/run_distribution_lens_reranker.py
```

## Links

- [[w42-book-claim-synthesis-and-ai-directions]] — parent proposal for utility-family lens
- [[w42-phase2-distribution-aware-ev-report]] — prior per-world distribution feature definitions
- [[w42-phase3-joined-claim-row-model-table]] — master claim-tag row table (different corpus)
- [[w42-phase4-claim-completion-board]] — baseline 64-row claim ledger
- Parent bead: t42-4zi6 (epic) / this bead: t42-ybo6
