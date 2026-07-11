---
title: "Book Validation v1 Wave 1 — Cross-AI Agreement Analysis"
experiment_id: w42-bookval-v1-wave1-cross-ai-agreement
bead: t42-m2i7
parent_bead: t42-4zi6
wave: "1.4"
status: complete
date: 2026-05-03
tags: [cross-ai, agreement, oracle, gus, detectors, distribution-lens, book-validation]
related:
  - [[w42-claim-tag-model-probe]]
  - [[w42-phase3-joined-claim-row-model-table]]
  - [[w42-phase4-claim-completion-board]]
  - [[w42-book-claim-synthesis-and-ai-directions]]
---

## Question

Where do book detectors, the Gus row-model, scalar EV, and the distribution-lens
reranker all disagree on the same decision? Those decisions are the most
pedagogically valuable — they are where the project's tools actively conflict,
which is exactly where the book's vocabulary either holds up or fails.

## Motivation

[[w42-claim-tag-model-probe]] showed detector tags improve the Gus row model.
[[w42-phase3-joined-claim-row-model-table]] improved mean regret from 1.36 → 1.13
by adding all claim families. Now we map where the four tools diverge on the same
decisions to expose structural book-vocabulary gaps.

## Slice

| Corpus | Decisions | Action rows | Seeds |
|--------|-----------|-------------|-------|
| `joined_claim_action_rows.csv` (primary) | 28,000 | 75,079 | corpus_v2_train, seeds 0–99 |
| `branch_atlas_scaled_v0` + Wave 1.1 (sub-corpus) | 280 | 773 | seed 9430 only |

## Source Definitions

| Source | Operationalization | Coverage |
|--------|-------------------|----------|
| `ev_top_action` | `is_best_mean` from joined table (mean_regret == 0); exact EV oracle | 28,000 decisions |
| `gus_top_action` | `is_best_threshold` from labeled_handshape (proxy — individual model scores not saved) | 28,000 decisions |
| `detector_endorsed` | Any ch03/ch04/ch05 positive label from `paired_contrasts.csv` fires on this candidate | 9,667 decisions with detector active |
| `dist_lens_top_action` | Top-1 under any non-EV utility (p_make, cvar_10, mark_ev, robust_q25, threshold) from Wave 1.1 | 280 decisions (1.0% of primary corpus) |

**Gus proxy note:** The Gus row-model was trained to predict `is_best_mean`.
Individual candidate scores are not saved in the artifacts; `model_metrics.csv`
reports only aggregate match rates (64.5% with EV for public_features_only).
The `is_best_threshold` proxy has an aggregate agreement rate with EV of 78.3%,
making it the best available row-level stand-in.

## Headline Agreement Rates

| Source pair | Agreement rate | N decisions |
|-------------|---------------|-------------|
| EV vs Gus | **78.3%** | 28,000 |
| EV vs Detector | **43.7%** | 9,667 |
| Gus vs Detector | **47.0%** | 9,667 |
| EV vs Dist-lens | **39.5%** | 539 |

Full pairwise matrix (from `agreement_matrix.csv`):

|  | ev | gus | detector | dist |
|--|-----|-----|---------|------|
| **ev** | 1.000 | 0.783 | 0.437 | 0.395 |
| **gus** | 0.783 | 1.000 | 0.470 | 0.408 |
| **detector** | 0.437 | 0.470 | 1.000 | 0.253 |
| **dist** | 0.395 | 0.408 | 0.253 | 1.000 |

## Divisiveness Distribution

| Category | Count | Fraction |
|----------|-------|---------|
| All sources fully agree (N=1 distinct pick) | 18,378 | 65.6% |
| 2-source spread | 8,529 | 30.5% |
| 3-source spread | 1,082 | 3.9% |
| 4-source spread (all different) | 11 | 0.04% |

The 11 four-way-split decisions are the most extreme divergence points. They span
declarations (3× twos, 3× sixes, 2× fives, 2× doubles-suit, 1× no-trump), occur in
early-to-middle tricks (trick_idx 0–3), and have 4–7 legal actions
(from `divisive_decisions.csv`, `n_distinct_picks == 4`).

## Per-Claim-Family Agreement

| Family | Decisions | Detector fires | EV–Gus agree | EV–Detector agree | Mean det. regret | Gus error when det ≠ EV |
|--------|-----------|----------------|--------------|-------------------|-----------------|------------------------|
| ch03 (Bidder Play) | 2,761 | 1,705 | 68.7% | 22.2% | 4.44 | 38.7% |
| ch04 (Partner Support) | 7,000 | 1,360 | 80.5% | 48.7% | 3.04 | 30.7% |
| ch05 (Setter Defense) | 14,000 | 6,602 | 78.8% | 48.2% | 3.33 | 37.6% |

**Interpretation:**
- **ch03 is most contested:** EV–Detector agreement is only 22.2%, the lowest across families.
  Bidder lead sequencing rules from the book frequently endorse the wrong domino.
- **ch04 is most reliable:** EV–Detector agreement 48.7%, lowest mean detector regret (3.04).
  Partner support plays are more validatable from the book's vocabulary.
- **ch05 is numerically dominant:** 6,602 detector-fired decisions; 48.2% EV agreement.
  When Gus and the detector disagree with EV, Gus sides with EV 62.4% of the time in ch05.

## Five Most Surprising Disagreement Patterns

### 1. Detector endorses count-dump at high regret (ch05_reckless_count)
`ch05_reckless_count_to_bidder_control` fires 2,300 times on non-EV actions with mean
regret **9.18** — the second-highest regret among all conflict labels. The book says
"give count to help partner," but the oracle finds many situations where holding back
is strictly better. This is a context-sensitivity failure: the book rule is correct in
certain positions but the detector labels the entire regime rather than the qualifying
sub-condition.

### 2. ch03_called_non_double fires on non-EV 86% of the time
Non-double trump leads endorsed by ch03 are consistently suboptimal. The double (boss
trump) is typically EV-superior at opening lead. This validates the positive-contrast
direction from the phase-4 paired_contrasts analysis: commanding_called_double beats
called_non_double. The detector tags the action correctly within its contrast pair but
the absolute endorsement (vs the oracle's first choice) is wrong most of the time.

### 3. Setter-regime label is too coarse (ch05_setter_pressure_regime)
The coarsest label in ch05 fires on 100% of setter action rows, and 60% of the rows
it fires on are non-EV-best. It marks the game regime, not the action quality. Because it fires on
every candidate in a setter-regime decision, it cannot discriminate the best action.
This is a regressor/detector design gap: regime labels need action-discriminating
subconditions before they can endorse a specific domino.

### 4. CVaR and robust-Q25 track EV much better than make-rate
On the 280-decision dist-lens sub-corpus (seed 9430; top-1 argmax agreement with EV,
utility-side ties broken by row order — EV itself has no tied maxima):
- `cvar_10` agrees with EV: **82.5%** (tie-free)
- `robust_q25` agrees with EV: **81.8%** (tie-free)
- `p_make` agrees with EV: **74.6%** (91.1% if ties count as agreement)
- `threshold_mass_low` agrees with EV: only **45.7%** (58.9% if ties count)

Risk-adjusted utilities are near-substitutes for EV; make-rate and especially the
lower-threshold-mass lens diverge more. This suggests the distribution-lens reranker's
most EV-consistent lenses are CVaR/robust-Q25, while threshold-mass is the true outlier.

### 5. Detector endorses 5-5 (the double) against the EV-best non-count
Twenty of the top-100 most divisive decisions (915 decisions corpus-wide) have the detector
endorsing 5-5 when EV picks a different domino. The most
extreme case (4-way spread, regret 31.03) is a no-trump setter closure: the detector
endorses 5-5 (count, double) at regret 31, while EV picks 6-0 (non-count, non-double),
Gus picks 1-1 (minimal regret 1.33), and dist-lens picks 5-1. The 5-5 looks safe
in the book vocabulary (doubles are weapons) but is a catastrophic misfire in the
specific no-trump closure context. This directly targets [[w42-claim-tag-model-probe]]'s
finding that 5-5 is 2.8× enriched in winners: the enrichment is real on average but
the detector over-generalizes it into contexts where it does not apply.

## Dist-Lens Gap

Wave 1.1 operates on a 280-decision sub-corpus (seed 9430). The primary 28,000-decision
corpus does not yet have distribution-lens utilities. On the overlapping 539 action rows
(those from the primary corpus that happen to be in branch_atlas), the EV–dist-lens
agreement is 39.5%. The orchestrator should integrate the missing column when Wave 1.1 is
extended to the full corpus.

## Artifacts

| File | Description |
|------|-------------|
| `per_action_source_picks.csv` | 75,079 rows × {ev_top, gus_top, detector_endorsed, dist_lens_top} |
| `agreement_matrix.csv` | 4×4 pairwise agreement rates |
| `divisive_decisions.csv` | Top 100 most divisive decisions |
| `spot_check_pack.json` | Human-readable pack for the same 100 decisions |
| `per_claim_family_agreement.csv` | Per-family agreement summary |
| `summary.json` | Machine-readable headline numbers |
| `manifest.json` | Input SHA256 hashes, reproducibility command |

**Reproduce:**
```bash
source forge/venv/bin/activate
python3 w42/book_validation_v1/wave1/t42-m2i7_cross_ai_agreement/run_cross_ai_agreement.py
```

## Claim-Ledger Impact

`underpowered` — This analysis is diagnostic. It identifies where vocabulary fails and
which claim families are most contested, but agreement rates alone are not direct tests
of individual claim truth values.
