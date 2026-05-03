# W42 Book Validation v1 — Wave 1 — Hidden Threat Impact Ranker

**Bead:** t42-c2y9  
**Parent epic:** t42-4zi6  
**Wave:** 1.3

## Question

For each decision where hidden-holder labels exist, which specific unseen tile (in which seat) drives the largest swing in the outcome distribution, and is that "load-bearing tile" predictable from book-detector vocabulary?

## Slice

| Dimension | Value |
|-----------|-------|
| Corpus A | `w42/branch_atlas_v1/hidden_threat_rows.csv` (1026 rows, 54 decisions) |
| Corpus B | `w42/branch_atlas_scaled_v0/hidden_threat_rows.csv` (5955 rows, 270 decisions) |
| Cross-tab | `w42/joined_claim_row_model_table/joined_claim_action_rows.csv` (75079 rows, seeds 0-99) |
| Seeds | 9420, 9421, 9430 (hidden-threat) vs 0-99 (joined claims) — no seed overlap |
| Declarations | All 10 (blanks, ones, …, sixes, no-trump, doubles, doubles-suit) |
| Seats | All 4 (bidder, left_setter, right_setter, bidder_partner) |

**Join note:** The hidden-threat corpora and joined-claim table do not share seeds. Cross-tab is categorical (by decl_name × seat_role), not row-level.

## N

- 6981 total hidden-threat rows (combined corpora)
- 324 unique decisions
- 895 unique (decision, action_slot) pairs
- 1549 top-K rows (K=5 per decision)
- 100 single-tile-dominant rows (by concentration)
- 100 inferable-vs-not rows (top-100 by impact)
- 113 detector-correlation rows (by decl_name × seat_role × tile_category)

## Paired or Unpaired

- Within-decision: unpaired tile ranking (max impact across action slots)
- Cross-tab: categorical bridge via (decl_name, seat_role)

## Metric

- `impact_score = |mean_q_delta| + 10*(|tail_low_mass_delta| + |shelf_high_mass_delta|)` (reused from legacy mining)
- `concentration_pct = top1_impact / sum(all_tile_impacts_in_decision)`
- `inferability = conditioned_mass` (fraction of worlds placing tile at that position)

## Results

### Tile Category Enrichment (top-1 load-bearing tile vs all top-5)

| Tile Category | Rate as Top-1 | Rate in Top-5 | Enrichment Ratio |
|---------------|--------------|--------------|-----------------|
| trump_double | 0.216 | 0.163 | **1.33x** |
| trump_count | 0.062 | 0.061 | 1.02x |
| trump_plain | 0.228 | 0.241 | 0.95x |
| offsuit_double | 0.173 | 0.284 | 0.61x |
| plain_tile | 0.284 | 0.451 | 0.63x |
| count_tile | 0.037 | 0.069 | 0.54x |

Trump doubles are 1.33x enriched as the single most load-bearing tile. Plain tiles and offsuit doubles are de-enriched (they appear often in the top-5 but rarely as the single highest-impact tile).

### Single-Tile Dominance

Only **1.85%** of decisions have one tile accounting for ≥60% of total decision impact. The median top-tile concentration is 20.6%. The distribution of impact across tiles is diffuse — the book's "one tile changes the plan" claim holds in rare but real cases, not systematically.

### Tile Category by Mean Impact Score

| Category | Mean Impact | Mean Downside | Direction (pct helpful) |
|----------|------------|--------------|------------------------|
| trump_count | 30.16 | 0.00 | 100% helpful |
| trump_double | 23.48 | 2.49 | — |
| offsuit_double | 20.54 | 3.56 | 70% helpful |
| trump_plain | 17.99 | 2.15 | — |
| count_tile | 16.09 | 2.93 | — |
| plain_tile | 13.75 | 3.47 | — |

**Trump-count tiles are 100% directionally "helpful"** (mean_q_delta always positive): knowing the trump-suit count tile is in a specific seat consistently improves the outlook. This is the tile category most consistent with the book's description of "one tile that changes the plan."

### Highest Single-Decision Impact Tiles

1. **5-5 held by bidder (twos game)** — impact 56.8, helpful: in a twos declaration, the holder of the double-five controls a non-trump trick that is otherwise contested
2. **4-4 held by bidder_partner (no-trump)** — impact 47-49: in no-trump, the 4-4 is a suit-boss; knowing partner holds it enables safe count timing
3. **4-2 held by right_setter (fours game)** — impact 44.3: trump-plain tile that determines whether the setter can extend the trump suit

### Inferability (Top-100 High-Impact Decisions)

| Inferability | Count | Mean conditioned_mass |
|-------------|-------|----------------------|
| moderately_inferable (mass 0.30-0.49) | 69 | 0.352 |
| hard_to_infer (mass < 0.30) | 31 | 0.195 |

No decisions in the top-100 by impact had `easily_inferable` (mass ≥ 0.50) load-bearing tiles. The most dangerous unknown tiles are precisely the ones with low conditioned mass — they could be anywhere. The 31 "hard_to_infer" decisions include 9 offsuit_doubles and 7 trump_plain tiles, not the trump_count tiles that would be most consequential.

### Detector Cross-Tab (Categorical Bridge)

The `setter_pounce_sequence_window` detector fires at ~52% rate for left_setter and ~51% for right_setter across all declarations. In those same seats, trump_double tiles appear as load-bearing at 12% (left_setter) and 30% (right_setter) of decisions. The pounce-window and load-bearing trump tile overlap in the setter seats but cannot be linked at the row level due to seed mismatch.

The `hidden_proxy_early_high_uncertainty` detector fires at ~51% for bidders and ~33% for all other seats. Bidder trump_double tiles have higher mean impact (24-44) in this high-uncertainty context, consistent with the book's claim that early uncertainty concentrates risk on hidden trump strength.

## Claim Ledger Impact

**Status: `underpowered`**

The wave produces offline tile-ranking labels and a categorical cross-tab. The claim that "one specific unseen tile changes the plan" is weakly supported at the distribution level (trump_double enrichment 1.33x, trump_count 100% helpful direction), but the 1.85% single-tile-dominant rate means it is rare in the systematic sense. The seed mismatch prevents a row-level detector↔tile join. No claim status promoted to `supported`.

## Caveats

1. Only 324 decisions across seeds 9420/9421/9430 — small generated slice, not live auction corpus.
2. No row-level join to joined_claim_action_rows (different seeds); all detector cross-tab is categorical.
3. Inferability heuristic (`conditioned_mass`) does not use actual game history — approximation only.
4. `impact_score` formula is unchanged from hidden_threat_legacy_mining; no tuning for this wave.
5. Hidden-holder labels are offline eval only. Not proposed as live features.

## Artifacts

| File | Description |
|------|-------------|
| `per_decision_top_k_tiles.csv` | 1549 rows: top-5 tiles per decision with impact, holder, direction |
| `single_tile_dominant_decisions.csv` | 100 rows: highest concentration decisions |
| `inferable_vs_not.csv` | 100 rows: top-100 impact split by inferability |
| `detector_correlation.csv` | 113 rows: (decl_name × seat_role × tile_category) cross-tab |
| `summary.json` | Machine-readable headline numbers |
| `manifest.json` | Input SHAs, provenance, leakage boundary |

## Reproducibility

```bash
.venv/bin/python3 w42/book_validation_v1/wave1/t42-c2y9_hidden_threat_impact_ranker/run_hidden_threat_impact_ranker.py
```
