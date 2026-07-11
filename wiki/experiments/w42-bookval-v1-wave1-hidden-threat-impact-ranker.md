---
title: W42 Book Validation v1 Wave 1 — Hidden Threat Impact Ranker
bead: t42-c2y9
parent_epic: t42-4zi6
status: complete
date: 2026-05-03
wave: "1.3"
---

# Hidden Threat Impact Ranker

Ranks unseen tiles by their per-decision outcome swing and cross-tabs load-bearing tile
patterns against book-chapter detector vocabulary. Produces offline labels for the future
belief-attention head described in [[w42-book-claim-synthesis-and-ai-directions]].

## Related pages

- [[w42-hidden-threat-legacy-mining]] — established the impact-score formula and streaming
  aggregation infrastructure this wave reuses
- [[w42-phase2-hidden-domino-threat-attribution]] — sketched the per-decision tile-ranking schema
- [[w42-powered-branch-atlas-v1]] — source of the 1026-row hidden_threat_rows.csv corpus (seeds 9420-9421)
- [[w42-branch-atlas-scaled-v0]] — source of the 5955-row hidden_threat_rows.csv corpus (seed 9430)
- [[w42-book-claim-synthesis-and-ai-directions]] — motivates the belief-attention head this wave labels for

## Question

For each decision where hidden-holder labels exist, which specific unseen tile (in which seat)
drives the largest swing in the outcome distribution, and is that "load-bearing tile"
predictable from book-detector vocabulary?

## Data slice

| Dimension | Value |
|-----------|-------|
| Hidden-threat corpus A | `w42/branch_atlas_v1/hidden_threat_rows.csv` — 54 decisions, seeds 9420-9421 |
| Hidden-threat corpus B | `w42/branch_atlas_scaled_v0/hidden_threat_rows.csv` — 270 decisions, seed 9430 |
| Detector cross-tab corpus | `w42/joined_claim_row_model_table/joined_claim_action_rows.csv` — 75079 rows, seeds 0-99 |
| Total unique decisions | 324 |

**Seed mismatch:** The two corpora do not share seeds. All detector cross-tabulation is
categorical (bridged via decl_name × seat_role), not row-level.

## Method

For each (decision, action_slot), the `hidden_threat_rows` schema provides one row per
(hidden_domino, relative_holder) triplet with:

```
impact_score = |mean_q_delta| + 10*(|tail_low_mass_delta| + |shelf_high_mass_delta|)
downside_score = (-mean_q_delta).clip(0) + 10*tail_delta.clip(0) + 5*(-shelf_delta).clip(0)
```

(Formula unchanged from [[w42-hidden-threat-legacy-mining]].)

Per-decision tile ranking: for each decision, aggregate to max-impact across action slots
per (hidden_domino, holder) pair, then rank descending. Output top-K (K=5).

Tile categories:
- `trump_double` — double in the called suit
- `trump_count` — count tile (5-0=5pts, 5-5/6-4=10pts) in the called suit
- `trump_plain` — non-count, non-double trump
- `offsuit_double` — double in non-trump suit
- `count_tile` — count tile outside trump suit
- `plain_tile` — everything else

## Results

### Tile category enrichment (top-1 load-bearing tile vs all top-5 pool)

| Tile Category | Rate as Top-1 | Rate in Top-5 | Enrichment |
|---------------|--------------|--------------|------------|
| trump_double | 0.216 | 0.163 | **1.33x** |
| trump_count | 0.062 | 0.061 | 1.02x |
| trump_plain | 0.228 | 0.241 | 0.95x |
| offsuit_double | 0.173 | 0.284 | 0.61x |
| plain_tile | 0.284 | 0.451 | 0.63x |
| count_tile | 0.037 | 0.069 | 0.54x |

**Trump doubles are 1.33x enriched** as the single highest-impact hidden tile. Plain tiles
and offsuit doubles appear frequently in the broader top-5 but rarely as the number-one
driver — they represent diffuse background threat, not the specific load-bearing tile the
book is pointing at.

### Single-tile dominance

Only **1.85%** (6/324) of decisions have the top-1 tile accounting for ≥60% of total
decision impact. The median top-tile concentration is 20.6%. This quantifies the gap
between the book's rhetorical framing ("one tile changes the plan") and the mathematical
reality: the effect is real and locally important but diffuse across most hands.

The six cases that do cross the 60% threshold are degenerate (very low absolute impact)
rather than high-stakes — a structural observation consistent with the finding that
high-impact decisions tend to have distributed threat across several tiles.

### Mean impact by tile category

| Category | Mean Impact | Downside | Direction |
|----------|------------|----------|-----------|
| trump_count | 30.2 | 0.0 | **100% helpful** |
| trump_double | 23.5 | 2.5 | mixed |
| offsuit_double | 20.5 | 3.6 | 70% helpful |
| trump_plain | 18.0 | 2.1 | mixed |
| count_tile | 16.1 | 2.9 | mixed |
| plain_tile | 13.7 | 3.5 | mixed |

**Trump-count tiles are uniquely directional: 100% helpful.** Knowing the trump-suit
count tile (e.g., 4-0 and 4-5 in fours) is held by a specific seat always raises the
actor's Q-distribution mean. This makes trump-count tiles the most "legible" load-bearing
tile from the perspective of belief-attention: conditioning on them produces a clean
unidirectional signal, not a mixed one.

### Highest-impact single-decision tile occurrences

1. **5-5 held by bidder, twos declaration** — impact 56.8, helpful: in twos the double-five
   is an offsuit boss that determines trick capture in the non-trump suits; its location
   determines whether the bidder can time count safely
2. **4-4 held by bidder_partner, no-trump** — impact 47-49, helpful: in no-trump the 4-4
   is the dominant suit boss; knowing partner holds it enables aggressive count timing
3. **4-2 held by right_setter, fours declaration** — impact 44.3: trump-plain tile that
   determines whether the setter can survive until trump depletion

### Inferability (top-100 by impact)

| Inferability bucket | N | Mean conditioned_mass |
|---------------------|---|-----------------------|
| moderately_inferable (0.30-0.49) | 69 | 0.352 |
| hard_to_infer (< 0.30) | 31 | 0.195 |

No top-100 decision had an easily_inferable (mass ≥ 0.50) load-bearing tile. The 31
"hard_to_infer" cases include 9 offsuit_doubles and 7 trump_plain tiles — the most
threatening tiles are also the ones with the least public-state signal, which is why
a belief-attention head is necessary for live inference.

### Detector cross-tab (categorical bridge)

Exact row-level join is not possible (seed mismatch). The categorical bridge via
(decl_name × seat_role) reveals:

- `setter_pounce_sequence_window` fires at 52% for left_setter and 51% for right_setter
  across all declarations. In those same seats, trump_double tiles are load-bearing at
  12-30% of decisions — consistent with the book's ch05 claim that setter-pounce windows
  are where trump-holding knowledge is most strategically consequential.
- `hidden_proxy_early_high_uncertainty` fires at 51% for bidders. Bidder trump_double
  impact (mean 22-44) is highest in this regime, aligning with the book's framing of
  opening-phase uncertainty as the primary hidden-tile risk surface.
- `ch05_setter_pounce_count_before_certainty` fires at 1-3% of setter decisions. This
  low base rate means the targeted claim is rare but co-occurs with the highest-impact
  tile patterns when it fires.
- Trump-count tiles as load-bearing appear in **fours** (12 decisions), **blanks** (6),
  **sixes** (2). The 4-0/4-5 tiles in fours declaration are uniquely impactful because
  fours has a constrained trump suit (7 tiles vs 13 in a suit-declared game), so each
  trump count tile resolves a larger fraction of positional uncertainty.

## Claim ledger impact

**Status: `underpowered`**

Evidence is offline diagnostic only. The categorical trump_double enrichment (1.33x) and
trump_count directional consistency (100% helpful) provide weak support for the claim
family "one specific unseen tile changes the plan." The 1.85% single-tile-dominant rate
refutes the strong version of the claim as a systematic property of all decisions. The
seed mismatch prevents row-level verification against detector-tagged decisions.

No central ledger status promoted to `supported`.

## Caveats

- 324 decisions from two small generated game slices; not live auction corpus
- All detector cross-tab is categorical, not row-level (seed mismatch)
- `conditioned_mass` inferability heuristic does not reconstruct actual game history
- Hidden-holder labels are offline eval only; not proposed as live features in this wave

## Artifacts

Output directory: `w42/book_validation_v1/wave1/t42-c2y9_hidden_threat_impact_ranker/`

| File | N rows | Purpose |
|------|--------|---------|
| `per_decision_top_k_tiles.csv` | 1549 | Top-5 tiles per decision with impact, holder, direction, tile_category |
| `single_tile_dominant_decisions.csv` | 100 | Highest concentration decisions (top-1 tile share) |
| `inferable_vs_not.csv` | 100 | Top-100 impact split by inferability bucket |
| `detector_correlation.csv` | 113 | Categorical cross-tab by decl_name × seat_role × tile_category |
| `summary.json` | — | Machine-readable headline numbers |
| `manifest.json` | — | Input SHAs, command, leakage boundary |

## Reproducibility

```bash
.venv/bin/python3 w42/book_validation_v1/wave1/t42-c2y9_hidden_threat_impact_ranker/run_hidden_threat_impact_ranker.py
```
