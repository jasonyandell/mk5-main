# Distribution Lens Reranker — Wave 1.1 / t42-ybo6

## Question

Where does scalar EV "lie" relative to seven alternative utility lenses (p_make,
p_set, threshold_mass_low, threshold_mass_high, CVaR_10, mark_ev, robust_q25)?
Which detector-family tags explain each disagreement regime?

## Slice

**Primary corpus:** branch_atlas_scaled_v0 — seed 9430, 10 declarations (blanks
through no-trump), bid=30 fixed, 1000 sampled worlds per decision.

**Supplemental corpus:** branch_atlas_v1 — seeds 9420-9421, 2 declarations, bid=30,
1000 sampled worlds per decision.

- N decisions: 211 (after deduplication within each 7-slot action set)
- N legal action rows: 907 (773 from v0, 134 from v1)
- Paired: yes — same decision, same 1000-world sample, different utility lens

## Metrics

| Utility | Definition |
|---------|------------|
| ev | q_per_world.mean() |
| p_make | P[Q >= 18] (ordinary bid=30 make threshold) |
| p_set | P[Q < 18] (= 1 - p_make) |
| threshold_mass_low | P[Q <= -18] (danger zone) |
| threshold_mass_high | P[Q >= 18] (= p_make by construction) |
| cvar_10 | mean of lowest 10% of Q worlds |
| mark_ev | E[team0_marks] per world using Ch10 transform (ordinary bid: 1 mark if made) |
| robust_q25 | 25th percentile of q_per_world |

**Note:** p_make, p_set complement each other; threshold_mass_high == p_make; mark_ev ==
p_make for ordinary bid=30 (all three share the same Q >= 18 threshold). These are
identical lenses and confirm internally consistent computation.

## Status

complete — underpowered for ledger promotion (single seed, fixed bid)

## Key Findings

1. **EV disagrees with at least one alternative in 64.9% of decisions** (137/211).
   This is the headline "EV lying" rate on this corpus.

2. **CVaR_10 and robust_q25 are the most divergent from EV** (25.6% and 29.9%
   respectively), meaning the tail-risk lenses pick a different top-1 action roughly
   one decision in four.

3. **p_make / p_set / threshold_mass_high / mark_ev all agree perfectly** — they are
   mathematically identical for ordinary bid=30 (Q >= 18 threshold). EV disagrees
   with this cluster 38.9% of the time.

4. **EV never loses in EV-units** when an alternative picks differently: mean EV gap
   is +1.1 to +1.4 points in EV's favor. The alternative-preferred action is never
   strictly better in expected value (n_alt_dominates_ev = 0). EV's top-1 genuinely
   scores higher in expectation — but may score worse on tail risk.

5. **No-trump regime is the highest EV-lying regime** (90% of decisions flagged).
   In no-trump, multiple actions cluster near the make threshold (Q ≈ 18), making
   the fine EV ordering fragile while tail-risk ordering diverges sharply.

6. **Multi-peak PDF and large_lower_tail shape tags predict disagreement** (67-69%
   lying rate vs overall 65%). Distribution shape detectors from phase-2/3 are
   moderate predictors of EV-vs-alternative disagreement.

7. **The doubles-trump regime shows 39% CVaR disagreement** — higher than average,
   consistent with the claim that doubles lead strategy is risk-sensitive.

## Caveats

- Single base seed (9430) and fixed bid=30 — no bid variation, no seed variation.
- The joined_claim_row_model_table (seeds 0-99) is on a different corpus; no direct
  row join was possible. Claim-family tags sourced from branch atlas internal CSVs.
- mark_ev and p_make are identical for bid=30 ordinary. Multi-mark bids (84+) not
  present.
- n_alt_dominates_ev = 0 does NOT mean EV is globally correct — it means within the
  same 1000-world sample, EV's pick cannot be strictly dominated in EV. Cross-world
  bias is not tested.
- The "no_disagree" base is 211 decisions, not 75k — detector-family rates are noisy
  for tags with < 20 decisions.

## Exact Command

```
python -u w42/book_validation_v1/wave1/t42-ybo6_distribution_lens_reranker/run_distribution_lens_reranker.py
```

## Artifacts

| File | Description |
|------|-------------|
| action_utility_scalars.csv | 907 rows; one per legal action; all 8 utility scalars + ranks + tags |
| utility_disagreement_matrix.csv | 8x8 top-1 disagreement rates between utility pairs |
| detector_explained_disagreements.csv | 31 detector tags; EV-lying rate and per-utility disagree rate |
| top_disagreements.csv | Top 200 decisions by n_disagree_with_ev (for human spot-check) |
| summary.json | Machine-readable headline numbers |
| manifest.json | Provenance, input SHAs, exact command |
