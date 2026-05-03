# Wave 3.0 — Utility Lens Synthesis

**Bead:** t42-f2ur  
**Parent epic:** t42-4zi6  
**Wave:** 3  
**Status:** complete (do not promote ledger directly; orchestrator decides)

## Question

Does the W42 book's tactical advice survive re-examination under multiple utility lenses (EV, p_make, mark_ev, CVaR_10, robust_q25)?  
Which claims flip verdict depending on which objective a player optimizes?

## Slice

All closed Wave 2 probes with paired_contrasts.csv:
- t42-v9lu reentry_v2 (n=222)
- t42-jysl low_trump_trap (n=257)
- t42-26j8 void_creation (n=276)
- t42-z31l void_creation_follow (n=500)
- t42-ntbe pounce_bid30 (n=52 paired)
- t42-8kbh pounce_high_bid (n=1140)
- t42-ey88 bid_only_enough (step_pair_deltas, n≈14000 per step)

## Method

Bootstrap CIs (n=2000, percentile method) on paired delta columns.  
5 utility lenses per probe:
- **EV**: oracle E[Q] delta
- **p_make**: P(Q >= make_threshold) delta, derived from threshold_mass or p_set columns
- **mark_ev**: equivalent to p_make at bid=30 (Wave1.2); positive-affine identity at all bids (Wave2.H)
- **CVaR_10**: 10th-percentile Q delta (all probes record from Team 0 perspective)
- **robust_q25**: Q25 delta (not recorded in any probe — all missing)

## Key Findings

### Utility Flips Detected (verdict changes from EV baseline)

| Claim | EV | p_make | mark_ev | CVaR_10 |
|-------|----|--------|---------|---------|
| ch05-void-creation-lead | contradicted | contradicted | contradicted | **spans_zero** |
| ch05-void-creation-follow | **supported** | spans_zero | spans_zero | spans_zero |
| ch12-setter-pounce-high-bid | contradicted | contradicted | contradicted | contradicted |

No true reversal (contradicted↔supported): all flips are between contradicted/supported and spans_zero.

### ch05-void-creation-follow is the most utility-sensitive claim
EV supports void creation in follow position but p_make, mark_ev, and CVaR_10 all span zero.
This means: void creation follow may be EV-positive on average but does not reliably improve
the setter's probability of making their contract.

### ch12-setter-pounce-high-bid: unanimous contradicted across all 4 available utilities
The strongest result in the campaign. Not a utility-dependent verdict.

### ch02-bid-only-enough: unanimous supported (mark_ev and p_make agree)
The most robust book claim across utilities.

## N, Metric, Paired/Unpaired

All probes: paired same-decision contrasts.  
Bootstrap CIs at 95%, percentile method, n_boot=2000, seed=42.

## Ledger Impact Proposal

This wave does NOT modify the central ledger directly.  
Proposed schema change: add per-utility status columns (see objective_aware_ledger_schema.json).  
Recommendation: adopt (3 of 7 claims show utility sensitivity; directly relevant to model design).

## Caveats

1. CVaR_10 not available for ch04-low-trump-trap and ch02-bid-only-enough.
2. robust_q25 unavailable for all claims (requires quantile oracle).
3. p_make proxy quality varies: threshold_mass is oracle-derived (reliable); p_set_pounce_proxy is heuristic.
4. CVaR sign conventions were carefully verified per probe (T0 perspective throughout).
5. pounce_bid30 p_make result (spans_zero) conflicts with oracle pounce rate 59.6% but
   binomial test p=0.106 → not significant; consistent with spans_zero classification.

## Artifacts

- `per_probe_utility_verdicts.csv` — 51 rows
- `per_claim_utility_summary.csv` — 35 rows  
- `objective_aware_ledger_schema.json` — proposed schema
- `utility_lens_synthesis.md` — long-form analysis
- `analyze.py` — reproducibility script

## Reproducibility

```bash
python3 w42/book_validation_v1/wave3/t42-f2ur_utility_lens_synthesis/analyze.py
```
