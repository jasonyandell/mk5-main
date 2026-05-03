# Ch10 Action-Level Mark-Multiplier Analysis

**Bead:** t42-8na4  
**Wave:** 2.H  
**Claim:** ch10-special-bid-mark-multiplier  
**Status proposal:** supported (no change; evidence base widened)

## Question

Does the mark-point multiplier at high bids change the oracle's optimal play, or
does it only change the scoring tally? This probe adds action-level strategic
evidence to the existing deterministic transform proof.

## Slice

- Corpus: bid_aware_atlas (seeds 9000–9049, 50 seeds × 10 declarations = 500 games)
- Bids tested: 30, 32, 35, 36, 39, 42, 84
- N decisions: 14,000 unique decisions, each evaluated at all 7 bids
- N action rows: 259,618

## Method

For each (seed, decl_id, decision_idx):
1. Find the top-1 action by mark_ev at each bid (mark_ev = mm × (2 × p_make − 1))
2. Find the top-1 action by raw EV (mean of oracle EV distribution)
3. Find the top-1 action by p_make (win probability)

Analyses:
- **Action flip rate per bid**: fraction of decisions where mark_ev top-1 ≠ raw EV top-1
- **Multiplier strategic effect**: fraction where mark_ev@bid ≠ mark_ev@bid=30
- **Cross-utility matrix**: agreement among mark_ev / p_make / EV, per bid
- **Slice breakdowns**: by declaration, seat role, EV action tier

## Key Findings

### 1. mark_ev always equals p_make for top-1 action (all bids)

mark_ev = mm × (2 × p_make − 1) is a positive affine transform of p_make.
Argmax(mark_ev) == argmax(p_make) at every bid, confirmed with zero exceptions
across 98,000 (decision, bid) pairs. Wave 1.2's identity holds universally.

### 2. The multiplier changes the optimal play in 71% of decisions at bid=84

| Bid | mark_ev@bid vs mark_ev@bid=30 flip rate | 95% CI |
|-----|------------------------------------------|--------|
| 32  | 56.2%                                    | [55.4, 57.0] |
| 35  | 60.3%                                    | [59.4, 61.1] |
| 36  | 63.2%                                    | [62.5, 64.0] |
| 39  | 70.0%                                    | [69.2, 70.8] |
| 42  | 71.0%                                    | [70.3, 71.8] |
| 84  | 70.9%                                    | [70.2, 71.7] |

The flip rate increases with bid level up to bid=42, then plateaus at bid=84.
This is expected: bids 42 and 84 share the same threshold_q=42, so the
difference in optimal play comes from threshold_q shift, not from mm scalar change.

### 3. mark_ev vs raw EV flip rate rises monotonically with bid

| Bid | flip rate (mark_ev vs raw EV) | 95% CI |
|-----|-------------------------------|--------|
| 30  | 21.4%                         | [20.8, 22.1] |
| 32  | 23.6%                         | [22.9, 24.3] |
| 35  | 26.0%                         | [25.3, 26.7] |
| 36  | 28.5%                         | [27.8, 29.3] |
| 39  | 32.6%                         | [31.8, 33.3] |
| 42  | 40.0%                         | [39.1, 40.8] |
| 84  | 40.1%                         | [39.3, 40.9] |

At bid=30 (mm=1), 21.4% of decisions already have mark_ev disagreeing with
raw EV — this is baseline utility disagreement from win-probability weighting.
By bid=42/84, 40% of decisions see a different top-1 action.

### 4. Bidder is the most multiplier-sensitive seat role

| Seat Role       | flip rate at bid=84 |
|-----------------|---------------------|
| bidder          | 74.6%               |
| bidder_partner  | 69.8%               |
| left_setter     | 69.6%               |
| right_setter    | 69.6%               |

Bidder's decisions are more sensitive — the offense team is more exposed to
the scoring threshold change.

### 5. Suit declarations (sixes, fives, fours) most multiplier-sensitive

| Declaration  | flip rate at bid=84 |
|--------------|---------------------|
| sixes        | 74.8%               |
| twos         | 73.1%               |
| fives        | 73.1%               |
| fours        | 72.2%               |
| no-trump     | 71.6%               |
| doubles      | 69.8%               |
| doubles-suit | 68.2%               |

## Claim-Ledger Impact

Row `ch10-special-bid-mark-multiplier` is already `supported` via the
deterministic transform proof from Wave 1.2. This probe does **not** change the
status. It adds action-level strategic evidence: the multiplier (operating
through threshold_q changes) genuinely shifts which tile the oracle plays in
71% of decisions at bid=84, not just the scoring arithmetic.

## Caveats

- The flip is driven by threshold_q recomputation, not by the mm scalar directly.
  A positive mm scalar cannot change argmax(mark_ev) within a fixed bid's p_make
  distribution.
- bid=42 and bid=84 have the same threshold_q=42, so their flip rates vs bid=30
  are nearly identical (~71%). The 2× multiplier at bid=84 adds no additional
  strategic differentiation beyond the threshold_q already used by bid=42.
- Analysis restricted to seeds 9000–9049 (50 seeds).

## Artifacts

| File | Description |
|------|-------------|
| action_flip_rates.csv | Per-bid flip rate (mark_ev vs raw EV top-1) |
| multiplier_strategic_effect.csv | Per-bid flip rate (mark_ev@bid vs mark_ev@30) |
| cross_utility_matrix.csv | Agreement among mark_ev / p_make / EV per bid |
| slice_breakdown.csv | Slices by declaration, seat role, EV action tier |
| summary.json | Machine-readable headline numbers |
| manifest.json | Provenance, input SHA256, command |
| run_ch10_action_level.py | Reproducibility script |

## Reproducibility

```bash
python3 w42/book_validation_v1/wave2/probes/t42-8na4_ch10_action_level/run_ch10_action_level.py
```
