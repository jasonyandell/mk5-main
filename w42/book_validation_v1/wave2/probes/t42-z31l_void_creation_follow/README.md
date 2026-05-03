# t42-z31l: Void Creation — Follow Position

**Wave:** 2  
**Bead:** t42-z31l  
**Parent epic:** t42-4zi6  
**Claim tested:** ch05-void-creation (follow-position sub-claim)

## Question

Does a setter voiding themselves in a non-led, non-trump suit by discarding their last
tile from that suit (while following a non-trump trick they cannot follow) improve their
setting prospects, compared to preserving that singleton suit by playing a different
off-suit discard?

The book's canonical scenario: setter cannot follow a non-trump lead, holds exactly 1 tile
in some other off-suit, and discards it to create a future trump-in window.

## Slice

- **Corpus:** oracle-greedy legacy corpus (`gus/data/corpus_train_chunk_*.pt`, 3 of 100 chunks)
- **Declarations:** all 10 (decl_id 0–9)
- **Role:** setter (player 1 or 3)
- **Position:** following (trick has ≥1 play before setter's turn)
- **Led suit:** non-trump only
- **Setter constraint:** cannot follow led suit; holds exactly 1 tile in some non-led, non-trump suit (singleton)
- **Alternative constraint:** must have at least 1 other legal play from a different suit
- **Trick filter:** trick number ≥1
- **Bid value:** 30 (all corpus games)

## N

- Candidates scanned: 7,827 decision points
- Filter hit rate: 6.39%
- Snapshots collected: 500 (hit target cap)
- Snapshots with valid paired contrast: 500
- **Paired contrasts run: 500**

## Metric

For each snapshot, two continuations are evaluated using the E[Q] oracle
(100 world samples per branch, MPS device):

- **A (preserve):** play a tile from a different non-led, non-trump suit (keeps singleton alive)
- **B (void):** play the singleton tile (voids that suit)

Deltas are B − A from setter perspective (positive = void creation is better for setter).

| Metric | Mean | 95% CI |
|--------|------|--------|
| EV delta (setter) | **+0.77** | [+0.12, +1.42] |
| p_set delta | +0.0087 | [−0.0018, +0.0192] |
| CVaR_10 delta | +0.17 | — |
| threshold_mass delta | −0.0087 | — |

52.6% of contrasts favor void creation by EV (setter perspective).

## Slice breakdown

| Slice | N | EV delta (setter) | p_set delta | % void better |
|-------|---|-------------------|-------------|---------------|
| phase=early | 208 | +0.37 | +0.0067 | 52.9% |
| phase=mid | 292 | +1.05 | +0.0102 | 52.4% |
| count_exposed=True | 30 | +0.46 | +0.0633 | 43.3% |
| count_exposed=False | 470 | +0.79 | +0.0052 | 53.2% |
| bidder_winning_trick | 326 | +0.79 | +0.0051 | 52.8% |
| setter_winning_trick | 174 | +0.72 | +0.0155 | 52.3% |

## Claim-ledger impact

**context-limited**

EV CI excludes zero on the positive side ([+0.12, +1.42]), weakly supporting the book
direction. However, the p_set CI straddles zero ([−0.0019, +0.0192]), and the fraction
voting for void creation (52.6%) is only marginally above chance. The benefit is real but
small and noisier than would qualify as "supported". The follow-position scenario differs
strikingly from the lead-position result (Wave 2.C: contradicted, EV delta = −2.63):
here the oracle finds a slight positive value to voiding, but the signal is not robust
enough to endorse the book claim without qualification.

## Caveats

1. All corpus games use bid=30; high-bid regimes (bid ≥ 35) are untested.
2. The oracle is greedy-oracle trained; Burl-level play may differ.
3. The "preserve" action (A) was selected as the lowest-pip-sum tile from the largest
   multi-tile off-suit — this is one reasonable alternative, not all alternatives.
4. No late-game cases (trick ≥ 6) passed the filter (all caught by trick_number ≥1
   but the corpus did not yield late-game follow-position voids at these conditions).
5. The count_exposed slice (N=30) is small; p_set delta there (+0.063) is suggestive
   but not powered for a sub-slice verdict.

## Artifacts

| File | Description |
|------|-------------|
| `paired_contrasts.csv` | 500 row-level paired contrast results |
| `slice_breakdown.csv` | Slice-level aggregates |
| `summary.json` | Headline numbers, machine-readable |
| `manifest.json` | Provenance, SHA256, command |
| `run_void_creation_follow_probe.py` | Reproducibility script |

## Command

```bash
python w42/book_validation_v1/wave2/probes/t42-z31l_void_creation_follow/run_void_creation_follow_probe.py \
  --snapshots w42/book_validation_v1/wave2/snapshots/void_creation_follow/snapshots.jsonl \
  --checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt \
  --samples 100 \
  --device mps \
  --output-dir w42/book_validation_v1/wave2/probes/t42-z31l_void_creation_follow/
```
