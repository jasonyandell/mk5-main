# t42-26j8 Void Creation Probe

## Question

Does a setter gain by deliberately voiding themselves in an off-suit — leading
their last tile from a singleton suit — so they can later trump in or punish a
count tile led to that suit? (Book claim: ch05-setter-defense, void-creation
section.)

## Slice

- Corpus: oracle-greedy legacy corpus (`gus/data/corpus_train_chunk_*.pt`)
- Role: setter (player 1 or 3; bidder is always player 0 in legacy corpus)
- Position: on lead (trick_plays empty), trick >= 1
- Condition: setter holds >= 2 distinct suits, at least 1 suit is a singleton
  (one tile remaining — the void-creation candidate)
- Declarations tested: all 10 (pip trumps 0-6, doubles-trump, doubles-suit,
  no-trump)
- Bid value: 30 (fixed by legacy corpus)

## N

- Input snapshots: 500
- Valid for GameStateTensor: 500
- Paired contrasts produced: 276
  (224 skipped: 45 had no singleton suit, 179 had no multi-tile preserve candidate)

## Methodology

For each snapshot, two actions are compared:

- **A (preserve suit)**: lead the lowest-sum tile from the suit with the most
  tiles (never the singleton suit, avoiding trump when possible).
- **B (create void)**: lead the lowest-pip-sum tile from the smallest singleton
  non-trump suit, voiding the setter in that suit.

`generate_eq_from_snapshots` is called independently for each action (100 world
samples per decision, MRV sampler, greedy continuation). The E[Q] and full Q-PDF
are read from `decisions[0]` at the action slot.

All Q values are in Team 0 (bidder) perspective. For setter analysis:
- `ev_delta_setter = -(ev_b_t0 - ev_a_t0)` — positive = void creation benefits setter
- `p_set_delta = p_set(B) - p_set(A)` — positive = void creation raises bidder failure prob

## Metrics

| Metric | Value | 95% CI |
|--------|-------|--------|
| N pairs | 276 | — |
| p_set delta (B-A) | **-0.0155** | [-0.028, -0.003] |
| EV delta setter (B-A) | **-2.63** | [-3.42, -1.84] |
| % contrasts void-creation better (EV) | 32.6% | — |
| CVaR_10 delta | -0.213 | — |
| threshold_mass delta | +0.016 | — |

## Slice Breakdown

| Slice | N | p_set delta | EV delta (setter) | % void better |
|-------|---|-------------|-------------------|---------------|
| phase=early | 156 | -0.0165 | -3.79 | 28.8% |
| phase=mid | 120 | -0.0143 | -1.13 | 37.5% |
| bidder_count_exposure=count_exposed | 18 | -0.0528 | -3.28 | 22.2% |
| bidder_count_exposure=no_count | 258 | -0.0129 | -2.58 | 33.3% |

All suit_type values are `off_suit` (by corpus filter design).

## Status Proposal

**contradicted**

Both p_set delta and EV delta (setter perspective) are negative, with 95% CIs
that exclude zero: oracle-greedy play systematically favors preserving suit over
creating a void. The effect is consistent across phases (early and mid) and count-
exposure sub-groups. Void creation does not help the setter in the positions
sampled.

## Caveats

1. **Oracle-greedy baseline only.** Snapshots are from greedy oracle play, not
   human games. The oracle may have already optimized its subsequent play to not
   require the void — the paired contrast only measures first-move choice.

2. **Leading position only.** All 500 snapshots capture the setter LEADING a
   trick (won the prior trick). The book's primary void-creation scenario may
   be during a following (off-suit discard) position, which is not represented here.

3. **Singleton suite definition.** The mining criterion requires exactly 1 tile
   in the voided suit. Voids created from 2-tile suits (playing one, leaving one)
   are not captured.

4. **Bid value fixed at 30.** All p_set metrics use bid_value=30. Games with
   higher bids (31, 32+) are not tested.

5. **No late-game phase.** The corpus filter requires trick >= 1; most positions
   fall in early/mid. Late-game (tricks 6-7) is absent from the slice breakdown
   due to insufficient samples.

6. **N=18 for count_exposed sub-group.** The count-exposure slice is small and
   should be treated with caution.

## Artifacts

```
w42/book_validation_v1/wave2/probes/t42-26j8_void_creation/
├── README.md                  (this file)
├── manifest.json              (provenance + SHA256 of snapshot file)
├── summary.json               (headline numbers + slice breakdown)
├── paired_contrasts.csv       (276 row-level pair records)
├── slice_breakdown.csv        (per-slice summary rows)
└── run_void_creation_probe.py (probe script)
```

## Reproducibility Command

```bash
python w42/book_validation_v1/wave2/probes/t42-26j8_void_creation/run_void_creation_probe.py \
    --snapshots w42/book_validation_v1/wave2/snapshots/void_creation/snapshots.jsonl \
    --checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt \
    --samples 100 \
    --device mps \
    --output-dir w42/book_validation_v1/wave2/probes/t42-26j8_void_creation
```
