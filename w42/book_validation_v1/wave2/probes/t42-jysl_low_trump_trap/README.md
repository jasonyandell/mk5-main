# Probe t42-jysl: Low Trump Trap (ch04)

**Bead:** t42-jysl  
**Wave:** wave2  
**Campaign:** W42 Book Validation v1 (epic t42-4zi6)

## Question

Does playing a *low* trump instead of the *dominant* trump cost the bidder
significant EV in trick-following situations — the "low trump trap" described in
ch04 of Winning 42?

Specifically: when the bidder is following (not leading), holds the highest
unplayed trump globally plus at least one lower trump, and is either void in
the led suit or following a trump lead — which action gives higher oracle EV?

## Slice

- **Role:** Bidder (player 0)
- **Position:** Following in the trick (not on lead), trick number ≥ 1
- **Declarations:** Pip-trump only (decl_id 0–6)
- **Condition:** Bidder void in led non-trump suit, OR led suit is trump; bidder holds dominant trump AND ≥ 1 other (low) trump
- **Source:** Oracle-greedy legacy corpus, 300 filtered snapshots

## N

- 300 snapshots consumed
- 43 skipped (follow-suit law prevented bidder from playing any trump — vacuous for claim)
- **257 paired contrasts** (85.7% usable rate)

## Method

**Paired/unpaired:** Paired  
Within-snapshot contrast using oracle E[Q] values embedded in each snapshot:
- **A (low trump):** Play the lowest-ranking trump legally available to the bidder
- **B (dominant trump):** Play the highest-ranking unplayed trump (globally), which the bidder holds

**EV delta = Q(B) − Q(A).** Positive delta means dominant trump is better (book claim territory).

No live model query — Q values come from oracle-greedy corpus metadata (`_e_q`, `_legal_mask`).

## Metrics

| Metric | Value |
|--------|-------|
| N paired contrasts | 257 |
| Mean EV delta (B−A) | **−1.95** |
| 95% CI (bootstrap, 1000 iter) | **[−2.94, −1.07]** |
| % dominant trump is better (δ > 0) | 42.4% |
| % low trump is better (δ < 0) | 57.6% |
| Trap-severe rate (δ > 5 pts) | 7.4% |
| Oracle chose dominant trump | 32.3% |
| Oracle chose low trump | 50.6% |
| Oracle chose non-trump | 17.1% |

### Count-in-trick subgroup (book's specific scenario)

| Metric | Count in trick (N=86) | No count in trick (N=171) |
|--------|-----------------------|---------------------------|
| Mean EV delta | **−0.015** | −2.921 |
| 95% CI | **[−2.16, +2.15]** | [−3.98, −1.97] |

The count-in-trick slice (N=86) is the closest match to the book claim. Its CI
straddles zero with a near-zero mean, yielding no reliable directional signal.

### Phase breakdown

| Phase | N | Mean delta | CI | Trap rate |
|-------|---|------------|----|-----------|
| early | 40 | −2.81 | [−5.39, +0.08] | 30% |
| mid | 181 | −1.69 | [−2.84, −0.52] | 48% |
| late | 36 | −2.30 | [−4.71, −0.48] | 28% |

## Status Proposal

**context-limited**

**Reason:** The overall CI [−2.94, −1.07] is entirely negative, meaning oracle
prefers the low trump globally. This reflects non-count situations where hoarding
the dominant trump for future tricks is the correct strategy. In the book's
specific scenario — count tile in the trick — the effect is near zero
(mean = −0.02, CI [−2.16, +2.15]), with no directional signal at N=86.
The book claim that "low trump can be a trap" is not contradicted in count
context but is not confirmed either; the evidence is context-limited and
underpowered for that subgroup.

## Caveats

1. Q values from oracle-greedy corpus (not live model). They represent E[V] under
   optimal subsequent play by both teams.
2. Purely within-snapshot contrast. No simulation of what happens next.
3. "Dominant trump" = highest-ranking unplayed trump globally held by bidder.
4. "Low trump" = worst (lowest-rank) legally playable other trump.
5. 43/300 snapshots (14.3%) skipped: follow-suit prevents any trump choice.
6. Count-in-trick subgroup N=86 is borderline for detecting small effects.
7. Oracle chose a non-trump in 17.1% of cases (preferred neither arm).
8. Overall negative delta is driven by non-count situations where holding the
   big trump for later is correct — the book never said to always lead dominant.

## Artifacts

| File | Description |
|------|-------------|
| `paired_contrasts.csv` | 257 rows, one per paired contrast |
| `slice_breakdown.csv` | Phase / trump-proximity / count breakdown |
| `summary.json` | Machine-readable headline numbers |
| `manifest.json` | Provenance + artifact SHAs |
| `run_analysis.py` | Reproducible analysis script |

## Reproducibility

```bash
cd /Users/jason/code/mk5-main
python w42/book_validation_v1/wave2/probes/t42-jysl_low_trump_trap/run_analysis.py
```

## References

- [[winning42-ch04-partner-support]] — Ch04 claim source
- Snapshot corpus: `w42/book_validation_v1/wave2/snapshots/low_trump_trap/`
- Campaign rules: `w42/book_validation_v1/AGENTS.md`
