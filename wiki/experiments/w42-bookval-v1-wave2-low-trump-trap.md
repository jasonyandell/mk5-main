---
title: w42 Book Validation v1 Wave2 — Low Trump Trap (ch04)
kind: experiment
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: complete
bead: t42-jysl
wave: wave2
epic: t42-4zi6
---

## Summary

Wave 2 probe `t42-jysl` tests the [[winning42-ch04-partner-support]] claim
that playing a *low* trump when the partner leads or a count tile is dumped
can be a trap — the opposition's dominant trump then captures the count.

**Status proposal: context-limited.**

257 paired contrasts from 300 oracle-greedy snapshots. The overall oracle
prefers the low trump over the dominant trump (mean EV delta = −1.95,
CI [−2.94, −1.07] entirely negative), primarily in non-count situations
where hoarding the dominant trump for later is the correct play.
In the book's specific scenario — a count tile already in the trick (N=86) —
the effect is near zero (mean = −0.02, CI [−2.16, +2.15]), crossing zero
with no reliable directional signal. The book claim cannot be confirmed or
contradicted at this sample size for that subgroup.

## Claim Under Test

Ch04 claims: when partner leads or a count tile is dumped, playing a *low*
trump is a trap. The partner side (or opposition) may then play their dominant
trump over the bidder's low trump, capturing the count.

Formal contrast tested:
- **A** = play the lowest-rank legally available trump (the "trap" move)
- **B** = play the dominant trump (highest-rank unplayed trump globally, held by bidder)
- **EV delta = Q(B) − Q(A)**: positive means dominant trump is better (book claim)

## Data

| Field | Value |
|---|---|
| Source | Oracle-greedy legacy corpus (`gus/data/`) |
| Filter | Bidder following, pip-trump decls, void in led suit or trump led, holds dominant + low trump |
| Snapshots loaded | 300 |
| Snapshots skipped | 43 (follow-suit prevents any trump choice) |
| Paired contrasts | **257** |
| Usable rate | 85.7% |

## Headline Results

| Metric | Value |
|---|---|
| Mean EV delta (B−A) | **−1.95** |
| 95% CI (bootstrap 1000 iter) | **[−2.94, −1.07]** |
| Dominant trump is better | 42.4% of cases |
| Low trump is better | 57.6% of cases |
| Trap-severe (delta > 5 pts) | 7.4% |
| Oracle chose dominant trump | 32.3% |
| Oracle chose low trump | 50.6% |
| Oracle chose non-trump | 17.1% |

### Count-in-Trick Subgroup (Book's Specific Scenario)

| Subgroup | N | Mean delta | 95% CI |
|----------|---|------------|--------|
| Count in trick | 86 | **−0.02** | [−2.16, +2.15] |
| No count in trick | 171 | −2.92 | [−3.98, −1.97] |

The count-in-trick CI straddles zero. No reliable directional effect detected
for the specific scenario the book describes.

### Phase Slice

| Phase | N | Mean delta | CI | Trap rate |
|-------|---|------------|----|-----------|
| Early (tricks 0–1) | 40 | −2.81 | [−5.39, +0.08] | 30% |
| Mid (tricks 2–5) | 181 | −1.69 | [−2.84, −0.52] | 48% |
| Late (tricks 6–7) | 36 | −2.30 | [−4.71, −0.48] | 28% |

## Interpretation

The oracle does **not** uniformly endorse throwing the dominant trump.
In the majority of filtered positions (57.6%), playing the low trump is
the better action by oracle-EV. This is consistent with standard 42 strategy:
hoarding the dominant trump for later tricks is often correct.

In count-specific situations (N=86), the delta is near zero and the CI crosses
zero. This means: when a count tile is already in the trick, the oracle shows
no consistent preference between low and dominant trump at this sample size.
The book's "trap" may be real in specific sub-configurations that require
finer segmentation than is available here (e.g., opponent's trump holding,
count magnitude, trick position within hand).

## Status

**context-limited**

The claim is not contradicted in its core scenario but is not confirmed. The
evidence base for the count-specific subgroup (N=86) is underpowered for the
near-zero effect observed. Overall, the oracle prefers low trump broadly,
which reflects situations outside the book's specific warning scope.

## Artifacts

| File | Path |
|------|------|
| Paired contrasts | `w42/book_validation_v1/wave2/probes/t42-jysl_low_trump_trap/paired_contrasts.csv` |
| Slice breakdown | `w42/book_validation_v1/wave2/probes/t42-jysl_low_trump_trap/slice_breakdown.csv` |
| Summary | `w42/book_validation_v1/wave2/probes/t42-jysl_low_trump_trap/summary.json` |
| Manifest | `w42/book_validation_v1/wave2/probes/t42-jysl_low_trump_trap/manifest.json` |
| Analysis script | `w42/book_validation_v1/wave2/probes/t42-jysl_low_trump_trap/run_analysis.py` |

Reproducibility:
```bash
cd /Users/jason/code/mk5-main
python w42/book_validation_v1/wave2/probes/t42-jysl_low_trump_trap/run_analysis.py
```

## Caveats

1. Q values from oracle-greedy corpus, not live model. Represent E[V] under
   optimal play from this state forward.
2. "Dominant trump" = highest-rank unplayed trump globally that bidder holds.
3. "Low trump" = lowest-rank legally playable other trump in bidder's hand.
4. Follow-suit law excluded 43/300 snapshots (trump play not an option there).
5. Count-in-trick N=86 is below the n=100 comfortable power floor for small effects.
6. No simulation of subsequent plays — purely single-decision EV comparison.

## Related

- [[winning42-ch04-partner-support]] — chapter source
- [[w42-book-validation-campaign]] — wave overview
- Campaign rules: `w42/book_validation_v1/AGENTS.md`
