---
title: w42 Champion Teaching Battery — Wave 5
kind: experiment
status: complete
wave: wave5
task_id: wave5-champion-teaching-battery
first_seen: 2026-06-13
last_updated: 2026-06-13
---

# w42-champion-teaching-battery

## Summary

**Question:** Does the champion (GusBidder + GusPointsEvaluator auction, lens:ev play)
obey the Winning 42, 2nd ed. tactical claims in its own play trajectories?

This is the pedagogical north star question for the book-validation campaign's "teaching half":
rather than asking whether the book is right in general, we ask whether the champion — the
best player we can run — independently arrives at the book's recommendations.

**Headline (128 seeds × 2 halves, n_worlds=10):**

The champion agrees with the book on 3 of 6 checkable ch04/ch05 tactical claims.
Both contradicted claims are negative controls (reckless count donation is worse,
unsafe partner count is worse), which confirms the champion correctly avoids those
patterns. The within-CI result (safe partner donation) is essentially a draw.

## Slice

- **Seeds:** 0–127 (128 deal seeds)
- **Halves:** 2 (champion as absolute team 0 and team 1 on each seed)
- **Games:** 256 total
- **Decisions:** 7,168 (28 per game average)
- **Action rows:** 19,264 (all legal candidates at every decision)
- **Labeled rows:** 1,484 with at least one tactical label
- **n_worlds per decision:** 10 (MPS device, fast path)
- **Model:** `domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`

## Claims tested

The 6 GUS_TACTICAL_SPECS ch04/ch05 play claims, re-run on champion data.
This is a clean replication pipeline: the same harness, same labels, same
bootstrap methodology as the prior Gus corpus runs.

## Results

| claim_id | obey_rate | N_paired | delta (pts) | 95% CI | verdict |
|----------|-----------|----------|-------------|--------|---------|
| ch05-pounce-count | 0.593 | 117 | +4.93 | [+3.26, +6.73] | **supported-on-slice** |
| ch05-pounce-count-before-certainty | 0.604 | 87 | +3.95 | [+2.24, +5.72] | **supported-on-slice** |
| ch05-extra-count-to-set | 0.654 | 23 | +10.00 | [+4.65, +15.79] | **supported-on-slice** |
| ch05-reckless-count-to-bidder | 0.317 | 550 | -7.66 | [-8.62, -6.77] | **contradicted** (negative control) |
| ch04-safe-partner-count-donation | 0.571 | 150 | +0.05 | [-1.12, +1.20] | within-CI |
| ch04-unsafe-partner-count-donation | 0.320 | 201 | -9.21 | [-10.96, -7.69] | **contradicted** (negative control) |

### Label distribution

| label | action rows |
|-------|-------------|
| ch05_reckless_count_to_bidder | 812 (most common — negative control) |
| ch04_partner_unsafe_count_to_defense | 303 |
| ch04_partner_safe_count_donation_current_control | 224 |
| ch05_setter_pounce_count | 145 |
| ch05_setter_pounce_count_before_certainty | 111 |
| ch05_setter_pounce_count_sets_now | 26 |

## Narrative

**Champion agrees with book on setter-pounce claims (supported-on-slice).**
When a defender can take an offense-controlled trick with count, the champion
prefers to do so at a +4.9 pt E[Q] advantage (CI entirely above zero, N=117).
The "before certainty" variant (not the last player in the trick) shows the same
pattern at +3.95 pts (N=87). When the pounce would actually set the bidder right
now, the advantage grows to +10 pts (N=23, CI [+4.6, +15.8]).

**Negative controls confirm champion avoids reckless play.**
`ch05_reckless_count_to_bidder` (playing count into offense-won trick without
beating it) is strongly negative: -7.66 pts CI [-8.62, -6.77], N=550. The champion's
obey rate is only 32% — meaning the champion mostly declines to play recklessly, but
when it does, it is substantially worse. This validates the detector and confirms
the champion learns Roberson's "don't throw count to the bidder" rule.

Similarly, `ch04_partner_unsafe_count_to_defense` (partner plays count into defense-
controlled trick they can't beat) is -9.21 pts CI [-10.96, -7.69], N=201. The champion
avoids this pattern 68% of the time.

**Safe partner count donation is a draw (within-CI).**
The `ch04_partner_safe_count_donation_current_control` contrast shows +0.05 pts
CI [-1.12, +1.20], N=150. The book says to donate count when your partner is winning
the trick — the champion's lens:ev is indifferent. This may reflect that the claim
is context-limited (e.g., only valuable when you have nothing better to do with the
count domino, not when you have other high-value plays).

## Self-selection caveat

**Critical:** The champion plays its own lens:ev trajectories. Tactical scenarios
arise only when the game reaches those states under champion play. The battery
measures "does the champion's OWN play obey the book," not "is the book right in
general" — which IS the teaching-half question, but the slice must be stated honestly.

Specifically:
- The champion rarely reaches 84-contract territory (no 84 results observed).
- Ch07/Ch08 endgame claims (depth ≤ 4) are not tested here.
- Ch03/Ch10 multi-step claims are structurally out of scope for single-decision contrasts.
- Only ch04/ch05 claims (count donation and setter pounce) appear in meaningful quantity.

## Promotion guard

Paired same-decision contrasts are present for all 6 claims. However, **the central
ledger (`phase4_claim_completion_board/completion_board.csv`) was NOT modified** by
this probe. Reconciliation is the foreground orchestrator's job per AGENTS.md.

The supported-on-slice verdict for the 3 ch05 claims replicates the prior Gus corpus
finding and provides additional evidence, but formal ledger movement requires the
orchestrator's reconciliation step.

## Utility coverage

Per the AGENTS.md wave3 utility-coverage requirement, all 5 lenses are recorded:
- ev_setter / ev_bidder
- p_make_setter / p_make_bidder
- mark_ev_setter / mark_ev_bidder
- cvar_10_setter / cvar_10_bidder
- robust_q25_setter / robust_q25_bidder

## Reproducibility command

```bash
python -u w42/book_validation_v1/wave5/probe_champion_teaching_battery.py \
    --n-seeds 128 --n-samples 10 --device mps \
    --out-dir w42/book_validation_v1/wave5/champion_teaching_battery \
    --checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt
```

Smoke test (2 seeds, ~20s on Apple Silicon M5):
```bash
python -u w42/book_validation_v1/wave5/probe_champion_teaching_battery.py \
    --smoke-only --n-samples 10 --device mps
```

## Artifacts

| File | Description |
|------|-------------|
| `w42/book_validation_v1/wave5/champion_teaching_battery/action_rows.csv` | 19,264 per-action rows |
| `w42/book_validation_v1/wave5/champion_teaching_battery/receipts.csv` | Per-claim receipt table |
| `w42/book_validation_v1/wave5/champion_teaching_battery/summary.json` | Machine-readable headline |
| `w42/book_validation_v1/wave5/champion_teaching_battery/manifest.json` | Provenance |
| `w42/book_validation_v1/wave5/champion_teaching_battery/harness_output/` | Standard harness artifacts |
| `w42/book_validation_v1/wave5/probe_champion_teaching_battery.py` | Probe script |
| `w42/claim_analysis/registry.py` | CHAMPION_SPECS appended (wave5 addition) |

## What is out of scope

- 84-contract claims (ch07/ch08): champion rarely bids 84
- Multi-step sequence claims (ch03/ch10): require trajectory contrasts, not single-decision
- Auction claims (ch02 bid-only-enough): requires paired bid counterfactuals;
  descriptive auction stats in game_meta but not a paired contrast
- Static/ruleset claims (ch09, ch16): no dynamic data needed
- Any claim requiring opponent hands not held by the champion: structurally impossible
  under the self-selection slice
