---
title: "W42 Book Validation v1 — Wave 1 Mark Utility Transform"
kind: experiment
status: complete
bead: t42-c6sa
parent_epic: t42-4zi6
wave: wave1
first_seen: 2026-05-03
last_updated: 2026-07-13
---
## Summary

Applying the deterministic Chapter 10 mark/match utility transform to the existing
branch_atlas_scaled_v0 Q PDFs produces a **23.2% top-1 action flip rate** (65/280
decisions). However, **81.5% of those flips have zero mark gain** — they are
surface-flattening artifacts of the binary mark threshold, not strategic preference
changes. Only **12 decisions (4.3% of total)** show genuine objective flips where
the mark-preferred action has a higher probability of winning the hand under marks,
at a mean point-EV cost of 1.32 points and a mean mark-gain of 0.030 marks
(10 of the 12 are detector-endorsed; that subset has mean cost 1.46 pts, mean gain 0.032).

The book's claim that tournament (marks) play demands different decisions from money
(points) 42 does show up at the action-ranking level, but only modestly at bid=30.
The effect is clearest for early-hand decisions by the bidder and bidder_partner,
and in the no-trump and sixes declarations.

## Slice

- Corpus: `branch_atlas_scaled_v0` — single seed (9430), 10 declarations, bid=30
- N decisions: 280 (28 per game × 10 games)
- N legal actions: 773
- N worlds per decision: 1000
- Bid value: 30 throughout (mark multiplier = 1; no special-bid effects)

## Method

The deterministic mark scoring transform from [[w42-phase4-scoring-objective-tests]]
is applied per-world to the `q_per_world` tensors in
`eq_pdf_s9430_d10_bid30_joint_1000s_v2.pt`.

**Q semantics:** `q_per_world[w,a]` is the remaining (team0 - team1) point
differential from taking action `a` in world `w`. This is *not* raw capture totals;
it is a net differential.

**Transform:**
1. `remaining_total = 42 - pre_t0 - pre_t1`
2. `remaining_t0[w,a] = (q_per_world[w,a] + remaining_total) / 2`
3. `final_t0[w,a] = pre_t0 + remaining_t0[w,a]` (clamped 0..42)
4. Apply `score_hand_marks(bid, bidder_team, (final_t0, 42-final_t0))` (verbatim reuse)
5. `mark_utility[w,a] = mark_team0 - mark_team1`
6. `mark_EV[a] = mean over 1000 worlds of mark_utility[w,a]`

The per-world hook is fully maintained: mark_utility is an (n_worlds, n_actions) tensor
before the mean, satisfying the constraint in the agent task brief.

## Findings

### Headline flip rate: 23.2%

65 of 280 decisions flip top-1 action when scored under mark EV instead of point EV.
54 of 65 flips (83%) have at least one book position detector active at that decision.

### True objective flips vs surface-flattening flips

The binary mark transform collapses a continuous 42-point outcome space into a binary
+1/−1 mark outcome. Near the 30-point make threshold, small Q differences may not change
whether a world is "made" or "set", producing near-identical mark EV across actions.

- **12 decisions (18.5%)**: positive mark_gain — genuine preference shift
- **53 decisions (81.5%)**: mark_gain = 0 — tie-breaking artifact of the flat binary surface

For the genuine flips: mean EV-cost = 1.32 pts, mean mark-gain = 0.030 marks.
Mark utility is the ±1 (team0 − team1) mark differential, so a 0.030 mean gain is
~1.5 percentage points better odds of winning the hand under marks.

### Declaration-level finding: no-trump is most sensitive

No-trump has a 46.4% flip rate (13/28 decisions), far above the mean. Under no-trump,
there is no dominant trump sequence, so actions near the 30-point make boundary are
more exchangeable. The mark transform amplifies this near-boundary sensitivity.

Blanks (7.1%) has the lowest flip rate — trump dominance locks in the make/set outcome
earlier, reducing the mark-vs-points preference divergence.

### Seat/role finding: bidder flips most

The bidder has a 32.9% flip rate. The bidder is making decisions that directly affect
whether the contract is made, so the bidder's actions sit closest to the make threshold
in more worlds. Setters flip less (14-19%).

### Ch10 claim correlations

From [[winning42-ch10-tournament-scoring]]:

| Ch10 claim | Flip enrichment |
|-----------|----------------|
| ch10-early-terminal-under-marks | **1.21x** (enriched) |
| ch10-tournament-speed-tradeoff | **1.21x** (enriched) |
| ch10-timed-marks-advancement-objective | **1.21x** (enriched) |
| ch10-nonbidder-partial-points-erased | **1.13x** (enriched) |
| ch10-set-severity-compression | 0.95x (neutral) |
| ch10-special-bid-mark-multiplier | **0x** (not activated, bid=30) |

The early-terminal enrichment is the clearest signal: decisions at mid-hand (where
tricks have been played but the hand is not yet resolved) flip more often, consistent
with the book's claim that marks create early-terminal pressure.

### Top 5 detector-endorsed flips

1. **game 0, decision 18, blanks, bidder, mid_hand**: 10.2pt cost, 0.0 mark gain.
   `late_trick_threshold_closure` active. Surface-flattening artifact at the largest
   cost in the dataset.

2. **game 1, decision 23, ones, bidder, mid_hand**: 7.6pt cost, 0.0 mark gain.
   `last_to_act_closure_policy|late_trick_threshold_closure` active. Tie-breaking flip.

3. **game 4, decision 7, fours, bidder_partner, early_hand**: 3.8pt cost, **0.070 mark gain**.
   `last_to_act_closure_policy` active. **Genuine objective flip** — partner-third-seat
   decision where a count-safe line sacrifices point EV but improves make odds by
   3.5 percentage points (0.070 in ±1 mark-utility).

4. **game 7, decision 17, doubles, left_setter, mid_hand**: 3.7pt cost, 0.0 mark gain.
   `doubles_regime_plan|doubles_trump_regime` active. Doubles-as-trump creates a
   surface-flattening effect.

5. **game 6, decision 4, sixes, left_setter, early_hand**: 3.4pt cost, **0.028 mark gain**.
   `setter_lead_pressure|defender_damage_lead_class` active. **Genuine setter flip**
   — opening lead that forfeits 3.4 pts of EV improves setter's mark-winning odds by
   1.4 percentage points (0.028 in ±1 mark-utility).

### Blocker finding: ch10-special-bid-mark-multiplier cannot be tested here

All 10 games use bid=30 (mark multiplier = 1). The special-bid effects (84x gives 2 marks,
126x gives 3 marks, 168x gives 4 marks) documented in [[winning42-ch10-tournament-scoring]]
and validated in [[w42-phase4-scoring-objective-tests]] are structurally inactive.
A follow-up run against a mixed-bid atlas (bid ∈ {42, 84, 126, 168}) is required
to test whether high-stakes special bids produce qualitatively different flip rates.

## Caveats

1. Single seed (9430), bid=30 only. No generalization claim across seeds or bid values.
2. 81.5% of flips are surface-flattening artifacts; the strategic content is in the
   12 genuine mark_gain > 0 flips.
3. The `joined_claim_action_rows.csv` corpus (seeds 0-9, corpus_v2_train) does not
   overlap with branch_atlas_scaled_v0 (seed 9430), so full feature-label join was
   not possible. Book endorsement uses `matched_position_detectors` from the atlas CSV.
4. Claim-ledger impact: `underpowered` — n=280 decisions from one seed and one bid
   value is insufficient to promote any Ch10 claim to `supported` at the action-ranking
   level. Evidence is directionally consistent but not powered.

## Artifacts

| Path | Description |
|------|-------------|
| `w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/run_mark_utility_transform.py` | Analysis script |
| `w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/manifest.json` | Provenance + SHAs |
| `w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/summary.json` | Headline numbers |
| `w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/action_mark_ev_scalars.csv` | Per-action point/mark EV |
| `w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/flip_rate_by_slice.csv` | Flip rate slices |
| `w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/detector_endorsed_flips.csv` | 54 endorsed flips |
| `w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/per_ch10_claim_correlation.csv` | Ch10 claim correlations |

## Provenance

- Transform source: `w42/phase4_scoring_objective_tests/run_phase4_scoring_objective_tests.py`
  (SHA256: a43adfe5e5a052b47945fb14d703da0f2ab00080d8710ca64367e7a1141f198d)
- Input PT: `w42/branch_atlas_scaled_v0/eq_pdf_s9430_d10_bid30_joint_1000s_v2.pt`
  (SHA256: c0d124b0733dedd08227de7a3a4ca8cb200a9900d585721d654b461b798af9b8)
- Repo commit: 12064bf1e67052bb1515fe59f4393104f2a3766b
- Reproduce: `source forge/venv/bin/activate && python3 -u w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/run_mark_utility_transform.py`

## Links

- [[w42-phase4-scoring-objective-tests]] — transform source and Ch10 claim evidence
- [[winning42-ch10-tournament-scoring]] — book source for mark/match utility claims
- [[w42-book-claim-synthesis-and-ai-directions]] — synthesis and AI directions
- [[w42-phase4-claim-completion-board]] — claim ledger baseline
- [[w42-book-validation]] — wave overview route
