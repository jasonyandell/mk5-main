# Dropped-30 eval: E[Q] n=10 defeats rob (exact info-set solver) at pure play

2026-07-30. Three independent instances (base seeds 1/2/3), 384 games each,
`bid30+rob` vs `bid30+lens:ev` (n_samples=10), mirrored pairs by construction
(paired halves, deal seeds team-assignment-independent, Bid30Bidder never
passes out). Every hand a forced 30 bid, declaration = best pip trump of the
forced bidder's hand, 1 mark per hand, games to 7. Engine scoreboard commits:
17807fd4 + offense/defense split. rob seated via the texas-42 rob_bridge
(subprocess line protocol); zero rules divergences across all ~180k decisions
(every reply cross-checks trick leader and team points).

## Headline

**E[Q] n=10 beats rob at dropped-30 play, ~6.5σ.**

| | seed 1 | seed 2 | seed 3 | pooled |
|---|---|---|---|---|
| rob games won | 182/384 | 163/384 | 180/384 | **525/1152 (45.6%)** |
| rob marks/game | −0.27 [−0.59,+0.06] | −0.57 [−0.87,−0.26] | −0.30 [−0.65,+0.02] | ≈ **−0.38** |
| rob pts/hand | −0.54 | −1.21 | −0.79 | ≈ **−0.85** |
| rob made (offense) | 33.0% | 31.1% | 33.3% | 32.4% |
| E[Q] made (offense) | 35.4% | 36.2% | 36.0% | 36.1% |

## Paired mirror analysis (the clean cut)

12,866 hands → 6,028 mirrored pairs (same deal, same forced contract, teams
swapped; unpaired straggler hands excluded).

- On identical deals: rob made **1,956 (32.4%)**, E[Q] made **2,176 (36.1%)**.
- 4,890 pairs concordant (deal decided). **1,138 discordant** — the pure
  play-skill signal: **E[Q] converted 679 deals rob couldn't; rob converted
  459 E[Q] couldn't.** Discordant share rob 40.3% [37.5%, 43.2%];
  McNemar z = −6.52.
- Make-rate edge rob−E[Q]: **−3.65pp [−4.75, −2.55]** per contract.
- The gap is negative in **all seven declarations** (largest in fours:
  152 vs 246 discordants / 2,169 pairs).
- rob also loses raw points/hand in all three seeds — E[Q] wins on rob's own
  objective (point differential), not just at the make/set boundary.

## Interpretation (held loosely)

"Exact" is exact *given an opponent model*. rob computes the true info-set
best response under his model of the hidden seats; E[Q] assumes every sampled
world plays double-dummy (strategy fusion bias, known). Facing each other,
E[Q]'s approximation is the better prior in this protocol: whatever rob's
opponent/partner model gives up against a PIMC field costs more than
strategy fusion costs E[Q]. Where it lives (offense vs defense, count
management, declaration structure) is recoverable from per_hand.csv.

Context: the 2026-07-29 24-game heuristic-auction probe (+0.46 marks/game,
CI crossing zero) was noise with the wrong sign. The walt→E[Q] meeting
(mk5 #72) registered the prior "edge shrinks hard, stays positive" — the
full-hand exact-solver analogue just came back **negative**.

Caveat: this measures the dropped-30 random-deal distribution, pip-trump
declarations only, and rob's Points lens at the normative window budget
(B=2^28) with budget-overflow decisions routed to the response-class
counting engine. A different lens, budget, or realistic contract
distribution could move it.

Artifacts: dropped30_384{,_s2,_s3}/{summary.json,per_hand.csv,per_game.csv};
paired analysis script in the Claude job dir (paired_analysis.py).
