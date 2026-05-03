---
title: w42 Claim Analysis Synthesis Report
kind: experiment
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: active
---

## Summary

[[w42]] phase 2, bead `t42-0b4l`, is complete as a recovery and claim-analysis
program. The crashed session's mission was reconstructed, the 111 GB [[gus]]
seed corpus was inventoried, every claim family received either a powered test,
a generated fixture probe, a reusable artifact, or a precise blocker, and the
next epic is now `t42-qtwb`.

The central lesson is that the book is useful as detector vocabulary, but broad
strategy claims only become evidence when the operationalization is narrow. The
best-supported phase-2 claims are action-local: setter pounce, reckless count
negative controls, closure taking, and no-trump double control. The most
important remaining gaps are generated counterfactuals: real auction discipline,
sequence/seat plans, 84 late-hand tableaux, paired regime tests, and a joined
claim-row model table.

Claim-ledger impact: no broad central claim status changes in the final
synthesis. Narrow result pages record their local recommendations and caveats.

## Family Results

| family | bead | evidence | headline | phase-2 status | next |
|---|---|---|---|---|---|
| claim matrix / harness | `.1`, `.2` | matrix plus reusable analyzer | 64 claims routed; harness emits label metrics and paired contrasts | infrastructure complete | `t42-qtwb.4` |
| tactical pounce / donation | `.3` | same-decision contrasts | pounce and reckless-count controls replicate strongly; safe donation is small-positive | narrow operationalized support | `t42-qtwb.3` |
| hidden-threat impact | `.4` | offline branch diagnostics | close-mean lower-hidden-downside choices cut tail mass without mean cost | diagnostic evidence | `t42-qtwb.4` |
| bidding risk / only enough | `.5` | enumeration plus generated margin probe | static risk arithmetic holds; unnecessary same-contract margin is negative-or-tie in the smoke probe | partial empirical support | `t42-qtwb.1` |
| seat and position | `.6` | slices plus action-local contrasts | structural roles are slice-only; pounce/support/closure labels pair cleanly | row-level evidence | `t42-qtwb.3` |
| 84 preservation | `.7` | explicit bid-84 fixtures | measurement surface exists; preserve/spend result is small and targeted | fixture evidence | `t42-qtwb.2` |
| doubles / no-trump | `.8` | legacy within-regime mining | no-trump double control is strong; broad doubles-trump proxies need sharper gates | within-regime support | `t42-qtwb.4` |
| claim-tag model | `.9` | direct row-model probe | claim tags modestly improve held-out row-model regret and match | model-feature evidence | `t42-qtwb.4` |
| data inventory | `.11` | schema/corpus inventory | legacy data supports broad play/hidden diagnostics but lacks real auction margin | recovery complete | `t42-qtwb` |

Machine-readable synthesis: `w42/claim_analysis_synthesis/claim_family_synthesis.csv`.

## What Moved

The strongest empirical movement is tactical and action-local:

- [[w42-tactical-claim-replication]] keeps the pounce story sturdy on 28,000
  decisions and 75,079 legal actions.
- [[w42-seat-position-strategy-map]] now distinguishes structural seat slices
  from pairable action labels. Closure take-trick and pounce closure are large
  positive same-decision contrasts; unsupported count donation into defense is
  sharply negative.
- [[w42-doubles-no-trump-legacy-mining]] shows no-trump double control is not
  just static lore. It survives within-regime action mining.
- [[w42-claim-tag-model-probe]] shows detector tags help a small legal-action
  row model, with pounce/donation carrying the clearest family-drop signal.

## What Did Not Move

The conservative blockers are just as important:

- Bid-only-enough is only partially tested. The same-contract margin direction
  is clear, but real auction discipline still needs auction histories or a
  bid-policy simulator.
- 84 preservation has generated evidence, but only in six hand-picked fixtures.
  It needs late-hand state injection or large seed mining.
- Hidden-threat labels are powerful report targets, but hidden truth is not a
  live feature. Phase 3 needs public-safe belief proxies.
- Doubles/no-trump regime-choice claims still need paired same-hand regime
  generation. Seed-modulo declaration slices are not enough.
- Model-feature gains do not prove book claims. They say the detector vocabulary
  is useful enough to keep.

## Next Epic

`t42-qtwb` is the phase-3 generated counterfactual strategy epic.

| bead | focus |
|---|---|
| `t42-qtwb.1` | auction-aware bid discipline corpus |
| `t42-qtwb.2` | 84 endgame state injection |
| `t42-qtwb.3` | sequence and seat counterfactuals |
| `t42-qtwb.4` | joined claim-row model table |

`t42-qtwb.1` now has its first completed artifact,
[[w42-phase3-auction-bid-discipline-corpus]]. It strengthens the
bid-only-enough operational slice under generated auction pressure and keeps
natural bucket / partner-signal claims context-limited.

`t42-qtwb.2` now has [[w42-phase3-84-seed-mining-corpus]], a documented natural
seed corpus for 84 candidate structures and defender asset patterns. It is a
route to powered dynamic tests, not a preserve/spend proof by itself.

`t42-qtwb.3` now has [[w42-phase3-sequence-seat-counterfactuals]], a generated
branch-value table over 75,079 legal actions. It strongly supports follow-seat
control, closure, setter pounce, and unsafe-count negative controls; partner
support remains timing-gated; bidder lead sequencing stays context-limited
because called-suit-first is negative when pooled while called-double command is
strongly positive.

`t42-qtwb.4` now has [[w42-phase3-joined-claim-row-model-table]], a joined
public-safe legal-action table and model-feature ablation probe. Public claim
families improve held-out row-model mean regret from `1.3598` to `1.1257`; the
dominant ablation is sequence/seat, while bidding risk and public 84 structure
carry smaller positive signal and hidden public proxies remain diagnostic.

## Validation

```bash
python -m py_compile \
  w42/bid_only_enough_claim_tests/run_bid_only_enough_probe.py \
  w42/seat_position_claim_tests/run_seat_position_claim_tests.py \
  w42/seat_position_claim_tests/validate_outputs.py \
  w42/claim_tag_model_probe/run_claim_tag_model_probe.py \
  w42/claim_tag_model_probe/validate_outputs.py

python w42/seat_position_claim_tests/validate_outputs.py \
  --artifact-dir w42/seat_position_claim_tests \
  --min-actions 75079 \
  --min-paired-contrasts 8

python w42/claim_tag_model_probe/validate_outputs.py \
  --artifact-dir w42/claim_tag_model_probe
```

## Links

[[w42]] | [[w42-claim-data-inventory]] |
[[w42-tactical-claim-replication]] |
[[w42-hidden-threat-legacy-mining]] |
[[w42-bidding-risk-budget-claim-validation]] |
[[w42-phase2-seat-position-strategy-map]] |
[[w42-phase2-84-weapon-preservation-probe]] |
[[w42-doubles-no-trump-legacy-mining]] |
[[w42-claim-tag-model-probe]]
