---
title: w42 Phase4 Bidding Count Exposure Tests
kind: experiment
first_seen: 2026-05-03
last_updated: 2026-05-03
status: active
---

## Summary

[[w42]] bead `t42-br7n.7` closes the phase-4 scope gap found by
[[w42-phase4-claim-completion-board]]. It writes claim-specific bidding,
count-exposure, and partner-prior evidence under
`w42/phase4_bidding_count_exposure_tests/`.

The artifact covers seven rows: `ch16-partner-two-plus-doubles-prior`,
`ch02-three-plus-trumps-good-start`, `ch02-risk-budget-threshold`,
`ch02-strong-trump-bad-risk-trap`, `ch02-four-five-off-danger`,
`ch12-natural-bid-bucket-anomaly`, and `ch02-double-side-protection`.

The main result is conservative support for static/generated-contract surfaces,
not full auction truth. Three-plus trump contracts are much stronger in the
generated corpus (`p_make_30` delta `+0.293` versus 0-2 trumps). Risk <=12 is
directionally better but modest (`p_make_30` delta `+0.037`, mark swing delta
`+0.075`). Four/five-off exposure is clearly worse (`p_make_30` `0.337` versus
`0.406`). Natural 30/31 or 35/36 max-profitable buckets appear in 66 / 384 rows.
Partner two-plus-double prior is `58.099%` exactly and `57.031%` in the
generated contract rows. Double-side protection covers only `16.775%` of exposed
count points, supporting the side-specific warning.

The strong-trump trap remains nuanced: 24 high-risk strong-trump rows exist, but
their make rate remains high. The result supports the exposure warning and
lower bid-ceiling caution more than a simple make-rate penalty.

## Method

| field | value |
|---|---|
| bead | `t42-br7n.7` |
| artifact directory | `w42/phase4_bidding_count_exposure_tests/` |
| runner | `w42/phase4_bidding_count_exposure_tests/run_bidding_count_exposure_tests.py` |
| validation | `w42/phase4_bidding_count_exposure_tests/validate_outputs.py` |
| contract rows | 384 |
| side exposure rows | 2672 |
| claim summary rows | 7 |

The runner joins prior generated contract/action rows with exact hand-shape,
risk, side-exposure, natural-bucket, and partner-double detectors. It does not
run observed human auctions or full bid/pass policy rollouts.

## Claim Results

| claim | recommendation | headline |
|---|---|---|
| `ch02-three-plus-trumps-good-start` | keep context-limited | three-plus trump `p_make_30` delta `+0.292901`, but not sufficient by itself |
| `ch02-risk-budget-threshold` | context-limited support | risk <=12 `p_make_30` delta `+0.037471`, mark swing delta `+0.074942` |
| `ch02-strong-trump-bad-risk-trap` | static surface supported, outcome mixed | 24 trap rows; make rate stays high but max-profitable ceiling trails low-risk strong trump |
| `ch02-four-five-off-danger` | context-limited exposure supported | four/five off rows have `p_make_30` `0.336857` |
| `ch02-double-side-protection` | static side detector supported, context-limited | protected side accounts for only `16.774838%` of exposed count points |
| `ch12-natural-bid-bucket-anomaly` | partial empirical bucket evidence | natural max-profitable buckets appear in 66 / 384 rows |
| `ch16-partner-two-plus-doubles-prior` | context-limited static prior supported | exact prior `58.098713%`, generated contract rate `57.03125%` |

## Blockers

All seven rows still list full policy or sequence counterfactuals as the next
step for promotion beyond static/generated-contract evidence. The needed
generator is auction-aware bid/pass policy rollout or state-injected E[Q]/make
set counterfactuals that can separate bidding behavior from static hand shape.

## Links

[[w42]] | [[winning42-ch02-bidding]] |
[[winning42-ch12-advanced-bidding-playing]] |
[[winning42-ch16-statistical-odds]]
