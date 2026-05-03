---
title: w42 Phase4 84 Dynamic Seed Tests
kind: experiment
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: active
---

## Summary

[[w42]] bead `t42-br7n.2` turns the phase-3 natural 84 seed inventory into a
dynamic reached-state branch-atlas artifact under
`w42/phase4_84_dynamic_seed_tests/`. It selects 24 mined natural seed games,
runs schema-v2 E[Q] branch generation, and emits action labels, label metrics,
paired contrasts, examples, blockers, and the generated game tensor.

The run covers 672 decisions, 1872 legal actions, and 14386 hidden-threat rows.
Selected surfaces include 9 protected one-off games, 10 straight-off games, 4
two-off same-suit games, 1 natural laydown, and 23 games each with defender
live-double, same-suit-pair, protector, and dead-asset surfaces.

The strongest dynamic result is preservation: playing expendable/dead assets
instead of spending live weapons or protectors is `+1.946` mean Q across 62
paired decisions. Releasing dead doubles instead of spending live assets is
`+0.504` mean Q across 29 pairs. On the bidder side, trump-pull before final-off
is `+4.295` mean Q across 76 paired decisions.

Claim-ledger impact: 84 preservation and bidder-plan proxies gain reached-state
support, but the artifact remains policy-trace evidence. It is not arbitrary
late-state injection and does not prove final set causality.

## Method

| field | value |
|---|---|
| bead | `t42-br7n.2` |
| artifact directory | `w42/phase4_84_dynamic_seed_tests/` |
| runner | `w42/phase4_84_dynamic_seed_tests/run_84_dynamic_seed_tests.py` |
| validation | `w42/phase4_84_dynamic_seed_tests/validate_outputs.py` |
| selected seed games | 24 |
| decisions | 672 |
| legal action rows | 1872 |
| hidden-threat rows | 14386 |
| paired contrasts | 3 |

Full-deal defender assets are used only as offline eval labels. Live features
remain public state plus actor hand.

## Main Contrasts

| contrast | paired decisions | mean Q delta |
|---|---:|---:|
| defense preserve expendable vs spend live asset | 62 | `+1.946` |
| dead-asset release vs spend live asset | 29 | `+0.504` |
| offense trump-pull vs final-off proxy | 76 | `+4.295` |

## Blockers

Final set attribution, full throwaway-ladder bottlenecks, score 42-vs-84
terminal bid counterfactuals, and the straight-off "nearly two-thirds set by
good players" population claim remain blocked. They need either arbitrary
late-state injection, terminal bid/match counterfactuals, or a stronger player
population than the current reached greedy policy trace.

## Links

[[w42]] | [[w42-phase3-84-seed-mining-corpus]] |
[[w42-phase2-84-weapon-preservation-probe]] |
[[winning42-ch07-taking-every-trick-84]] | [[winning42-ch08-setting-84]]
