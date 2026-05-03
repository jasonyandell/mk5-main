---
title: w42 Phase4 Doubles Notrump Regime Tests
kind: experiment
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: active
---

## Summary

[[w42]] bead `t42-br7n.5` moves Chapter 9 beyond within-regime legacy mining by
generating paired same-hand doubles-trump versus no-trump simulations under
`w42/phase4_doubles_notrump_regime_tests/`. The runner fixes each P0 hand and
uses the same simulation seed for both regimes, giving 192 same-hand regime
pairs and 12288 paired opponent-world samples.

The result leans strongly toward the book's caveat: many doubles alone is not a
safe reason to choose doubles-trump. No-trump beats doubles-trump at bid 30 on
`66.1%` of four-plus-double hands with mean `NT - DT` mark swing `+0.126`.
Missing-top / low-exposure hands are sharper: no-trump is preferred `75.9%` of
the time with `+0.177` mean mark swing. No-trump support-double hands are also
strong for no-trump: `76.2%` preferred, `+0.220` mean mark swing.

The pro-doubles slice is high/top double control. That bucket is essentially
flat: `+0.031` for no-trump with confidence crossing zero, and doubles-trump is
preferred `45.2%` versus no-trump `47.6%`. Five-plus doubles also softens the
no-trump advantage enough that the interval crosses zero.

Claim-ledger impact: the old "four-plus doubles" rule should stay
context-limited. The generated evidence supports the caveat structure:
top-double control matters, while low/missing-top exposure favors no-trump.

## Method

| field | value |
|---|---|
| bead | `t42-br7n.5` |
| artifact directory | `w42/phase4_doubles_notrump_regime_tests/` |
| runner | `w42/phase4_doubles_notrump_regime_tests/run_doubles_notrump_regime_tests.py` |
| validation | `w42/phase4_doubles_notrump_regime_tests/validate_outputs.py` |
| selected hands | 192 |
| regime contract rows | 384 |
| paired opponent-world samples | 12288 |
| claim contrasts | 8 |

This is greedy `forge.bidding` policy simulation, not exhaustive oracle proof.
P0 is always treated as bidder; auction pressure is outside the bead.

## Bucket Results

| bucket | hands | mean `NT - DT` mark swing | no-trump preferred |
|---|---:|---:|---:|
| four-plus doubles | 121 | `+0.126` | `66.1%` |
| five-plus doubles | 31 | `+0.082` | `61.3%` |
| four-plus missing top double | 79 | `+0.177` | `75.9%` |
| high double control | 42 | `+0.031` | `47.6%` |
| no-trump support | 105 | `+0.220` | `76.2%` |
| regime switch proxy | 55 | `+0.204` | `78.2%` |

## Links

[[w42]] | [[winning42-ch09-doubles-no-trump]] |
[[w42-doubles-no-trump-legacy-mining]]
