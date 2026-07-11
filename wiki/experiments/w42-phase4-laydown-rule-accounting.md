---
title: w42 Phase4 Laydown Rule Accounting
kind: experiment
first_seen: local-2026-05-03
last_updated: afd4802
status: complete
---

## Summary

[[w42]] bead `t42-br7n.4` turns the Chapter 1 rule substrate and the Chapter 3
laydown warning into deterministic proof artifacts. The run writes table-driven
Forge checks plus a tiny full-information laydown enumerator under
`w42/phase4_laydown_rule_accounting/`.

The artifact has 29 deterministic assertions and zero failures: 23 rule
assertions, 7 rule fixtures, 6 laydown fixtures, and 3 rejected-laydown
counterexamples. It supports the low-level accounting claims for count identity,
42-point hand totals, suit membership, trump exclusivity, follow-suit masks,
trick winner, lead control, count capture, and contract scoring.

The laydown checker proves boss-trump and suit-exhaustion walker fixtures, and
rejects false claims when an opponent can still take the lead. It reproduces the
Chapter 3 final-deuce warning as an explicit counterexample: a late low deuce is
not a proven laydown if an opponent still holds a higher deuce.

Claim-ledger impact: deterministic substrate rows are supported on the fixture
surface, and `ch03-laydown-correctness` now has an executable proof checker.
The artifact does not yet consume arbitrary saved engine snapshots or Burl
claims, so corpus/model false-positive rates remain a follow-up.

## Method

| field | value |
|---|---|
| bead | `t42-br7n.4` |
| artifact directory | `w42/phase4_laydown_rule_accounting/` |
| runner | `w42/phase4_laydown_rule_accounting/run_laydown_rule_accounting.py` |
| validation | `w42/phase4_laydown_rule_accounting/validate_outputs.py` |
| rule assertions | 23 |
| laydown fixtures | 6 |
| counterexamples | 3 |
| failures | 0 |

The proof criterion is intentionally strict: for a claimed laydown, every legal
continuation by every player must give the claimant's team every remaining
trick. Rejected fixtures write concrete counterexample lines.

## Findings

| family | result |
|---|---|
| count identity | exact count set, count values, 35 count points, and 42 total hand points pass |
| suit membership | each pip suit has seven tiles before trump; doubles have one natural suit |
| trump exclusivity | called-suit trump does not also follow its secondary suit |
| follow-suit masks | legal responses match Forge tables in trump, secondary-suit, and void cases |
| trick winner / lead control | led-suit, trump, doubles-trump, and next-leader fixtures pass |
| contract scoring | bid 30/take 33, bid 35/take 33, and bid 35/take 30 examples pass |
| laydown proof | true boss-trump/walker claims prove; false off/trump-exclusion/deuce claims reject |

## Links

[[w42]] | [[winning42-ch01-in-a-nutshell]] |
[[winning42-ch03-bidder-play]] | [[forge]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- The strict all-continuations criterion could be relaxed to "wins regardless of opponents but assuming partner cooperates" to match how humans actually declare laydowns — a one-flag variant worth trying.
