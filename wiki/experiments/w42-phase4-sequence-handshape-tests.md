---
title: w42 Phase4 Sequence Handshape Tests
kind: experiment
first_seen: local-2026-05-03
last_updated: afd4802
status: complete
---

## Summary

[[w42]] bead `t42-br7n.1` sharpens Chapter 3, Chapter 4, and Chapter 5 tactical
claims over the full 75,079-row legal-action table from
[[w42-tactical-claim-replication]]. The artifact lives under
`w42/phase4_sequence_handshape_tests/` and writes public/action-local labels,
decision-shape rows, paired contrasts, slices, examples, and blocker rows.

The main result is the same folk-wisdom refinement seen in phase 3, but with
sharper labels. Commanding called doubles beat off leads by `+1.316` Q across
576 paired bidder lead decisions, while generic non-double called-suit leads
lose to off leads by `-3.682` Q across 1263 pairs. The book supports
"commanding trump" much better than blanket trump-first.

Partner and setter claims also split cleanly. Exact closure count donation is
strong (`+3.143` Q across 65 pairs), but partner count before closure is weak
and uncertain (`+0.308` Q with confidence interval crossing zero). Partner
count-liability leads are bad (`-3.425` Q). Setter count-calling leads
(`+3.028` Q), pounce take-count-now (`+2.681` Q), and the reckless-count
negative control (`-4.921` Q) all move in the book-consistent direction.

Claim-ledger impact: several tactical slices gain direct support, but exact
private hand-shape claims remain context-limited. The current table is fixed at
bid `30` and lacks exact reentry, void-creation, and generated high-bid pressure
state fields.

## Method

| field | value |
|---|---|
| bead | `t42-br7n.1` |
| artifact directory | `w42/phase4_sequence_handshape_tests/` |
| runner | `w42/phase4_sequence_handshape_tests/run_sequence_handshape_tests.py` |
| validation | `w42/phase4_sequence_handshape_tests/validate_outputs.py` |
| input | `w42/tactical_claim_replication/all_action_rows.jsonl` |
| action rows | 75079 |
| decision rows | 28000 |
| label metric rows | 47 |
| paired contrasts | 10 |
| blocker rows | 8 |

Labels use public/action-local row columns and same-decision legal-candidate
sets. Branch mean, threshold mass, lower-tail mass, and regret are offline
evaluation labels.

## Main Contrasts

| contrast | paired decisions | mean Q delta |
|---|---:|---:|
| commanding called double vs off lead | 576 | `+1.316` |
| non-double called-suit lead vs off lead | 1263 | `-3.682` |
| partner closure count donation vs noncount | 65 | `+3.143` |
| partner count before closure vs noncount | 549 | `+0.308` |
| partner count-liability lead vs noncount | 526 | `-3.425` |
| setter reckless count to bidder vs noncount | 2380 | `-4.921` |
| setter pounce take-count-now vs hold | 368 | `+2.681` |
| setter count-calling lead vs non-count-calling | 1866 | `+3.028` |

## Blockers

Exact reentry, low-trump traps, count-protection throwaways, trump-rich setter
recognition, opponent void creation, effective doubles, and high-bid off-pounce
pressure need additional state fields or generated high-bid contracts. The
35/36 pressure claim is blocked rather than contradicted because the source row
table is entirely bid `30`.

## Links

[[w42]] | [[winning42-ch03-bidder-play]] |
[[winning42-ch04-partner-support]] | [[winning42-ch05-setter-defense]]
