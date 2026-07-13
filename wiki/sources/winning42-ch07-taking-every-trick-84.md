---
title: Winning 42 Ch07 Taking Every Trick 84
kind: source
first_seen: 2026-05-01
last_updated: 2026-05-01
status: complete
---

## Summary

This page is the bead-backed work surface for `t42-ni1l.7`: Chapter 7, "Taking Every
Trick / The 84 Bid." It harvests bidder-side 84 concepts from Winning 42 into
measurable hypotheses for [[gus]], [[burl]], and [[forge]].

## Work Surface

The chapter slice is `scratch/winning42/winning42.with_figures.md` lines 3062-3547.
Completion requires source-backed concepts, detector inputs, metrics/tests, likely
data sources, implementation notes, and readiness labels for enumeration, oracle
rollouts, Gus analysis, or Burl traces.

This chapter belongs to the [[winning42-strategy-measurement]] workstream. It extends
the cheap public-state strategy-tag idea from [[gus-strategy-tags-probe]] into the 84
contract regime: a bidder is no longer trying to make enough points, but to win every
trick. That changes the relevant state accounting from ordinary bid margin to proof,
threat exhaustion, final-walker construction, and score-aware exposure.

## Source Claims

- An 84 bid doubles the all-seven-tricks contract: if opponents take any trick, the
  bidder is set for 84 points. (`scratch/winning42/winning42.with_figures.md` lines
  3073-3086)
- A lay-down 84 is a proof object: seven dominoes no one can beat, often four high
  trumps plus doubles, where play order is unnecessary because no losing branch exists.
  (`scratch/winning42/winning42.with_figures.md` lines 3090-3114, 3485-3517)
- The ordinary non-lay-down 84 shape is three or four trumps, two or three doubles,
  and usually one off protected by the double ahead of that off. The bid is a gamble
  that all tiles capable of beating the final off will be forced out before the last
  trick. (`scratch/winning42/winning42.with_figures.md` lines 3122-3154)
- The recommended bidder sequence is trumps first, doubles second, final off last,
  while tracking the off suit until it becomes a walker. (`scratch/winning42/winning42.with_figures.md`
  lines 3168-3209)
- Two failure modes are emphasized for protected-off 84 hands: one opponent holding
  too many trumps, or one opponent preserving enough tiles in the off suit until the
  last two tricks. (`scratch/winning42/winning42.with_figures.md` lines 3236-3245,
  3274-3282)
- Three-trump and two-off variants are still candidate 84 hands when doubles can force
  the relevant threat suits before the final trick; this creates an ordering problem
  when the higher same-suit tile has not yet appeared. (`scratch/winning42/winning42.with_figures.md`
  lines 3247-3296)
- Straight-off 84 is explicitly riskier: the book states a two-to-one chance that
  opponents hold the needed double, but allows higher offs or desperate score states
  as exceptions. (`scratch/winning42/winning42.with_figures.md` lines 3318-3324)
- Protected-off 84 should usually be bid, except that within 42 points of 250 the
  same hand should bid 42 to win the match while reducing set penalty. (`scratch/winning42/winning42.with_figures.md`
  lines 3326-3333)
- Overcalling 84 requires adding 42 points, creating 126, 168, or Game bids; the book
  treats straight-off escalation as poor risk. (`scratch/winning42/winning42.with_figures.md`
  lines 3335-3352)
- Against repeated strong defenders, occasional straight-off 84 can be used as an
  opponent-modeling counter when they over-discard doubles because they assume all
  good 84 bidders have protected offs. (`scratch/winning42/winning42.with_figures.md`
  lines 3354-3383)
- Partner support for an 84 bid centers on saving doubles, discarding doubles only
  after their suit is dead, keeping the double with the most still-live watched tiles,
  and sometimes preserving low bait tiles that overload defenders' double choices.
  (`scratch/winning42/winning42.with_figures.md` lines 3385-3483)

## Concept Table

| Concept | Detector / state inputs | Metric / test | Likely data source | Priority | Readiness | Implementation notes |
|---|---|---|---|---:|---|---|
| `84_contract_regime` | Winning bid value, contract target, tricks won by bidder team, score before hand | Correct regime label; set/make accounting; regret split for 42 vs 84 decisions | Engine game records; forge bidding traces | P0 | enumeration, oracle, Gus, Burl | This is the gate for every other detector. It should tag all bid values in the 84 ladder, not just exactly 84. |
| `laydown_84_proof` | Bidder hand, declared trump, trump ranks held, remaining non-trumps, same-suit double protection | Exhaustive proof that every legal defense loses every trick; false-positive rate of lay-down claims | Perfect-information enumeration over hands; forge/oracle solver | P0 | enumeration, oracle | Treat as a proof predicate, not a heuristic. The book's sufficient condition is at least four trumps including top three, plus doubles or double-protected second-high tiles. |
| `protected_one_off_84_shape` | Trump count/rank, doubles count, exactly one off, off suit, double ahead of off, live higher same-suit threats | Make probability; tail set rate; regret of 84 vs 42/41/36; bucketed by off rank | Dealt hands plus oracle rollouts; strategy-tag analyzer | P0 | enumeration, oracle, Gus | This is the core chapter-7 model feature. It should emit shape tags before play and update threat counts during play. |
| `trump_first_double_second_plan` | Bidder action order, lead ownership, outstanding trump count, doubles led before final off, off held until last trick | Plan-faithfulness rate; regret when deviating; survival of final off | Full game traces; forge E[Q] action records; Burl traces | P1 | oracle, Gus, Burl | This tests whether the policy follows the book's sequencing template and whether the oracle agrees with exceptions. |
| `final_walker_counter` | Final off tile, led suit, higher same-suit tiles live/played/in hand, outstanding trumps, void evidence | Precision/recall of final off becoming unbeatable; regret on last two tricks | Engine state logs; Gus strategy features; oracle states | P0 | enumeration, oracle, Gus | Generalizes `effective_double_or_walker` into an 84-specific terminal proof. |
| `off_suit_exhaustion_counter` | Off suit, same-suit played count, known in-hand same-suit tiles, trumped same-suit tile, doubles that force suit | Calibration of "threats remaining" count; missed-force opportunities | Public play history; belief-labeled worlds; Gus belief outputs | P1 | enumeration, Gus, Burl | Must be public-state first, then optionally belief-weighted. Burl traces should show whether it tracks the suit in language/tool calls. |
| `three_trump_three_double_84` | Three trumps, three doubles, one off, missing trump count/rank, double matching off suit | Make/set rate vs four-trump shape; sensitivity to one opponent holding three trumps | Enumerated deals; oracle rollouts | P1 | enumeration, oracle | The book claims four trumps are not required. This bucket isolates the cost of the missing trump. |
| `two_off_same_suit_84` | Two offs sharing a suit, protecting double, higher same-suit threats, final-off ordering | Success rate; correct order of final two offs; regret when high threat remains | Enumerated candidate hands; oracle action traces | P1 | enumeration, oracle, Gus | Requires an ordering detector: lead the higher off first if the still-live same-suit threat can beat the lower off. |
| `protected_off_failure_modes` | Opponent trump concentration, one opponent's ability to preserve two off-suit threats, forced-follow availability | Attribution of set cause: trump stack vs off-suit preservation vs other | Perfect-info postmortems; oracle rollouts | P1 | enumeration, oracle | This is the bridge to Chapter 8 defense. It can label why an 84 failed without implying the bid was bad ex ante. |
| `straight_off_84_risk` | Off lacks double-ahead protection, off rank, matching double ownership probability, score deficit, opponent skill/style | Empirical set rate; compare to book's two-to-one double ownership claim; regret of 84 vs lower bid | Enumeration for ownership odds; oracle/game rollouts for set rate; style corpora later | P0 | enumeration, oracle, Gus | Ownership odds are easy; "good players set nearly two-thirds" needs policy assumptions and should remain context-limited until simulated. |
| `score_42_vs_84_gate` | Score before hand, distance to 250, hand's 42 make probability, hand's 84 make probability, set penalty | Regret of bidding 84 when 42 wins match; terminal win-rate delta | Engine score states; forge rollouts | P0 | enumeration, oracle, Burl | This is an unusually clean rule: near game point, avoid unnecessary 84 exposure even with a strong hand. |
| `84_overcall_ladder` | Current auction, prior 84 bid, candidate hand solidity, bid increments 84/126/168/Game, straight-off flag | Overcall frequency; regret/tail loss by hand solidity; legal ladder correctness | Bidding logs; oracle rollouts; tournament scoring variants | P2 | enumeration, oracle | Separate legality from strategy. The strategic surface is whether escalation is justified by proof strength. |
| `opponent_model_keep_honest` | Repeated opponent identity, defenders' double-discard tendency, bidder straight-off frequency, prior revealed 84 shapes | Style-conditioned make/set rate; adaptation after punished double discard | Requires player-id corpora or synthetic style policies | P3 | oracle, Burl | Not an MVP unless style simulation exists. It is valuable later because it turns book strategy into population dynamics. |
| `partner_save_double_84` | Partner doubles, bidder trump declaration, played suit counts, live watched tiles, forced discard choices | Correct double-preservation rate; regret of discarding live winning double; final-set attribution | Full traces with partner seat; oracle/action records | P1 | enumeration, oracle, Gus, Burl | Belongs partly to Chapter 4/8, but Chapter 7 frames it as bidder-team support for making 84. |
| `dead_double_discard` | Partner-held double, all relevant same-suit off candidates exhausted, trick number, discard legality | Precision of "safe to release" label; regret when holding dead double too long | Public play history; oracle traces | P2 | enumeration, oracle, Gus | The detector should distinguish "double can no longer set straight off" from "double may still be useful as count/control." |
| `double_choice_most_live_threats` | Partner has multiple doubles before next-to-last trick, unplayed tiles in each watched suit, rank of possible bidder offs | Accuracy of keeping the double with higher remaining threat mass; regret of choice | Enumerated public states; oracle rollouts | P2 | enumeration, oracle, Burl | Good small eval bucket for Burl because the reasoning is explicit and countable. |
| `low_bait_partner_support` | Partner low same-suit tiles, opponents likely saving doubles, discard opportunities, final trick composition | Cases where low-tile preservation causes defender double misallocation; policy-dependent gain | Synthetic defender policies; Burl/Gus traces later | P3 | oracle, Burl | This is strategic deception and needs modeled opponents; not a deterministic first detector. |
| `84_memory_pressure` | 84 regime, trick-history visibility, off-suit watched count, immediate past two tricks vs full history | Trace faithfulness to suit counting; Gus/Burl errors after long tracking interval | Burl traces; Gus belief/tag ablations | P2 | Gus, Burl | The chapter explicitly says players must keep counts mentally when old tricks are not visible; that maps cleanly to attention/memory probes. |

## First Detector Set

1. `laydown_84_proof`: exhaustive predicate for hands that cannot lose any trick under
   a chosen declaration.
2. `protected_one_off_84_shape`: opening-hand classifier for the canonical 84 candidate.
3. `final_walker_counter`: live-state detector for when the saved off has become a
   forced winner.
4. `straight_off_84_risk`: ownership-odds and rollout bucket for unprotected final offs.
5. `score_42_vs_84_gate`: score-aware bid restraint when 42 already wins the match.
6. `partner_save_double_84`: partner-seat preservation bucket for doubles that could set
   or protect the last trick.

## Readiness Notes

Enumeration-ready:

- `laydown_84_proof`, `84_contract_regime`, `straight_off_84_risk` ownership odds,
  `score_42_vs_84_gate` legality/terminal thresholds, `two_off_same_suit_84` threat
  counts, and `partner_save_double_84` public-state labels.

Oracle-ready:

- Protected-off make probability, three-trump/four-trump comparisons, straight-off
  rollout set rate, overcall ladder regret, plan-deviation regret, and set-cause
  attribution after failed 84 hands.

Gus-ready:

- Public strategy tags for `84_contract_regime`, `protected_one_off_84_shape`,
  `final_walker_counter`, `off_suit_exhaustion_counter`, `straight_off_84_risk`,
  `score_42_vs_84_gate`, and partner double preservation.

Burl-ready:

- Trace buckets where Burl should mention or tool-check the 84 regime, final off suit,
  outstanding trumps, watched same-suit threats, score gate, and partner/defender double
  preservation. The most interpretable prompt/eval cases are `score_42_vs_84_gate`,
  `double_choice_most_live_threats`, and `final_walker_counter`.

## Claim Ledger

[[w42-phase3-84-seed-mining-corpus]] and
[[w42-phase4-84-dynamic-seed-tests]] now give Chapter 7 natural seed inventory
and reached-state dynamic evidence. The evidence supports bidder plan proxies
and natural surface frequency, but score gates and straight-off population set
rates still need stronger generators.

| Claim | Source | Status | Next empirical check |
|---|---|---|---|
| 84 is set if opponents win one trick. | Lines 3073-3086 | supported by rules/accounting | Add `84_contract_regime` unit tests over scoring records. |
| Lay-down 84 can be proved from high trumps plus doubles/protected seconds. | Lines 3090-3114, 3485-3517 | Untested | Exhaustively enumerate the stated sufficient condition and measure false positives/negatives. |
| Protected one-off 84 hands have dramatically low set risk. | Lines 3122-3154, 3326-3329 | context-limited seed/dynamic support | Phase 3 finds 22176 protected-one-off candidate rows; phase 4 selects 9 and tests reached-state action proxies. Direct terminal set rate remains open. |
| Main protected-off failure modes are opponent trump concentration or preserved off-suit threats. | Lines 3236-3245, 3274-3282 | Untested | Postmortem failed 84 rollouts with perfect-info attribution. |
| Straight-off 84 has roughly two-to-one matching-double ownership against the bidder and is set nearly two-thirds by good players. | Lines 3318-3324 | context-limited / blocker | Phase 4 selects 10 straight-off games, but the "good players" set-rate claim still needs population rollouts. |
| Within 42 points of 250, bid 42 instead of 84 even with a protected-off hand. | Lines 3326-3333 | blocked | `t42-br7n.2` records the score 42-vs-84 gate as static only; terminal match counterfactuals remain outside the play generator. |
| Overcalling 84 with a straight off is poor because the set penalty becomes harder to recover from. | Lines 3335-3352 | Untested | Bucket 126/168/Game overcalls by proof strength and score context. |
| Partner should save live doubles, discard dead doubles, and keep the double with the most remaining watched tiles. | Lines 3385-3450 | context-limited support | Phase 4 preserve/spend proxy is `+1.946` Q and dead-asset release is `+0.504` Q; full throwaway ladder remains blocked. |
| Partner can help by preserving low bait tiles that make defenders save the wrong doubles. | Lines 3452-3483 | Underpowered until style policies exist | Requires synthetic defender policies or repeated-player traces. |

## Notes For Implementation

- Keep bidder-side 84 and defender-side 84 separate. Chapter 7 provides bidder and
  partner-making-84 concepts; Chapter 8 should own defender preservation and active set
  tactics.
- Distinguish proof from probability. `laydown_84_proof` should be exact; protected-off
  and straight-off 84 should be empirical risk buckets.
- Include score in all 84 bidding buckets. The same hand can be strategically correct
  as 84 when behind and strategically dominated by 42 when already within match range.
- Treat OCR gaps in the preview as source limitations. The chapter's figures define
  concrete hand examples, but the surrounding prose is enough to specify detector shapes.
