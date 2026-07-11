---
title: Winning 42 Ch02 Bidding
kind: experiment
first_seen: local-2026-05-01
last_updated: local-2026-05-01
status: active
---

## Summary

This page is the bead-backed work surface for `t42-ni1l.2`: Chapter 2, "Bidding."
It harvests bidding-as-risk-budget concepts from Winning 42 into measurable hypotheses
for [[gus]], [[burl]], and [[forge]].

Chapter 2 treats a bid as a public commitment to a loss budget. The source frame is:
choose a plausible trump suit, count the tricks and count dominoes that might escape,
subtract that loss budget from 42, then bid only enough to win the auction. The chapter's
most useful empirical hook is not a single "good bid" rule, but a family of state tags:
trump control, off exposure, duplicate count exposure, partner-help dependence, and
auction-position restraint.

## Work Surface

The chapter slice is `scratch/winning42/winning42.with_figures.md` lines 735-1406.
Completion requires source-backed concepts, detector inputs, metrics/tests, likely
data sources, implementation notes, and readiness labels for enumeration, oracle
rollouts, Gus analysis, or Burl traces.

The source cited here is the Chapter 2 preview slice, pages 9-22. In this worktree the
large OCR source is absent, so the slice was read from the main checkout at the same
repository path and treated as read-only provenance.

Readiness labels:

- `enumeration` means the claim can be checked from dealt hand plus declaration candidates.
- `oracle` means it needs E[Q] rollout, perfect-information solve, or generated games.
- `Gus` means it can become a public-state/action feature or belief/regret bucket.
- `Burl` means it can be audited in thought/tool traces.

## Source-Backed Concepts

| concept | source basis | detector/state inputs | metric/test | likely data source | priority / readiness | implementation notes |
|---|---|---|---|---|---|---|
| Minimum biddable trump suit | Ch2 opens with the beginner rule that a good bidding hand starts from at least three tiles in one suit, often strengthened by doubles. | Hand, candidate declaration, count of suit members, count/rank of doubles. | Make/set rate and bid-regret by trump-count bucket; compare three-trump candidates against pass and alternative declarations. | Exhaustive hand enumeration; [[forge]] bidder games; Gus corpus decision 0. | P0 / `enumeration`, `oracle`, `Gus` | Emit `candidate_trump_count`, `candidate_trump_rank_coverage`, and `double_count_outside_trump`; do not hard-code "three trumps bid" because later examples pass strong-looking four/five-trump hands. |
| Backward loss-budget bidding | The chapter says to predict the count dominoes and tricks the hand might not win, then bid from that loss budget rather than from optimism. | Candidate declaration, own hand, vulnerable trump count, vulnerable offs, count at risk, auction high bid. | Calibration curve: predicted loss budget vs oracle make probability / realized set rate; regret for overbid and underbid. | Enumeration for static budget; E[Q] rollout for realized value; generated full games for auction outcomes. | P0 / `enumeration`, `oracle`, `Gus`, `Burl` | First-pass detector can compute `safe_bid_ceiling = 42 - estimated_at_risk_points`; later attach uncertainty bands rather than a single ceiling. |
| Trump count risk | Ch2 warns to ask whether the bidder could lose a trump trick and whether trump count could fall on it; examples include missing critical five-trump count. | Candidate trump suit, trump tiles held, highest missing trump, trump count dominoes held/missing, count in candidate trump. | Frequency and cost of losing a trump trick by rank coverage; make probability when missing boss/second boss/count trump. | Enumeration for missing-trump buckets; oracle rollouts for outcome delta. | P0 / `enumeration`, `oracle`, `Gus` | Distinguish "many trumps" from "safe trumps"; hand 10 is the canonical bucket where four trumps still carry 16 points of trump risk. |
| Off-side count exposure | The chapter repeatedly analyzes an off by both pips and maps each side to count dominoes that could be lost. | Non-trump, non-double tiles in hand; both suits of each off; count dominoes associated with each side; candidate trump suit. | Expected lost count and tail set risk by off side; false-safe rate when only high side is considered. | Enumeration for exposure table; oracle rollouts for decision value. | P0 / `enumeration`, `oracle`, `Gus`, `Burl` | Feature should be action-local and hand-level: `off_high_side_count_risk`, `off_low_side_count_risk`, `off_max_side_risk`. |
| Four/five suit danger | The source singles out four-offs and five-offs because one trick can expose two count dominoes and 15 points. | Candidate off suits, whether four/six plus four/ace or double-five/five-blank are live, in hand, or trump. | Set rate and catastrophic loss rate for four/five off buckets vs other off buckets. | Enumeration; generated games; oracle counterfactuals. | P0 / `enumeration`, `oracle`, `Gus` | Use as an adversarial eval bucket because it is a tail-risk concept, not just mean EV. |
| Trump-suit neutralization of off risk | Ch2 notes that count normally exposed by an off may disappear when the associated count domino is trump. | Candidate trump suit, off pips, count domino identity, whether that count tile is trump or held by bidder. | Delta in loss-budget and oracle value when the same hand is evaluated under alternative declarations. | Candidate-declaration enumeration; [[forge]] declaration counterfactuals. | P1 / `enumeration`, `oracle`, `Gus` | Good first target for "choose fives vs treys" style candidate-declaration ranking; avoid evaluating a hand under only the finally chosen declaration. |
| Double-ahead protection | The source introduces "double ahead of your off" as reducing the chance that count is lost on that suit, while warning it protects only the matching side. | Off tile, matching doubles in hand, side-specific count exposure, lead/follow position, whether protected side is high or low side. | Protected vs unprotected loss delta; cases where a double falsely reassures because the other side remains exposed. | Enumeration for labels; oracle rollout for value; Burl traces for whether reasoning mentions both sides. | P0 / `enumeration`, `oracle`, `Gus`, `Burl` | Emit side-specific protection rather than a Boolean `protected_off`; `double_ahead_high_side` and `double_ahead_low_side` matter separately. |
| Duplicate count accounting | Ch2 says not to add the same count domino twice when multiple offs expose it. | All offs, candidate trump, exposed count tile set, duplicate exposures by count tile. | Error rate of naive summed-risk vs unique-count-risk; regret for over-passing hands with duplicated exposure. | Enumeration; generated bidding corpus. | P1 / `enumeration`, `Gus`, `Burl` | Natural unit test for the strategy tag analyzer: two blank offs should not count `blank-five` twice. |
| Lead-planned off exception | The source says if the bidder knows an off will be led early, only the high side matters because low-side risk is bypassed before opponents lead that suit. | Candidate opening lead plan, off tile, seat/lead control assumptions, high-side and low-side count exposure. | Oracle value of leading off early vs pulling trump first; success/failure by opponent void/trump distribution. | Oracle rollouts; Burl traces; generated games with opening lead. | P1 / `oracle`, `Gus`, `Burl` | Static detector can mark `planned_off_lead_reduces_low_side`; empirical validation needs play-sequence policy because leading off is a plan, not just a hand property. |
| One-off pessimism vs multi-off partner help | The chapter gives the paradoxical rule: with one vulnerable off, assume opponents hold the winner; with two or more vulnerable tricks, some partner help becomes more reasonable. | Number of loss opportunities, count exposure per opportunity, partner seat ownership probability for rescue tiles, auction partner signal. | Make probability and bid regret by `loss_opportunity_count`; partner-rescue rate conditioned on one-off vs multi-off hands. | Enumeration for prior odds; oracle rollouts for EV; Gus belief ownership; generated games. | P0 / `enumeration`, `oracle`, `Gus` | This is a prime "book claim" because it can be quantified: compare one 15-point off to two 6-point offs with partner rescue probability. |
| Partner prior-bid signal | Ch2 suggests confidence can rise when partner has entered the auction, because it signals useful dominoes. | Auction history, partner bid/pass, partner seat, candidate declaration, hidden-hand posterior. | Information value of partner bid: make probability shift, posterior ownership of rescue/count/trump tiles, bid-regret delta. | Generated auction corpus; Gus belief outputs; Burl traces. | P1 / `oracle`, `Gus`, `Burl` | Needs a trustworthy bidding corpus or logged bidder policy; keep separate from pure hand enumeration because partner bid is behavioral evidence. |
| Bid only enough to win | The chapter says a hand worth 35 should bid 31 if 31 wins, because the score received is not penalized by bidding lower. | Auction high bid, hand bid ceiling, candidate increment, scoring rules. | Overbid regret/tail risk when bid exceeds minimum winning bid; distribution of unnecessary high bids. | Generated games with auctions; Burl traces; strategy-probe action buckets. | P0 / `enumeration`, `oracle`, `Gus`, `Burl` | Detector: `unnecessary_bid_margin = bid - (current_high_bid + 1)` under ordinary point scoring; gate variants/scoring modes later. |
| Natural bid buckets and odd-bid behavior | Ch2 examples cluster around 30/31/35/36 and says there is often no need to open 34 because raises usually jump to 35. | Bid value, auction context, estimated ceiling, prior high bid, score mode. | Frequency and value of odd bids; regret of 32-34 vs 31/35 alternatives; opponent response distribution. | Auction logs; generated games; Burl traces. | P2 / `oracle`, `Gus`, `Burl` | Keep as an analysis bucket before making a training rule; odd bids may be table/meta strategy rather than hand strength. |
| Pass strong-looking trumps with bad count risk | Multiple examples emphasize passing hands with excellent trump quantity when one off/trump trick risks too much count. | Candidate trump count/rank, total at-risk points, one-trick catastrophic count, partner rescue probability. | False-positive biddable classifier rate for high-trump hands; tail set rate. | Enumeration; oracle rollouts; Gus regret bucket. | P0 / `enumeration`, `oracle`, `Gus` | Use as a guard against shallow "more trumps = bid" features; label `strong_trump_bad_off_trap`. |
| Confidence/frequency as skill frontier | The chapter ends by moving from beginner caution toward confident frequent bidding when risks are justified. | Player/bot identity, bid/pass frequency, hand risk bucket, score context. | Style-conditioned bid calibration: aggression residual after controlling for hand strength; set rate vs missed-opportunity regret. | Generated/self-play games; future population traces; Burl full-game arena. | P2 / `oracle`, `Gus`, `Burl` | Useful later for style and partnership ecology, but not first-pass static tags. |

## First Detectors To Build

1. `candidate_bid_loss_budget`: for every declaration candidate, compute unique at-risk count
   plus trick risk from vulnerable trumps and offs, then expose a conservative bid ceiling.
2. `side_specific_off_risk`: for every off, emit high-side and low-side count exposure,
   whether each side is neutralized by trump, already held, or double-protected.
3. `catastrophic_four_five_off`: flag four/five off situations where one lost trick can
   carry 15+ points, with sublabels for held/trumped/duplicated count.
4. `partner_help_dependency`: classify one-loss-opportunity hands versus multi-loss
   hands and estimate whether the bid requires a partner rescue.
5. `unnecessary_bid_margin`: compare the actual bid to the minimum bid needed to win the
   auction and to the detector's bid ceiling.
6. `strong_trump_bad_risk_trap`: find hands with four or more candidate trumps that should
   still pass because trump/off count risk exceeds the chapter's 12-point beginner budget.

## Readiness Notes

Enumeration can cover candidate trump count, rank coverage, count exposure, duplicate
count accounting, double-ahead labels, and rough one-off versus multi-off rescue priors.

Oracle rollout is needed for make probability, set risk, bid-regret, declaration
counterfactuals, and whether leading an off early is actually better than pulling trump.

Gus work should start with public-state tags and concept buckets: `bid_loss_budget`,
`off_side_risk`, `off_protection`, `partner_help_dependency`, `unnecessary_bid_margin`,
and `strong_trump_bad_risk_trap`. These fit the existing [[gus-strategy-tags-probe]]
finding that action-local and human-legible strategy tags carry policy signal.

Burl work should audit whether traces mention both sides of an off, avoid double-counting
the same count domino, distinguish "safe because trump" from "safe because in hand," and
justify bid increments relative to the current auction rather than announcing an absolute
hand value.

## Claim Ledger

[[w42-phase4-bidding-count-exposure-tests]] now supplies static-detector and
generated-contract evidence for the main Chapter 2 risk-budget rows. The current
evidence is useful but still not full auction truth: promotion beyond
context-limited support needs bid/pass policy rollout or state-injected
make/set/E[Q] counterfactuals.

| claim | status | first empirical check |
|---|---|---|
| A conservative bid can be modeled as `42 - at_risk_points`, with bids generally justified when at-risk points are 12 or less. | context-limited support | `t42-br7n.7` finds risk <=12 modestly better than >12: `p_make_30` delta `+0.037471`, mark swing delta `+0.074942`. |
| Three or more candidate trumps plus doubles is a useful initial biddability prior. | context-limited support | Three-plus trump contracts are much stronger in generated labels (`p_make_30` delta `+0.292901`), but the result is not sufficient by itself. |
| Offs, especially four/five offs, dominate many bidding failures more than trump count does. | context-limited support | Four/five off exposure rows have lower `p_make_30` (`0.336857`) and worse generated value; stronger line-of-play punishment tests remain open. |
| Double-ahead protection reduces only the protected side of an off, not both sides. | static side detector supported, context-limited | Side-specific protection covers only `16.774838%` of exposed count points; live sequence value remains untested. |
| Duplicate count exposures should be counted once in the bid risk budget. | context-limited / not run | Unit-testable enumeration against naive risk-sum baseline. |
| One vulnerable trick should be bid pessimistically, while two or more vulnerable tricks can justify some partner-help expectation. | context-limited / not run | Partner rescue probability and bid-regret interaction by loss-opportunity count. |
| A bidder should bid only enough to win the auction when the same captured points will score regardless of bid size. | **supported** on same-hand bid-margin slice | [[w42-bookval-v1-wave2-ch02-multistep]] paired all 5 step pairs (30↔32, 32↔35, 35↔36, 36↔39, 39↔42) over n=8,168-10,052 paired decisions per step: mark_ev deltas `+0.048` to `+0.146`, all CIs exclude zero in book direction. Cohen d ranges `0.16`–`0.47` (smallest at 35↔36, largest at 39↔42; not monotone in step order). All 85 slice cells (5 step pairs × [10 decls + 4 seat roles + 3 phases]) support book direction. Transitive 30→42 cumulative matches sum-of-steps within 0.81%. Ledger row promoted from `context-limited` to **`supported`** — the campaign's first promotion above `context-limited` from a non-supported start. Cross-contract bid choice (different declarations at different bids) and full auction-policy testing remain separate scopes. |

## Wave 2 Findings (Book Validation v1)

The campaign's first ledger promotion landed here. Wave 2.B.2's
50-seed bid-aware MPS sweep (259,618 action rows) provides
**paired same-hand counterfactual evidence** for the bid-only-enough
principle:

- For every (seed, decl_id, decision_idx) actually-played action across
  50 seeds × 10 decls, mark_ev and p_make are computed under both
  bid=30 and bid=32 incentives.
- The paired delta (bid=32 minus bid=30) is `-0.076` for mark_ev and
  `-0.038` for p_make. Both have tight CIs that exclude zero in the
  book direction.
- N=14,000 paired decisions per bid bucket. Power: sufficient for
  mark_ev and p_make; borderline for threshold_mass.

This is the cleanest possible same-contract test of the book's claim:
on the same hand at the same decision, evaluating under bid=32 makes
you mark-worse than under bid=30 by ~0.08 marks per decision. The
audit's original overclaim risk was that bid-only-enough should not
be promoted broadly without auction evidence; this paired-same-hand
slice is conservative because it tests only the "captured-points
identity" mechanism the book proposes, not auction strategy.

The high-bid steps (bid=35, 36, 39, 42, 84) and full auction policy
remain untested. Wave 2.G (bead `t42-ey88`) will extend the analysis to
multi-step bid margins; auction-policy testing remains in the Wave 3
plan.

**Update**: Wave 2.G landed at [[w42-bookval-v1-wave2-ch02-multistep]]
and extended the evidence to all 5 adjacent step pairs in {30, 32, 35,
36, 39, 42}. Result: overbid penalty in the book direction at every step, no reversals
(Cohen d `0.16`–`0.47`), 85 of 85 slice cells in book direction,
transitive cumulative matches sum-of-steps. Ledger row promoted from
`context-limited` to **`supported`** on the same-hand bid-margin
slice — the campaign's first non-trivial `supported` promotion. The
remaining caveat is cross-contract bid choice (different declarations
at different bids) and real auction-policy response, both Wave 3 work.

## Audit (2026-07-07)

Two-pass audit against code and artifacts; 2 corrections applied in place and independently re-verified; second pass amended 1.

- Bead IDs on this page (`t42-ni1l.2`, `t42-br7n.7`, `t42-ey88`) no longer resolve via `bd` (retired 2026-06) but all three exist in the archived `.beads/issues.jsonl`; a cheap pass could rewrite them as plain-text or archive references.
- The 35↔36 step's small delta (`+0.048`, a count-point-only threshold shift) is a cheap probe for whether the overbid penalty is driven by threshold mass rather than bid magnitude.
- [[w42-bookval-v1-wave2-ch02-multistep]]'s own Monotonicity section says Cohen d "grows" while its table shows the 35↔36 dip; that page could use the same rewording.

## Links

[[winning42-strategy-measurement]] · [[gus-strategy-tags-probe]] · [[gus]] · [[burl]] · [[forge]]
