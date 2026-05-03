---
title: Winning 42 Ch03 Bidder Play
kind: experiment
first_seen: local-2026-05-01
last_updated: local-2026-05-01
status: active
---

## Summary

This page completes bead `t42-ni1l.3`: Chapter 3, "How to Play Your Hand after
Winning the Bid." The chapter turns bidding from a static hand score into a play
sequence: the bidder must plan how to pull trumps, expose or delay offs, invite
partner count donations, preserve a reentry trump, and prove the final tricks.

The source slice is `scratch/winning42/winning42.with_figures.md` lines 1407-2057,
covering book pages 23-35. The source is used as a hypothesis generator for
[[gus]], [[burl]], and [[forge]], not as ground truth. No empirical run was
performed for this bead, so every claim below is recorded as untested until
enumeration, oracle rollouts, Gus belief outputs, or Burl traces check it.

## Grounding

Chapter 3 sits directly inside [[winning42-strategy-measurement]]: bidder play is
the first chapter where book advice naturally becomes per-decision tags rather
than whole-hand labels. It extends [[gus-strategy-tags-probe]] by naming action
local features that should be cheap to compute: outstanding trumps, dangerous
offs, live count, partner donation windows, setter pounce windows, and claim
proof.

The chapter also maps cleanly onto existing project roles:

- [[forge]] can enumerate visible state, legal moves, trick winners, live count,
  and E[Q] deltas for alternative bidder sequences.
- [[gus]] can be audited for whether public-state/action tags reduce regret in
  bidder-sequencing buckets.
- [[burl]] can be audited for whether its tool calls and thoughts mention the
  same public facts: remaining trump, off risk, partner donation safety, and
  whether a laydown is actually proven.

## Measurable Concepts

| Concept | Source-backed claim | Detector/state inputs | Metric/test | Likely data source | Priority | Readiness | Implementation notes |
|---|---|---|---|---|---:|---|---|
| Pull trumps before dumping offs | The default bidder plan is to lead trumps first so opponents must spend their trumps before vulnerable offs are led. | Bidder seat, declaration, bidder hand, legal lead, outstanding trump count, opponents' forced-follow status, off inventory. | Paired E[Q] delta of trump-first vs off-first lead when both are legal; set rate by outstanding-trump bucket. | Forge generated games plus counterfactual oracle rollouts. | P0 | enumeration, oracle, Gus, Burl | Emit `bidder_pull_trump_window` and `off_before_trump_exception` tags. |
| Early off exception | The book allows leading the off early when the bidder has one off and at least four trumps, because opponents may void themselves if trumps are pulled first. | Off count, trump count, off suit, count callable by off, outstanding trump rank, whether opponents can trump if off is led now. | Compare make/set rate for early-off vs trump-first in one-off/four-trump hands; tail loss from immediate trump-in. | Exhaustive hand-shape enumeration, then Forge rollouts. | P0 | enumeration, oracle, Gus | This is a high-value exception bucket because it tests whether models learn "default rule plus guarded exception." |
| Preserve a reentry trump | The bidder should usually save at least one trump until after offs are played, even when extra trump leads could collect partner count. | Bidder trump count after candidate play, offs remaining, current lead control, live dangerous count, available doubles/walkers. | Regret of spending final trump before all offs are resolved; probability of failing to regain lead. | Forge rollouts and Gus policy buckets. | P0 | enumeration, oracle, Gus, Burl | Emit `last_trump_spent_before_off_clear` and `reentry_trump_preserved`. |
| Partner donation on guaranteed trump trick | If partner is void in trump or has only one trump, bidder can sometimes lead a guaranteed high trump so partner can donate count safely. | Partner follow failure or highest-remaining-trump signal, bidder's trick-winning guarantee, partner free-discard status, partner count availability if known in world. | Good donation rate; unsafe donation rate when bidder trick was not guaranteed; E[Q] lift from one extra guaranteed trump lead. | Perfect-info enumeration for availability; public-state oracle rollouts for decision value; Burl traces for rationale. | P0 | enumeration, oracle, Gus, Burl | Public detector should separate "donation window exists" from hidden "partner actually has count." |
| Double ahead of off | When bidder has a double in the same suit as an off, leading the double first can draw a vulnerable count tile before the off trick. | Off tile pips, same-suit double in hand, live count in that suit, count already captured, partner/opponent ownership in sampled worlds. | Verify the book's stated success-rate shape; compare double-first vs off-first E[Q]. | Exhaustive deal enumeration and Forge rollouts. | P0 | enumeration, oracle, Gus | Good first statistical-analysis claim because the book gives explicit approximate percentages. |
| Extra trump can create setter pounce | Leading an unnecessary extra trump before an off may let an opponent discard their last card in the off suit, creating a void and pounce window. | Candidate extra-trump lead, opponents' off-suit holdings in world, public void evidence after the trick, off still in bidder hand, live count. | Setter-pounce creation rate; regret of extra trump in one-off hands; tail loss conditional on opponent void creation. | Perfect-info enumeration for mechanism; public-state oracle for policy impact. | P0 | enumeration, oracle, Gus | This is the defensive shadow of the default trump-pull rule. |
| Dangerous outstanding trump | If any trump not in bidder hand remains, the bidder treats it as dangerous, especially when it can beat bidder's remaining trump or win count. | Outstanding trump count, highest missing trump, bidder remaining trump rank, current trick count pressure, offs unresolved. | Regret and set rate when bidder ignores dangerous trump; recovery success after emergency trump-in. | Forge rollouts, Gus concept buckets, Burl trace checks. | P1 | enumeration, oracle, Gus, Burl | Emit `highest_missing_trump` and `dangerous_outstanding_trump`. |
| Sacrificial low trump to retain command | In a sample hand, the bidder leads a lower trump first so an opponent wins with a high trump, leaving bidder with boss trump and a safer structure. | Trump rank ordering, bidder holds boss plus lower trumps, count at stake on first trick, number of remaining trumps after sacrifice. | Compare low-trump-first vs boss-trump-first on hands with boss plus vulnerable lower trump; measure command retention. | Forge rollouts and hand-shape enumeration. | P1 | enumeration, oracle, Gus, Burl | Important because it contradicts the naive "always lead highest trump" simplification. |
| Count-at-risk inventory | Bidder continuously tracks which count dominoes are already captured, in hand, live, or only available to opponents. | Count tiles by status, tricks won, current bid margin, maximum affordable future loss, count callable by candidate leads. | Catastrophic count-dump rate; regret by `count_at_risk` bucket; set conversion when unexpected count appears. | Existing generated games plus strategy tag analyzer. | P0 | enumeration, oracle, Gus, Burl | This should become a core `strategy_tags` feature, not a chapter-only metric. |
| Opponent void and pounce timing | Opponents often set the bidder by being void in the off suit exactly when the bidder leads the off and then dumping live count. | Public void evidence, hidden off-suit holdings in sampled worlds, live count available to void opponent, bid margin. | Pounce-window precision/recall; missed-pounce and created-pounce attribution; bidder regret on off timing. | Perfect-info labels plus public belief rollouts. | P0 | enumeration, oracle, Gus, Burl | Pairs with Chapter 5 setter analysis; this page owns the bidder-side risk detector. |
| Endgame suit-exhaustion proof | A late low off can become unbeatable if prior double leads force all higher same-suit tiles out. | Remaining tiles by suit, previous forced-follow events, live higher tiles, bidder last lead, trick depth. | Claim-proof accuracy; regret when bidder leads a double to exhaust a suit vs prematurely leads the off. | Enumeration and Forge rollouts. | P1 | enumeration, oracle, Gus, Burl | Emit `effective_walker_after_exhaustion` and `last_off_proven`. |
| Claim or laydown correctness | The bidder may declare the rest if every remaining trick is forced, but a wrong claim loses the hand under the book's rule. | Remaining hand, unseen legal responses, trumps remaining, doubles/walkers, current lead, possible opponent trick wins. | Exact proof checker: all completions make bidder win every remaining trick; false-positive claim rate. | Deterministic engine enumeration. | P0 | enumeration, Burl | This is the cleanest no-ML detector in the chapter and a strong tool for Burl. |
| Bid justification by loss budget | The book judges a bid by whether planned and rare loss scenarios fit the contract, not whether the sample deal happened to make or set. | Bid value, expected planned losses, maximum affordable count loss, rare set-worlds, observed outcome. | Calibration: make probability and expected regret by planned-loss budget; distinguish good bid that got unlucky from bad bid. | Forge rollouts, bid corpus, Gus policy buckets. | P1 | oracle, Gus, Burl | Bridges Chapter 2 bidding and Chapter 3 play sequencing. |

## First Detectors

The first implementation batch should prefer detectors that are deterministic,
low-leakage, and immediately useful for Gus/Burl bucketed evaluation:

1. `bidder_reentry_trump_preserved`
   - Inputs: bidder remaining trumps, offs remaining, candidate action, current lead.
   - Positive when the bidder still has at least one trump after the action while
     unresolved offs remain; negative when the final trump is spent before off
     risk is gone.

2. `double_ahead_of_off_window`
   - Inputs: off tiles, same-suit doubles in bidder hand, live count in that suit,
     already captured count.
   - Positive when leading the double can force or capture count before the off is
     exposed.

3. `partner_safe_count_donation_window`
   - Inputs: partner's public inability to follow trump or singleton-high-trump
     signal, bidder's guaranteed trick status, partner free-discard status.
   - Public label marks a window; perfect-info label marks whether count donation
     was actually available.

4. `setter_pounce_created_by_extra_trump`
   - Inputs: optional extra trump lead, off still in bidder hand, opponent last-card
     off-suit status in sampled world, live count.
   - Attribution bucket for extra trump leads that make the future off trick worse.

5. `dangerous_outstanding_trump`
   - Inputs: highest missing trump, remaining bidder trump ranks, live count and
     off status, trick position.
   - Gives Gus and Burl a cheap public reason to explain emergency trump-ins and
     delayed off dumping.

6. `claim_all_remaining_tricks_proven`
   - Inputs: exact remaining public/hidden state under a sampled full world, legal
     continuations, current lead.
   - Deterministic proof checker for laydown correctness and false claim risk.

## Readiness Notes

- Enumeration-ready: claim proof, suit exhaustion, double-ahead-of-off combinatorics,
  partner donation availability under full worlds, and last-trump/reentry labels.
- Oracle-ready: alternative lead sequences, early-off exception, extra-trump pounce
  creation, sacrificial low-trump sequencing, and bid loss-budget calibration.
- Gus-ready: public/action-local tags for `outstanding_trumps`, `highest_missing_trump`,
  `reentry_trump_preserved`, `count_at_risk`, `off_risk`, and `partner_donation_window`.
- Burl-ready: trace audits for whether the model asks about trump status, live count,
  partner donation safety, off risk, and claim proof before committing.

## Phase-3 Counterfactual Follow-Up

[[w42-phase3-sequence-seat-counterfactuals]] gives Chapter 3 its first broad
phase-3 branch-value follow-up. It does not yet expose full hand-shape gates for
trump count, off count, reentry, or live-count inventory, but it does compare
legal candidate moves in naturally occurring sequence states.

The main lead-plan result is context-limited rather than promotional. Pooled
bidder called-suit leads trail off-suit leads by `-1.935` Q across 1657 paired
states, including early slices. However, called double leads beat lower
called-suit leads by `+5.720` Q across 182 paired states. The project should
therefore split the old "pull trumps first" row into sharper detectors:
commanding trump/double lead, early-off exception, low-trump command exception,
count-leading risk, and reentry/off-clear preservation.

Chapter 3's partner-support and count-inventory claims also get new boundaries:
partner count when the bidder side controls the trick is only `+0.608` Q across
614 pairs, while partner count into defensive control is `-8.351` Q across 802
pairs and bidder count leads are `-4.877` Q versus non-count leads. The folk rule
survives as a gated timing rule, not a blanket invitation to dump count.

## Phase-4 Hand-Shape And Laydown Follow-Up

[[w42-phase4-sequence-handshape-tests]] gives Chapter 3 a sharper public/action
local pass over the same full legal-action table. Commanding called doubles beat
off leads by `+1.316` Q across 576 paired bidder lead decisions, while generic
non-double called-suit leads lose to off leads by `-3.682` Q across 1263 pairs.
This preserves the "commanding trump" intuition and further rejects the blanket
trump-first simplification.

[[w42-phase4-laydown-rule-accounting]] gives Chapter 3 its first exact
laydown-proof artifact. The checker enumerates every legal continuation in
small full-information late-state fixtures and requires the claimant's team to
win every remaining trick. It proves boss-trump and suit-exhaustion walker
claims, and rejects false laydowns when an opponent can still take control. The
book-like final deuce warning is now executable: if an opponent still holds a
higher deuce, the claim is rejected with a concrete counterexample.

## Claim Ledger

Phase-3 follow-up exists for several rows, but full hand-shape/reentry tests are
still pending. Status values describe the current evidence state, not the truth
of the book claims.

| Claim | Status | Next check |
|---|---|---|
| Trump-first is usually better than off-first for bidder play. | context-limited | `t42-br7n.1` supports commanding called doubles over off leads, but generic non-double called-suit leads are strongly negative. Next bucket by exact trump count, off count, and live count. |
| One-off/four-trump hands can justify an early off. | context-limited | Enumerate one-off/four-plus-trump shapes and compare off-first tail risk against trump-first void creation. |
| The bidder should preserve at least one trump until after offs are resolved. | context-limited | Measure final-trump-spent-before-off-clear regret and failure-to-regain-lead rate. |
| Partner should donate count only when bidder has a guaranteed winning trick. | context-limited | `t42-qtwb.3` shows small-positive support under bidder-side control and strong negative support into defense; next add guaranteed-trick and partner-free-discard gates. |
| Leading a double ahead of an off wins the vulnerable count substantially more often than leading the off first. | context-limited | Reproduce the book's double-deuce/deuce-blank style odds by exhaustive enumeration. |
| Extra trump leads can create opponent voids that set the bidder later. | context-limited | Attribute future off-trick pounces to optional earlier trump leads. |
| Low-trump-first can preserve command better than boss-trump-first in some hands. | context-limited | Counterfactual oracle rollout on hands with boss trump plus unresolved lower trumps and offs. |
| A laydown is valid only when every legal continuation wins the rest. | supported-by-fixture-proof, corpus-untested | `t42-br7n.4` implements an exact tiny-state proof checker and rejects false claims, including the final-deuce warning. Next: wire saved engine snapshots and Burl trace claims into the checker. |

## Links

[[winning42-strategy-measurement]] · [[gus-strategy-tags-probe]] · [[gus]] · [[burl]] · [[forge]]
