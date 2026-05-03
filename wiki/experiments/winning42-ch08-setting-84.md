---
title: Winning 42 Ch08 Setting 84
kind: experiment
first_seen: local-2026-05-01
last_updated: local-2026-05-01
status: active
---

## Summary

This page is the bead-backed work surface for `t42-ni1l.8`: Chapter 8, "Setting the
84 Bid." It harvests 84-defense concepts from Winning 42 into measurable hypotheses
for [[gus]], [[burl]], and [[forge]].

Chapter 8 treats 84 defense as hidden-state preservation. The defender usually cannot
win by ordinary trump or count tactics; the relevant skill is identifying the few tiles
that can beat the bidder's last off, preserving one of them through forced follow-suit
pressure, and abandoning dead assets when the public record proves they no longer matter.
This is a natural `84_defender_preservation` bucket for the strategy-tags workstream.

## Source Anchors

The source slice is `scratch/winning42/winning42.with_figures.md` lines 3548-3818.

- The chapter opens by separating 84 defense from ordinary hands: trumps rarely lose an
  84 bid, count dominoes do not decide it, and defenders should focus on tiles that can
  win the last trick against the bidder's final off (lines 3559-3581).
- The book claims each opponent usually starts with one to four possible last-trick
  weapons, often doubles, but must choose which one to keep when several remain live
  (lines 3583-3593).
- When two doubles remain live near the end, the book recommends keeping the double whose
  suit has the most still-unplayed possible bidder offs (lines 3602-3615).
- If the bidder has the double ahead of the off, saved doubles become useless; the set
  requires preserving two same-suit tiles until the last two tricks (lines 3627-3648).
- Same-suit pairs are vulnerable because either side can be forced out by a led double or
  other suit lead; the defender may need separate protector tiles to avoid breaking the
  pair early (lines 3661-3729).
- The throwaway ladder is: live doubles, live same-suit pairs, pair protectors,
  non-winning tiles, then low tiles that help the partner decide what to keep. Pair
  protectors outrank partner-readable low discards (lines 3733-3784).
- The book gives the key odds claim: before play begins there is a two-to-one likelihood
  an opponent has a setting tile, but many roadblocks prevent carrying that tile to the
  last trick; straight-off 84 is "nearly" set two out of three times, while double-ahead
  84 heavily favors the bidder (lines 3799-3811).

## Concept Table

| concept | detector/state inputs | metric/test | likely data source | priority | implementation notes |
|---|---|---|---|---|---|
| 84-defense mode switch | Declaration value `84`, defender seat, trick index, bidder seat, bidder remaining tiles, public trump and played tiles | Compare ordinary defensive features vs 84-specific features; regret when model spends count/trump resources instead of preserving final weapon | Generated E[Q] games filtered to 84 contracts; future Burl traces with 84 decisions | P0; enumeration-ready, oracle-ready, Gus-ready, Burl-ready | This gates every other detector. It should suppress ordinary `count_liability` priority unless count is relevant to legal follow or final trick ownership. |
| live last-trick weapon inventory | Defender hand, bidder's possible final-off suits from public state, remaining unseen tiles, trump declaration, bidder known played tiles | Precision/recall of tiles that can beat at least one plausible bidder final off; distribution check for the book's one-to-four claim | Exhaustive deals for bid-84 shapes; oracle game records; strategy-tag feature extraction | P0; enumeration-ready, oracle-ready, Gus-ready | A weapon is live if there exists a legal hidden world and final-off tile where the defender's tile wins the last trick. This should include doubles and non-double same-suit tops. |
| live double preservation | Defender doubles, live unseen tiles in each double's native suit, already-played suit tiles, trump suit, trick depth | Voluntary discard regret for throwing a live double; final weapon recall by trick 6/7 | Forge state logs plus counterfactual action scoring | P0; enumeration-ready, oracle-ready, Gus-ready, Burl-ready | Source lines 3583-3625 identify doubles as obvious last-trick candidates but force a choice when several remain. |
| double choice by live-target count | Two or more live doubles at next-to-last decision, count of unplayed possible final offs by suit | Accuracy of choosing the double with more live target tiles; E[Q] delta when choosing lower-count target | Exhaustive hidden-world enumeration at trick 6; oracle rollouts for action regret | P1; enumeration-ready, oracle-ready, Gus-ready | This operationalizes the double-four vs double-deuce example in lines 3602-3615. It is a clean small detector. |
| double-ahead-off vulnerability | Bidder still holds possible same-suit double ahead of final off; defender owns two same-suit tiles that can split the next-to-last and last tricks | Set conversion when pair preserved; missed-pair-set rate; final two tricks explained by pair rather than double | Generated 84 hands with known full deal; oracle action labels | P0; enumeration-ready, oracle-ready, Gus-ready | Source lines 3627-3659 say saved doubles fail against double-ahead 84 and only a same-suit pair can set it. |
| live same-suit pair inventory | Defender hand pip multiplicities, pair suit, trump exclusions, unseen higher/lower same-suit tiles, bidder possible double-ahead plans | Pair survival rate; pair potential recall; regret of breaking a live pair | Strategy-tag pass over full games; constructed 84 endgames | P0; enumeration-ready, oracle-ready, Gus-ready, Burl-ready | A pair is live when it can answer the bidder's next-to-last double with the lower tile and final off with the higher tile. |
| pair vulnerability under follow-suit | Pair tile side suits, led suit, defender alternatives in led suit, forced-follow legality, trick history | Forced break vs voluntary break classifier; regret only assigned to voluntary breaks | Engine legal-play traces; oracle counterfactuals | P0; enumeration-ready, oracle-ready, Gus-ready | Lines 3646-3659 and 3700-3706 distinguish unsafe pairs from safer doubles because the defender can be forced to follow. |
| pair protector inventory | Extra defender tiles in the side suits of a live pair, led double threats, trick depth, remaining bidder doubles | Protector availability; pair survival conditional on protector; regret of discarding protector early | Generated games with full deal; Burl traces for explanation faithfulness | P1; enumeration-ready, oracle-ready, Gus-ready, Burl-ready | Lines 3719-3729 describe protectors as tiles that absorb led suits so the same-suit pair survives until the end. |
| dynamic abandonment of dead assets | Live weapon/pair/protector status before and after each played tile; target tiles exhausted; both pair targets already played | Correct release rate for dead doubles/pairs; regret of continuing to preserve dead assets | Public-state replay from generated games | P1; enumeration-ready, oracle-ready, Gus-ready, Burl-ready | Lines 3692-3699 state that if the watched sixes are played early, the defender can abandon the six pair. |
| throwaway priority ladder | Legal actions when unable to follow suit, live doubles, live pairs, protectors, non-winning tiles, low partner-readable tiles | Ladder violation rate; paired regret of chosen discard vs best ladder-preserving discard | Engine traces plus oracle E[Q] labels; Burl thought traces | P0; enumeration-ready, oracle-ready, Gus-ready, Burl-ready | The book's ordering is directly testable from lines 3733-3784. Priority 3 should beat priority 5. |
| partner-readable low discard | Free discard opportunities, partner seat, partner possible live weapons, tile low/high identity, whether tile is also a protector | Belief shift in partner's possible weapon set; regret of low discard when it sacrifices protector | Pair/team generated games; Burl traces if partner reasoning is explicit | P2; oracle-ready, Gus-belief-ready, Burl-ready | The book allows low tiles to help partner decide what to keep, but explicitly subordinates that to protecting a pair. |
| endgame asset bottleneck | Number of live assets at trick 5/6, legal discard slots remaining, forced-follow risk, partner asset overlap | Bottleneck rate; choice regret when more assets remain than can be held | Full-deal generated 84 hands; constructed adversarial endgames | P1; enumeration-ready, oracle-ready, Gus-ready | Lines 3589-3593 and 3668-3671 frame the endgame as choosing among too many plausible assets. |
| straight-off vs double-ahead odds | Bidder final off structure, bidder double-ahead presence, defenders' initial setting assets, legal pressure paths | Enumerated set probability; compare straight-off set rate to "nearly two out of three" and double-ahead advantage | Exhaustive or sampled deal enumeration under 84-bidder filters | P0; enumeration-ready, oracle-ready | This is the headline statistical claim from lines 3799-3811 and should be checked before training on it. |
| set attribution | Final trick winner, preserved asset lineage, earlier forced/voluntary breaks, partner tile ownership, bidder off structure | Label set as preserved double, preserved pair, partner rescue, bidder error, forced break avoided, or opponent blunder | Full game replay with complete deal; oracle counterfactuals | P1; enumeration-ready, oracle-ready, Gus-ready, Burl-ready | Attribution makes the claim ledger useful: the project should know whether a set came from correct preservation or accidental survival. |
| tracking-load curriculum | Count of watched target tiles, live assets, protectors, forced-follow threats, trick depth | Difficulty bucket vs Gus regret, Burl tool-use depth, and human-readable trace errors | Strategy-tag reports; Burl harvest traces; synthetic contrast pairs | P2; Gus-ready, Burl-ready | Lines 3617-3625 and 3692-3697 describe the cognitive load. This is a curriculum signal, not a rule. |

## First Detectors

1. `is_84_defense_decision`: flag defender decisions during an 84 contract and attach
   bidder seat, trick index, legal action count, and whether the defender is free to discard.
2. `live_final_weapon_tiles`: for each defender tile, mark whether it can still beat at
   least one plausible bidder final off in at least one hidden world.
3. `live_same_suit_pair`: mark two-tile same-suit assets that can beat double-ahead-off
   84, plus the lower-then-higher play order needed on tricks 6 and 7.
4. `pair_protector_status`: for each live pair, count side-suit protectors and identify
   legal actions that would discard a protector before the pair is dead.
5. `throwaway_ladder_rank`: assign each legal discard a Ch8 priority rank and report
   ladder violations, especially pair-protector-before-partner-signal errors.
6. `84_set_attribution`: after an 84 hand resolves, attribute any set to preserved double,
   preserved pair, partner rescue, bidder structure, forced break, or voluntary blunder.

## Readiness

Enumeration can check the static claims: initial defender setting-asset distribution,
straight-off vs double-ahead set frequency, live double counts, live same-suit pair counts,
and whether the book's two-to-one setting-tile claim holds under the project's bidding
filters.

Oracle rollouts are needed for action-quality claims: whether discarding a live double,
breaking a pair, spending a protector, or choosing the lower-live-target double actually
changes E[Q] under the current state.

[[gus]] can use these as public-state and action-local strategy tags. The strongest
training/eval buckets are `84_defender_preservation`, `live_final_weapon`, `live_pair`,
`pair_protector`, `forced_break`, `voluntary_break`, and `throwaway_ladder_rank`. The
existing [[gus-strategy-tags-probe]] result says tags have signal, but this chapter needs
bucketed regret and belief calibration rather than aggregate regret only.

[[burl]] can be evaluated through traces: whether it recognizes the 84-defense mode, asks
tools for unseen/live suits instead of ordinary count math, preserves pair protectors, and
explains forced breaks without claiming illegal hidden knowledge. The [[burl]] tool loop is
well-suited to this because the useful behavior is reasoning with visible state and then
committing, not memorizing a book rule.

[[forge]] supplies the rule authority and E[Q] baseline. The first pass should use engine
state replay and E[Q] counterfactuals; later work can add adversarial 84 endgames where the
only winning action is preservation rather than point capture.

## Claim Ledger

[[w42-phase4-84-dynamic-seed-tests]] now supplies reached-state action evidence
for 84 defense over mined natural seeds. It supports preservation and dead-asset
release proxies, while final set attribution, throwaway-ladder bottlenecks, and
population set-rate claims remain blocked.

| claim | status | next empirical check |
|---|---|---|
| 84 defense should use a different mode from ordinary defense because count and trump incentives rarely decide the set | context-limited support | Phase 4 reached 84 defense decisions and preservation labels; ordinary-vs-84 policy comparison remains open. |
| Each opponent usually starts with one to four possible last-trick weapons | context-limited seed inventory | Phase 3 mines defender live-double and same-suit-pair surfaces at scale; exact per-opponent weapon distribution needs a dedicated table. |
| When choosing between live doubles, the better double is the one with more unplayed possible final-off tiles | context-limited | Test E[Q] deltas in next-to-last-trick states with multiple live doubles. |
| Double-ahead-off 84 can only be set by preserving a same-suit pair, not by saving a double | context-limited | Enumerate double-ahead structures and verify set paths by final two-trick simulation. |
| Pair protectors should be kept ahead of partner-readable low discards | context-limited / blocker | Phase 4 found 12 protector actions and positive preservation evidence, but full throwaway-ladder bottlenecks remain blocked. |
| Straight-off 84 is set nearly two out of three times, while double-ahead-off heavily favors the bidder | blocked | Phase 4 selects straight-off games, but population set rate and final set attribution are not tested by reached greedy traces. |
| Tracking load is a real difficulty signal for models and agents | underpowered | Bucket Gus regret and Burl trace errors by live assets plus watched target tiles. |

## Links

[[winning42-strategy-measurement]] | [[gus-strategy-tags-probe]] | [[gus]] | [[burl]] | [[forge]]
