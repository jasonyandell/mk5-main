---
title: Winning 42 Ch12 Advanced Bidding Playing
kind: experiment
first_seen: local-2026-05-01
last_updated: 5233160
status: active
---

## Summary

This page completes the bead-backed work surface for `t42-ni1l.12`: Chapter 12,
"Advanced Bidding and Playing." The chapter is a concentrated exception-handling
source for [[gus]], [[burl]], and [[forge]]: the same hand can move from safe
count-forcing to crisis management once public evidence reveals dangerous trump
distribution, and many locally natural plays become wrong once bid margin, count
location, partner expectations, or 84 endgame assets are considered.

The source slice is `scratch/winning42/winning42.with_figures.md` lines 4485-6225.
The grounding workstream is [[winning42-strategy-measurement]], especially its
MVP detectors for `bid_risk_budget`, `off_risk_and_protection`,
`count_liability_surface`, `trump_control_and_reentry`, `partner_count_donation`,
`setter_pounce`, `84_bidder_plan`, and `84_defender_preservation`. The first Gus
probe, [[gus-strategy-tags-probe]], makes this chapter immediately useful because
its action-local tags already improved a tiny policy student, and this chapter
mostly adds action-local exception tags rather than new rules.

## Source Claims

The chapter frames advanced play as disciplined reasoning through trumps, offs,
hand shape, and what other players reveal by playing or failing to play specific
tiles (`scratch/winning42/winning42.with_figures.md` lines 4505-4526). Its early
worked hand contrasts two worlds for the same hand: if trumps clear, protection
stripping and count-forcing doubles make a 35/36 bid safe; if one opponent retains
a trump, the bidder must switch to crisis management and force only the count
needed before the trump can matter (lines 4537-4695).

The middle hands are full-perspective traces. They expose bid-risk accounting,
auction restraint, partner-help dependency, double-not-always-right decisions,
low-trump-first exceptions, opponent pounces, donation/non-donation inference,
and maximum-damage leads from the same public state surface (lines 4837-5707).
The 84 section turns the same ideas into preservation problems: defenders should
save live doubles and same-suit pairs, discard assets only after partner play kills
their use, and use apparently dead discards to communicate public information
without table talk (lines 5747-6051). The close adds natural-bid-bucket and
style/reputation hypotheses: 30, 31, 35, and 36 are treated as the reasoned bid
centers; 32/33 are usually overbid or underbid except in last-seat auction cases;
wild bidding has measurable style and reputation effects (lines 6072-6213).

## Concept Table

| Concept | Source anchor | Detector / state inputs | Metric / test | Likely data source | Priority | Readiness | Implementation notes |
|---|---|---|---|---|---:|---|---|
| Protection stripping before count forcing | Lines 4553-4579 | Bidder hand, live count by suit, protector tile ownership possibilities, legal double leads, current trick leader | Delta in expected captured count and make rate when a non-count protector-stripping double precedes the count-forcing double | Enumerated hands plus [[forge]] oracle rollouts | P0 | Enumeration + oracle + Gus | Emit `protector_strip_window`, `forced_count_target`, and `protector_removed` tags; compare against immediate count lead and random double lead. |
| Terminal set impossibility proof | Lines 4595-4603 | Bid amount, points already secured, max opponent points remaining, trick slots remaining, legal count co-occurrence | Precision/recall of `cannot_be_set_now`; false safe claims are high severity | Engine enumeration over public state | P0 | Enumeration + Burl | Useful as a [[burl]] tool-result summary: when the proof holds, the agent can shift from avoiding all losses to maximizing remaining count. |
| Score-context 84 escalation | Lines 4541-4545 and 4605-4616 | Match score/marks, candidate ordinary bid EV, 84 set probability, comeback deficit, opponent score | Utility crossover where 84 becomes rational despite high set risk | Match simulator using oracle policies and score states | P1 | Oracle + Burl | Chapter gives "way behind" examples; measurement should learn the mark-score utility threshold instead of hard-coding 120 points. |
| Dangerous outstanding trump crisis trigger | Lines 4622-4676 and 5339-5359 | Outstanding trump count, owner belief, bidder trump count, bid margin, needed count, remaining count by suit | Regret of continuing the original trump plan versus switching to count-first crisis play | Oracle rollouts over sampled worlds; Gus belief buckets | P0 | Oracle + Gus + Burl | Emit `crisis_trump_owner`, `needed_count`, and `dangerous_trump_unspent`; bucket by whether opponent must follow the next count-forcing suit. |
| Same hand, different opponent-world value | Lines 4657-4664 and 4688-4695 | Candidate trump suit, off identity, opponent distribution, live count, public evidence after first tricks | Variance of best action/policy by hidden distribution for identical private hand | Paired hidden-world analysis through [[forge]] | P0 | Oracle + Gus | This is a direct Gus belief-value target: the same private hand should not map to a fixed script after public evidence changes. |
| Maximum follow-burden lead | Lines 4631-4656 and 4764-4772 | Legal leads, remaining suit lengths, known voids, dangerous trump holder, desired count target | Probability target holder must follow instead of trumping; make-rate delta | Enumeration with public known voids; oracle validation | P0 | Enumeration + oracle | Generalizes "lead the double whose suit has more dominoes out/unplayed" into a detector over follow burden and trump-denial value. |
| Defender maximum-damage lead after winning | Lines 4706-4729 and 5438-5456 | Defender lead options, callable count, bidder must-trump risk, bid margin, partner void probability | Set conversion and expected opponent points for each lead class | Oracle counterfactual over defender lead choices | P0 | Oracle + Burl | Good Burl trace bucket: did the model identify the lead that either sneaks count or forces the bidder to spend a scarce trump? |
| Emergency trump-in to prevent count catastrophe | Lines 4724-4736 | Current trick count threat, bidder trump scarcity, bid margin, needed count, future trump control | Regret of trumping now versus saving trump; catastrophic set avoidance rate | Oracle rollouts | P1 | Oracle + Burl | This should be separate from ordinary "win trick" because the text says the bidder is not happy to spend the trump but must prevent immediate set risk. |
| Partner-help dependent bid and auction restraint | Lines 4837-4892 and 5274-5289 | Current auction, partner bid status, candidate bid risk, expected partner help, score context | Overcall regret, partner-raise set rate, underbid loss-to-opponent rate | Bidding corpus plus match simulator | P1 | Oracle + Burl | Distinguish "opponent has bid 30" from "partner has bid 30"; the same hand may justify catch-up aggression against an opponent but support when partner already owns the bid. |
| Low-trump-first exception | Lines 5042-5046 | Trump count, highest/second-highest trump held, probability all follow, count in first trick | Regret of low trump first versus boss/high trump first | Oracle rollouts on three-trump hands | P1 | Enumeration + oracle | Adds a chapter-specific exception to the common "lead boss trump" heuristic. |
| Double-not-always-right | Lines 4992-5008, 5230-5235, and 6011-6014 | Double identity, live lower tiles, current bid margin, last-trick value, whether double calls count, whether double is dead | Mistaken-double rate and regret; cases where playing or discarding the double is optimal | Oracle-labeled action buckets and Burl traces | P0 | Enumeration + oracle + Burl | The detector should flag tempting doubles that are dead, misleading, or inferior because another tile sets/makes the bid under the current margin. |
| Safe donation and non-donation inference | Lines 5252-5267 and 5332-5346 | Partner winning/likely winning, count held, alternative count held, partner bid role, observed donation choice | Owner-belief update accuracy for hidden count; donation safety regret | Gus belief eval; Burl trace audit | P0 | Gus + Burl | Non-donation is evidence: if partner donated blank-five instead of double-five, the bidder infers an opponent has double-five. |
| Benign off / decoy lead to induce trump spend | Lines 5360-5367 and 5373-5380 | Bidder protected off, suspected trump holder, partner rescue tile, count at risk in led suit | Regret and trump-spend probability of bait lead versus direct trump/count lead | Oracle rollouts with hidden-world labels | P1 | Oracle + Gus + Burl | This is a compact adversarial bucket for "play looks strange but encodes a hidden protector and a desired response." |
| Setter pounce with count on bidder off | Lines 5517-5558 and 5692-5707 | Bidder off lead, defender void, count in defender hand, partner likely winner/double, bid value and margin | Missed-pounce rate, set conversion, count-held-never-used rate | Oracle rollouts; Burl traces | P0 | Oracle + Burl | The chapter explicitly says to play the count even without knowing who will win, especially against 35/36 bids. |
| Count-trump preservation and delayed pounce | Lines 5609-5665 | Count trump in defender/partner hand, bidder trump exhaustion, current trick winner, remaining count | Regret of saving count trump versus spending it; later set/make delta | Oracle rollouts | P1 | Oracle + Burl | The hand where the defender saves four-five while partner later donates deuce-trey is the positive control. |
| 84 defender live-asset preservation | Lines 5747-5886 and 5906-6000 | Bidder 84, live doubles, same-suit low pairs, partner discards, killed target pairs, forced discards | Pair survival to last two tricks; set conversion; voluntary break regret | 84-specific oracle rollouts | P0 | Enumeration + oracle + Burl | Emit `live_84_double`, `live_84_pair`, `partner_killed_target`, and `must_forsake_pair_for_double`; "never forsake a double for domino pairs" is a measurable priority rule. |
| 84 partner decoy and public information discards | Lines 5789-5798, 5876-5886, and 6039-6048 | Partner hand, defender target assets, low-tile decoys, discards that eliminate partner search branches | Belief entropy reduction for partner; set conversion from cooperative discards | Gus belief eval and Burl trace audit | P1 | Gus + Burl | This is legal inference, not table talk: discards can make partner stop saving dead assets or keep a protected pair. |
| Natural bid bucket anomaly | Lines 6072-6105 | Candidate bid, auction position, last-seat raise need, expected tricks lost, count loss model | Calibration of 30/31/35/36 versus 32/33; overbid/underbid regret | Bidding enumeration plus match simulator | P0 | Enumeration + oracle | Treat 32/33 as a suspicious bucket except when last seat needs only to raise 31; useful for bidding-data quality checks. |
| No-doubles four-off low bid | Lines 6107-6145 | No doubles, four offs, trump count/rank, count trumps held, off risk by side, partner-help requirement | Make rate and set-tail for "weak-looking" 30 bids | Bidding enumeration and oracle rollout | P1 | Enumeration + oracle | Good counterexample to a naive "many offs/no doubles means pass" tag. |
| Style and reputation prior | Lines 6147-6213 | Player/team identity, bid aggression history, low-bid willingness, set-defense skill, partner reputation | Style-conditioned bid/pass regret; opponent overcall response probability | Long-run self-play or human-style simulations | P2 | Gus + Burl | Needs repeated identities. Useful later for Burl because reputation priors can change how an opponent interprets a partner's bid. |

## Highest-Value First Detectors

1. `dangerous_trump_crisis`: flag bidder positions where a retained opponent trump
   makes the original trump-pull script inferior to a count-first crisis plan. This
   is the chapter's clearest same-hand/different-world lesson and should be scored
   by paired oracle regret.
2. `protector_strip_count_force`: identify double leads that remove a non-count
   protector before a later double forces the target count tile. This is cheap to
   enumerate and directly extends the existing `count_liability_surface`.
3. `setter_pounce_high_bid_off`: bucket defender plays where the bidder exposes an
   off and any available count must be played immediately, even if the defender is
   unsure who wins the trick.
4. `double_not_always_right`: mark doubles that are tempting but dead, inferior, or
   wrong under the bid margin. This should become both a Burl trace audit and a Gus
   action-local feature.
5. `84_live_asset_preservation`: track live doubles, same-suit pairs, partner-killed
   targets, and voluntary pair breaks in 84 defense. This is the most important
   advanced 84 bucket in the chapter.
6. `natural_bid_bucket_anomaly`: compare 32/33 bids against 30/31/35/36 and last-seat
   auction exceptions. This can start as a bidding-corpus sanity check before oracle
   game simulation.

## Readiness Notes

Enumeration-ready checks include terminal set impossibility, live/dead count status,
maximum follow burden, natural bid bucket anomaly detection, 84 live asset inventory,
and many static risk decompositions. These do not require a learned belief model;
they require the engine state, legal moves, and remaining unseen tile sets.

Oracle-ready checks include crisis-management switches, maximum-damage defender
leads, emergency trump-in, decoy off leads, low-trump-first exceptions, pounce
choices, and 84 pair-preservation decisions. These need paired action comparisons
under [[forge]] E[Q] because the book's advice is often about regret under hidden
world distributions rather than deterministic legality.

Gus-ready checks include donation/non-donation inference, same-hand/different-world
belief value, 84 partner discard information, dangerous trump owner localization,
and style priors once repeated identities exist. The most natural Gus metrics are
owner Brier/log loss, entropy reduction after public evidence, concept-bucket regret,
and tail-regret under the tags introduced by this page.

Burl-ready checks include whether traces mention the live exception: crisis mode,
protector stripping, pounce urgency, dead double, partner donation inference, or 84
asset preservation. The audit should grade not only the committed action but whether
Burl used legal public evidence rather than an oracle-like shortcut.

## Claim Ledger

[[w42-phase4-bidding-count-exposure-tests]] and
[[w42-phase4-sequence-handshape-tests]] now cover two advanced rows: natural bid
bucket evidence and high-bid off-pounce pressure. Other rows remain
underpowered until state-injected or high-bid generators exist.

| Claim | Status | Next empirical check |
|---|---|---|
| A cleared-trump 35/36 hand can become mathematically unsettable after enough count is forced in. | underpowered | Enumerate max opponent score after each trick and validate `cannot_be_set_now` against exact legal continuation. |
| The same hand should switch from protector stripping to crisis management when a dangerous trump remains. | underpowered | Run paired oracle rollouts for original-script versus crisis-plan actions across hidden worlds with retained opponent trump. |
| 84 is justified from some non-perfect 35/36 hands only under score-context desperation. | underpowered | Simulate match utility by score deficit and compare ordinary bid versus 84 escalation. |
| Doubles are not always the right play; bid margin and dead-suit status can make a non-double superior. | underpowered | Bucket double-available decisions and compare oracle regret for double versus book-recommended non-double. |
| Count pounces on bidder offs are mandatory against high bids even when the defender is unsure who wins. | **contradicted** at bids 35/36/39/42 | [[w42-bookval-v1-wave2-pounce-high-bid]] snapshot-level paired contrast on n=1,140 high-bid pounce-eligible positions: pooled EV delta `-10.42` (CI `[-11.25, -9.59]`), p_set delta `-0.047` (CI `[-0.056, -0.038]`). All 4 bid buckets contradict; pounce-better fraction 21.6-26.6% across bids; effect 3x larger in magnitude than at bid=30. The earlier `context-limited` promotion (Wave 2.B.2, aggregate Q-delta proxy) was reversed by snapshot-level paired evidence. Final status: `contradicted`. |
| 84 defenders should preserve live doubles and same-suit pairs, but discard them once partner play kills the target. | underpowered | Track live asset survival and set attribution in 84 rollouts. |
| 32/33 bids are usually anomalous except as last-seat raises over 31. | partial empirical bucket evidence | `t42-br7n.7` finds natural 30/31 or 35/36 max-profitable buckets in 66 / 384 generated contract rows; real auction-increment behavior remains untested. |
| Reputation for disciplined bidding can provoke opponent overcalls and should affect style priors. | context-limited | Needs repeated-player simulation or human logs; not measurable from single independent forge games alone. |

## Wave 2 Findings (Book Validation v1)

[[w42-bookval-v1-wave2-pounce-window-bid30]] is the most consequential
Ch 12 finding so far. A paired-contrast probe on 52 oracle-greedy
snapshots (filtered from 500 candidates by 1-legal-move and
setter-led-trick exclusions) tested the book's "pounce on bidder
exposed count" instruction against the oracle's choice and against
scalar EV.

The result splits the book's advice along the objective function:

- **Right under `p_make`**: the oracle (which optimizes `p_make` at
  the contract threshold) chose pounce in `59.6%` of paired
  contrasts. The book's instruction is directionally correct under
  this objective.
- **Wrong under scalar EV**: scalar EV said decline was better in
  `65.4%` of paired contrasts. The 10-point count subgroup (n=5)
  was sharpest: EV delta `+15.68` with CI `[+1.60, +29.76]`
  (decline strictly better) yet the oracle still pounced `80%` of
  the time.

Status: `context-limited` for the bid=30 slice. The book's pounce
instruction encodes an implicit `p_make` objective at the contract
threshold; under tail-aware utilities (CVaR, robust_q25 — see
[[w42-bookval-v1-wave1-distribution-lens-reranker]]) it would
recommend a different action substantially more often. The high-bid
extension (35/36/39/42 - the regime the chapter most cares about) has
since run on the bid-aware corpus (bead `t42-8kbh`):
[[w42-bookval-v1-wave2-pounce-high-bid]] contradicted the claim at all
four high bids (see Claim Ledger above).

This finding generalizes: several book claims that read as universal
advice may actually encode `p_make` reasoning that is directionally
correct at threshold-sharp bids and incorrect under different
utilities. A future audit pass should re-classify each
`context-limited` row by which objective function it survives under.

## Links

[[winning42-strategy-measurement]] - [[gus-strategy-tags-probe]] - [[gus]] - [[burl]] - [[forge]]
