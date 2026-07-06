---
title: Winning 42 Ch11 Table Talk
kind: experiment
first_seen: local-2026-05-01
last_updated: afd4802
status: retired
---

**Phantom plan.** This one-session chapter harvest (2026-05-01) registered 14
measurable hypotheses and got zero empirical follow-up in the two months
since — none of its detectors (`table_talk_leakage`,
`trace_public_evidence_faithfulness`, etc.) were ever implemented against
Burl or Gus, and none of its claim-ledger rows were absorbed into the
central 64-row ledger ([[w42-phase4-final-claim-audit]]). The project's
actual frontier moved through the phase-4 closure and on to the
[[champion]]/[[jud]] ladder.

## Summary

This page is the bead-backed work surface for `t42-ni1l.11`: Chapter 11, "Talking
Across the Board." It harvests legal-inference and leakage concepts from Winning 42
into measurable hypotheses for [[gus]], [[burl]], and [[forge]].

Chapter 11 is not mainly a manners chapter. It defines the information boundary that
makes bidding, belief, and play strategy meaningful: players may infer from public bids
and legal play, but they may not transmit private hand facts through verbal or physical
cues. The same boundary matters for model work. [[gus]] strategy tags must not encode
private partner information; [[burl]] traces should justify actions from tool-visible
facts; [[forge]] rollouts should separate legal public-state inference from contaminated
knowledge.

## Source Frame

The chapter slice is `scratch/winning42/winning42.with_figures.md` lines 4332-4484.

The book gives these source claims:

- Direct or indirect remarks during bidding or play can unfairly inform a partner about
  hand strength, helping strength, what to play, or what is in hand.
- A player should not verbally identify trump when a low trump is led; attention to the
  led suit is the player's own responsibility.
- Beginner coaching is allowed only by mutual agreement; normal competitive play has no
  cross-board hints after hands are drawn.
- Table-talk penalties vary by house rules, but repeated or serious violations may void
  or lose the hand.
- A delayed renege can be impossible to reconstruct fairly because later play reveals
  hidden hands; tournament play treats renege as automatic loss of hand.
- Any bidding-phase verbal or physical cue about specific dominoes, doubles, or support
  is cheating.
- The strategic reward of 42 depends on bidders taking risk without knowing partner
  support, then discovering hidden information through the first few tricks.

These claims connect directly to [[winning42-strategy-measurement]] and
[[gus-strategy-tags-probe]]: strategy labels are valuable only if they preserve the
same public-information discipline the game expects from human players.

## Concept Table

| concept | detector/state inputs | metric/test | likely data source | priority | implementation notes | readiness |
|---|---|---|---|---:|---|---|
| `table_talk_leakage` | private facts in current player's hand, partner's legal/public observations, trace text or annotation channel, timing relative to bidding/play | leakage classifier precision/recall; mutual information between non-public facts and partner-facing text/action annotations | Burl traces, synthetic contaminated traces, human-authored chapter examples | P0 | Detect statements that reveal hand quality, helping strength, specific tiles, doubles, count, or trump identity before those facts are public. Use as a guardrail, not as a strategy feature. | Burl-ready; Gus-ready for feature audits; not oracle-ready |
| `bidding_phase_private_signal` | auction history, speaker seat, private hand features, partner next-bid behavior, prohibited cue labels | partner bid shift after cue; illegal-support signal rate; overbid/underbid regret under contaminated vs clean observations | synthetic auction perturbations, Burl bid traces if bids enter harness, generated bidding logs | P0 | Chapter treats bidding cues as worse than in-hand talk because they collapse the bidder's intended risk problem. | enumeration-ready for signal availability; Gus/Burl-ready once bidding traces exist |
| `legal_inference_boundary` | public bids, legal plays, failures to follow, trick history, private hands, model feature tensors | private-feature leakage test: can a probe predict hidden partner support beyond public evidence? | Gus strategy feature dumps, generated game records, oracle world samples | P0 | Audit `strategy_tags` and future features to ensure owner beliefs arise from public history, not true hands. | Gus-ready; enumeration-ready; oracle-ready for outcome deltas |
| `trump_identification_assist` | declaration, led tile, led suit, whether led tile is low trump, trace text/tool calls, player whose turn follows | frequency of explicit trump-identification in trace; downstream play correction rate after illegal hint | Burl traces, synthetic trace tests, engine state snapshots | P1 | The book's example is low trump led during the hand: saying "this is trump" helps inattentive players and should be forbidden in normal play. | Burl-ready; enumeration-ready for state detection |
| `attention_responsibility` | current trick lead, declaration, legal-follow set, model/chosen action, trace claims about suit/trump | misread-led-suit rate; illegal/misprioritized follow choices; regret when led suit is ambiguous-looking but public | generated games, Burl traces, Gus eval buckets | P1 | Measures whether a player can track led suit and trump identity without being told. Useful as an eval bucket separate from hidden-state inference. | Gus-ready; Burl-ready; oracle-ready |
| `agreed_coaching_mode` | ruleset flag, player skill mode, trace messages, before/after hands-drawn boundary | contamination rate under `coaching_allowed=false`; legal explanation rate after hand completion | harness config, trace logs, UI/session metadata | P2 | The chapter allows beginner coaching only by mutual agreement. Treat this as a ruleset/mode gate to avoid mixing teaching traces with competitive play traces. | Burl-ready; ruleset-ready |
| `violation_penalty_policy` | ruleset, violation type, timing, severity, repeated warning flag, current contract state | policy consistency: void hand vs loss of hand; tournament automatic-loss behavior; reward target difference | ruleset variants, simulator annotations, tournament-mode evals | P2 | House rules differ. This should be a variant flag, not a universal game rule baked into base engine policy. | enumeration-ready; oracle-ready with rule variants |
| `renege_immediate_detection` | led suit, player hand, played tile, legal-follow status, declaration, turn index | legal-follow violation precision/recall; immediate catch rate; penalty assignment correctness | engine logs, generated games with injected illegal plays | P0 | This is the clean form: the moment a player fails to follow when able. Existing legality tools should make this deterministic. | enumeration-ready; Burl-ready through `is_legal`/engine |
| `delayed_renege_contamination` | renege trick index, discovery trick index, later exposed tiles, changed public beliefs, contract result | reconstruction ambiguity count; outcome swing between corrected and original line; claim marked void/loss instead of replay | injected-reneges corpus, oracle rollouts from pre/post discovery states | P1 | The book's key empirical claim is not just that renege is illegal, but that late correction contaminates hidden information. | oracle-ready; enumeration-ready |
| `bidder_credit_protection` | bidder team, renege owner, contract made/set under original/corrected worlds, partner culpability | false bidder-punishment rate; cases where bidder clearly made absent opponent renege | injected-reneges corpus, oracle outcome comparison | P2 | The chapter says the bidder should not lose credit for a hand clearly won unless bidder's partner misplayed. Needs policy plus counterfactual outcome attribution. | oracle-ready |
| `early_trick_belief_discovery` | bid history, first few tricks, follow/slough events, count/trump movement, hidden owner posterior | entropy reduction after each early trick; owner Brier/log-loss by trick; regret improvement after update | Gus belief outputs, oracle-consistent world samples, generated games | P0 | This is the positive mirror of anti-leakage: great play comes from discovering where tiles are through public evidence. | Gus-ready; oracle-ready |
| `risk_under_partner_uncertainty` | bidder hand, partner private support, public auction only, contract value, make probability | make/set calibration by true partner support bucket hidden from bidder; overbid regret when support assumed | generated bidding/deal corpus, oracle bid/play rollouts | P1 | The book says bidders must balance max bid and risk without knowing partner help. Measure whether policies rely on legal priors rather than leaked partner state. | oracle-ready; Gus-ready |
| `trace_public_evidence_faithfulness` | Burl thought/tool trace, referenced facts, tool observations, public state, hidden true hand | unsupported-private-claim rate; citation-to-tool/public-state coverage; paired regret for faithful vs leaky traces | Burl STaR traces, synthetic counterfactual traces | P0 | A Burl trace should say "partner is likely void because..." only when public play supports it, not because the training record knows the true hand. | Burl-ready |

## Highest-Value First Detectors

1. `trace_public_evidence_faithfulness`: scan Burl traces for claims about partner support,
   trump ownership, count, doubles, or voids and require a public-state or tool observation
   that supports each claim.
2. `legal_inference_boundary`: audit Gus strategy features and any new concept tags for
   accidental true-hand leakage, especially partner-support and owner-belief tags.
3. `renege_immediate_detection`: deterministic engine bucket for failure-to-follow legality
   and penalty assignment; useful because it is exact and cheap.
4. `delayed_renege_contamination`: inject late-discovered reneges and measure when replaying
   or correcting the hand is epistemically contaminated by revealed hidden information.
5. `early_trick_belief_discovery`: evaluate Gus belief calibration and Burl reasoning after
   the first few tricks, where legal inference should replace forbidden table talk.
6. `trump_identification_assist`: catch traces or UI/helper text that tells a player "this is
   trump" after the declaration already made that public but attention-dependent.

## Readiness Notes

Enumeration-ready:
- `renege_immediate_detection`, `trump_identification_assist` state detection,
  `legal_inference_boundary` feature audits, and ruleset flags for coaching/penalty variants.
- These need no oracle value estimate; they are public-state and legality checks.

Oracle-ready:
- `delayed_renege_contamination`, `bidder_credit_protection`,
  `risk_under_partner_uncertainty`, and regret deltas for attention failures.
- These need [[forge]] E[Q] or perfect-information rollouts to ask whether a violation or
  mistaken reconstruction changes expected outcome.

Gus-ready:
- `legal_inference_boundary`, `early_trick_belief_discovery`,
  `attention_responsibility`, and partner-support risk buckets.
- These should report belief calibration, hidden-hand entropy reduction, owner Brier/log
  loss, and regret by concept bucket.

Burl-ready:
- `trace_public_evidence_faithfulness`, `table_talk_leakage`,
  `trump_identification_assist`, `agreed_coaching_mode`, and renege-tool use.
- These should be applied to traces before using them as STaR or SFT material; leaky
  traces should be rejected or rewritten to cite public observations.

## Claim Ledger

No empirical run was performed for this chapter harvest. The ledger therefore records
analysis status, not evidence status.

| claim | status | next check |
|---|---|---|
| Bidding-phase cues about hand quality or partner help collapse the intended risk problem. | underpowered | Compare bid EV and partner bid shift under clean vs synthetic leaked-support annotations. |
| Verbal trump identification after a low trump lead is an unfair attention assist. | context-limited | Build a trace/UI guard and measure whether Burl or humans misplay less when given the illegal hint. |
| Delayed renege correction is often epistemically contaminated by later revealed hidden hands. | underpowered | Inject reneges at trick `t`, discover at `t+k`, and count correction states whose legal reconstruction is no longer unique or value-stable. |
| Immediate renege detection is deterministic from public led suit, declaration, hand, and played tile. | supported | Validate against engine legality tables and injected illegal-play fixtures. |
| Strong 42 strategy comes from legal early-trick inference rather than private partner information. | underpowered | Evaluate Gus belief entropy/Brier and Burl trace faithfulness after tricks 1-3. |
| Coaching is mode-dependent and should not contaminate competitive traces. | supported | Add `coaching_allowed`/`teaching_mode` metadata to any trace source that includes advice during a live hand. |

## Implementation Hooks

- Extend `strategy_tags` with audit metadata rather than new private facts: `public_evidence_count`,
  `owner_belief_source`, `attention_required`, `renege_possible`, and `coaching_mode`.
- Add a Burl trace lint that marks unsupported private claims before traces enter training.
- Add injected-illegal-play fixtures for immediate and delayed renege checks.
- Keep penalty policy behind a `rule_variant` flag so tournament automatic loss and friendly
  void-hand rules do not bleed into base strategy measurement.

## Links

[[winning42-strategy-measurement]] - [[gus-strategy-tags-probe]] - [[gus]] - [[burl]] - [[forge]]
