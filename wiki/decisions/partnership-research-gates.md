---
title: Partnership Research Gates
kind: decision
first_seen: bc4eb386
last_updated: c7f74f5c
status: active
---

The project selects architecture only after evidence discriminates among the
explanations preserved in [[partnership-wall-research]]. The smallest passing
mechanism earns the next build rung. This decision records eligibility
criteria, not a selected experiment or implementation queue.

## Decision

CFR, a larger network, an LLM, a symbolic strategy library, and [[jud|Jud]] v2 are not
default next steps. Each remains eligible only through a mechanism-specific
gate. This preserves falsifications, adequately measured nulls, bounded null
observations, confounds, and instrument insufficiency as distinct outcomes. It
also prevents interpretability, book agreement, belief accuracy, or E[Q]
imitation from substituting for higher marks.

## Architecture-selection gates

| causal result | eligible build | result that withholds selection |
|---|---|---|
| Per-move targets improve move ranking with capacity held fixed. | Add per-move auxiliary supervision to the existing leaf before increasing model size. | No held-out ranking or marks gain over hand-level targets. |
| Capacity improves only in interaction with the better target. | Scale the leaf to the smallest capacity that clears the interaction. | Capacity-only paper metrics without a policy gain. |
| Auction decoding improves same-hand belief and paired marks across role/order controls. | Build an auction-focused likelihood model and condition bidding/defense on it. | Gain disappears under masked-auction comparison, fails outside one convention population, or the consumer prices double-dummy rather than realized outcomes ([[w42-champion-selfplay-fixed-point]]). |
| Voluntary-action likelihood improves calibrated belief. | Add an action-likelihood update to the posterior. | Better fit without held-out log-loss improvement. |
| Sender-by-decoder interaction improves fixed-pair marks beyond shuffled pairs. | Put partner/opponent models inside the interactive rollout. | Belief improves but the interaction and partnership lift remain null. |
| Persistent state wins only in merged-information or partner-visible plan fixtures. | Add the smallest recurrent option/plan state that carries the demonstrated variable. | Lift exists only in perfect-information worlds already priced by [[forge|Forge]] Q. |
| Full PDF shape changes exact contextual decisions and improves held-out marks. | Add a role/score/future-information-conditioned distribution consumer. | A fixed risk collapse or diagnostic-only PDF difference. |
| A passing mechanism still requires information-set consistency that belief-conditioned rollout cannot supply. | Test bounded information-set resolving or CFR on the demonstrated microgames. | No residual inconsistency after the simpler consumer is installed. |
| A per-move-supervised leaf must also serve as a look-ahead evaluator. | Bounded CFR+ / multi-valued states over the demonstrated microgames — the repair Kubíček & Lisý specify for distilled leaves ([[lamir1-ceiling]]). | Look-ahead gains already achievable with per-move targets alone, or no look-ahead consumer passes its own gate. |
| An LLM selector beats deterministic and small learned selectors on the same microgames and passes reasoning verification. | Use the LLM only at the narrow selection seam it wins. | Better narration, rationale agreement, or tool use without marks lift. |
| Per-move supervision and action-updating opponents pass independently. | A Jud-v2-shaped integration becomes eligible. | Either premise remains unmeasured or fails independently. |
| Reusable plan families beat minimal plan state across held-out fixtures. | A bounded symbolic strategy library becomes eligible. | Fixture-specific rules, ordinary within-world plans, or no generalization. |

## Universal gates

Every promoted mechanism satisfies all of the following:

- runtime inputs are public information plus the actor's own hand;
- value labels match the policy that generated them;
- role and auction-order effects generalize beyond one seat assignment;
- full matches to seven marks reproduce on two held-out seed blocks;
- predictions, uncertainty bands, and falsifiers are registered before the
  final run;
- removing the claimed mechanism removes the marks advantage.

[[w42-jud-v1]] makes policy matching load-bearing.
[[w42-style-partnership-concept-buckets]] makes partner shuffling load-bearing
for a partnership-value claim, not for every route through the wall.
[[champion-design-review]] makes an information-reactive harness load-bearing
for signaling and concealment claims.
[[w42-champion-selfplay-fixed-point]] makes realized-outcome pricing
load-bearing for any auction consumer: its converged belief-conditioned
bidder lost on marks because it priced double-dummy P(make)
([[strategy-fusion]]).

## Build ladder

Current rung: [[partnership-failure-atlas-v0]],
[[world-sampler-mrv-audit]], and [[partnership-decision-record-v1]] land the
joined archive spine, exact-fixture sampler repair, and future Arena identity.
Two-block C0 reproduction remains open; no research mechanism, causal
microgame, or successor architecture is selected by this PR.

1. The joined record schema and exact-enumeration audit land first.
2. `C0 = margin:wp(head_8) + lens:ev` reproduces with bidder, player, sampler,
   utility, score, role, and partner fingerprints.
3. A later research decision selects one causal question from the evidence
   ledger; the current PR does not make that selection.
4. One passing causal mechanism enters as a narrow adapter over `C0`.
5. Policy-conditioned values retrain only when the adapter changes the policy
   distribution.
6. The claimed mechanism is removed in an otherwise identical ablation.
7. Capacity, search depth, self-play, or broader strategy vocabularies scale
   only after the ablation survives.
8. [[w42]] detectors and [[burl]] narration explain the demonstrated gain
   afterward; they do not define promotion.

## Stop and retire rules

- Better belief with no decision or marks effect remains a diagnostic result.
- A marks gain equal under fixed and shuffled partners is not partnership
  value.
- A gain without a mechanism-removal loss is not a causal strategic claim.
- Agreement with [[expected-q-value|E[Q]]] is imitation, not passage through
  the wall.
- Agreement with [[w42-book-second-pass]] validates a hypothesis only in the
  tested information regime; it does not validate the book wholesale.
- A general plan library remains withheld when Forge Q already prices the
  proposed sequence within each exact world.

## General promotion criterion

A candidate replaces the full-match champion only when it:

1. beats `margin:wp(head_8) + lens:ev` in paired marks-to-7 matches on two
   held-out seed blocks;
2. loses that advantage when the demonstrated strategic mechanism is removed.

This criterion applies to every registered route through [[the-wall]]. It
selects a policy with a demonstrated strategic marks gain. Interpretability and
E[Q] imitation remain instruments.

## Additional partnership-value criterion

A partnership claim additionally requires a larger advantage with a matched,
mutually legible pair than with the same policies and deals under shuffled
partners, and that interaction must disappear when the claimed partnership
channel is removed. A gain equal under fixed and shuffled partners may still
break the wall, but it is not [[partnership-value]].

## Links

[[partnership-wall-research]] [[partnership-value]] [[the-wall]] [[champion]]
[[w42-jud-v1]] [[champion-design-review]] [[w42-book-second-pass]]
