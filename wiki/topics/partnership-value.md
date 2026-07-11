---
title: Partnership Value
kind: topic
first_seen: bc4eb386
last_updated: bc4eb386
status: active
---

Partnership value is the marks lift produced by mutually legible interaction,
not the sum of two strong individual policies. It is the target isolated by
[[partnership-wall-research]].

## Operational measure

The proposed primary estimate is:

`partnership lift = paired team marks with a fixed, mutually legible pair - paired team marks with shuffled partners`

Individual policy strength, deals, seats, bidder share, and opponent strength
remain controlled. A gain that survives partner shuffling unchanged is team
strength, not partnership value. [[w42-style-partnership-concept-buckets]]
shows that current corpora lack repeated policy identities and shuffled-pair
cohorts, so this quantity has not yet been measured.

## Distinctions the metric must preserve

| dimension | meaning | surviving evidence | required observation |
|---|---|---|---|
| Uncertainty | Posterior shape, outcome variance, and action fragility under hidden worlds. It is not itself coordination. | [[belief-bayes-ceiling]] places top-1 belief near its consistency-information ceiling; [[gus-drama-atlas]] finds 26.2% foggy-fragile decisions and localizes most drama to leads. | Calibrated posterior and per-world action consequences before and after public evidence. |
| Role and order | Bidder, partner, left/right setter, leader/follower/closer, and auction position change the meaning of the same tile. | [[w42-phase2-seat-position-strategy-map]] maps distinct role lenses; removing sequence/seat erased most of the joined W42 gain in [[w42-phase3-joined-claim-row-model-table]]. | Role-composed strata and counterfactual seat/order controls, not raw seat tokens alone. |
| Partner coordination | A sender's action changes a receiver's useful response under a shared convention. | Partner-legibility proxies are directional but underpowered; partner-fit residual is design-only in [[w42-style-partnership-concept-buckets]]. | Sender-by-decoder interaction plus fixed-versus-shuffled partner marks. |
| Action-derived inference | Voluntary choices change holding likelihoods beyond hard consistency, legal follow, and void evidence. | [[w42-book-second-pass]] supplies explicit choice-to-holding hypotheses; auction actions improved [[gus|Gus]] belief in [[w42-champion-auction-belief]]. | Actor-policy likelihood, posterior update, and shuffled-action negative controls. |
| Plan persistence | One information-set policy maintains a continuation or partner-visible intent across decisions. | [[book-strategy-player]] defines a plan lifecycle and the book supplies candidate sequences. | Persistent-versus-erased plan state in merged information sets, with partner-visible and partner-blind arms. |
| Distributional utility | The policy uses the shape of an action-outcome PDF only when future information, score, or role licenses it. | [[w42-lens-v1-utility-head-to-head]] leaves EV as the top-scoring tested fixed collapse; the full E[Q] PDF still contains multimodal structure. | Matched-mean PDF fixtures with exact contextual utility; a generic risk preference is insufficient. |
| Bidding | Auction actions price contracts and reveal policy-conditioned information. | [[w42-jud-v1]] and [[w42-plateau-probe]] establish realized-outcome bidding as the current marks gain. | Partner-versus-opponent bid semantics, failed bids, pass value, bid lattice, and policy fingerprint. |
| Match score | Marks-to-go changes auction and continuation utility. It is distinct from raw hand score and raw contract value. | Score is available to [[champion|Champion]] bidding; score-conditioned play lost. No ablation attributes head_8's gain to score. | Same-hand score counterfactuals at `0-0`, `6-6`, `6-0`, and `0-6`, isolated separately for bidding and play. |

## Current champion boundary

The full-match [[champion]] converts available information into two things
that are already valuable:

- `lens:ev` ranks moves across worlds consistent with public history;
- `margin:wp(head_8)` prices bids under realized policy outcomes.

It has not demonstrated:

- behavioral likelihood updates from voluntary play;
- sender/receiver conventions or fixed-partner lift;
- information-set persistent plans;
- partner-visible concealment or signaling value;
- contextual use of the full outcome distribution;
- a causal match-score mechanism.

The current failure is therefore not simply "missing belief." It is the
unmeasured sequence from public action to likelihood-weighted belief, from
belief and role to a partner-legible choice, and from that choice to a marks
gain against an information-reactive opponent.

## Plan boundary

[[champion-design-review]] corrects the early generic-plan claim attached to
[[book-strategy-player]]. [[forge|Forge]]'s perfect-information Q already values a
candidate action through the end of each exact world. Ordinary within-world
plans are not absent merely because the consumer makes a fresh choice each
turn.

The remaining plan hypotheses are narrower:

- one policy must remain consistent across several possible worlds;
- an action may preserve or reveal information to a partner;
- an opponent may exploit revealed intent;
- a later choice may need to honor a convention not recoverable from the
  current scalar value alone.

The concrete sequences in [[w42-book-second-pass]] test those hypotheses only
when embedded in merged information sets and reactive partnerships. They are
not prior evidence for a symbolic strategy library.

## Mechanism seams

The research surface is a set of joins rather than one model choice:

- auction action -> actor likelihood -> belief;
- voluntary play -> actor likelihood -> belief;
- belief + role/order -> partner-legible action;
- action + prior convention -> partner response;
- plan state + new evidence -> continuation or retirement;
- outcome distribution + score/context -> utility;
- policy-conditioned outcome -> bid price;
- fixed-pair interaction -> full-match marks.

Each seam needs its own ablation. Improvements in belief accuracy,
explanation quality, or oracle imitation do not establish partnership value
unless they cross the final fixed-pair marks seam.

## Links

[[partnership-wall-research]] [[partnership-research-gates]] [[champion]]
[[expected-q-value]] [[gus]] [[w42]] [[w42-book-second-pass]]
