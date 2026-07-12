---
title: Partnership Value
kind: topic
first_seen: bc4eb386
last_updated: c7f74f5c
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
| Partner coordination | Familiar policies make actions mutually legible: a partner can infer likely holdings, priorities, or continuation from how the other partner tends to play. Explicit conventions are one sparse subset, not the whole phenomenon. | Partner-legibility proxies are directional but underpowered; partner-fit residual is design-only in [[w42-style-partnership-concept-buckets]]. | Fixed-versus-shuffled partner marks remain the defining outcome for a partnership-value claim. |
| Action-derived inference | Voluntary choices change holding or intent likelihoods beyond hard consistency, legal follow, and void evidence. The action may reveal information without having been chosen to send it. | [[w42-book-second-pass]] supplies explicit choice-to-holding hypotheses; auction actions improved [[gus|Gus]] belief in [[w42-champion-auction-belief]]. | Actor-policy likelihood and posterior change are not yet retained by the live harness. |
| Plan persistence | One information-set policy maintains a continuation or partner-visible intent across decisions. | [[book-strategy-player]] defines a plan lifecycle and the book supplies candidate sequences. | Persistent-versus-erased plan state in merged information sets, with partner-visible and partner-blind arms. |
| Distributional utility | The policy uses the shape of an action-outcome PDF only when future information, score, or role licenses it. | [[w42-lens-v1-utility-head-to-head]] leaves EV as the top-scoring tested fixed collapse; the full E[Q] PDF still contains multimodal structure. | Matched-mean PDF fixtures with exact contextual utility; a generic risk preference is insufficient. |
| Bidding | Auction actions price contracts and reveal policy-conditioned information. | [[w42-jud-v1]] and [[w42-plateau-probe]] establish realized-outcome bidding as the current marks gain. | Partner-versus-opponent bid semantics, failed bids, pass value, bid lattice, and policy fingerprint. |
| Match score | Marks-to-go changes auction and continuation utility. It is distinct from raw hand score and raw contract value. | Score is available to [[champion|Champion]] bidding; score-conditioned play lost. No ablation attributes head_8's gain to score. | Same-hand score counterfactuals at `0-0`, `6-6`, `6-0`, and `0-6`, isolated separately for bidding and play. |

## Natural legibility is broader than intentional signaling

Expert partnership discussion commonly asks what could have been known from a
partner's play, how it could have been known, and what was only a guess. The
answer depends on the public state and on how that particular partner tends to
play. It includes intention as well as holdings.

Most information-bearing actions are not selected primarily as messages. A
player may discard to become void because becoming void creates better future
options; the play also reveals something about priorities or shape. The signal
is a byproduct of good play. Familiar partners can interpret that byproduct,
but a simple rule such as "this discard means the player is trying to become
void" is noisy and can be misleading because the tactical reason remains
primary and context-dependent.

Intentional, successful signaling in ordinary play is comparatively sparse and
unreliable. Weak hints that slightly move a belief are ubiquitous. Their
aggregate partnership value is plausible, but its decision-relevant density and
marks contribution are unmeasured. Establishing that signaling exists is
therefore different from establishing it as a promising current attack on
[[the-wall]]. A clean explicit convention can demonstrate a capability without
representing the frequency, ambiguity, or value of natural partnership
legibility.

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

The current failure is therefore not simply "missing belief." The project has
not measured whether the sequence from public action to policy-aware belief,
from belief and role to a partner-legible choice, and from that choice to marks
is dense or valuable enough to explain the wall.

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
[[convention-aware-blueprint-search]]
