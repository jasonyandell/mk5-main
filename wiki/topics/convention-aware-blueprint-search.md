---
title: Convention-Aware Blueprint Search
kind: topic
first_seen: 1a4482fe
last_updated: local-2026-07-12
status: active
---

Classification: **IDEATED, unbuilt, unselected.** No current code artifact
implements this architecture. This page preserves the proposal and its
project-record refinement; it is not a build decision.

## Proposal

### Diagnosis

[[expected-q-value|E[Q]]] evaluates each legal action across sampled hidden
worlds, then uses a perfect-information continuation inside each exact world.
That is strong at ordinary tactical continuation, but the simulated future
does not have to use one policy across worlds the acting seat cannot
distinguish.

The proposal identifies the remaining defect as information-set strategy:

- strategy fusion permits different continuation choices in indistinguishable
  worlds;
- future seats inside the exact-world evaluation already possess the hidden
  information that a real partner or opponent would have to infer;
- a move whose only benefit is changing another seat's later belief has no
  channel through which to earn that benefit;
- concealment, action-derived inference, and convention-dependent partner
  response are therefore absent from the continuation objective.

The proposal does not claim that such moves lack direct tactical value. It
claims that perfect-information continuation cannot price the part of their
value carried only through a later imperfect-information response.

### Blueprint search

The proposed repair adapts SPARTA-style search from cooperative partially
observable games ([Lerer et al., 2020](https://arxiv.org/abs/1912.02318)). A
fixed, commonly known policy acts as the **blueprint**. Each seat is assumed to
choose from its own information set under that same policy.

Beliefs are filtered by policy consistency as well as deal consistency:

`P(world | public action) proportional to P(blueprint chooses action | actor information in world) * P(world)`

Candidate actions are evaluated by sampling consistent worlds and rolling
forward imperfect-information seat policies. Each simulated seat receives
only its private hand and public history. A candidate lead or discard can then
change the partner's posterior and subsequent play inside the rollout, giving
an already understood convention or naturally legible action a measurable
continuation value.

The smallest version is single-agent search: only the acting seat deviates
from the blueprint while partner and opponents remain fixed to it. A later
multi-agent version could allow other seats to search at public states where
the relevant search procedure is common knowledge.

### Iteration

The proposal closes the loop through expert iteration:

1. search for profitable deviations from the current blueprint;
2. distill the improved choices into a new blueprint generation;
3. use that same generation for action likelihoods and rollout behavior;
4. repeat only if the new policy remains a coherent shared codebook.

This creates a possible route for established conventions to improve and for
coordinated conventions to form. The latter is a hypothesis, not an automatic
consequence of unilateral search.

### Existing infrastructure it could consume

- a stochastic policy supplies the blueprint and action likelihoods;
- consistent-world particles represent hidden hands;
- the repaired [[world-sampler-mrv-audit|uniform world sampler]] supplies
  rejuvenated particles when likelihood filtering collapses;
- public-history and private-hand views define each rollout seat's information
  set;
- a policy-matched continuation value truncates expensive rollouts;
- [[partnership-decision-record-v1]] records policy, sampler, belief, role,
  partner, and action provenance for attribution.

### Known failure modes

- **Particle collapse:** repeated likelihood weighting can destroy effective
  sample size; resampling and consistent-deal rejuvenation are required.
- **Likelihood miscalibration:** an overconfident blueprint can eliminate
  plausible worlds; likelihood floors or tempering may be required.
- **Rollout cost:** full-hand imperfect-information rollouts can be expensive;
  truncation is useful only when the leaf preserves the post-action belief and
  continuation policy.
- **Convention drift:** a searcher, likelihood filter, partner policy, or leaf
  trained from different blueprint generations can read a different codebook.
- **Opponent mismatch:** opponents are not cooperative blueprint partners and
  may exploit predictable conventions.

### Clairvoyance headroom idea

The proposal also preserves a diagnostic ladder using the same policy
consumer:

1. current information (`C0`);
2. partner hand revealed;
3. all hidden hands revealed.

Paired action flips and outcome changes could localize decisions sensitive to
hidden information, especially bidding and early leads. This is a cheap
headroom probe before a blueprint-search build.

## Refinement against the project record

### The missing object is narrower than planning

The [[champion]] does not lack planning in the general sense. [[forge|Forge]] Q
solves through the end of a hand within each exact world. Singleton setup,
trump pulling, re-entry, count cashing, and other ordinary within-world plans
are already priced there.

The proposal's surviving target is narrower: one information-set-consistent
continuation across worlds, with beliefs and partner responses changed by
public actions. “Information-set planning and coordination” is accurate;
“planning is absent” is not.

### The prior-attempt ledger must remain exact

The project built AlphaZero-style work, belief heads, LAMIR, MCCFR, and STaR.
It did not run every nearby family in Texas 42:

- partial ISMCTS was set aside by argument and deployment constraints, not
  established as a Texas 42 empirical negative;
- world-model directions were largely ideated or retracted, not completed
  wall attacks;
- ReBeL ran in a Go Fish testbed, not in Texas 42;
- ReBeL- and Student-of-Games-shaped directions therefore remain theoretical
  comparisons rather than Texas 42 negative results.

ReBeL's guarantee is specifically for two-player zero-sum games
([Brown et al., 2020](https://arxiv.org/abs/2007.13544)); Player of Games
likewise presents its sound search formalism for two-player games
([Schmid et al., 2021](https://arxiv.org/abs/2112.03178)). Texas 42 combines
partnership cooperation with an adversarial opposing team and private
information not shared within either partnership. That weakens direct transfer
of those guarantees, but it does not by itself prove blueprint search is the
only principled route.

### The current champion is not the proposed blueprint system

The current full-match champion is
`margin:wp(head_8) + lens:ev`, not a policy/value-network searcher.
[[w42-jud-v1|JudSearch]] is closer to the described search shape and still lost
to `lens:ev`.

No current policy has all four properties the design requires:

- C0-level play strength;
- stochastic, calibrated action likelihoods;
- execution from every seat's legal information set;
- a continuation value matched to that policy and blueprint generation.

SPARTA-style search offers a policy-improvement guarantee relative to its
blueprint under its cooperative assumptions. It does not guarantee improvement
over C0 when the available blueprint is weaker than C0.

### Policy filtering adds information, not accuracy for free

The `39.184%` figure in [[belief-bayes-ceiling]] is a top-1 ceiling for one
consistency-information posterior and corpus. An observed action's likelihood
under a policy supplies information outside that posterior, so the figure does
not cap action-conditioned inference.

The added information is useful only when the likelihood model is calibrated
and policy-matched. “Potentially sharper than consistency-only belief” is
supported structurally. “Far sharper” or “improved for free” is not yet a
Texas 42 result.

### Search can read a convention before it can invent one

Single-agent blueprint search can exploit an existing decoded convention or
the natural legibility of an established policy. It cannot make a novel signal
valuable when the fixed partner blueprint does not interpret it. Unilateral
expert iteration also need not cross the coordinated sender/receiver barrier:
the sender's novel action is initially costly or neutral because the receiver
still uses the old codebook.

Claims of convention creation therefore require a synchronized joint-policy
change, a multi-agent common-knowledge search step, or another explicit
coordination mechanism. This is distinct from improving play within a fixed
convention.

### The Hanabi guarantee does not transfer unchanged

SPARTA assumes a fully cooperative game with a common reward across all
agents. Texas 42 has common reward only within each partnership; opponents are
adversarial. Treating opponents as fixed environment policies remains a
plausible engineering adaptation, but the cooperative improvement guarantee
does not automatically survive it.

Team-maxmin work with ex-ante correlation supports coordinated distributions
over joint team strategies
([Farina et al., 2021](https://www.mit.edu/~gfarina/2021/tmecor-correlation-icml21/)).
It supports the relevance of pre-agreed coordination; it does not establish
one fixed convention as the unique solution.

### The leaf must preserve the mechanism

A generic value-net truncation can erase the proposed gain. The leaf must be
conditioned on the post-action belief, blueprint generation, convention, and
continuation policy. Changing the blueprint changes the policy distribution
being valued. [[w42-jud-v1]]'s policy-conditioned pricing law therefore applies
inside every expert-iteration round; an existing value net is not automatically
reusable.

### PIMC theory is a hypothesis source, not a Texas 42 measurement

Perfect-information Monte Carlo performs well when leaf values are correlated
across hidden worlds, information is revealed quickly, and bias is favorable
([Long, Sturtevant, and Buro, 2010](https://ojs.aaai.org/index.php/AAAI/article/view/7562)).
Those predictors have not been measured directly in Texas 42.

The project does establish a bidding marks gain and a concentration of
decision drama in early leads. It has not established that the residual value
is concentrated specifically in bidding, the first two leads, and signal
management. Signal-management marks remain unmeasured.

### Clairvoyance is sensitivity, not a universal bound

Revealing hidden hands to a fixed policy consumer measures how much that
consumer's behavior and results change when given the information. It does not
measure the total attainable value of hidden information and is not a universal
upper bound on imperfect-information improvement. A consumer may fail to use
revealed information, or may use it in ways unavailable to any legal policy.

A small clairvoyance gap also cannot imply that the missing value was “mostly
conventions.” Conventions are themselves a way to convey or infer hidden
information, so a genuinely small information-sensitive gap would normally
reduce, not isolate, that headroom. The useful surviving artifact is a
per-decision C0 sensitivity atlas, interpreted as consumer-specific evidence.

## Book-seeded coordinated initialization

[[w42-book-second-pass]] supplies a possible escape from the convention
bootstrap problem. A new signal has no unilateral value when the fixed partner
policy does not decode it. A book convention can instead be installed as the
same sparse policy overlay on both partnership seats: one side produces the
specified action in its stated context and the other performs the specified
belief update or response. Single-agent search can then value deviations
within an already coordinated codebook.

This is **coordinated initialization**, not wholesale imitation. The book is a
trustworthy hypothesis source, not an optimal full policy. A learned fallback
must supply action probabilities outside the convention states, and the
overlay and fallback must form one policy with one policy-matched continuation
value.

### Mine interactions, not oracle-disapproved rules

Double-dummy value can measure the direct tactical component of a book action,
but it cannot classify every oracle-endorsed action as “mere technique.” A move
can be tactically best and also make later partner behavior better. That is the
common case where useful information is a byproduct of good play.

The cleanest convention signature is still strong: the exact-world oracle
dislikes the sender's sacrifice, a blind imperfect-information partnership
also loses from it, and a decoding partner makes the joint policy win. It is a
sufficient example of convention value, not its definition.

Each mined convention therefore needs a sender-by-reader factorial rather than
a binary oracle verdict:

| sender overlay | partner reader | opponent reader | quantity exposed |
|---|---|---|---|
| off | off | off | ordinary-policy baseline |
| on | off | off | direct technique value or sender sacrifice |
| on | on | off | partner decoding increment |
| on | on | on | net value after opponent information leakage |

The full design also keeps the corresponding `sender=off` reader controls so
the partner and opponent effects are interactions, not merely stronger policy
substitutions. With team marks denoted by `V(sender, partner, opponents)`:

- direct action effect is `V(1,0,0) - V(0,0,0)`;
- partner convention value with opponent reading off is
  `[V(1,1,0) - V(0,1,0)] - [V(1,0,0) - V(0,0,0)]`;
- opponent leakage with partner reading on is
  `[V(1,1,1) - V(0,1,1)] - [V(1,1,0) - V(0,1,0)]`;
- net convention value is the full paired marks change, reported with those
  components rather than substituted for them.

Agreement with [[expected-q-value|E[Q]]] identifies a direct tactical
component. It neither proves nor removes the decoding component. The wall is
convention-shaped only to the extent that sender-by-reader interactions
survive paired full-match marks and disappear under the matching ablation.

### The book is an opponent-policy hypothesis, not a universal prior

A book convention is public enough to seed an opponent model as well as a
partner codebook. That creates extra action-derived inference unavailable to a
cooperative-only SPARTA formulation: opponent actions can be weighted by the
likelihood that a book-following policy would choose them from each candidate
hand.

Actual players need not follow the book. Opponent filtering therefore requires
a latent policy type, not unconditional book likelihoods:

`P(action | world) = sum_z P(action | world, policy_type=z) P(policy_type=z)`

The controlled partner can have a known book-overlay type. Each opponent's
book-likeness must be inferred from public history among book-like, learned
baseline, and other policy types. This preserves a second effect that a
partner-only test would miss: the same legibility that helps a partner may help
both opponents more.

### Initial mining surface

The book already supplies several distinct convention shapes:

- top-unplayed-trump as common knowledge that unlocks partner donation;
- donate-highest and absence-of-donation as two-sided inference;
- keep-priority and dump-to-inform choices in 84 defense;
- bids as hand-shape, role, and score-conditioned messages;
- Plunge/Splash as an explicit bounded signaling channel in a separate rule
  variant, not evidence about ordinary straight-42 frequency.

Each extraction must record the public preconditions, sender alternatives,
receiver update, intended response, opponent-readable consequence, and the
policy population for which the interpretation is claimed. Technique, natural
legibility, explicit partnership convention, deception, and variant-specific
communication remain separate labels.

### Open question

Can a sparse [[w42-book-second-pass|Winning 42]] convention overlay plus a
learned fallback become a complete, calibrated, policy-matched blueprint whose
partner-decoding gain remains positive after opponent decoding and full-match
evaluation?

The book now supplies a concrete coordinated-initialization mechanism and an
attribution design. Whether that mechanism is dense enough, complete enough,
or valuable enough to attack [[the-wall]] remains open. No book-seeded build,
experiment selection, or architecture selection follows from preserving it.

## Status

Convention-aware blueprint search is a serious candidate design with strong
structural fit to action-derived inference and partner response. It is also
unbuilt, unmeasured in Texas 42, and missing a C0-strength policy-matched
blueprint. It remains in the competing-explanation ledger until evidence earns
a build under [[partnership-research-gates]].

## Links

[[partnership-wall-research]] [[partnership-value]]
[[partnership-research-gates]] [[champion]] [[expected-q-value]]
[[strategy-fusion]] [[w42-book-second-pass]] [[w42-jud-v1]]
[[world-sampler-mrv-audit]] [[partnership-decision-record-v1]]
