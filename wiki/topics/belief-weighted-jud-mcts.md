---
title: Belief-Weighted Jud MCTS
kind: topic
first_seen: f6b691da
last_updated: f6b691da
status: active
---

Research position: **SURVIVING HYPOTHESIS.** Implementation status: IDEATED,
unbuilt.

Belief-weighted Jud MCTS combines three surviving project assets:

- belief particles concentrate search on plausible hidden worlds;
- MCTS allocates depth and simulations to consequential branches;
- [[jud]] supplies information-honest rollout behavior and a
  policy-conditioned `V_realized` leaf.

The idea is strongest when the tree represents information sets rather than
separate perfect-information worlds. In that form it becomes a concrete search
engine for [[convention-aware-blueprint-search]], not merely a deeper
determinization.

## Why this survives

[[w42-jud-v1]] supplies direct positive evidence for the core search/leaf
pairing. Its `JudSearch` consumer improved greedy Jud play from `-3.44` to
`-1.16` marks/game against `lens:ev`, a `+2.28` gain. The Jud leaf was useful
once search converted its sharp post-trick evaluations into a move ranking.

The next JudSearch result localizes the remaining question: doubling worlds
from 10 to 20 added only `+0.11`, below the registered band. Flat sampling
volume was not the binding constraint. MCTS changes two things that the worlds
sweep did not:

- it allocates simulations adaptively instead of spending equally across every
  root action and world;
- it can carry a continuation beyond the current trick.

The project has also shown bounded positive value from belief-sampled
aggregation: [[gus-qmean-router]] reduced held-out regret when a Q-mean second
opinion was used selectively. That result does not establish MCTS, but it keeps
belief-weighted candidate generation in the positive evidence ledger.

## The decisive fork — what identifies a tree node?

### Determinized Jud MCTS

The simpler version samples a world from the root belief, uses the complete
hidden deal as the simulation state, and belief-weights root action statistics
across worlds. Even when the rollout policy receives only its legal information,
separate hidden-world node identities permit different continuations.

This is deeper, adaptively allocated JudSearch. It may add tactical and
multi-trick value, but separate world trees may choose incompatible plans in
states the real actor cannot distinguish. Root weighting alone therefore
retains [[strategy-fusion]] and gives information-only actions no later
imperfect-information response to influence.

### Information-set Jud MCTS

The structural version shares action statistics across worlds whenever the
acting policy cannot distinguish them. Each simulated seat receives its own
hand and the public history, not the complete deal. Public simulated actions
update or reweight the beliefs used by later seats.

Node identity is actor-relative:

`N = (public history, acting seat, I_acting seat)`

Two worlds share statistics at a node only when the acting seat cannot
distinguish them. Seat 0 must choose one action across worlds differing only
in unseen hands; when seat 2 later acts, worlds with different seat-2 hands
are legitimately different information sets, while worlds with the same seat-2
hand and public history but different unseen hands must still share seat-2's
statistics. A node keyed only by public history over-merges; a node keyed by
the complete sampled world leaks and retains [[strategy-fusion]]; a node keyed
permanently to the root player's information fails once another seat acts.

The load-bearing rule is:

> A hidden world may determine what is true in a simulation; it may not grant
> the acting policy a different continuation unless that policy can observe
> the difference.

This gives search a channel for information-set value. A lead can change the
partner's posterior and later action inside the tree. A discard can reveal a
priority to both partner and opponents. A plan can persist because one action
choice is backed up across the worlds for which it must remain the same.

## Candidate stack

`belief particles -> information-set MCTS -> blueprint rollout policy -> Jud V_realized leaf`

### Belief particles

The root contains consistent hidden deals with explicit weights. Hard public
evidence removes worlds; bids and voluntary plays can reweight them through
policy likelihoods. [[world-sampler-mrv-audit]] supplies the repaired uniform
completion sampler for valid particle rejuvenation when effective sample size
falls.

### Tree policy

MCTS selection allocates compute among legal actions. Priors can come from a
learned policy; a [[w42-book-second-pass]] overlay can supply shared
convention behavior in its declared states. Partner and opponent seats act
from their own information sets under the rollout blueprint.

### Mid-tree inference

After each simulated public action, the later actor's particle weights change
according to the likelihood of that action under the acting policy. This is
the difference between “belief-weighted at the root” and “belief-aware through
the continuation.” The latter can price action-derived inference, signaling,
concealment, and counter-inference.

### Leaf and backup

Jud's `V_realized` values terminal or truncated nodes from the searching team's
perspective. Common-random-number particles keep root action comparisons
paired.

Backup semantics are load-bearing, and exactly two forms are legal
([[search-literature-transfer]]):

- **Semantics 1 — unilateral online search.** Only the currently acting seat
  optimizes. Every later seat — the partner included — is a stochastic node
  drawn from its information-honest blueprint policy. There is no max backup
  at partner nodes and no min backup at opponent nodes; both are
  policy/chance nodes. This is the SPARTA-shaped form
  [[convention-aware-blueprint-search]] adopts and the clean first
  implementation.
- **Semantics 2 — common-knowledge coordinated search.** Both teammates run
  the same public search procedure, and at a public team decision the search
  selects a prescription `γ: I_j → Δ(A_j)` — an action for every private
  information state the acting teammate could hold — using only information
  common to the team; the teammate executes the branch matching its actual
  hand. This is the team-public-information coordinator construction from the
  adversarial-team literature.

A search that simply maximizes separately at each sampled partner hand, using
information available to the root process, is neither of these. It is a
centralized controller with access to both hands, even when each individual
action superficially looks legal. This distinction decides whether an apparent
J3/J4 gain is a real decentralized strategy or information leakage.

[[w42-jud-v1]]'s policy-conditioned pricing law is load-bearing: the Jud leaf
prices the policy distribution that generated its corpus. When MCTS or a
blueprint overlay materially changes that continuation policy, the leaf must be
retrained or explicitly tested for transfer.

## Relationship to prior work

### JudSearch is the nearest built predecessor

JudSearch already combines hidden-world lifting, information-honest current-
trick responses, and a realized-value leaf. Belief-weighted Jud MCTS extends
its consumer from a flat one-trick action comparison to an adaptive tree.

### Zeb does not answer this combination

[[zeb]] built MCTS and self-play and showed that the shape can learn competent
hidden-information play. Its learned
value, training objective, node semantics, and evaluation are not the Jud
realized-value / explicit-belief / information-set combination here. Zeb is
supporting prior art for the engine, not a negative result for this design.

### LAMIR supplies a real warning and a useful discriminator

Every tested [[lamir1-ceiling|LAMIR-1]] look-ahead mode lost to direct
`pi_me`; distilled scalar leaf noise and rollout compounding flipped move
orderings. MCTS does not make that failure disappear. The countervailing
project evidence is that JudSearch gained `+2.28` with a different,
realized-value leaf and a post-trick boundary where that leaf is sharp.

The unresolved question is therefore specific: can adaptive search extend the
demonstrated JudSearch gain before leaf bias or rollout-policy drift dominates?

### Blueprint search supplies the semantics

[[convention-aware-blueprint-search]] defines why later seats react to public
actions: a shared policy supplies action likelihoods and partner responses.
Belief-weighted Jud MCTS supplies one way to allocate and back up that search.
The blueprint is the codebook; MCTS is the engine; Jud is the realized-value
leaf.

## Causal decomposition

The design contains four nested consumers that preserve the explanation for
any gain:

| arm | search behavior | question isolated |
|---|---|---|
| J0 | current JudSearch: consistent worlds, current trick | built predecessor |
| J1 | J0 plus explicit root belief weights | does belief weighting improve the existing consumer? |
| J2 | deeper MCTS on the same root particles, determinized world nodes | does adaptive depth/allocation add value? |
| J3 | J2 plus information-set action sharing and mid-tree belief updates | does information-set consistency add value beyond depth? |
| J4 | J3 plus the book/learned convention blueprint | does coordinated policy legibility add value? |

`J1 - J0` attributes root belief weighting. `J2 - J1` attributes depth and
branch allocation. `J3 - J2` attributes information-set node identity and
action-derived belief. `J4 - J3` attributes the installed convention layer.
After its isolated root-weighting comparison, every later arm uses the same
deals, root particles, compute budget, leaf generation, and marks-to-7 arena.

This decomposition also protects a useful partial result. Determinized MCTS may
improve tactics even if mid-tree inference is neutral; information-set MCTS may
improve play even if the first book conventions do not. The result remains
about the mechanism that moved.

Build-order constraint ([[research-lane-selection]]): J3's mid-tree updates
require calibrated action likelihoods and a coherent stochastic blueprint,
neither of which currently exists. Built before the [[auction-decoder]]-class
likelihood instrument passes held-out evaluation, J3 collapses operationally
into J2 — a deeper determinizer with a more elaborate name. J2 remains
independently eligible through its own gate. Root-belief determinized MCTS is
reported as a search improvement, never as a solution to the information
problem.

## Engineering requirements

- information-set keys that never expose another seat's private hand;
- particle ancestry, weights, rejuvenation, and effective-sample diagnostics
  at public-history nodes;
- calibrated action likelihoods for mid-tree belief updates;
- one coherent rollout blueprint across searcher, partner, opponent model, and
  leaf generation;
- team-aware selection and backup rules for two cooperating seats against two
  adversarial seats;
- transposition rules that merge only observations the acting policy truly
  cannot distinguish;
- a compute ledger separating more depth from merely more leaf evaluations.

## Research question

Can belief-weighted information-set MCTS extend JudSearch's demonstrated
`+2.28` search gain far enough to beat `lens:ev`, and can mid-tree belief updates
then produce additional marks for a demonstrated information-set or partnership
reason?

The question joins search depth, action-derived inference, plan persistence,
and blueprint coordination without conflating them. It remains a live candidate
alongside target quality, capacity, auction decoding, and distributional
utility in [[partnership-wall-research]].

## Links

[[jud]] [[w42-jud-v1]] [[convention-aware-blueprint-search]]
[[partnership-wall-research]] [[partnership-research-gates]]
[[world-sampler-mrv-audit]] [[gus-qmean-router]] [[lamir1-ceiling]]
[[alphazero-under-imperfect-information]] [[strategy-fusion]] [[champion]]
[[search-literature-transfer]] [[auction-decoder]] [[research-lane-selection]]
