---
title: Search Literature Transfer Map
kind: topic
first_seen: b28fb55a
last_updated: b28fb55a
status: active
---

Maps the published imperfect-information-search literature onto this project's
frontier, which is not "find a better generic algorithm" but: turn [[forge]]'s
exact per-world counterfactuals into one legal, decentralized,
information-set-consistent continuation policy, while public actions change
what partners and opponents believe and therefore do
([[partnership-wall-research]], [[strategy-fusion]]). No primary-source system
was found that combines this project's four assets in one full-game stack: an
exact per-world oracle, policy-conditioned realized values ([[jud]]), four
information-honest seats, and measurable partnership conventions. The closest
pieces exist separately, mapped below.

Source: an external deep-research synthesis reviewed 2026-07-13. Citations
marked `(verify)` await primary-source confirmation.

## Transfers directly

### Policy-based inference in trick-taking games (Skat)

Rebstock et al.'s policy-based inference (verify) uses a model of how players
choose actions to reweight states inside the acting information set, and
improved both inference quality and the playing strength of an existing
determinized Skat system. This maps one-for-one onto the seam this project has
built but never closed ([[pi-opp-head]] exists; nothing updates belief from
it):

`log b_t+1(w) = log b_t(w) + τ · log π_θ(a_t | I_actor(w), role, score, auction, policy type)`

followed by normalization, likelihood flooring or a uniform mixture, ESS
monitoring, resampling, and exact-world rejuvenation via the repaired
[[world-sampler-mrv-audit|uniform-completion sampler]]. The [[auction-decoder]]
is this transfer instantiated on the auction first.

The project's own record adds the mandatory caveat the paper does not carry:
better inference is not automatically better play. A belief improvement was
marks-neutral when the consumer could not react to it
([[belief-propagation-gap]], [[champion-design-review]]). The likelihood model
and the decision consumer therefore gate separately
([[partnership-research-gates]]).

## Transfers as shape, not as guarantee

### SPARTA-style single-agent search

SPARTA ([Lerer et al., 2020](https://arxiv.org/abs/1912.02318)) improves one
acting agent while every other agent is fixed to a commonly known blueprint —
a legal way to evaluate actions under imperfect information without granting
simulated actors hidden state. This is the search shape
[[convention-aware-blueprint-search]] adopts and the **Semantics 1** backup
rule in [[belief-weighted-jud-mcts]]. Its policy-improvement guarantee is for
fully cooperative games; Texas 42 keeps the partner on the blueprint, models
opponents as inferred policies, and lets paired full-match marks carry the
claim instead of the theorem.

### Learned Belief Search

Learned Belief Search (Hu et al., verify) adds an approximate learned belief
and a public/private policy architecture to the same cooperative setting. It
fits the project's particle and policy-head assets, with the identical
cooperative-only caveat.

## Bounded comparators, not architectures

### Alpha-mu and EPIMC

Alpha-mu (Cazenave & Ventos, verify) attacks [[strategy-fusion]] and
non-locality directly by backing up *vectors* of outcomes over possible worlds
instead of resolving each world independently; EPIMC (verify) postpones
perfect-information resolution and reports its largest gains where strategy
fusion matters most — consistent with the game-property analysis of
[Long, Sturtevant & Buro, 2010](https://ojs.aaai.org/index.php/AAAI/article/view/7562)
([[pimc]]). Neither is a drop-in: alpha-mu assumes a perfect-information
opposing side (bridge declarer play) and neither supplies partner conventions,
actor-policy likelihoods, or decentralized team search. Correct use here:
bounded comparators on late-hand microgames where few worlds remain, to expose
residual strategy fusion in whatever approximate search the project builds.

## The formal repair for two-versus-two

Texas 42 is not two-player zero-sum. Merging a partnership into one player
gives that player both private hands — illegal. Treating four seats as
independent players loses the team's ex-ante correlation and every 2p0s
convergence property.

The adversarial-team literature supplies the proper construction: a
**team-public-information coordinator** sees only information common to the
team and chooses a *prescription* `γ_j: I_j → Δ(A_j)` — an action for every
private state the acting teammate might hold — which is payoff-equivalent to
the decentralized team game (TMECor line;
[Farina et al., 2021](https://www.mit.edu/~gfarina/2021/tmecor-correlation-icml21/)).
Solving the corresponding equilibria is NP-hard and the equivalent
perfect-recall belief game may be exponentially larger; TB-DAG representations
(Zhang, Farina & Sandholm, verify) improve the representation without removing
the hardness. Correct use here: a legality specification for team search
semantics (**Semantics 2** in [[belief-weighted-jud-mcts]]), a solver for
bounded auction or late-hand microgames, and a verifier for approximate search
policies — not the next whole-game build.

## Does not transfer

- **ReBeL** ([Brown et al., 2020](https://arxiv.org/abs/2007.13544)): the
  convergence guarantee is explicitly two-player zero-sum. Its lasting lesson
  does transfer: values attach to public belief states, not hidden states, and
  a scalar hidden-state value is insufficient once search changes future
  strategies.
- **Depth-limited solving** (Brown & Sandholm, verify): imperfect-information
  states have no context-free values, so robust search needs multiple
  continuation strategies or richer leaf objects. Relevant only when a search
  jointly optimizes future strategies or performs resolving; a fixed-blueprint
  single-agent consumer legitimately uses a scalar policy-conditioned
  `V_realized` ([[jud]]).
- **Ordinary ISMCTS by label**: re-determinizing ISMCTS (verify) exists
  precisely because naive ISMCTS leaks hidden information into simulated
  player models. Node identity must be actor-relative —
  `N = (public history, acting seat, I_acting seat)`. Public-history-only keys
  over-merge; complete-world keys leak and retain strategy fusion; keys pinned
  to the root player's information fail once another seat acts
  ([[belief-weighted-jud-mcts]]).
- **Whole-game CFR over four seats**: representable only through the
  team-public-information construction above; four independent CFR players do
  not represent ex-ante correlated partnership strategies.
- **Double-dummy bidding evaluation**: bridge bidding systems often price
  contracts double-dummy; this project's own fixed-point experiment rejects
  that target — the converged belief-conditioned bidder over-bid via
  double-dummy optimism and lost on marks
  ([[w42-champion-selfplay-fixed-point]]). Double-dummy data remains a
  feature, upper bound, or auxiliary — never the bidder's target.

## Consequences already registered

- [[belief-weighted-jud-mcts]] carries the two legal backup semantics and the
  actor-relative node-identity rule.
- [[auction-decoder]] carries the policy-based-inference transfer with a
  latent policy-type mixture.
- [[research-lane-selection]] holds back whole-game CFR/ReBeL, centralized
  team players, and public-history-only ISMCTS, and reserves bounded
  TB-DAG/CFR solving for microgame verification after a simpler consumer
  demonstrates a residual inconsistency.

## Links

[[partnership-wall-research]] [[research-lane-selection]]
[[belief-weighted-jud-mcts]] [[convention-aware-blueprint-search]]
[[auction-decoder]] [[strategy-fusion]] [[pimc]] [[lamir1-ceiling]]
[[w42-champion-selfplay-fixed-point]] [[jud]] [[expected-q-value]]
