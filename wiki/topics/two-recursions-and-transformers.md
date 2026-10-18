---
title: "The two recursions: belief, planning, attention and autoregression"
kind: topic
first_seen: 2026-10-17
last_updated: 2026-10-17
status: active
---

<!-- SPDX-FileCopyrightText: 2026 Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Finding and source scope

The two recursions in the Texas 42 mathematics are **forward information updating**
and **backward continuation evaluation**. Attention and autoregression give a learned
model ways to use history and predict continuations. They do not, by definition,
implement those two game recursions. A trained network could learn parts of either;
establishing that requires more than identifying its architecture.

The exact source heading is “Two orthogonal recursions,” §6.9 of the
[unified information geometry draft][geometry-recursions]. It advances
`(K, e, β, R)`—mechanical/support kernel, evidence, belief and relational answers—after an
observation, and separately constructs continuation objects backward from terminal
outcomes. `walt/FACTOR-BELIEF.md` also records Jason's framing as two recursions
running in opposite directions. These sources are pinned to Texas-42 commit
`9d6a5a2e12958183d49d0cd95c8617c4ef7a12e4`.

The geometry document is a mathematical draft, with explicit proof-status labels.
The predictive-algebra document contains conditional prose proofs and an experimental
implementation program. Its filename says `v0.6`, while its internal status says
handoff `v0.1`; citations below identify both the file and the section. No claim here
turns these drafts into a freshly checked Lean theorem or an opening-game solver.

The neural comparison refers to the actual [[lem-nanogpt-foundations]] observer at
mk5 commit `118b4904aabdfd5e10a227812f62507dfd27da1c`, not a generic future Lem.

## Forward: what the observed play changes

Let `ξ` denote a possible hidden situation: the deal and, when modeled, persistent
player types and other necessary latent state. Let `β_t(ξ)` be its current probability.
After seeing another seat play `o` with positive probability under the model,
a static-latent version of the update is

$$
\beta_{t+1}(\xi)
=\frac{\beta_t(\xi)L(o\mid\xi,h_t)}
        {\sum_\eta\beta_t(\eta)L(o\mid\eta,h_t)}.
$$

The likelihood is zero when the play is illegal in that situation. Otherwise it
comes from a specified behavioral model, evaluated using **the acting seat's own
information** in that situation. Thus a play can change both the possible holdings
and the relative plausibility of different players or styles. Rules alone supply
possibility, not discretionary choice probabilities.

When the latent coordinate itself changes, §6.3 also sums over predecessor states
and applies the state-transition kernel. Support, remaining tiles, context and
relational roles advance along with the probabilities. This is more than looking
up an earlier token. A player's own choice is treated as an intervention under
the draft's stated independence conditions, not automatically as new evidence
about its own hidden-deal uncertainty. [Geometry §§6.2–6.3][geometry-update].

“Learning” in this forward recursion means updating information during play.
Network training is a different operation: changing parameter values across examples.
A fixed trained network can respond differently to a longer history without changing
any weights.

## Backward: what each available choice would lead to

Planning starts with terminal outcomes and a declared objective, then evaluates
earlier choices through their possible continuations. For a fixed behavioral field,
a schematic finite-horizon belief-state recursion with terminal utility is

$$
V(B)=\max_{a\in A(B)}\sum_o
 P(o\mid B,a)\,V\bigl(T(B,a,o)\bigr),
\qquad V(B_{\rm terminal})=U(B_{\rm terminal}).
$$

Here `B` retains the focal player's information, including relevant history and
belief; `T` includes the forward update after an action and observation. The displayed
form groups intervening modeled play into the next observation. At an individual
other-seat node, its prescribed behavior is averaged rather than chosen by the
focal player. Immediate rewards can be included when the objective requires them.
The terminal `U(B)` is the belief expectation of the specified terminal payoff;
when the result is public, it reduces to that known payoff.

The critical order is **one choice for the entire information state, then averaging
over what is hidden**. A plan can choose differently after observations distinguish
the cases. It cannot choose differently merely because the analyst named different
hidden deals or player types. This is the [[strategy-fusion-fence]].

The geometry draft first retains policy-indexed terminal outcome laws, before
applying a particular utility. The scalar Bellman form above is a specialization,
not the full valuation-independent object. See [geometry §§7.1–7.2][geometry-policy]
and [predictive algebra §§8.1–8.4][predictive-policy].

## Where they meet: knowledge times consequences

For a **fixed lawful continuation policy** `ρ`, the join is

$$
J_\rho(\beta)=\sum_\xi\beta(\xi)V_\rho(\xi).
$$

`β` supplies what is plausible now; `V_ρ` supplies what that same information-honest
plan would achieve in each situation. Writing the latter as a world-indexed function
does not allow it to choose a different plan for each world. Optimization is over
whole lawful policies after the expectation, not independently inside its sum.

The predictive-algebra proposal makes a stronger connection to learned representations.
If a basis `F` spans the selected controlled continuation tests and closes under their
transitions, the full belief can be replaced for those tests by predictive moments:

$$
\psi(\beta)=\mathbb E_\beta[F(\xi)],\qquad
J_\rho(\beta)=\psi(\beta)c_\rho.
$$

With a constant-one basis coordinate and certified action/observation-specific
matrices `M`, the forward update for a positive-probability observation is
`ψ′ = (ψ M)/(ψ M)₀`. Hidden emissions with the same player observation are summed
in `M`; coordinate zero gives that observation's probability.
Backward construction produces the lawful policy vectors `c_ρ`; the root action
value maximizes `ψ c_ρ` among policies beginning with that action. Hidden events
sharing one player observation must share their continuation choice.

This is an exact factorization **under the stated spanning and closure conditions**;
it compresses only when the resulting rank is smaller than the concrete dimension.
The finite rank need not be small, and a short vector alone is no certificate.
Its relevance to Lem is a research question: can training learn a compact state
that retains the distinctions needed for the chosen futures? Success at one-step
prediction under the recording policies does not establish sufficiency for every
alternative action, continuation policy or objective.
[Predictive algebra §§7–8][predictive-filter].

The later [information-geometry rulings][rank-ruling] sharpen this limitation. Under
their stated finite Straight-42 setup, if complete attributed observations distinguish
every latent world, an exact predictive closure containing the constant function has
rank equal to the number of worlds. Scalar expected-value closure is a different
object and has no general small-rank guarantee either. This is a barrier to claiming
universal **exact linear** compression, not a theorem against useful approximate,
nonlinear or restricted-purpose neural representations. The rulings retain their
exploratory source tier.

## What attention actually computes

At token position `t`, one causal attention head computes

$$
\alpha_{tj}=\operatorname{softmax}_{j\le t}
 \left(\frac{q_t\cdot k_j}{\sqrt{d_k}}\right),\qquad
z_t=\sum_{j\le t}\alpha_{tj}v_j.
$$

The current representation supplies a query; earlier/current representations supply
keys and value vectors. Learned matching scores determine how their information is
combined. Later positions are masked out. In this observer, two such blocks, their
MLPs and residual connections transform the token sequence. The second block can
read representations already enriched by the first.

**The attention weights range over token positions. Belief weights range over hidden
situations.** Normalized weights in both formulas do not make them the same object.
An attention head may help retrieve an earlier lead or discard; the surrounding
network must learn what follows from that evidence. No Bayesian update, legal-support
intersection or utility maximization is imposed by the attention formula.
[Observer `Attention` and `Block`][observer-model]; [Vaswani et al., §§3.1–3.2][attention-paper].

## What autoregression actually computes

Autoregressive modeling factors a sequence into successive conditional predictions:

$$
p_\theta(x_{1:T}\mid C)
=\prod_{t=1}^{T}p_\theta(x_t\mid C,x_{<t}),
$$

where `C` is the initial visible context. In the observer, the next actor is supplied
as an additional public token at every step; it is not a predicted variable. The
conditional has the actual form `pθ(tile_t | visible prefix, actor_t)`. For a lawful
generated prefix, the rules determine the next actor. A generative table would have
to supply those actor/turn transitions as well as tiles.

This has a direct relative in the game mathematics: the likelihood of a history
under a candidate hidden situation is a product of the acting policies' conditional
action probabilities. [`Belief.lean::likelihoodFrom_append`][lean-likelihood] proves
the corresponding concatenation identity. The conditionals differ: that likelihood
is conditioned on a candidate hidden situation, while the observer predicts from
its visible prefix, implicitly averaging whatever remains unknown. No future actor
tokens are included in an earlier prediction.

Training minimizes the sum of `−log pθ(recorded tile | recorded prefix, actor)`.
Every prefix contains the actual earlier plays, called **teacher forcing**. The
causal mask permits training many positions together without letting a later play
inform an earlier prediction. Free-running generation would instead append a
sampled output before predicting again; the reported evaluations do not do that.
[Encoder and targets][observer-data]; [training loss][observer-training].

This sequence factorization extends probability through time. It has no terminal
objective, comparison of alternative policies, or backward value backup. Sampling a
future is consequently not the same calculation as deciding which future to pursue.
Backpropagation travels backward through the network to adjust prediction parameters;
that is also distinct from backward game planning.

## The precise connection, and its limits

An exact observer under a specified population would predict another seat's action by

$$
p(a\mid I_t)=\sum_\xi\beta_t(\xi)
 \pi_{s,\tau_s(\xi)}\bigl(a\mid I_s(\xi,h_t)\bigr).
$$

`I_t` is the viewer's information; `I_s` is the actor's information in a candidate
hidden situation, and `τ_s(ξ)` selects its modeled behavioral type. A next-action
network can learn this predictive mixture directly,
without exposing its factors as explicit probabilities over deals and players.
It may also imitate choices whose consequences were already evaluated by its teacher.
Neither possibility demonstrates that its internal mechanism runs an exact planner.

The observer's final score has another suggestive shape:
`logit(a) = LayerNorm(h_t) · E_a`. A context vector pairs with a tile vector.
In the predictive algebra, a belief-moment vector pairs with a continuation-policy
vector. **The domains and certificates differ:** a learned tile embedding has not
been shown to be a lawful policy vector, and a logit is not a game value. The
resemblance is a useful hypothesis for [[recursive-generalization]] and
[[clean-neurons-42]], not a result of the existing training runs.

| Mathematical object | Possible role in a learned model | Established here |
|---|---|---|
| Forward information update | History-sensitive representation of holdings and behavior | Architecture can read the legal prefix; no exact belief semantics established |
| Predictive observation law | Next-tile probabilities conditioned on that prefix | Trained and measured in [[lem-observer-data-scale-2026-10-17]] |
| Backward lawful planning | Learned evaluation, distilled decisions, or a separate search using a model | No equivalence or planning mechanism established by next-tile accuracy |

## A small arithmetic example

This is a schematic two-situation example, not a recorded 42 hand. Suppose hidden
possibilities `A` and `B` begin equally likely. An observed legal play has likelihood
`4/5` in `A` and `1/5` in `B`. Forward updating changes their probabilities to
`4/5` and `1/5`.

Suppose a lawful plan `X` succeeds only in `A`, and another lawful plan `Y` succeeds
only in `B`. Backward evaluation supplies these two success patterns. Pairing them
with the new belief values `X` at `4/5` and `Y` at `1/5`.

Choosing `X` in `A` and `Y` in `B` before any distinguishing observation would claim
success probability one. That is an unavailable information advantage. A learned
predictor can recognize the observed play and favor `X` in its output without that
alone establishing that it represented these beliefs and evaluated both plans.

## Implication for Nello and subsequent experiments

[[rules-of-42]] changes the continuation question for Nello: the bidder must avoid
every trick, the partner sits out, and the engine's doubles form an unpowered separate
suit. Earlier plays can remove a holding, reveal a void or alter which remaining tile
is dangerous. Forward information and backward consequences therefore remain separate
questions even when they concern the same tile.

The geometry sources above concern Straight 42. Their information/planning distinction
guides a Nello formulation; their game-specific results do not automatically transfer.
Neither existing observer corpus contains Nello. No Nello experiment is claimed here.

A useful future separation is: first test predictions on controlled histories that
carry different evidence; then test choices against independently computed lawful
continuation values for a small declared player family. Hold mechanical state and
rule support fixed where possible, so remembering legality is distinguishable from
reading behavior. These are proposed tests, not a registration or completed result.

## Other recursions with different meanings

- [[scenario-player]] writes `R_live[R_model[U]]`: a response ladder over modeled
  players. Its depth concerns which player models which other player; each response
  can itself involve forward belief and backward planning.
- [[belief-policy-value-algebra]] closes a policy-improvement loop: changing behavior
  changes action likelihoods, beliefs and the values of responses. This is distinct
  from processing one hand with a fixed behavioral field.
- Transformer depth composes representation updates. It is not automatically either
  the number of future tricks searched or the depth of reasoning about another mind.

## Sources

All local implementation claims are pinned to mk5 `118b4904`; all Texas-42 mathematical
claims to `9d6a5a2e`. Sources were inspected, not newly compiled or experimentally rerun.

The Lean source also separates the objects: [`Strategic.lean`][lean-strategic] defines
the Bayes filter and fixed-strategy belief recursion, proves its equality to latent
expectation, and derives the finite-strategy-class best-response equality. The
[proof plan][lean-plan] explicitly leaves the concrete Straight-42 instantiation of
that generic process as further work. This supports the abstract distinction without
certifying the observer, the predictive compression proposal, or a Nello solver.

- [Unified information geometry][geometry-recursions], §6.9; §§6.3, 7, and its opening
  claim-discipline statement distinguish information, control and proof status.
- [Factor-belief report, lines 617–626][factor-belief]: project provenance of the
  two-directions framing; its numerical horizon remains an exploratory corpus result.
- [Predictive algebra][predictive-filter], §§6–8: closure, filtering and lawful policy
  coordinates; these conditions are essential to the proposed bridge to representations.
- [Model-belief intake][model-intake]: augmented hidden state, behavioral likelihoods
  and persistent types. [[lem-nanogpt-foundations]] records its scoped proof boundary.
- [Observer model][observer-model], [data][observer-data], [training][observer-training].
- [Attention Is All You Need][attention-paper], §§3.1–3.2, for the original causal
  decoder and attention definitions.

[geometry-recursions]: https://github.com/jasonyandell/texas-42/blob/9d6a5a2e/walt/math/unified_information_geometry_v0.4.md#L1193-L1213
[geometry-update]: https://github.com/jasonyandell/texas-42/blob/9d6a5a2e/walt/math/unified_information_geometry_v0.4.md#L1040-L1073
[geometry-policy]: https://github.com/jasonyandell/texas-42/blob/9d6a5a2e/walt/math/unified_information_geometry_v0.4.md#L1217-L1246
[predictive-filter]: https://github.com/jasonyandell/texas-42/blob/9d6a5a2e/walt/math/predictive_algebra_v0.6.md#L576-L817
[predictive-policy]: https://github.com/jasonyandell/texas-42/blob/9d6a5a2e/walt/math/predictive_algebra_v0.6.md#L703-L817
[factor-belief]: https://github.com/jasonyandell/texas-42/blob/9d6a5a2e/walt/FACTOR-BELIEF.md#L617-L626
[model-intake]: https://github.com/jasonyandell/texas-42/blob/9d6a5a2e/walt/math/model_belief_base_player_v0.1_intake.md
[observer-model]: https://github.com/jasonyandell/mk5-main/blob/118b4904/lem/observer/model.py#L32-L100
[observer-data]: https://github.com/jasonyandell/mk5-main/blob/118b4904/lem/observer/data.py#L37-L126
[observer-training]: https://github.com/jasonyandell/mk5-main/blob/118b4904/lem/observer/train.py#L127-L144
[attention-paper]: https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
[rank-ruling]: https://github.com/jasonyandell/texas-42/blob/9d6a5a2e/wiki/walt-math-information-geometry.md#L24-L80
[lean-likelihood]: https://github.com/jasonyandell/texas-42/blob/9d6a5a2e/lean/Texas42/Belief.lean#L152-L181
[lean-strategic]: https://github.com/jasonyandell/texas-42/blob/9d6a5a2e/lean/Texas42/Strategic.lean#L106-L290
[lean-plan]: https://github.com/jasonyandell/texas-42/blob/9d6a5a2e/wiki/proof-assistant-plan.md#L65-L77
