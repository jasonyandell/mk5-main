---
title: Research night 2026-10-12 — Jason's note: optimality gap and repairs (verbatim)
kind: source
first_seen: 2026-10-12
last_updated: 2026-10-12
status: complete
---

# Walt: Finite Stabilization, the Optimality Gap, and Repairs

*Discussion from October 10, 2026.*

Buddy, **we have a specific diagnosis, a useful bound, and a route to a stronger algorithm.**

The precise result from today is **finite stabilization without a general optimality guarantee**. We have not measured how far the actual 42 player is from optimal. What we have established is that the mathematical construction can settle while leaving achievable win probability on the table.

The two main gaps are:

- **Belief consistency:** the moves a policy chooses convey information, and the beliefs used afterward may fail to reflect that information.
- **Partnership coordination:** even with correct beliefs, changing one player at a time can miss improvements that require both partners to change together.

The first has a direct repair within Walt’s existing planning structure. The second requires enlarging what we optimize. And if we want an actual **finite-time optimality theorem**, we can get one by retaining and comparing complete policies, with exact response searches providing a certificate.

## 1. What the stabilized Walt actually guarantees

Our idealized operator is

$$
Q_i^F(\beta_i(I),a)
=
\max_{\rho_i\in\Pi_i^a(I)}
\mathbb E_{\beta_i(I),\,\rho_i,\,F_{-i}}[u_i],
$$

where $\rho_i$ is a complete legal continuation policy: a plan that can respond to whatever that player subsequently observes.

Walt selects the maximizing current action. At stabilization,

$$
P^\infty=\mathcal R_\beta[P^\infty].
$$

That means each player’s current action begins an optimal continuation **under the belief and other-player policies supplied to that calculation**.

There are two consistency requirements hiding in that sentence:

1. Does the belief correctly describe how the observed situation arose?
2. When this player reaches a later decision, will its new calculation actually follow an equally good continuation of the earlier plan?

A collection of locally optimized decisions can fail both requirements if each decision starts from an independently specified belief.

“Fixed beliefs” here does **not** mean ignoring the transcript altogether. The mapping $I\mapsto\beta_i(I)$ can remove impossible deals after tiles are played or someone shows a void. The missing part is whether that mapping accounts for **why a player with a particular hand would choose the observed move**.

### The earlier counterexample identifies that defect exactly

In today’s example, the stabilized sender plays

$$
H\longmapsto A,\qquad L\longmapsto B.
$$

Those moves reveal the hidden type. But the receiver continues to assign probability $1/2$ to each type after either message.

Consequently:

| Observed move | Frozen belief | Actual belief under the sender’s policy | Frozen choice | Correct choice |
|---|---|---|---|---|
| $A$ | $P(H)=1/2$ | $P(H)=1$ | $X$ | $Y$ |
| $B$ | $P(H)=1/2$ | $P(H)=0$ | $Y$ | $X$ |

That is the difference between the stabilized make probability $3/4$ and the attainable probability $1$.

The sender’s behavior has already supplied the information needed to improve. **The receiver’s belief rule prevents that information from changing its decision.**

### How large can this gap be?

We can strengthen the counterexample: **there is no nontrivial universal quality bound from stabilization alone.** The loss can approach the entire range from zero to one, even in a game with only two decisions.

Here is a family that makes this precise.

Nature chooses a hidden type $\theta\in\{1,\ldots,m\}$, with

$$
P(\theta=1)=\frac{2}{m+1},
\qquad
P(\theta=j)=\frac{1}{m+1}\quad(j>1).
$$

The sender sees $\theta$ and chooses a public message $a$. The receiver sees $a$ and guesses $b$. Their common success probability is

$$
u(\theta,a,b)
=
\alpha\,\mathbf1\{a=\theta\}
+
(1-\alpha)\,\mathbf1\{b=\theta\}.
$$

This can be an ordinary binary make/fail game: a final coin flip determines which success condition is checked.

Freeze the receiver’s beliefs at the initial prior after every message.

The receiver then strictly prefers guessing $1$ everywhere. The sender strictly prefers sending $a=\theta$, because truthful reporting earns the $\alpha$ component. These policies are reached immediately from the uniform starting policy and remain unchanged.

Their value is

$$
V_{\mathrm{stable}}
=
\alpha+(1-\alpha)\frac{2}{m+1}.
$$

A receiver that uses the message and guesses $b=a$ achieves $1$. Thus

$$
\boxed{
V^*-V_{\mathrm{stable}}
=
(1-\alpha)\left(1-\frac{2}{m+1}\right).
}
$$

For $m=100$ and $\alpha=0.01$, the stabilized player makes about **2.96%**, while the repaired player makes **100%**.

As $m$ grows and $\alpha$ shrinks, the gap approaches one. Every message occurs, the frozen beliefs have full support, and every selected action is strictly preferred. This does not depend on ties or unreachable situations.

That is a worst-case result about the mathematical class, not an estimate for 42. Its implication is clear: **the number of levels needed to settle does not bound the quality of the settled policy.**

## 2. The smallest repair: make the beliefs and the plan agree

For a fixed model of the other players, the repair is to weight each possible world by its likelihood of producing the observations.

In a card or domino game with public actions, the posterior has the form

$$
\boxed{
\beta_i^F(w\mid I)
\propto
\mu(w)\,
\mathbf1\{w\text{ is compatible with }I\}
\prod_{\substack{t<|h|\\j_t\ne i}}
F_{j_t}\!\left(a_t\mid I_{j_t}(w,h_t)\right).
}
$$

Here:

- $\mu(w)$ is the initial deal probability.
- Compatibility removes worlds contradicted by the observations.
- $F_j(a\mid I_j)$ is the probability that modeled player $j$ chooses action $a$ from its information.

The practical distinction is:

> “That player could legally have made this move with that hand” and “that player would probably choose this move with that hand” provide different information.

Under perfect recall, the player’s own past action probabilities cancel within its information set. Using chance and the other players’ reach probabilities also lets us evaluate situations reached by changing our own earlier actions. This is the same reach-weight construction underlying counterfactual values in extensive-form game solving.[^cfr]

One important Walt-specific detail: **$\operatorname{pmake}(a)$ is not the probability of choosing $a$.** It is the success probability conditional on choosing it. For deterministic Walt, action likelihood comes from its argmax and tie rule. For sampled Walt, it comes from the distribution of choices induced by its sampling.

### Two ways to make the repair concrete

**First: return and follow the whole optimized plan.**

The mathematical maximization already produces some maximizing $\rho_i^*$. Instead of retaining only the first action and its value, retain the contingent policy and follow its branches as observations arrive.

Against the fixed modeled opponents, exact execution of that plan realizes its calculated value.

**Second: replan using beliefs conditioned from the same model.**

At each later decision, update the earlier world distribution using the observations’ likelihoods under the modeled opponents. Then recompute.

With exact full-continuation optimization and perfect recall, those later calculations are consistent with the earlier optimization. They can choose a different tied continuation without losing value.

This gives a positive theorem:

> **With the other players fixed, coherent beliefs and exact search over all legal own continuation policies produce an actual best response.**

This is a finite computation in a finite game. It directly closes the gap between “the plan we valued” and “the policy we actually execute.”

It does not require a different objective. It requires the probability model to remain consistent across the decisions contributing to that objective.

## 3. We can bound the remaining loss

The worst case is unrestricted, but we can say something useful when the belief error is controlled.

Hold the other players fixed. Let

$$
\nu_i(I)
$$

be the coherent posterior at information set $I$, and let $\beta_i(I)$ be the belief Walt uses. Define

$$
\delta(I)
=
\operatorname{TV}\bigl(\beta_i(I),\nu_i(I)\bigr),
$$

where total variation is half the sum of the absolute probability differences.

For a make/fail objective, every continuation policy has a value between zero and one. Therefore changing the belief by total variation $\delta$ changes the value of any fixed continuation by at most $\delta$. Taking the maximum over the same legal continuation policies preserves that bound:

$$
\boxed{
|Q_\beta(I,a)-Q_\nu(I,a)|\le\delta(I).
}
$$

If Walt chooses $\widehat a$ using $\beta$, then

$$
\boxed{
\max_a Q_\nu(I,a)-Q_\nu(I,\widehat a)
\le 2\delta(I).
}
$$

One $\delta$ allows the selected action to have been overestimated; the other allows a better action to have been underestimated.

### This gives an action certificate

Suppose Walt’s estimated best action beats its estimated runner-up by $\Delta(I)$. If

$$
\boxed{\Delta(I)>2\delta(I),}
$$

then correcting the belief cannot change the best action.

That is useful because we do not need perfect beliefs everywhere. We need enough accuracy to preserve the decisions that matter.

In the earlier $H/L$ example, the receiver switches to the correct action after $A$ when $P(H\mid A)>2/3$, and after $B$ when $P(H\mid B)<1/3$. The frozen probability $1/2$ misses both thresholds. Exact certainty is unnecessary; crossing those boundaries suffices.

### The decision losses add up along actual play

Define the one-seat improvement gap at a profile $P$ as

$$
g_i(P)
=
\max_{\rho_i}V_i(\rho_i,P_{-i})-V_i(P).
$$

This asks how much that seat could improve if everyone else kept playing the same policy.

For the exact stabilized operator, the local optimality deficits telescope through the player’s decision tree. We obtain

$$
\boxed{
g_i(P)
\le
\min\left\{
1,\;
2\,\mathbb E_P
\left[
\sum_{\text{own decisions reached }I}\delta(I)
\right]
\right\}.
}
$$

If there are at most $d$ remaining decisions for that player and every belief error is at most $\bar\delta$, the simpler bound is

$$
g_i(P)\le\min\{1,2d\bar\delta\}.
$$

With a certified action-value approximation error $\eta(I)$, replace $\delta(I)$ by $\delta(I)+\eta(I)$. A sampling certificate must account for the search over candidate policies; an ordinary confidence interval around a selected winner is not automatically sufficient.

### Coherence gives us an even cleaner result

Suppose the entire policy is optimized under one approximate initial prior $\widehat\mu$, and all later beliefs are obtained by conditioning that same model. With correct dynamics and fixed opponents,

$$
\boxed{
V_\mu^*-V_\mu^{\pi_{\widehat\mu}^*}
\le 2\,\operatorname{TV}(\mu,\widehat\mu).
}
$$

There is **no horizon factor**.

The proof is short: every complete policy’s value differs between the two priors by at most their total variation. Apply that once to the truly optimal policy and once to the approximate-model optimum.

This is an encouraging result. We can tolerate an imperfect starting model and still obtain a global guarantee. Repeatedly replacing beliefs with incompatible local distributions introduces an additional problem.

These bounds are derived guarantees under the stated assumptions. They become numerical estimates for Walt only after we measure or bound the relevant errors.

## 4. Correct inference changes the convergence problem—and leaves partnership coordination

There is a price for making beliefs depend on the modeled policies: **today’s finite-depth stabilization proof no longer transfers automatically.**

Previously, a decision depended on modeled decisions strictly later in the game. That gave us the decreasing rank used in the proof.

With policy-dependent beliefs:

- Earlier choices determine what later observations mean.
- Later reactions determine which earlier choices are attractive.

That introduces feedback. The circular dependence between beliefs and strategies is a central issue in dynamic games with asymmetric information.[^dynamic-games]

We can demonstrate the problem with hidden matching pennies. The first player chooses $H/T$; the second chooses without seeing that choice. The first wants a match, and the second wants a mismatch.

With simultaneous best-response updates and a fixed tie rule, the pure profiles can cycle:

$$
(H,H)\to(H,T)\to(T,T)\to(T,H)\to(H,H).
$$

The beliefs are accurately describing the previous modeled policy. The iteration still cycles.

So the repaired claims are:

- **Frozen opponents:** exact coherent planning gives an optimal response.
- **Everyone changing together:** convergence needs its own argument.

### Even a converged solution can leave a team improvement

For 42, “no player can improve alone” is weaker than “our partnership cannot improve.”

Consider two partners choosing $S$ or $R$, with a common success probability:

| Your choice | Partner chooses $S$ | Partner chooses $R$ |
|---|---:|---:|
| $S$ | $\varepsilon$ | $0$ |
| $R$ | $0$ | $1$ |

At $(S,S)$, changing alone makes things worse. Changing together improves success from $\varepsilon$ to one.

There is no belief error here. The limitation is the unit of optimization.

That small table illustrates a bad fixed point; uniform initialization would choose the good outcome in that particular table. But uniform initialization does not rescue a general guarantee either. We can construct a version with many actions where:

- Action $0$ gives $c$ when both choose it.
- Choosing $0$ against any other action gives $c/2$.
- Both choosing action $1$ gives one.
- All other pairs give zero.

With $m+1$ actions and $c=3/(m+1)$, action $0$ is the unique best response to uniform play and remains the unique best response to itself. Both players settle there, while the coordinated optimum is one. As the number of actions grows, $c$ approaches zero.

Thus **even correct beliefs, strict best responses, and a uniform start can leave an arbitrarily large partnership gap.**

For our game, the appropriate partnership deviation is a pair of policies,

$$
p_A=(\pi_0,\pi_2),
$$

where each policy still sees only its own seat’s information. Joint optimization does not authorize either seat to inspect the partner’s hand. The literature on team-maxmin play treats this distinction explicitly, including whether partners may coordinate their policy choices before play.[^team-maxmin]

## 5. Where your complementary-plan idea fits

Your idea about one plan succeeding where another fails is relevant. The key question is **what lets us choose between them**.

Suppose plans $A$ and $B$ succeed on complementary hidden worlds.

Choosing between them now, without more information, gives at most

$$
\max\{\mathbb E U_A,\mathbb E U_B\}.
$$

Randomizing between them cannot exceed that value against the same fixed opponents. It produces a weighted average.

Selecting whichever one succeeds separately in each hidden world would give

$$
\mathbb E[\max(U_A,U_B)].
$$

That is generally a stronger, unavailable-information operation.

But suppose we can first observe something $Z$, before the plans need to diverge. Then a legal contingent policy can obtain

$$
\boxed{
\mathbb E_Z
\left[
\max\{
\mathbb E[U_A\mid Z],
\mathbb E[U_B\mid Z]
\}
\right].
}
$$

That is the mathematical form of useful hedging through future choice.

For example, if each plan wins on half the worlds and their winning halves are disjoint, each individually makes with probability $1/2$. A preliminary move that reveals which half we occupy can let us select the right continuation and make with probability one.

**Our exact unified $Q$ already includes that possibility**, because it maximizes over full observation-contingent policies. The “max over actions, sum over observations, then optimize within the resulting observation groups” structure captures it.

So if move $C$ preserves a valuable future choice, its exact $Q$ should already include that value. If a practical search misses it, the missing piece lies in the searched continuations, sampled observations, beliefs, or opponent model.

This suggests a precise interpretation of your intuition:

> Find moves whose future observations separate the circumstances in which different continuations succeed.

Two qualifications matter. The observation must arrive before the differing action, and it must concern something the player can actually observe. Complementary successes caused only by different rollout random seeds do not supply usable information.

## 6. A finite-time route to an actual optimum

We can recover a finite-time optimality theorem, but we need to specify the target.

| Target | What must be optimized |
|---|---|
| Best play for one seat against fixed others | One complete legal seat policy |
| Best partnership play against fixed opponents | A pair of legal partner policies |
| Best worst-case play against an opposing team | A minimax solution over the permitted team strategy classes |

The first is repaired by coherent planning. The second can, in principle, be solved by enumerating legal partner-policy pairs.

For the third, a particularly natural extension is **double oracle**: maintain a collection of policies, solve their competition, and search for policies that can improve on that solution. Exact algorithms using this framework exist for finite zero-sum games with imperfect information.[^double-oracle]

### Walt can generate candidates for a growing policy collection

Let $\mathcal P$ and $\mathcal Q$ be the finite sets of legal complete plans for the two sides, and let

$$
U(p,q)=P(\text{declaring team makes}\mid p,q).
$$

For partnerships, a plan is a tuple of seat-local policies. A distribution over such tuples permits coordination of plan choice before private information arrives; if our intended player allows only independent partner randomization, that restriction defines a different optimization problem.

Maintain subsets

$$
\mathcal P_t\subseteq\mathcal P,
\qquad
\mathcal Q_t\subseteq\mathcal Q.
$$

Then repeat:

1. Solve the game restricted to the currently retained plans, obtaining mixed strategies $x_t,y_t$.
2. Find a best complete response to $y_t$.
3. Find a best opposing complete response to $x_t$.
4. Add those responses to the collection.

Walt can seed the collection and propose new plans. The responses must be evaluated as legal policies across information sets.

There are also extensive-form versions that expand available responses within information states, rather than mixing only whole policies at the root. That is the direction explored by XDO; its stated guarantees apply to its two-player setting.[^xdo]

### The important addition is a certificate

Define

$$
H_t=\max_{p\in\mathcal P}U(p,y_t),
\qquad
L_t=\min_{q\in\mathcal Q}U(x_t,q).
$$

Then the full-game minimax value satisfies

$$
\boxed{L_t\le v^*\le H_t.}
$$

The current maximizing strategy guarantees at least $L_t$. The current minimizing strategy permits at most $H_t$. Therefore

$$
\boxed{\varepsilon_t=H_t-L_t}
$$

measures the unresolved optimality gap.

This gives a stopping condition with strategic meaning.

### Why exact search terminates in finite time

Let $v_t=U(x_t,y_t)$ be the restricted-game value.

If $H_t-L_t>0$, then either

$$
H_t>v_t
\quad\text{or}\quad
L_t<v_t.
$$

At least one improving best response must therefore be absent from the current collection. A plan already included could not improve on the restricted equilibrium in that way.

Every nonterminal iteration adds a new plan. There are finitely many plans. Consequently, with exact response searches, exact evaluation, and exact restricted-game solving, the process terminates at zero gap.

That is a finite-time optimality proof. Its bound can be enormous: the number of legal complete policies grows with all the information sets at which choices could be made. It is a foundation for a solver and a small-game benchmark, not a promise of practical trick-one runtime.

### The limitation we must preserve

An approximate Walt response alone does not certify those bounds.

A found maximizing response can underestimate $H_t$. A found minimizing response can overestimate $L_t$. Both errors can make the apparent gap falsely small.

With certified response errors $\epsilon_A,\epsilon_B$, we can instead use

$$
\widetilde L_t-\epsilon_B
\le v^*
\le
\widetilde H_t+\epsilon_A.
$$

Without certificates, the procedure is still a useful empirical improvement process.

Regret minimization supplies another established route: in finite two-player zero-sum games with perfect recall, CFR bounds regret and the error of an appropriately averaged strategy, with the basic bound decreasing as $1/\sqrt T$. That offers finite computation for a requested approximation tolerance, rather than a claim that the current pure policy settles exactly after a fixed number of levels. The partnership information constraints still require care.[^cfr]

## 7. What I would do next for Walt

**I would first measure the gap between the continuation Walt values and the continuation it executes, with the other players held fixed.**

At a starting information set $I_0$, calculate

$$
V_{\mathrm{plan}}
=
\max_a Q_i^F(\beta_i(I_0),a),
$$

then evaluate the actual local replanning policy against the same modeled opponents:

$$
V_{\mathrm{execute}}
=
\mathbb E_{\beta_i(I_0),\,\mathcal R_\beta[F]_i,\,F_{-i}}[u_i].
$$

The difference

$$
\boxed{
\Delta_{\mathrm{execution}}
=
V_{\mathrm{plan}}-V_{\mathrm{execute}}
}
$$

is nonnegative for the exact specification. It directly exposes an optimized continuation that the assembled policy fails to realize.

On tiny games where we can compute everything exactly, compare:

| Comparison | What it establishes |
|---|---|
| Extracted optimal plan versus current replanning | Whether later decisions lose value the initial search already found |
| Current beliefs versus coherent likelihood updates | How much that loss comes from belief inconsistency |
| Best one-seat response versus best partnership response | How much requires coordinated policy changes |
| Restricted policy solution versus exact response bounds | The remaining distance to the specified minimax solution |

I checked the arithmetic of the counterexample families above; these would make useful exact cases for that experiment.

For larger 42 positions, a stronger response that finds an improvement supplies a **lower bound** on the exploitable gap. Failure to find an improvement does not establish an upper bound. Also, our full-deal oracle is not automatically the right upper bound: revealing the deal to both teams strengthens both sides and changes the information problem.

My first repair would therefore be **coherent world weights and coherent continuation execution**, followed by joint partnership response searches on small cases. That tells us which loss matters before we expand the algorithm.

The strongest conclusion we can carry forward is this:

**Walt already has the machinery to value legal contingent plans. We can make that valuation correspond to the policy actually played. To go further toward optimal team play, we need to compare complete coordinated policies—and measure the improvements still available against them.**

## References

[^cfr]: Martin Zinkevich, Michael Johanson, Michael Bowling, and Carmelo Piccione. [*Regret Minimization in Games with Incomplete Information*](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf). NIPS 2007. Source for counterfactual reach weights, regret bounds, and averaged-strategy equilibrium guarantees.

[^dynamic-games]: Yi Ouyang, Hamidreza Tavafoghi, and Demosthenis Teneketzis. [*Dynamic Games with Asymmetric Information: Common Information Based Perfect Bayesian Equilibria and Sequential Decomposition*](https://arxiv.org/abs/1510.07001). Source for the circular dependence between strategies and beliefs in dynamic games with asymmetric information.

[^team-maxmin]: Youzhi Zhang, Bo An, and Jakub Černý. [*Computing Ex Ante Coordinated Team-Maxmin Equilibria in Zero-Sum Multiplayer Extensive-Form Games*](https://arxiv.org/abs/2009.12629). Source for team strategy coordination and the distinction between team and individual optimization.

[^double-oracle]: Branislav Bošanský, Christopher Kiekintveld, Viliam Lisý, and Michal Pěchouček. [*An Exact Double-Oracle Algorithm for Zero-Sum Extensive-Form Games with Imperfect Information*](https://www.cs.utep.edu/kiekintveld/papers/2014/bklp_double_oracle.htm). Journal of Artificial Intelligence Research, 2014. Source for restricted-game solving, best-response expansion, and exact convergence guarantees.

[^xdo]: Stephen McAleer, John Lanier, Kevin Wang, Pierre Baldi, and Roy Fox. [*XDO: A Double Oracle Algorithm for Extensive-Form Games*](https://arxiv.org/abs/2103.06426). NeurIPS 2021. Source for expanding responses within information states.
