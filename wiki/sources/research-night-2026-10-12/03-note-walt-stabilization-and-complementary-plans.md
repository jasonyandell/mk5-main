# Walt: Stabilization Rates and Complementary Plans

**Research note — October 12, 2026**

This note develops two questions about Walt: how quickly its response ladder stabilizes, and how complementary ways of winning can be combined into a stronger plan. It collects the mathematical analysis from the discussion into a standalone account.

The results concern the stated mathematical construction. They are not a proof-assistant verification, a verification of the implementation, or a measured playing-strength result. Numerical examples are illustrative; none is asserted to be an actual Texas 42 position.

## 1. The planning construction and the scope of stabilization

Let $i$ be the deciding player, $I$ its information history, $\beta_i(I)$ its belief, and $F_{-i}$ its model of the other players. A legal contingent plan $\rho_i$ specifies the player's present and future choices using only information available when each choice is made.

For a legal first action $a$, define

$$
Q_i^F(\beta_i(I),a)
=
\max_{\rho_i\in\Pi_i^a(I)}
\mathbb E_{\beta_i(I),\rho_i,F_{-i}}[u_i(Z)].
$$

The response operator returns a maximizing first action, using a fixed tie rule:

$$
\mathcal R_\beta[F]_i(I)
=
\operatorname*{argmax}^{\prec_i}_{a\in A_i(I)}
Q_i^F(\beta_i(I),a).
$$

The response ladder starts from uniform legal play:

$$
P^{-1}=\mathsf U,
\qquad
P^k=\mathcal R_\beta[P^{k-1}].
$$

The deciding player's own future choices are optimized within its plan. Other players' future decisions are supplied by the lower-rung model.

The stabilization results below assume:

- A finite game and finite action and observation sets.
- A belief constructor $\beta_i(I)$ held fixed as the ladder level changes.
- Exact full-horizon evaluation and a fixed deterministic tie rule.
- A remaining-game rank that strictly decreases along future player-model dependencies.
- Information histories that retain the observations and own actions needed for lawful contingent choices.

In particular, changing a lower-rung policy does not itself change the starting belief at a fixed information history. If beliefs are instead recomputed from the previous rung's probabilities of earlier actions, the following dependency argument requires reconsideration.

Stabilization identifies a fixed point of this local response construction. It does not by itself establish equilibrium play, correct beliefs about actual opponents, or agreement between a planned continuation and later invocations that rebuild their beliefs.

## 2. How quickly the ladder stabilizes

### 2.1 The dependency-height bound

A response at one information state depends on certain later other-player decisions. Those decisions may depend on still later decisions. Eventually the chain reaches a decision that requires no relevant lower-policy input.

Write a coordinate as $c=(i,I)$. Place an edge $c\to d$ whenever the response at $c$ can depend on the lower-rung policy at the later coordinate $d$. Forced-action coordinates have constant outputs and can be omitted from relevant dependencies.

Let $h(c)$ be the longest number of edges in a dependency chain starting at $c$:

$$
h(c)=
\begin{cases}
0, & \text{if there are no relevant dependencies},\\
1+\max_{d:c\to d}h(d), & \text{otherwise}.
\end{cases}
$$

Then

$$
\boxed{
P^k(c)=P^\star(c)
\quad\text{for every }k\ge h(c).
}
$$

Because $P^{-1}=\mathsf U$, this is $h(c)+1$ applications of the response operator.

**Proof.** A coordinate with no relevant lower-policy dependencies is settled after one response application. Once every coordinate on which a response depends has settled, the next response at that coordinate receives the same continuation problem at every higher level. Exact evaluation and the fixed tie rule therefore return the same action. Induct on dependency height. The same induction shows that the eventual result is independent of the base policy. $\square$

This describes finite backward propagation through dependencies. It does not assert a geometric error rate or monotone improvement in playing strength.

The statement concerns returned actions. A forced action can already be settled while its estimated outcome value still depends on other players' unsettled future choices.

### 2.2 A tighter bound for Texas 42

Consider the tile-play phase with contract and trump fixed. Let $n_j$ be the number of remaining tiles held by each active seat $j$, and define

$$
R=\sum_j\max(n_j-1,0).
$$

This counts the maximum number of remaining potentially nonforced plays. Each seat's final tile requires no choice.

A genuine decision has at least two legal actions, so its acting seat holds at least two tiles. Playing the selected tile reduces $R$ by one. Any later genuine modeled decision therefore has smaller $R$.

For a current decision with multiple legal actions,

$$
h(i,I)\le R-1,
\qquad
\boxed{P^{R-1}_i(I)=P^\star_i(I).}
$$

At the start of an ordinary four-seat, seven-tile hand,

$$
R=4(7-1)=24.
$$

Thus **24 exact response applications**, reaching $P^{23}$, give a conservative theoretical ceiling for this fixed-belief formulation. Follow-suit restrictions, forced moves, and an already determined make/set result can reduce the relevant depth.

This is a bound on reasoning levels. It is not a practical runtime estimate, a claim that the bound is tight for 42, or evidence that real positions require anything close to 24 levels. A forced current move is already determined.

### 2.3 The general bound is sharp, and plateaus do not certify stability

There are finite games in which the root chooses the same action through arbitrarily many successive levels, then changes at the last relevant level.

Construct a perfect-information game with $n$ decisions, alternating between two teammates. At decision $t$, the acting player can:

- Stop for shared payoff $s_t=1-3^{-(n-t)}$.
- Continue to the next decision; continuing at the final decision pays $1$.

The payoffs lie in $[0,1]$ and can be implemented as terminal win probabilities.

Start from uniform Stop/Continue. The final decision strictly chooses Continue. At any earlier decision, the next actor is the other player. Against the uniform base, even allowing optimal own continuation, Continue is worth at most

$$
\frac{s_{t+1}+1}{2}<s_t.
$$

Consequently, $P^0$ continues only at the final decision. Inductively, $P^k$ continues at exactly the final $k+1$ decisions. The root first changes at $P^{n-1}$. All decisive comparisons are strict.

Therefore the general dependency-height bound cannot be uniformly improved. Several unchanged root choices are not proof that the next level cannot change the move.

Increasing depth also need not monotonically improve actual play. Settling to a fixed point is a different property from improving at every intermediate rung.

### 2.4 An early-stopping certificate

Suppose we can bound how much the still-unsettled lower-policy decisions could change each current action's value:

$$
|Q_k(a)-Q_\star(a)|\le\varepsilon_a,
$$

where $Q_\star$ evaluates against the eventual lower-policy field.

Then $a$ is already the eventual unique choice if

$$
\boxed{
Q_k(a)-\varepsilon_a
>
\max_{b\ne a}\bigl(Q_k(b)+\varepsilon_b\bigr).
}
$$

One way to obtain a bound is to control the probability of reaching an unsettled modeled decision before the outcome is determined. Couple the same hidden situation and randomness under the provisional and eventual models. Their trajectories agree until the first unsettled modeled decision. A make indicator can differ by at most one afterward.

For each action $a$, an upper bound on that encounter probability, taken over **all legal own continuations beginning with $a$**, therefore bounds the possible change in its optimized value. Checking only the currently selected rollout is insufficient.

The set of settled coordinates must itself be justified by a structural argument or a valid certificate; adjacent-rung agreement alone does not establish it.

This provides a concrete stopping principle: stop when the remaining possible effects are too small to change the winner. The bound may be unhelpfully large early in a hand. Near a decided contract or a largely forced ending, it can be useful.

### 2.5 Three different meanings of speed

| Quantity | What changes | What the result establishes |
|---|---|---|
| Ladder depth $k$ | Reasoning attributed to modeled players | Finite action stabilization under the fixed-belief dependency assumptions |
| Sample count $N$ | Accuracy of a chosen rung's empirical evaluation | A separate statistical convergence question |
| Elapsed runtime | Actual work performed by search and nested invocations | Depends on tree size, repeated calculations, and implementation |

The dependency bound alone supplies no affordable runtime guarantee. Fresh samples can also produce changing answers even when the corresponding exact policy has settled.

## 3. Preserve where each plan succeeds

For a plan $\rho$ and sampled situation $\omega$, retain

$$
r_\rho(\omega)
=
\text{the outcome of plan }\rho\text{ in situation }\omega.
$$

When all relevant randomness is included in the scenario, this can be a binary make/no-make value. If some randomness is integrated out, it can be a conditional value in $[0,1]$.

For nonnegative weights normalized so that $\sum_\omega w_\omega=1$, the usual scalar is

$$
V(\rho)=\sum_\omega w_\omega r_\rho(\omega).
$$

Two plans with the same scalar can have very different outcome vectors. One may succeed precisely where another fails.

**The scalar is sufficient to rank complete plans at the current belief. The vectors preserve information that can be used to construct a better plan.**

This connects to the existing value geometry:

$$
V(\mu)=\max_\rho\langle\mu,r_\rho\rangle.
$$

A plan with a lower value under today's belief can become the best continuation after a later observation changes the relevant belief. A low current average alone does not establish that the plan is useless on future observable branches.

## 4. An example where complementary continuations make C best

Imagine 100 equally weighted sampled situations. Playing root action $C$ leads to a later observable cue, red or blue, before another choice. Each cue occurs in 50 situations. At that later choice, there are two candidate continuations, $p$ and $q$.

| Continuation | Make among red situations | Make among blue situations | Overall make |
|---|---:|---:|---:|
| $C$, followed by $p$ | 100% | 90% | 95% |
| $C$, followed by $q$ | 90% | 100% | 95% |
| $D$, with its continuation | 96% | 96% | 96% |

For the $D$ row, red and blue label the same initial scenario classes that would produce those cues under $C$; $D$ need not reveal a cue.

Comparing the two retained $C$ continuations separately favors $D$. But after playing $C$, use

$$
\text{red}\mapsto p,
\qquad
\text{blue}\mapsto q.
$$

The resulting legal plan makes in every situation:

$$
\boxed{Q(C)=100\%>Q(D)=96\%.}
$$

The observation need not reveal the whole hidden situation. It only needs to distinguish which continuation is preferable. In 42, the relevant observation could be a played domino, failure to follow suit, who wins a trick, or a longer observed sequence.

Contingent planning accounts for both what an action accomplishes and what it lets the player learn before subsequent choices. This is part of the established planning structure for partially observable systems. [1]

### What makes the combination playable?

The candidate continuations share a prefix through the switching point. The player then uses an observation available at that point to choose between them.

If red and blue remain hidden, any blind mixture of $p$ and $q$ still makes only 95%:

$$
\lambda(0.95)+(1-\lambda)(0.95)=0.95.
$$

Complementary failures identify an opportunity. An executable choice rule determines how much of that opportunity can be collected.

The player cannot play $C$, observe what happens, and retroactively choose root move $D$. Switching must occur between continuations that remain available after a common prefix.

## 5. The exact formula for repairing failures

Let $\pi$ be the current continuation and $\nu$ an alternative. They start after the same prefix. Let $E$ be a condition the player can recognize at the switching decision.

Use $\nu$ on $E$ and keep $\pi$ otherwise. With binary outcomes, the change in make probability is

$$
\begin{aligned}
\Delta
&=\mathbb E\!\left[\mathbf1_E(r_\nu-r_\pi)\right]\\
&=\Pr(E,\ \pi\text{ loses},\ \nu\text{ wins})\\
&\quad-\Pr(E,\ \pi\text{ wins},\ \nu\text{ loses}).
\end{aligned}
$$

Thus the gain is

$$
\boxed{\text{recovered losses}-\text{sacrificed wins}.}
$$

If $O$ represents everything observable at the switching point, the best selector between the two continuations gains

$$
\boxed{
\mathbb E\!\left[
\left(\mathbb E[r_\nu-r_\pi\mid O]\right)_+
\right],
}
$$

where $x_+=\max(x,0)$. For each observation, switch exactly when the alternative has positive conditional advantage.

This suggests a concrete use for scheme/fix: search for observable conditions that identify profitable switches. Paired outcome differences supply labels for that search. The condition itself must use only information available to the acting player at the patched decision.

## 6. Maximizing pmake globally across sampled situations

For a fixed weighted sample, the global objective is

$$
\boxed{
\max_{\pi\in\Pi_{\mathrm{legal}}}
\sum_\omega w_\omega
\mathbf1\{\pi\text{ makes in }\omega\}.
}
$$

The legality constraint couples sampled situations: whenever two situations produce the same information history for the player, the policy must make the same choice. Once observations distinguish them, the policy may branch.

Three quantities are useful to distinguish:

| Quantity | Meaning |
|---|---|
| Best single candidate | Value of the best currently retained complete candidate |
| Best observable combination | Value achievable by selecting continuations using available information |
| Best candidate in each hidden situation | An optimistic bound that may require unavailable information |

For candidates sharing a prefix and a fixed later observation $O$,

$$
\max_\rho\mathbb E[r_\rho]
\;\le\;
\mathbb E\!\left[\max_\rho\mathbb E[r_\rho\mid O]\right]
\;\le\;
\mathbb E[\max_\rho r_\rho].
$$

The middle expression describes the achievable combination. The last expression allows an oracle to select a candidate separately in every hidden situation. It bounds what can be obtained from that candidate menu, but is not automatically an upper bound over every as-yet-undiscovered legal policy.

Keeping both quantities can expose complementary successes and show how much of their potential is currently executable.

### Where this fits in exact Walt

The exact specification already maximizes over all legal contingent plans. An exhaustive implementation of that definition would already value $C$ at 100% in the example above.

Therefore, if $D$ truly has the highest exact, fully optimized pmake, failure correlations cannot create an improvement outside that optimum. The practical opportunity lies in how finite computation discovers and represents plans:

- Which continuations survive pruning?
- Are their outcome vectors retained?
- Can strengths discovered on different scenarios be combined lawfully?
- Does the search directly seek repairs to the current plan?

Those choices can change which root action practical Walt finds.

### Expected points and make probability

If an EV denotes expected points rather than expected binary make, a larger mean score can have a smaller chance of reaching the contract. A bimodal score distribution can therefore matter directly to the objective.

If EV already means the expectation of a binary make indicator, it is pmake. In that case the useful extra structure lies in conditional success patterns and executable continuations.

Against a fixed model, a blind random choice between complete policies satisfies

$$
V\bigl(\lambda\pi+(1-\lambda)\nu\bigr)
=\lambda V(\pi)+(1-\lambda)V(\nu)
\le\max\{V(\pi),V(\nu)\}.
$$

Here the mixture denotes choosing which complete policy to execute, independently of hidden information. Improving worst-case performance against an adapting opponent is a different objective from maximizing current-hand pmake against a fixed model.

## 7. A concrete portfolio-and-repair algorithm

For each root action, retain a small portfolio of legal continuations and their outcomes on shared scenarios.

1. Find failures of the current composed plan.
2. Generate alternatives that recover some of them.
3. Find observable branches where an alternative improves the conditional result.
4. Compose a revised legal policy, counting both recovered losses and newly sacrificed wins.
5. Evaluate the entire revised policy on fresh scenarios.

A candidate can earn a place despite a lower standalone average because it handles a branch the other candidates handle poorly.

For a fixed observation partition, let $v_{o,\rho}$ be the conditional make probability of continuation $\rho$ after observation $o$, and let $p_o$ be that observation's probability under the common prefix. The achievable value of portfolio $\mathcal P$ is

$$
F(\mathcal P)
=\sum_o p_o\max_{\rho\in\mathcal P}v_{o,\rho}.
$$

Adding continuation $\nu$ contributes

$$
\boxed{
F(\mathcal P\cup\{\nu\})-F(\mathcal P)
=
\sum_o p_o
\left[
v_{o,\nu}-\max_{\rho\in\mathcal P}v_{o,\rho}
\right]_+.
}
$$

This scores additional usable coverage. For a fixed partition and fixed candidate evaluations, adding candidates never decreases the objective. A candidate's marginal contribution decreases as the portfolio becomes richer, because the existing maximum in each observation cell can only increase.

These claims apply to the fixed-prefix, fixed-partition menu objective. They are not a general theorem about unrestricted policy synthesis in which new choices change the observation process.

### Two kinds of stabilization

- The ladder settles as its modeled future decisions settle.
- Portfolio search improves as additional legal ways to handle observations are discovered.

Exhaustively enumerating all legal policies eventually reaches the empirical optimum in finite time. A heuristic repair search needs a separate completeness argument. When its present repair generator finds nothing useful, that is only a local stopping condition.

Empirical improvement also needs evaluation beyond the scenarios used to choose and compose the repair. The procedure can otherwise select a policy that fits its construction sample unusually well.

### Joint teammate policies

Optimizing several teammates' policies together expands the problem beyond one player's response against fixed teammate models:

$$
\max_{\pi_{\mathrm{team}}}
\sum_\omega w_\omega
\mathbf1\{
(\pi_{\mathrm{team}},F_{\mathrm{opponents}})
\text{ makes in }\omega
\}.
$$

Each teammate's policy must still use only that teammate's own information. Joint planning does not permit pooling hidden hands during execution. Coordinated changes can improve team performance even when no individual policy change would. Exact finite-horizon joint-policy planning is studied in decentralized partially observable models. [2]

## 8. The next experiment: the success matrix

The proposed experiment retains per-scenario outcomes for candidate continuations, measures complementary wins, and then measures how much complementarity survives the observable-switching constraint.

For each root action:

1. Fix a shared set of weighted scenarios and a declared other-player model.
2. Record a scenario-by-continuation success matrix for the candidate plans.
3. Record the information available at each proposed switching point.
4. Compare the best retained candidate, the best legal observable composition, and the per-scenario candidate oracle.
5. Evaluate the composed policy on fresh scenarios from the same declared target distribution.
6. Check whether the resulting root-action ranking changes, particularly whether a slightly worse-looking $C$ overtakes $D$.

Candidate continuations must share the prefix preceding a switch. Every shared information history must reuse the same policy choice across scenarios. If multiple starting hands are combined, their sampling weights should define the intended root distribution, with each episode's terminal result counted once.

The experiment asks a specific question: **Is practical Walt missing useful combinations of the plans it already finds, and can those combinations be selected using the information available during play?**

## References

The Walt-specific bounds and constructions above are derivations for the stated response operator. These references supply related foundations for contingent and joint-policy planning; they are not citations for a claim that the present Walt implementation has been verified.

1. Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. *Planning and acting in partially observable stochastic domains*. Artificial Intelligence 101 (1998), 99–134. [Author-hosted paper](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf).
2. *Mixed Integer Linear Programming For Exact Finite-Horizon Planning In Decentralized POMDPs* (2007). [arXiv:0707.2506](https://arxiv.org/abs/0707.2506).

