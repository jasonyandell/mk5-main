<!-- SPDX-FileCopyrightText: 2026 Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

# Walt: one policy across hidden worlds

Jason Yandell · 18 October 2026 · CC0

**Setting.** A finite hand-phase game with perfect recall; focal seat $i$ knows its own hand and public history $I=(x,h)$. Hidden deal $w$ has belief $\beta$. A fixed behavioral field $F$ specifies lawful policies for the other seats, including the partner, using only each actor's information. Terminal utility $u$ lies in $[L,U]$; write $M=U-L$. Make probability, points and match equity are distinct choices of $u$.

## 1. The response and its recursion

Let $\Pi^a(I)$ be the finite set of deterministic, information-consistent continuation policies beginning with legal action $a$. Randomization cannot improve a one-seat response to a fixed field. Define

$$Q^F_\beta(a)=\max_{\rho\in\Pi^a(I)}\mathbb E_{\beta,\rho,F}[u],\qquad
R_\beta[F](I)=\operatorname*{argmax}^{\prec}_a Q^F_\beta(a),$$

with a fixed tie order $\prec$. One policy is chosen across indistinguishable worlds; its outcomes are averaged before optimization.

For deterministic $F$, use unnormalized world masses $m_w$. A cell $C$ contains worlds sharing the focal observation history. With $C_o=\{w\in C:F(w,h)=o\}$,

$$V(C,h)=\begin{cases}
\sum_{w\in C}m_wu(w,h)&\text{terminal},\\
\max_a V(C,ha)&\text{focal turn},\\
\sum_oV(C_o,ho)&\text{field turn}.
\end{cases}$$

**Bellman theorem (derived).** At a focal decision, $V(C,ha)/m(C)=Q^F_{\beta_C}(a)$, where $m(C)=\sum_{w\in C}m_w>0$ and $\beta_C(w)=m_w/m(C)$. Induct on remaining plays: the focal seat must choose one action throughout a cell; distinct public observations permit independently chosen continuations. There is no opponent MIN because the field specifies opponent behavior. Stochastic fields require likelihood-weighted splitting, or integration over random tapes as well as deals. See the [theory](https://github.com/jasonyandell/mk5-main/blob/28edece47b5cf11d57f32095d835b82d7e169bf0/wiki/sources/research-night-2026-10-12/10-theory.md#L154-L198) and [cell recursion](https://github.com/jasonyandell/mk5-main/blob/28edece47b5cf11d57f32095d835b82d7e169bf0/scenario/planner.py#L827-L887).

**Geometry (derived).** Each fixed policy has an outcome vector $r_\rho(w)$, averaging field randomness. On nonnegative measures,

$$V(\mu)=\max_\rho\langle\mu,r_\rho\rangle,\qquad
V(\mu_1+\mu_2)\le V(\mu_1)+V(\mu_2).$$

Thus $V$ is piecewise linear, convex and positively homogeneous. The inequality follows by moving the maximum inside the sum. Its right side permits observing the split label before choosing: a focal information relaxation. Giving every player hidden information also strengthens opponents, so averaged double-dummy minimax is not a general upper bound against a fixed field.

## 2. Belief coherence and execution

For prior $\mu$, the posterior consistent with $F$ is

$$\beta_I^F(w)\propto\mu(w)\mathbf1_{\{w\sim I\}}
\prod_{t:j_t\ne i}F_{j_t}(a_t\mid I_{j_t}(w,h_{<t})).$$

Here $w\sim I$ means physical compatibility. Perfect recall makes the focal seat's own past action likelihoods constant within its information set, so they cancel. Deterministic field likelihoods are 0/1; zero total likelihood leaves the posterior undefined. A fallback changes the model.

**Coherent best response (derived).** Full-horizon, unrestricted recursion with $\beta=\beta_I^F$ is an exact one-seat best response to $F$. At positive-probability histories, if every subsequent invocation conditions the same prior and field, replanning realizes its root value: conditioning in stages gives the same posterior, and backward induction gives the same optimal continuation value. Enumerating physical worlds alone does not supply these behavioral likelihoods.

For total variation $\delta=\operatorname{TV}(\beta,\nu)$, fixed-policy expectations differ by at most $M\delta$. Taking maxima preserves that bound, hence

$$|Q_\beta(a)-Q_\nu(a)|\le M\delta,\qquad
\max_aQ_\nu(a)-Q_\nu(R_\beta)\le2M\delta.$$

A unique action gap under $\beta$ greater than $2M\delta$ therefore survives. The second inequality follows by inserting the two values under $\beta$ and using its maximizing action. Optimizing a complete policy under one approximate prior gives the same $2M\delta$ regret bound without a horizon factor; independently replacing later beliefs does not. The finite weighted algebra is [mechanized](https://github.com/jasonyandell/mk5-main/blob/28edece47b5cf11d57f32095d835b82d7e169bf0/scenario/lean/Stab/Coherence.lean); the full game-level execution statement is derived.

## 3. Why a response ladder can settle

Let a coordinate $c$ be an acting seat's information state. Suppose a single response operator $R$ reads an input policy profile only at dependencies $D(c)$, and a natural height $d(c)$ strictly decreases along every dependency. Beliefs, tie rules and all other settings—including any samples and random tapes—are fixed across applications. Heterogeneous rung budgets do not meet this premise.

**Stabilization theorem (mechanized).** For any starting profiles $P,Q$,

$$R^nP(c)=R^mQ(c)\quad\text{whenever }n,m\ge d(c)+1.$$

Dependency-free coordinates settle after one application; induction settles each next height after one more. A uniform height bound gives a unique fixed point. Repeated identical root actions alone certify nothing: an unpropagated change can arrive later. See [the proof](https://github.com/jasonyandell/mk5-main/blob/28edece47b5cf11d57f32095d835b82d7e169bf0/scenario/lean/Stab/Ladder.lean).

For ordinary four-seat 42, let $n_j$ be remaining hand sizes and

$$r=\sum_{j=0}^3\max(n_j-1,0).$$

Every play weakly decreases $r$; a genuine choice requires at least two held tiles and decreases it strictly. Drop forced coordinates. **If** dependencies occur only after the coordinate's own play, positive rank gives height $r-1$. Opening rank is 24: the opening response settles after **24 applications**, or $P^{23}$ under $P^{-1}=P_{\rm base}$. This [mechanized corollary](https://github.com/jasonyandell/mk5-main/blob/28edece47b5cf11d57f32095d835b82d7e169bf0/scenario/lean/Stab/Rank.lean#L94-L175) assumes locality and later-only dependencies; their refinement to the concrete planner is not mechanized.

Recomputing beliefs from each preceding policy profile can make later beliefs depend on earlier decisions, destroying the height argument. An [abstract four-cycle](https://github.com/jasonyandell/mk5-main/blob/28edece47b5cf11d57f32095d835b82d7e169bf0/scenario/lean/Stab/Cycle.lean) demonstrates that the missing hypothesis matters; it is not a full-42 cycle.

## 4. Samples, certificates, and objectives

An exact solve on a finite sample optimizes that sample. Its maximizing continuation has optimistic expected training value; seven root actions do not mean only seven policies were selected among. A thin cell can overfit without structurally violating information consistency. Floors that hand thin cells to a fixed policy restrict the policy class.

**Statistical certificate (derived).** Draw independent iid samples $S_1,S_2$ from the same stated belief; either integrate field randomness exactly or sample iid joint deal-and-tape scenarios. For each of $A$ root actions, let $\widehat Q_1(a)$ be the empirical optimum over $\Pi^a(I)$; sampling tapes does not expose them to the policy. For each $a$, construct from $S_1$ alone a fixed policy in $\Pi^a(I)$, completed at unseen information states, and score it on $S_2$ as $\widetilde V_2(a)$. For $0<\eta<1/2$ and positive sample sizes, set $\epsilon_N=M\sqrt{\log(A/\eta)/(2N)}$. Simultaneously with probability at least $1-2\eta$,

$$\widetilde V_2(a)-\epsilon_{|S_2|}\le Q(a)
\le\widehat Q_1(a)+\epsilon_{|S_1|}.$$

For the upper side, apply Hoeffding to each fixed population-optimal policy, which the empirical optimum dominates. For the lower side, condition on $S_1$ and apply Hoeffding to its fixed policies on independent $S_2$. Union-bound over actions on both sides. A lower bound exceeding every competitor's upper bound certifies the action. This [certificate](https://github.com/jasonyandell/mk5-main/blob/28edece47b5cf11d57f32095d835b82d7e169bf0/wiki/sources/research-night-2026-10-12/10-theory.md#41-sampling-a-certificate-that-survives-the-selection-effect) is mathematics, not a claim that the runtime emits it.

Finally, one-seat response fixes the partner; partnership optimization chooses two lawful policies jointly; team maxmin additionally quantifies over adversarial opponents. A unilateral guarantee is stronger still:

$$\exists\pi_i\ \forall w\in W\ \forall\pi_{-i}^{\rm legal}:\ \text{success}.$$

Here even the partner is universally quantified. Neither $\forall w\exists\pi_w$ nor probability one on sampled worlds proves this statement. Settling a response ladder, optimizing its stated model, and guaranteeing success are three different claims.

**Match objective (derived).** For one-mark alternatives sharing continuation contexts and a fixed future law, let $E_+,E_-$ be match-win probabilities after winning or losing the mark. An alternative with mark-win probability $p$ has equity $E_-+p(E_+-E_-)$. Thus maximizing $p$ maximizes equity if $E_+\ge E_-$, strictly preserving ordering when $E_+>E_-$. This [affine identity](https://github.com/jasonyandell/mk5-main/blob/28edece47b5cf11d57f32095d835b82d7e169bf0/wiki/topics/match-equity.md#one-mark-calls-are-score-blind) does not make $p$ independent of score-sensitive play.
