---
title: Research night 2026-10-12 — theory agent (verbatim)
kind: source
first_seen: 2026-10-12
last_updated: 2026-10-12
status: complete
---

# Does Zeb find the optimal move in finite time? (theory role)

Research night 2026-10-12. Author: theory role (Opus 5.5). Pure mathematics.
Prior art: note 1 (`notes/1_walt_optimality_gap_and_repairs.md`, Oct 10) and
note 2 (`notes/2_walt_stabilization_and_complementary_plans.md`, Oct 12).
Code read: `scenario/planner.py` (the recursion, the floor, the tie rule),
`scenario/fiber.py` (`sample_worlds`: iid uniform draws, or the whole fiber when it
has at most N members), `arena/walt_play.py` (spec defaults). Measured numbers are
quoted from `wiki/entities/scenario-player.md` (the hub) and cited as such.

## 0. The answer, first

**Exact Walt finds the optimal move in finite time, against a fixed field, when its
belief is that field's posterior.** With any other belief it can fail, by up to the
whole range (note 1). That is T1 below, proved. "Optimal"
there means O1: the best move for one seat when the other three seats play a fixed,
declared policy F.

**Deployed Zeb does not find the O1 move in general.** It differs from exact Walt in
four places. Each is a real gap with a constructible counterexample, and each has an
exact repair:

| # | gap | what deployed Zeb does | exact repair | cost at the table |
|---|---|---|---|---|
| 1 | belief | uniform over the void-consistent fiber, $\beta^U$ | the field's posterior $\beta^F$ (with the net, a 0/1 consistency filter) | cheap if the acceptance rate is not tiny; see §6 |
| 2 | sampling | 512 iid worlds (160 on the phone) | a two-sided certificate (§4.1); or the whole fiber | exact late in the hand; infeasible to certify at the opening |
| 3 | floor | thin cells hand the move to the net | floor off when the fiber is enumerated exactly | free; small, late-hand only (§4.2) |
| 4 | referent | F = the net | F = the players actually at the table | unaffordable when they are Zeb itself (§6) |

The tie rule is not a gap: it never costs value (§4.3).

**The ladder stabilizes in finite time under fixed beliefs.** That is T2, proved in a
form the Lean role can formalize: it is a three-line induction on a rank function, with
no game theory in it. For 42 the opening lead is settled by the 24th application of
$\mathcal R$ from the uniform start, i.e. $P^{23}$ in note 2's indexing, and the general
bound is sharp. It applies to exact Walt or a rung-independent seed, not to the sampled
ladder as configured, whose seed mixes in the rung (§3.4). With policy-dependent (coherent) beliefs the ladder can cycle forever;
§3.6 gives a game where it cycles from the uniform start, checked in exact arithmetic.

**O1 is the most we can get cheaply.** An O1 fixed point is "person-by-person
optimal", not partnership-optimal (O2) and not team-maxmin (O3). The gap can approach
the whole range (note 1, §4). O3 is finite to certify (double oracle) and far beyond
any budget.

Verdict table:

| claim | status |
|---|---|
| T1: exact Walt with $\beta^F$ is O1 at the root, finite time | **proved** (§2.2) |
| T1′: replanning with $\beta^F$ every turn is a best response for the whole hand | **proved** (§2.3) |
| coherence bound, no horizon factor, $2\,\mathrm{osc}\cdot\mathrm{TV}$ | **proved**; make constant 2, points constant $4r$ ($\le 168$) (§2.4) |
| T2: the fixed-belief ladder stabilizes at height $h(c)$; 42: settled by the 24th application of $\mathcal R$ from the uniform start, i.e. $P^{23}$ | **proved** (§3), also in Lean; exact Walt or rung-independent seed only |
| T2 with policy-dependent beliefs | **false**: a 4-cycle from the uniform start (§3.6, verified exactly) |
| C2 sampling: a certificate that survives the selection effect | **proved** (§4.1) |
| C2 floor: loss bounded by the mass reaching floored cells | **proved** (§4.2) |
| C2 tie rule: harmless for value | **proved** (§4.3) |
| C3: O1 fixed points are not O2/O3 optimal | **cited** from note 1, plus a 42-shaped sketch (§5) |
| expert iteration (pour → net → Zeb) is the T2 ladder | **proved** only if imitation is exact and the seed is fixed across turns; that the plateau is $P^\star$ is a **conjecture** (§3.7) |
| the sampled ladder as configured stabilizes | **false in general**: the seed mixes in the rung, so each rung is a different operator (Lean `rung_seeded_no_settle`, §3.4) |

---

## 1. Definitions

### 1.1 The hand-phase game $\Gamma$

Fix the declaration (trump), the bidder seat $b$, and the bid $B$. Seats are
$\{0,1,2,3\}$. Teams are $\{0,2\}$ and $\{1,3\}$; the bidder team is $T_b \ni b$.

- **Worlds.** $W$ is the set of deals $w=(w_0,w_1,w_2,w_3)$, a partition of the 28
  tiles into four 7-tile hands. $w_j$ is seat $j$'s dealt hand.
- **Prior.** $\mu\in\Delta(W)$ is the deal distribution conditioned on the auction.
  Deployed Zeb uses $\mu$ uniform (it ignores the auction). We keep $\mu$ as a
  parameter.
- **Public history.** $s=(a_1,\dots,a_t)$, the tiles played so far in order. The
  seat to move, $j(s)$, is a function of $s$ alone (the leader and the trick winners
  are determined by the tiles). The banked points, the led suit, and the record of
  `scenario/record.py` are functions of $s$.
- **Legality.** $a_{t+1}$ is legal in $w$ after $s$ iff
  $a_{t+1}\in L\bigl(w_{j(s)}\setminus \mathrm{played}(s),\ \mathrm{led}(s)\bigr)$,
  where $L(h,\ell)$ is "follow the led suit if able, else anything in $h$".
  $s$ is *feasible in* $w$ iff every play in it was legal in $w$.
- **Terminals.** $s$ is terminal when $|s|=28$. Utilities depend only on $s$:
  $u^{\mathrm{make}}(s)=\mathbf 1\{\mathrm{banked}_{T_b}(s)\ge B\}$ and
  $u^{\mathrm{pts}}(s)=\mathrm{banked}_{T_b}(s)-\mathrm{banked}_{\bar T_b}(s)\in[-42,42]$.
  Seat $i$'s utility is $u_i=u$ if $i\in T_b$ and $u_i=-u$ otherwise (for `make`,
  $1-u$; the same up to a constant). This is the code's "bidder units, defenders
  minimize".
- **Information.** At $(w,s)$, seat $i$ knows $(w_i,s)$. An **information set** of $i$
  is $I=(x,s)$ with $x$ a 7-tile hand and $j(s)=i$.

**Lemma 0 (perfect recall).** $\Gamma$ has perfect recall.
*Proof.* Seat $i$'s information at any later decision is $(w_i,s')$ with $s'$
extending $s$. It contains its information at $s$ and every action it took
(they are in $s'$). $\square$

**Fiber.** $\mathrm{Fib}(I)=\{w\in\operatorname{supp}\mu:\ w_i=x,\ s \text{ feasible in } w\}$.
With $\mu$ uniform this is the set the counting DP of `scenario/fiber.py` samples:
the played tiles' owners are fixed by $s$, and feasibility reduces to the shown voids
(a seat that failed to follow suit $\sigma$ holds no $\sigma$ tile now). At the opening
lead $|\mathrm{Fib}|=21!/(7!)^3=399{,}072{,}960$.

### 1.2 Fields and beliefs

A **field** $F=(F_j)_{j\ne i}$ is a policy for the three other seats:
$F_j(\cdot\mid w_j,s)\in\Delta\bigl(L(w_j\setminus\mathrm{played}(s),\mathrm{led}(s))\bigr)$.
It may read only seat $j$'s information. The deployed net reads a function of
(remaining hand, record), a coarsening of $(w_j,s)$, so it is a legal field. It is
deterministic: integer inference, ties to the lowest tile id.

**Uniform-over-fiber belief.** $\beta^U(I)=\mu(\cdot\mid\mathrm{Fib}(I))$.
"That seat could legally have played this."

**Coherent posterior.**
$$
\beta^F(I)(w)\ \propto\ \mu(w)\,\mathbf 1\{w\in\mathrm{Fib}(I)\}
\prod_{t<|s|,\ j_t\ne i} F_{j_t}\bigl(a_t\mid w_{j_t},s_{<t}\bigr)
$$
(note 1, §2). "That seat *would* have played this." With $F$ deterministic every factor
is 0 or 1, so
$$
\beta^F(I)=\mu\bigl(\cdot\mid \mathrm{Fib}^F(I)\bigr),\qquad
\mathrm{Fib}^F(I)=\{w\in\mathrm{Fib}(I):\ F_{j_t}(w_{j_t},s_{<t})=a_t\ \ \forall t,\ j_t\ne i\}.
$$
It is undefined when $\mathrm{Fib}^F(I)=\varnothing$: the observed plays are
impossible under $F$. Define the **consistency fraction**
$\rho(I)=\mu(\mathrm{Fib}^F(I))/\mu(\mathrm{Fib}(I))$.

**Lemma 1 (Bayes).** Let the other seats play $F$ and seat $i$ play any policy
$\pi_i$ under which $I$ has positive probability. Then
$\Pr(w\mid I\text{ reached})=\beta^F(I)(w)$.
*Proof.* $\Pr(w, I)=\mu(w)\cdot\prod_{t:\,j_t\ne i}F_{j_t}(a_t\mid w_{j_t},s_{<t})\cdot
\prod_{t:\,j_t=i}\pi_i(a_t\mid x,s_{<t})\cdot\mathbf 1\{w_i=x,\ s\text{ feasible}\}$.
The last product depends on $w$ only through $w_i=x$, which is the same for every
$w$ in the fiber (perfect recall: $i$'s own factors are functions of its own
information). It cancels on normalizing. $\square$

**Lemma 2 (the TV of the filter).** If $\mu$ is uniform and $F$ is deterministic,
$\mathrm{TV}\bigl(\beta^U(I),\beta^F(I)\bigr)=1-\rho(I)$.
*Proof.* With $S=\mathrm{Fib}$, $S_F=\mathrm{Fib}^F\subseteq S$:
$\mathrm{TV}=\sum_{w\in S_F}\bigl(\tfrac1{|S_F|}-\tfrac1{|S|}\bigr)=1-\tfrac{|S_F|}{|S|}$. $\square$

This one number, $\rho$, is what the empirical role is measuring. Every TV bound below
becomes a bound in $\rho$.

### 1.3 The exact Walt operator

Fix $i$ and $I=(x,s)$ with $j(s)=i$. A **continuation policy** $\rho$ assigns to every
own information set $(x,s')$, $s'\sqsupseteq s$, $j(s')=i$, a legal tile. $\Pi^a(I)$ is
the set of those with $\rho(I)=a$. For a belief $\beta$ on $\mathrm{Fib}(I)$,
$$
V^F_\beta(\rho)=\sum_w \beta(w)\ \mathbb E\bigl[u_i(z)\mid w,s,\rho,F\bigr],\qquad
Q_i^F(\beta,a)=\max_{\rho\in\Pi^a(I)}V^F_\beta(\rho),
$$
$$
\mathcal R_\beta[F]_i(I)=\operatorname*{argmax}^{\prec}_{a}\ Q_i^F\bigl(\beta(I),a\bigr),
$$
where $\prec$ is the tie rule: among maximizers, the lowest tile id (`_best`).

**What the code computes.** For a finite weighted set $(S,\omega)$ of worlds, the
recursion in `_Invocation._value` is
$$
V(C,s')=\begin{cases}
\omega(C)\,u(s') & s'\ \text{settled or terminal}\\[2pt]
\max_{a}\ V\bigl(C,\ s'a\bigr) & j(s')=i\quad(\min\text{ for a defender})\\[2pt]
\sum_{a} V\bigl(C_a,\ s'a\bigr) & j(s')\ne i,\quad C_a=\{w\in C: F_{j(s')}(w,s')=a\}
\end{cases}
$$
with $C=S$ at the root. Under `make`, "settled" means the contract is already decided,
so every completion has the same utility and stopping is exact.

**Lemma W (the cell recursion is the max over legal policies).** With no floor and
no horizon, for every legal root action $a$,
$$
V(S,\ s a)\,/\,\omega(S)=Q_i^F\bigl(\omega/\omega(S),\ a\bigr).
$$
*Proof.* Induction on the number of plays remaining. Three facts carry it.
(i) At an own turn every world in the cell $C$ has the same information set
$(x,s')$, so "one action for the whole cell" is exactly one policy choice at that
information set. (ii) Each cell is indexed by its public history $s'$, so distinct
cells at own turns are distinct information sets. A policy's choices in different
cells are therefore independent, and the max of a sum of independent terms is the sum
of the maxes. This is where perfect recall is used: if two cells shared an information
set, choosing separately in each would be an illegal policy (strategy fusion).
(iii) At another seat's turn the deterministic field sends each world to exactly one
child, so the children's cells partition $C$ and values add. For a stochastic field,
replace the partition by a split with weights; the proof is the same. (The memo key is
(record, cell), so it merges histories with the same record and the same live worlds.
Their continuation problems are identical: field moves, legality and the utility all
read the record. Merging changes no value.) $\square$

So inside one invocation Walt **is** coherent: the cells are exactly the root sample
conditioned on $F$'s plays. The only incoherence is at the root, where the sample is
drawn from $\beta^U$ instead of $\beta^F$. This is the cleanest way to say where Zeb's
belief gap lives.

### 1.4 The ladder

A **coordinate** is $c=(j,I_j)$: a seat and one of its information sets. A
**profile** $P$ assigns a legal action to every coordinate. The response operator
acts on all seats at once:
$$
\mathcal R_\beta[P](j,I)=\operatorname*{argmax}^{\prec}_a\ Q_j^{P_{-j}}\bigl(\beta_j(I),a\bigr),
\qquad P^{-1}=\mathsf U,\quad P^k=\mathcal R_\beta[P^{k-1}].
$$
**Fixed beliefs** means $\beta_j(I)$ does not depend on $k$ or on $P^{k-1}$.
The deployed modeled minds use uniform weights (`_invocation`: "modeled minds weigh
their scenarios uniformly"), so the deployed ladder has fixed beliefs.

### 1.5 Deployed Zeb and the champion

- **Zeb** (`walt:k0,n512,F=<net>`, utility `make`, full horizon, floor 16):
  $$
  \mathrm{Zeb}(I)=\operatorname*{argmax}^{\prec}_a\ \widehat Q^{\,f}_S(a),
  \qquad S\sim\beta^U(I)^{\otimes N}\ \text{iid, or } S=\mathrm{Fib}(I)\text{ if }|\mathrm{Fib}(I)|\le N,
  $$
  with $F$ the net, $N=512$ in the arena and 160 on the phone (the package default),
  and $\widehat Q^{\,f}$ the cell recursion with the floor of §4.2. One rung: no
  ladder. It replans from scratch every turn.
- **Champion** (`walt:k1,n512,m8`): the same with $F$ = a rung-0 Walt at 8 worlds and
  a 4-play inner horizon, dice tape beyond.

The utility is bounded. We write
$\mathrm{osc}(I)=\sup u_i-\inf u_i$ over terminals reachable from $I$.
For `make`, $\mathrm{osc}\le 1$. For `points` with $r(s)=42-\mathrm{banked}_0(s)-\mathrm{banked}_1(s)$
points still unbanked, the final margin lies in an interval of width $2r$, so
$\mathrm{osc}\le 2r\le 84$.

---

## 2. T1: exact Walt with the coherent belief is an O1 best response

### 2.1 Statement

**Theorem T1.** Let $\Gamma$ be a finite extensive-form game with perfect recall, $F$
a fixed field, and $I$ an information set of seat $i$ reached with positive probability
under $(\mu,F)$. Let $\hat a=\mathcal R_{\beta^F}[F]_i(I)$. Then:

1. (**value**) $Q_i^F(\beta^F(I),a)=\max_{\rho\in\Pi^a(I)}\mathbb E\bigl[u_i\mid I,\rho,F\bigr]$
   for every legal $a$: the operator's value is the true conditional value.
2. (**optimality, O1 at the root**) $\hat a$ begins a continuation policy that
   maximizes $\mathbb E[u_i\mid I,\cdot,F]$ over all continuation policies.
3. (**finite time**) the cell recursion over $S=\mathrm{Fib}^F(I)$ with weights $\mu$
   computes every $Q_i^F(\beta^F(I),a)$ exactly in at most
   $28\cdot|\mathrm{Fib}^F(I)|\cdot n_i!$ cell evaluations, where $n_i$ is the
   number of tiles seat $i$ holds.

### 2.2 Proof

(1) By Lemma 1, the conditional law of $w$ given $I$ is $\beta^F(I)$ whatever $i$ did
before. Given $w$, the continuation under $(\rho,F)$ is a fixed (or, for stochastic
$F$, fixed-law) path. So $\mathbb E[u_i\mid I,\rho,F]=V^F_{\beta^F(I)}(\rho)$. Maximize
both sides over $\Pi^a(I)$.

(2) $\Pi(I)=\bigsqcup_a\Pi^a(I)$, so
$\max_\rho\mathbb E[u_i\mid I,\rho,F]=\max_a Q_i^F(\beta^F(I),a)=Q_i^F(\beta^F(I),\hat a)$.

(3) By Lemma W the recursion computes $Q$. Each cell is indexed by a public history
$s'$. Every visited $s'$ is the path of some world $w\in S$ under some sequence of own
choices. There are at most $n_i!$ such sequences per world (at most $n_i$ choices, then
$n_i-1$, and so on), each path has at most 28 prefixes. $\square$

The bound in (3) is finiteness, not a runtime: at the opening lead it is about
$5.6\times10^{13}$ for the full fiber. The memo merges most of it.

### 2.3 T1′: replanning every turn with $\beta^F$ realizes the plan

Zeb does not execute its plan; it replans. With coherent beliefs that costs nothing.

**Lemma D (decomposition).** For an own information set $I$ and action $a$, let $I'$
range over the next own information sets (or terminals) reached after $a$. Under
perfect recall and $\beta^F$,
$$
Q_i^F(\beta^F(I),a)=\mathbb E\Bigl[\ W(I')\ \Bigm|\ I,a,F\Bigr],\qquad
W(I')=\max_a Q_i^F(\beta^F(I'),a)
$$
(with $W$ = the utility at a terminal).
*Proof.* After $a$, the continuation policy's choices at different $I'$ are
independent (perfect recall: each $I'$ is a distinct public history), so the max
passes inside the expectation over $I'$. The law of $w$ given $(I,a,I')$ is
$\beta^F(I)$ conditioned on reaching $I'$, which is $\beta^F(I')$ by Lemma 1. $\square$

**Theorem T1′.** Let $\pi^R$ play $\mathcal R_{\beta^F}[F]_i(I)$ at every own information
set $I$ (replanning from scratch, coherent belief, exact evaluation). Then $\pi^R$ is a
best response to $F$ from every information set reached with positive probability; in
particular $\Delta_{\mathrm{execution}}=V_{\mathrm{plan}}-V_{\mathrm{execute}}=0$
(note 1, §7).
*Proof.* Backward induction on the number of own decisions left. With none left the
claim is trivial. Otherwise $\pi^R$ picks $\hat a$ with $Q(\hat a)=W(I)$. By Lemma D,
$W(I)=\mathbb E[W(I')\mid I,\hat a]$. By induction $\pi^R$ achieves $W(I')$ from each
$I'$. So it achieves $W(I)$ from $I$. $\square$

### 2.4 What carries T1, and what breaks without each hypothesis

| hypothesis | why it is needed (one line) |
|---|---|
| finiteness | with actions $a\in\mathbb N$ and payoff $1-1/a$ the sup is not attained, so no argmax exists; with an unbounded horizon the recursion does not terminate. |
| perfect recall | the absent-minded driver (Piccione–Rubinstein 1997): the best plan exits with probability 1/3, and no per-node argmax attains it; Lemma W's step (ii) fails. |
| $\beta^F$ coherent with $F$ | note 1's sender–receiver game: the frozen receiver makes 3/4 where 1 is attainable; the $m$-type family drives the gap to $(1-\alpha)(1-\tfrac{2}{m+1})\to1$. |
| exact evaluation | sampling and the floor change the argmax in constructible cases (§4). |
| $F$ is what the other seats actually play | O1 is relative to the model. If the table is not $F$, $\mathrm{Fib}^F(I)$ can be empty (β^F undefined) and nothing is claimed about actual results (§6, row "referent"). |

### 2.5 Belief-error bounds, with constants

**Lemma 3 (oscillation).** For any $f$ and beliefs $\beta,\nu$ on the same finite set,
$|\mathbb E_\beta f-\mathbb E_\nu f|\le\mathrm{osc}(f)\cdot\mathrm{TV}(\beta,\nu)$.
*Proof.* With $m$ the midpoint of the range of $f$,
$\mathbb E_\beta f-\mathbb E_\nu f=\sum_w(f(w)-m)(\beta(w)-\nu(w))\le
\tfrac{\mathrm{osc}}2\,\|\beta-\nu\|_1=\mathrm{osc}\cdot\mathrm{TV}$. $\square$

Let $\delta(I)=\mathrm{TV}(\beta(I),\nu(I))$ with $\nu=\beta^F$ the coherent posterior.

**Prop A (one decision).** $|Q_\beta(I,a)-Q_\nu(I,a)|\le\mathrm{osc}\cdot\delta(I)$, so the
action $\hat a$ chosen under $\beta$ has
$\max_aQ_\nu(I,a)-Q_\nu(I,\hat a)\le 2\,\mathrm{osc}\cdot\delta(I)$, and if the
$\beta$-gap between the best and second-best action exceeds $2\,\mathrm{osc}\cdot\delta(I)$,
the coherent argmax is the same action.
*Proof.* Lemma 3 for each fixed $\rho$, then take the max over the same $\Pi^a(I)$ on
both sides. For the regret: $Q_\nu(a^\star)\le Q_\beta(a^\star)+\mathrm{osc}\,\delta\le
Q_\beta(\hat a)+\mathrm{osc}\,\delta\le Q_\nu(\hat a)+2\,\mathrm{osc}\,\delta$. $\square$

**Prop B (no horizon factor; note 1, §3).** Let $\pi^\star$ be optimal under prior
$\mu$ and $\hat\pi$ optimal under $\hat\mu$, both complete policies from $I_0$ against the
same $F$. Then $V_\mu(\pi^\star)-V_\mu(\hat\pi)\le 2\,\mathrm{osc}\cdot\mathrm{TV}(\mu,\hat\mu)$.
*Proof.* $V_\mu(\pi^\star)-V_\mu(\hat\pi)=[V_\mu(\pi^\star)-V_{\hat\mu}(\pi^\star)]
+[V_{\hat\mu}(\pi^\star)-V_{\hat\mu}(\hat\pi)]+[V_{\hat\mu}(\hat\pi)-V_\mu(\hat\pi)]$.
The middle term is $\le0$; Lemma 3 bounds each outer term by $\mathrm{osc}\cdot\mathrm{TV}$. $\square$

*Reading for Zeb.* By Lemma W, Walt-uniform's root plan is exactly $\hat\pi$ with
$\hat\mu=\beta^U(I_0)$ and $\mu=\beta^F(I_0)$ (its cells condition on $F$). By Lemma 2,
**if Zeb committed to its root plan, it would lose at most $2(1-\rho(I_0))$ of pmake to
the O1 optimum** (exact evaluation assumed). Zeb does not commit; it replans with a
fresh $\beta^U$, which is not the conditioning of the root belief. That needs Prop C.

**Prop C (replanning; note 1, §3).** Let $\pi^R$ replan at every own $I$ with belief
$\beta(I)$ and exact evaluation. Then
$$
W_\nu(I_0)-V^{\pi^R}_\nu(I_0)\ \le\ \min\Bigl\{\mathrm{osc},\ 2\,\mathrm{osc}\cdot
\mathbb E_{\nu,\pi^R,F}\Bigl[\sum_{\text{own }I\text{ reached}}\delta(I)\Bigr]\Bigr\}.
$$
*Proof.* At each own $I$, with $\hat a=\pi^R(I)$: $W_\nu(I)-Q_\nu(I,\hat a)\le2\,\mathrm{osc}\,\delta(I)$
(Prop A), $Q_\nu(I,\hat a)=\mathbb E[W_\nu(I')\mid I,\hat a]$ (Lemma D), and
$V^{\pi^R}(I)=\mathbb E[V^{\pi^R}(I')\mid I,\hat a]$. Subtract and unroll along the play of
$\pi^R$. The $\mathrm{osc}$ cap is trivial. $\square$

**Constants.**

| utility | osc | action certificate (Prop A) | commitment bound (Prop B), deterministic F, uniform μ |
|---|---|---|---|
| make | 1 | $\Delta>2\,\delta$ | pmake loss $\le 2(1-\rho)$ |
| points, opening | 84 | $\Delta>168\,\delta$ points | margin loss $\le 168(1-\rho)$ |
| points, $r$ points unbanked | $2r$ | $\Delta>4r\,\delta$ | $\le 4r(1-\rho)$ |

These bounds are worst-case. With $\rho\le\tfrac12$ the make certificate needs
$\Delta>1$ and can never fire. So the TV route only certifies decisions when the
filter keeps most of the fiber. When it does not, the honest move is to compute
$Q_{\beta^F}$ directly (the repair), not to bound it.

---
## 3. T2: the fixed-belief ladder stabilizes (the Lean target)

The theorem is purely combinatorial. It needs an operator whose output at a coordinate
depends only on the input at a set of coordinates, and a rank that strictly decreases
along that dependency. No argmax, no probabilities, no ties appear in the proof. The
game content is all in checking the two hypotheses (§3.3, §3.4).

### 3.1 The abstract setting

- $C$: a type of coordinates (no finiteness needed).
- $A : C\to\mathrm{Type}$: the actions at each coordinate.
- $\mathcal P=\prod_{c\in C}A(c)$: profiles.
- $\mathcal R:\mathcal P\to\mathcal P$: the operator.
- $D: C\to\mathcal{S}et(C)$: the dependency sets.
- $\mathrm{rk}:C\to\mathbb N$: a rank.

**(Loc) Locality.** For all $c$ and $P,Q\in\mathcal P$:
if $P(d)=Q(d)$ for all $d\in D(c)$, then $\mathcal R(P)(c)=\mathcal R(Q)(c)$.

**(Dec) Rank decreases.** For all $c$ and $d\in D(c)$: $\mathrm{rk}(d)<\mathrm{rk}(c)$.

The **dependency height** is the least such rank:
$h(c)=0$ if $D(c)=\varnothing$, else $h(c)=1+\max_{d\in D(c)}h(d)$
(well defined whenever some rank exists and every $D(c)$ is finite;
the theorem below is stated for any rank, so Lean need not construct $h$).

Write $\mathcal R^n$ for the $n$-fold iterate, $P^{-1}$ for any base profile, and
$P^k=\mathcal R^{k+1}(P^{-1})$.

### 3.2 The theorem and its proof

**Lemma T2.1 (agreement).** For all $n\in\mathbb N$, all $c$ with $\mathrm{rk}(c)<n$,
and all $P,Q$: $\mathcal R^n(P)(c)=\mathcal R^n(Q)(c)$.

*Proof.* Induction on $n$.
*Base* $n=0$: no $c$ has $\mathrm{rk}(c)<0$. Vacuous.
*Step* $n\to n+1$: take $c$ with $\mathrm{rk}(c)\le n$. For $d\in D(c)$,
$\mathrm{rk}(d)<\mathrm{rk}(c)\le n$ by (Dec), so $\mathcal R^n(P)(d)=\mathcal R^n(Q)(d)$
by the induction hypothesis. By (Loc) applied to $\mathcal R^nP$ and $\mathcal R^nQ$,
$\mathcal R^{n+1}(P)(c)=\mathcal R(\mathcal R^nP)(c)=\mathcal R(\mathcal R^nQ)(c)=\mathcal R^{n+1}(Q)(c)$. $\square$

**Lemma T2.2 (stability).** If $\mathrm{rk}(c)<n\le m$ then $\mathcal R^m(P)(c)=\mathcal R^n(P)(c)$.
*Proof.* $\mathcal R^m(P)=\mathcal R^n(\mathcal R^{m-n}P)$. Apply T2.1 with
$Q=\mathcal R^{m-n}P$. $\square$

**Theorem T2 (finite stabilization).** Define $P^\star(c)=\mathcal R^{\mathrm{rk}(c)+1}(P^{-1})(c)$. Then:
1. $P^k(c)=P^\star(c)$ for every $k\ge\mathrm{rk}(c)$ (that is, after $\mathrm{rk}(c)+1$ applications);
2. $P^\star$ does not depend on the base $P^{-1}$;
3. $\mathcal R(P^\star)=P^\star$;
4. $P^\star$ is the only fixed point of $\mathcal R$.

*Proof.* (1) is T2.2 with $n=\mathrm{rk}(c)+1$, $m=k+1$. (2) is T2.1.
(3) Fix $c$ and let $Q=\mathcal R^{\mathrm{rk}(c)}(P^{-1})$. For $d\in D(c)$,
$\mathrm{rk}(d)+1\le\mathrm{rk}(c)$, so by T2.2, $P^\star(d)=\mathcal R^{\mathrm{rk}(d)+1}(P^{-1})(d)=Q(d)$.
By (Loc), $\mathcal R(P^\star)(c)=\mathcal R(Q)(c)=\mathcal R^{\mathrm{rk}(c)+1}(P^{-1})(c)=P^\star(c)$.
(4) If $\mathcal R(Q)=Q$ then $Q=\mathcal R^{\mathrm{rk}(c)+1}(Q)$, so
$Q(c)=\mathcal R^{\mathrm{rk}(c)+1}(Q)(c)=\mathcal R^{\mathrm{rk}(c)+1}(P^{-1})(c)=P^\star(c)$ by T2.1. $\square$

With $\mathrm{rk}=h$ this is note 2's statement: $P^k(c)=P^\star(c)$ for $k\ge h(c)$.

**Indexing, explicitly (agreed with the Lean role).** $P^{-1}=\mathsf U$ and
$P^k=\mathcal R[P^{k-1}]$, so $P^k=\mathcal R^{k+1}(\mathsf U)$ is the result of $k+1$
applications. Coordinate $c$ is settled in $P^{h(c)}$, which is the
$(h(c)+1)$-th application. A coordinate with no dependencies ($h=0$) still needs one
application: the Lean witness is a one-coordinate game with a constant response and a
wrong starting policy. For 42, "$R-1=23$" means $P^{23}$, the 24th application.

**Uniqueness (Lean corollary).** Under a uniform height bound $H$, $\mathcal R$ has
exactly one fixed point, and every fixed point equals $\mathcal R^{H+1}$ of any start
(T2(4) above, proved without $H$ by working coordinate-wise). So fixed-belief
Walt-vs-Walt has one stabilized profile, whatever the base.

**Tightness (abstract).** An abstract local operator can keep the same root answer for
$n$ applications and change it at application $n+1$; §3.5 is the game-shaped instance.
Unchanged root moves across rungs certify nothing.

The Lean role has formalized T2 (sorry-free, core Lean); see
`scratch/research-night/lean/`.

**Suggested Lean 4 shape** (a sketch for the Lean role, not checked):

```lean
variable {C : Type*} {A : C → Type*}

structure LocalOp (C : Type*) (A : C → Type*) where
  R    : ((c : C) → A c) → ((c : C) → A c)
  D    : C → Set C
  rk   : C → ℕ
  loc  : ∀ c P Q, (∀ d ∈ D c, P d = Q d) → R P c = R Q c
  dec  : ∀ c, ∀ d ∈ D c, rk d < rk c

theorem LocalOp.agree (O : LocalOp C A) :
    ∀ n c, O.rk c < n → ∀ P Q, (O.R^[n] P) c = (O.R^[n] Q) c
theorem LocalOp.stable (O : LocalOp C A) (P) (c) (k : ℕ) (hk : O.rk c ≤ k) :
    (O.R^[k+1] P) c = (O.R^[O.rk c + 1] P) c
def LocalOp.star (O : LocalOp C A) (P₀) : (c : C) → A c := fun c => (O.R^[O.rk c + 1] P₀) c
theorem LocalOp.star_fixed (O : LocalOp C A) (P₀) : O.R (O.star P₀) = O.star P₀
theorem LocalOp.star_unique (O : LocalOp C A) (P₀ Q) (h : O.R Q = Q) : Q = O.star P₀
```

Note `Function.iterate_succ_apply'` gives `R^[n+1] P = R (R^[n] P)`, which is the
form the induction step wants; `Function.iterate_add_apply` gives T2.2.

### 3.3 Forced coordinates

**Lemma T2.3 (forced coordinates are constant).** If $A(d)$ is a singleton (or any
subsingleton), then $P(d)=Q(d)$ for all $P,Q$. Hence if $D$ satisfies (Loc), so does
$D'(c)=D(c)\setminus\{d:\ A(d)\text{ subsingleton}\}$.
*Proof.* `Subsingleton.elim`. For the second part: agreement on $D'(c)$ plus automatic
agreement on the forced coordinates is agreement on $D(c)$. $\square$

So forced coordinates can be dropped from every dependency set, and they need no rank.
In 42 a coordinate is forced exactly when the legal set has one tile (the last tile in
hand, or the only tile of the led suit).

### 3.4 The instance: the Walt ladder with fixed beliefs

Coordinates are $c=(j,I)$ with $j=j(s)$, actions the legal tiles, and
$\mathcal R=\mathcal R_\beta$ from §1.4 (any deterministic tie rule).

**Dependencies.** $Q_j^{P_{-j}}(\beta_j(I),a)$ is computed from three things: the
fixed belief $\beta_j(I)$, the utility, and the values of $P$ at coordinates
$(j',I')$ with $j'\ne j$ and $I'$ strictly later than $I$ (the modeled seats' future
moves). With a cell floor, also $(j,I')$ with $I'$ strictly later: the floored decision
is the lower rung's value at the viewer's **own** later coordinate (`_handed_down` =
`_modeled_move(k-1, viewer, hand, rec)`), so the dependency set must include own later
coordinates; they are still later, the rank still drops, and the bound of 24
applications stands (Lean: `floored_local`). Let $D(c)$ be all non-forced such
coordinates. (Loc) holds because
$\beta_j(I)$ does not depend on $P$. This is the only place "fixed beliefs" is used.

**The 42 rank.** For a public history $s$ with $n_j(s)$ tiles left in seat $j$'s hand
(public: every seat started with 7 and the plays are public), let
$$
\mathsf R(s)=\sum_{j=0}^3\max\bigl(n_j(s)-1,\,0\bigr),\qquad
\mathrm{rk}(j,I)=\mathsf R(s)-1\ \ \text{for non-forced } c,\qquad \mathrm{rk}=0\ \text{for forced } c.
$$

**Lemma T2.4.** (Dec) holds for this rank.
*Proof.* Let $c=(j,(x,s))$ be non-forced, so seat $j$ holds $n_j(s)\ge2$ tiles. Let
$d=(j',(x',s'))\in D(c)$, non-forced, $s'$ strictly extending $s$. The play at $s$ is by
seat $j$ with $n_j\ge2$, so it lowers $\mathsf R$ by exactly 1; no play raises $\mathsf R$.
So $\mathsf R(s')\le\mathsf R(s)-1$ and $\mathrm{rk}(d)<\mathrm{rk}(c)$. And
$\mathrm{rk}(c)\ge0$ because a non-forced decision has $n_j\ge2$, so $\mathsf R\ge1$. $\square$

**Corollary T2-42.** At the opening lead $\mathsf R=4\cdot6=24$, so
$\mathrm{rk}=23$: **the opening lead is settled by the 24th application of
$\mathcal R_\beta$ from the uniform start, i.e. $P^{23}=\mathcal R_\beta^{24}(\mathsf U)$ in note 2's
indexing**, for every base and every fixed belief constructor. Every later non-forced
decision settles no later. This is note 2, §2.2, now proved.

The bound is on reasoning levels, not runtime, and it is not claimed tight for 42.

**The seed catch (Lean: `rung_seeded_no_settle`).** T2 iterates *one* operator. The
code does not: `_invocation` (planner.py line 304) mixes the rung index $k$ into the
sample seed, so rung $k$ is its own operator $\mathcal R_k$, with different samples and
so different decisions even when the `Rung` configs are identical. Lean exhibits
operators that are local for the **empty** dependency (height 0 everywhere) whose
rung-indexed iterates alternate forever. So:

- T2 applies to **exact Walt** (fiber enumerated, which `fiber.py` does when the fiber
  is no larger than the sample) and to a **rung-independent seed** with identical rung
  configs. There the sample is a fixed function of the coordinate and $\mathcal R$ is one
  local operator.
- T2 does **not** apply to the sampled ladder as configured. The rungs differ by
  sampling noise at every height. Base-independence after $\mathrm{rk}(c)+1$ rungs
  survives (T2.1's proof works for a composition of operators all local for the same
  $D$); stability does not.
- Neither deployed player is an iteration anyway. The champion `k1,n512,m8` is one
  application of $\mathcal R_1$ to $\mathcal R_0[\text{dice}]$; Zeb is one application of
  $\mathcal R_0$ to the net. What they inherit from T2 is locality, floor included, not
  stabilization.

### 3.5 Sharpness in general (note 2, §2.3)

Note 2's family: $n$ alternating decisions between two teammates; at decision $t$ stop
for $s_t=1-3^{-(n-t)}$ or continue; continuing at the last decision pays 1. The
dependency chain $1\to2\to\cdots\to n$ has $n-1$ edges, so $h(\text{root})=n-1$, and the
root first changes at $P^{n-1}$. The key inequality is
$\tfrac{s_{t+1}+1}2<s_t\iff 3^{-(n-t)}<\tfrac12\,3^{-(n-t-1)}\iff\tfrac13<\tfrac12$.
So the height bound is attained, and **several unchanged rungs never certify that the
next rung will not move the root**. Verified in exact arithmetic for $n=2,\dots,8$:
`theory/sharp_check.py` prints the root's action by rung (`S…SC…C`, first change at
$P^{n-1}$ every time).

### 3.6 Why policy-dependent beliefs break it: a cycle from the uniform start

If $\beta_j(I)=\beta^{P^{k-1}}(I)$ (coherent with the rung below), then $\beta_j(I)$
depends on $P^{k-1}$ at the *earlier* coordinates that produced the record. $D(c)$ then
contains coordinates before $c$, edges point both ways in time, and no decreasing rank
exists. That is a failure of the proof. Here is a failure of the conclusion.

**Game (a 42-shaped signalling game).** Nature deals seat 1 a type $\theta\in\{H,T\}$,
each with probability 1/2. Seat 1 publicly plays $m\in\{A,B\}$. Seat 2, an opponent,
sees $m$ (not $\theta$) and guesses $g\in\{H,T\}$. Seat 1 scores
$u_1=\varepsilon\cdot\mathbf 1\{m=\mathrm{nat}(\theta)\}+\mathbf 1\{g\ne\theta\}$, with
$\mathrm{nat}(H)=A$, $\mathrm{nat}(T)=B$, $0<\varepsilon<1$. Seat 2 scores
$\mathbf 1\{g=\theta\}$. Think of a defender reading the bidder's discard, where each
hand has a slightly better natural discard. Coherent beliefs: seat 2's belief after
$m$ is Bayes from seat 1's previous-rung policy. Tie rule: lowest letter. Simultaneous
response, $P^{-1}$ uniform.

| rung | seat 1 (H, T) | seat 2 (after A, after B) |
|---|---|---|
| $P^0$ | A, B | H, H |
| $P^1$ | A, B | H, T |
| $P^2$ | B, A | H, T |
| $P^3$ | B, A | T, H |
| $P^4$ | A, B | T, H |
| $P^5$ | A, B | H, T |
| $P^6$ | B, A | H, T $\;=P^2$ |

$P^2\to P^3\to P^4\to P^5\to P^2$ is a 4-cycle. Every response from $P^1$ on is
strict. Verified in exact rational arithmetic:
`scratch/research-night/theory/cycle_check.py` (prints `P^2 == P^6: True`,
`P^2..P^5 distinct: True`). With the same game and **fixed** beliefs ($\tfrac12$ after
either message), seat 2's response is "always H" at every rung, seat 1's is "A, B", and
the ladder is settled at $P^0$, as T2 says.

This is note 1's hidden matching pennies, rebuilt so that it cycles from the uniform
start under a fixed tie rule. Coherent beliefs and the fixed-height guarantee do not
coexist in general. Any repair that makes the *ladder* coherent needs its own
convergence argument (fictitious play, regret matching, or the double oracle of §5).
This is the strongest reason the T1 repair is for **one seat against a frozen field**,
not for the ladder.

### 3.7 Two remarks for Zeb as deployed

**Sampled rungs.** See the seed catch in §3.4. The seed mixes in $k$, and the rungs
also differ in $N$ and horizon, so the sampled ladder applies a different operator
$\widehat{\mathcal R}_k$ at each level. Base-independence survives (T2.1 with
$\mathcal R^n$ replaced by $\widehat{\mathcal R}_n\circ\cdots\circ\widehat{\mathcal R}_1$);
stability does not. Dropping $k$ from the seed and sharing $N$ and horizon across rungs
would make the sampled operator one fixed local operator, and T2 would then apply to
the sampled ladder exactly. Deployed Zeb is a single application, so none of this
changes what Zeb plays.

**Expert iteration is the T2 ladder only under two idealizations.** The pour is played
by the deployed player, $\widehat{\mathcal R}_0[\text{net}_t]$, and $\text{net}_{t+1}$
imitates its moves. That loop iterates **one** map,
$\text{net}\mapsto\text{imitation of }\widehat{\mathcal R}_0[\text{net}]$, only if
(i) the imitation is exact on every coordinate and (ii) the planner's seed is fixed
across turns of the loop (the pour seeds do change: 1001.., 100001.., 200001..). Under
both, the loop is the T2 ladder, it reaches $P^\star$ by the 24th turn, and by T2(2) it
reaches **the same $P^\star$ an exact rung-independent ladder reaches from any base**,
dice or net. Two cautions. Imitation is
partial (held-out agreement 53–55%, hub), so the idealization is loose. And $P^\star$ is
a fixed point of *frozen-belief* responses, so even reaching it exactly does not give
O1 (§3.6, §5). **Conjecture (with both caveats above):** the measured plateau (50k: +0.11, 250k: +0.13 marks/game
vs the champion, hub) reflects the loop approaching the neighbourhood of $P^\star$, where
better imitation stops changing the response. Testable: measure root-move agreement
between successive nets' Zebs on a fixed position set; T2 predicts it rises toward
100% if the loop is converging.

---
## 4. C2: sampling, the floor, the tie rule

Fix an information set $I$ and a referent belief $\beta$ ($\beta^U$ for "exact
Walt-uniform", $\beta^F$ for O1). Write $Q(a)=Q_i^F(\beta,a)$ for the exact values and
$A$ for the legal root actions, $|A|\le7$.

### 4.1 Sampling: a certificate that survives the selection effect

Zeb's $\widehat Q_S(a)$ is a max over continuation policies of an empirical mean, and
the policy is chosen on the same sample. So $\mathbb E\,\widehat Q_S(a)\ge Q(a)$ (a max of
unbiased estimates is biased up), and a confidence interval around $\widehat Q_S(\hat a)$
is not valid. A union bound over the 7 root actions does not help, because the
selection happens over continuation policies, not root actions. The hub measured the
size of this effect at an opening lead: floor-1 forecast +30.1 margin / 83% made
versus +9.6 / 45% in real endings. This is note 1's warning ("an ordinary confidence
interval around a selected winner is not automatically sufficient"), quantified.

The repair uses the optimism where it is safe and a fresh sample where it is not.

**Setup.** $S_1$: $N$ iid worlds from $\beta$. $S_2$: $M$ iid worlds from $\beta$,
independent of $S_1$.
- $\widehat Q^{\,1}_{S_1}(b)$: the **unfloored** cell recursion on $S_1$ (floor 1).
  By Lemma W it equals $\max_{\rho\in\Pi^b(I)}\widehat V_{S_1}(\rho)$.
- $\hat\rho_a$: any continuation policy built from $S_1$ alone and defined at every
  information set. The natural choice is Zeb's floored plan, completed by the net at
  information sets $S_1$ never reached. (With the floor, that completion is what the
  plan already does in thin cells.)
- $\widetilde V_{S_2}(a)$: the mean utility of the fixed policy $(a,\hat\rho_a)$
  against $F$ over $S_2$. This is a replay, not a solve: linear in $M$.
- $\varepsilon_K=\mathrm{osc}\cdot\sqrt{\ln(|A|/\eta)/(2K)}$.

**Theorem C2.1.** With probability at least $1-2\eta$, simultaneously for all $a,b\in A$:
$$
Q(b)\ \le\ \widehat Q^{\,1}_{S_1}(b)+\varepsilon_N,\qquad
Q(a)\ \ge\ \widetilde V_{S_2}(a)-\varepsilon_M .
$$
Consequently, on that event: (i) if $\widetilde V_{S_2}(\hat a)-\varepsilon_M>\max_{b\ne\hat a}\widehat Q^{\,1}_{S_1}(b)+\varepsilon_N$,
then $\hat a$ is the unique exact argmax; and (ii) always,
$\max_bQ(b)-Q(\hat a)\le\max_b\widehat Q^{\,1}_{S_1}(b)+\varepsilon_N-\widetilde V_{S_2}(\hat a)+\varepsilon_M$,
a computable regret bound for the move actually played.

*Proof.* Upper side: let $\rho^\star_b$ be an exact maximizer for $Q(b)$. It is fixed
before sampling. Hoeffding (one-sided, range osc) gives
$\Pr[\widehat V_{S_1}(\rho^\star_b)<Q(b)-\varepsilon_N]\le\eta/|A|$. And
$\widehat Q^{\,1}_{S_1}(b)\ge\widehat V_{S_1}(\rho^\star_b)$. Union over $b$.
Lower side: conditional on $S_1$, each $(a,\hat\rho_a)$ is fixed and $S_2$ is
independent, so Hoeffding applies to $\widetilde V_{S_2}(a)$ around
$V(a,\hat\rho_a)\le Q(a)$. Union over $a$. (i) and (ii) follow by chaining. $\square$

The point of the construction: the **upper** bound may be optimistic (that is the safe
direction), so the clairvoyant floor-1 value serves. The **lower** bound must be a
fixed policy scored on fresh worlds, so the honest floored plan serves. The hub's
two forecasts, floor 1 and floor 16, are exactly these two sides.

**Numbers** ($|A|=7$, $\eta=0.025$ per side, so 95% overall):

| $K$ worlds | $\varepsilon_K$, make (pp of pmake) | $\varepsilon_K$, points at the opening (osc 84) |
|---|---|---|
| 160 | 13.3 | 11.2 |
| 512 | 7.4 | 6.2 |
| 1,024 | 5.2 | 4.4 |
| 4,096 | 2.6 | 2.2 |
| 16,384 | 1.3 | 1.1 |

**What this means at the table.** At $N=M=512$ the certificate needs the honest value
of the chosen move to beat every rival's clairvoyant value by about 15 pp. At the
opening the clairvoyance gap alone was 38 pp (hub), and it does not shrink with $N$
while cells stay thin. So **the certificate essentially never fires early in a hand.**
It fires late, and it is trivial when the fiber is enumerated: then $S=\mathrm{Fib}$,
$\varepsilon=0$, and Lemma W makes $\widehat Q=Q$ exactly. `sample_worlds` already
enumerates whenever $|\mathrm{Fib}|\le N$. Fiber sizes at the start of a trick with $k$
tiles in each hidden hand, before voids (voids only shrink them):

| tiles per hidden hand $k$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| $(3k)!/(k!)^3$ | 6 | 90 | 1,680 | 34,650 | 756,756 | 17,153,136 | 399,072,960 |

At $N=512$ every decision of the last two tricks is exact (below the root the viewer's
own moves there are forced); at $N=2{,}048$ the last three, before voids.

A tighter upper side is a research lever, not a result: information-relaxation duality
(Brown, Smith and Sun, *Operations Research* 2010) subtracts a penalty for using
information early, which can close part of the clairvoyance gap while staying a valid
bound.

### 4.2 The floor: what it computes and what it can cost

**What it computes.** At an own decision below the root whose cell has fewer than $f$
sample worlds, Zeb plays $h(C)=\mathrm{net}_i(\text{own remaining hand},\ \text{record})$,
one tile for the whole cell (`_floored`, `_handed_down`). Cells only shrink along a
branch, so once a branch is floored every later own decision in it is floored: the
viewer's seat is handed to the net for the rest of that branch. By Lemma W with the
floored nodes restricted to one action,
$$
\widehat Q^{\,f}_S(a)=\max_{\rho\in\Pi^{a,f}_S(I)}\widehat V_S(\rho),\qquad
\Pi^{a,f}_S=\{\rho\in\Pi^a(I):\ \rho=\text{net at own info sets whose }S\text{-cell has }<f\text{ worlds}\}.
$$
The root is never floored.

**Prop F1 (value lost).** Let $\hat\rho^{\,1}_a$ maximize the unfloored value on $S$.
Then
$$
0\ \le\ \widehat Q^{\,1}_S(a)-\widehat Q^{\,f}_S(a)\ \le\ \mathrm{osc}\cdot
\Pr_{w\sim S}\bigl[\text{the play of }(a,\hat\rho^{\,1}_a,F)\text{ from }w\text{ reaches an own cell with }<f\text{ worlds}\bigr].
$$
*Proof.* The left inequality: $\Pi^{a,f}_S\subseteq\Pi^a$. For the right, let $\rho'$
equal $\hat\rho^{\,1}_a$ except at floored information sets, where it plays the net.
Then $\rho'\in\Pi^{a,f}_S$. Whether an information set is floored depends only on its
public history, so in each world the plays of $\rho'$ and $\hat\rho^{\,1}_a$ coincide
until the first floored own decision. They can differ by at most osc only in worlds
that reach one. So $\widehat Q^{\,f}_S(a)\ge\widehat V_S(\rho')\ge\widehat Q^{\,1}_S(a)-\mathrm{osc}\cdot\Pr[\text{reach}]$. $\square$

**Two regimes.**
- *Exact regime* ($S=\mathrm{Fib}$, enumerated). Cells are exact information-set
  supports; a max over a small exact cell is honest, not clairvoyant. Here the floor
  is a pure loss under the model, bounded by Prop F1 with the true reach probability.
  **The code still floors in this regime** (`_floored` counts live worlds whatever the
  sampler did). The regime is narrow at $N=512$: a floor only matters at a non-forced
  own decision below the root, so the viewer needs at least three tiles while the
  fiber fits in 512. Before voids that is mainly the third trick from the end with the
  viewer playing third (hidden hands 2,2,3: 210 worlds) or fourth (2,2,2: 90 worlds),
  plus fibers that voids have shrunk. There, cells fall under 16 within a trick, and
  the viewer hands its next decision to the net in a position it could solve exactly.
  **Repair: floor 1 whenever the fiber was enumerated. Cost: a few extra cells, late in
  the hand. It closes this row of the gap completely; the value at stake is small.**
- *Sampled regime.* The floor is not an approximation of $Q$; it is a guard against
  the selection effect of §4.1. It trades a bounded value loss (Prop F1) for less
  optimism. For one cell, the optimism is controlled by the maximal inequality: with
  $n$ iid worlds and $k$ fixed continuations of range osc,
  $\mathbb E[\max_a\bar X_a]-\max_a\mu_a\le\mathrm{osc}\sqrt{\ln k/(2n)}$, which is about
  $0.98\,\mathrm{osc}$ at $n=1,k=7$ and $0.25\,\mathrm{osc}$ at $n=16$. **Conjecture:** the
  total optimism of $\widehat Q^{\,f}$ is at most osc times the expected sum, over the
  free cells on the plan's path, of $\sqrt{\ln 7/(2|C|)}$ weighted by the cell's mass.
  The composition through the recursion is not proved; the one-cell lemma is.
  The hub's measurement is consistent with it: floors 8 to 161 gave identical forecasts
  (+10.4 / 35%) within a point of real endings (+11.5 / 34%).

### 4.3 The tie rule

**Prop T.** Any member of $\operatorname{argmax}_aQ(a)$ has the same value, so the tie
rule never costs value at the decision where it is applied. *Proof:* definition. $\square$

It matters in three other places:
1. It decides *which* optimal move Zeb returns; O1 does not care.
2. In the ladder it defines the modeled policy $P^k$, so it changes higher rungs'
   values. T2 needs only that it is fixed.
3. Under $\beta^F$ it is part of $F$: the filter must reproduce the net's tie rule
   bit for bit or it discards worlds the net would have produced. The integer net does
   this by construction (referee and core are bit-identical, hub).

Under sampling, make values are multiples of $1/N$, so exact sample ties are common.
Lowest-tile-id then picks deterministically among moves the sample cannot separate.
That costs nothing beyond what Theorem C2.1's regret bound already accounts for.

### 4.4 What we would need: the theorem we can actually get

**Corollary Z.** Let Zeb′ be Zeb with three changes: (a) the root sample is drawn from
$\beta^{\text{net}}(I)$ (exact rejection on uniform draws: keep a draw iff the net
replays every other-seat play of the record from it); (b) floor 1 when the sample is
the whole filtered fiber; (c) the certificate of C2.1 computed at each decision. Then, with $F$ = the net (and so, at a
table that really plays the net, against the table itself):
1. whenever $|\mathrm{Fib}(I)|\le N$ (enumerate the fiber, then filter it), Zeb′'s move
   is an O1 best response, with certainty (T1 + Lemma W);
2. the probability that the certificate fires on a move that is not the O1 best
   response is at most $2\eta$ (T1 + C2.1; a statement over the sampling, not a
   conditional probability given that it fired);
3. with probability at least $1-2\eta$, its regret is at most the bound C2.1(ii);
4. by T1′, if every decision of the hand is exact in sense 1, the whole hand is a best
   response, and $\Delta_{\mathrm{execution}}=0$.

*Proof.* Rejection on iid uniform draws gives iid draws from $\mu(\cdot\mid\mathrm{Fib}^{\text{net}})=\beta^{\text{net}}$
(Lemma 2's setting). The rest is the cited results. $\square$

That is the precise sense in which "Zeb finds the optimal move in finite time" can be
made true: **against the net, late in the hand, exactly; earlier, only when a
certificate fires, which at the opening it will not.** "Against the net" is the catch.
§6 row 9 explains why it does not transfer to a table of Zebs.

---

## 5. C3: why O1 is not O2 or O3

### 5.1 The referents

- **O2 (partnership).** The bidder's team chooses a *pair* $(\pi_i,\pi_{i+2})$ of
  seat-local policies, against fixed opponents. Each policy still reads only its own
  seat's information.
- **O3 (team maxmin).** $\max_{x}\min_{y}U(x,y)$ over the two teams' (correlated, ex ante
  coordinated) team strategies (note 1, §6).

An O1 fixed point under coherent beliefs is a profile where each seat best-responds to
the others: in team-theory language, **person-by-person optimal** (Marschak and Radner,
*Economic Theory of Teams*, 1972). That is necessary for team optimality, not sufficient.

### 5.2 The counterexamples (cited, note 1 §4)

| | partner $S$ | partner $R$ |
|---|---|---|
| you $S$ | $\varepsilon$ | 0 |
| you $R$ | 0 | 1 |

$(S,S)$ is person-by-person optimal; the pair $(R,R)$ makes 1. The uniform start
escapes this table, but note 1's $(m{+}1)$-action family does not: with $c=3/(m+1)$,
action 0 is the unique response to uniform play ($\tfrac{3(m+2)}{2(m+1)^2}$ against
$\tfrac{2m+5}{2(m+1)^2}$ for action 1; checked) and to itself, while the pair (1,1)
makes 1. **The O1-to-O2 gap is $1-\tfrac{3}{m+1}\to1$**, with correct beliefs, strict
responses and a uniform start.

**A 42-shaped sketch (not a verified position).** Conventions are exactly this shape.
Say the partner, on lead, can return a trump to the bidder ($R$) or cash a side double
($S$), and the bidder, at the next lead, can pull the last trump ($R$) or play off side
suits ($S$). If "return trump" only pays when the bidder pulls, and "pull" only pays
after a return, each seat's lone switch to $R$ loses and the pair's switch wins. A
learned field that never plays $R$ makes $R$ a losing deviation for Zeb at every seat.

**Conjecture (self-confirming conventions).** Expert iteration cannot discover such a
convention. The net imitates Zeb, and Zeb best-responds to the net one seat at a time
(§3.7). A convention neither partner plays is never in the pour, so it never enters the
net, so it is never a best response. This is a mechanism for "a stronger player will
not come from more of this pour" (hub). It is not proved that 42 has such conventions
of material size. The probe in §6 row 11 would measure it.

### 5.3 The barriers

**O2 is finite and enormous.** With opponents fixed, the partnership's problem is a
finite-horizon decentralized POMDP. Those are NEXP-complete in general (Bernstein,
Givan, Immerman and Zilberstein 2002). 42 has a helpful structure: every action is
public, so the partners' common information is the entire record. The
common-information approach (Nayyar, Mahajan and Teneketzis, IEEE TAC 2013) turns the
team into one coordinator who, at each public history, picks a *prescription* (a map
from the mover's possible hands to tiles). That is a single-agent POMDP, exactly
solvable by backward induction. The prescription space is the problem: at the opening
the mover's possible hands number $\binom{21}{7}=116{,}280$, so prescriptions number
up to $7^{116{,}280}$. Alternating one-seat improvement (JESP-style) only reaches
person-by-person optima, which is the trap of §5.2 again.

**O3 is finite to certify and further out.** Note 1, §6: a double oracle keeps finite
policy sets, solves the restricted game, and adds best responses; with exact oracles it
stops at gap $H_t-L_t=0$ in finitely many iterations. Each oracle call is an O2 problem
(a best team response to a mixed opposing team). The number of iterations is bounded
by the number of pure team policies. A crude lower bound on that number: after trick 1
there are on the order of $7\cdot21\cdot20\cdot19\approx5.6\times10^4$ public histories
at which seat 0 next decides with at least two legal tiles, so seat 0 alone has more
than $2^{10^4}$ pure policies (a rough count, not a careful one). Computing
team-maxmin equilibria in extensive form is hard in general; I cite Celli and Gatti
(AAAI 2018) from memory and the orchestrator should verify the exact statement. **O3
for full 42 is out of reach. For endgames of two or three tricks it is plausible and
would give a real exploitability certificate.**

---
## 6. The Q ledger: what we would need, what it costs, what to measure

### 6.1 Timings it is priced in (all from the hub unless marked)

| what | wall per decision |
|---|---|
| Zeb in WASM (Node 24, M5), opening lead, 160 worlds | 0.044 s |
| Zeb in WASM, seven-declaration forecast, 160 / 512 worlds | 0.30 s / 1.2 s (about 0.04 / 0.17 s per root analysis) |
| compiled core, one thread, `k0,n160` dice field, play 0 | 0.056 s |
| compiled core, one thread, `k1,n160,m8,g4` / `gfull`, play 0 | 1.12 s / 10.5 s |
| champion `k1,n512,m8`, seven-declaration forecast | 4.7 s |
| field-net forward, **Python referee**, features included (measured tonight, `theory/net_timing.py`, 300 positions, 43 active features on average) | 26 µs |
| field-net forward, Rust / WASM | not measured; at most the Python figure |

Jason's bar: 2–3 s per decision on a phone; 20 s "really really takes you out of the
game". The phone's slowdown against the M5 is not measured here.

### 6.2 The ledger

$\rho$ is the consistency fraction of §1.2, $m$ the number of other-seat non-forced
plays so far, $a$ the per-play agreement between the net and the players actually at
the table.

| # | target | gap | exact repair | cost in Zeb's units | affordable at the table? | experiment that measures the gap |
|---|---|---|---|---|---|---|
| 1 | O1 vs net | belief $\beta^U\ne\beta^{\text{net}}$ (TV $=1-\rho$, Lemma 2) | **consistency filter**: draw uniform worlds, replay the record with the net on each, keep the consistent ones, plan on $N$ kept; this *is* $\beta^{\text{net}}$ (Lemma 2, Cor. Z) | $(N/\rho)$ draws $\times\ m$ forwards. At $N=512$, $m=12$, 26 µs (Python bound): **0.16 s/ρ**, so 0.16 s at ρ=1, 1.6 s at ρ=0.1, 16 s at ρ=0.01; early exit on the first mismatch lowers it; Rust lowers it further | **yes if ρ ≳ 0.1** at the Python bound; ρ comes from the empirical role | empirical role (C1): ρ by plays; move, pmake and plan changes with filtered worlds. Then an arena block **Zeb-filtered vs Zeb, both against a table of three nets**: the one setting where O1-vs-net is the true referent |
| 2 | O1 vs net | filter risk: effective-sample collapse | sample from the posterior, never reweight: rejection (row 1), or sampling-importance-resampling from a pool | pool $=N/\text{ESS-fraction}$; at the hub's measured ESS fractions for the ε=0.1 field tilt (0.02–0.08) that is 6k–26k draws to keep 512; filtering cost only, planning stays at $N$ | yes, by row 1's arithmetic | ESS fraction and distinct-world count of the kept set, by plays |
| 3 | O1 | sampling, $N$ finite | certificate C2.1 (split sample, floor-1 upper side, held-out lower side); or enumerate the fiber | one floor-1 solve on $S_1$ (more nodes than floor 16: 297M vs 119M at one hub opening, `k1`) + a linear replay on $S_2$; exactness at the opening needs $N\ge3.99\times10^8$ | the check: yes late, of little use early (it will not fire). Exactness: already true for the last two tricks at $N=512$ | fraction of decisions where C2.1 fires, by plays; the regret bound C2.1(ii) averaged over a pour |
| 4 | O1 | floor active on an **enumerated** fiber | floor 1 when the sample is the whole fiber | a few extra cells, third trick from the end and later | **yes, free** | count floored decisions on enumerated fibers in a pour; arena block "f=16, f=1 when exact" vs Zeb, 384 mirrored games. Under the model this is a pure gain (Prop F1); at a real table it is an empirical question |
| 5 | O1 | floor in the sampled regime | none short of exact enumeration; the floor is the optimism guard | none | keep it | $\widehat Q^1-\widehat Q^{16}$ and held-out values per position (the two sides of C2.1) |
| 6 | O1 | tie rule | none needed (Prop T) | 0 | yes | none |
| 7 | O1 | replanning with a fresh $\beta^U$ | coherent $\beta^F$ (then T1′: zero), or commit to the root plan (then Prop B: $\le2(1-\rho)$) | row 1's cost | as row 1 | $\Delta_{\mathrm{execution}}=V_{\text{plan}}-V_{\text{execute}}$ on exactly solvable positions, three tricks left, full fiber of 1,680 (note 1 §7; empirical role) |
| 8 | O1 | prior $\mu$ ignores the auction | multiply in the auction likelihood (built: the `auction` belief) | negligible | yes | measured: auction belief −0.05 [−0.37, +0.27] marks/game with the heuristic bidders (hub); with real bidders, unmeasured |
| 9 | O1 vs the **actual** table | referent: $F$ = net, table = Zeb (or people) | $F$ := Zeb itself; likelihood = Zeb's own replayed decision per world per past play | $(N/\rho_Z)\times m\times t_{\text{Zeb}}$; the replayed decisions are this hand's earlier positions, 0.01–0.044 s each (hub scaling), so at $N=512$, $m=12$: **60–270 s even at ρ=1** | **no** | the net-as-proxy risk: the true world survives the net filter with probability $\approx a^m$; with $a\approx0.54$ (held-out agreement, hub): 16% at $m=3$, 2.5% at $m=6$, 0.4% at $m=9$, 0.06% at $m=12$. Measure $a$ and the survival rate of the true world directly in self-play |
| 10 | O2 | partnership coordination (§5.2) | common-information coordinator over prescriptions (§5.3); exact | prescriptions up to $7^{116{,}280}$ at the opening | **no** | exact joint vs single best response on endgames of two tricks (90 worlds before voids): the O1-to-O2 gap where it is computable |
| 11 | O2 | self-confirming conventions in expert iteration (§5.2, conjecture) | joint deviation search: perturb both partners' policies together in the field, keep what the pair gains | offline only | n/a (offline) | (a) T2's prediction: root-move agreement between successive nets' Zebs on a fixed position set should rise toward 100%; (b) a pair-deviation probe on endgames |
| 12 | O3 | team maxmin | double oracle with exact O2 oracles and the certificate $H_t-L_t$ (note 1 §6) | at least the O2 cost per iteration, up to $2^{10^4}$ iterations | **no**; endgames offline only | exploitability **lower bound**: train a best-response opposing pair against frozen Zeb; any gain is a proven lower bound on the O3 gap (no upper bound follows from failing to find one) |

### 6.3 The cheapest row, said plainly

The consistency filter (row 1) is $\beta^{\text{net}}$ exactly, with zero new
machinery: a forward pass per world per past play, then the same planner. By Lemma W
it makes Zeb's plan coherent end to end, and by Corollary Z it makes Zeb an exact O1
player late in the hand. Its cost is $0.16\,\text{s}/\rho$ at the Python bound.

It carries two risks, and they pull in opposite directions.

1. **Collapse.** If $\rho$ is small, the kept set is small. Reweighting 512 uniform
   draws by a sharp likelihood is the field tilt the hub measured at −0.65 and −0.96
   marks/game, and the mechanism was effective sample size (0.02–0.08 of the worlds
   kept at ε=0.1). The cure is known: "a strong g must be sampled from" (hub). Rejection
   sampling is exactly that.
2. **Misspecification.** The filter keeps the worlds in which *the net* would have
   played what the table played. If the table is not the net, the true world is
   usually not among them (row 9: 2.5% survival after six discretionary plays at 54%
   agreement). Softening the filter to $\varepsilon/n+(1-\varepsilon)\mathbf1[\text{match}]$
   restores the true world's support and brings back risk 1.

So the filter is the right repair for O1 against the net, and a doubtful one for play
against Zeb or people, until the net's per-play agreement with the table is high. This
squares with the hub's finding that uniform beat every belief tried: when the model of
the other seats is only about half right per play, the compatibility-only belief is the
robust choice.

**Prediction for tonight's measurement (empirical role).** In the self-play pour the
other three seats *are* the deployed player. Its per-play agreement with the net is
about $a=0.53$ on held-out positions (hub; 0.55 for the wide net). If disagreements are
roughly independent across plays, the real deal survives the net's consistency filter
with probability about $a^m$, where $m$ is the number of other-seat **non-forced** plays
so far:

| other-seat non-forced plays $m$ | 3 | 6 | 9 | 12 |
|---|---|---|---|---|
| predicted survival of the real deal, $a=0.53$ | 14.9% | 2.2% | 0.33% | 0.05% |
| same, $a=0.55$ | 16.6% | 2.8% | 0.46% | 0.08% |

A measured curve near this refutes nothing and confirms the misspecification risk.
A curve well **above** it means the net's errors cluster on positions the real deal
does not visit (agreement on the hand's own path is higher than on held-out rows), and
the filter is less dangerous than this section says. A curve well **below** it means
errors compound. Compare the curve with the filter's acceptance rate $\rho$ at the same
$m$. **Conjecture:** the filter helps at a real table when survival stays well above
$\rho$ (the real deal is over-represented among kept worlds) and hurts when it falls
to $\rho$ or below.

---

## 7. What I could not do, and what is conjecture

- **Not measured here:** $\rho$ in real positions (the empirical role's C1 run), the
  Rust/WASM net forward time (only the Python bound, 26 µs), the phone's slowdown, the
  survival rate of the true world under the net filter in self-play.
- **Conjectures, marked in the text:** the composition of the one-cell optimism bound
  through the floored recursion (§4.2); the plateau of expert iteration as the
  neighbourhood of $P^\star$ (§3.7); self-confirming conventions as a material O2 gap in
  42 (§5.2); the filter's value at a real table being governed by $a$ (§6.3).
- **Cited, not re-derived:** note 1's counterexample families (the $m$-action
  arithmetic is re-checked in §5.2) and double-oracle termination.
- **From memory, to verify:** Celli and Gatti (AAAI 2018) on the hardness of team-maxmin
  in extensive form; Bernstein et al. (2002) NEXP-completeness of finite-horizon
  Dec-POMDPs; Nayyar, Mahajan and Teneketzis (2013) common-information approach;
  Brown, Smith and Sun (2010) information relaxations; Piccione and Rubinstein (1997)
  absent-minded driver; Marschak and Radner (1972) person-by-person optimality.
- **Not attempted:** a real 42 position for C3. The 42-shaped sketch is a story with a
  payoff table, not a position.
- **Files:** this report; `theory/cycle_check.py` (the §3.6 cycle, exact arithmetic,
  under a second); `theory/sharp_check.py` (note 2's sharpness family, $n\le8$, under a second); `theory/net_timing.py` (the 26 µs bound, under 10 s).
