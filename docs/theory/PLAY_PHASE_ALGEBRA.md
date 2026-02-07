# Texas 42: Play-Phase Algebra (Beyond Suits)

This note formalizes the **play phase only** for a fixed **(deal, declaration)**, building on the suit/trump mechanics in `docs/theory/SUIT_ALGEBRA.md`.

It is written to match the representation in `scripts/solver2/` (GPU solver / training-data generator), where the packed state is intentionally **score-free**: all scoring appears as **per-trick transition rewards**.

---

## §1. Scope and Inputs

Fix:

- Seats: $S = \mathbb{Z}/4\mathbb{Z} = \{0,1,2,3\}$ (clockwise).
- Teams: $T_0=\{0,2\}$ and $T_1=\{1,3\}$.
- Domino set $\mathcal{D}$ (double-six) as in `docs/theory/SUIT_ALGEBRA.md`.
- Declaration $\delta \in \Delta$ as in `docs/theory/SUIT_ALGEBRA.md` / `scripts/solver2/declarations.py`.
- Deal: a partition $\mathcal{D}=H_0 \sqcup H_1 \sqcup H_2 \sqcup H_3$ with $|H_s|=7$.
- Initial leader $L \in S$ (in `solver2` this is fixed to $L=0$).

We model only:

- The 7-trick play phase (28 plays total)
- Trick legality (follow-if-possible)
- Trick winner resolution
- Trick scoring (42 points total per hand)

We explicitly do **not** model bidding, contract success/failure, or early termination.

---

## §2. Local Indices (The “Gauge” of a Deal)

Let $I=\{0,1,2,3,4,5,6\}$ be **local indices** inside a player’s 7-domino hand.

Choose bijections (one per seat):

$$
\Lambda_s : I \to H_s
$$

Intuition: $\Lambda_s(i)$ is the **global domino** held by seat $s$ at local position $i$.

This choice is *not part of the game*; it is a representational freedom (a “gauge”). In `solver2`, the chosen gauge is the sorted order of global domino IDs per hand (`scripts/solver2/rng.py` + `scripts/solver2/context.py`).

---

## §3. State Space as a Sufficient Statistic

Define remaining subsets $R_s \subseteq I$ for each seat $s$ (which local indices are still in that player’s hand).

Let:

- $L \in S$ be the current trick leader
- $k \in \{0,1,2,3\}$ be the number of plays made in the current trick (the “trick length”)
- $p_0,p_1,p_2 \in I \cup \{\bot\}$ be the leader’s, second player’s, and third player’s played local indices for the current trick prefix (with $\bot$ meaning “unset”)

A play-phase state is:

$$
s = \big((R_s)_{s\in S},\, L,\, k,\, p_0, p_1, p_2\big)
$$

The **current player** is determined (no extra state needed):

$$
\mathsf{player}(s) = L + k \pmod 4
$$

### Markov Property (Why This State Is Enough)

Future legality and outcomes depend only on:

- which dominoes remain in each hand ($R_s$), and
- the current trick prefix ($L,k,p_0,p_1,p_2$),

not on the earlier order of play. This is the key quotient that turns the full game tree into a much smaller **state DAG** (the graph that `solver2` enumerates).

**Implementation note:** `solver2` packs $(R_0,R_1,R_2,R_3,L,k,p_0,p_1,p_2)$ into one `int64`, storing each $R_s$ as a 7-bit mask and using the sentinel value `7` for $\bot$ (`scripts/solver2/state.py`, `scripts/solver2/expand.py`).

---

## §4. Legality as “Follow-If-Possible”

All suit/trump mechanics are delegated to the suit algebra (`docs/theory/SUIT_ALGEBRA.md`) via:

- `led_suit_for_lead_domino(d0, δ) ∈ {0..6,7}`
- `can_follow(d, led_suit, δ) ∈ {True, False}`
- `trick_rank(d, led_suit, δ) ∈ \mathbb{N}` (tiered rank)

Let the leader’s played domino (once it exists) be:

$$
d_0 = \Lambda_L(p_0)
$$

Define the led suit:

$$
\ell = \mathsf{ledSuit}(d_0,\delta) \in \{0,1,2,3,4,5,6,7\}
$$

For a seat $s$ and local index $i\in I$, define “can follow” at the local level:

$$
\mathsf{followable}(s,i;\ell,\delta)\iff \mathsf{canFollow}(\Lambda_s(i),\ell,\delta)
$$

Now define the followable set in the current state:

$$
F_s(s;\ell,\delta) = \{i \in R_s : \mathsf{followable}(s,i;\ell,\delta)\}
$$

Legal action set $A(s)$ for the current player $u=\mathsf{player}(s)$:

- If $k=0$ (leading): $$A(s)=R_u$$
- If $k>0$ and $F_u \neq \varnothing$: $$A(s)=F_u$$
- If $k>0$ and $F_u = \varnothing$: $$A(s)=R_u \quad\text{(must slough)}$$

This is exactly the rule in `docs/theory/SUIT_ALGEBRA.md` §10 and is realized in `solver2` as bitmask intersections using the precomputed `SeedContext.LOCAL_FOLLOW` table (`scripts/solver2/context.py`, `scripts/solver2/expand.py`).

---

## §5. Trick Completion as a Reducer

When $k=3$, the current player plays the 4th domino $p_3 \in I$, completing the trick:

$$
(d_0,d_1,d_2,d_3) =
\big(
\Lambda_L(p_0),\,
\Lambda_{L+1}(p_1),\,
\Lambda_{L+2}(p_2),\,
\Lambda_{L+3}(p_3)
\big)
$$

Winner offset:

$$
w = \underset{j\in\{0,1,2,3\}}{\arg\max}\;\tau(d_j,\ell,\delta)
$$

Winner seat:

$$
W = L + w \pmod 4
$$

Then the state update is:

- remove $p_j$ from $R_{L+j}$ for $j=0,1,2,3$
- set $L := W$
- reset the trick prefix: $k:=0$, $p_0=p_1=p_2:=\bot$

For $k<3$, the update is “mid-trick”: remove the played local index from the current player’s $R_u$, increment $k$, and store the play into the appropriate $p_j$ field.

**Implementation note:** `solver2` precomputes trick winner and points for all $7^4$ local tuples, per leader seat, in `SeedContext.TRICK_*` tables (`scripts/solver2/context.py`), and uses them in `expand_gpu` (`scripts/solver2/expand.py`).

---

## §6. Scoring and the Signed-Reward Decomposition

Define the counting value function $c:\mathcal{D}\to\{0,5,10\}$:

$$
c(d)=
\begin{cases}
10 & \text{if } d\in\{(5,5),(6,4)\} \\
5 & \text{if } d\in\{(5,0),(4,1),(3,2)\} \\
0 & \text{otherwise}
\end{cases}
$$

Define trick points:

$$
\mathsf{pts}(d_0,d_1,d_2,d_3)=1+\sum_{j=0}^3 c(d_j)
$$

The “+1” is the per-trick point; across 7 tricks that contributes 7 points total. The counting dominoes contribute 35 points total, so:

$$
\sum_{\text{7 tricks}} \mathsf{pts} = 42
$$

### Signed Reward (Two-Team Reduction)

Let $\mathsf{team}(s)=s\bmod 2$. Define the signed reward on the *completing move* of a trick:

$$
r = \begin{cases}
\mathsf{pts} & \text{if } \mathsf{team}(W)=0\\
-\mathsf{pts} & \text{if } \mathsf{team}(W)=1
\end{cases}
$$

All mid-trick moves have reward $0$.

**Key identity:** the sum of signed rewards over the entire hand equals the team point differential:

$$
\sum r = \text{Team0Points} - \text{Team1Points} = 2\cdot\text{Team0Points} - 42
$$

So optimizing Team 0’s points is equivalent to optimizing $\sum r$.

### Consequence: Score-Free State

Because all scoring can be pushed onto trick-completing transitions (and the terminal has “no more reward”), we can set:

$$
V(\text{terminal}) = 0
$$

and avoid storing an explicit score in the state. This is why `scripts/solver2/state.py` has `compute_terminal_value ≡ 0` and why the packed state has no score field.

---

## §7. The Graded State DAG and Backward Induction

Define the **level** (dominoes remaining):

$$
\mathsf{level}(s)=\sum_{s\in S}|R_s|
$$

Each legal move removes exactly one local index from exactly one $R_s$, so:

$$
\mathsf{level}(s')=\mathsf{level}(s)-1
$$

Therefore the reachable state graph is a finite **DAG** graded by `level`. This grading is exactly what `solver2` computes with popcount (`scripts/solver2/state.py`) to solve by backward induction.

Define the value function $V$ by:

$$
V(s)=
\begin{cases}
0 & \text{if } \mathsf{level}(s)=0 \\
\max\limits_{a\in A(s)} \big(r(s,a)+V(s')\big) & \text{if } \mathsf{team}(\mathsf{player}(s))=0 \\
\min\limits_{a\in A(s)} \big(r(s,a)+V(s')\big) & \text{if } \mathsf{team}(\mathsf{player}(s))=1
\end{cases}
$$

This is the Bellman/minimax recursion implemented in `scripts/solver2/solve.py`.

---

## §8. Isomorphisms (Structural Symmetries)

### 8.1 Local-Index Gauge: $S_7^4$ Action

For each seat $s$, let $\pi_s \in S_7$ be a permutation of $I$ and define:

$$
\Lambda'_s = \Lambda_s \circ \pi_s^{-1}
$$

This is a pure relabeling of local indices; it does not change the underlying deal.

Transport a state by:

$$
R'_s = \pi_s(R_s)
$$

and transport actions by applying $\pi_{\mathsf{player}(s)}$ to the chosen index.

Under this transport, the game graph and payoff structure are isomorphic. Concretely: it permutes bits inside each 7-bit hand mask and correspondingly permutes the context tables (`LOCAL_FOLLOW`, `TRICK_*`).

**Practical corollary:** choosing sorted global domino IDs as the local order is a convenient gauge-fix that makes contexts reproducible.

### 8.2 Table Symmetries: Dihedral Action on Seats

Any rotation/reflection of the 4 seats (the dihedral group $D_4$ of the square) acts on the deal by permuting the hands and leader.

- If the seat permutation preserves the team partition $\{0,2\}$ / $\{1,3\}$, it preserves payoff values.
- If it swaps the teams, it negates the signed payoff (and swaps max/min roles).

This is a useful normalization lever when comparing equivalent problems, even though `solver2` fixes the initial leader to seat 0.

### 8.3 Contrast with Suit Algebra Symmetry

`docs/theory/SUIT_ALGEBRA.md` identifies large pip-relabeling symmetries at the level of suit incidence structure. The full play-phase game breaks most of those symmetries because:

- ranks use numeric pip sums, and
- scoring singles out specific counting dominoes.

So the “big” $S_7$ symmetry of the suit covering does not lift to an automorphism group of the full play-phase + scoring system.

---

## §9. Code Correspondence (solver2)

The main mathematical objects above correspond to:

- $\delta$ (declaration): `scripts/solver2/declarations.py`
- Domino set $\mathcal{D}$, suit/rank predicates: `scripts/solver2/tables.py`
- Deal + gauge-fix $\Lambda_s$: `scripts/solver2/rng.py`, `scripts/solver2/context.py` (`SeedContext.L`)
- State packing + level/team functions: `scripts/solver2/state.py`
- Move legality + transition function: `scripts/solver2/expand.py`
- Precomputed legality/outcome tables: `scripts/solver2/context.py` (`LOCAL_FOLLOW`, `TRICK_WINNER`, `TRICK_POINTS`, `TRICK_REWARD`)
- Backward induction / minimax: `scripts/solver2/solve.py`
