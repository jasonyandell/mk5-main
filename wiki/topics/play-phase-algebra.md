---
title: Play-phase algebra — state model, rewards, and the solved game
kind: topic
first_seen: 2025-12-28
last_updated: 2026-07-11
status: active
---

The formal model of the **play phase only** for a fixed (deal, declaration), building on
[[suit-algebra-spec]]. This is the mathematics [[the-oracle]] implements: the packed
score-free state, follow-if-possible legality, the signed-reward decomposition, and the
graded DAG that backward induction solves. Written to match the representation in
`forge/oracle/` (the GPU solver / training-data generator, née `scripts/solver2/`).

Merged at this ingest from `docs/theory/PLAY_PHASE_ALGEBRA.md` (first committed 69d636ac)
and its spec companion `PLAY_PHASE_SPEC.md` (869f71ea); code paths updated from
`scripts/solver2/` to `forge/oracle/`.

## §1. Scope and inputs

Fix:

- Seats: $S = \mathbb{Z}/4\mathbb{Z} = \{0,1,2,3\}$ (clockwise).
- Teams: $T_0=\{0,2\}$ and $T_1=\{1,3\}$.
- Domino set $\mathcal{D}$ (double-six) and declaration $\delta \in \Delta$ as in [[suit-algebra-spec]].
- Deal: a partition $\mathcal{D}=H_0 \sqcup H_1 \sqcup H_2 \sqcup H_3$ with $|H_s|=7$.
- Initial leader $L \in S$ (the oracle fixes $L=0$).

Modeled: the 7-trick play phase (28 plays), trick legality (follow-if-possible), trick
winner resolution, trick scoring (42 points total per hand). **Not** modeled: bidding,
contract success/failure, marks, early termination.

## §2. Local indices (the "gauge" of a deal)

Let $I=\{0,1,2,3,4,5,6\}$ be **local indices** inside a player's 7-domino hand.

Choose bijections (one per seat):

$$\Lambda_s : I \to H_s$$

$\Lambda_s(i)$ is the **global domino** held by seat $s$ at local position $i$. This choice
is *not part of the game*; it is a representational freedom (a "gauge"). The oracle's chosen
gauge is the sorted order of global domino IDs per hand (`forge/oracle/rng.py` +
`forge/oracle/context.py`).

## §3. State space as a sufficient statistic

Define remaining subsets $R_s \subseteq I$ for each seat $s$. Let $L \in S$ be the current
trick leader, $k \in \{0,1,2,3\}$ the number of plays in the current trick, and
$p_0,p_1,p_2 \in I \cup \{\bot\}$ the current trick prefix's played local indices.

A play-phase state is:

$$s = \big((R_s)_{s\in S},\, L,\, k,\, p_0, p_1, p_2\big)$$

The **current player** is determined: $\mathsf{player}(s) = L + k \pmod 4$.

### Markov property

Future legality and outcomes depend only on which dominoes remain in each hand ($R_s$) and
the current trick prefix ($L,k,p_0,p_1,p_2$) — not on the earlier order of play. This
quotient turns the full game tree into a much smaller **state DAG**, the graph the oracle
enumerates.

**Implementation:** the oracle packs $(R_0,R_1,R_2,R_3,L,k,p_0,p_1,p_2)$ into one `int64`,
each $R_s$ a 7-bit mask, sentinel `7` for $\bot$ (`forge/oracle/state.py`,
`forge/oracle/expand.py`).

## §4. Legality as follow-if-possible

Suit/trump mechanics are delegated to [[suit-algebra-spec]] via `led_suit_for_lead_domino`,
`can_follow`, and `trick_rank` (`forge/oracle/tables.py`).

With the leader's played domino $d_0 = \Lambda_L(p_0)$ and led suit
$\ell = \mathsf{ledSuit}(d_0,\delta)$, define the followable set for seat $s$:

$$F_s = \{i \in R_s : \mathsf{canFollow}(\Lambda_s(i),\ell,\delta)\}$$

Legal action set $A(s)$ for the current player $u$:

- If $k=0$ (leading): $A(s)=R_u$
- If $k>0$ and $F_u \neq \varnothing$: $A(s)=F_u$
- If $k>0$ and $F_u = \varnothing$: $A(s)=R_u$ (must slough)

This is the only legality constraint in the play phase. The oracle realizes it as bitmask
intersections against the precomputed `SeedContext.LOCAL_FOLLOW` table
(`forge/oracle/context.py`, `forge/oracle/expand.py`).

## §5. Trick completion as a reducer

When $k=3$, the current player plays the 4th domino $p_3$, completing the trick:

$$(d_0,d_1,d_2,d_3) = \big(\Lambda_L(p_0),\, \Lambda_{L+1}(p_1),\, \Lambda_{L+2}(p_2),\, \Lambda_{L+3}(p_3)\big)$$

Winner offset $w = \arg\max_{j} \tau(d_j,\ell,\delta)$; winner seat $W = L + w \pmod 4$
(unique by the unique-winner theorem, [[suit-algebra-spec]] §7). The update: remove each
$p_j$ from $R_{L+j}$, set $L := W$, reset $k:=0$ and $p_0=p_1=p_2:=\bot$. For $k<3$ the
update is mid-trick: remove the played index, increment $k$, store the play.

**Implementation:** the oracle precomputes trick winner and points for all $7^4$ local
tuples, per leader seat, in `SeedContext.TRICK_*` tables (`forge/oracle/context.py`), used
by `expand_gpu` (`forge/oracle/expand.py`).

## §6. Scoring and the signed-reward decomposition

Counting value $c:\mathcal{D}\to\{0,5,10\}$: 10 for 5-5 and 6-4; 5 for 5-0, 4-1, 3-2;
0 otherwise. Trick points:

$$\mathsf{pts}(d_0,d_1,d_2,d_3)=1+\sum_{j=0}^3 c(d_j)$$

The "+1" per trick contributes 7 across the hand; count dominoes contribute 35; total 42.

### Signed reward (two-team reduction)

With $\mathsf{team}(s)=s\bmod 2$, define the reward on the *completing move* of a trick:

$$r = \begin{cases} \mathsf{pts} & \text{if } \mathsf{team}(W)=0\\ -\mathsf{pts} & \text{if } \mathsf{team}(W)=1 \end{cases}$$

All mid-trick moves have reward 0.

**Key identity:** the sum of signed rewards over the hand equals the team point differential:

$$\sum r = \text{Team0Points} - \text{Team1Points} = 2\cdot\text{Team0Points} - 42$$

Optimizing Team 0's points is equivalent to optimizing $\sum r$.

### Consequence: score-free state

All scoring lives on trick-completing transitions and the terminal has no more reward, so
$V(\text{terminal}) = 0$ and the packed state needs no score field. This is why
`forge/oracle/state.py` has `compute_terminal_value ≡ 0`.

## §7. The graded state DAG and backward induction

Define $\mathsf{level}(s)=\sum_{s\in S}|R_s|$ (dominoes remaining). Each legal move removes
exactly one local index, so $\mathsf{level}(s')=\mathsf{level}(s)-1$: the reachable graph is
a finite **DAG graded by level** (computed by popcount in `forge/oracle/state.py`), solvable
by backward induction:

$$V(s)= \begin{cases} 0 & \text{if } \mathsf{level}(s)=0 \\ \max\limits_{a\in A(s)} \big(r(s,a)+V(s')\big) & \text{if } \mathsf{team}(\mathsf{player}(s))=0 \\ \min\limits_{a\in A(s)} \big(r(s,a)+V(s')\big) & \text{if } \mathsf{team}(\mathsf{player}(s))=1 \end{cases}$$

This is the Bellman/minimax recursion in `forge/oracle/solve.py` — the ground truth behind
[[expected-q-value]] and everything distilled from it ([[gus]], [[student-distillation]]).

## §8. Isomorphisms (structural symmetries)

### 8.1 Local-index gauge: $S_7^4$ action

For each seat, a permutation $\pi_s \in S_7$ of $I$ with $\Lambda'_s = \Lambda_s \circ \pi_s^{-1}$
is a pure relabeling: transport states by $R'_s = \pi_s(R_s)$ and actions by
$\pi_{\mathsf{player}(s)}$, and the game graph and payoffs are isomorphic. Concretely it
permutes bits inside each 7-bit hand mask and the context tables. **Practical corollary:**
sorted global domino IDs as the local order is a gauge-fix that makes contexts reproducible.

### 8.2 Table symmetries: dihedral action on seats

Any rotation/reflection of the 4 seats ($D_4$) acts on the deal by permuting hands and
leader. If it preserves the team partition it preserves payoffs; if it swaps teams it
negates the signed payoff (and swaps max/min roles). A useful normalization lever, though
the oracle fixes the initial leader to seat 0.

### 8.3 Contrast with the suit-algebra symmetry

[[suit-algebra-spec]] §9 identifies large pip-relabeling symmetries of the suit incidence
structure. The full play-phase game breaks most of them: ranks use numeric pip sums, and
scoring singles out specific counting dominoes. The big $S_7$ symmetry does not lift to the
scored game — and per-deal it does not even lift to the unscored legality tree, because a
mixed domino leads its higher end (only the monotone relabeling transports play trees; see
the lead-direction bullet in [[suit-algebra-spec]] §9). The measured consequence — exact
equivalence over deals/worlds is empty at H4 while within-hand ties are real — is
[[endgame-equivalence-census]].

## §9. Checkable properties

For any fixed (deal, $\delta$) the play phase is finite, deterministic, and
perfect-information. Implementation-agnostic check suite:

- **Follow compliance**: no legal move allows sloughing when a follower exists.
- **Winner uniqueness**: each completed trick has exactly one winner.
- **Score conservation**: total points across 7 tricks is 42; Team0 + Team1 = 42.
- **State progression**: exactly 28 plays, grouped into 7 tricks.

## §10. Code correspondence (forge/oracle)

- $\delta$ (declaration): `forge/oracle/declarations.py`
- Domino set, suit/rank predicates: `forge/oracle/tables.py`
- Deal + gauge-fix $\Lambda_s$: `forge/oracle/rng.py`, `forge/oracle/context.py` (`SeedContext.L`)
- State packing + level/team functions: `forge/oracle/state.py`
- Move legality + transitions: `forge/oracle/expand.py`
- Precomputed legality/outcome tables: `forge/oracle/context.py` (`LOCAL_FOLLOW`, `TRICK_WINNER`, `TRICK_POINTS`, `TRICK_REWARD`)
- Backward induction / minimax: `forge/oracle/solve.py`

## Links

[[suit-algebra-spec]] · [[suit-algebra]] · [[rules-of-42]] · [[the-oracle]] ·
[[expected-q-value]] · [[breakthrough-and-oracle]]
