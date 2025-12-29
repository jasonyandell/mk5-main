# Texas 42: The Three-Axis Power System

## Overview

This document formalizes the three interlocking systems that determine game value in Texas 42:

1. **Rank Algebra**: Domino strength under suit/trump context
2. **Void Lattice**: Who can respond to which suits
3. **Control System**: Lead token dynamics and tempo

The conjecture is that game value decomposes (or nearly decomposes) as a function of these three axes, enabling efficient computation without full tree enumeration.

This document assumes the core definitions from:

- `docs/theory/SUIT_ALGEBRA.md` (led suit, following, trump power, trick ranking)
- `docs/theory/PLAY_PHASE_ALGEBRA.md` (play-phase state, transitions, signed rewards)

---

## §1. Preliminaries

### 1.1 The Domino Set

$\mathcal{D} = \{(i,j) : 0 \le i \le j \le 6\}$, the edges of $K_7$ (complete graph on 7 vertices).

$|\mathcal{D}| = 28$

Partition $\mathcal{D}$ into doubles and mixed:

$$\mathcal{D}^\circ = \{(p,p) : p \in \{0..6\}\} \qquad \mathcal{D}^\times = \mathcal{D} \setminus \mathcal{D}^\circ$$

### 1.2 Suits (Natural, Called, Effective)

We use the suit language of `docs/theory/SUIT_ALGEBRA.md`.

For pip value $p \in \{0,1,2,3,4,5,6\}$, the **natural suit** is:

$$\sigma_p = \{d \in \mathcal{D} : p \in d\}$$

This is a covering (not a partition): each mixed domino belongs to exactly two $\sigma_p$, and each double belongs to exactly one $\sigma_p$.

Under a declaration $\delta$, dominoes are **called** into an 8th suit via the called-set function $\kappa(\delta)$. The called suit is:

$$\sigma_7^\delta = \kappa(\delta)$$

and the effective (followable) pip suits are:

$$\hat{\sigma}_p^\delta = \sigma_p \setminus \kappa(\delta)$$

The 8 led-suit values are $\ell \in \{0..6,7\}$, where $\ell=7$ denotes the called suit.

For notational convenience, extend the definition with:

$$\hat{\sigma}_7^\delta := \sigma_7^\delta$$

**Important:** there is no always-present "doubles suit" in straight 42. The doubles set $\mathcal{D}^\circ$ becomes the called suit only when $\delta \in \{\mathsf{doubles\text{-}trump}, \mathsf{doubles\text{-}suit}\}$.

### 1.3 Declarations (Called Set and Power)

Declarations are:

$$\delta \in \Delta = \{0,1,2,3,4,5,6\} \cup \{\mathsf{doubles\text{-}trump}, \mathsf{doubles\text{-}suit}, \mathsf{notrump}\}$$

They induce:

- the called set $\kappa(\delta)$ (which dominoes are in suit 7)
- the power function $\pi(\delta) \in \mathcal{P}(\mathcal{D}) \cup \{\bot\}$ (which suit, if any, beats all others)

As in `docs/theory/SUIT_ALGEBRA.md`, $\pi(\delta)=\sigma_7^\delta$ for pip-trump and doubles-trump, and $\pi(\delta)=\bot$ for doubles-suit and notrump.

### 1.4 Seats and Teams

- Seats: $\mathcal{S} = \mathbb{Z}/4\mathbb{Z} = \{0,1,2,3\}$
- Teams: $T_0 = \{0,2\}$, $T_1 = \{1,3\}$
- $\text{team}(s) = s \mod 2$

### 1.5 Deals

A deal is a partition $\mathcal{D} = H_0 \sqcup H_1 \sqcup H_2 \sqcup H_3$ with $|H_s| = 7$.

---

## §2. The Rank Algebra

### 2.1 The Tier Function

For declaration $\delta$ and led suit $\ell \in \{0..7\}$, define the tier function (matching `docs/theory/SUIT_ALGEBRA.md`):

$$\text{tier}_{\delta,\ell} : \mathcal{D} \to \{0, 1, 2\}$$

$$\text{tier}_{\delta,\ell}(d) = \begin{cases}
2 & \text{if } \pi(\delta) \neq \bot \text{ and } d \in \pi(\delta) & \text{(power / trump)}\\
1 & \text{if } d \in \hat{\sigma}_\ell^\delta & \text{(follows led)}\\
0 & \text{otherwise} & \text{(slough)}
\end{cases}$$

Tier 2 dominates tier 1 dominates tier 0.

### 2.2 Within-Tier Rank (Double-High + Exceptions)

Define pip sum:

$$\text{pips}(d) = i + j \quad \text{for } d = (i,j)$$

Within a non-slough tier, the ordering is:

- doubles rank highest (rank 14) in a pip suit
- exception: when doubles form the called suit ($\kappa(\delta)=\mathcal{D}^\circ$), doubles rank by pip value (6-6 high)

Concretely:

$$\text{rank}_{\delta,\ell}(d)=\begin{cases}
p & \text{if } \kappa(\delta)=\mathcal{D}^\circ \land d=(p,p) \land \text{tier}_{\delta,\ell}(d)>0\\
14 & \text{if } d \in \mathcal{D}^\circ \land \text{tier}_{\delta,\ell}(d)>0\\
\text{pips}(d) & \text{if } \text{tier}_{\delta,\ell}(d)>0\\
0 & \text{otherwise}
\end{cases}$$

### 2.3 The Total Rank Function

Define the total ordering key:

$$\tau_{\delta,\ell}(d)=\big(\text{tier}_{\delta,\ell}(d)\ll 4\big)+\text{rank}_{\delta,\ell}(d)$$

All sloughs map to 0. This is safe because in any legal trick the lead is never a slough, so a slough tier cannot contain the winner.

### 2.4 The Dominance Relation

Fix $(\delta, \ell)$. Define dominance:

$$d_1 \succ_{\delta,\ell} d_2 \iff \tau_{\delta,\ell}(d_1) > \tau_{\delta,\ell}(d_2)$$

This is a total preorder (ties among off-suit losers).

### 2.5 Threat and Victim Sets

For domino $d$ in context $(\delta, \ell)$:

**Threat set** (dominoes that beat $d$):
$$T(d, \ell, \delta) = \{d' \in \mathcal{D} : d' \succ_{\delta,\ell} d\}$$

**Victim set** (dominoes that $d$ beats):
$$V(d, \ell, \delta) = \{d' \in \mathcal{D} : d \succ_{\delta,\ell} d'\}$$

**Peer set** (dominoes that tie—both off-suit losers):
$$P(d, \ell, \delta) = \{d' \in \mathcal{D} : \tau_{\delta,\ell}(d') = \tau_{\delta,\ell}(d)\}$$

### 2.6 The Power Profile

For domino $d$ under declaration $\delta$, define its **power profile**:

$$\mathrm{prof}_\delta(d) : \{0..7\} \to \mathbb{N} \times \mathbb{N}$$

$$\mathrm{prof}_\delta(d)(\ell) = \big(|T(d,\ell,\delta)|, |V(d,\ell,\delta)|\big)$$

This maps each possible led suit to the (threats, victims) pair.

### 2.7 Remaining-Aware Refinement

Given remaining dominoes $R \subseteq \mathcal{D}$:

$$T_R(d, \ell, \delta) = T(d, \ell, \delta) \cap R$$
$$V_R(d, \ell, \delta) = V(d, \ell, \delta) \cap R$$

As play progresses, threats diminish. Dominoes **promote** as higher-ranked dominoes are played.

### 2.8 Rank Algebra Summary

The rank algebra provides, for any context $(\delta, \ell, R)$:
- A total ordering on playable dominoes
- Threat/victim structure for each domino
- Static computation: $O(1)$ per query with precomputation

---

## §3. The Void Lattice

### 3.1 Void Patterns

A **void pattern** is a function recording known voids:

$$V : \mathcal{S} \times \{\text{suits}\} \to \{\top, \bot\}$$

- $V(s, \ell) = \top$: seat $s$ is known void in suit $\ell$
- $V(s, \ell) = \bot$: seat $s$ may have cards in suit $\ell$

### 3.2 Void Pattern Space

The space of void patterns:

$$\mathcal{V} = \{V : \mathcal{S} \times \{0..7\} \to \{\top, \bot\}\}$$

Theoretical maximum: $2^{32}$ patterns (4 seats × 8 suits).

Reachable patterns are far fewer due to constraints:
- Can't be void in all suits with dominoes remaining
- Certain void combinations are deal-impossible

### 3.3 The Void Lattice Structure

Define refinement ordering:

$$V_1 \le V_2 \iff \forall s, \ell: V_1(s,\ell) = \top \Rightarrow V_2(s,\ell) = \top$$

"$V_2$ has at least as many known voids as $V_1$."

This forms a lattice:
- Bottom $\bot_\mathcal{V}$: no voids known (all $\bot$)
- Top $\top_\mathcal{V}$: all possible voids known
- Meet $\wedge$: intersection of void knowledge
- Join $\vee$: union of void knowledge

### 3.4 Void Monotonicity

**Key property**: Voids only accumulate during play.

$$V_{\text{trick } n} \le V_{\text{trick } n+1}$$

Once void, always void. The void pattern traverses the lattice monotonically upward.

This is the **funnel property** at the void level.

### 3.5 Void Inference

We write the follow predicate as:

$$\mathsf{follows}(d,\ell,\delta)\iff d \in \hat{\sigma}_\ell^\delta$$

Voids are learned through play:

**Direct observation (perfect info)**: Seat $s$ has remaining domino $d$ such that $\mathsf{follows}(d,\ell,\delta)$.
- Confirms: $s$ is not currently void in the led suit $\ell$ (with respect to $\hat{\sigma}_\ell^\delta$).

**Void inference (legal play)**: Seat $s$ plays domino $d$ with $\neg\mathsf{follows}(d,\ell,\delta)$ when suit $\ell$ was led.
- Implies: $V(s, \ell) = \top$ (seat $s$ is void in the led suit, i.e. has no remaining domino in $\hat{\sigma}_\ell^\delta$).

This inference is **binary and certain**—not probabilistic.

### 3.6 Interrupt Capability

A void enables **interrupt**: playing off-suit **and still winning** by using the power suit (when power exists).

Define interrupt capability:

$$\text{canInterrupt}(s, \ell, V, R_s, \delta) = V(s, \ell) \land (\pi(\delta) \neq \bot) \land (R_s \cap \pi(\delta) \neq \varnothing)$$

Seat $s$ can interrupt suit $\ell$ if:
1. Void in $\ell$ (can't follow)
2. The declaration has power ($\pi(\delta)\neq\bot$)
3. Holds at least one power-suit domino (can over-rank)

### 3.7 The Interrupt Topology

For void pattern $V$, define the **interrupt graph** $G_V^\delta$:

- Nodes: seats $\mathcal{S}$
- Edges: directed, labeled by suit

$$s_1 \xrightarrow{\ell} s_2 \in G_V^\delta \iff \text{canInterrupt}(s_2, \ell, V, H_{s_2}, \delta) \land s_1 \text{ would lead } \ell$$

This graph shows who can steal control from whom under which leads.

### 3.8 Safe and Contested Suits

For a seat $s$ with lead token:

**Safe suits** (no interrupt possible):
$$\text{Safe}(s, V, \delta) = \{\ell : \forall s' \neq s, \neg\text{canInterrupt}(s', \ell, V, H_{s'}, \delta)\}$$

**Contested suits** (interrupt threat exists):
$$\text{Contested}(s, V, \delta) = \{\ell : \exists s' \neq s, \text{canInterrupt}(s', \ell, V, H_{s'}, \delta)\}$$

A key strategic decision: lead safe to guarantee win, or lead contested to probe/bait?

### 3.9 Void Lattice Summary

The void lattice provides:
- A compact representation of "who can respond to what"
- Monotonic traversal (funnel structure)
- Interrupt topology (control flow implications)
- Deterministic inference from observed play

---

## §4. The Control System

### 4.1 The Lead Token

Define the **lead token** as a resource held by exactly one seat:

$$\text{Leader} : \{\text{game states}\} \to \mathcal{S}$$

The token determines who names the comparison function (led suit) for the next trick.

### 4.2 Token Dynamics

The lead token obeys:

**Initial condition**: Leader$(t=0)$ determined by bidding winner.

**Transition rule**: 
$$\text{Leader}(t+1) = \text{TrickWinner}(t)$$

Where TrickWinner is determined by the rank algebra applied to the four played dominoes.

### 4.3 Token as Team Resource

The token is partially shared within a team:

$$\text{TeamControl}(t) = \text{team}(\text{Leader}(t))$$

If my partner holds the token, my team still has agenda control—partner can lead to my strength.

### 4.4 Token Acquisition

The token is acquired by winning a trick. Ways to win:

1. **Rank dominance**: Play highest in led suit
2. **Trump intrusion**: Interrupt with trump when void in led suit (only when $\pi(\delta)\neq\bot$)
3. **Default**: All others play off-suit, any on-suit card wins

### 4.5 Token Value

The value of holding the token depends on:

**Hand strength**: Can you lead suits where you're dominant?
$$\text{LeadStrength}(s, R_s, \delta) = \left|\left\{\ell : \max_{d \in R_s \cap \hat{\sigma}_\ell^\delta} \tau_{\delta,\ell}(d) = \max_{d' \in R \cap \hat{\sigma}_\ell^\delta} \tau_{\delta,\ell}(d')\right\}\right|$$

(Count of suits where you hold the current high card.)

**Interrupt immunity**: Can opponents steal the token back?
$$\text{InterruptRisk}(s, V, \delta) = |\text{Contested}(s, V, \delta)|$$

**Partner coordination**: Can you pass to partner advantageously?

### 4.6 The Control Graph

For game state $\sigma$, define the **control graph** $C_\sigma$:

- Nodes: $\mathcal{S} \times \{0..7\}$ (seat × suit pairs)
- Edge $(s_1, \ell_1) \to (s_2, \ell_2)$: 
  - If $s_1$ leads $\ell_1$, then $s_2$ wins and would lead $\ell_2$

This graph encodes all possible control flow paths.

### 4.7 Tempo

**Tempo** measures the rate of token movement:

$$\text{Tempo}(T_i, \text{tricks } n..m) = \frac{|\{k \in [n,m] : \text{team}(\text{Leader}(k)) = i\}|}{m - n + 1}$$

High tempo = your team leads most tricks = more agenda control.

### 4.8 The Ducking Principle

Sometimes optimal play is to **lose** a trick intentionally:

- Preserve high cards for later
- Let partner win (better lead position)
- Exhaust opponent's trump
- Set up endgame squeeze

Ducking trades immediate token acquisition for future positional value.

### 4.9 Control System Summary

The control system provides:
- A transferable resource (lead token)
- Deterministic transition rules
- Team-level aggregation
- Tempo as a measurable quantity
- Strategic ducking decisions

---

## §5. Positional Power

### 5.1 Within-Trick Position

In each trick, seats play in order. Position $p \in \{1,2,3,4\}$:

| Position | Role | Information | Agency |
|----------|------|-------------|--------|
| 1 (Lead) | Names the game | None about this trick | Maximum (suit choice) |
| 2 | First response | Sees lead | Signal/force/duck |
| 3 | Pivot | Sees lead + response | React to dynamic |
| 4 | Finisher | Sees all three | Perfect local info |

### 5.2 Information Sets

For position $p$, the information set is:

$$I_p = \{d_1, \ldots, d_{p-1}\}$$

Position 1: $I_1 = \varnothing$
Position 4: $I_4 = \{d_1, d_2, d_3\}$

### 5.3 Positional Power Function

Define positional modifier:

$$\rho(p, d, I_p, \delta, \ell) \in \mathbb{R}$$

This modifies raw domino rank by positional context:
- Position 1 with a strong suit: high $\rho$ (can name favorable game)
- Position 4 with exact knowledge: $\rho$ deterministic (knows if $d$ wins)
- Position 2 with count: $\rho$ includes signaling value

### 5.4 Stake Manipulation

Position 2 and 3 can alter trick stakes:

**Raising stakes**: Play count, encouraging desperate measures
$$\text{stake}(\{d_1, d_2\}) > \text{stake}(\{d_1\}) \implies \text{increased pressure on positions 3, 4}$$

**Signaling partner**: Play high (strength) or low (weakness) without count

**Baiting trump**: Play count to draw out opponent's trump

### 5.5 Position-Control Interaction

The lead position for trick $n+1$ is determined by winning trick $n$.

This creates a **two-level positional structure**:
- Within-trick: positions 1-4
- Across-tricks: who gets position 1

$$\text{PositionalValue}(s) = \sum_{t} \mathbb{1}[\text{Leader}(t) = s] \cdot \text{LeadValue}(s, t)$$

### 5.6 Positional Summary

Position adds:
- Information asymmetry within tricks
- Agency asymmetry (leader has most)
- Stake manipulation opportunities
- Cross-trick linkage through lead inheritance

---

## §6. The Unified Framework

### 6.1 The State Space

In perfect-information play (the `solver2` setting), a sufficient play-phase state is the one in `docs/theory/PLAY_PHASE_ALGEBRA.md`:

$$\sigma = (R_0, R_1, R_2, R_3, L, k, p_0, p_1, p_2)$$

Where:

- $R_s$: remaining dominoes for seat $s$ (or remaining local indices under a fixed deal gauge)
- $L$: current leader (token holder)
- $k$: plays so far in the current trick ($0..3$)
- $p_0,p_1,p_2$: the first 3 plays of the current trick (unset when $k<j$)

Derived (not stored in `solver2`'s packed state):

- the void pattern $V$ (actual voids in each effective suit, given $(R_s)$ and $\delta$)
- the trick number $t$ (from the total number of plays made)

### 6.2 The Three-Axis Decomposition Conjecture

**Conjecture**: Game value decomposes approximately as:

$$\Phi(\sigma) = \underbrace{\Phi_{\text{rank}}(\sigma)}_{\text{material}} + \underbrace{\Phi_{\text{void}}(\sigma)}_{\text{interrupt structure}} + \underbrace{\Phi_{\text{control}}(\sigma)}_{\text{tempo/position}} + \epsilon(\sigma)$$

Where:
- $\Phi_{\text{rank}}$: aggregate rank strength (threat/victim balance)
- $\Phi_{\text{void}}$: interrupt topology value
- $\Phi_{\text{control}}$: lead token + tempo value
- $\epsilon$: interaction/correction term

If $\epsilon$ is small, the game is approximately separable.

### 6.3 Candidate Component Functions

**Rank component**:
$$\Phi_{\text{rank}}(\sigma) = \sum_{s \in T_0} \sum_{d \in R_s} w_r(d, \delta) - \sum_{s \in T_1} \sum_{d \in R_s} w_r(d, \delta)$$

Where $w_r(d, \delta)$ is a rank-based weight (e.g., average $|V(d, \ell, \delta)|$ across suits).

**Void component**:
$$\Phi_{\text{void}}(\sigma) = \sum_{s \in T_0} \text{InterruptPower}(s, V) - \sum_{s \in T_1} \text{InterruptPower}(s, V)$$

Where InterruptPower counts interrupt capabilities weighted by suit value.

**Control component**:
$$\Phi_{\text{control}}(\sigma) = \alpha \cdot \mathbb{1}[\text{team}(L) = 0] + \beta \cdot \text{LeadStrength}(L)$$

Where $\alpha$ is the base value of holding the token, $\beta$ weights lead strength.

### 6.4 The Potential Function Hypothesis

An exact identity of the form:

$$\Phi(\sigma) - \Phi(\sigma') = r(\sigma \to \sigma')$$

for *all* legal transitions would make cumulative reward essentially path-dependent only through the endpoint, which is incompatible with nontrivial trick-taking play.

The meaningful analogue is: find a simple $\Phi$ that approximates the true minimax value function $V$ (low Bellman error), so that greedy $\Phi$-improvement is a strong move-ordering / heuristic.

### 6.5 The Weaker Conjecture

**Weak hypothesis**: The decomposition holds approximately:

$$|V(\sigma) - \Phi(\sigma)| < \epsilon$$

For some small $\epsilon$ (e.g., < 2 points on average).

This would enable:
- Fast heuristic evaluation
- Effective move ordering
- Pruning without missing optimal plays

---

## §7. The Interaction Terms

### 7.1 Rank-Void Interaction

Rank strength is mediated by voids:

- High trump is strong, but if opponents are trump-void, they can't be interrupted
- Suit length matters: more cards = less likely to void = can follow longer

Let $\ell_d = \mathsf{ledSuit}(d,\delta)$ be the suit induced by leading domino $d$ (as in `docs/theory/SUIT_ALGEBRA.md` §10). Then:

$$\text{RankVoidInteraction}(d, V, \delta) = \sum_{s \neq \text{holder}(d)} \mathbb{1}[V(s, \ell_d)] \cdot f(\tau_{\delta,\ell_d}(d))$$

### 7.2 Void-Control Interaction

Voids determine interrupt topology, which affects control flow:

- More voids = more volatile control (token changes hands unpredictably)
- Specific void patterns create "safe harbors" (suits with guaranteed token retention)

$$\text{VoidControlInteraction}(V, L) = |\text{Safe}(L, V)| - |\text{Contested}(L, V)|$$

### 7.3 Rank-Control Interaction

Holding the token amplifies rank strength:

- Leader uses their strongest suit
- Non-leader must respond to leader's choice

$$\text{RankControlInteraction}(L, R_L) = \max_\ell \sum_{d \in R_L \cap \hat{\sigma}_\ell^\delta} \tau_{\delta,\ell}(d)$$

### 7.4 The Full Tensor

The complete interaction might be a 3-way tensor:

$$\Pi_{r,v,c} = \text{power contribution for rank } r, \text{ void pattern } v, \text{ control state } c$$

Game value is then a contraction:

$$\Phi(\sigma) = \sum_{r,v,c} \Pi_{r,v,c} \cdot \phi_r(\sigma) \cdot \phi_v(\sigma) \cdot \phi_c(\sigma)$$

Where $\phi_r, \phi_v, \phi_c$ are indicator/feature functions for each axis.

---

## §8. Experimental Validation

### 8.1 Phase 1: Void Lattice Alone

Test whether void pattern predicts value:

```python
def test_void_sufficiency(logs):
    # Group by (void_pattern, level, declaration)
    # Compute within-group variance
    # Compare to total variance
    return variance_ratio
```

**Success criterion**: Variance ratio < 0.1

### 8.2 Phase 2: Rank Features Alone

Test whether rank-based features predict value:

```python
def compute_rank_features(state, context):
    features = []
    for seat in range(4):
        hand = get_remaining(state, seat)
        for suit in range(8):
            # Count dominoes in suit
            # Sum of ranks in suit
            # Max rank in suit
            features.extend([count, rank_sum, max_rank])
    return features
```

**Success criterion**: Linear model R² > 0.9

### 8.3 Phase 3: Control Features Alone

Test whether control state predicts value:

```python
def compute_control_features(state, context):
    leader = get_leader(state)
    leader_team = leader % 2
    # Lead strength
    # Safe suit count
    # Tempo history (if tracked)
    return [leader_team, lead_strength, safe_suits]
```

### 8.4 Phase 4: Combined Model

Test the full decomposition:

```python
def compute_all_features(state, context):
    return (
        compute_rank_features(state, context) +
        compute_void_features(state, context) +
        compute_control_features(state, context)
    )
```

**Key question**: Does the combined model significantly outperform individual components?

If combined R² ≈ max(individual R²), one axis dominates.
If combined R² >> max(individual R²), the interaction matters.

### 8.5 Phase 5: Tensor Decomposition

If interactions matter, test whether the tensor has low rank:

```python
def test_tensor_rank(value_tensor):
    # Apply tensor decomposition (CP, Tucker, etc.)
    # Check reconstruction error vs. rank
    return optimal_rank, reconstruction_error
```

Low tensor rank → separable structure → efficient computation.

---

## §9. Algorithmic Implications

### 9.1 If Void Lattice Suffices

Build a lookup table:

```
Key: (void_pattern, level, declaration, leader_team)
Value: minimax value (or optimal move)
```

Estimated size: ~100M entries, few GB storage.

**Solve any position in O(1) lookups.**

### 9.2 If Linear Decomposition Holds

Compute value directly:

```python
def evaluate(state):
    return (
        w_rank @ rank_features(state) +
        w_void @ void_features(state) +
        w_control @ control_features(state)
    )
```

**Solve any position in O(1) arithmetic.**

### 9.3 If Tensor Structure Holds

Use tensor contraction:

```python
def evaluate(state):
    r = rank_embedding(state)
    v = void_embedding(state)
    c = control_embedding(state)
    return contract(Pi, r, v, c)
```

**Solve any position in O(k³) where k is tensor rank.**

### 9.4 Move Ordering Heuristic

Even if not exact, the decomposition provides move ordering:

```python
def order_moves(state, moves):
    return sorted(moves, key=lambda m: heuristic_value(apply(state, m)), reverse=True)
```

Better move ordering → more alpha-beta pruning → faster exact solve.

### 9.5 Bidding Application

The decomposition directly enables bidding evaluation:

```python
def evaluate_bid(hand, declaration):
    # Compute expected void pattern trajectory
    # Compute expected rank distribution
    # Compute expected control dynamics
    # Integrate to get expected value
    return expected_points
```

---

## §10. Code Correspondence

### 10.1 Rank Algebra Implementation

Location: `scripts/solver2/tables.py`, `scripts/solver2/context.py`

```python
# Core predicates (computed functions)
trick_rank(domino_id, led_suit, decl_id)  # τ(d,ℓ,δ) ordering key
can_follow(domino_id, led_suit, decl_id)  # d ∈ \hat{σ}_ℓ^δ

# Precomputed per-seed tables
SeedContext.LOCAL_FOLLOW   # follow masks for each (leader, lead_local, follower_offset)
SeedContext.TRICK_WINNER   # winner offset for each (leader, p0, p1, p2, p3)
SeedContext.TRICK_POINTS   # points for each completed trick
SeedContext.TRICK_REWARD   # signed reward (+points if team0 wins else -points)
```

### 10.2 Void Lattice Implementation

Location: `scripts/solver2/void_analysis.py` (to be created)

```python
def extract_void_pattern(state: int, context: SeedContext) -> int:
    """Extract 32-bit actual-void pattern (4 seats × 8 led-suits)."""
    pattern = 0
    for seat in range(4):
        remaining = extract_remaining(state, seat)
        for led_suit in range(8):
            follow_mask = 0
            for local_idx in range(7):
                domino_id = int(context.L[seat, local_idx])
                if can_follow(domino_id, led_suit, context.decl_id):
                    follow_mask |= (1 << local_idx)
            if (remaining & follow_mask) == 0:
                pattern |= (1 << (seat * 8 + led_suit))
    return pattern

def void_lattice_order(v1: int, v2: int) -> bool:
    """Test if v1 ≤ v2 in the void lattice (v2 has more voids)."""
    return (v1 & v2) == v1
```

### 10.3 Control System Implementation

Location: `scripts/solver2/state.py`

```python
def extract_leader(state: int) -> int:
    """Extract current leader from packed state."""
    return (state >> 28) & 0x3

def compute_lead_strength(state: int, context: SeedContext) -> int:
    """Count suits where leader holds the high card."""
    leader = extract_leader(state)
    remaining = extract_remaining(state, leader)
    strength = 0
    for suit in range(7):
        if has_suit_high(remaining, suit, state, context):
            strength += 1
    return strength
```

### 10.4 Feature Extraction

Location: `scripts/solver2/features.py` (to be created)

```python
@dataclass
class StateFeatures:
    # Rank features
    rank_sums: Tuple[int, int, int, int]  # per seat
    max_ranks: Tuple[int, int, int, int]  # per seat
    
    # Void features
    void_pattern: int
    void_counts: Tuple[int, int, int, int]  # voids per seat
    
    # Control features
    leader: int
    leader_team: int
    lead_strength: int
    safe_suit_count: int
    
    # Derived
    team_rank_differential: int
    interrupt_balance: int

def extract_features(state: int, context: SeedContext) -> StateFeatures:
    """Extract all features from packed state."""
    ...
```

---

## §11. Open Questions

1. **Counting domino treatment**: Do counters need a separate axis, or do they fold into rank?

2. **Declaration dependence**: Does the decomposition quality vary by declaration? (Trump vs. no-trump)

3. **Game stage effects**: Does the decomposition hold better early or late?

4. **Partner coordination**: How does team structure interact with the three axes?

5. **The correction term**: If $\epsilon$ is non-negligible, what is its structure?

6. **Computational complexity**: What is the complexity class of exact 42 evaluation? Is it in P?

---

## §12. Summary

The three-axis framework posits that Texas 42 game value decomposes into:

| Axis | What It Captures | Key Object |
|------|------------------|------------|
| Rank | Who wins comparisons | Dominance relation $\succ_{\delta,\ell}$ |
| Void | Who participates in comparisons | Void lattice $\mathcal{V}$ |
| Control | Who chooses the comparison | Lead token dynamics |

The experimental program tests whether this decomposition holds exactly, approximately, or not at all.

**If exact**: Closed-form solution, O(1) evaluation.
**If approximate**: Powerful heuristic, fast pruning.
**If neither**: The game's complexity is irreducibly entangled.

The structure of 42—built on $K_7$, exactly 42 points, 7 tricks, the suit covering—strongly suggests that exploitable structure exists. This framework provides the language to find and prove it.

---

## Appendix A: Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\mathcal{D}$ | Domino set (28 elements) |
| $\sigma_p$ | Natural suit for pip $p$ |
| $\kappa(\delta)$ | Called set (dominoes summoned to suit 7) |
| $\hat{\sigma}_p^\delta$ | Effective (followable) suit after calling |
| $\pi(\delta)$ | Power suit (or $\bot$) |
| $\delta$ | Declaration |
| $\ell$ | Led suit (0..7, with 7=called) |
| $\tau_{\delta,\ell}$ | Trick ordering key |
| $\succ_{\delta,\ell}$ | Dominance relation |
| $T(d,\ell,\delta)$ | Threat set |
| $V(d,\ell,\delta)$ | Victim set |
| $\mathrm{prof}_\delta(d)$ | Threat/victim profile across led suits |
| $V$ | Void pattern |
| $\mathcal{V}$ | Void lattice |
| $L$ | Leader (token holder) |
| $\Phi$ | Potential function |
| $\sigma$ | Game state |
| $T_0, T_1$ | Teams |

## Appendix B: Key Conjectures

**Conjecture 1 (Void Sufficiency)**: States with identical void patterns have value variance < 10% of total variance.

**Conjecture 2 (Linear Decomposition)**: A linear model on (rank, void, control) features achieves R² > 0.95.

**Conjecture 3 (Low Bellman Error)**: There exists a simple $\Phi$ such that the Bellman residual is small:

$$\Phi(\sigma) \approx
\begin{cases}
\max_a \big(r(\sigma,a)+\Phi(\sigma')\big) & \text{Team 0 to act}\\
\min_a \big(r(\sigma,a)+\Phi(\sigma')\big) & \text{Team 1 to act}
\end{cases}$$

**Conjecture 4 (Low Tensor Rank)**: The interaction tensor $\Pi_{r,v,c}$ has rank < 10.

---

## Appendix C: Experimental Results (2025-12-28)

Validation performed using `scripts/solver2/validate_conjecture.py` on minimax-solved positions from `data/solver2/`.

### Setup

- **Data**: 50,000 states sampled from 4 parquet files (seeds 0-1, declarations blanks/ones/fives)
- **Features extracted** (15 total):
  - Rank axis (8): rank differential + 7 max-rank differentials per suit
  - Void axis (5): void differential + 4 void counts per seat
  - Control axis (2): leader team + level (normalized)
- **Models**: Linear regression (closed-form) and MLP (128-128-1, 50 epochs)
- **Device**: CUDA GPU

### Results

| Phase | Description | R² |
|-------|-------------|-----|
| 1 | Void features only | 5.7% |
| 2 | Rank features only | 14.5% |
| 3 | Control features only | 8.2% |
| 4 | Linear combined | 19.2% |
| 5 | MLP combined | 22.8% |

### Conjecture Status

| Conjecture | Criterion | Observed | Status |
|------------|-----------|----------|--------|
| Void Sufficiency | within-group variance < 10% | ~95% residual | **REFUTED** |
| Linear Decomposition | R² > 0.95 | R² = 0.19 | **REFUTED** |

### Analysis

1. **Rank dominates**: The rank axis (14.5%) contributes more than void (5.7%) or control (8.2%) individually.

2. **Synergy exists but weak**: Combining axes adds +4.7% beyond the best individual axis.

3. **Nonlinearity is minor**: MLP gains only +3.6% over linear model.

4. **77% unexplained**: The three-axis framework captures only ~23% of minimax value variance.

### Interpretation

The simple aggregate features (team differentials, counts) miss critical information:

- **Specific card positions**: Which exact dominoes are in each hand matters more than aggregate counts
- **Trick context**: The current trick state (p0, p1, p2 plays) affects who can win
- **Card interactions**: Specific combinations (e.g., holding both 6-5 and 5-5 in fives) have non-additive value

### Implications

1. **Not a simple decomposition**: Game value does not factor cleanly into rank + void + control.

2. **Features need enrichment**: To reach R² > 0.9, likely need:
   - Per-domino presence indicators (28 binary features per seat)
   - Trick-in-progress context
   - Suit-specific control analysis

3. **MLP is promising**: Even with weak features, MLP extracts some nonlinear signal. With richer features, neural evaluation may approach minimax quality.

4. **Heuristic value**: The 23% explained variance may still provide useful move ordering, even if not sufficient for accurate evaluation.
