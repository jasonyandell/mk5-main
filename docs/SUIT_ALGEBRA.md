# Texas 42: Algebraic Suit Structure

> **Scope.** This document defines the algebraic structure for **trick resolution**: suit membership, following, and winner determination. It covers the four standard declarations (pip-trump, doubles-trump, nello, notrump). Point scoring (counters), special contracts (splash, plunge, sevens), and strategic objectives (nello's "lose all tricks") are handled elsewhere—see `docs/rules.md` for the complete game specification.

## §1. The Domino Set

Let $\mathbb{P} = \{0, 1, 2, 3, 4, 5, 6\}$ be the set of pip values.

The **domino set** $\mathcal{D}$ is the upper triangle of $\mathbb{P} \times \mathbb{P}$:

$$\mathcal{D} = \{(i, j) \in \mathbb{P}^2 : i \leq j\}$$

with $|\mathcal{D}| = 28$.

We define the **pip membership** predicate:

$$p \in d \iff d = (p, \_) \lor d = (\_, p)$$

and partition $\mathcal{D}$ into **doubles** and **mixed**:

$$\mathcal{D}^\circ = \{(p, p) : p \in \mathbb{P}\} \qquad \mathcal{D}^\times = \mathcal{D} \setminus \mathcal{D}^\circ$$

---

## §2. The Natural Covering

For each $p \in \mathbb{P}$, the **natural suit** is:

$$\sigma_p = \{d \in \mathcal{D} : p \in d\}$$

This yields a **covering** $\Sigma = \{\sigma_0, \sigma_1, \ldots, \sigma_6\}$ with properties:

| Property | Statement |
|----------|-----------|
| Cardinality | $\forall p: |\sigma_p| = 7$ |
| Double membership | $d \in \mathcal{D}^\circ \implies |\{p : d \in \sigma_p\}| = 1$ |
| Mixed membership | $d \in \mathcal{D}^\times \implies |\{p : d \in \sigma_p\}| = 2$ |
| Pairwise intersection | $p \neq q \implies \sigma_p \cap \sigma_q = \{(p, q)\}$ |

This is **not** a partition—each mixed domino inhabits two suits simultaneously.

---

## §3. Declarations and the Called Suit

A **declaration** $\delta$ specifies how dominoes are called into the 8th suit.

$$\delta \in \Delta = \mathbb{P} \cup \{\mathsf{doubles}, \mathsf{nello}, \mathsf{notrump}\}$$

The **called set** function $\kappa : \Delta \to \mathcal{P}(\mathcal{D})$:

$$\kappa(\delta) = \begin{cases}
\sigma_p & \text{if } \delta = p \in \mathbb{P} \\[4pt]
\mathcal{D}^\circ & \text{if } \delta \in \{\mathsf{doubles}, \mathsf{nello}\} \\[4pt]
\varnothing & \text{if } \delta = \mathsf{notrump}
\end{cases}$$

The **called suit** is $\sigma_7^\delta = \kappa(\delta)$.

---

## §4. The Effective Suit Structure

Under declaration $\delta$, the covering transforms. Define the **effective suit**:

$$\hat{\sigma}_p^\delta = \sigma_p \setminus \kappa(\delta)$$

The full **effective structure** is:

$$\hat{\Sigma}^\delta = \{\hat{\sigma}_0^\delta, \hat{\sigma}_1^\delta, \ldots, \hat{\sigma}_6^\delta, \sigma_7^\delta\}$$

### Key Theorem: Effective Partition

Under any declaration $\delta$, each domino belongs to **exactly one or two** effective suits:

$$\forall d \in \mathcal{D}: \quad 1 \leq |\{s \in \hat{\Sigma}^\delta : d \in s\}| \leq 2$$

with equality to 1 iff $d \in \kappa(\delta) \lor d \in \mathcal{D}^\circ$.

---

## §5. Following: The Membership Predicate

When suit $\ell$ is led under declaration $\delta$, a domino $d$ **can follow** iff:

$$\mathsf{follows}(d, \ell, \delta) \iff d \in \hat{\sigma}_\ell^\delta$$

Equivalently:

$$\mathsf{follows}(d, \ell, \delta) \iff (\ell \in d) \land (d \notin \kappa(\delta))$$

A domino in the called suit **cannot** follow an off-suit lead—it has been called away.

---

## §6. Power: The Dominance Relation

Declarations also specify **power**—which suit dominates others.

Define the **power function** $\pi : \Delta \to \mathcal{P}(\mathcal{D}) \cup \{\bot\}$:

$$\pi(\delta) = \begin{cases}
\sigma_7^\delta & \text{if } \delta \in \mathbb{P} \cup \{\mathsf{doubles}\} \\[4pt]
\bot & \text{if } \delta \in \{\mathsf{nello}, \mathsf{notrump}\}
\end{cases}$$

The suit with power beats all other suits. When $\pi(\delta) = \bot$, no suit has power—Tier 1 is empty, so the highest follower wins. (In nello, the strategic objective is to *lose* tricks, but the trick-taking mechanics are unchanged.)

---

## §7. Rank Within Suit

Within any suit (trump or led), **the double ranks highest**. This is the fundamental ranking rule of Texas 42.

For the **called suit** specifically:

For $\delta = p \in \mathbb{P}$ (pip trump):

$$\rho(d, p) = \begin{cases}
14 & \text{if } d = (p, p) \\[4pt]
\mathsf{sum}(d) & \text{otherwise}
\end{cases}$$

For $\delta = \mathsf{doubles}$:

$$\rho((p,p), \mathsf{doubles}) = p$$

When doubles are trump, rank is by pip value (6-6 beats 5-5 beats ... beats 0-0).

The general principle—double beats all, then by pip sum—applies equally to Tier 1 (following the led suit). See §8 for the complete ranking function.

---

## §8. Trick Ranking: The Three-Tier Function

Given declaration $\delta$ and led suit $\ell$, we define the **trick ranking** $\tau : \mathcal{D} \times \{0..7\} \times \Delta \to \mathbb{N}$:

$$\tau(d, \ell, \delta) = (\mathsf{tier}(d, \ell, \delta) \ll 4) + \mathsf{rank}(d, \ell, \delta)$$

where $\ll$ denotes left bit-shift, and:

$$\mathsf{tier}(d, \ell, \delta) = \begin{cases}
2 & \text{if } d \in \pi(\delta) & \text{(Tier 2: trump)} \\[4pt]
1 & \text{if } d \in \hat{\sigma}_\ell^\delta & \text{(Tier 1: follows led)} \\[4pt]
0 & \text{otherwise} & \text{(Tier 0: slough)}
\end{cases}$$

$$\mathsf{rank}(d, \ell, \delta) = \begin{cases}
p & \text{if } \delta = \mathsf{doubles} \land d = (p,p) & \text{(doubles-trump: by pip)} \\[4pt]
14 & \text{if } d \in \mathcal{D}^\circ \land \mathsf{tier}(d, \ell, \delta) > 0 & \text{(other double in play)} \\[4pt]
\mathsf{sum}(d) & \text{if } \mathsf{tier}(d, \ell, \delta) > 0 & \text{(non-double in play)} \\[4pt]
0 & \text{otherwise} & \text{(slough)}
\end{cases}$$

*Note: Conditions are evaluated top-to-bottom; the first match applies.*

**Key rules:** 
- The double always ranks highest in its suit, whether trump or not
- Exception: when doubles are trump, they rank by pip value (6-6 highest)
- In Tier 2 (pip-trump), the only double is the trump double
- In Tier 1, the only double that can follow is $(\ell, \ell)$—other doubles either are trump or don't contain $\ell$

This yields a 6-bit encoding:

| Tier | Binary Pattern | Decimal Range |
|------|----------------|---------------|
| 2 (trump) | `10_xxxx` | 32–46 |
| 1 (follows) | `01_xxxx` | 16–30 |
| 0 (slough) | `00_0000` | 0 |

**Extraction:**
$$\mathsf{tier}(\tau) = \tau \gg 4 \qquad \mathsf{rank}(\tau) = \tau \land \mathsf{0xF}$$

**Note on Tier 0:** Sloughs are unordered—all map to 0. This is consistent because the lead domino always belongs to Tier 1 or Tier 2 (it either follows itself or is trump), so Tier 0 is never the highest non-empty tier. Slough ordering is strategically irrelevant to trick resolution.

### Architectural Note: Configuration vs Context

This function decomposes into two dependencies:

| Aspect | Depends On | Known When |
|--------|-----------|------------|
| Tier 2 membership | $\delta$ only | Hand starts |
| Tier 1 membership | $\delta$ and $\ell$ | Trick starts |
| Tier 0 | default | Always |

The called set $\kappa(\delta)$ and power $\pi(\delta)$ are **configuration-dependent**—computable from $\delta$ alone, suitable for lookup tables.

The "follows led suit" predicate $d \in \hat{\sigma}_\ell^\delta$ is **context-dependent**—requires knowing $\ell$, which isn't determined until the first card is played.

---

## §9. Theorem: Unique Winner

**Claim:** For any legal trick $T = \{d_1, d_2, d_3, d_4\}$ under declaration $\delta$ with led suit $\ell$:

$$\exists! \, d^* \in T : \forall d \in T, \; \tau(d^*, \ell, \delta) \geq \tau(d, \ell, \delta)$$

*Proof.* We show that the highest non-empty tier always contains a unique maximum.

### Lemma 9.0: Tier 0 Is Never Highest

The lead domino $d_\ell$ always satisfies either:
- $d_\ell \in \pi(\delta)$ (it's trump → Tier 2), or  
- $d_\ell \in \hat{\sigma}_\ell^\delta$ (it follows itself → Tier 1)

Therefore at least one domino is in Tier 1 or above. Tier 0 can never contain the winner. $\square$

### Lemma 9.1: Pip Sums are Injective Within Follow-Sets

Let $F \subseteq \mathcal{D}$ be any set of dominoes that all contain a common pip $p$:

$$F \subseteq \sigma_p = \{d : p \in d\}$$

**Claim:** $\mathsf{sum}$ is injective on $F$.

*Proof.* Each $d \in \sigma_p$ has the form $(p, k)$ for some $k \in \mathbb{P}$. Thus $\mathsf{sum}(d) = p + k$. Since the map $k \mapsto p + k$ is injective and each domino in $\sigma_p$ has a unique "other pip" $k$, no two dominoes in $\sigma_p$ share the same sum. $\square$

### Lemma 9.2: Tier 2 Has No Collisions

For pip-trump $\delta = t \in \mathbb{P}$:

The called set is $\kappa(t) = \sigma_t$. Non-double members have distinct sums by Lemma 9.1. The double $(t,t)$ gets rank 14, which exceeds $\max_{d \in \sigma_t} \mathsf{sum}(d) = t + 6 \leq 12$. Therefore ranks are injective on $\kappa(t)$. $\square$

For doubles-trump $\delta = \mathsf{doubles}$:

The called set is $\kappa(\mathsf{doubles}) = \mathcal{D}^\circ = \{(0,0), (1,1), \ldots, (6,6)\}$.

All members are doubles, so all get rank 14... but wait, this creates a collision!

**Resolution:** For doubles-trump, we use pip value as rank instead:

$$\rho((p,p), \mathsf{doubles}) = p$$

Their ranks are $\{0, 1, 2, 3, 4, 5, 6\}$—all distinct. $\square$

### Lemma 9.3: Tier 1 Has No Collisions

The effective suit $\hat{\sigma}_\ell^\delta = \sigma_\ell \setminus \kappa(\delta)$ is a subset of $\sigma_\ell$.

The only double in $\hat{\sigma}_\ell^\delta$ (if any) is $(\ell, \ell)$—and only if it wasn't called to trump. This double gets rank 14.

All non-double members have distinct sums by Lemma 9.1, with $\max \mathsf{sum} = \ell + 6 \leq 12 < 14$.

Therefore ranks are injective on $\hat{\sigma}_\ell^\delta$. $\square$

### Lemma 9.4: Tiers Are Strictly Separated

The tier ranges are:
- Tier 2: $32 + \mathsf{rank} \in [32, 46]$ (rank 0–14)
- Tier 1: $16 + \mathsf{rank} \in [16, 30]$ (rank 0–14)
- Tier 0: $\{0\}$

These intervals are pairwise disjoint: $[32,46] \cap [16,30] = \varnothing$ and both exclude 0. $\square$

### Completion of Proof

By Lemma 9.0, the winner comes from Tier 1 or Tier 2. By Lemma 9.4, the maximum comes from the highest non-empty tier. By Lemmas 9.2–9.3, that tier has no internal collisions.

Therefore $\arg\max$ is unique. $\square$

---

## §10. The Complete Decision Function

Given a trick with lead domino $d_\ell$ under declaration $\delta$:

**Step 1.** Determine led suit:
$$\ell = \begin{cases}
7 & \text{if } d_\ell \in \kappa(\delta) \\[4pt]
\max(d_\ell) & \text{otherwise}
\end{cases}$$

> **Rule: "Big End Up."** In Texas 42, mixed dominoes always lead on their higher pip. A player leading 3-5 leads the fives suit, not the threes. The only exception is when the domino is trump—then it leads suit 7 regardless of pip values. This is not a strategic choice; it is a fixed rule of the game.

**Step 2.** For each subsequent domino $d$, legal play requires:
$$\mathsf{legal}(d, \ell, \delta, H) \iff \mathsf{follows}(d, \ell, \delta) \lor \neg\exists h \in H: \mathsf{follows}(h, \ell, \delta)$$

**Step 3.** Winner is determined by:
$$\mathsf{winner} = \underset{d \in \text{trick}}{\arg\max} \; \tau(d, \ell, \delta)$$

By Theorem §9, this is well-defined.

---

## §11. The Symmetry Group

The symmetric group $S_7$ acts on $\mathbb{P}$ and induces automorphisms of $\mathcal{D}$:

$$\phi_\tau(i, j) = (\tau(i), \tau(j)) \quad \text{for } \tau \in S_7$$

All pip-trump declarations are **isomorphic** under this action:

$$\hat{\Sigma}^p \cong \hat{\Sigma}^q \quad \forall p, q \in \mathbb{P}$$

There is essentially **one** pip-trump structure, instantiated seven ways.

---

## §12. Summary: The 8-Suit Model

| Suit Index | Name | Contents under $\delta$ |
|------------|------|-------------------------|
| $\sigma_0$ | Blanks | $\hat{\sigma}_0^\delta = \sigma_0 \setminus \kappa(\delta)$ |
| $\sigma_1$ | Aces | $\hat{\sigma}_1^\delta = \sigma_1 \setminus \kappa(\delta)$ |
| $\sigma_2$ | Deuces | $\hat{\sigma}_2^\delta = \sigma_2 \setminus \kappa(\delta)$ |
| $\sigma_3$ | Treys | $\hat{\sigma}_3^\delta = \sigma_3 \setminus \kappa(\delta)$ |
| $\sigma_4$ | Fours | $\hat{\sigma}_4^\delta = \sigma_4 \setminus \kappa(\delta)$ |
| $\sigma_5$ | Fives | $\hat{\sigma}_5^\delta = \sigma_5 \setminus \kappa(\delta)$ |
| $\sigma_6$ | Sixes | $\hat{\sigma}_6^\delta = \sigma_6 \setminus \kappa(\delta)$ |
| $\sigma_7$ | **Called** | $\kappa(\delta)$ |

The word **called** reflects the game's vocabulary: *"I called fives"* summons all 5-bearing dominoes into $\sigma_7$.

---

## Appendix: Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\mathbb{P}$ | Pip values $\{0..6\}$ |
| $\mathcal{D}$ | The 28 dominoes |
| $\mathcal{D}^\circ$ | Doubles |
| $\mathcal{D}^\times$ | Non-doubles (mixed) |
| $\sigma_p$ | Natural suit for pip $p$ |
| $\delta$ | Declaration (what was called) |
| $\Delta$ | Set of all declarations |
| $\kappa(\delta)$ | Called set (dominoes summoned to $\sigma_7$) |
| $\hat{\sigma}_p^\delta$ | Effective suit $p$ after calling |
| $\pi(\delta)$ | Suit with power (or $\bot$) |
| $\rho(d, \delta)$ | Rank of $d$ within called suit |
| $\tau(d, \ell, \delta)$ | Trick ranking (three-tier function) |
| $\ell$ | Led suit (context-dependent) |
| $\mathsf{sum}(d)$ | Pip sum of domino $d$ |
| $\ll, \gg$ | Bit shift left/right |
| $\land$ | Bitwise AND |

---

## Appendix: GPU Suitability

The algebraic model is designed for massively parallel evaluation on GPUs.

### Why This Matters

Monte Carlo methods (PIMC, MCCFR) require evaluating thousands to millions of game states. Traditional branching logic (`if isDouble then... elif isTrump then...`) serializes poorly on SIMT architectures. GPUs want uniform operations across lanes.

### The Algebraic Advantage

Every game rule reduces to **table lookup** or **bitmask operation**:

| Operation | Implementation | Complexity |
|-----------|----------------|------------|
| Is trump? | `(calledMask >> d) & 1` | O(1), no branch |
| Can follow? | `(suitMask[δ][ℓ] >> d) & 1` | O(1), no branch |
| Legal moves | `handMask & suitMask[δ][ℓ]` | O(1), parallel |
| Trick winner | `argmax(τ)` over 4 values | O(1), reducible |

### Memory Layout

| Table | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `EFFECTIVE_SUIT` | 28 × 9 | 252 bytes | Domino × AbsorptionId → Suit |
| `SUIT_MASK` | 9 × 8 | 72 × 4 bytes | AbsorptionId × Suit → 28-bit mask |
| `HAS_POWER` | 28 × 9 | 252 bytes | Domino × PowerId → bool |
| `RANK` | 28 × 9 | 252 bytes | Domino × PowerId → 6-bit rank |

**Total: < 1 KB.** Fits entirely in L1 cache. Shared across all threads in a warp.

### State Representation

A complete game state for AI search:

| Component | Representation | Bits |
|-----------|----------------|------|
| Hands (4 players) | 4 × 28-bit mask | 112 |
| Played dominoes | 28-bit mask | 28 |
| Declaration | 4-bit enum | 4 |
| Current trick | 4 × 5-bit domino id | 20 |
| Trick winner | 2-bit player id | 2 |
| Score | 2 × 8-bit | 16 |

**Total: ~182 bits per state.** Thousands of states fit in shared memory.

### Parallelization Strategy

For Perfect Information Monte Carlo (PIMC):

1. **Sample generation**: Each thread samples one possible world (hidden cards assignment)
2. **Rollout**: All threads evaluate tricks in lockstep using table lookups
3. **Reduction**: Parallel reduction to aggregate win rates

For Counterfactual Regret Minimization (MCCFR):

1. **Information sets**: Encoded as (visible cards, declaration, trick history)
2. **Strategy lookup**: Regret tables indexed by info set hash
3. **Update**: Atomic adds to shared regret accumulators

### The Key Insight

The branching logic that plagued the old implementation:

```
if (isTrump(d, δ)) { ... }
else if (canFollow(d, ℓ, δ)) { ... }
else { ... }
```

becomes branchless:

```
tier = hasPower[d][δ] ? 2 : (canFollow[d][ℓ][δ] ? 1 : 0)
isDouble = IS_DOUBLE[d]
rank = (tier > 0) ? (isDouble ? 14 : SUM[d]) : 0
value = (tier << 4) | rank
```

Special case for doubles-trump: rank by pip value, not 14.

And even this can be precomputed into a single lookup when both $\delta$ and $\ell$ are known.

### Theoretical Throughput

With 28 dominoes, 9 absorption configs, 8 possible led suits:
- Complete ranking table: 28 × 9 × 8 = 2,016 entries
- At 1 byte each: **2 KB**

A modern GPU with 48 KB shared memory per SM can hold the entire game rule system **24 times over**, leaving room for game states, strategy tables, and intermediate results.

The algebra doesn't just clarify the rules—it makes them *fast*.
