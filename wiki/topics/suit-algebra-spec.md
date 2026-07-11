---
title: Suit algebra — the formal specification
kind: topic
first_seen: 69d636ac
last_updated: pending-this-ingest
status: active
---

The complete algebraic specification of Texas 42's suit structure: called sets, power,
effective suits, the three-tier trick order, the unique-winner theorem, the S₇ symmetry, and
the machine encoding. [[suit-algebra]] tells the story of how this model came to be and what
it replaced; this page is the mathematics itself. The play phase built on these primitives
is [[play-phase-algebra]].

Merged at this ingest from `docs/theory/SUIT_ALGEBRA.md` (first committed 69d636ac) and
`SUIT_ALGEBRA_PURE.md` (19373119) — two ~90%-identical documents; the pure-mathematical
formulation is the body, the machine encoding is the appendix.

## §1. The domino set

Let $\mathbb{P} = \{0, 1, 2, 3, 4, 5, 6\}$ be the set of pip values.

A **domino** is a 2-element multiset $\{i, j\}$ of pip values ($i = j$ allowed). The **domino set** is:

$$\mathcal{D} = \{\, \{i, j\} : i, j \in \mathbb{P} \,\} \qquad |\mathcal{D}| = 28$$

Treating dominoes as multisets rather than ordered pairs makes **pip membership** literal—$p \in d$ is plain set membership—and spares every later definition from min/max bookkeeping.

Define $\mathsf{sum}(\{i,j\}) = i + j$ and $\mathsf{max}(\{i,j\}) = \max(i, j)$, and partition $\mathcal{D}$ into **doubles** and **mixed**:

$$\mathcal{D}^\circ = \{\, \{p, p\} : p \in \mathbb{P} \,\} \qquad \mathcal{D}^\times = \mathcal{D} \setminus \mathcal{D}^\circ$$

## §2. The natural covering

For each $p \in \mathbb{P}$, the **natural suit** is:

$$\sigma_p = \{d \in \mathcal{D} : p \in d\}$$

This yields a **covering** $\Sigma = \{\sigma_0, \sigma_1, \ldots, \sigma_6\}$ with properties:

| Property | Statement |
|----------|-----------|
| Cardinality | $\forall p: |\sigma_p| = 7$ |
| Double membership | $d \in \mathcal{D}^\circ \implies |\{p : d \in \sigma_p\}| = 1$ |
| Mixed membership | $d \in \mathcal{D}^\times \implies |\{p : d \in \sigma_p\}| = 2$ |
| Pairwise intersection | $p \neq q \implies \sigma_p \cap \sigma_q = \{\{p, q\}\}$ |

This is **not** a partition—each mixed domino inhabits two suits simultaneously.

## §3. Declarations: called set and power

A **declaration** $\delta$ does two things: it **calls** dominoes into the 8th suit, and it may grant that suit **power** over the others.

$$\delta \in \Delta = \mathbb{P} \cup \{\mathsf{doubles\text{-}trump}, \mathsf{doubles\text{-}suit}, \mathsf{notrump}\}$$

The **called set** function $\kappa : \Delta \to \mathcal{P}(\mathcal{D})$:

$$\kappa(\delta) = \begin{cases}
\sigma_p & \text{if } \delta = p \in \mathbb{P} \\[4pt]
\mathcal{D}^\circ & \text{if } \delta \in \{\mathsf{doubles\text{-}trump}, \mathsf{doubles\text{-}suit}\} \\[4pt]
\varnothing & \text{if } \delta = \mathsf{notrump}
\end{cases}$$

The **called suit** is $\sigma_7^\delta = \kappa(\delta)$.

A declaration is **powered** iff $\delta \in \mathbb{P} \cup \{\mathsf{doubles\text{-}trump}\}$. The **power set** $\pi : \Delta \to \mathcal{P}(\mathcal{D})$:

$$\pi(\delta) = \begin{cases}
\kappa(\delta) & \text{if } \delta \text{ is powered} \\[4pt]
\varnothing & \text{otherwise}
\end{cases}$$

Dominoes with power beat all others (§6). When $\pi(\delta) = \varnothing$, no suit has power and the highest follower wins.

The pair $(\kappa(\delta), \pi(\delta))$ is the **entire semantic content** of a declaration. The ten declarations yield ten distinct pairs but only **nine** distinct called sets: doubles-trump and doubles-suit share $\kappa = \mathcal{D}^\circ$ and are distinguished by power alone. This asymmetry matters for table layouts (see the machine-encoding appendix).

**Nello note:** Doubles-suit serves nello contracts. The strategic objective is to *lose* tricks, but the trick mechanics are unchanged: doubles form suit 7, rank by pip value (6-6 highest), and the highest follower wins—no domino has power.

**Sevens note:** The sevens contract is deliberately absent from $\Delta$: it has no suit structure and no decisions. Play is forced—order your hand by $|\mathsf{sum}(d) - 7|$ and play it out. Nothing in this algebra applies. (See [[rules-of-42]] §Special contracts.)

## §4. The effective suit structure

Under declaration $\delta$, the covering transforms. Define the **effective suits**:

$$\hat{\sigma}_p^\delta = \sigma_p \setminus \kappa(\delta) \quad (p \in \{0..6\}) \qquad \hat{\sigma}_7^\delta = \kappa(\delta)$$

The full **effective structure** is:

$$\hat{\Sigma}^\delta = \{\hat{\sigma}_0^\delta, \hat{\sigma}_1^\delta, \ldots, \hat{\sigma}_6^\delta, \hat{\sigma}_7^\delta\}$$

### Theorem: effective membership

Under any declaration $\delta$, each domino belongs to **exactly one or two** effective suits:

$$\forall d \in \mathcal{D}: \quad 1 \leq |\{s \in \hat{\Sigma}^\delta : d \in s\}| \leq 2$$

with equality to 1 iff $d \in \kappa(\delta) \cup \mathcal{D}^\circ$.

Called dominoes live only in $\hat{\sigma}_7$; uncalled doubles live only in their own pip suit; uncalled mixed dominoes still straddle two suits. The covering narrows under calling, but it does not become a partition.

## §5. Following and the led suit

When suit $\ell \in \{0..7\}$ is led under declaration $\delta$, a domino $d$ **can follow** iff:

$$\mathsf{follows}(d, \ell, \delta) \iff d \in \hat{\sigma}_\ell^\delta$$

Unpacking the two cases:

$$\mathsf{follows}(d, \ell, \delta) \iff \begin{cases}
(\ell \in d) \land (d \notin \kappa(\delta)) & \text{if } \ell \in \{0..6\} \\[4pt]
d \in \kappa(\delta) & \text{if } \ell = 7
\end{cases}$$

A called domino cannot follow a pip lead—it has been called away. Only called dominoes follow a called lead.

When domino $d$ is **led**, it determines the led suit:

$$\ell(d, \delta) = \begin{cases}
7 & \text{if } d \in \kappa(\delta) \\[4pt]
\mathsf{max}(d) & \text{otherwise}
\end{cases}$$

Under notrump $\kappa(\delta) = \varnothing$, so $\ell = 7$ never occurs.

## §6. The trick order

### Rank

Rank within a suit depends only on the domino and the declaration—never on what was led. Let $R = \{0 < 1 < \cdots < 12\} \cup \{\top\}$ be the linear order with $\top$ above every number, and define $\mathsf{rank} : \mathcal{D} \times \Delta \to R$ (cases read top-down):

$$\mathsf{rank}(d, \delta) = \begin{cases}
p & \text{if } \kappa(\delta) = \mathcal{D}^\circ \land d = \{p,p\} & \text{(doubles as a suit: by pip)} \\[4pt]
\top & \text{if } d \in \mathcal{D}^\circ & \text{(lone double in a pip suit)} \\[4pt]
\mathsf{sum}(d) & \text{otherwise} & \text{(non-double: by pip sum)}
\end{cases}$$

Key facts:

- **The double tops its suit**—the fundamental ranking rule of Texas 42. $\top$ exceeds every pip sum.
- **Exception:** when doubles form their own suit ($\kappa(\delta) = \mathcal{D}^\circ$), they rank by pip value: 6-6 beats 5-5 beats … beats 0-0.
- Within suit $\ell$, pip sum $\ell + k$ is monotone in the other pip $k$—so sum-ordering is exactly the familiar "rank by the other end."

### Tier

Given led suit $\ell$, each domino lands in a tier (cases read top-down):

$$\mathsf{tier}(d, \ell, \delta) = \begin{cases}
2 & \text{if } d \in \pi(\delta) & \text{(trump)} \\[4pt]
1 & \text{if } d \in \hat{\sigma}_\ell^\delta & \text{(follows the led suit)} \\[4pt]
0 & \text{otherwise} & \text{(slough)}
\end{cases}$$

### Trick order

Order dominoes by the pair (tier, rank) under the **lexicographic order**, with all sloughs identified at the bottom:

$$\tau(d, \ell, \delta) = \begin{cases}
(0, 0) & \text{if } \mathsf{tier}(d, \ell, \delta) = 0 \\[4pt]
\bigl(\mathsf{tier}(d, \ell, \delta),\; \mathsf{rank}(d, \delta)\bigr) & \text{otherwise}
\end{cases}$$

Sloughs are unordered among themselves. This is consistent because the lead domino always lands in tier 1 or tier 2 (Lemma 7.1), so tier 0 is never the highest occupied tier—slough order is irrelevant to trick resolution.

### Architectural note: configuration vs context

The ranking decomposes into two dependencies:

| Aspect | Depends on | Known when |
|--------|-----------|------------|
| Tier 2 membership | $\delta$ only | Hand starts |
| Tier 1 membership | $\delta$ and $\ell$ | Trick starts |
| Rank | $\delta$ only | Hand starts |

The called set $\kappa(\delta)$, power set $\pi(\delta)$, and rank are **configuration-dependent**—computable from $\delta$ alone, suitable for lookup tables. Only "follows the led suit" is **context-dependent**: it requires $\ell$, which isn't determined until the lead is played.

## §7. Theorem: unique winner

**Claim:** Let $d_1, d_2, d_3, d_4 \in \mathcal{D}$ be distinct, with $d_1$ the lead and $\ell = \ell(d_1, \delta)$. Then:

$$\exists! \, d^* : \forall i, \; \tau(d^*, \ell, \delta) \geq \tau(d_i, \ell, \delta)$$

Distinctness is the only hypothesis—legality of the trick is not needed.

*Proof.* Under the lexicographic order, the maximum lies in the highest occupied tier; we show that tier is never 0 and never contains two dominoes of equal rank.

### Lemma 7.1: the lead is never tier 0

If $d_1 \in \kappa(\delta)$ then $\ell = 7$: under a powered declaration $d_1 \in \pi(\delta)$ (tier 2); otherwise $d_1 \in \hat{\sigma}_7^\delta = \kappa(\delta)$ (tier 1). If $d_1 \notin \kappa(\delta)$ then $\ell = \mathsf{max}(d_1)$ and $d_1 \in \sigma_\ell \setminus \kappa(\delta) = \hat{\sigma}_\ell^\delta$ (tier 1). Either way at least one domino is in tier 1 or above, so tier 0 never contains the winner. $\square$

### Lemma 7.2: pip sums are injective within a natural suit

Each $d \in \sigma_p$ has the form $\{p, k\}$, so $\mathsf{sum}(d) = p + k$. Distinct members have distinct other-pips $k$, hence distinct sums. $\square$

### Lemma 7.3: tier 2 ranks are injective

$\pi(\delta) \neq \varnothing$ only for powered declarations, where $\pi(\delta) = \kappa(\delta)$.

*Pip trump* $\delta = t$: the only double in $\sigma_t$ is $\{t,t\}$, with rank $\top$. Non-doubles have distinct sums by Lemma 7.2, all below $\top$. Injective.

*Doubles-trump*: $\pi(\delta) = \mathcal{D}^\circ$, ranked by pip value—ranks $\{0, \ldots, 6\}$, all distinct. $\square$

### Lemma 7.4: tier 1 ranks are injective

The tier 1 occupants are $\hat{\sigma}_\ell^\delta \setminus \pi(\delta)$.

*Case $\ell \in \{0..6\}$:* $\hat{\sigma}_\ell^\delta \subseteq \sigma_\ell$. The only possible double is $\{\ell,\ell\}$ (others are called away or lack pip $\ell$), with rank $\top$. Non-doubles have distinct sums by Lemma 7.2, all below $\top$. Injective.

*Case $\ell = 7$:* If $\delta$ is powered, $\hat{\sigma}_7^\delta = \kappa(\delta) = \pi(\delta)$, so tier 1 is empty—vacuously injective. If $\delta = \mathsf{doubles\text{-}suit}$, tier 1 is $\mathcal{D}^\circ$ ranked by pip value—ranks $\{0, \ldots, 6\}$, all distinct. Under notrump $\ell = 7$ is unreachable (§5). $\square$

### Lemma 7.5: tiers are strictly separated

The lexicographic order places every tier-2 pair above every tier-1 pair, and every tier-1 pair above $(0,0)$. (In the 6-bit integer encoding below this is literal: tier 2 occupies $[32,46]$, tier 1 $[16,30]$, tier 0 is $\{0\}$—pairwise disjoint since rank $\leq 14 < 16$.) $\square$

### Completion

By Lemma 7.1 the highest occupied tier is 1 or 2. By Lemma 7.5 the maximum $\tau$ comes from that tier. By Lemmas 7.3–7.4 ranks are injective there, so distinct dominoes get distinct $\tau$. Therefore $\arg\max$ is unique. $\square$

## §8. The complete decision function

Given a trick with lead $d_1$ under declaration $\delta$ and a hand $H$:

**Step 1.** Led suit: $\ell = \ell(d_1, \delta)$ (§5).

**Step 2.** For each subsequent domino $d \in H$, legal play requires:
$$\mathsf{legal}(d, \ell, \delta, H) \iff \mathsf{follows}(d, \ell, \delta) \lor \neg\exists h \in H: \mathsf{follows}(h, \ell, \delta)$$

**Step 3.** Winner:
$$\mathsf{winner} = \underset{d \in \text{trick}}{\arg\max} \; \tau(d, \ell, \delta)$$

By Theorem §7, this is well-defined.

## §9. The symmetry group

The symmetric group $S_7$ acts on $\mathbb{P}$ and induces automorphisms of $\mathcal{D}$:

$$\phi_g\{i, j\} = \{g(i), g(j)\} \quad \text{for } g \in S_7$$

(well-defined on multisets—no ordering to repair). The action is equivariant on suits, $\phi_g(\sigma_p) = \sigma_{g(p)}$, so it carries effective structures to effective structures:

$$\hat{\Sigma}^p \cong \hat{\Sigma}^{g(p)} \quad \forall g \in S_7$$

All pip-trump declarations share **one** following/legality structure, instantiated seven ways. This is the isomorphism that lets training-data generation sample one pip-trump representative instead of all seven (see [[suit-algebra]]).

**What the symmetry does not preserve.** The action preserves suit *membership* only—not the ranked game:

- *Intra-suit order:* the swap $0 \leftrightarrow 6$ carries trump-0 to trump-6 but sends $\{3,6\}$—the top non-double of suit 3—to $\{3,0\}$, the bottom. Pip sums are not equivariant.
- *Count points:* $\{5,0\}, \{4,1\}, \{3,2\}, \{5,5\}, \{6,4\}$ are fixed targets, not symmetric under relabeling.

Pip-trump declarations are isomorphic as legality structures, not as games. ([[play-phase-algebra]] §8.3 develops why the full scored game breaks the big symmetry.)

## §10. Summary: the 8-suit model

| Suit index | Name | Contents under $\delta$ |
|------------|------|-------------------------|
| $\sigma_0$ | Blanks | $\hat{\sigma}_0^\delta = \sigma_0 \setminus \kappa(\delta)$ |
| $\sigma_1$ | Aces | $\hat{\sigma}_1^\delta = \sigma_1 \setminus \kappa(\delta)$ |
| $\sigma_2$ | Deuces | $\hat{\sigma}_2^\delta = \sigma_2 \setminus \kappa(\delta)$ |
| $\sigma_3$ | Treys | $\hat{\sigma}_3^\delta = \sigma_3 \setminus \kappa(\delta)$ |
| $\sigma_4$ | Fours | $\hat{\sigma}_4^\delta = \sigma_4 \setminus \kappa(\delta)$ |
| $\sigma_5$ | Fives | $\hat{\sigma}_5^\delta = \sigma_5 \setminus \kappa(\delta)$ |
| $\sigma_6$ | Sixes | $\hat{\sigma}_6^\delta = \sigma_6 \setminus \kappa(\delta)$ |
| $\sigma_7$ | **Called** | $\kappa(\delta)$ |

The word **called** reflects the game's vocabulary: *"I called fives"* summons all 5-bearing dominoes into $\sigma_7$. (The term is Jason's coinage replacing the earlier "absorbed suit" jargon — see [[suit-algebra]].)

## Notation reference

| Symbol | Meaning |
|--------|---------|
| $\mathbb{P}$ | Pip values $\{0..6\}$ |
| $\mathcal{D}$ | The 28 dominoes (2-element multisets) |
| $\mathcal{D}^\circ$ | Doubles |
| $\mathcal{D}^\times$ | Non-doubles (mixed) |
| $\mathsf{sum}(d)$, $\mathsf{max}(d)$ | Pip sum / larger pip of $d$ |
| $\sigma_p$ | Natural suit for pip $p$ |
| $\delta$, $\Delta$ | Declaration; set of all declarations |
| $\kappa(\delta)$ | Called set (dominoes summoned to $\sigma_7$) |
| $\pi(\delta)$ | Power set: $\kappa(\delta)$ if powered, else $\varnothing$ |
| $\hat{\sigma}_\ell^\delta$ | Effective suit $\ell \in \{0..7\}$; $\hat{\sigma}_7^\delta = \kappa(\delta)$ |
| $\ell(d, \delta)$ | Suit led by domino $d$ |
| $\top$ | Top element of the rank order; the double's rank in a pip suit |
| $\mathsf{rank}(d, \delta)$ | Rank within suit (declaration-only) |
| $\tau(d, \ell, \delta)$ | Trick order: (tier, rank), lexicographic |
| $\phi_g$ | Automorphism of $\mathcal{D}$ induced by $g \in S_7$ |

## Appendix: machine encoding

The algebra was designed from the start for massively parallel evaluation — the substrate for
[[the-oracle]]'s GPU solver and the PIMC world-sampling behind [[expected-q-value]]. (An
MCCFR direction was explored and retired separately; see [[pre-ml-ai-attempts]].) Branching
rules logic serializes poorly on SIMT architectures; the algebra reduces every rule to table
lookup or bitmask operation.

### The 6-bit τ encoding

$\top$ becomes 14 (any constant exceeding the max pip sum 12 works; 14 fits the 4-bit rank field):

$$\tau(d, \ell, \delta) = \begin{cases}
0 & \text{if } \mathsf{tier} = 0 \\
(\mathsf{tier} \ll 4) \;|\; \mathsf{rank}(d, \delta) & \text{otherwise}
\end{cases}$$

| Tier | Binary pattern | Decimal range |
|------|----------------|---------------|
| 2 (trump) | `10_xxxx` | 32–46 |
| 1 (follows) | `01_xxxx` | 16–30 |
| 0 (slough) | `00_0000` | 0 |

Extraction: $\mathsf{tier}(\tau) = \tau \gg 4$, $\mathsf{rank}(\tau) = \tau \,\&\, \mathsf{0xF}$.

### Index spaces

Lookup tables index on distinct *semantic* values, not the ten declarations:

| Space | Values | Count |
|-------|--------|-------|
| Called config | $\kappa(\delta) \in \{\sigma_0, \ldots, \sigma_6, \mathcal{D}^\circ, \varnothing\}$ | 9 |
| Power config | $\pi(\delta) \in \{\sigma_0, \ldots, \sigma_6, \mathcal{D}^\circ, \varnothing\}$ | 9 |
| Declaration class | $(\kappa(\delta), \pi(\delta))$ | **10** |

The full ranking needs the *pair*: doubles-trump and doubles-suit share a called set but differ in $\tau$—a sloughed double wins under one and loses under the other.

### Tables and operations

| Operation | Implementation | Complexity |
|-----------|----------------|------------|
| Has power? | `(POWER_MASK[δ] >> d) & 1` | O(1), no branch |
| Can follow? | `(SUIT_MASK[δ][ℓ] >> d) & 1` | O(1), no branch |
| Legal moves | `handMask & SUIT_MASK[δ][ℓ]` (empty ⟹ whole hand legal) | O(1), parallel |
| Trick winner | `argmax(τ)` over 4 values | O(1), reducible |

| Table | Dimensions | Size | Contents |
|-------|------------|------|----------|
| `LED_SUIT` | 28 × 9 | 252 B | Domino × CalledConfig → $\ell(d, \delta)$ |
| `SUIT_MASK` | 9 × 8 | 288 B | CalledConfig × Suit → 28-bit mask of $\hat{\sigma}_\ell$ |
| `POWER_MASK` | 9 | 36 B | PowerConfig → 28-bit mask of $\pi(\delta)$ |
| `RANK` | 28 × 2 | 56 B | Domino × (doubles called?) → $\mathsf{rank}(d, \delta)$ |

(Following is a *relation*—a mixed uncalled domino belongs to two effective suits—so membership lives in `SUIT_MASK`; the single-valued `LED_SUIT` answers only "which suit does $d$ lead?")

**Total: under 1 KB**, L1-resident, shared across a warp. Ranking logic is branchless, mirroring §6:

```
tier  = ((POWER_MASK[δ] >> d) & 1) ? 2 : ((SUIT_MASK[δ][ℓ] >> d) & 1)
value = tier ? (tier << 4) | RANK[d][doublesCalled] : 0
```

The fully-precomputed ranking table is 28 dominoes × 10 declaration classes × 8 led suits = 2,240 entries — ~2.2 KB at a byte each.

### State representation

A complete game state for search:

| Component | Representation | Bits |
|-----------|----------------|------|
| Hands (4 players) | 4 × 28-bit mask | 112 |
| Played dominoes | 28-bit mask | 28 |
| Declaration | 4-bit enum | 4 |
| Current trick | 4 × 5-bit domino id | 20 |
| Trick winner | 2-bit player id | 2 |
| Score | 2 × 8-bit | 16 |

~182 bits per state; thousands of states fit in GPU shared memory. ([[play-phase-algebra]]
tightens this further to a score-free int64 packing for the oracle.)

### Live implementations

- **TypeScript engine**: `src/game/core/domino-tables.ts` (DOMINO_PIPS, getAbsorptionId, getPowerId, EFFECTIVE_SUIT, SUIT_MASK, RANK, HAS_POWER), tested in `src/tests/unit/domino-tables.test.ts`.
- **Oracle solver**: `forge/oracle/tables.py` (`led_suit_for_lead_domino`, `can_follow`, `trick_rank`, `resolve_trick`) — see [[play-phase-algebra]] §9.
- **Tokenizer**: τ's rank-relative-to-trump idea survives as `trump_rank` / `TRICK_RANK_TABLE` in `forge/eq/game_tensor.py`, consumed by [[forge]]'s training pipeline and `forge/analysis/bias/` interpretability scripts.

## Links

[[suit-algebra]] · [[play-phase-algebra]] · [[rules-of-42]] · [[the-oracle]] ·
[[expected-q-value]] · [[engine]]
