---
title: Constellations resurrected as Zeb's late-hand table (2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: complete
---

## Summary

Jason, 2026-10-13: "resurrect the constellation work. I abandoned it because it didn't get
us to trick 1 and case recognition was slow. that was then. let's see if we can conceivably
use that as an endgame lookup table and train Zeb with it. experiment". And: "there's
nothing strategy fusion about the constellations. that's the point of them. they're true no
matter what came before. that's also what makes lookup complicated."

**What it shows.**

- **Recognition is no longer slow.** A seat's constellation costs about 0.5 µs in Rust at
  tricks 4–7. texas-42's store paid 4.6 µs. A deployed Zeb decision at the same positions
  costs 1.3–28 ms.
- **The table is exact where its field reads only constellations.** Under the structural
  ladder, two positions with one constellation and different histories plan to the same
  values. That held on 2,240 pairs with two tricks left and 82 pairs with three, with no
  exception.
- **Deployed Zeb is not a function of the constellation.** Its field net reads pips and dead
  tiles. On the same pairs its values differ on 3.4% (two tricks left) and 20% (three).
- **The table does not reach trick 1, for the same reason as before.** Seat-level classes
  number about 10^6 with one trick left, 10^8 with two and at least 10^10 with three. Fresh
  positions almost never repeat before the last trick.
- **The late-hand ladder settles by its third level.** The deployed net already plays the
  settled policy's best move 96% of the time, with a regret of 0.006 make probability against
  0.021 for a random pick. A warm fine-tune toward it moves held-out regret to 0.0048, inside
  the noise at 449 positions. In 150 endgames the play test showed one flip for the tuned net
  and none against.

## Part 1: what the constellation work was

All texas-42 citations are read-only, at its working tree on 2026-10-13.

**The object.** `wiki/idea-retrograde-rank.md` (texas-42, 2026-07-31 → 08-01, `ddd2f96a`)
defines it:

- A tile's **standing** is its relational position among the *living* tiles: what it
  follows, what it sloughs against, and what takes precedence over it, per lead context.
- A **constellation** is the arrangement of standings, counts, holders and lead, with the
  suits as colors that have no order and no pip names. Trump is answered by realization,
  never named.
- A **substitution** swaps a live tile for a dead one with the same standing among the
  living.

The claim was that "the suffix minimax value is a function of the constellation, not the
tiles."

**What was measured to preserve it.** Both measurements used the perfect-information,
both-teams-optimal suffix value:

- 32,886 substitution checks at 1–3 tricks left, with zero divergences
  (`rob/.../retrograde_rank_probe.rs`).
- 817,896 checks at k=2 across two 10-tile sub-universes and every declaration, again with
  zero divergences (`constellation_k2_probe.rs`).
- An exhaustive k=1 census: 2,211,300 last-trick positions → 15,680 fine classes (all four
  lead contexts) → 1,753 coarse classes (the led context only) → 14 outcomes.

The factorization proof (C1, exchange x:009) was adversarially step-checked, which is an
external tier, not a kernel proof. The Lean files `ConstellationCore.lean` and
`ConstellationSuffix.lean` define the suffix machine and its minimax, not the theorem. The
backward step for the pooled, declaration-free key was refuted.

**The Walt census and store.**

- **Lemma E** (`walt/CENSUS-RULINGS.md` ~1045). Equal r1 canonical forms give a tile
  bijection and a seat rotation that carry one remaining extensive game to the other. Every
  count-free fold by an operator that reads only those relations is therefore equal.
- **r1/r3.** Trick-6 roots went 647 → 306 and trick-5 roots 16,112 → 12,924. The compression
  sits late: 2,010 → 16 at trick 7, and the root ratio weakens from 2.1:1 to 1.25:1 one trick
  earlier.
- **The S5j store** (`11beaff1`). Its level 1 was 55,036,800 states → 32,532 forms → 64
  classes. It ran 1.57–2.69× slower than a plain cache. Canonicalization cost 4.6 µs against
  0.1 µs for a state key, and inside the memo a hit saved a subtree that was already
  collapsed. Its floor table was 41× slower than closed-form last-trick resolution.
- **S5k.** At the first play the seat-level quotient is the identity: 1,184,040 hands.

**Why it was abandoned.** It did not reach trick 1, because compression is bought with
deadness and nothing is dead at the opening. Where it did compress, recognition cost more
than it saved under a memo.

## Part 2: the property, and what may be stored

**The property.** Two positions with equal constellations are the same remaining game. A
tile bijection carries every world, every legal move, every trick winner and every point of
one to the other. The remaining game never reads a dead tile, so this holds whatever the
history. It is a statement about games, not a choice of value. It is not strategy fusion and
needs no fence.

**What a table may hold.** The fence ([[strategy-fusion-fence]]) concerns only the stored
value:

- **Fenced if used as a seat's value:** texas-42's probes stored the perfect-information
  minimax per world, and S5j stored a world-informed focal max that its own ruling labels
  "not a seat value."
- **Stored here:** Zeb's own plan value on the seat's information state, which is safe use
  (b). It is one action per cell of the viewer's worlds, values added over what the viewer
  can distinguish, and the other seats played by a fixed field.

**Why lookup is complicated.** Zeb's value is a fold over the remaining game *and over the
seat's information*, and that information carries history:

- The voids the record shows are kept in the key, per tile.
- The banked points are kept, as what the contract still needs.
- The field is the problem. The plan's value is a function of the seat's constellation
  exactly when the field and the belief are equivariant under the bijection:
  - the other seats' policies read only their own constellations and break ties without tile
    ids;
  - the belief is uniform on the fiber.
- The deployed net reads pips and the dead tiles. The posterior draw (#111) also reads the
  history. Both break the condition. That analysis came from a design consultation and was
  confirmed by measurement below.

## Part 3: experiment

Code: `scenario/core/src/constellation.rs` holds the form (`seat_form`, `world_form`) and
`scenario/constellation.py` holds the substitutions and the ladder. Scripts and outputs are in
the research archive under `zeb/constellations/`. Every run was under `timeout 300`. The
positions come from random legal play of random deals (pip trumps, bid 30), not from Zeb's
own games.

### The form

Each live tile and each tile of the open trick gets a label-free code: holder (own, hidden,
or on the table at a relative seat), the seats the record has shown void of it, count, and
trump standing. Each color that can still be led gets its followers' codes in standing order.
Colors are sorted by that invariant and permuted only where invariants tie. The least tile
list is the form, and its order is the labeling.

The header holds the leader and trick position relative to the seat, the led color, and the
points the bidding side still needs, clipped to what is left. The world form keeps exact
holders and drops voids. A count-free variant is Lemma E's object.

Tests check several properties:

- the seven trump declarations fold to one form wherever counts survive the relabeling;
- the dead tiles are ignored;
- the labeling is a bijection;
- every dead-tile substitution keeps the form;
- the world form is blind to voids and the seat form is not.

### Recognition speed

Rust, 11,821 positions with three tricks left or fewer and 17,397 with four or fewer (2,000
deals), on an M5 shared with other jobs (load average about 55, so plan times vary by run):

| per decision | tricks 5–7 | tricks 4–7 |
|---|---|---|
| constellation (canonicalize) | 0.50 µs | 0.55–0.70 µs |
| table probe (canonicalize + hash) | 0.63 µs | 0.6–2.1 µs |
| plain cache probe on the exact record | 17–25 ns | 27–30 ns |
| deployed Zeb plans it (`k0,n512,f=16`, the 250k net) | 1.3–5.4 ms | 7–28 ms |

The first version, with a heap-allocated permutation search, took 1.3–1.4 µs. A tie-free fast
path on stack arrays brought it to 0.5 µs; ties among color invariants are rare.

The old store still loses where it lost, as a replacement for a plain probe: about 25× the
cost. A lookup that replaces a plan is 2,500× or more cheaper.

Inside the late-hand ladder, the planner's exact cache answers first. Of the modeled
decisions that miss it, the constellation table answers **13.2%** at level 1 and **6.9%** at
level 2 (1,000 deals, two tricks left or fewer). Each such hit saves a plan of milliseconds
for a form of under a microsecond, which is the regime S5j's lesson said symmetry would
pay in.

### Table size by tricks left

These are sampled counts, not exact: trick boundaries of 200,000 random-play deals, one
position per trick boundary per deal. "Repeat" is the share of positions whose class was
already seen. Chao1 is a lower-bound estimate of the class count.

| tricks left | seat form (make): distinct / repeat / Chao1 | world, count-free (Lemma E's object): distinct / repeat / Chao1 |
|---|---|---|
| 1 | 131,161 / 34% / ~6.6×10^5 | 2,059 / 99% / ~2.1×10^3 (saturated) |
| 2 | 199,621 / 0.2% / ~6×10^7 | 185,916 / 7% / ~2×10^6 |
| 3 | 199,999 / 0% / ≥10^10 | 199,997 / 0% / ≥10^9 |
| 4 | 199,999 / 0% / no estimate | 200,000 / 0% / no estimate |

What these sizes mean for tractability:

- **One trick left.** A complete seat table is about a million entries.
- **Two tricks left.** About 10^8 entries, a few CPU-hours for a Rust ladder.
- **Three tricks left or more.** No complete table, and no hits on fresh positions. What is
  left is a lazy table inside a search, which is where the 13% came from.

The Python ladder's plan takes 38 ms at level 1 and about 1 s at level 2 with three tricks
left (1,680-world fibers). A three-trick study needs the ladder ported to the core. That
cannot be done in a minute, and it is a design finding.

### Use 1: the lookup, identity by differential test

Every position was paired with dead-tile substitutions that keep the mover's constellation.
The isomorphism used is the one the two canonical labelings define. It is not always the
swap: a new top trump pushes the others down.

| | two tricks left or fewer (2,240 pairs, 1,177 positions) | three tricks left or fewer (82 pairs) |
|---|---|---|
| structural ladder, plan vs plan, levels 1–3 (1–2 at three tricks) | **0 differ** | **0 differ** |
| structural ladder, table lookup vs plan | **0 differ** | **0 differ** |
| deployed Zeb, values | 77 differ (3.4%) | 16 differ (20%) |
| deployed Zeb, tile with equal values | 115 differ (the tie rule reads tile ids) | 9 |

Symmetric constellations, where the field's tie between automorphic tiles is broken by one
labeling of several, occurred 839 times at level 1 with no divergence.

### Use 2: the ladder settles, and the net already sits near it

The ladder is one-rung Zeb over the whole enumerated fiber against the level below. Level 0
is the canonically first legal tile. On 4,563 decisions with two tricks left or fewer (1,500
deals):

| level → next | roots that change | stored states that change |
|---|---|---|
| 1 → 2 | 140 | 2,209 of 152,272 |
| 2 → 3 | 26 | 113 of 152,272 |
| 3 → 4, 4 → 5, 5 → 6 | 0 | 0 |

Level 3 is a fixed point on every state reached. Its values are exact information-state
targets for the late hand. The run took 226 s in Python, over the one-minute mark; the
ladder belongs in the core.

The deployed field net, graded against them on held-out deals (every tenth deal, 449
decisions):

| field | regret, make probability | at a best move |
|---|---|---|
| uniform legal pick | 0.0206 | — |
| deployed `margin_d250k_h128x64` | **0.0059** | 96.2% |
| fresh net trained on the 4,114 other rows (30 epochs) | 0.0201 | 94.4% (memorizes: 0.0005 in training) |
| deployed, fine-tuned warm (`lr` 3e-5, 10 epochs) | **0.0048** | 97.1% |
| deployed, fine-tuned warm (`lr` 1e-4, 10 epochs) | 0.0106 | 95.8% |

**Play** used 150 held-out trick-boundary endgames, paired. With the mover's team on deployed
Zeb, it succeeds in 87. With the settled ladder playing for it, also 87, with no flips. With
Zeb planning over the fine-tuned field, 88: one flip for, none against.

**Reading.**

- The fixed point of Zeb's own ladder in the late hand is nearly what the 250k net already
  plays. The net imitates Zeb, and Zeb is one step of the ladder.
- Moving the net toward the fixed point is the right direction on held-out deals. The size
  is small and unproven at this n.
- The loop Jason named ("evolve the model to a new fix point") runs as: settle the ladder on
  a constellation-keyed table, warm-start the net, re-pour. Its first turn is measured here,
  at the scale Python allows.

## A faster, ranked index (designed, not built)

The question went to a design consultation; the outcome is recorded here.

**A rooted traversal is a canonical labeling.** BFS from each color, followers in standing
order, labeling a color at first sight, with each mixed tile emitted at both ends. The
minimum over at most 7 roots replaces the permutation search, at an estimated 0.1–0.2 µs.
Invariance follows because an isomorphism maps root to root and the traversals run in
lockstep. Completeness follows because the string is a construction recipe.

**No root-free single pass exists.** Tiles with equal codes need not be automorphic.

**A dense index (a bijection onto 0..N_k) is blocked.** Constellations are orbits under a
global relabeling, and orbits do not decompose into per-tile choices the way the labeled
worlds of Silas's `Fiber` do. Past two tricks the space is too large for an array anyway.

**The near-dense variant ranks two parts and multiplies.** A skeleton orbit id comes from a
sorted list of canonical strings. The attributes are ranked over the canonical tile order: the
own set by the combinatorial number system with DP state (tile, own-left), the void matrix by
its closure, and the header. The index is `g·A_k + attr`.

## Links

[[scenario-player]] · [[endgame-equivalence-census]] · [[strategy-fusion-fence]] ·
[[splat-search-bounds]] · [[research-night-2026-10-12]] · [[walt]]
