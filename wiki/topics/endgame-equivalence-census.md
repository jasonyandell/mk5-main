---
title: Endgame equivalence census — exact fungibility measured, and where it lives
kind: topic
first_seen: 2026-07-20
last_updated: 2026-07-20
status: complete
---

Can exact equivalence classes over "every legal domino combination" shrink the
[[hoyt]] solve space with fully correct results? Measured 2026-07-20 on the
frozen 200-root H4 evalset with a certified instrument (`hoyt/equivcensus.py`):
**no at the deal/world level — by mechanism, not accident — and yes in exactly
one place: inside a single known hand.**

## The instrument: the residual signature

The play phase touches tiles only through led_suit, can_follow,
τ-comparisons, and count ([[suit-algebra-spec]] §5–§7, [[play-phase-algebra]]
§4–§6). The **residual signature** of a trick-boundary root encodes exactly
that and nothing else: per-lead legality rows, per-row trick-order dense
ranks, counts, holders, void-forbidden bits, and a seat/payoff header — no
pip identity. Any signature isomorphism is a game isomorphism (values
transport exactly). Canonicalization is individualization-refinement over
the 16 live tiles; certification gates: canonical keys invariant under index
relabeling (50/50), planted symmetries found exactly
(`hoyt/tests/test_equivcensus.py`).

Priors were registered before running (session 2026-07-20): cross-root
collapse 1.0–1.05×; ≥70% trivial automorphism groups; voids suspected as the
main symmetry killer.

## Verdicts

| quantity | measured |
|---|---|
| cross-root canonical classes (strict, points-payoff, relaxed, no-void regimes) | **200/200 distinct in every regime** — collapse 1.000× |
| within-root world orbits (physics-u, all 200 roots) | **3,206,646 → 3,206,646 (exactly 1.000×)**; zero hidden-moving automorphisms |
| within-hand interchangeable pairs (relaxed signature) | **25/200 roots, 29 pairs** (23 roots |Aut|=2, two S3 triples) |
| solver receipt for the 29 pairs | `br_solve` vs uniform field: **29/29 value ties at dv = 0.0 (bitwise)**; non-paired moves differ, median 1.54 pts |
| pip-relabeled roots matching their originals | **0/200** (trump-fixing and cross-declaration both) |

## The co-occurrence theorem (why worlds never compress)

In a 4-seat trick game, any two comparable hidden tiles can land in the same
trick — some consistent world puts them in different seats. So their relative
rank is always game-live, and no exact swap of hidden tiles exists. The only
tiles that can never resolve a trick against each other are two tiles of one
hand (one play per seat per trick) — which is why interchangeability exists
**only within the observer's own hand**. This kills world-level dedup at
every horizon, not just H4: the mechanism is depth-independent. (The lone
loophole — mutually-incomparable "dead trash" pairs whose suits are entirely
dead — occurred 0 times in 200 roots.) Voids turned out irrelevant to the
null: the no-void diagnostic also found nothing (prior refuted).

Corollary for the S7 story: **pip relabeling is not a per-deal game
isomorphism.** A mixed domino leads its higher end and in-suit ranks are pip
sums, so only the unique *monotone* relabeling transports even the unscored
per-deal play tree; arbitrary g transports suit membership only
([[suit-algebra-spec]] §9 refined). The game re-reads its pips at every
mixed lead — endgames do not forget their pips.

## What survives, and its consumers

Within-hand ties are provable exact Q-ties, found structurally in ~1 ms
without solving — "my 2-2 and my 4-4 are the same domino now" (evalset
555009 under doubles-trump; 555040 has an interchangeable 6-0/6-1/6-2
triple). Consumers, per the distill-for-what rule:

- **Solver testing**: value-tie invariants are a new parity-style gate class
  — the 29/29 bitwise receipt doubles as a correctness witness for the BR
  lane (`test_equivcensus.py` E3).
- **Indifference detection**: provable argmax ties for [[jud]] serving and
  claim logic — tie-breaks among these pairs are pure convention, never
  value.
- **Tree width**: merging tied root branches at build time (build is the top
  fleet bucket, [[perf-log]] 19a) — bounded by the 12.5% incidence; census
  gives per-root pairs for free.
- **Narration/teaching**: the tie is the family-table "equals" concept made
  exact ([[w42]] voice anchor).

What is dead with receipts: world-orbit quotients, cross-root reference
dedup, orbit-aware world caps, and any "solve one representative per class"
scheme over hidden-tile assignments. CFR σ among tied moves is NOT a valid
tie observable (RM+'s max(0,·) kink amplifies fp accumulation noise between
value-tied branches — measured 4.5e-04 σ drift on an exact tie); per-move
values are.

Scope: exactness claims are u-lane (physics worlds, uniform weights). The
σ-filtered production line ([[jud]] in the loop) and rng world caps are not
signature-respecting; nets read pips.

## Sizing the class-CFR lane

The strongest surviving consumer is **class-CFR**: constrain every seat's
strategy to be uniform across its own current equals at each info set (one
regret accumulator per equals-class). Before building the solver lane we
sized the merge — across *all* reachable [[cfr-primer|CFR]] info sets (four
seats, every depth), what fraction of NON-FORCED strategy slots would merge?

Priors registered first (2026-07-20): (a) root-level pair incidence for one
seat ≈12.5% (the 25/200 fact above); (b) incidence RISES with depth (fewer
live tiles → adjacency easier); (c) fraction of non-forced slots removed:
single-digit to 15%; (d) some isets become *class-forced* (n_classes = 1
with n_legal > 1).

The instrument is the per-info-set EQUALS predicate: a pair in the acting
seat's hand is equal iff it survives the same residual-signature test used
for the root — with one refinement that carries the theorem to mid-tree
nodes. **Rank comparisons (Q) drop the acting seat's own-hand tiles** (a
tile the seat also holds can never resolve a trick against the pair, so it
must not split them — `relax_meme` made local); follow-legality (C) keeps
all live tiles; a mid-trick guard adds the tiles already on the table.
Certified against `interchangeable_pairs` on three anchors including the
6-0/6-1/6-2 triple — where the naive "compare against every live tile" test
fails (the middle tile 6-1 splits 6-0/6-2), which is exactly the me-me
comparison the relaxed signature forbids. Numba predicate ≈0.9 µs/iset.

Sample: 10 stratified roots (every 10th of the evalset). World caps change
which hidden hands exist, so incidence is cap-sensitive; we measured both
cap-16 and cap-64 on the same roots. **cap-64 is the primary run at 10 roots,
not 20 — the full 20-root cap-64 pass exceeds the session budget (≈40M info
sets at cap-64 already); cap-16 was also run on the full 20 roots for
breadth (2.86%).** cap-256 was not run (out of budget); the 16→64 axis is
the sensitivity witness.

| quantity (cap-64, 10 roots) | measured |
|---|---|
| total reachable info sets | 39,389,562 |
| forced (n_legal = 1) | **82.6%** (matches the known ~83% at H4) |
| non-forced slots removed under own-equals | **2.94%** (14,044,679 → 13,631,733) |
| non-forced isets with ≥1 merge | 6.00% |
| class-forced (n_classes = 1, n_legal > 1) | **376,486** = 5.50% of non-forced |
| per-root spread (non-forced reduction) | min 0.63%, median 3.27%, max 4.79% |

**Prior (c) confirmed but lands low** — the merge is real and single-digit,
near the bottom of the predicted band (~3%, not 15%). **Prior (b) refuted:
incidence FALLS with depth**, it does not rise. Non-forced slot reduction by
trick (cap-64): leading trick (plies 0-3) **4.38%**, plies 4-7 **3.67%**,
plies 8-11 **2.87%**, last trick (plies 12-15) **100% forced — zero
non-forced isets** (one tile each, nothing to merge). The room to merge is
widest where hands are largest, and the last trick contributes nothing.
**Prior (d) confirmed:** class-forced isets are common (376k), essentially
all at the 2-tile trick (plies 8-11), where a single merge collapses both
legal moves into one class — a provable argmax tie CFR can skip.

**Cap sensitivity:** more worlds → slightly *less* merging (cap-16 3.22% →
cap-64 2.94% on the same 10 roots; cap-16 on 20 roots 2.86%). Extra hidden
hands introduce more distinguishing tiles, so pairs split more often; the
headline is robust at ~3% and the full-u value most likely plateaus just
below 2.9%.

**Surprise:** the merge budget is not evenly spread across seats. The
visible seat's slots reduce only **0.68%**, the three hidden seats **3.60%**
(cap-64) — and hidden seats are 75% of all info sets. Class-CFR's savings
live almost entirely in the hidden half of the strategy space (where each
seat is enumerated across every consistent world; the visible seat walks one
known hand). **Verdict for the lane:** own-equals class-CFR is exact and
cheap to detect (~1 µs/iset, no solve) but buys a ~3% strategy-slot
reduction on top of the ~83% forced-slot compression already in the fused
lane (#82) — a correctness-preserving tidy and an indifference-detection
gate, not a width win. Receipts: `scratch/equiv-census/class_cfr_census.py`,
`class_cfr_results_n10.json`.

## Links

[[hoyt]] · [[walt]] · [[suit-algebra-spec]] · [[play-phase-algebra]] ·
[[jud]] · [[perf-log]]
