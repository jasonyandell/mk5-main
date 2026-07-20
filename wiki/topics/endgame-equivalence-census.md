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

## Links

[[hoyt]] · [[walt]] · [[suit-algebra-spec]] · [[play-phase-algebra]] ·
[[jud]] · [[perf-log]]
