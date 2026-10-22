---
title: "The trick-1 ladder: how far the first trick is from exact"
kind: experiment
first_seen: 2026-10-22
last_updated: 2026-10-22
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

The trick-1 session of the exact-player track (Exact-T1, 2026-10-22), the other end from
[[exact-player-s0-2026-10-22]]. At trick 1 the bidder (S0, 30, a pip declaration) is not
close to exact. Under one declared type, how far is the seat from proving its information-state
best tile? The pre-registration measures it by horizon and by budget: does the proof settle, how
many tiles survive, how large is the certified regret Γ, and what does it cost? Walt-free:
nothing of mk5 `scenario/` or texas-42's walt is a component.

## The player (built, `silas/src/player/trick1.rs`)

- **The model.** The other three seats each play one deterministic law, `SeatLaw::Lowest`, as
  the bootstrap type. σ0 was the brief's type, but it is a sampled Walt ("Walt free"; approved).
  The deal prior is uniform, and the auction is not modelled.
- **The belief.** The exact counted posterior. Likelihoods are 0/1, so masses are integer world
  counts.
  - At the lead nothing has been seen: all 116,280 × 3,432 = **399,072,960 worlds**.
  - A follower behind the recorded lead reads the record through its voids only
    (`FiberUniform`). The recorded (champion) lead has likelihood zero under Lowest at every
    follow root: **the record refutes the one-type model**. That is a finding, and the reason
    for a library of types (S1).
- **The ladder** is the focal-horizon sandwich. The viewer's next k + 1 decisions are branched per
  world, and each horizon state is keyed by its exact public history. Then:
  - Reduction: the viewer's node takes the max of each side separately; another seat's node sums.
  - **Lower L**: the best lawful tail (the viewer's own Lowest or Highest below the horizon),
    chosen per public history after the sum over its worlds. This is lawful, one choice per
    information state.
  - **Upper U**: per world, an OR-search over the viewer's own plays against the others' fixed
    laws. It is clairvoyant, and its fence status is the sanctioned use, *a bound that refutes*
    ([[strategy-fusion-fence]]). U drops tiles and sizes Γ, and never chooses.
  - The act is argmax L (lowest tile on ties). Survivors = {a : U_a ≥ max L}.
    Γ = max U − L(act).
  - Settled is read two ways. **Strict**: L_b > every other U. **Proved**: Γ = 0 exactly, a
    certified optimal set, ties included. Proved counts as settled for the predictions
    (orchestrator ruling, amendment 6).
- **Budget** is in plays, every tile any simulation lays, split tree / tails / god. It is
  deterministic per shard, a shard being one hand of the first hidden seat (FAIR CLOCKS).
  - Gathers over shard ranges are bytes that merge.
  - Gathers per root tile join in the rung (root tiles never meet below the root's max).
  - A rung consumes its tile gathers one at a time, so tiles that do not fit in memory together
    still reduce.
  - A rung refuses a partial cover, a missing or repeated tile, or a total past its budget.
- **Metal.** Exact-GPU's `silas-metal` gather (exact/gpu) writes the same bytes, Work included.
  At k = 2 the leaf maps are too big to keep, so Metal writes each root tile's exact counts
  (L, U, mass, leaves, plays) with the SHA-256 of its gather bytes. A tile too big for one gather
  is split by the next seat's reply at depth 1. That node sums, so the pieces' counts add exactly.
  The rung (choice, survivors, Γ, settled) is single-sourced in `trick1.rs`. The runner's
  `--triples` builds the same `Rung` from per-tile counts.
- **CLI.** `cargo run --release --example trick1_roots -- DIR --deals D --viewer V --k K`,
  with `--from/--to/--root --save` for pieces, `--load` for rungs from bytes, `--triples` for
  rungs from counts, and `--counts` for the exact (L, U, mass) per tile.

## Gates

- **Tests** (`silas/tests/trick1.rs`, ~2 s):
  - The flat rule tables equal `Physics`.
  - On 24 seeded endgames (4–6 tricks left; both roots, both laws), at every horizon: every
    interval contains `player::recount`'s exact value (Exact-1's independent construction),
    rungs nest in k, and at k + 1 ≥ tiles left L = Q = U exactly.
  - On six of them, at every horizon: three pieces through bytes, merged out of order, equal the
    whole gather byte for byte and give the same rung; per-tile gathers joined equal the
    one-gather rung; a double shard, a partial cover, a missing tile and a repeated tile are each
    refused.
- **Engines, one CPU rung per k**, all on deal 0's lead, each from capped CPU pieces merged,
  byte-identical to Metal's gather (SHA-256, Work included):

  | k | CPU pieces | SHA-256 |
  |---|---|---|
  | 0 | 4 | `b126067c…92f1e` |
  | 1 | 3 | `2423ca29…3d481f` |
  | 2 (tile 6-5) | 8 | `b16dcc5d…4c60fa` |

- **The k = 2 gate** (amendment 7): deal 1's seven k = 2 tiles, full bytes (33 GB), through
  `trick1.rs`'s rung equal Metal's per-tile counts exactly, 7 of 7. Only after that were the
  other roots taken as counts.
- **Stop rules:** 147 lead intervals (7 leads × 7 tiles × k = 0, 1, 2), none empty, and every
  one of the 98 consecutive pairs nests in k.

## Pre-registration

`scratch/desks/family-2026-10-22/PREREG-T1.md`, reviewed by the orchestrator, with eight
amendments, each fixed before the numbers it governs:

1. predictions in numbers;
2. B2 set from a probe;
3. the follow roots' belief is a finding;
4. order, and Metal pieces once byte-equal;
5. B2 = 3×10¹² plays per lead root, and the per-tile split;
6. strict vs proved;
7. k = 2 receipts and the deal-1 gate;
8. tiles over the 1.2×10⁸ leaf cap are split by the depth-1 reply. That is a storage partition:
   B2 is unchanged.

- **Roots**: 7 leads (S0's trick-1 lead, the first deal of each pip declaration in pour
  `pour_make_s1001_n512_d5000`: deals 0, 1, 2, 5, 12, 15, 39) and 21 follow roots (seats 1–3
  on the same deals), reported apart.
- **Budgets**: k = 0 and 1 for every lead, and k = 2 at B2 per lead root; a 300 s cap per piece.
  Follow roots climb until settled or past 10¹² plays.
- **Predictions** (top budget: k = 2, or the highest rung within B2):
  - Jason: settled at k ≤ 2 on ≥ 4/7 leads; and the act ≠ the law's tile with gain ≥ 10‰ on
    ≥ 4/7.
  - Orchestrator: survivors > 1 and Γ > 50‰ on ≥ 4/7.
  - Exact-T1: no lead settles at k ≤ 2; Γ at k = 1 > 150‰ on ≥ 5/7; survivors ≥ 3 at k = 1 on
    ≥ 5/7; act ≠ law with gain ≥ 10‰ on ≥ 5/7; follow roots settle at k ≤ 2 on ≥ 1/3.

## Results: the seven leads

Each lead is 399,072,960 worlds. Γ is in ‰ of worlds; the value is the share of worlds in which
S0's side makes 30. The law's tile is Lowest's own lead. Walt's tile is the champion's recorded
lead, shown as data only.

**Γ by horizon** (act, Γ‰, survivors):

| deal (decl) | k = 0 | k = 1 | k = 2 | law's tile | Walt's tile |
|---|---|---|---|---|---|
| 0 (P4) | 6-5, 302.3, 7 | 6-5, 163.8, 7 | 6-5, **83.9**, 7 | 2-1 | 4-0 |
| 1 (P6) | 5-5, 6.4, 6 | 5-5, **0 proved**, 5 | 2-2, **0 proved**, 5 | 2-2 | 6-5 |
| 2 (P5) | 3-3, 271.5, 6 | 3-3, 231.8, 6 | 6-4, **154.9**, 5 | 2-0 | 6-4 |
| 5 (P2) | 2-2, 228.7, 7 | 2-2, 163.6, 7 | 6-4, **74.6**, 7 | 1-1 | 2-2 |
| 12 (P3) | 4-3, 271.6, 7 | 4-3, **165.2**, 7 | refused: 3.33×10¹² plays > B2 | 1-1 | 3-0 |
| 15 (P0) | 6-0, 392.7, 7 | 6-0, 301.6, 7 | 6-0, **191.0**, 7 | 0-0 | 0-0 |
| 39 (P1) | 6-2, 589.6, 7 | 2-1, 469.6, 7 | 2-1, **295.2**, 7 | 1-0 | 1-0 |

**At the top budget** (k = 2; deal 12 at k = 1). Intervals are exact counts over the mass, to
four places:

| deal | act [L, U] | law's tile [L, U] | gain‰ = L(act) − L(law) | certified‰ = L(act) − U(law) |
|---|---|---|---|---|
| 0 | 6-5 [0.1889, 0.2728] | 2-1 [0.1553, 0.2259] | 33.6 | −37.0 |
| 1 | 2-2 [1, 1] | 2-2 [1, 1] | 0 | 0 |
| 2 | 6-4 [0.4329, 0.5879] | 2-0 [0.3428, 0.4385] | 90.2 | −5.6 |
| 5 | 6-4 [0.8635, 0.9313] | 1-1 [0.8262, 0.9298] | 37.3 | −66.3 |
| 12 | 4-3 [0.8344, 0.9993] | 1-1 [0.7866, 0.9996] | 47.7 | −165.2 |
| 15 | 6-0 [0.7801, 0.9711] | 0-0 [0.7688, 0.9711] | 11.3 | −191.0 |
| 39 | 2-1 [0.6008, 0.8960] | 1-0 [0.5580, 0.8960] | 42.8 | −295.2 |

Plays per rung: k = 0, 1.9–12×10¹¹; k = 1, 4.9×10¹¹–1.7×10¹²; k = 2, 6.1×10¹¹–2.8×10¹². The
upper bound costs 37–92% of them. Leaves (public histories at the horizon): k = 0,
3–10×10⁴; k = 1, 7–13×10⁶; k = 2, 2.4–6.8×10⁸.

- **Γ falls with every rung on every lead**, by a factor of 1.2 to 2.2 per rung. The intervals
  nest, and none is empty.
- **One lead is proved, none strictly settled.** Deal 1 (sixes trump) is proved at k = 1. At
  k = 2 five tiles make in every one of its worlds (2-2, 3-3, 5-5, 6-5, 6-6), so the act is the
  lowest of them, 2-2, which is the law's own tile. Walt's recorded 6-5 is one of the five.
- **The act moves with k** on three leads:
  - deal 2: 3-3 → 6-4, where 6-4 leads 3-3 by 45,773 worlds (0.11‰), on L;
  - deal 5: 2-2 → 6-4;
  - deal 39: 6-2 → 2-1.

  A choice made at k = 0 or 1 is provisional while Γ is large.
- **Where Walt's recorded tile is the act at the top budget:** deal 2 (6-4) only. Where it
  survives: on every lead, since every tile survives on five of them.
- **The gain is in guarantees, not yet a refutation.** The act's lower bound beats the law's
  tile's lower bound by ≥ 10‰ on 6 of 7 leads. But the law's tile still survives on all seven:
  U(law) ≥ L(act). No lead's ladder has yet shown the law's tile to be worse.

**The registered scorecard** (top budget):

- **Strict settled: 0 of 7. Proved: 1 of 7** (deal 1).
- **Jason**:
  - settled on ≥ 4/7: **not met** (1/7);
  - act ≠ law with gain ≥ 10‰ on ≥ 4/7: **met** (6/7; deal 1's act is the law's tile).

  The gain is the registered L − L. The certified L − U is ≤ 0 on 7/7.
- **Orchestrator**, survivors > 1 and Γ > 50‰ on ≥ 4/7: **met** (6/7; deal 1 is the exception).
- **Exact-T1**:
  - "no lead settles at k ≤ 2": **fails** (deal 1's proved tie, under amendment 6);
  - Γ at k = 1 > 150‰ on ≥ 5/7: **met** (6/7);
  - survivors ≥ 3 at k = 1 on ≥ 5/7: **met** (7/7);
  - act ≠ law with gain ≥ 10‰ on ≥ 5/7: **met** (6/7);
  - the follow-root clause is below.

## Results: the follow roots

<!-- FOLLOWS -->

## The shape of the frontier

**How far trick 1 is from exact.** At the lead, the seat's best lawful plan is certified to
within 75–295‰ of its information-state optimum, on the five leads that are not proved and fit
B2. Γ roughly halves every one to two rungs. The ladder collapses to exact at k = 6 (seven own
tiles at the lead). The plays would allow it: they grow only ×1.3–3.4 per rung. **The leaves
would not.** They grow ×30–80 per rung after k = 1. So k = 3 would hold about 10¹⁰–5×10¹⁰
public histories per root, 0.4–3 TB of leaf map, against 10¹²–10¹³ plays. That is 1–15 min per root on
Metal at its measured rate (1.3–1.4×10¹⁰ plays/s, [[exact-gpu-2026-10-22]]), or 15 min–3 h on
12 CPU threads (1.24×10⁹). The bottleneck is the leaf map, not the compute.

**What would close it**, in order of leverage:

1. **The next reduction: reduce as you go.** The leaf map exists only to sum each public
   history over its worlds before the max. Turn the loop inside out. Walk the public-history
   tree once, carrying the list of worlds still consistent with it. Laws are deterministic, so
   each world goes down exactly one branch at each other seat's node. Reduce each subtree as
   it closes. Memory becomes O(worlds × the viewer's depth) instead of O(leaves). That is
   399M hands × 16 B ≈ 6.4 GB per level, or one copy if the list is partitioned in place. The work in plays is unchanged, and the whole ladder to k = 6 becomes a compute question.
2. **Lumpability.** Worlds that the declared laws cannot tell apart for the rest of the hand
   (the same futures under every line the viewer can play) are one atom. Counting by atom
   instead of by world shrinks the 399M factor itself. Exact-1's S0 course builder already
   merges such atoms below the viewer's turn.
3. **A tighter upper side.** The upper bound `god` is clairvoyant per world and costs 37–92%
   of the plays. How much of Γ is its looseness, rather than the lower side's, is not measured
   here. A lawful upper bound that is not clairvoyant would shrink Γ without
   any k. One example is the best response to the others' laws by information state, relaxed
   only at the horizon.
4. **A second type.** Under one type, the recorded lead is impossible at every follow root.
   A family of types (Exact-1's S1 library) changes the model, not the cost. It is what the
   follow roots need before their beliefs mean anything.

## Scope

- One declared type (all three others play `Lowest`), uniform deal prior, no auction, S0 at 30.
- Seven deals, one per pip declaration, from one pour. These are first deals, not a sample
  built for coverage; nothing here estimates a rate over deals.
- The value is make-or-set for the bid, not marks.
- The upper bound refutes; it never chooses.
- Cost: plays are the registered budget. Wall times are descriptive. Instruction counts from
  runs with `/usr/bin/time` inside `scripts/cpu-slot` measured `time` itself, not the runner,
  so none are reported (#249).

## Data

Per-tile counts and receipts (SHA-256 per tile gather), every rung's line, the scorecard script
and the follow-root log: [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10)
at `exact/trick1-2026-10-22/`. The gather bytes themselves (33 GB for deal 1's k = 2 alone)
are not published; their digests are.
