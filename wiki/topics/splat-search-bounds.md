---
title: "Searching over splats: what is sound, what is tractable"
kind: topic
first_seen: 2026-10-13
last_updated: 2026-10-13
status: active
---

## What this is

The answer to "can a player pathfind with splats": an admissible heuristic between margin distributions, dominance pruning, or a best-first (Dijkstra) exploration of the heaviest lumps of worlds. Asked 2026-10-13; the question had been re-derived twice before. A splat is the per-world margin or make distribution of a move ([[splat-visualizer]], [[metasplat-2026-10-13]]); the player is Walt ([[scenario-player]]). The probe measured seat 0, tricks 2-4, champion pour `walt:k1,n512,m8` and the netfield pour.

## Verdict

- **Sound:** world-by-world dominance under fixed continuations; first-order stochastic dominance (FSD) under a fixed belief and fixed continuations; lump envelopes as upper bounds; an unexplored-mass bound with exact masses and one fixed plan's values.
- **Unsound:** average double-dummy (DD) value as an admissible heuristic; dominance on a sampled cell as zero-risk; summed per-lump plans as the value.
- **Tractable:** trick 4 and later, full cell, cheap, provided the planner accepts explicit world sets. Trick 3 plausibly seconds (estimated). Trick 2 minutes (estimated); mass-ordering gains almost nothing over enumeration there. Earlier than trick 4, statistical certification is the realistic route.

## Prior work, per hypothesis

### Paired dominance prunes soundly, under fixed continuations

World-by-world dominance is proved in texas-42 `walt/math/decision_sparse_exact_solving_v0.1.md` section 6.2 (line 535) and 6.3, only under fixed continuation policies. Per-world-optimal continuations are unsound (errata non-theorem E4', line 487). Implemented as `Dominated` in texas-42 `walt/walt/src/solver/field_swap.rs:914`, on exact fibers only.

### Distances bound differences, not subtrees

W1 and Kantorovich-Rubinstein (KS) distances bound the difference in mean and in P(make) between two known splats. True, but not a heuristic for an unexplored subtree, and they discard the pairing.

### Anytime bound from unexplored mass times range

- texas-42 proved and built it: `anytime_proof_state_score_v0.1.md` sections 6-9, 22, 42 give lump envelopes. Section 20 warns that the per-threshold best tile is not playable by one policy. The frontier (`1ceadb7c`) is ordered by bound-per-cost, not mass. Slice F (`246cb260`): at the opening, 80.5% of mass is action-exact at 36,923 classes; the residual fragments into 116,280 singletons.
- mk5: the cell floor's Prop F1 (`10-theory.md` ~722), and Theorem C2.1 (`:654`), a statistical certificate with epsilon = 7.4 pp of P(make) at K=512 that "essentially never fires early" (`:690`).

### Average DD is not admissible against the fixed field

The average DD value over the cell is not an admissible upper bound against Walt's fixed stochastic field: texas-42 errata C4 (~450) and section 8.2 (729); [[belief-policy-value-algebra]] lines 107-117. What is valid is the mover clairvoyant against the fixed field with the root action common (Lemma E3), gap 6-22 per mille at trick 4 (`walt/FACTOR-BELIEF.md` ~604). Silas's double-dummy solver is fenced for this reason: `ClairvoyantSolver` and `(clairvoyant-dd-make)` sit behind a cargo feature and carry the [[strategy-fusion-fence]] token.

### Search respects information sets

One tile per decision across indistinguishable worlds: mk5 Lemma W (`10-theory.md:180`), texas-42 Lean `sumBeforeMax`.

## Probe measurements, 2026-10-13

Every run under `timeout 300`. Code and outputs: `scratch/splat-search/`, published 2026-10-14 as `zeb/splat-search/` in [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10).

### Legal moves pruned by dominance

| dominance test | t2 | t3 | t4 |
|---|---|---|---|
| world-by-world on margin | 1.2% | 3.7% | 7.6% |
| on make | 8.7% | 20.5% | 33.3% |
| FSD on marginals | 29.5% | 34.8% | 37.3% |

- On sampled cells it is not zero-risk: 5.6% (margin) and 14.2% (make) of dominances seen on 256 worlds fail on the other 256.
- Only enumerated cells are exact: 0% / 0.4% / 13% of positions at t2 / t3 / t4.
- Detecting dominance requires planning every move, so it saves nothing at the root.

### Lumps by mass

- Lumps by holdings of the remaining trumps and count: median per cell 175 / 53 / 24 (t2 / t3 / t4). Only about 54% of the mass is in lumps whose make is constant.
- Lumps by full value vector: nearly singletons.
- The hard bound closes linearly in mass whatever the order. Separating the best move needs unexplored mass below a median make gap of 0.02, so it separates in 55-69% of positions at a median 98% explored; ties never separate.
- Exact cell sizes, median: 2^22.4 / 2^16.5 / 2^12 worlds.
- Planning lumps separately inflates value (+0.6 to +3.5 pp), so it gives an upper side only.

### Average DD against the planner

Not a bound in either direction. The planner beats average DD's best at about 28% of positions, about 14% beyond 2 SE. The mean gap is +1.6 to +1.8 pp. Argmax agreement is 68 / 77 / 86%.

## Smallest next experiment

A trick-4 sandwich over the full cell (at most 2^16 worlds) at k0. Lower side: the 1,024-world plan replayed on every world. Upper side: summed holdings-lump plans. Count how often the best move's lower side beats every other move's upper side. Before building it, check texas-42's `frontier.rs` (`1ceadb7c`).

## Links

[[splat-visualizer]] | [[metasplat-2026-10-13]] | [[candlewax]] | [[scenario-player]] | [[belief-policy-value-algebra]] | [[endgame-equivalence-census]] | [[the-wall]] | [[strategy-fusion-fence]]
