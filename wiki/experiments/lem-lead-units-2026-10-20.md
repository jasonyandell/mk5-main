---
title: "Already-clean test 7d: the two lead-plus-trick-state units, read in both directions — one thing, two related jobs, or packing? (Lem, #232)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: active
---

Part of THE ALREADY-CLEAN TEST (#193; [[clean-neurons-42]]). [[lem-unit-origin-2026-10-20]] left two
broadened units that passed parts 1–3 of its refutation clause. Since then:
- part 3 was shown to misread the essay (see that page's correction);
- both pairs turn out to be **mutually exclusive**: `lead` co-occurs with b on 0 of 434,535 rows,
  because when you lead, nothing is on the trick.

This test reads both units the way the essay reads a neuron, upstream at how it's built and
downstream at what it does. It ends in one of four verdicts.

Scope: these nets, one position against uniform beliefs, no history (#202), lists v1 and v2,
signals conditioned on the play count.

## The review before sealing (two turns; the reader read all 944 lines of the essay)

- **Turn 1** changed what the test asks.
  - Exclusivity alone shows only that a pair *could* be packed. What decides it is "why its things
    are together in the first place": locality, formation, reuse and the negative pole.
  - Absence of explanations is "superposition of the gaps", so a packing verdict needs a positive
    signature.
  - It added exact upstream votes, per-half downstream use with cancelling weights, partner
    specificity, raw shape instead of the percentile window, and recurrence across nets.
- **Turn 2** removed constructions that decided verdicts before any data was read.
  - Partner rank on the same top 1% that chose the union is circular. So partners are ranked on
    held-out deals and judged by near-ties.
  - The `lead`-exclusive pool is nearly all trick-state, so its composition is published.
  - "Cancelled on one half" is also differentiation. So packing requires **demultiplexing**, with
    targeted, baselined cancellation.
  - The verdicts are made exclusive with a precedence.
  - The offshoot mass rule is dropped for units born coarse.
  - Recurrence excludes the units' bit-identical twins and gets a chance baseline.

## Pre-registration

- **Units.**
  - **C1:** 16×8 seed 0, layer-1 unit 14 (`lead`, `ten-on-trick`).
  - **C2:** 128×64 seed 0, layer-1 unit 59 (`lead`, `safe-trick`).

  Both are final checkpoints of the formation runs. b is the half that arrived later in
  [[lem-unit-origin-2026-10-20]]. Rows are split by deal into **fit** and **held-out** halves
  (seed 20261020). Top sets are the top 1% of the conditioned signal.
- **Check 1, upstream votes.** On each half of the held-out top set (A: `lead`; B: b), list the 10
  inputs with the largest mean contribution wᵢx̄ᵢ. A **shared input** carries both halves: its
  weight is in the unit's top 10% by |w|, and its activity on each half exceeds its activity on
  the unit's non-firing rows at the same play counts by at least 0.3. The halves sit at different
  positions in the trick, so the position bits separate them, and that is recorded rather than
  scored.
- **Check 2, downstream use and cancelling.**
  - **Per-half use:** the KL divergence from the base move distribution over legal tiles when the
    unit is ablated, mean over each half's rows.
  - **Readers:** every layer-2 unit i with w₂[i, j] ≠ 0, either sign. Its *other vote* is
    z₂ᵢ − w₂[i, j]·a₁ⱼ.
  - **Targeted cancellation on a half:** the reader's mean other-vote on that half is below its
    mean on the unit's non-firing rows, at the same play counts, by at least one standard
    deviation of that baseline. The reader also has w₂ > 0 (for w₂ < 0 the sign is reversed).
  - **Demultiplexing:** the readers that carry ≥ 0.8 of Σ|w₂[:, j]| split into two disjoint groups,
    each cancelling a different half, and no reader is targeted-positive on both halves.
- **Check 3, partner specificity on held-out deals.** The pool is every v1 or v2 concept that
  co-occurs with `lead` on fewer than 0.5% of `lead` rows; its composition by family is published.
  On the held-out top set, compute the union precision of {`lead`, c} for each c in the pool.
  - **Near-ties:** the count of partners within 0.05 of b's union precision, and how many families
    they come from.
  - **Distant control:** the best union of `lead` with a concept from score, voids or doubles,
    matched on base rate, with exclusivity not required.
  - **Specific:** b leads by > 0.05, or its near-ties are all trick-state.
  - **Arbitrary:** near-ties from ≥ 3 families, or the distant control within 0.05 of b.
- **Check 4, raw shape.** For each half, its **rarity** (share of rows) and its **weakness** (mean
  raw a₁ per row as a fraction of the unit's maximum), reported separately.
- **Check 5, recurrence.** Count the broadened layer-1 units whose best union is `lead` with a
  trick-state concept, in the six formation runs and 21 drop-out nets. C1's and C2's twins (the
  drop-out 16×8 s0 and 128×64 s0 nets) are excluded. The baseline is the same count for `lead` with
  a concept from any other family.
- **Verdicts, in order of precedence, sealed.**
  1. **Packing:** an arbitrary partner, *and* demultiplexing, *and* no recurrence beyond baseline.
  2. **One thing:** a shared input, *and* a specific partner, *and* no demultiplexing.
  3. **Two related jobs:** a specific partner, *and* recurrence above baseline, *and* no
     demultiplexing.
  4. **Unresolved:** anything else.

  A packing verdict is scoped as the toy-world or bad-learning boundary of a tiny MLP over
  hand-built features, and as pending #202.
- **Predictions.**
  1. Both partners are specific (check 3).
  2. Neither unit demultiplexes (check 2).
  3. `lead` + trick-state recurs above its baseline (check 5).
  4. Neither unit gets the packing verdict.
- **Cost.** Two nets, a few forward passes each.

## Results

Not yet run.

## Links

[[lem-unit-origin-2026-10-20]] · [[lem-formation-2026-10-17]] · [[lem-dropout-order-2026-10-17]] · [[clean-neurons-42]]
