---
title: "Gus: did the actor acquire the computed TRAIN decisions?"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->
# Gus: did the actor acquire the computed TRAIN decisions?

Registered and launched at 18:40:01 UTC. [Public registration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/ac44438dbb93af520d8ea200eb556053ad325d7b/lem/gus-contingent-fit-2026-10-20/registration) freshly verifies all 10 files and 55 archive members. Parent and independent static reviews cleared the source; five synthetic recovery/value/input tests passed before scientific calls. Recovery: `scratch/lem/gus-contingent-fit-v1/`.

The prize is to use continuation machinery to discover better contingent decisions, then train Gus on them. The [cost comparison](gus-contingent-cost-2026-10-20.md) completed training and development but exhausted its reserved test pool at 24 of 32 eligible roots. Its transfer result remains unevaluated. This diagnostic asks a prior question: did the actor acquire the useful decisions in its existing training examples?

## Fixed scope

Inspect five saved policies: original parent, cost step 16 selected on development, cost final step 64, teacher-label CE step 48 selected on development, and CE final step 64. Final snapshots are diagnostic specimens, not newly selected candidates. Use all 32 original TRAIN roots and 622 discretionary unsettled prefixes, with their unchanged actor inputs, exact teacher Q values and sealed fixed-field caches. There is no training, new collection, held-out analysis, temperature change or checkpoint reselection. All 609611 reserved deals remain future training exclusions.

Categories are fixed from the teacher Q values and original parent: 18 strict corrections, 48 other non-flat decisions, and 556 all-action ties. Root-balanced local measurements distinguish probability assigned to optimal actions, actual legal argmax repairs, and newly introduced consequential mistakes. Disagreement on an all-action tie is not a utility error.

## From local choices to the complete policy

Capture each native actor call once, including tokens, deterministic position metadata where used, raw tile logits and the returned legal action. Reconstruct every derived record from saved raw output on resume. The actor receives no Q values, hidden world or referee state as input.

Evaluate each captured policy across each existing root's exact worlds using only recorded G0 and pinned Zeb responses. Cache misses stop the diagnostic. Compare the existing settlement evaluator against an independent full legal-tail walk; canonical tails are permitted only after the contract outcome is publicly settled. Check that occupancy-weighted local teacher regret sums exactly to the teacher's root value minus the student's complete-policy value. This distinguishes errors on actual paths from errors in unvisited branches.

The maximum is 3110 actor forwards and 160 policy/root evaluations, in two sequential segments of 45 seconds cooperative and 90 seconds outer, through `scripts/cpu-slot`. Each small root walk is atomic within the segment; failure, an unclosed attempt or exhausted cap stops work. A separate saved-evidence audit adds no actor calls or game replay.

## Interpretation boundary

Perfect TRAIN policy value would establish acquisition under this finite teacher. Residual TRAIN regret identifies unfinished acquisition or interference, whose magnitude and source should guide the next comparison. Neither outcome proves new-hand generalization or actual full-hand strength, and this cannot rescue the stopped cost test. No player promotion follows from this diagnostic.
