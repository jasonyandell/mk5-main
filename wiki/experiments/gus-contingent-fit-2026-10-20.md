---
title: "Gus: did the actor acquire the computed TRAIN decisions?"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
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

## Measured acquisition

All 3110 native captures and 160 root values completed in one 6.839-second segment, with no failures or field calls. Both independent value walks and every exact occupancy identity agreed. The independent saved-data auditor reconstructed 187637 checks across all 6541 output files without loading models or replaying games. It checks saved outputs and arithmetic, not the neural forward implementation. Its pre-execution provenance lookup correction and original source are preserved.

| Fixed policy | Complete TRAIN success | Gap to exact teacher | Corrections repaired /18 | New non-flat errors /48 |
|---|---:|---:|---:|---:|
| Parent | 36.1636% | 2.4673 pp | 0 | 0 |
| Cost selected16 | 37.1937% | 1.4373 pp | 9 | 4 |
| Cost final64 | 37.3557% | 1.2752 pp | 14 | 4 |
| CE selected48 | 37.5401% | 1.0909 pp | 13 | 1 |
| CE final64 | 37.6558% | 0.9751 pp | 14 | 1 |
| Exact teacher | 38.6310% | 0 | — | — |

Both final students acquired 14 requested consequential corrections. Cost final assigns more optimal probability on the correction subset (87.40% versus CE final 74.46%, root-node weighted), yet its complete TRAIN policy has lower value. Local probability, argmax repair and visited-path outcome are distinct. Remaining missed corrections and newly introduced errors both matter. This supports acquisition under the finite training model; it does not identify the cause of interference, establish transfer, select final64 for deployment, or promote a player.

[Public evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/7cf364398d400a0de0cd9e7213c98fcf6c32532e/lem/gus-contingent-fit-2026-10-20/evidence) freshly verifies 9 files and 6565 archive members. Prior source data and checkpoints are linked by immutable identities, not duplicated. The next candidate question is how to preserve already-good consequential decisions while acquiring the remaining corrections; no next recipe is selected by this descriptive readout.

The [report chapter](https://gus-first-look.psychometrix.chatgpt.site/#contingent-fit) is deployed from `d25b592cde78afdbfa0889d617a09005e82e2bb5`, preserving owner-private access. New mobile score cards, navigation and disclosure passed 390px visual/interaction QA; existing JS controls were unchanged.
