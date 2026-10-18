---
title: Texas 42 and Walt — compact mathematical notes
kind: topic
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---

<!-- SPDX-FileCopyrightText: 2026 Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Publication

[Texas 42 and Walt: finite mathematics](../../docs/math/README.md) is a three-file,
CC0 manuscript: a short introduction and two self-contained mathematical notes.
[42](../../docs/math/42.md) covers incidence, calling versus power, unique trick winners,
scoring, finite play, exact allocation counting and measure-preserving query reduction.
[Walt](../../docs/math/walt.md) covers lawful policies, cell recursion, coherent beliefs,
value geometry, response stabilization and statistical certification.

The source snapshot is `28edece47b5cf11d57f32095d835b82d7e169bf0`. Three Sol audits
screened the repository's mathematical sources; Astra consolidated the manuscript and
independently reviewed proofs. The notes distinguish derived arguments, abstract Lean
theorems, implementation connections and empirical evidence. The fifteen exposed Lean
theorems passed a fresh `scripts/check lean`; concrete game and runtime refinement remain
outside that formal proof. Internal AI review is not external peer review.

The final manuscripts total 1,900 whitespace-delimited words. Author and proof reviewer
iterated before a fresh Astra adversarial review; all three approved the revised text.
Review resolved cell-value normalization, action-conditioned holdout policies and hidden
random-tape restrictions, then corrected two stale wiki summaries. No material finding
remained within the stated scope. `scripts/check --fast` passed wiki, conflict-marker and
license checks; pre-existing wiki warnings remain outside this publication's scope.

## Corrections and boundaries

The audit corrected current summaries in [[suit-algebra-spec]], [[play-phase-algebra]],
[[texas-42]], [[strategy-fusion]], [[belief-policy-value-algebra]],
[[endgame-equivalence-census]] and [[match-equity]]. Incidence symmetry does not preserve
numeric lead selection; fixed-clockwise seats admit rotations, not reflections.
All-four-clairvoyant minimax has no general bound against a fixed field. Posterior movement
does not imply information growth, and a finite symmetry census does not prove that hidden
worlds never compress. Match-value comparisons require the stated continuation assumptions.

Historical snapshots remain unchanged. In [[research-night-2026-10-12]], the one-path-per-world
complexity argument applies to a deterministic field; stochastic exactness also requires
integrating its randomness. Enumerating deals alone neither performs that integration nor
supplies behavioral likelihoods. The stabilization theorem requires one fixed operator,
including fixed settings, samples and tapes, rather than heterogeneous deployed rungs.
The historical oracle symmetry report is not authority for scored-game relabeling.

[[scenario-player]] · [[silas]] · [[splat-search-bounds]]
