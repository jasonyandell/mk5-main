---
title: "Gus value-versus-routing: stopped instrument audit"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

This registered study is terminally complete as a stopped instrument audit;
its primary model comparison was never reached.

## Question and registered prediction

Following [[gus-route-rescue-2026-10-20]], the registered diagnostic was to
separate what the final actor query of layer-0/head1 reads from how strongly it
routes to that source. The single primary prediction was that, in cosine-final
Gus, a dose-0.5 value-only interpolation from the paired other-lead token would
lower the positive-minus-negative paired signed raw pre-GELU U65 contrast in
both existing panels. The endpoint was raw activation units, not standard
deviations. **The cosine model was never measured, so this prediction is
unevaluated, not failed or supported.**

The frozen design used the same 48 inspected contexts: 12 matched
positive/negative physical pairs per old panel, with original parent followed by
cosine-final. At the final actor query of layer-0/head1 it specified
routing-only score attenuation, value-only interpolation toward the paired
other-lead token's first-block V at the same recipient position, and their
combination. TILE and preceding ACTOR sources, doses 0.25 and 0.5, and exact
identity/self shams were registered. Recipient tokens, position, Q/K and all
other query/head writes were held fixed. The donor V uses only the donor token
embedding plus recipient position through the first block's LayerNorm and QKV;
it imports no other donor-row history. This remains an internal
counterfactual, not an alternate lawful hand or semantic-concept isolation.
The [frozen registration, SHA256 `2bb988f3…`](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/a2e15dea3afb271c4d7505fd9c6122c1d2b6f5c3/lem/gus-value-route-2026-10-20/registration)
was anonymously verified before trained responses.

## Stop and partial coverage

The first instrument run stopped at a numeric/nonfinite assertion during
independent-forward audit. It completed 42 immutable original-parent row/model
records: all 24 development sides and 18 confirmation sides (pairs 00–08).
Each accepted record contains 20 intervention cases plus its baseline: **840
intervention cases and 42 baselines, 882 accepted forward cases total**. There
are no cosine-final records. All accepted record checksums and audit flags pass;
their maximum hook-versus-functional forward difference was `1.34706497e-5`
against the frozen `2e-5` tolerance, and maximum local-write algebra error was
`3.64190755e-7`. These figures describe accepted cases only.

The next unit in the frozen order was original-parent, confirmation pair 09,
positive row `292:6`. The assertion was `err < 2e-5 and all(isfinite(...))`.
The failure receipt did not save the failing condition, computed error,
per-field differences or finite-value flags. Therefore the exact condition,
whether tolerance or finiteness triggered, and the discrepancy magnitude are
unknown. No specific numerical cause can be inferred. The row was not
atomically committed. There is one failed attempt and no normal completion
receipt; no automatic retry, resume, summary or threshold change followed.
The preserved failure archive, including the frozen instrument and partial
records, was anonymously verified at immutable dataset revision
[`6e9b3a68fa11b965e9242b773205cad5df05e8d6`](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/6e9b3a68fa11b965e9242b773205cad5df05e8d6/lem/gus-value-route-2026-10-20/failure-evidence).

The registered primary comparison cannot be evaluated from the partial
records. There is no semantic, route, output, cleanliness or player conclusion
from this run. In particular, the failure is not evidence for or against its
registered value prediction. The existing 48-row inputs, original/cosine
checkpoint pair, G0/Zeb field and 607,306 reserved-root exclusions were not
changed; no training, game, new root, player search or policy change occurred.

An earlier, unpublished preparation and its reason note are preserved under
`scratch/lem/gus-value-route-v1/preparation-v1/`. Its guard review and repair
preceded final registration; that preparation was not a trained-model
measurement. Keep it distinct from this registered run's `failure.json` and
unclosed attempt. A later, separate fixed-row diagnostic measured one float32
threshold crossing on row `292:6`; see [[gus-value-numerics-2026-10-20]].
Because the original attempt did not save its condition or discrepancy, the
later result does not identify what caused the original failure.

## Audit disposition and next step

The saved partial output is evidence only for those accepted original-parent
rows. Its maximum accepted audit error does not characterize the unsaved
failing condition. The separate completed numerical diagnostic is not a resume
or completion of this value/routing comparison and does not evaluate its
primary prediction. Any later registered value/routing test remains a separate
decision.

The same [First Look report](https://gus-first-look.psychometrix.chatgpt.site)
includes this stopped study and separate diagnostic; its owner-private
deployment succeeded from source `4ce39042b6002fb7e74cae08e37f2f42cbc4e414`. See also
[[gus-route-rescue-2026-10-20]] and [[gus-book-silas-tools]].
