## [2026-10-18 | ba31124a | Gus probability reward and lazy frozen continuations]
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

[[gus-success-probability-planning]] now selects direct verified expected
success plus the existing KL/rehearsal recipe; search and future-splat auxiliary
training are deferred. The trainer pins the reward choice and preserves complete
signal/resume state. Frozen anchor supplements append deterministic row/sample
continuations with unchanged nonlearner seating, full recipe identities, atomic
records, replay verification and immutable snapshots. `gus-training` documents
append, resume, recheck and the next matched probability-versus-correctness test.

[[gus-lazy-rlvr-smoke-2026-10-18]] records byte-identical lazy reconstruction,
paired fixed-seat continuations and an eight-round trainer smoke. Evidence is
published at the immutable Hub revision linked there; no strength claim or
model promotion. All 116 Lem tests and the required repository gates passed.
