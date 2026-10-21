---
title: "Gus masked teaching breadth — 64 to 128 roots"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-21
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

Does a broader collection of exact contingent lessons help the same tiny learner generalize? [[gus-contingent-maskfit-2026-10-20]] established nearly complete acquisition on 64 TRAIN roots; [[gus-contingent-transfer-2026-10-20]] did not establish reliable fresh transfer. This comparison adds 64 new TRAIN roots while retaining the masked loss and 512-update budget.

This is the data-coverage branch of the textbook diagnostic: good training fit with weak unseen performance motivates testing more data. [Deep Learning §11.3](https://www.deeplearningbook.org/contents/guidelines.html) recommends substantial increments such as doubling. This is a compute-matched breadth comparison; it also reduces repetitions per root and can change the nonflat fraction. It does not isolate data quantity from optimization exposure or prove a general scaling law.

## Frozen comparison

- Original pretrained parent initializes one new masked-CE student. Original 64 TRAIN roots / 1,356 rows and teacher labels are reused exactly; the first 64 structurally eligible roots in a fresh 2,048-source pool are added. No selection by teacher gain or test behavior.
- Same 512 scheduled AdamW updates, batch 32, seed 42, root-uniform then within-root-uniform sampling. Only exact all-action-tie CE terms are zeroed; the full batch denominator remains. All scheduled optimizer steps occur.
- Historical masked64-final512 is the narrow control. Both endpoints are fixed final512; no development selection or checkpoint reselection. Architecture, learning rate, original inputs and fixed field remain unchanged.
- First 128 structurally eligible roots from a separate 4,096-source pool form the sole primary finite test. Wide must exceed both original parent and narrow by at least 0.5 percentage points, with each paired physical-root 95% lower bound positive. TRAIN old/new strata are descriptive acquisition diagnostics and cannot change the endpoint.
- No full-hand stage or promotion. A finite pass would require a separately registered actual-play confirmation.

The existing 624,843 reservations plus 6,144 new potential deals yield 630,987 future training exclusions, including rejected/unplayed. Original clean bid30 inputs; fixed G0 partner and pinned Zeb; uniform legal-consistent root masses carried through future observed field actions. Hidden worlds are summed before one shared learner maximum. The continuation teacher remains offline; deployed Gus has no search.

## Bounded execution and verification

Sequential segments: 45 seconds cooperative, 90 seconds outer; caps 48 collection, 2 training, 4 TRAIN diagnostic, 8 test. Pool shortfall, failure, unclosed attempt, runtime drift or exhausted cap stops the study for inspection. Exact runtime is Python 3.12.13 / Torch 2.14.1 / NumPy 2.5.4 via cpu-slot and the unresolved scratch/venv/bin/python path. Sources, reuse and outputs are hash-bound. Independent source review and serialized synthetic-resume tests precede registration and new responses. Completed old measurements are reused, not rerun.

Independent saved-data auditing reconstructed split/source order, original-label preservation, initialization/stream identity, training completion, complete-policy values, paired intervals and closed-output manifests. That audit does not itself constitute independent native execution, gradient replay or game replay. One training seed and historical control limit inference.

## Steering and capability refresh

Jason wants partner play with table talk to identify consequential decision nodes. A collaborator proposed a future equal-count comparison of talk-nominated versus ordinary examples. No human nomination corpus exists yet; this uniform-breadth study supplies a reference without treating dialogue as actor input. Plunge gameplay is separate work.

Lem207 remains closed with its09:48 reply; Book121 is pending and Silas166 open. Lem's new six-unit papers and report-only baseline/on-off columns concern another16×8 field net; no Gus semantics transfer. Existing finite verifier suffices. The prize remains discovery of better contingent decisions followed by learning them.

## Evidence

[Public registration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/28d64232f670a49153e351f49d22d985a1fb55fd/lem/gus-contingent-maskbreadth-2026-10-20/registration) freshly verified all11 files and 89 archive members before scientific responses. Registration SHA `21bdcd3d56dd45af5065ec18524c3b189b3d5883ceb1f1b9c2f745a03aa9dfb4`. Independent source review matched32 files and215 imported-source pins;11 synthetic tests and an isolated process startup check passed. Preserved metadata ancestry and import-order preparation failures were repaired before registration. Sources and endpoints remained frozen throughout the completed run. Local recovery: scratch/lem/gus-contingent-maskbreadth-v1/CURRENT.md. Preceding completed evidence is linked from [[gus-contingent-transfer-2026-10-20]].

## Result

All 23 scientific segments closed normally: 20 collection, 1 training, 1 TRAIN diagnostic and1 test. The added 64 TRAIN roots came from the first 687 source deals; the 128 TEST roots from the first 1780. The dataset contains 3,000 rows (420 nonflat, 2,580 flat), preserving the original 1,356-row prefix.

| Fixed policy | Mean exact success on 128 fresh finite roots |
| --- | ---: |
| Parent |37.3162%|
| Narrow 64 final 512 |36.5458%|
| Wide 128 final 512 |39.5881%|

Wide minus narrow is +3.0423 percentage points, paired physical-root 95% interval [+0.9060,+5.6521]. Wide minus parent is +2.2719 points [-0.2944,+5.5247]. The sole conjunctive gate **failed** because the parent interval crosses zero. No full-hand test, reselection or promotion. Against narrow, 17 roots improve, 11 worsen, 100 have equal values; against parent, 24/15/89. Equal values need not mean equal actions. Descriptive halves cannot rescue the pooled endpoint.

On original 64 TRAIN roots, wide 42.6641% versus narrow 43.0605% and teacher 43.1126%. On added 64, wide 37.5957%, narrow 30.2855%, teacher 37.7260%. Across all 128, wide 40.1299% versus teacher 40.4193%. Thus acquisition of the added lessons is nearly complete, with a small loss on old lessons. These are known-example values, not transfer evidence. Broader teaching has a favorable finite contrast with the historical narrow student; improvement over the pretrained parent remains unestablished. Fixed updates also reduce repetitions per root, and one seed limits inference.

Independent saved-data audit passed 6,272,340 checks. It reconstructs provenance, serialized initialization/stream/completion, saved policy outcomes, exact probabilities, intervals and closed-output hashes. It does not replay native inference, gradients or transitions; stored trick legality/points and later learner-action/cache correspondence are outside that reconstruction. Two auditor-only schema failures are preserved: scalar support identity lives in the companion receipt, while label support identity hashes canonical JSON rather than file bytes. Inspected repairs follow the original writers and retain public archive bindings. Scientific sources and results were unchanged.

[Verified evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/762b5a71a622a1ff6df8fed47e50ebe5b2a43e44/lem/gus-contingent-maskbreadth-2026-10-20/evidence) contains 12 files and 144,960 members. Original TRAIN/control archives are referenced, not repeated. Lossless publication retains every scientific member, reconstructs the full publication descriptor byte-identically, and preserves the inspected packaging failures under unchanged time/size limits. All 630,987 reservations remain future exclusions.
