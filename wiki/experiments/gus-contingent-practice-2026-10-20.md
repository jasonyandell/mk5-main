---
title: "Gus contingent decisions — more practice on the same lessons"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

Does a longer teacher-action CE training budget on the same 64 exact teaching roots improve complete contingent decisions on fresh positions?

[[gus-contingent-fit-2026-10-20]] demonstrated partial acquisition on familiar positions. [[gus-contingent-breadth-2026-10-20]] did not establish transfer after 64 updates on 64 roots. Saved minibatch losses continued falling, but changing minibatches and clipped gradients do not establish underfitting or predict playing improvement. The next lever is additional optimization on the same discovered decisions. The comparison follows the distinction between a training surrogate and the validation objective in [Deep Learning §8.1.2](https://www.deeplearningbook.org/contents/optimization.html).

## Registered comparison

The short arm reuses the completed breadth trajectory at steps 0/16/32/48/64. A new long arm forks its exact step-64 model, optimizer, RNG and metric history, then performs only 448 additional updates. Its selectable candidates are 0/128/256/384/512. Both offer five candidates; the grids differ and are not nested. This compares the declared budget and selection procedures, with one training seed and a historical control.

All 64 TRAIN roots, 1356 rows, teacher labels, input encoding, parent, optimizer and minibatch generator remain fixed. The extended stream must reproduce the original 64-step prefix exactly. No completed training or teacher solving is repeated. Ordinary legal teacher-action CE, root-uniform then node-uniform sampling, batch 32, seed 42, AdamW 1e-4 and clipping at 1 remain unchanged.

Fresh 32-root development selects each arm by exact complete-policy success, with earliest exact ties. Only then does 64-root test open. Long must exceed parent and selected short by at least 0.5 percentage points, with both paired physical-root bootstrap 95% lower bounds positive. Only a pass opens 128 actual full hands, with separate 2-point parent and 1-point short-control gates and positive lower bounds. No test reselection or automatic promotion.

## Information, exclusions and caps

The finite teacher uses the explicit uniform legal-consistent root prior carried through observed field actions. Indistinguishable worlds are summed before choosing one shared learner action. Original clean bid30 inputs, fixed G0 partner and pinned Zeb opponents remain unchanged. Deployed Gus has no search. Acquisition, finite-model transfer and full-hand strength are separate claims.

Fresh pools reserve 1536 development, 2048 test and 128 conditional full-hand deals after all 613323 prior exclusions. All 3712 reservations become future training exclusions, totaling 617035, including rejected and unused deals. Eligibility and the bounded learner-last trick-five endgame remain unchanged. No targeted selection by teacher advantage.

Caps: 24 collection, 2 training, 4 development, 4 test and 8 conditional full-hand segments, sequential 45-second cooperative and 90-second outer limits. Runtime, sources and closed-output snapshots are checked through cpu-slot. Failure, insufficient eligible roots or exhausted caps stops without automatic retry or growth. The prior cost study remains stopped; its 24 roots are not a replacement test.

## Status and evidence

[Public registration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/859a7495c8d226d8d0e6386a0135a45199d3fba3/lem/gus-contingent-practice-2026-10-20/registration) `859a7495c8d226d8d0e6386a0135a45199d3fba3` freshly verified all 12 files and 60 archive members before launch. Registration SHA256 `c64a37a16677e8287033e2acf52214b0abbe9f7a46d75538e962d56b02f0b8d9`. Independent source review and seven synthetic tests passed, including exact fork/resume state parity and retained fork-origin bytes. A preparation-only receipt collision was preserved and repaired by limiting test hashes to explicit science/test sources; no scientific evidence was changed.

Controller launched at 2026-10-20T19:43:37Z after the pinned runtime check. It stopped during the fourth attempted segment, before any new optimizer update. The scientific question is UNEVALUATED. Recovery root: `scratch/lem/gus-contingent-practice-v1`.

Lem's new field-net unit result concerns a different architecture and information scope; it supplies no recipe or neuron claim for this comparison. Book's pending reply and Silas's unmerged constructor do not block the existing exact verifier. Capability refresh and the collaborator's prize check are retained with preparation evidence.

## Training-start stop

Three collection segments completed all 32 fresh development roots and preserved the 64 TRAIN roots / 1356 rows. Training initialization passed a JSON string path into a hash helper that requires a Path object. The resulting AttributeError stopped the controller before creating a training directory or running an optimizer update. No checkpoint selection or test collection occurred; there is no transfer result or promotion.

The original failure, unclosed attempt 003 and frozen sources remain intact. Synthetic tests had exercised the fork helper and dataset loader separately, missing the full initialization entrypoint with JSON path values. Any repair must test that boundary and receive a separate registration; it may reuse completed collection under unchanged criteria and remaining caps. This stopped run is not resumed silently. Independent saved-data audit passed 4235 stop checks and 466224 certificate/root checks across 96 selected roots, 2360 dataset rows and 49127 snapshot entries. It confirms 339 scanned development candidates, 32 accepted, unchanged TRAIN rows and exact equality with the last closed output snapshot. No native model or game replay was performed. [Verified failure evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/4c45a7507a2c675cd8f58be75375b30679a5d638/lem/gus-contingent-practice-2026-10-20/failure-evidence) `4c45a7507a2c675cd8f58be75375b30679a5d638` freshly verifies 11 files and 23463 archive members.
