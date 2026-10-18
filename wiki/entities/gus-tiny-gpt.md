---
title: "Gus — the tiny history-conditioned Texas 42 GPT"
kind: entity
first_seen: 2026-10-18
last_updated: 2026-10-18
status: active
---

## Identity

Jason named the current [[lem-the-teacher]] nanoGPT model **Gus** on 2026-10-18.
This is the October 107,392-parameter, 500k-deal next-tile observer implemented
in `lem/observer/`. The April oracle-era [[gus]] project is separate history;
its code, weights and findings are not this model's.

Public model: [jasonyandell/gus](https://huggingface.co/jasonyandell/gus),
release `pretrained-v0`. The initial release is supervised pretraining. The authorized reward/depth campaign
is registered in [[gus-population-depth-2026-10-18]].
The card includes safe-format weights, an adapter-compatible inference checkpoint,
exact input/output schema, a pinned loading example, provenance and scoped metrics.
Weights are CC0; the nanoGPT-derived implementation retains MIT attribution.

## Routes

- [[lem-observer-corpus-squeeze-2026-10-17]] — G0's training and formation archive.
- [[lem-observer-first-h2h-2026-10-17]] — measured play versus the predecessor.
- [[lem-nanogpt-foundations]] — clean inputs, math, pending splat/Nello and Zeb methodology.
- [[gus-population-iteration]] — latest five generations plus permanent Zeb; proposed first reward round.

## Release verification

Release commit [`378813257486`](https://huggingface.co/jasonyandell/gus/tree/37881325748671214bcedddf3953615958eeaaf7),
tag `pretrained-v0`: 11 release files, 900,351 bytes. Fresh anonymous download
verified every checksum; safetensors and adapter checkpoint tensors match
exactly; the CPU float32 opening-prefix logits match the source model exactly.
The pinned model-card example selects six-six. Verification used PyTorch 2.14.1
and safetensors 0.8.0. No new training or H2H games were run for publication.

Source full checkpoint SHA256:
`99fd106a15ae468dca9c02426541816f1121a2cdef0cc7b96fd37f2122f31d8d`.
Exported safetensors SHA256:
`8ea269f24892d1e852d1ab7d2de3c3963298d01a85cf29388d9c9843070fa9f0`.
[Release checksums](https://huggingface.co/jasonyandell/gus/blob/37881325748671214bcedddf3953615958eeaaf7/SHA256SUMS).
