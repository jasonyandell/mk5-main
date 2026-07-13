---
title: Scratchpad Validation v2 Iteration 0 (retired)
kind: experiment
first_seen: 2026-04-11
last_updated: 2026-04-11
status: retired
---

## Summary

A [[star]] iteration with engine-verified scratchpad validation layered on top of [[k1-grading]]. The model was required to produce a structured HAND / VOIDS / COUNTS / PLAY scratchpad before committing to a play; each claim was checked against engine ground truth. The experiment failed due to format-bootstrapping: the model had never seen the scratchpad format and could not produce one correctly.

([sources/380f3fa](../sources/380f3fa.md) through [sources/78ba940](../sources/78ba940.md))

## Setup

- **Format:** narration v2 with ground-truth fields embedded; model prompted to fill HAND / VOIDS / COUNTS / PLAY sections before deciding
- **Validation rules** (from [[380f3fa]]):
  1. HAND must match remaining dominoes exactly
  2. COUNTS must correctly track which count dominoes are played/out
  3. PLAY must be legal
- **Grading categories:** `valid_pass` (gold — keep), `valid_fail` (rationalize), `invalid` (discard — wrong facts), `illegal` (discard), `parse_fail` (discard)
- **Infrastructure:** [[star-harness]] on [[modal]] B200; random subset sampling per iteration via `--subset N`

## Results

- `invalid` rate: 64.5% (facts wrong in scratchpad)
- Training traces that qualified: 5 (out of the iteration's full set)
- 5 traces is too few to produce a useful LoRA update

## Interpretation

The model produced incorrect scratchpad facts not because it lacks rules knowledge but because it had never seen the scratchpad format. Imposing strict validation before format-bootstrapping starves the training pipeline. This is a sequencing failure, not a validation design failure.

The 64.5% invalid rate is consistent with the illegal-rate diagnostic from [[discard-illegal-traces]]: the model's rules comprehension is still shaky, and adding a new output format simultaneously compounds the difficulty.

## Retirement path

1. **[[b12fcec]]** — relax to hand-only validation; counts logged but not used to reject traces
2. **[[78ba940]]** — full revert to simple [[k1-grading]]; scratchpad code retained for later; random subset sampling added
3. **[[5946c94]]** — `iterate.sh` switched back to v1 narrations

## Lesson

Format-bootstrapping must precede fact-validation. The model needs to produce well-formed scratchpads before those scratchpads can be validated. See [[scratchpad-validation]].

## Related pages

[[lem]] · [[star]] · [[k1-grading]] · [[scratchpad-validation]] · [[discard-illegal-traces]] · [[learned-by-playing]] · [[380f3fa]] · [[b12fcec]] · [[78ba940]] · [[5946c94]]
