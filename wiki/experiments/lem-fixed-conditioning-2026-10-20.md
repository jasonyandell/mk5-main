---
title: "The position baseline, done right: the squeeze and Gus re-read raw with a play-count baseline in the labeller (Lem, #233)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
---

Part of THE ALREADY-CLEAN TEST (#193; [[clean-neurons-42]]). [[lem-lead-units-2026-10-20]] found a
flaw in the conditioned audit of [[lem-conditioned-audit-2026-10-20]]: z-scoring each play count by
its own spread amplified noise where a unit is nearly silent. Every conditioned read since then
carries it, including CLAUDE.md's "24 neurons cleaner than 192 (0.549 vs 0.406)"
([[lem-squeeze-conditioned-2026-10-20]]). This test fixes the instrument and re-reads the headline.

Scope: these nets, these inputs and concept list v2. **The headline is quoted only as "reads
cleaner against list v2"**, never as support for "coarser, not mixed", until a coarse-versus-mixed
test passes ([[lem-unit-origin-2026-10-20]] is partial).

## The review before sealing (one turn; the reader read all 944 lines of the essay)

- **The fix is right but not enough.**
  - Standardizing activations at all works against the essay, which trusts a unit's "natural
    sparsity and activation ranges" and warns that percentiles flatten a neuron's raw shape.
  - Centring can erase real meaning in the field net, where the play count can be a unit's job.
  - Centring can bring the artifact back where a unit's stratum mean is strongly negative.
  - So the **primary read is raw**, with the play count handled in the labeller as a baseline.
    The pooled-scale conditioned read is secondary.
- **The headline prediction had no margin, and the old P5 was rigged.** P5's 0.15 tolerance on
  absolute shares exceeded the whole gap (0.143), so it could pass while the ordering flipped.
  Both are rewritten below.
- **The deeper point.** The essay predicts squeezed units grow *coarser, not mixed*, and that
  bigger nets have narrower, more recognisable neurons. A coarse unit can match one of 70 coarse
  concepts while a finer one cannot, so "narrow reads cleaner against list v2" may be vocabulary
  coarseness (#202). That is why the headline is quoted only as above, and why CLAUDE.md's framing
  has gone to the Mayor.

## Pre-registration

- **Instrument, two reads.**
  - **Read R, primary:** raw pre-activations; the top 1%; labels list v2 plus the raw input bits;
    and **the play-count baseline** (`audit --baseline n_plays`). A concept counts for a pole only
    where its precision on the top set beats, by ≥ 0.1, what the top set's play-count mix alone
    predicts for it, P(concept | play count) weighted by the top set's counts. Clean means a counting
    concept with precision ≥ 0.9.
  - **Read C, secondary:** pre-activations centred within each play count and divided by one
    pooled within-stratum standard deviation (`audit --condition n_plays`, now fixed), with no
    baseline.
- **Nets.**
  - The squeeze: the 21 trained drop-out nets and their 21 untrained twins (7 widths × 3 seeds),
    on the 434,535-row pip sample.
  - Gus: `pretrained-v0` at step 200,000 and three untrained nets, on its 43,927 test-split rows
    of pour A.
  - 20 shuffles on the 16×8 seed-2 net and on Gus's final checkpoint.
- **Predictions, sealed.**
  1. **The headline, read R.** The 16×8 mean clean share minus the 128×64 mean is at least 0.05,
     *and* 16×8 beats 128×64 in every seed pair (s0, s1, s2).
  2. **Floors, read R.** Every width's untrained mean is below 0.15, and so is Gus's.
  3. **Gus learned, read R.** Trained Gus is at least its floor + 0.20.
  4. **The artifact is gone, read C.** At most 0.05 of the s0 nets' layer-1 `+` poles have more
     than a quarter of their top rows below 5% of the unit's maximum. This is not automatic, since
     a strongly negative stratum mean can still shift rows.
  5. **The sign survives.** The 16×8 − 128×64 gap has the same sign under reads R and C.
- **Descriptive: raw shape.** For each layer-1 unit of the s0 nets at 16×8 and 128×64:
  - its firing rate;
  - its best counting label's precision at the top 0.1%, 1% and 5%;
  - a tag: **detector** if it fires on < 10% of rows, **dial** otherwise.
- **What the headline sentence will say.** The read-R gap with its seed pairs and floors, quoted
  "against list v2", with this page's caveat.
- **Cost.** 84 squeeze audits and 8 Gus audits on the Mac's CPU through `scripts/cpu-slot`, about
  2 hours.

## Results

Run 2026-10-20 as registered. Evidence is `lem/fixed-conditioning-2026-10-20/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/b0ce48462bd03b3181ad07a6b00e98eeb6ed7708/lem/fixed-conditioning-2026-10-20):
92 audits, the artifact check, `grade.py`, `grade.txt` and `grade.json`. Shuffled labels gave 0.000
on all 20 shuffles of the 16×8 seed-2 net and of Gus.

**Read R (primary): raw pre-activations, with the play-count baseline in the labeller.**

| width | neurons | trained, seeds 0 / 1 / 2 | mean | untrained floor (mean) | margin |
|---|---:|---|---:|---:|---:|
| 128×64 | 192 | 0.511 / 0.577 / 0.537 | **0.542** | 0.276 | +0.266 |
| 128×8 | 136 | 0.607 / 0.615 / 0.619 | 0.614 | 0.292 | +0.322 |
| 32×16 | 48 | 0.615 / 0.548 / 0.622 | 0.595 | 0.299 | +0.296 |
| 16×8 | 24 | 0.792 / **0.543** / 0.750 | **0.695** | 0.375 | +0.320 |
| 8×4 | 12 | 0.565 / 0.727 / 0.739 | 0.677 | 0.264 | +0.413 |
| 4×2 | 6 | 0.667 / 0.500 / 0.111 | 0.426 | 0.278 | +0.148 |
| 2×1 | 3 | 0.333 / 0.333 / 0.600 | 0.422 | 0.222 | +0.200 |

Gus, read R: trained **0.312**, untrained 0.017 / 0.021 / 0.004 (mean 0.014).

**Read C (secondary): fixed conditioning, pooled scale.** The 128×64 mean is 0.514 and the 16×8
mean 0.731, a gap of +0.216. Gus reads 0.327 against a floor of 0.216. All values are in
`grade.json`.

**The predictions, graded.**

| | measured | |
|---|---|---|
| 1. read R: 16×8 − 128×64 ≥ 0.05 *and* every seed pair | gap +0.153, but seed 1 reverses (0.543 against 0.577) | **wrong** |
| 2. read R: every floor < 0.15, and Gus's | field floors 0.222–0.375; Gus 0.014 | **wrong** (field) |
| 3. read R: Gus ≥ floor + 0.20 | 0.312 against 0.014 | holds |
| 4. read C: ≤ 0.05 of units with > ¼ near-silent top rows | 16×8: 0.0; **128×64: 0.213** | **wrong** |
| 5. the gap has the same sign under R and C | +0.153 and +0.216 | holds |

**Reading.**

- **The headline weakens, and it now says what it measures.** Read raw, against list v2:
  - the field net retrained at 24 neurons averages 0.695 clean against 0.542 at 192;
  - one of three seed pairs reverses;
  - the narrow nets' own floor is higher (0.375 against 0.276).

  Against each width's floor, the margins are +0.320 and +0.266, a difference of +0.054. The
  ordering mostly holds, by less than the old reads said.
- **The field nets' floor is not position.** Their untrained floors stay at 0.22–0.38 under the
  play-count baseline. Random field units are named by raw input bits, and an untrained integer MLP
  unit's strongest rows share input features. That is the honest floor for this architecture,
  higher than the 0.077 the flawed conditioned read gave.
- **Gus's floor is position, and the labeller baseline removes it.** 0.014 under read R, with
  trained Gus at 0.312. The baseline does in the labeller what conditioning tried to do to the
  activations, and it keeps the raw shape the essay asks for.
- **The conditioned read stays flawed for wide nets.** Under the pooled scale, 0.213 of the
  128×64 units still have a quarter of their top rows near-silent. This is the reviewer's
  negative-mean case: centring lifts rows near raw zero where a stratum's mean is strongly
  negative. **Read R replaces conditioning as the program's instrument.** Every earlier
  conditioned share is superseded where read R exists (the squeeze, Gus), and flagged elsewhere
  (#199, #228, #207, #231, #232).
- **Descriptive raw shape.** Every layer-1 unit of the 16×8 s0 net is a dial, firing on ≥ 10% of
  rows. At 128×64 seed 0, 29 of 127 are detectors.

The essay predicts squeezed units grow coarser, not mixed. This test reads cleanliness against a
fixed list, which cannot tell coarser from mixed.

Scope: these nets, these inputs, list v2.

## Links

[[lem-lead-units-2026-10-20]] · [[lem-conditioned-audit-2026-10-20]] · [[lem-squeeze-conditioned-2026-10-20]] ·
[[lem-gus-formation-2026-10-20]] · [[clean-neurons-42]]
