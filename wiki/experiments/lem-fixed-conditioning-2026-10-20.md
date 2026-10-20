---
title: "The position baseline, done right: the squeeze and Gus re-read raw with a play-count baseline in the labeller (Lem, #233)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: active
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

Not yet run.

## Links

[[lem-lead-units-2026-10-20]] · [[lem-conditioned-audit-2026-10-20]] · [[lem-squeeze-conditioned-2026-10-20]] ·
[[lem-gus-formation-2026-10-20]] · [[clean-neurons-42]]
