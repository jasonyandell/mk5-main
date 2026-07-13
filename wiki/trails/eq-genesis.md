---
title: "E[Q] Genesis — Era 3 (2026-01-09 .. 01-31)"
kind: trail
first_seen: 2026-01-10
last_updated: 2026-01-31
status: complete
---

## Overview

Era 3 is the era where E[Q] got built. Not a bead tree, not a design doc — `forge/eq/` from
nothing, a marginalization fix that gives "E[Q]" its literal meaning in code, a 12,325x GPU
rewrite, three Q-value checkpoints still catalog-live today, and, on the last night of the
window, [[zeb]] founded as a second, parallel bet. It is also the era before the wall existed.
The window opens on a confession about the perfect-information oracle and closes on
unshaded triumph — "we did it buddy." See [[strategy-fusion]] for the diagnosis-through-founding
arc and [[expected-q-value]] for the mechanism itself. Source: [[era3-eq-era|conversation digest]].

## The story

The oracle carried a diagnosed disease in from the prior era: strategy fusion — it played as
a god because it was trained via reverse induction on fully-observed states, and so it "knows
the right answer because the right answer was answered before it was even asked." [[strategy-fusion]]
covers the diagnosis, the abandoned first fix, and the discipline moment ("grok, don't
converge") that preceded the actual founding of E[Q]: the sentence "we train on games where
every move is chosen by averaging oracle values over all hidden worlds consistent with what's
been publicly played so far," asked to be checked against a "karpathy himself" standard
(2026-01-10T23:56:04).

It shipped fast. `ece6dcf` (Jan 10) births `forge/eq/` — 18 files, 2,860 insertions, zero
deletions. The next day, `341dd53` is the fix that defines the mechanism: reconstruct a
hypothetical initial hand per sampled world (`remaining + played_by`) instead of using the true
deal. See [[expected-q-value]] for the full build timeline (12,325x vectorization, the 74%
ceiling, the slot-0 bias hunt).

A five-day gap (Jan 12-16), then an engineering avalanche: posterior-weighted sampling with
ESS mitigation ([[qval-over-policy-models]] lands the same week), schema v2, GPU
saturation, the 12,325x rewrite, CUDA-only enumeration, the full 85-bin E[Q] histogram
(`ef199b0`, Jan 24 — the mechanical ancestor of the later name [[candlewax]], not the same
thing — see Corrections below), the slot-0 positional-bias investigation ([[q0-positional-bias]]),
and a sampling-convergence result Jason almost misread as a failure before flipping it into a
celebrated breakthrough (Jan 27).

The era closes by opening a second front: `62e3b53` (Jan 31) births `forge/zeb/`, an
AlphaZero-style self-play learner reusing the oracle's tables and the Stage-2 tokenization
just built. No trained result lands in-window — [[zeb]] ends the era as scaffolding and a
promise. See [[zeb]]'s Precursor section for the full founding-commit inventory.

**Running underneath it all**, non-load-bearing: the ralph/ralf loop tooling (`.claude/skills/ralph-loop/`,
`scripts/ralph.md`) — a bid to automate the recurring "de-slop" tax so more hours went to the
real problem, instantiated once (Jan 11) and referenced in research through Jan 21, then quiet
for the rest of the window — and a handful of one-off conversational spikes (a live GPU-solver
debugging session nicknamed "42 ai buddy"; a lookup of an external Texas 42 web-game maker;
a memory-store naming brainstorm called "engram," never built, its beads-pairing premise since
moot). None of these produced a lasting artifact beyond what's named above.

## What this era established

1. **E[Q] is cheap at scale and it's real.** 6,493 games/sec after the 12,325x rewrite, 99.3%
   transfer cut, bit-for-bit verified (`0db7750`, `9098e5d`). This never becomes a bottleneck
   again.
2. **There is a hard, provable ceiling on argmax-Q play: ~74%.** Not a training deficiency — a
   fact about the game's tie structure. See [[argmax-q-ceiling]].
3. **Naive uniform-world sampling is measurably wrong ~1/3 of the time** (Jason's own
   measurement, 2026-01-27, asserted not re-verified against a logged artifact). The mechanism
   used to marginalize changes a third of the labels.
4. **Data provenance can silently poison a model.** `deal_from_seed()` sorting hands by domino
   ID injected a 1.74-bit artifact that 20 parallel architectural investigations couldn't
   explain — see [[q0-positional-bias]].
5. **"Grok, don't converge."** See [[grok-not-converge]].
6. **No-legacy is exercised discipline, not a slogan.** SMC deleted same-day it was superseded;
   CPU fallbacks introduced and stripped within minutes; `EQ_MVP.md` retired the day its
   successor shipped.

## What it ruled in / out about the wall

**Ruled in:** the value-computation half of the wall is done. E[Q] is cheap, GPU-scaled,
bit-for-bit correct, validated against a theory-matched ceiling. Nothing in later eras
re-litigates whether E[Q] can be computed.

**Ruled out, by deliberate deferral, not failure:** any weighting of sampled worlds by
likelihood/counterfactual evidence (belief-weighting, deferred Jan 10 and Jan 14 — "no
heuristics... that's just an example of something that should arise naturally if it is correct
to do"), and any consumption-side plan whatsoever — bidding, signaling, CFR, and a symbolic
"threat class" abstraction (Jason's own coinage, 2026-01-14, never implemented in-window) were
all named and cut from scope. "I literally don't care about signaling. yet."

The wall does not exist yet in this era. There is no undefeated champion, no "I dunno what to
do with it" — that framing belongs to a later moment describing this same founding event; see
[[strategy-fusion]] for the dating correction. E[Q] is being born as a validated input while the
question of what a player does with it is left completely untouched, on purpose. Zeb's birth on
the last night is the first parallel bet on that question — not "consume E[Q] better" but
"sidestep labeling entirely via reward." A second bet placed, not yet a data point.

## Corrections to the received story

- **"Candlewax was born here" is wrong.** The string "candlewax" does not appear in the repo or
  corpus until 2026-04, ~3 months after this window closes, naming an unrelated subproject
  ([[candlewax-spike]]). What is real in January is the mechanical ancestor: `ef199b0` (Jan 24)
  storing the full 85-bin `e_q_pdf` histogram — a distribution-over-Q the later name happened to
  echo, not the same artifact. See [[candlewax]]'s concordance section.
- **The in-era euphoria quote is "we did it buddy," not "I dunno what to do with it."** The
  January register is unshaded triumph; the doubt is retrospective.
- **Zeb has no rename ancestry**, and its first evaluated result lands 2026-02-06 — era 4, not
  this window.
- **The `t42-64uj` "all 8 phases in one day" claim is off by a day**: five commits Jan 17,
  four Jan 18.
- **Beads tracking for this window doesn't exist**: 0 of 652 exported issues have
  created/closed timestamps in 2026-01-09..31. All in-era tracking is commit-message
  bead-IDs only.

## Links

[[strategy-fusion]] [[expected-q-value]] [[argmax-q-ceiling]] [[q0-positional-bias]]
[[zeb]] [[candlewax]] [[candlewax-spike]] [[grok-not-converge]]
[[qval-over-policy-models]] [[forge]]
