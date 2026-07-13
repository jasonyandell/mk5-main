---
title: Suit algebra — absorption vs. power, S₇ symmetry, τ-encoding
kind: topic
first_seen: 2026-07-06
last_updated: 2026-07-11
status: active
---

## What it is

The factored algebraic model that replaced Texas 42's ad-hoc, per-mode rules logic (trump,
nello, notrump each had their own special-cased ranking code) with a single encoding split
into two independent operations:

- **Absorption** — which dominoes belong to which suit.
- **Power** — which suit beats which, and by how much.

Nello (doubles-suit) is the proof that these are genuinely orthogonal, not merely
usually-aligned: doubles separate into their own suit (absorption) with no power suit at all
(nothing beats anything). Jason diagnosed the prior abstraction as rotten before building
anything new: "I think my fundamental abstraction has been misguided... I want to actually
represent these actual rules actually correctly in code. and the naive approach has not led
us to elegance." (conv b132ad92, 2025-12-24).

## Built as code

Shipped `5dbd59e` (2025-12-25, "Factored algebraic model for dominoes"), with the load-bearing
claim in `c1d0ac6`'s commit message: **"The S₇ symmetry insight: all pip absorptions are
isomorphic."** `src/game/core/domino-tables.ts` (DOMINO_PIPS, getAbsorptionId, getPowerId,
EFFECTIVE_SUIT, SUIT_MASK, RANK, HAS_POWER) plus 11 tests in
`src/tests/unit/domino-tables.test.ts` (the commit message's "28 tests" refers to the 28
dominoes tested, not test blocks — corrected here after direct recount). `rules-base.ts` was
changed so absorbed dominoes lead suit 7, a real semantic change, not just a table.

A single rank encoding, **τ(d,ℓ,δ) = (tier<<4)+rank**, unified move-ranking across pip-trump,
doubles-trump, and no-trump modes (`366a005`, 2025-12-26), deleting a pile of per-mode special
cases across ~9 files. τ ranks a domino relative to trump ("3rd-highest trump") rather than by
raw identity ("domino 14") — the seed-invariant basis that survived every model built on top
of it. When a cross-seed-generalization diagnostic later showed a raw-domino-ID MLP had
memorized seeds instead of learning structure (test loss 0.040 vs. target <0.02, bead
t42-wzsq, 2025-12-29), the τ idea was folded directly into the transformer's tokenization
rather than shipped as a standalone re-encoded MLP — it is the direct ancestor of the
production tokenizer's `trump_rank` feature and `TRICK_RANK_TABLE`
(`forge/eq/game_tensor.py`), still live and consumed by `forge/analysis/bias/`'s
interpretability scripts today.

**Spec:** the full formal treatment now lives in the wiki as [[suit-algebra-spec]] (suits,
trick order, unique-winner theorem, machine encoding — merging `docs/theory/SUIT_ALGEBRA.md`
and `SUIT_ALGEBRA_PURE.md`, which were ~90% identical) and [[play-phase-algebra]] (state
model, signed rewards, backward induction — from `PLAY_PHASE_ALGEBRA.md` + `PLAY_PHASE_SPEC.md`).
The docs/theory/ originals were retired at this ingest. The algebra is still cited by the
Jan-8 epistemic-audit closeout and by later training-data-generation decisions
(S₇-isomorphism-driven declaration sampling means only one pip-trump representative needs
training data, not all seven).

## Renamed, on Jason's own coinage

- "absorbed suit" → **"called suit"**, Jason's own term, retiring jargon no real player would
  recognize: "there are natural suits and then there's what we've called absorbed suit, but
  that turns out to be quite an unfamiliar and strange word... I've been playing 42, very
  traditional game, for a long time. I've never heard of it. nobody has." (conv a42a9394,
  2025-12-26). "after working with you (opus) in claude code we called the concept 'called'
  suit."
- "nil" → **"no-trump"**; "nello" → **"doubles-suit" / "doubles-trump"** (bfbffcec, a42a9394,
  2025-12-26/27) — landed in code the same window (`d4d62fb`, `d3dd1ed`).

## A real bug, caught honestly

Gemini's cross-check flagged that the formalization as first written implied 5-5 loses to 5-6
in a fives-trump lead (sum 10 < sum 11), which is wrong per the actual game. Jason endorsed
the catch and named the mechanism honestly — the algebra didn't produce it on its own, a
manual rules-recall did: "hmm. that's a big miss we made there buddy i read this and missed it
too." (conv a42a9394, 2025-12-26).

## What did not survive, or never shipped

- **Sevens is explicitly out of scope**, not a later extension target: `docs/theory/SUIT_ALGEBRA.md`
  states "The sevens contract is deliberately absent from Δ... Nothing in this algebra
  applies." A Sevens-extension bead (t42-d2ia) sat `pending` and was closed only by an
  automated stale-sweep four months later — never implemented.
- **Three-axis feature decomposition** (a downstream feature-engineering effort riding on the
  algebra) failed: R²=22.8% at best (MLP-combined), both its Void Sufficiency and Linear
  Decomposition conjectures marked REFUTED (bead t42-bp9q, 2025-12-28). `docs/theory/SUIT_STRENGTH.md`
  was later deleted as obsolete (`0113be4`, 2026-01-08).
- **A "3blue1brown-style visualization"** produced a real, working static LaTeX PDF of the
  algebra's equations ("look at it it's beautiful. worked great. holy cow," conv fdac8110,
  2026-01-01) — not an animated/Manim video, despite the aspirational label.
- **No CFR work and no Lean/formal-proof-assistant exploration** exist anywhere in this
  window. The DP/backward-induction solver ([[the-oracle]]) that does belong here is a
  distinct technique from CFR, which belongs to an earlier, separately-abandoned
  count-abstraction line.

## Where it sits in the era

Suit algebra is upstream of [[the-oracle]] and everything built on it — it fixed the game's
*representation* so the GPU solver, [[forge]], and the Q-value models had a trustworthy
substrate. It does not touch how a computed value gets consumed into a plan; that question
([[strategy-fusion]]) surfaces later in the same era, once bidding-by-simulation started
using the oracle the algebra had made trustworthy. See [[breakthrough-and-oracle]] for the
full era narrative.

## Links

[[suit-algebra-spec]] · [[play-phase-algebra]] · [[the-oracle]] ·
[[breakthrough-and-oracle]] · [[forge]] ·
[[sources/claude/era2-breakthrough-oracle|conversation digest]]
