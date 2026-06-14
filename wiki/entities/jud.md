---
title: Jud — the unified belief-conditioned core
kind: entity
first_seen: local-2026-06-14
last_updated: local-2026-06-14
status: active
---

## What it is

Jud is the name for the unified belief-conditioned core the [[champion]] work points
at: one organ that bids and plays as the *same act*, by conditioning search on a
learned belief and training that belief by self-play over whole games. At this point
it is a **direction and a vocabulary**, not a build. The engineering — how the
foundation is unified, how the value is trained, the shape of the loop — is
deliberately deferred.

The [[champion]] ladder (#20–#28) produced the pieces as separable artifacts: the
[[forge]] solve, its distilled oracle, [[expected-q-value]] search, the [[gus]]
belief head, the EV utility ([[w42-lens-v1-utility-head-to-head]]). Those pieces are
stapled together today, and rung #26 reached a fixed point of a *partial* loop. Jud
is the conception in which they become one thing, and a commitment to precise words
so they stop being conflated.

## Vocabulary (use these words precisely)

"Oracle" is colloquially overloaded; this project's pieces are specific.

- **solve** — perfect-information Texas 42, solved exactly: per deal, the true value
  of every move by exhaustive backward induction over the enumerated game
  (`forge/oracle/solve.py`, [[forge]]). Ground truth. *Not* "the oracle."
- **oracle** — the fast distillation of the solve into a value network
  (`domino-qval-large-3.3M`, qmae 0.94, ≈97% accurate). The runtime stand-in for the
  solve. solve and oracle share one **type** — the value of a *complete* world — and
  therefore one wall.
- **the brick wall** — because solve and oracle require a complete world, neither can
  be applied to a hidden-information decision directly. This is not a defect to
  repair; it is what a perfect-information value *is*: a function of perfect
  information.
- **eq** — the lift past the wall ([[expected-q-value]]): sample worlds consistent
  with what is known, evaluate each with the oracle, keep the spread. Its output per
  action is a **distribution** over outcomes — the honest object, because under hidden
  information the value of an action genuinely *is* a distribution, not a number. eq
  is the correct bridge. Its limit: it draws worlds by *consistency only*, conditions
  on no belief, and does not learn or grow.
- **the blob** — eq's per-action distribution. Real, but *melted*: smeared across
  every consistent world at equal weight.
- **belief** — a learned, conditioned weighting over worlds, P(world | all evidence).
  It exists today (the [[gus]] belief head; auction-conditioned at
  [[w42-champion-auction-belief]], #24) but only as a *post-hoc reweight*, which
  measured marks-neutral in play (#25) — a result [[champion-design-review]] reframes
  as the [[arena]] being information-blind, not the belief inert ([[pimc]],
  [[belief-bayes-ceiling]]).
- **utility** — the collapse of a blob to a scalar to choose by (EV, p_make,
  marks-to-7). EV is the established winner for fixed one-step utilities
  ([[w42-lens-v1-utility-head-to-head]], +5.42 pts/hand). This half is settled.

## The idea

Belief belongs **inside the search**, not as a reweight applied after it.
Conditioning eq's worlds on belief sharpens the blob onto the worlds actually in play;
a utility then reads a shape instead of a smear. The same operation at every depth
makes **bidding and play one act** — a decision over a belief at the root is a bid;
the same decision deeper is a play; the same on the other side is defense. Belief
carries information when the opponents *inside a rollout* also update belief from
actions — the only place signaling can exist. And the belief that conditions the
search is trained on the whole games the search produces, so it learns toward a
self-consistent fixed point ([[champion]] self-consistency; [[champion-design-review]]
forward design). This is the design's "one thing" — laid over Fable's verbatim
forward design ([[champion-design-review]], confirmed byte-exact against the transcript)
and the participant's memory ([[belief-conditioned-self-play]]).

## What the solve is, and is not

Solving perfect-information 42 is real, rare, and load-bearing: ground truth, a
bootstrap, and an exact referee. It is **not** the player. It lives on the
perfect-information axis; the player's problem is hidden information, an orthogonal
axis. The over-bidding measured at rung #26 ([[w42-champion-selfplay-fixed-point]]) is
not the oracle's ≈3% distillation error and would survive a 100% solve: it is the gap
between perfect-information value and achievable hidden-information play (strategy
fusion, [[pimc]]). Consequently jud's value is *not* the oracle distilled harder — its
target is realized whole-game outcomes (belief-state value), with the solve and oracle
serving as bootstrap and referee rather than as the thing being copied.

## First picture

A session sketch (`scratch/jud_demo/`, uncommitted, 2026-06-14) rendered one mid-game
position's eq blob two ways: melted (128 worlds, uniform) and belief-weighted. Even
the weak post-hoc reweight visibly un-melts the comb — on the headline action, world
ESS 128 → 10.5 and the contract's p_make 0.20 → 0.61 — and it sharpens toward *truth*,
not optimism (other lines resolve to losing). It is the #25 belief value seen directly
in the distribution rather than through the information-blind [[arena]] that could not
score it. A melted blob is illegible; the same blob, belief-sharpened, is both the
stronger move and the teachable lesson — the legibility the [[champion]] teaching half
wants.

## Honest status

A direction and a vocabulary, captured 2026-06-14. Not built. The [[champion]] #26
loop reached a fixed point of a *crippled* version — belief converged while the value
stayed perfect-information, yielding a calibrated over-bidder. Jud names the loop in
which the value itself is belief-native. The how is open and deferred until the
conception settles.

## Links

- [[champion]] — the player-and-teacher this is the core for; the ladder that built
  the pieces
- [[champion-design-review]] — Fable's verbatim forward design (confirmed byte-exact vs the transcript) + a graded predictions ledger; the two load-bearing caveats
- [[belief-conditioned-self-play]] — what jud trains and how (the loop shape, sourced to Fable; the training mechanics, flagged as open)
- [[expected-q-value]] — eq, the lift; [[pimc]] — the flaw belief-in-the-search
  addresses
- [[forge]] — the solve and the distilled oracle; [[gus]] — the belief head;
  [[zeb]] — the parked learned model (a different thing, not this)
- [[w42-lens-v1-utility-head-to-head]] — EV as the utility;
  [[w42-champion-auction-belief]] — the auction belief;
  [[w42-champion-selfplay-fixed-point]] — the #26 fixed point
- [[arena]] — the information-blind measuring stick belief value cannot be scored
  through
