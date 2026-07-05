---
title: Jud — the unified belief-conditioned core
kind: entity
first_seen: local-2026-06-14
last_updated: local-2026-07-05
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

## The engineering, first cut (Fable 5, 2026-07-05)

The deferred "how" has a first cut, from a Fable 5 session reasoning over this
record (a new session; no continuity of memory with `0a708a4e` is claimed). The
ruling on the question [[belief-conditioned-self-play]] left open: **value-native
is endorsed** — promoted from the optional summit to the spine *for the pricing
path* — because #26 measured exactly the failure a fixed perfect-information value
forces, and [[rank-vs-price]] shows the promotion is mandatory for bidding while
play can stay PIMC in v0. The one-organ doc's "may be load-bearing for the bidder,
not optional" is confirmed as the design position.

### jud v0 — the value-native bidder (the smallest true slice)

The stack today runs two value backends across three consumers: play *ranks*
with the double-dummy oracle (`arena/lens_play.py`), `net:wp` *prices* with
distilled Gus-sim realized play (`gus/bidding/simulate.py` — Gus in all four
seats, "skips the oracle entirely"), and the belief bidder *prices* with the
oracle again (`champion/belief_bidder.py`). [[rank-vs-price]] says the ranking
consumer is fine and every price consumer must share the realized backend. v0
does exactly that — one head is added; nothing else moves:

- **V_realized** — a distributional belief-state value: info-state (own hand +
  full auction + play history so far) → distribution over the hand's final points
  margin (equivalently, the per-declaration threshold-exceedance curve at the
  root). Trained by categorical cross-entropy on the **realized** outcome of
  arena self-play hands — Monte Carlo targets, no bootstrapping (a 7-trick
  horizon makes TD machinery pointless). The corpus is #26's bridge
  (`arena.cli --emit-snapshots` → stamped `GameRecordGPU`) extended to stamp the
  realized outcome already recorded in `arena/results/per_hand.csv`.
- **The bidder** prices each candidate contract by querying V_realized at the
  hypothetical-completed-auction root (the in-distribution trick
  `champion/belief_bidder.py` built), reads tail mass at the threshold, and
  collapses through the existing `MarksToSeven` — so the untested auction-side
  score-conditioning lever ([[champion-design-review]] caveat 2) rides along for
  free. The oracle E[Q] path leaves the pricing loop entirely.
- **Play unchanged** (`lens:ev`) — per [[rank-vs-price]], play consumes rankings,
  which are near-ceiling; value-native is load-bearing only where prices are
  consumed.
- **The loop** — #26's machinery, now training belief AND V_realized each round.
  Convergence reads: belief-KL (exists), V calibration (predicted vs realized
  exceedance, reliability curve), and the referee gap (below).
- **Coverage** — on-policy corpora starve V of off-policy contracts (a bidder
  that never bids 84 generates no 84 data). Blend ε-exploration bids into arena
  generation and/or forced-bid corpora (`generate_eq_continuous --bid-value seed`
  exists since rung #23).
- **The pass alternative** — arena corpora contain opponent-won contracts, so
  V_realized also trains on defend-side info-states ("suppose they win"),
  pricing the pass counterfactual from data. This closes with data the hole #26
  flagged as genuinely OOD for the oracle path (the suppose-opponent-wins
  hypothetical was never queried) and gives `MarksToSeven`'s pass baseline a
  learned leg to stand on.
- **Belief's role in v0** — implicit: V_realized conditioned on the auction
  learns what the belief would say. The explicit belief head keeps training in
  the loop (the #24/#26 line) for legibility ([[belief-trajectory]]) and for
  v1's search; belief-inside-the-search arrives at v1, not v0. v0's unification
  is bidder+player sharing one bank account.
- **What this is not** — not the rejected realized-make-rate calibration. That
  was a static scalar painted over a double-dummy search. V_realized replaces
  the evaluator with the full outcome distribution — EV and tails both survive,
  utilities collapse only at decision time (the Lens v1 law) — and it trains
  *inside* the loop, so it is self-consistent rather than corrected.

### Factorization law

Learn what is unknown; compute what is exact. The hand-outcome distribution
under the real policy is unknown → learned (V_realized). The marks race is
exact → computed (`race_wp`, Pascal). Do not learn marks-to-7; do not hand-tune
hand outcomes (`pmake_scale` retires).

### The referee instrument

Per position, oracle EV − V_realized EV = **the price of hidden information** —
the one-organ doc's "ruler" made concrete (`champion/optimism_meter.py` is its
static ancestor). When later rungs put the model inside the rollouts, this gap
narrates conventions emerging: signaling is exactly what moves realized value
toward double-dummy.

### Registered predictions (falsifiable)

1. The v0 bidder reaches ≥ parity with `net:wp`: the −2.2 residual is optimism,
   and `net:wp` is a frozen, belief-blind, score-blind slice of V_realized — the
   design subsumes it.
2. V_realized's root exceedance curve matches `optimism_gap.json`'s realized
   curve (0.52 @ 30 falling to ~0.19 @ 41), not the oracle's.
3. The value-native loop's fixed point is not an over-bidder: auction escalation
   was value–policy inconsistency, which is impossible by construction once the
   only deposits into V are cleared outcomes.

### The ladder past v0

- **v1** — V_realized at the leaves of shallow belief-state search in play and
  defense (the summit made spine; Student-of-Games shape). Defense is where
  information-set value concentrates.
- **v2** — opponents inside rollouts update belief from actions: signaling gets
  priced, conventions emerge, and the referee gap tells the story with receipts.

## Honest status

A direction, a vocabulary, and now a first-cut engineering spec (v0, 2026-07-05,
GitHub issue filed). Not built. The [[champion]] #26 loop reached a fixed point
of a *crippled* version — belief converged while the value stayed
perfect-information, yielding a calibrated over-bidder. Jud names the loop in
which the value itself is belief-native.

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
