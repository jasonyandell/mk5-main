---
title: Belief-conditioned self-play — what jud trains and how (Fable's approach)
kind: topic
first_seen: 2026-06-14
last_updated: 2026-07-11
status: active
---

This page records the training approach behind [[jud]] — what is trained, how, the
arena, the self-play loop — separating **what Fable specified** (sourced and clear)
from **what this session extended** and **what is not yet clear**. The last is marked
explicitly rather than guessed.

## The single objective

The approach has one objective: the self-consistent fixed point of belief-conditioned
self-play. It is an attractor in a precise sense — self-consistency is a fixed point
of the training loop (the policy that bids is the policy the belief is trained on is
the policy that plays), so the destination does not move. Work either advances the
organ toward it, or builds the instruments that gauge the descent (the **solve** as
exact referee, the [[arena]] as measuring stick), or does not land. One caution
already on the record: the loop is proven to *converge*, but #26
([[w42-champion-selfplay-fixed-point]]) converged to a *shallow* fixed point — a
calibrated over-bidder — because it trained only the belief and left the value
perfect-information. The deep, tournament-strong fixed point is hypothesis, not result.

## What the design review records

[[champion-design-review]] reproduces Fable's **verbatim forward-design turn** — confirmed
byte-exact against the original transcript (session `0a708a4e`; difflib similarity 1.000
over 5590 normalized chars, checked 2026-06-14), with a later-pass graded predictions
ledger and asset map synthesized *on top*. (An earlier provenance pass wrongly demoted the
page to "a recovered summary, not Fable's words"; that demotion was itself an
over-correction and is reversed — see [[log]].) So the spine below is sourced to Fable's
actual words, not a paraphrase:

- **The spine** — at every decision, bid or play: maintain a posterior over the
  hidden tiles from all evidence; sample worlds from that posterior (not uniform);
  evaluate each world with the exact solver / its [[forge]] oracle and marginalize;
  choose under marks-to-7 win probability given the score. Bidding and play are the
  same operation at different depths.
- **The arena** — full games, real auctions, marks to 7. "Until this exists, 'best
  player' isn't a measurable sentence." Every player is judged by game win-rate here.
  ([[arena]], champion rung #20.)
- **The self-play loop, as written** — "play full games with the current
  bidder+player → **retrain the belief model** on those games → re-derive the policy
  with belief-weighted oracle search → repeat. Two or three rounds and your
  partnership has an actual bidding system — conventions that emerged because they're
  optimal." So in Fable's written loop the thing **trained** each round is the
  **belief**; the **policy** is belief-weighted [[expected-q-value|eq]] search over the
  retrained belief and the **fixed** oracle value.
- **Build order** — arena → auction v0 → beliefs everywhere (condition the belief on
  auction + play, then swap uniform world sampling for belief-weighted sampling
  *inside* the search) → the self-play fixed point → marks-to-7 utility over
  everything → (optional summit) depth-limited subgame re-solving on late tricks, with
  *Gus V as leaf values*.
- **Marginal-value ranking** — auction ≫ belief-weighted worlds > score-conditioned
  utility ≫ card-play polish.
- **The teaching half** — the distillation chain (oracle → champion → [[gus]] → burl →
  lem) is also a pedagogy chain; running the champion through the W42 detector battery
  yields a data-validated strategy guide.
- **The two caveats** — the arena is information-blind (both sides PIMC, so belief and
  concealment value cannot be scored via play-marks); score-conditioning belongs at
  the auction, not in play ([[champion-design-review]]).

## What is remembered of the coherent vision (memory, not transcript)

Independent of the design review, the participant who worked with Fable recalls the design as
**one coherent vision** — non-obvious, and elegant. The fragments retained: a trained
model that **changes how the game is played even during search** (the model is inside
its own rollouts, so improving it changes the game the search explores); the **search
must be belief-conditioned**; and **somehow it was all about bidding** — fitting together
as one thing, not parts.

Reconstructed (interpretation, *not* recovered Fable): one model plays the game; search
plays the hand out using that same model, so the beliefs and the play are the model's,
and as it learns the searched game changes; the game-tree is rooted at the bid, so a bid
is valued by playing the hand out under the beliefs the bid creates — bidding, signaling,
play, and defense are one belief-conditioned game the model both searches and is trained
on; self-play converges to a fixed point where bid, belief, and play are mutually
consistent, and a *bidding system* (conventions, signals) emerges as the legible product.
**Open:** "somehow it's all about bidding" is the least-recovered piece — whether bidding
dominates because it is the game-tree root and the marginal-value peak, because it is the
public channel where the fixed point becomes a shared convention, or for some further
reason, is not established.

What the [[champion]] ladder (#20–#28) built was *not* this. It used a fixed oracle (not
a model that reshapes the game during search), searched fixed perfect-information worlds
(not belief-conditioned model rollouts), and added a separate bidder over the oracle. The
over-bidder and the play-side nulls measured those **staples** — not the coherent vision,
which has not been built.

## What this session extended (beyond Fable's forward design)

- **Belief inside the search** — conditioning the *worlds the search visits* on
  belief (un-melting the eq blob), at every ply including inside the rollout, is the
  unifying move recorded at [[jud]].
- **The value itself trained belief-native** — Fable's written loop trains the
  *belief* and keeps the *oracle* (perfect-information value) **fixed**; the policy is
  search over it. This session argues the **value** must also become belief-native —
  trained on realized whole-game outcomes — because a fixed perfect-information value
  is exactly why #26 converged to an over-bidder (the belief converged while the value
  stayed double-dummy; [[pimc]] strategy fusion). In the source, learned values appear
  **only at the optional summit** ("Gus V as leaf values" for subgame re-solving), not
  in the main loop.

## What is not yet clear (flagged, not guessed)

- **Whether the value-native step was the 0a708a4e session's intent.** The source
  establishes a belief-only training loop over a fixed oracle value; promoting
  belief-native value from the optional summit to the spine was the 2026-06-14
  session's extension. The *historical* question stays unestablished (and likely
  permanently so). The *design* question is now closed: a Fable 5 session
  (2026-07-05, a new session claiming no memory of `0a708a4e`) reviewed this
  record and **endorsed value-native for the pricing path** — with a mechanism
  for why it is load-bearing for bidding specifically ([[rank-vs-price]]: play
  consumes rankings, bids consume prices; PIMC optimism cancels in the former
  and lands whole in the latter). The provenance line does not blur: the
  extension was the 2026-06-14 session's, and it was right.
- **The training mechanics** — first cut now specified at [[jud]] (v0): the
  value target is the realized hand-outcome distribution (categorical CE on
  realized margin, Monte Carlo credit assignment over the hand — no
  bootstrapping at a 7-trick horizon); the marks race stays analytic
  (`race_wp`); "the policy" in v0 is the existing `lens:ev` play plus
  V_realized-priced bids; rollouts with belief-updating opponents are deferred
  to v2. What remains genuinely open: the v1 search shape (depth, belief-update
  approximation at internal nodes) and the v2 opponent-model mechanics.
- **Whether the fixed point is reachable and good.** #26 proved a *shallow*
  fixed point (an over-bidder) is reachable. Whether the value-native loop's
  fixed point exists, is reachable, and is tournament-strong is hypothesis —
  now with three registered falsifiable predictions at [[jud]].

## Honest status

The architecture and the loop *shape* are clear and sourced to Fable. The training
*mechanics*, and whether the value-native extension is Fable's own intent, are **not
established**; the engineering is deferred ([[jud]]). This page records what is
established and fences off what is not.

## Origin of the "reasoning is the product" reframe

The reframe predates [[jud]]: it was named at the end of the [[gus]] Morning-5 session
(2026-04-22), resolving a planning loop in which PPO self-play, LAMIR-done-right,
meta-strategy heads, and [[detect-and-route]] each deflated under repeated stress-testing.
Jason's words: *"It's a lot to take in. Who has what, how did that change, what does that
mean, updated domino upon domino until the play. Ultimately it's the reasoning I care
about."* (gus/MORNING5_PLAN.md @ 233b7dc5). The reframe — the reasoning is the product,
not the play — made [[belief-trajectory]] extraction the first move and demoted
action-selection improvements to downstream depth.

## Links

- [[jud]] — the unified core this trains · [[champion]] — the player-and-teacher · the
  ladder
- [[champion-design-review]] — Fable's verbatim forward design + the two caveats
- [[w42-champion-selfplay-fixed-point]] — the #26 belief-only loop (the shallow fixed
  point) · [[arena]] — the measuring stick
- [[expected-q-value]] — eq, the search · [[pimc]] — the flaw belief-native value
  addresses · [[forge]] — solve + oracle · [[gus]] — the belief head
