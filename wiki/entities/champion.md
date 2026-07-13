---
title: Champion — unified belief-state player
kind: entity
first_seen: 2026-06-09
last_updated: 2026-07-13
status: active
phase: 2026-07-13 — the ladder stands past #33 with its measurement baseline closed. The current best player is jud's value-native bidder over oracle play ([[jud]] carries the fact, numbers, and the repaired-sampler reproduction via [[stage-0-closure]]). Per-move play targets at v1 capacity graded marks-null ([[jud-target-granularity]]); the next experiments are selected at [[research-lane-selection]] (auction decoder + target granularity), with promotion gated by [[partnership-research-gates]]. Rung-by-rung history — [[champion-ladder]].
---

## What it is

The champion is the unification target for the project's player work: one
belief-state player that bids and plays full games to 7 marks, built from
pieces that already exist as separate artifacts. The reframe (2026-06-09): the
[[forge]] oracle, [[gus]], the pre-wiki bidding evaluators, the mark-utility
machinery, and the [[w42]] concept vocabulary are stapled together today; the
champion is the architecture that makes them one thing — and the same object,
read from the other side, is the teacher.

The action ladder lives in GitHub issues, milestone **Champion** (beads retired
2026-06; historical beads remain readable in `.beads/issues.jsonl`). The
rung-by-rung record — what each rung asked, measured, and concluded, #20
through #33 and the measurement program after — is the [[champion-ladder]]
trail.

## Decision loop

At every decision, bid or play:

1. Maintain a posterior over the 21 hidden tiles conditioned on **all**
   evidence — auction history, every play, every failure to follow suit.
2. Sample worlds from that posterior, not uniformly over consistent worlds.
3. Evaluate each world with the perfect-info oracle ([[expected-q-value]]) or
   its student ([[gus]]); marginalize.
4. Choose under marks-to-7 win probability conditioned on the score, not raw
   points and not fixed p_make.

This is the belief-conditioned-search architecture behind the strongest
bridge, Skat, and poker programs. Texas 42 is small for the class (7 tricks,
~4×10⁸ worlds at deal collapsing rapidly with voids), so near-equilibrium play
is a realistic target. The residual [[pimc]] flaw ([[strategy-fusion]],
information value) is mitigated by self-play consistency and — optionally, the
summit — depth-limited subgame re-solving on late tricks, where 42's endgames
are small enough to solve exactly at the information-set level.

## Asset map

| Organ | Status | Where |
|---|---|---|
| Exact perfect-info value | **done** | [[forge]] oracle; [[expected-q-value]] |
| Fast student | **done** | [[gus]] v3-10k, 0.551 regret; 0.49 with routing ([[blunder-detector]]) |
| One-step utility ceiling | **done** — EV wins | Lens(ev); [[w42-lens-v1-utility-head-to-head]] |
| Belief posterior | **accuracy won, play-weighting dead** | +2.59pp auction conditioning ([[w42-champion-auction-belief]], #24); belief-weighted *play* sampling a decisive null (#25, [[champion-ladder]]); [[belief-bayes-ceiling]] |
| Mark utility | **v2**; play-risk measured negative (#27) | `champion/utility.py` (`race_wp`, `MarksToSeven`); rung receipts on [[champion-ladder]] |
| Bid pricing | **superseded by value-native** | `net:wp` distillation (#22) → `V_realized`/`margin:wp` ([[jud]], [[w42-jud-v0]]) |
| Auction policy | **ladder of three** | `heuristic` → `GusBidder` (#21) → `net:wp` (#22) → `ValueBidder` ([[jud]]); receipts on [[champion-ladder]] |
| Full-game arena | **done** | [[arena]]; decision provenance via [[partnership-decision-record-v1]] |
| Self-play loop | **converges; value-native form wins** | #26 fixed point ([[w42-champion-selfplay-fixed-point]]) → jud v0 loop ([[w42-jud-v0]]) |
| Unified bid+play organ | **built; play mechanism-limited** | [[w42-jud-v1]]; per-move targets null at v1 capacity ([[jud-target-granularity]]) |

## Why the auction dominates

Once card play is near-double-dummy, remaining edge in trick-taking games
concentrates in auction accuracy and belief quality (the bridge lesson). The
stack's card play is already near-oracle (0.49–0.55 regret). Marginal-value
ranking for the champion:

**auction ≫ belief-weighted worlds > score-conditioned utility ≫ card-play polish.**

Empirically the ranking holds **for the auction**, not for play: every
CI-excludes-zero arena win is an auction lever (#21, #22, and jud's #32), and
every play-side lever measured dead — score-conditioned play risk (#27),
belief-weighted play sampling (#25), greedy value play and per-move targets at
v1 capacity (#33, [[jud-target-granularity]]). So "belief-weighted worlds" and
"score-conditioned utility" earn their rank **as bidding and defense inputs**,
realized through self-play; applied to card play they collapse into the
card-play-polish floor. [[rank-vs-price]] carries the mechanism; the
[[champion-ladder]] trail carries the receipts.

## Self-consistency

A bid is information only if the policy that produces bids is the policy the
belief model is trained on. The champion reaches that fixed point by self-play
iteration: arena games with the current bidder+player → retrain the belief
model on those games → re-derive the policy → repeat, so bidding conventions
emerge as equilibrium artifacts rather than authored rules. The #26 loop
proved convergence but converged to a calibrated over-bidder
([[w42-champion-selfplay-fixed-point]]); jud v0's value-native loop dissolved
the over-bidder the principled way and produced the current best player
([[jud]]). The unified-core conception — one organ, belief conditioning the
search rather than reweighting it after, and the precise
solve / oracle / eq / belief / utility vocabulary — is recorded at [[jud]].

## Teaching half

A maximally strong player is mute; the [[w42]] campaign built the concept
vocabulary that makes it legible. Run the champion through the detector
battery: where the book is right, where it is wrong, and what to do instead —
with receipts. [[burl]] narrates in the Roberson idiom ([[at-risk-points]],
[[post-commit-q-and-a]]). The distillation chain **oracle → champion → gus →
burl → lem** is also the pedagogy chain: each level explains the one above to
the one below. Target artifact: a data-validated strategy guide — *Winning 42,
second edition*.

First receipts landed at rung #28: the champion agrees with the book on 3 of 6
checkable tactical claims, and the two "contradicted" rows are negative
controls confirming it learned Roberson's prohibitions
([[w42-champion-teaching-battery]]).

## Links

- [[champion-ladder]] — the rung-by-rung record, #20–#33 and after
- [[jud]] — the unified belief-conditioned core; home of the
  current-best-player fact and the jud v0/v1 grading
- [[partnership-wall-research]] · [[partnership-value]] ·
  [[partnership-research-gates]] — the post-v1 measurement program and gates
- [[the-wall]] — the central question this player is graded against
- [[champion-design-review]] — Fable's verbatim forward design + the graded
  predictions ledger and two load-bearing caveats
- [[forge]] · [[gus]] · [[burl]] · [[lem]] · [[w42]] — the organs
- [[arena]] — the measuring stick
- [[w42-lens-v1-utility-head-to-head]] — EV ceiling for fixed one-step utilities
- [[belief-bayes-ceiling]] · [[belief-co-train]] — belief evidence base
- [[pimc]] — the flaw the self-play loop and the summit address
- [[book-strategy-player]] — plan algebra; natural fit is contract plans at
  the auction, not play-phase overlay
- [[w42-book-claim-synthesis-and-ai-directions]] — the single-decision blind
  spot that started this thread
