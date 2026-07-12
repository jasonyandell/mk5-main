---
title: Jud — the unified belief-conditioned core
kind: entity
first_seen: local-2026-06-14
last_updated: local-2026-07-12
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
  every consistent world at equal weight. The melted blob IS [[candlewax]] — the
  Burl-era name for the same multi-peaked object, measured at founding (2026-01-06)
  and carrying the project's central consumption question ("distill it — for what
  purpose?"); see that page's concordance and "the wall, stated precisely."

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

### Registered predictions (falsifiable) — GRADED

Built and graded 2026-07-06 ([[w42-jud-v0]], evidence at `4080e07`):

1. The v0 bidder reaches ≥ parity with `net:wp`: the −2.2 residual is optimism,
   and `net:wp` is a frozen, belief-blind, score-blind slice of V_realized — the
   design subsumes it. **MISS at round 0** (−1.44 [−1.88, −0.95], a legible
   over-bidder that wins points and loses marks) → **PASS after the loop**
   (prediction 3): the final head reaches statistical parity (−0.07 [−0.66, +0.49]
   at the Step-3 seed). Parity at v0's scale — later dominance once data scaled
   ([[w42-plateau-probe]]: +0.38/+0.42 at 3× data/round).
2. V_realized's root exceedance curve matches `optimism_gap.json`'s realized
   curve (0.52 @ 30 falling to ~0.19 @ 41), not the oracle's. **PASS** — ECE 0.046,
   max |Δ| ≤ 0.029 over 13 thresholds, 6× closer to realized than to oracle, sits
   0.09–0.13 below the double-dummy oracle.
3. The value-native loop's fixed point is not an over-bidder: auction escalation
   was value–policy inconsistency, which is impossible by construction once the
   only deposits into V are cleared outcomes. **PASS** — 4 rounds carry the margin
   −1.44 → −0.31 → +0.24 → +0.24 → +0.22 (CI includes zero from round 2); made-rate
   49.9% → 60–64%; the notrump artifact dies in one on-policy round. The residual
   channel it exposed: optimism returns through *selection* (winner's curse) as
   well as the evaluator, and the loop is what closes it.

### The ladder past v0

- **v1 — built and graded ([[w42-jud-v1]], `3ac03de`).** One net for bid AND play:
  play-history-conditioned V_realized, greedy 1-ply play, then the loop and then search
  above the leaves. The bidding side validated the unification; the play side hit the
  per-move-discrimination wall (greedy is mechanism-limited; search recovers 2/3 of the
  gap but not parity; more worlds and better calibration add nothing). The stack went
  −4.37 → −1.43 oracle-free and named v2's target.
- **v2** — the play wall's two named cues: **bigger leaf + per-move targets** (distill
  E[Q] as an auxiliary policy/value signal — the solve-as-bootstrap law, oracle as
  bootstrap not copy) so the leaf can *discriminate* moves, not just *calibrate*
  positions; and **opponents inside rollouts** updating belief from actions, so
  signaling gets priced, conventions emerge, and the referee gap tells the story with
  receipts.

### jud v1 — the one organ, bid and play, built + graded ([[w42-jud-v1]], `3ac03de`)

The unified value organ exists and is graded: `champion/jud_net.py` extends
V_realized from bid-time to EVERY decision — info-state (own hand + canonical
auction + play history) → the same 43-bin realized-points categorical, where
bid-time is play-time with an empty history (one featurization, byte-identical
to `margin_net`'s at the root — tested). Two consumers ride one net: the v0
ValueBidder unchanged (`jud` spec) and `judplay` — greedy depth-1 value-native
play (price every legal move's post-move info-state, argmax E[pts], defenders
minimize), oracle-free and world-sample-free at runtime. The head's value
sharpens with depth (MAE 8.6 → 3.5 root → terminal) — the jud signature.

Six registered predictions ([[w42-jud-v1]], GitHub #33) settle the rung:
**the unification holds at the auction and is mechanism-limited at play.**

- **Bidding validates the unification.** The unified encoding beats v0's own
  round-0 bidder at round 0 (JP2), and jud r4's bidding matures to −0.27 vs
  `net:wp` *behind `lens:ev` play*. The value-native move survives being
  folded into the one organ.
- **Greedy play is a bad ranker.** The full stack loses −4.37 at round 0 (JP1,
  a play-channel loss — points negative, unlike v0's over-bidder), and the
  self-play loop moves play quality **zero** (JP3 falsified: judplay r0 −3.35
  vs r4 −3.44 with bidding fixed). All full-stack movement was the bidder
  adapting to its own weak play. 1-ply greedy value play over a hand-level-MC
  head is mechanism-limited: 28 decisions share one label, against E[Q] n=10's
  per-move oracle evaluation.
- **Search recovers most of the gap.** `judsearch` (belief-lift worlds N=10,
  current-trick rollout with the head playing every seat info-honestly,
  V_realized leaves, **no oracle anywhere**) plays −1.16 where greedy judplay
  scored −3.44 on identical deals — **JS1 PASS, +2.28.** The leaf was fine; the
  greedy consumer was the bottleneck. More worlds (JS2, +0.11) and a
  better-calibrated head (JS3, best test CE of the night, zero play gain) added
  nothing: the wall is per-move discrimination, not worlds or data. Full stack
  bottomed at −1.43 — from −4.37 in one night, oracle-free — never at parity.

The v1 stack is not the champion (that stays `margin:wp`(head_8)+`lens:ev`,
[[champion]]); it is the unification proven at the auction and the play wall
named with receipts. The greedy player is the leaf evaluator search calls, and
search is what makes the sharp post-trick leaf into a ranking.

### The policy-conditional pricing law (jud v1, new to the vocabulary)

`V_realized` prices honestly **only for the policy that generated its corpus.**
jud r4's bidding reads −0.27 with `lens:ev` play behind it, but its prices are
calibrated to its own weak play (offense 49.7% vs head_8's 64%); pair the head
with a *better* player and its prices go stale-pessimistic. The one organ
prices honestly *for itself*, not in the abstract — so a stack's consumers
cannot be mixed and matched, and a value head must be retrained whenever the
policy it prices for changes. This is the realized-outcome analogue of the
self-consistency requirement: a price is only as honest as the policy it was
trained against.

## Honest status

A direction, a vocabulary, and **v0 built, graded, and now past parity**
([[w42-jud-v0]], [[w42-plateau-probe]]): the value-native bidder is real,
calibrated (P2), and its self-play loop dissolves the #26 over-bidder the
principled way (P3). The v0 write-up (`4080e07`) reached statistical *parity* with
the hand-tuned `net:wp` baseline and left one question open — whether the plateau
was the [[pimc]] price of hidden information or a data-starved head. The plateau
probe (`68fda7b`/`0bdd4d5`) answered it: **data starvation, not structure.** Scaling
on-policy self-play 3× per round (rounds 5–8, 1000 games/round) carried the bidder
past `net:wp` — head_8 beats it **+0.38 [+0.09, +0.67]** (reserved seed) and **+0.42
[+0.12, +0.72]** (fresh seed), the first learned bidder to beat the hand-tuned
champion on marks. Rounds 9–12 confirmed **saturation at ≈ +0.3–0.4 marks/game** at
this net capacity (a tiny MLP). The round-0 miss (P1) exposed a second optimism
channel — the winner's curse on *selection*, independent of the oracle's
double-dummy assumption — and the fix was simply more of the on-policy data the loop
already used. The [[champion]] #26 loop had reached a fixed point of a *crippled*
version (belief converged while the value stayed perfect-information, a calibrated
over-bidder); jud v0 is the loop in which the value itself is realized-native, and it
now beats the baseline. The next binding constraint is capacity or mechanism, not
rounds — which was **v1's cue: ONE net for bid + play** (play-history-conditioned
`V_realized`, 1-ply argmax-EV play replacing E[Q] n=10 at runtime, then the same
self-play-loop method on the full stack). **v1 is now built AND graded**
([[w42-jud-v1]], `3ac03de`): one net for every decision, the loop, and search
above the leaves, all registered before measurement. The verdict: the
unification holds at the auction (the bidder survives the fold intact — it beats
v0's own round-0 bidder) and is **mechanism-limited at play** — greedy 1-ply
value play is a bad move-ranker (the loop moves it zero), search recovers
two-thirds of the gap (−3.44 → −1.16, JS1 PASS +2.28) but not parity, and
neither more worlds nor a better-calibrated head closes the rest. The wall is
per-move discrimination: a 470k MLP on hand-level Monte-Carlo labels cannot
out-rank E[Q] n=10's per-move oracle. So the E[Q]-beating edge did **not** live
in a searchless value net at this capacity; the stack reached −1.43 from −4.37
oracle-free in one night, and the current best player stays the value-native
bidder over oracle play (`margin:wp`(head_8)+`lens:ev`, [[champion]]). v2's cue
is now concrete: a bigger leaf trained on **per-move** targets (E[Q] distilled
as a bootstrap, the solve-as-bootstrap law) plus opponents-in-rollout.

[[belief-weighted-jud-mcts]] preserves a distinct consumer hypothesis over the
same result: JudSearch proved that search can use the post-trick leaf, while
the failed worlds sweep says flat sample count is not enough. Adaptive MCTS can
test deeper allocation; information-set node sharing and mid-tree belief
updates supply the structural increment that determinized JudSearch lacks.

**Zeb-protocol reconfirmation (2026-07-06, `afd4802`).** A further paired test
(dropped contracts, bid30 both sides, seed 7000000, 256 games) holds the line:
`judsearch` −1.39 [−1.75,−1.00] and `judplay` −2.73 [−3.04,−2.43] both still
lose to `lens:ev` — eq n=10 remains undefeated at pure play against every
learned challenger since Zeb. A bonus pilot priced `margin:wp`(r8) against the
**live** (non-distilled) `gus:10,wp` sim bidder: **+0.59 [-0.19,+1.39]** (64
games) — the bidding crown wasn't hiding behind the distillation.

## Links

- [[partnership-wall-research]] · [[partnership-research-gates]] — the
  measurement and causal gates that now stand between v1's play result and any
  Jud-v2-shaped build
- [[w42-jud-v1]] — v1 built and graded: the one organ for bid + play; unification holds
  at the auction, mechanism-limited at play (JS1 search PASS +2.28, JP3/JS3 falsified);
  the policy-conditional pricing law; the −4.37 → −1.43 oracle-free trajectory
- [[belief-weighted-jud-mcts]] — surviving consumer hypothesis: adaptive Jud
  search with belief particles, information-set node identity, and mid-tree
  inference; J0/J1/J2/J3/J4 separates root belief, depth, information-set
  updates, and convention
- [[w42-jud-v0]] — v0 built and graded (P2 pass / P1 miss→loop-recovered / P3 pass);
  the recipe-fork and A2 denial-bidding findings
- [[w42-plateau-probe]] — the parity plateau broken: 3× data/round carries the bidder
  past `net:wp` (+0.3–0.4), saturating at this capacity; registered prior falsified
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
- [[w42-book-second-pass]] — the book-sourced experiment queue for the auction-first
  frontier: the auction decoder (bid→hand posteriors, the {30,31,35,36} bid lattice,
  match-score-conditioned bidding) and the signaling/concealment material that is
  exactly the v2 opponents-in-rollout target
