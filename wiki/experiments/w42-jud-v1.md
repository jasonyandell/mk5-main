---
title: Jud v1 — the one organ, bid and play
kind: experiment
status: complete
task_id: gh-33
first_seen: f550205
last_updated: afd4802
---

# w42-jud-v1

## Summary

**Question:** [[jud]]'s v1 rung — ONE net for bid AND play. Extend `V_realized` from
bid-time to every decision (info-state → 43-bin realized-margin distribution, bid-time =
empty-history play-time), let the [[w42-jud-v0|v0]] `ValueBidder` consume it unchanged,
replace `lens:ev`'s oracle-per-move play with the same head consumed greedily, and iterate
the whole stack by the self-play loop that made the v0 bidder champion-beating
([[w42-plateau-probe]]). Does the unification hold, and does the loop that dissolved the
bidder's over-bidding also close the play gap against E[Q] n=10?

**Answer, in one line:** the unification holds at the auction and is mechanism-limited at
play. One night of registered rungs (organ → loop → search → search-in-loop) carried the
full stack from **−4.37 to −1.43** marks/game vs `net:wp+lens:ev`, oracle-free at runtime —
but never to parity. The bidding side validates the one-organ conception; the play side
names the next wall, and a searchless value net is not it.

Six registered predictions, each stated before its measurement (GitHub #33). **Two
falsified, one passed, three missed the band — all first-class results.** A falsified
registered prediction, run to its pre-committed falsifier, is the system working: it makes
the reversal unarguable instead of a matter of interpretation.

## Addendum — Zeb-protocol verdict (2026-07-06, `afd4802`)

The very next commit ran a further Zeb-protocol paired test (dropped contracts,
bid30 both sides, seed 7000000, 256 games), confirming the night's verdict
rather than overturning it: `judsearch` **−1.39 [−1.75,−1.00]** (30.9%) and
`judplay` **−2.73 [−3.04,−2.43]** (15.2%) both lose to `lens:ev` — "the noisy
mean over ten exact worlds remains the play champion, as it has against every
learned challenger since Zeb." A bonus pilot also ran: `margin:wp`(r8) vs the
**live** (non-distilled) `gus:10,wp` sim bidder, **+0.59 [-0.19,+1.39]** (64
games) — the bidding crown wasn't hiding behind the distillation. Neither
result changes the Night verdict below; both reinforce it.

## The organ

`champion/jud_net.py` — one net, two consumers, no special cases
(`champion/evidence/jud_v1/jud_build_report.md`):

- **Featurization** — `x = [hand 63 | canonical auction 28 | play 259]`, 350 dims. The
  auction block reuses `margin_net.canonical_auction` verbatim (the [[w42-jud-v0|v0]]
  level-blind, later-seat-masked selection-leakage defense); the play block is a POV-relative
  per-domino map + running-score tail, from which who-played-what, trick winners, and the
  score are all recoverable. **Bid-time is play-time with an empty play history** —
  byte-identical to `margin_net.featurize` at the root (tested).
- **Target** — the same 43-bin categorical over the declaring team's realized points, one
  Monte Carlo label per hand shared across all ~28 per-decision info-states (offense AND
  defense rows). No bootstrapping (7-trick horizon). MLP 350 → 512 → 512 → 43, ~470K params.
- **Bidding consumer** — `JudNet.pmake_table` has `MarginNet.pmake_table`'s exact signature,
  so `ValueBidder` consumes it with **zero adapter** (`jud` spec).
- **Play consumer** — `arena/jud_play.py::JudPlay` — greedy depth-1: price every legal move's
  post-move info-state, argmax E[pts], defenders flip the sign. One forward pass per tick.
  No world sampling, no oracle at runtime (`judplay` spec).

The head's value **sharpens with depth** — per-trick MAE 8.6 → 3.5 root → terminal — the jud
signature: the value gets sharper as evidence accrues.

## The registered ledger

### JP1 — round-0 full stack loses, legibly — MISS by a hair, shape CONFIRMED

*Registered:* judbid+judplay (trained on `lens:ev`-played corpus, so judplay's own play is
off-policy at round 0) vs `net:wp+lens:ev`, 512 games — loses, margin in [−4, −1]; mechanism
= play errors not bidding; diagnostic = point margin ALSO negative (unlike v0's over-bidder,
which won points while losing marks).

*Graded:* **−4.37 [−4.55, −4.19]** — a hair past the [−4, −1] band. Point margin −6.10/hand,
negative as predicted. The shape landed exactly: this is a play-channel loss, not a bidding
loss, and the searchless value player bleeds points rather than trading them.

### JP2 — the bid side survives the unification — MISS, wrong bar

*Registered:* judbid alone (judbid+`lens:ev` vs `net:wp+lens:ev`, 256+ games) within ±0.4 of
the [[w42-plateau-probe|margin-head]] level, i.e. margin in [−0.2, +0.8]; a big regression
would mean the unified encoding broke something.

*Graded:* **−1.09 [−1.50, −0.69]** — outside the band, but the band was wrong. It compared a
**round-0** head to the loop-matured head_8. The honest apples-to-apples is v0's own round 0
(−1.44), which the unified head **beats** at its own round 0. The encoding is not broken;
loop-maturity was the missing ingredient the prediction failed to name.

### JP3 — the loop closes most of the play gap — FALSIFIER FIRED

*Registered:* after ≥4 self-play loop rounds, the full-stack margin improves by ≥ half the
round-0 deficit (≥ +2.2). Falsifier for the hopeful half: no improvement round-over-round ⇒
1-ply value play is mechanism-limited, not data-limited, and search is the cue.

*Graded:* four loop rounds carried the full stack **−4.37 → −4.38 → −4.16 → −3.72 → −4.04** —
nowhere near half the deficit. The decisive isolation experiment, bidding held fixed so only
play varies: **judplay(r0) −3.35 vs judplay(r4) −3.44** — the loop moves play quality
**zero**. Every bit of full-stack movement was the bidder adapting to its own play (offense
share fell 64% → 49% as the bidder learned to stop over-bidding behind a weak player;
made-rate rose 32.5% → ~40%). Bidding-side learning works exactly as in v0; the play-side
signal is too weak to move. 1-ply greedy value play over a hand-level-Monte-Carlo head is
mechanism-limited: 28 decisions share one label, against an opponent that evaluates 10
sampled perfect-information worlds with a 97% oracle per move.

### The policy-conditional pricing finding (not registered — a bonus)

jud r4's **bidding** matured to **−0.27 [−0.57, +0.01]** — but only *with `lens:ev` play
behind it*. Its prices are calibrated to its own weak play (offense 49.7% vs head_8's 64%):
pair the head with a **better** player and its prices go stale-pessimistic. **`V_realized`
prices are calibrated to the policy that generated them** — the one organ prices honestly
*for itself*, not in the abstract. This is the deep reason the bidder and player cannot be
tuned independently, and a caution for any future stack that swaps one consumer's backend.

### JS1 — search above the leaves — PASS, +2.28

*Registered (amended pre-build):* `judsearch` — the v1 spec's literal shape. For each legal
move: sample N=10 consistent worlds (eq's belief lift, the exact `lens` sampler), roll the
CURRENT TRICK to resolution inside each world with the jud head playing every seat
info-honestly (offense argmax, defense argmin), evaluate the post-trick info-state with the
same head from the searcher's POV, average across worlds under common random numbers, argmax
EV. **No oracle anywhere — the leaf is `V_realized`.** (The first draft wrongly claimed no
world sampling; opponents' in-trick replies come from hidden hands, so honest lookahead needs
the belief lift — caught before building.) Prediction: search improves the play channel by
≥ +1.0 marks/game over greedy judplay (same head, play-only A/B, bid fixed), because greedy
1-ply cannot see who wins the count while the post-trick leaf is exactly where the head is
sharpest. Honest prior on reaching `lens:ev` parity: ~15%.

*Graded:* **PASS, +2.28.** Play-only, same r4 head, same deals (seed 7200000): greedy
judplay **−3.44 [−3.75, −3.14]** → judsearch:n10 **−1.16 [−1.55, −0.77]** (made-rate 46.1% →
62.7%). Registered bar ≥ +1.0, cleared at 2.3×. **The leaf was fine; the greedy consumer was
the bottleneck.** Still short of `lens:ev` (the CI excludes 0) — the ~15% prior on full
parity was right to be low. No oracle anywhere on the jud side.

### JS2 — worlds sweep — BELOW BAND

*Registered:* judsearch:n20, same seed/head — mild gain +0.2 to +0.6 (world-average noise
shrinks but the leaf's bias is shared); falsifier ≥ +1.0 (worlds are the binding constraint).

*Graded:* **−1.05 [−1.44, −0.64], gain +0.11** — below the band. Doubling worlds barely
moved it. **Worlds are not the constraint; the shared leaf bias is.** Don't push n.

### JS3 — search-in-the-loop — FALSIFIER FIRED

*Registered:* one loop round where self-play uses judsearch:n10 both sides (~600 games),
cumulative retrain, re-measure. Prediction: search-quality games improve BOTH the head's
prices and its leaf, worth ≥ +0.4 on the play channel beyond JS2, and the full jud stack
lands within [−1.0, +0.2] of `net:wp+lens:ev`. Honest prior on full parity tonight: ~25%.

*Graded:* one round of judsearch self-play (600 games, 6612 hands) + cumulative retrain
produced **the best paper head yet** (r5, test CE 2.09) — and bought **zero** play
improvement: play-only **−1.41 [−1.79, −1.02]** vs r4's −1.16 (registered ≥ +0.4). Full
stack jud:wp(r5)+judsearch(r5) at reserved seed 7000000: **−1.43 [−1.68, −1.16]** (registered
band [−1.0, +0.2]). **Better paper calibration did not buy better play.** The wall is
per-move discrimination: a 470k MLP trained on hand-level Monte-Carlo labels cannot match a
97%-accurate 3.3M perfect-information oracle evaluated per move over sampled worlds.
Capacity and per-move signal, not data or loop rounds.

## The two trajectories

The play channel across the JS ladder (play-only, same deals, bid fixed):

| play consumer | margin (A−B) | 95% CI | made-rate |
|---|---|---:|---:|
| greedy judplay (r4) | −3.44 | [−3.75, −3.14] | 46.1% |
| judsearch:n10 (r4) | **−1.16** | [−1.55, −0.77] | 62.7% |
| judsearch:n20 (r4) | −1.05 | [−1.44, −0.64] | — |
| judsearch:n10 (r5, retrained) | −1.41 | [−1.79, −1.02] | — |

Search recovered two-thirds of the greedy→`lens:ev` gap in one step; every lever after it
(more worlds, a better-calibrated head) added nothing. The full-stack trajectory over the
night's rungs, vs `net:wp+lens:ev`:

| rung | full stack | 95% CI |
|---|---|---:|
| JP1 round-0 (judbid+judplay) | −4.37 | [−4.55, −4.19] |
| JP3 loop round 4 | −4.04 | — |
| JS3 jud:wp(r5)+judsearch(r5) @ reserved seed | **−1.43** | [−1.68, −1.16] |

## Night verdict

The best player the project knows how to make at this time is **not** a jud stack. It is the
[[w42-plateau-probe|value-native bidder]] over oracle play:

**`margin:wp(head_8) + lens:ev`** — beats the previous champion `net:wp+lens:ev` by **+0.38
[+0.09, +0.67]** and **+0.42 [+0.12, +0.72]** (512 games at each of two reserved seeds). The
first learned bidder to beat the hand-tuned one. jud v1's one-organ stack reached −1.43 from
−4.37 in one night, oracle-free — its bidding validates the unification, its play names the
wall.

## Interpretation

1. **The unification is real at the auction.** One featurization, one net, one bank account:
   the `ValueBidder` consumes the play-conditioned head with zero adapter, and bid-time is
   byte-identically play-time with an empty history. The v0 bidder result transfers into the
   unified organ intact (JP2's honest bar) — the encoding did not break anything.
2. **Play consumes rankings, and the ranking gap survives.** [[rank-vs-price]] predicted the
   auction was where the value-native move pays; v1 confirms the other half. Greedy 1-ply
   value play (−3.44) is a bad *ranker* of moves even though the head is a fine *evaluator*
   of positions — search closes most of that gap (−1.16) precisely by turning the sharp
   post-trick leaf into a ranking, without ever touching the head. But the oracle's rankings
   are still unbeaten: `lens:ev` remains ahead, and no amount of worlds or retraining moved
   it.
3. **Prices are policy-conditional.** `V_realized` is honest only for the policy that
   generated its corpus (the r4 bidder's −0.27 evaporates when a stronger player sits behind
   it). Consumers cannot be mixed and matched.
4. **Better calibration ≠ better discrimination.** JS3's r5 head had the best test CE of the
   night and the worst outcome — proof that per-move move-ranking, not distributional
   calibration, is the play-side objective, and that a 470k MLP on hand-level labels does not
   have the per-move signal a per-move oracle does.

## v2's named target

Bigger leaf + **per-move targets** — distill E[Q] as an auxiliary policy/value signal (the
[[jud]] solve-as-bootstrap law: the oracle as bootstrap and referee, not the thing copied) —
plus **opponents-in-rollout** so belief updates from actions and signaling gets priced. The
wall v1 measured is capacity and per-move signal; v2 is the rung that supplies both.

## Provenance

Built and graded 2026-07-06 across four commits: `f550205` (the organ + play-history
snapshots + `judplay`), `9d30b25` (JP1/JP2/JP3 loop grades), `e596205` (judsearch, JS1), and
`3ac03de` (the JS ladder grades). Every rung was registered on GitHub #33 before its
measurement; evidence at `champion/evidence/jud_v1/`
(`jud_v1_predictions.md`, `jud_build_report.md`, `judsearch_build_report.md`). The
one-organ conception and the value-native pricing path are [[jud]]'s (Fable's forward design
by way of the 2026-07-05 Fable 5 session); the [[rank-vs-price]] mechanism it confirms is
Fable's.

## Links

- [[jud]] — the unified belief-conditioned core; v1 is its bid+play unification, built and graded
- [[w42-jud-v0]] — the value-native bidder v1 extends from bid-time to every decision
- [[w42-plateau-probe]] — where the night's best player (`margin:wp`(head_8)) was measured
- [[rank-vs-price]] — play consumes rankings, bids consume prices; v1 confirms both halves
- [[champion]] — the ladder; the current best player is stated there
- [[expected-q-value]] — E[Q] n=10, the oracle play v1's search could not out-rank; [[pimc]]
- [[arena]] — the harness; [[forge]] — the solve/oracle a v2 leaf would distill from

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- The per-trick MAE 8.6 → 3.5 figure traces to `jud_build_report.md`; the underlying eval
  JSON (`jud_net_eval_round0.json`) could be linked directly for provenance.
- Cheap next probe before v2: JS2 showed worlds aren't the constraint — a leaf-bias probe
  (same search, oracle leaf vs jud leaf, small n) would quantify how much of the remaining
  −1.16 gap is leaf vs rollout policy.
