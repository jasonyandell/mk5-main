---
title: Otis Phase 1 — the first lesson consumer (retention override on lens:ev)
kind: experiment
first_seen: 2026-07-15
last_updated: 2026-07-15
status: active
---

## Questions

Executing [[otis]] Phase 1 per the bridge ticket
([issue #55](https://github.com/jasonyandell/mk5-main/issues/55)), which is
[issue #53](https://github.com/jasonyandell/mk5-main/issues/53): the
unanswered half of [[count-fate-ledger]]'s open question. Overnight
2026-07-15→16, worktree branch `worktree-otis-night2`, arena-side
(contamination-free — no stored eq-corpus worlds are consumed). In question
form:

1. Does a slough-decision override consuming tied-rollout retention prices
   (the [[otis-v0]] P7 instrument, online ~1 s/decision) beat the incumbent
   `margin:wp(r8)+lens:ev` in paired marks? (V1)
2. Does the same override consuming otis fate-head participation scores
   (instant) beat the incumbent? (V2)
3. Mechanically: how often does the trigger fire on-policy, and how often
   does the override actually disagree with `lens:ev`'s default slough — is
   the lever non-vacuous? (M1, M2 — measured before any marks gate)

The trigger is [[otis-v0]] W6's genuine-retention predicate: the actor is
void in the led suit AND holds ≥ 2 legal non-trump junk (non-count) sloughs.
Everything else about the player is bit-identical to the incumbent (one flag
per variant). Play-side levers have a graveyard (#25 belief-weighted
marginalization, #27 score-conditioned utility, #33) — this is the
narrowest well-posed entry, and **a null at power is a graded result**: it
says prices this size don't cash at the table.

## Protocol pins

- Incumbent both sides: `margin:wp(model=champion/margin_net_r8.pt)+lens:ev`
  ([[stage-0-closure]] convention), n_samples=10, MPS, repaired sampler.
- Variant = same bidder, same lens, plus the one-flag slough override —
  pure play-side delta.
- Paired blocks per the stage-0 protocol: reserved block `--base-seed
  7000000`, fresh block `--base-seed 9000000`. Power discipline per
  [[otis-v0]] W7: 512-game blocks cannot resolve < 0.3 marks/game — ties
  are graded only after pooling to ≥ 2,048 paired games (V1, online cost)
  or a single 4,096-game block (V2, instant). Declared up front: V1 runs
  512-game blocks and pools; V2 runs 4,096-game blocks.
- The vacuity pre-gate (M2) runs BEFORE any marks gate: on ≥ 200 triggered
  decisions drawn from on-policy incumbent self-play, measure how often the
  override changes the action. If disagreement < 5%, the variant is graded
  **vacuous** (its marks gate cannot move and is skipped) — a distinct
  outcome, not a null.

## Predictions — registered before any run

Bands written 2026-07-15 before the override existed. Priors: the play-lever
graveyard says tie; W6 says the prices are real (median |tied Δ| 0.68,
p90 2.88 points on ordered cells) but small per decision; the marks
conversion runs through ~2 triggered decisions per hand for the overridden
team.

| # | claim | band (pass) | falsifier (genuine negative) | graded |
|---|---|---|---|---|
| M1 | Trigger fires on-policy | [0.8, 4.0] triggered decisions per hand summed over the overridden team's two seats (W6 prior: 2.2/hand on bid-30 fixtures) | < 0.3/hand — the fixture-derived trigger barely exists on-policy; the lever's surface is too small to gate | **PASS — 2.37/hand** (843 triggers / 355 hands, 32-game shadow run, seed 5100000) |
| M2 | The lever is non-vacuous | override disagrees with lens:ev's default slough on [15%, 70%] of triggered decisions (each variant separately) | < 5% — vacuous; marks gate skipped for that variant, graded as such | **PASS (V1) — 54.1%** (456/843); shadow mean claimed margin +0.73 pts among disagreements; 0.27 s median/trigger |
| V1 | Tied-rollout override, paired marks vs incumbent | REGISTERED MEDIAN: tie — pooled ≥ 2,048 games, \|Δ\| < 0.15 marks/game, CI includes 0. HOPE: Δ > 0 with CI excluding 0 | Δ < 0 with CI excluding 0 — the override actively hurts: tied prices on M=50 live worlds do not transfer to table play | |
| V2 | Fate-head override, paired marks vs incumbent | REGISTERED MEDIAN: tie — 4,096-game block, \|Δ\| < 0.10 marks/game, CI includes 0. HOPE: Δ > 0 with CI excluding 0 | Δ < 0 with CI excluding 0 — participation scores at incumbent-parity capacity mis-price retention | |
| V2-c | Play-state fate head calibrates (gate before V2 may run) | held-out per-tile fate NLL on PLAY states beats the marginal base rate by ≥ 0.15 nats (the [[otis-v0]] P2 band), improving with trick depth | ≤ 0.03 nats — play-state fates not learnable at this capacity; V2 stays unrun and that is the graded finding | **PASS — +0.5712 nats** (val 13,965 rows; every tile +0.56 to +0.64; by trick 0→6: 0.47→0.63, monotone) — V2's marks gate is licensed |
| V1-m | Mechanism receipt (V1, descriptive) | among disagreements, the tied price of the override's choice exceeds lens:ev's choice by a positive mean margin (the price it claims to cash); reported with its distribution | mean ≤ 0 — the override is not even claiming value where it acts (instrument bug or trigger mismatch) | |

"Best player yet" claims require the two-block standard (reserved 7M AND
fresh 9M, both CI-excluding-zero) — nothing weaker. If wall-clock forces a
choice, V2's 4,096-game block outranks V1's later pools (declared drop
order: V1 pooling beyond 1,024 games is the first thing dropped; an
ungraded pool extension is "not run", never "failed").

## Pre-run amendment: the play-state fate head (2026-07-15)

Discovered during implementation, before any run: the otis v0 net's
featurization is the 91-dim declarer-hand ⊕ canonical-auction row, imported
verbatim from `champion.margin_net` — **bid-time only**. The fate heads
cannot condition on play history, so the v0 net has no per-candidate-discard
score at a mid-hand slough decision; issue #53's V2 shape assumed a
play-state fate head that v0 never built.

Per Jason's same-session directive, the missing net is built tonight
rather than parked: **OtisPlayNet** — an info-state featurizer over the
actor's perspective (remaining hand, played set, current trick, voids,
declaration, auction context, score) feeding the same five 8-class fate
heads + trick head, trained on per-ply rows expanded from the
[[otis-v0]] W2b fate corpus (labels are per-hand; every ply inherits
them; the deal-hash split is per-hand so no ply leaks across splits).
V2 may run its marks gate ONLY if V2-c (the calibration gate above,
registered before training) passes. V2's marks band is unchanged.

## Method

Hook: `arena/lens_play.py::LensPlay._select` is the documented override seam
(rungs #25/#27 both hooked here). A `SloughOverridePlay(LensPlay)` recomputes
the action for triggered states only:

- **V1 (tied)**: sample M=50 valid worlds live (repaired `WorldSamplerMRV`),
  weight by the [[gus]] v3 belief head (the [[otis-v0]] W5/W6 weighting),
  and for each candidate junk slough force it as the first action of a
  tied-by-construction rollout (`otis/tiedroll.py` machinery: every seat a
  deterministic function of its own info-state); choose the candidate with
  the highest belief-weighted mean my-team points. Common random worlds
  across candidates.
- **V2 (fate heads)**: for each candidate junk slough, featurize the
  post-discard information state and read the otis fate heads; score =
  Σ over count tiles of value(t) · P(my team captures t) + trick-head
  E[my-team tricks]; choose the argmax. One batched forward per decision.

Gates run on the M5 (one GPU job at a time — they queue behind nothing;
Track A owns the remote 4090). Every gate names bidder, player, sampler,
utility per [[stage-0-closure]]; paired-CI from the same per-game rows as
prior gates.

## Receipts

### V1 blocks A+B (2026-07-15, LIVE override, 512 paired games each)

Specs pinned as registered; per-block and pooled CI via
`arena.match._bootstrap_ci_mean` (method-identical to prior gates):

| block | seed | games | mean Δmarks | 95% CI |
|---|---|---|---|---|
| A (reserved) | 7000000 | 512 | −0.1523 | [−0.4395, +0.1309] |
| B (fresh) | 9000000 | 512 | −0.2793 | [−0.5567, +0.0078] |
| **pooled** | | **1,024** | **−0.2158** | **[−0.4199, −0.0068]** — CI excludes 0 |

Both blocks independently negative; interim pool (half the registered
2,048) already excludes zero on the loss side. Mechanism receipts from the
live triggers (27,662 total): 54.3% disagreement, mean claimed margin
**+0.75 pts** among disagreements, 0.27 s median per trigger — the
override fires exactly as the shadow run predicted and *claims* value
every time it acts. The table pays the claim back as a marks loss. Final
V1 grade waits on the C/D pool to 2,048 (registered power).

(V2 4,096-game block + C/D fill here as runs land)

## Links

[[otis]] · [[otis-v0]] · [[count-fate-ledger]] · [[strategy-fusion]] ·
[[stage-0-closure]] · [[gus]] · [[jud]]
