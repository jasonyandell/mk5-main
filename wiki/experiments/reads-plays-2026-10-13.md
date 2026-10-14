---
title: Zeb reads the other seats' plays, and restricted choice is priced (2026-10-13)
kind: experiment
first_seen: 2026-10-13
last_updated: 2026-10-14
status: complete
---

## Summary

Deployed Zeb plans over worlds uniform on its cell. It reads how the other seats played only
through the voids. [[lumps-voi-2026-10-13]] found money tied to who holds a key tile (the 5-5
first), late in the hand. It also found that restricted choice exists in 42. Here the live
seat's worlds are drawn from a posterior that reads every other seat's plays, under two
likelihoods, and the reader plays plain Zeb in the [[arena]]. No perfect-information value
enters ([[strategy-fusion-fence]]): the model is the field net Zeb already plans against, and
every value is Zeb's plan.

**What it shows.** Reading is a **null in marks** under both likelihoods. Consistent:
−0.09 [−0.28, +0.11] marks/game. Uniform: +0.01 [−0.19, +0.22]. Restricted choice, the
uniform arm less the consistent one on the same deals, is **+0.10 [−0.15, +0.36]**: also a
null (320 games per arm, 160 deal-paired game pairs). Reading changes the move, though. On
exact fibers, where neither side samples, the reader plays a different tile at 10.0%
(consistent) and 11.1% (uniform) of decisions, against a chance level of 0. The uniform
likelihood moves 3.6% of decisions away from the consistent reader's tile. The changes
carry the lumps study's tiles: the 5-5 and the 6-4 lead the mass the reading moves, then
the 4-4 and the 5-count tiles. The model fails the record it reads. By trick 6 the real
deal fits it at 34% of decisions, and the posterior is empty at half of trick 6's
decisions, which then fall back to the uniform. So the binding constraint is the model of
the other seats, as #111 found.

## Prior work, and what this adds

- **#111, the posterior sampler** ([[scenario-player]] § Foundation repair). `d=posterior`
  draws from β^F, the net's ε-greedy field: a 0/1 filter on the net's single move at ε = 0.
  It pays +0.026 marks/hand against seats that *are* the net. In Zeb self-play its
  posterior was empty at most late positions and null at power.
- **Night 1's `dice` and `field` tilts** (retired 2026-10-12): 1/n over every legal tile and an
  ε-greedy rung-0 move, both *reweighting* 512 uniform worlds. Nulls and losses, with the
  ESS collapse named as the mechanism.
- **The convention reader**: a 0/1 likelihood on the partner's gated plays. [[silas]]
  `Reading`: exact consistent and uniform likelihoods for a book policy's allowed set,
  per seat, by counting.
- **Self-minds** in texas-42's bridge Walt are a different thing: future-me as a level-0 mind
  inside the plan, not a reading of past plays.

New here: the model's *set* rather than its argmax, so the two likelihoods differ and
restricted choice can exist. Also a draw that stays exact when the likelihood is sharp.

## Fixed before running

Stated in `scenario/reads_plays.py` before the measured runs. One addition, made after a
first look at one batch, is marked.

- **The model of every other seat** is the deployed field net `margin_d250k_h128x64`.
  Zeb's own policy as the model (self-minds) costs a median **12.2 s per decision on all 18
  cores** at its cheapest form (512 importance weights, no rejection): 505–2,188 distinct Zeb
  decisions per reading decision, against 0.03–0.08 s for the decision itself
  (`cost.json`). That is several hundred times the wall, so the net stands in. It is a
  distillation of Zeb's play, with a regret of 0.35 points per decision.
- **The set** at a play is the legal tiles the net scores within **δ = 2 points** of its best.
  That is one logit of its training temperature τ = 2, since the target was
  softmax(margin/τ). The net's integer unit is fitted as the temperature minimizing its own
  training loss: 2^22.125 on both pours, so the margin is 2^22. On the 5,000-deal pour the
  set averages 2.45 of 3.65 legal tiles, holds two or more at 74% of decisions, and holds a
  move Zeb's plan rates best at 96%.
- **Likelihoods**, per discretionary play of each other seat, multiplied over plays and
  seats. Each factorizes by seat, as Silas's do. *Consistent:* [tile ∈ A]. *Uniform:*
  [tile ∈ A]/|A|. `Config.reading`; arena `d=posterior,r=consistent|uniform:4194304`.
- **The draw.** A fiber of at most 512 worlds is weighted by L normalized (exact, given the
  model). A fiber of at most 2^20 is enumerated, and 512 worlds are drawn from it with
  replacement in proportion to L (exact draws). A larger one is drawn by rejection from the
  uniform stream (exact). A posterior that is empty, or a budget that runs out, plans over
  the uniform sample and counts as a *fallback*. Sampled draws are labeled Sampled with the
  config seed 0.
- **Comparison.** The arena's paired halves: `bid30` both teams, games to 7, each game
  seed played with the reader as team 0 and as team 1 over the same deals. Batches of 32
  game pairs at base seeds 1–5 per likelihood, each under `timeout 300`. Seed 6 overran the
  cap in both arms and was dropped, not rerun. Metric: the reader's mark margin per game.
  95% bootstrap interval over game pairs, so it is deal-clustered. Restricted choice is
  the uniform arm's pair margin less the consistent arm's, pair by pair.
- **Decision change.** At each reading-side decision with two or more legal tiles, the
  reader's tile against plain Zeb's tile at the same state, along the reader's own games.
  *Chance baseline* (added after one batch, before the measured runs): plain Zeb at seed 1
  against seed 0, which is what a fresh sample alone changes. It is zero on an exact fiber.
- **Where.** Change rates by trick, by seat relative to the bidder, and by draw path. Per
  hidden tile and lump (the mover's left, partner or right holds it), |P_read − P_plain|
  at changed decisions against unchanged ones.

## Marks

| arm | game pairs (games) | marks/game, reader − plain | contracts made, reader / plain bidding |
|---|---|---|---|
| consistent | 160 (320) | **−0.09 [−0.28, +0.11]** | 36.4% / 37.1% |
| uniform | 160 (320) | **+0.01 [−0.19, +0.22]** | 37.1% / 36.9% |
| uniform − consistent (restricted choice) | 160 | **+0.10 [−0.15, +0.36]** | |

All three are nulls. The intervals are about ±0.2 marks/game, so an effect the size of
#111's against nets (+0.026 marks/hand, roughly +0.3 marks/game) would show in neither direction.
Reading here does not help a Zeb that plays Zebs.

## How often reading changes the move

| | consistent | uniform | chance (plain at another seed) |
|---|---|---|---|
| every decision (n ≈ 30,500 each) | 14.9% | 17.6% | 14.8–14.9% |
| exact fiber (no sampling on either side; n ≈ 5,850) | **10.0%** | **11.1%** | 0 |
| enumerated draw (fiber 513 to 2^20) | 23.5% | 24.8% | 16.5% |
| rejection (trick 1–2) | 15.4% | 22.6% | 31.1% (not comparable: the reader's draw shares plain's stream) |
| fallback | 0 | 0 | 1–2% |
| uniform reader against the consistent reader | | 12.1% (exact fiber 3.6%) | |

Overall, the consistent reader's change rate is no higher than a resample. On exact
fibers the 10–11% is reading alone, with no chance level beneath it.

**By trick**, every decision (consistent / uniform / chance): trick 1 11% / 18% / 33%,
trick 2 22% / 28% / 26%, trick 3 25% / 26% / 20%, trick 4 19% / 21% / 11%, trick 5
10% / 11% / 1.5%, trick 6 2% / 3% / 0. On exact fibers (tricks 4–6): 20% / 22% at trick 4,
13% / 15% at trick 5, 4.5% / 5.2% at trick 6. The excess over chance peaks at **tricks 4–5**,
where [[lumps-voi-2026-10-13]] put the money.

**By seat** on exact fibers (consistent / uniform): bidder 9.2% / 11.2%, left setter
9.4% / 9.4%, **partner 11.1% / 12.3%**, right setter 10.3% / 11.3%. The partner changes
most, by a little.

**Tiles.** The changed moves are mostly swaps among small tiles that are close in value:
5-3 for 5-2, 2-1 for 1-0, 3-3 for 0-0, 6-1 for 3-1. On exact fibers they include the
5-5 for the 5-3 and the 6-4 for the 4-0. The mass the reading moves, at changed decisions
against unchanged ones (the lift, summed over the three lumps):

| | first | second | third | then |
|---|---|---|---|---|
| consistent, every decision | **5-5** +0.114 | **6-4** +0.109 | **4-4** +0.066 | 5-0, 4-1 |
| uniform, every decision | **6-4** +0.110 | **5-5** +0.105 | 5-0 +0.066 | 4-1, **4-4** |
| consistent, exact fibers | **5-5** +0.128 | **6-4** +0.125 | 3-2 +0.063 | 5-0, 4-1 |
| uniform, exact fibers | **6-4** +0.132 | **5-5** +0.122 | 5-0 +0.070 | 4-1, 0-0 |

These are lumps-voi's top three and its 5-count tiles. The partner's lump holds 34.5% of the
moved mass at changed exact-fiber decisions against 30.3% at unchanged ones (consistent),
and 34.7% against 33.3% (uniform): a lean toward the partner, much weaker than lumps-voi's
43% of the 5-5's VOI. Where reading moves Zeb, it moves on the tiles worth knowing. It
does not move enough, or rightly enough, to show in marks.

## The model and the sample

| trick | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| real deal fits the model (L > 0) | 94% | 82% | 67% | 53% | 42% | 34% |
| posterior empty, fallback (consistent arm) | 0% | 0.5% | 2.9% | 11% | 28% | 51% |
| draw path (consistent arm) | rejection | rejection 68%, enumerated 32% | enumerated | enumerated 79%, exact 10% | exact 52%, enumerated 20% | exact 49% |
| median rejection acceptance, consistent / uniform | 0.85 / 0.23 | 0.53 / 0.04 | | | | |
| importance ESS fraction on the uniform 512, consistent | 0.90 | 0.42 | 0.29 | 0.17 | 0.085 | 0.011 |
| the same, uniform likelihood | 0.64 | 0.28 | 0.18 | 0.084 | 0.050 | 0.011 |
| ESS zero (no uniform world explains the record) | 0 | 1% | 2% | 11% | 38% | 50% |

The ESS row comes from every 50th decision of the consistent batches, 96–107 per trick
(`ess.json`). The real-deal row is a diagnostic only: the true deal is never used to play
or to score.

- **The weights collapse.** As importance weights the uniform likelihood leaves about 43
  of 512 worlds at trick 4 and 25 at trick 5. Reweighting would be planning on a handful,
  as on night 1. The draw keeps 512 exact draws instead, and the ESS rows say how sharp the
  posterior is.
- **The model does not fit Zebs.** The net's 2-point set holds a move Zeb's plan rates best
  at 96% of decisions. The product over a hand's discretionary plays excludes the real deal
  at two thirds of trick-6 decisions. Late in the hand the posterior is often empty, and the
  reader then plays as plain Zeb. That fallback is half of trick 6 and a quarter of trick 5:
  exactly where lumps-voi's money sits. It caps any gain there.

## Restricted choice in the toy and in play

`scenario/tests/test_reading.py` reproduces the lumps study's constructed deal with the
reading: every seat read as allowing every legal tile. From the partner's chair after the
5-4 lead and the left setter's 5-0, P(left holds the 5-5) is 6/19 consistent and
**615/2708** uniform, and P(the 5-0 was his only five) is 715/3876 consistent and
**715/2031** uniform. Those are Silas's exact counts, to 10⁻¹². In play, the uniform
likelihood moves 3.6% of exact-fiber decisions off the consistent reader's tile, and the
marks it buys are +0.10 [−0.15, +0.36] per game: restricted choice is real in the
arithmetic and unpriced at the table at this power.

## Caveats

- One model (the net's set at δ = 2 points) and one δ. A wider δ fits Zebs better: the set
  holds a Zeb-best move 99% of the time at 4 points. It also says less. Not swept.
- The other seats are Zebs, which the net imitates but is not. Against seats that are the
  net, #111's argmax filter paid. This reader was not run against a net table.
- Marks are per game with games to 7 under bid30. The reader's change rate along its own
  trajectory includes positions plain Zeb would not have reached.
- The mass shifts compare two samples of the same cell. Sampling noise sits in both the
  changed and the unchanged means, and the lift removes it on average only.
- Self-minds were costed, not run.

## Next (#145)

Done 2026-10-14: [[reads-plays-soft-2026-10-14]]. The softened set pays at a net table, with no
fallbacks. Against Zeb its registered replication failed in the consistent form, +0.12
[−0.02, +0.27], and held in the uniform form, +0.23 [+0.08, +0.37]. Restricted choice is null at both tables.

- Soften the set so the real deal never vanishes: F_σ's ε over the set, or a temperature
  on the net's scores in place of a hard δ. Then rerun the arms. The fallback rate and the
  real-deal fit say this is where the null comes from, not the likelihood's form.
- Run the reader against a table of nets (#111's design), where the model is true by
  construction, to price restricted choice where it should exist.

## Data

Public in `jasonyandell/texas-42-zeb-research-2026-10` under `zeb/reads-plays/`
(CC0, `SHA256SUMS`; `NOTES.md` lists the columns): per batch `meta.json`, `games.jsonl`,
`decisions.jsonl.gz`; `summary.json`, `ess.json`, `cuts.json`, `cost.json`, the scale fits,
and the code. Recompute
([publish-data-to-huggingface](../playbooks/publish-data-to-huggingface.md)):

```bash
NET=scratch/field_net/v3/margin_d250k_h128x64.npz
python -u -m scenario.reads_plays scale --pour scratch/atlas/pour_make_s1001_n512_d5000 --net $NET
python -u -m scenario.reads_plays cost --net $NET --positions 8 --out cost.json
timeout 300 python -u -m scenario.reads_plays arena --net $NET --likelihood uniform --seed 1 --games 32 \
    --out runs/uniform_s1                      # and consistent; seeds 1-5 (run_all.sh)
python -u -m scenario.reads_plays summarize --runs runs/* --out .
python -u -m scenario.reads_plays ess --net $NET --runs runs/consistent_s* --every 50 --out ess.json
```

Code at mk5-main `5ee71159`. Tests: `scenario/tests/test_reading.py` (the reading in
both engines, every draw path, the restricted-choice toy against Silas) and
`scenario/tests/test_reads_plays.py` (the study's bookkeeping).

## Links

[[scenario-player]] · [[lumps-voi-2026-10-13]] · [[silas]] · [[arena]] ·
[[strategy-fusion]] · [[research-night-2026-10-12]]
