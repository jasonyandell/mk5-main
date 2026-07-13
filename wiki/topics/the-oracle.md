---
title: The oracle — three solver generations, one compression, one pivot
kind: topic
first_seen: 2025-12-27
last_updated: 2026-07-11
status: active
---

## What it is

The genesis of what is today `forge/oracle/` — the perfect-information solver that
[[forge]] treats as Stage 1 ground truth, described here as it was actually built: three
generations in rapid succession, one compression that made it fit on consumer hardware, and
one pivot away from value regression that redirected the whole downstream pipeline.

The question, in Jason's own subjunctive form before it was a design: "I wonder if play
itself can be bootstrapped from pimc starting at late game and bootstrapping earlier?" (conv
085ffa71, 2025-12-24) — i.e. can backward induction from the endgame stand in for a PIMC
search already found intractable ("pimc is checking everything and takes minutes for one
game... I am pretty sure bidding is intractable actually. 21 choose 7. yikes," same conv). The
scope was deliberately narrowed against an assistant-generated "solve your grandmother's
game" framing: "I just want to solve a few seeds completely, not the whole entire game for
every possible deal. are we thinking along the same lines?" (conv 7343283b, 2025-12-27).

## Three generations, one day (Dec 27, 2025)

1. **CPU solver** (`scripts/solver/`, `b541a4b`, 01:59) — Python backward induction over
   complete regret tables. Deleted the same morning at 10:32 (`bc2400e`), called a
   "quagmire" in its own commit body.
2. **First GPU solver** (`6f82f9f`, 13:08) — PyTorch backward induction, 47-bit packed state,
   95 passing tests: "Solves seed=100 decl=0 in 22s with 10.3M states." Deleted that night
   (`bcd49f0`, 21:57, commit message literally "delete false start") once solver2 superseded
   it.
3. **solver2** (`ba07bc6`, 18:56) — 28-bit packed state, BFS enumeration + backward
   induction, Parquet output. 63-69% faster after int8-popcount + vectorization
   (`19893fc`, 20:22).

## The compression that mattered most

Bead `t42-ze7i` (2025-12-27): removing score from the packed state — score only offsets
value by a constant, the optimal action doesn't depend on it — cut state count **79M → ~2M**
and peak VRAM **~3.5GB → ~88MB**. This is the single biggest lever in the solver's life: the
difference between needing a datacenter and fitting an 85M-state solve on a 4GB consumer
GPU. Jason's own summary of the moment: "42 is like a funnel and chess is like a fractal. we
can solve it. we have solved it. tonight we solved one hand of 42 for all suits including
doubles and no-trump." (conv 3dd0968b, 2025-12-28).

## Folded into Crystal Forge (Dec 30, 2025)

`2559818` ("Crystal Forge: Lightning-first ML pipeline," 2025-12-30) moved solver2 wholesale
into `forge/oracle/` — the directory that exists in the repo today (`generate.py`,
`campaign.py`, `expand.py`, `solve.py`, `state.py`, `tables.py`, `schema.py`,
`declarations.py`, `context.py`, `output.py`, `rng.py`, `timer.py`). The old `scripts/solver2/`
tree and its archive copy were both deleted the same day (`cf8e6d7`). The naming moment
itself, unprompted delight: "the crystal forge sounds so badass. I can say 'yeah over in the
forge' and I feel like a cool dude." (conv bec8b3d4, 2025-12-30). The `mv0-mv6` move-value
column naming was renamed to `q0-q6` the same day (`77f3823`, 23:04 local — not Dec 31 as an
earlier draft of this record stated), giving the "Q-value" vocabulary that runs throughout
the project since.

## Training the oracle at scale, and the value-head pivot

A scaling-curve experiment (Dec 30) showed accuracy climbing with no plateau to 13.75M
samples. An architecture sweep on rented Lambda Labs H100s ("EXCITING OMG," conv b00686d3,
2025-12-31) broke a 94.5%-accuracy plateau: **97.1% acc / 0.11 q-gap** (817K-param Large
model, `2ea5cc1`), then **97.8% acc / 0.072 q-gap** with a value head (`fc3acc7`). That same
commit message drew the era's cleanest engineering conclusion: **"bidding needs simulation,
not regression"** — the value head, meant for bidding, plateaued at 7.4 points MAE and was
abandoned. That redirected the project to `forge/bidding/` (simulate whole games with the
policy model, count P(make)) — built Jan 1, vectorized 135x the same night (0.33 → 44
hands/min). The delight was real and immediately self-skeptical: given a hand, "how can there
be 2% chance of getting 42 by choosing 6s for trump. you don't have the 6-6. can we see an
example?" (conv 896a2663, 2026-01-02) — the answer (partner drew a monster by chance) is
exactly the kind of thing an analytic heuristic would miss.

## What it found

- 10.3M states solved in 22s (`6f82f9f`).
- State-space compression 79M → ~2M states, VRAM 3.5GB → 88MB (`t42-ze7i`).
- Policy accuracy: 92.92% → 94.53% plateau → **97.1%/0.11** → **97.8%/0.072** (Large v1/v2,
  `2ea5cc1`, `fc3acc7`).
- Value-head-for-bidding regression failed at 7.4 pts MAE — abandoned for simulation.
- One cross-validation discrepancy against the TS minimax engine (seed=100/decl=0, off by one
  point) was found and fixed same-day (`withNoBid()` added to `StateBuilder`, bead t42-1a6e).

## Honest terminal status

**BUILT, and alive today.** `forge/oracle/` is the Stage-1 ground-truth generator that
`forge/eq/` (Stage 2, E[Q] under imperfect information) and `forge/bidding/` both consume.
The colloquial "perfect-information oracle" framing crystallized on Jan 3, 2026 (conv
adb6de51) into the Stage-1/Stage-2 architectural split the project still runs on — Jason's
own three-model plan, stated the day before: "we will have 3 models: the current 800k model
trained on perfect play that is a perfect information oracle; a bidding hidden information
model; a play hidden information model." (conv 896a2663, 2026-01-02).

## What it ruled in/out about the wall

The oracle era is where the wall's precursor problem got named with mathematical precision:
a perfect-information value is trustworthy and cheap (10.3M states/22s, 97.8% policy
accuracy), and simulate-then-count beats value-regression as a consumption strategy — but
naively averaging perfect-information rollouts is provably an upper bound on
imperfect-information value ([[strategy-fusion]]), ruling out simple averaging as a bidding
evaluator. It solved evaluation; it did not yet solve strategy. See
[[breakthrough-and-oracle]] for the full arc.

## Data format and state-space shape

Each shard is one (seed, declaration) Parquet file with three columns: `state` int64
(the 41-bit packed game state), `V` int8 (minimax value-to-go, Team-0 perspective),
and `q0`–`q6` int8 (Q-value per local action, −128 = illegal). Declaration IDs:
0–6 are pip trumps (blanks, ones, twos, threes, fours, fives, sixes), 7 doubles-trump,
8 doubles-suit, 9 no-trump. (docs/solver2-data.md @ 233b7dc5)

State-space shape, measured on the 999-seed / 180 GB continuous campaign (3.2M–115M
states per shard): doubles-trump generates the largest trees (avg 33.8M states, also
the highest variance, σ = 28.2M) and doubles-suit the smallest and most predictable
(avg 23.1M, σ = 14.0M); the seven pip trumps and no-trump fall between. One seed in
1,000 blew the 160M-state cap (seed 434, fours, ~190M states) and was skipped.
(docs/oracle-state-space-analysis.md @ 233b7dc5)

## Links

[[suit-algebra]] · [[strategy-fusion]] · [[forge]] · [[breakthrough-and-oracle]] ·
[[the-analysis-epic]] · [[era2-breakthrough-oracle|conversation digest]]
