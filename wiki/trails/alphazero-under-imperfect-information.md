---
title: AlphaZero Under Imperfect Information (Era 4, 2026-02-01 .. 2026-02-18)
kind: trail
first_seen: 2026-02-01
last_updated: 2026-02-16
status: complete
---

## Overview

Era 4 is 16 calendar days (129 commits, solo, spilling to a final 2026-02-18 note) spent
answering one question left open by [[expected-q-value|E[Q]]]'s early-2026 closeout: E[Q]
computes p(make)/EV with no plan and no judgment ([[candlewax|the wall]]) — can a policy
that actually plans be learned the way AlphaGo Zero learned Go, from self-play alone,
under 42's imperfect information? The project's own answer, delivered as a question on the
way out the door: *"maybe the finding is 'alphago under imperfect information can carry
you pretty far, actually!'"* (2026-02-16T15:38,
[[era4-zeb-era|conversation digest]]).

Everything in this era routes through [[zeb]], the self-play player it built. This page is
the narrative hub; [[zeb]] is the entity-level detail on the model itself.

## The through-line: is there even a "best move" here?

The era opens by closing a different door — *"there are a lot of interesting discoveries
in here about 42 but ultimately they do not lead anywhere"* (2026-02-01T01:31) — and pivots
immediately: *"ok left turn"* → *"why is alphago zero style approach not appropriate for 42
again?"* The night MCTS first ran, the doubt that recurs for the rest of the era surfaces:
*"nobody knows what 'better' really is... there simply is no spoon because it's imperfect
information and 'best' move has a random component and always will.. right?"*
(2026-02-02T00:02). The project checked its own drift against this doubt twice — evaluating
and rejecting an external RL library (Sample Factory) in one 85-minute session, and naming
the pattern directly: *"my gut says we haven't actually tried the alphazero approach, we've
tried proxies and standins and maybe-instead-ofs"* (2026-02-02T21:28). Underneath the
vigilance: *"I don't want to be disappointed again, I admit it. I thought PIMC was gonna be
sweet. it wasn't"* (2026-02-05T04:32).

## The wall, stated precisely, mid-build

Before any experiment tested it, the project named the actual shape of the problem it was
about to run into:

> "there's no such thing as an e[q] optimal policy unfortunately. it's a distribution, an
> often lumpy, often smooth histogram of discrete scores, not something you can just pick
> 'best' for that application." (2026-02-06T04:00)

> "but I don't know how to make great games buddy. ultimately I have to pick a next move
> from that melted candle wax in order to even make stage 2 training data and I have NO
> confidence that it selects moves well, with good judgement, only that it is extremely
> well informed about the statistical landscape." (2026-02-06T04:10)

This is [[candlewax]], said in Jason's own words at 4am mid-era, not synthesized after the
fact.

## What got built

- [[zeb]] itself: MCTS + self-play, three model sizes (75K/557K/3.3M params), a belief
  head, 1M+ then 1.7M+ self-play games, climbing from ~50% to a 76.3% peak vs random.
- [[zeb-fleet-ops]]: a self-healing Vast.ai spot-GPU fleet with HF Hub as the
  worker/learner exchange bus, reputation scoring, and a CQRS monitor rewrite — a solo
  hobbyist running a real cluster at pennies per hour.
  [[eval-matrix-bradley-terry]]: a Bradley-Terry Elo ranking putting `random`,
  `heuristic`, every `eq:n=*`, and every `zeb:*` checkpoint on one scale — `zeb-large-belief`
  landed at 1579, neck-and-neck with the `eq:n=100` anchor at 1600.
- [[full-teacher-eq-experiment]]: the era's central experiment, closed 2026-02-16 —
  feeding E[Q]'s per-action signal straight into the policy head, at 25% then 95% mix,
  never moved play past ~74% vs random.

## What it settled, and what it left open

**Ruled out:** naive E[Q]-as-policy-teacher. Distilling the oracle straight into the
policy head does not push play past the ceiling self-play alone already reached — "just
increase the oracle dose" is dead ([[full-teacher-eq-experiment]]).

**Ruled in, qualified:** self-play/MCTS can learn a competent imperfect-information
policy from scratch. 50% → ~76% vs random is real and checkpoint-backed. The prerequisite
question — can AlphaZero's shape even get off the ground under 42's hidden information —
is answered yes, capped.

**Left explicitly open:** whether the cap is capacity, architecture, or the structural
fact that E[Q] has no single "best" to pick — plus the era's real, underclaimed bequest:
[[belief-feeding-policy]], the belief head built 2026-02-08 that never fed a move.

**One hard fact for the record:** no Zeb-descended policy has ever been recorded beating
[[expected-q-value|E[Q]]] n=10 at pure play, in this window or since. The 2026-07-06
[[w42-jud-v1|jud v1]] verdict (`afd4802`) names E[Q] n=10 "the play champion, as it has
against every learned challenger since Zeb." Neck-and-neck (Elo 1579 vs 1600) is not beat.

## What the era's headline numbers actually measure

Every vs-random percentage in this era — 70.9%, 76.3%, the ~74% "ceiling" — is graded
against total points scored, not marks-to-7, the game's actual win condition. Jason flagged
this himself mid-era and it was never fixed in-window:
[[vs-random-eval-is-suspect]].

## Corrections to the received story

- **"Reinforce-to-flywheel" / "six weeks of RL training"** was never an experiment,
  module, or bead — the title of one conversation plus a recap sentence. The six-week arc
  it names continued entirely under Zeb.
- **Sample Factory was never built** — evaluated and rejected in one 85-minute session.
- **The "Crystal Palace word cloud" was a 3-minute aside**, not a course of work; "Crystal
  Palace" itself predates this era by 2-3 months.
- **Beads tracked zero activity in this window** — all era-4 work is git- and
  conversation-sourced.
- **Zeb was not abandoned at the era's close.** `forge/zeb/` survived on disk and received
  a commit as recently as 2026-07-06. The full-teacher experiment was formally closed by a
  deliberate ops commit stating a finding, not abandoned mid-run.

See [[zeb]] · [[zeb-fleet-ops]] · [[full-teacher-eq-experiment]] ·
[[eval-matrix-bradley-terry]] · [[vs-random-eval-is-suspect]] · [[belief-feeding-policy]] ·
[[candlewax]] · [[the-gestation]] (the era that follows) ·
[[era4-zeb-era|conversation digest]].
