---
title: Intermediate AI — the shipped PIMC opponent
kind: topic
first_seen: ba123884
last_updated: pending-this-ingest
status: active
---

The pre-ML AI that shipped and still runs the web game's opponents: [[pimc]]
— sample opponent hands under tracked constraints, evaluate each sampled
world with partnership minimax to terminal, pick the play with the best
average team outcome. [[pre-ml-ai-attempts]] covers the *retired* probes
(AlphaZero ideated-only, MCCFR built-then-killed); this page covers the
survivor as it exists in `src/game/ai/` today. Part of the
[[engine-architecture]] cluster.

"Intermediate AI" is the historical name (the Monte Carlo evaluator's file
header still carries it); the machinery now lives inside
`BeginnerAIStrategy` in `src/game/ai/strategies.ts` — "the standard playable
AI." The only strategy types are `'beginner' | 'random'`
(`src/game/ai/actionSelector.ts`); a separate IntermediateAIStrategy class no
longer exists.

## Wiring (verified 2026-07)

The Svelte app's opponents are this AI: `gameStore` → `createLocalGame` →
`attachAIBehavior(client, i, { type: 'beginner' })` → `selectAIAction` →
`BeginnerAIStrategy.chooseAction`. The AI is an ordinary GameClient
subscription with no privileged reads ([[multiplayer-pattern]]).

## Decision pipeline

`BeginnerAIStrategy` is phase-dispatched:

- **Bidding** — Monte Carlo (`evaluateBidActions`,
  `src/game/ai/monte-carlo.ts`): for each candidate bid, sample unconstrained
  opponent hands, pick trump for the sampled hand via `determineBestTrump`,
  roll the full hand out, and measure the make rate. Bid the highest value
  whose make rate ≥ 0.50; otherwise pass. Default 5 simulations per bid.
- **Trump selection** — heuristic `determineBestTrump`
  (`src/game/ai/hand-strength.ts`): doubles with 3+, else strongest suit.
- **Play** — full PIMC via `buildConstraints` + `selectBestPlay`: for each
  candidate play, sample opponent hands consistent with everything observed
  (default 10 worlds per candidate), inject them, apply the candidate, and
  evaluate the rest of the hand with minimax. Highest average team points
  wins. Team points, not individual tricks — partnership dynamics emerge
  from the objective, not from rules.

## Constraint tracking

`src/game/ai/constraint-tracker.ts` maintains `HandConstraints`: dominoes
already played, the AI's own hand, and per-player void suits inferred from
failures to follow. Void inference delegates to the composed
`rules.canFollow` through a precomputed `CanFollowCache` — it never
re-derives follow-suit logic. This is the institutionalized fix for the
system's founding bug: a hand-rolled suit check that ignored trump exclusion
(with 4s trump, 4-0 is *only* a trump — not following blanks with it proves
nothing about blanks), which produced contradictory constraints and sampling
failures (docs/INTERMEDIATE_AI.md @ 233b7dc5). Lesson, still binding: never
re-derive game rules; call the composed [[layer-system]] rules.

## Hand sampling

`src/game/ai/hand-sampler.ts` (`sampleOpponentHands`) distributes the unseen
pool via backtracking search over candidate sets, guaranteed to find an
assignment if one exists. The invariant: a valid distribution MUST always
exist, because the real game state is one — so a sampling failure is a
constraint-tracking bug and throws with debug info rather than degrading.

## Per-world evaluation: minimax, not policy rollout

`src/game/ai/minimax.ts` (`minimaxEvaluate`): partnership minimax — players
0/2 maximize, 1/3 minimize — with alpha-beta pruning and heuristic move
ordering, searched to the actual terminal state (no evaluation heuristic at
a depth cutoff). It executes through `ctx.rules`, so special contracts are
respected, and handles neutral auto-execute actions transparently.
Simulations override `playerTypes` to all-AI so the consensus layer passes
through (see `createSimulationContext` in [[engine-testing-patterns]]).

Minimax replaced greedy heuristic rollouts at `3e063ff` (2025-12-21), curing
the "depressed android" — the defeatist AI that dumped count when losing.
Game-theoretic search finds fighting lines even in lost positions
([[pre-ml-ai-attempts]]).

## Support files

`domino-strength.ts` (static strength analysis),
`strength-table.generated.ts` (precomputed at build time via
`npm run generate:strength-table`), `utilities.ts` (`analyzeHand`),
`gameSimulator.ts` (batch AI-vs-AI runs over HeadlessRoom, seed search),
`types.ts` (`AIStrategy` interface: pure
`chooseAction(state, validActions)`).

This PIMC baseline is what the E[Q]/ML line measures itself against — the
caveat that a PIMC-vs-PIMC harness is structurally blind to
concealment/signaling value lives at [[champion]].

---

*Source: docs/INTERMEDIATE_AI.md @ 233b7dc5 (which described an earlier
layout: `strategies/intermediate.ts`, rejection sampling, beginner-policy
rollouts, 50 sims); rewritten against `src/game/ai/` at this ingest.*
