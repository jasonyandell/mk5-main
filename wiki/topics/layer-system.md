---
title: The Layer System
kind: topic
first_seen: 2025-11-02
last_updated: 2026-07-11
status: active
---

The unified Layer system is how the engine supports special contracts (nello,
splash, plunge, sevens) and UX modes (speed, consensus, hints, oneHand,
tournament) without a single conditional in the executors. Part of the
[[engine-architecture]] cluster; origin story in [[web-game]] (the October
2025 "variants are different games with shared mechanics" crisis and the
`GameLayer` landing).

## The problem it solves

Special contracts change *both* what actions are possible *and* how execution
works — nello has 3-player tricks and can end a hand early. Flag-and-
conditional designs put `if (nello)` in every executor; the Layer system
instead gives each layer two orthogonal composition surfaces:

```typescript
interface Layer {
  name: string;
  rules?: { /* per-method overrides, each receiving prev */ };
  getValidActions?: (state, prev: GameAction[]) => GameAction[];
}
```

(`src/game/layers/types.ts`)

## Surface 1: execution rules (GameRules)

`GameRules` is an interface of 18 pure methods that determine HOW the game
executes. Executors delegate every decision to them and never inspect state
for mode:

- **WHO** (3): `getTrumpSelector`, `getFirstLeader`, `getNextPlayer`
- **WHEN** (2): `isTrickComplete`, `checkHandOutcome`
- **HOW** (6): `getLedSuit`, `suitsWithTrump`, `canFollow`, `rankInTrick`,
  `isTrump`, `calculateTrickWinner`
- **VALIDATION** (3): `isValidPlay`, `getValidPlays`, `isValidBid`
- **SCORING** (3): `getBidComparisonValue`, `isValidTrump`, `calculateScore`
- **LIFECYCLE** (1): `getPhaseAfterHandComplete`

The count is not fixed — the original threaded-rules design had 7 methods
(docs/archive/pure-layers-threaded-rules.md @ 233b7dc5) and it grew as modes
needed new execution semantics (`checkHandOutcome` for nello/plunge early
termination, `getPhaseAfterHandComplete` for oneHand's terminal state).
Adding a method — base default plus layer overrides — is the sanctioned
extension mechanism; adding a conditional to an executor is the anti-pattern.

`checkHandOutcome` returns the discriminated union `HandOutcome`:
`{ isDetermined: false }` or
`{ isDetermined: true; reason: string; decidedAtTrick?: number }` — the
reason is inaccessible unless determined, by type.

**Composition**: `composeRules(layers)` (`src/game/layers/compose.ts`) reduces
the layer list; each override receives the previous layer's result as its
last parameter (`prev`) — return `prev` to pass through, or a new value to
override. Layers implement only the methods they change.
`src/game/layers/rules-base.ts` is the single source of truth for base
trump/suit/follow-suit semantics.

## Surface 2: action generation

`getValidActions(state, prev)` transforms WHAT is possible. Four operations:
**filter** (tournament removes special bids), **annotate** (hints adds hint
metadata; speed adds `autoExecute`), **script** (oneHand injects a bidding
sequence), **replace** (consensus swaps `complete-trick`/`score-hand` for
`agree-trick`/`agree-score`). Also composed by reduce —
`composeGetValidActions` chains each layer over the previous output.

The decision rule for which surface to use: if an *executor* needs the
behavior, it's a GameRules method; if only action availability changes, it's
`getValidActions`.

## The layers (LAYER_REGISTRY, verified 2026-07)

`src/game/layers/registry.ts` registers exactly ten:

| Layer | What it does |
|---|---|
| `base` | Standard 4-player Texas 42; all 18 rule defaults |
| `nello` | Partner sits out: 3-play tricks, bidder must lose all tricks, early hand end |
| `plunge` | Partner selects trump and leads; bidder team must win all tricks |
| `splash` | Plunge-like, requires 3+ doubles |
| `sevens` | Closest-to-7 wins; no follow-suit |
| `tournament` | Filters special-contract bids out |
| `oneHand` | Single-hand mode: scripted bidding/trump, terminal `one-hand-complete` phase |
| `speed` | Auto-executes forced moves: single legal action → `autoExecute: true` with `authority: 'system'` |
| `hints` | Annotates actions with hint metadata |
| `consensus` | Gates `complete-trick`/`score-hand` behind `agree-trick`/`agree-score` from every *human* player (AI doesn't vote; all-AI games pass through) |

## Composition point and ordering

`createExecutionContext(config)` (`src/game/types/execution.ts`) composes
`[baseLayer, ...enabledLayers]` into a frozen ExecutionContext. The consensus
layer is **always appended last** (added automatically if not in the config's
layer list) so it can intercept `complete-trick`/`score-hand` produced by any
earlier layer, including speed. Only `Room` and `HeadlessRoom` call
`createExecutionContext` — enforced by ESLint and
`src/tests/architecture/composition.test.ts`.

## Layer state inspection is intentional

Layers check `state.trump.type` to decide whether their rules apply
(`state.trump?.type === 'nello' ? 3-play logic : prev`). This is not a
violation: nello is composed at config time but *activated* by player choice
at trump selection. The check is explicit, local to the layer's file, and
keeps executors mode-agnostic. Config-time-only alternatives were evaluated
and rejected — they move the complexity without eliminating it
(docs/ORIENTATION.md @ 233b7dc5).

## Adding a mode (recipe)

1. Create `src/game/layers/myLayer.ts` implementing only the overrides
   needed.
2. Register it in `LAYER_REGISTRY`.
3. Enable via `GameConfig.layers: ['myLayer']`.
4. If an executor needs new mode-specific behavior: add a GameRules method
   (interface → base default → layer override → thread through
   `composeRules`) instead of a conditional.

No executor changes. Rule conformance across base and special contracts is
enforced by `src/tests/guardrails/rule-contracts.test.ts`
([[engine-testing-patterns]]).

---

*Sources: docs/ORIENTATION.md, docs/CONCEPTS.md,
docs/archive/pure-layers-threaded-rules.md @ 233b7dc5, verified against
`src/game/layers/` at this ingest.*
