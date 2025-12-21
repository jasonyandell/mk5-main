# Trump Unification Plan (Crystal Palace)

## Vision
Build a single, elegant, rule-centered system where trump, suits, and follow-suit semantics live in exactly one place. Every decision in gameplay, AI, and UI derives from that source. There are no alternate implementations, no bypasses, and no hidden forks.

## Non-Negotiables
- One source of truth for trump/suit/follow-suit semantics.
- No legacy paths, no compatibility shims, no duplicated rule logic.
- All rule-aware computations flow through composed `GameRules` in the `ExecutionContext`.
- The client is dumb. It receives only filtered, serialized state and derived view fields.
- Special contracts (nello, sevens, plunge, splash) remain layer overrides with zero executor conditionals.

## Current Fractures (From Opus)
- Suit/trump logic is duplicated across `base.ts`, `compose.ts`, and `core/dominoes.ts`.
- Trick winner logic in `core/scoring.ts` bypasses layers.
- UI utilities implement trump logic (`dominoHelpers`, `domino-sort`), violating the client boundary.
- Multiple ad-hoc "isTrump" and follow-suit checks exist with slight variations.
- `suitAnalysis` is a cached field on state, creating a stale-cache risk and inflating state surface.

## Canonical Semantics
These are the only rule-aware entry points. Everything else is either pure pip math or a defect.

- `rules.getLedSuit(state, domino)`
- `rules.suitsWithTrump(state, domino)`
- `rules.canFollow(state, ledSuit, domino)`
- `rules.rankInTrick(state, ledSuit, domino)`
- `rules.calculateTrickWinner(state, trick)`
- `rules.isTrump(state, domino)`

## Final Layout (End State)

### Single Source of Truth
- `src/game/layers/rules-base.ts`
  - The sole implementation of base trump/suit/follow-suit behavior.
  - Exports `getLedSuitBase`, `suitsWithTrumpBase`, `canFollowBase`, `rankInTrickBase`, `isTrumpBase`.

### Layer System
- `src/game/layers/base.ts`
  - Thin wrapper that delegates to rules-base for all suit/trump semantics.
- `src/game/layers/compose.ts`
  - Composition only. No base logic. It calls rules-base for initial values.
- `src/game/layers/nello.ts`, `src/game/layers/sevens.ts`
  - Override only where behavior differs. No duplicated algorithm structure.

### Core Utilities (Pure, Rule-Agnostic)
- `src/game/core/dominoes.ts`
  - Pure pip math and deck utilities only. No trump logic.
  - Keep: deck creation, shuffle/deal, `dominoHasSuit`, `getNonSuitPip`, points helpers.
  - Remove: `getLedSuit`, `isTrump`, `getDominoValue`.

- `src/game/core/scoring.ts`
  - Points and marks only. No trick winner logic.
  - Remove `calculateTrickWinner` entirely.

### Server-Only Rule Derivations
- `src/kernel/kernel.ts`
  - Builds the server-side view projection using `ExecutionContext.rules` and filtered state.
  - Sends serialized projection to clients.
- `src/game/view-projection.ts`
  - Either moved server-side or reduced to a pure formatting helper that consumes derived data.

## Roll-Forward Plan (Clean Break)

### 1) Create the Rules Base Module
- Add `src/game/layers/rules-base.ts` and move all base suit/trump/follow-suit logic there.
- Ensure base behaviors are expressed once, in one file.

### 2) Rewire Composition
- Update `base.ts` to call the rules-base helpers for all suit/trump methods.
- Update `compose.ts` so its base values come from rules-base, not inline logic.

### 3) Remove All Rule Logic From Core Helpers
- Delete `getLedSuit`, `isTrump`, `getDominoValue` from `core/dominoes.ts`.
- Delete `calculateTrickWinner` from `core/scoring.ts`.
- Update all call sites to use `GameRules` equivalents.

### 4) Make `GameRules` Complete
- Add `isTrump(state, domino)` to the `GameRules` interface.
- Implement in base layer and allow overrides where needed.

### 5) Force All Rule Consumers Through `GameRules`
- Engine: Only `rules.*` used for trick winner, led suit, follow-suit validation.
- AI: Replace any core helper usage with `rules.*` equivalents.
- Utilities: Remove all local `isTrump` and led-suit computations.

### 6) Server-Side View Projection Only
- Build the UI projection in `buildKernelView` using `ExecutionContext.rules` + filtered state.
- Client no longer imports or calls any rule-aware helpers.
- Any UI that needs playability, trump labels, or led-suit info consumes server-provided fields.

### 7) Delete UI Rule Logic (Perfects Feature)
- Remove the Perfects feature and all UI rule logic it contains.
- Known current files to remove:
  - `src/PerfectsApp.svelte`
  - `src/lib/components/PerfectHandDisplay.svelte`
  - `src/lib/utils/dominoHelpers.ts`
  - `src/lib/utils/domino-sort.ts`
  - `scripts/find-perfect-hands.ts`
  - `scripts/find-perfect-partition.ts`
  - `scripts/find-3hand-leftover.ts`
  - `docs/PERFECTS.md`
- If Perfects must exist, it lives as a server-side tool or a separate repo, not in UI.

### 8) Remove `suitAnalysis` From GameState
- Delete `suitAnalysis` from `Player` and all derived state.
- Compute suit analysis on demand in server/AI where needed.
- This eliminates stale cache risk and shrinks serialized state.

### 9) Enforce the Crystal Palace Contract With Tests
- Rule contract tests for base + special contracts across all canonical methods.
- A "no bypass" test that fails if UI/AI import rule logic from `core/dominoes.ts` or `core/scoring.ts`.
- Projection tests to verify no hidden state leaks.

## Definition of Done
- Exactly one base implementation of trump/suit/follow-suit semantics.
- No rule logic in `core/dominoes.ts` or `core/scoring.ts`.
- Every rule-aware decision uses `ExecutionContext.rules`.
- No UI rule logic exists in the repo (Perfects removed or moved).
- `suitAnalysis` is not stored on state.
- Tests enforce no bypass and layer correctness.

## Guiding Principle
If the system needs to answer a question about trump, suits, following, or trick winners, the answer must come from `GameRules` or it is wrong.
