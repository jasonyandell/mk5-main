# Variant Composition Refactor v2

**Status:** Draft for alignment  
**Audience:** Gameplay, platform, and UX collaborators  
**Intent:** Reaffirm the original functional vision while simplifying the migration path

---

## 1. Executive Summary

- **Goal:** Replace the imperative variant system with a single, composable state-machine transformer pipeline that matches the architecture vision in `docs/remixed-855ccfd5.md`.
- **Strategy:** Treat every variant effect as an action-stream transform. Variants decorate `getValidActions`, optionally emit scripted follow-up actions, and rely on replay to surface game state. No additional hooks over initial-state or executor functions.
- **Outcomes:** 
  - Event sourcing remains the source of truth (`state = replay(initialConfig, actionHistory)`).
  - Capability → authorization → visibility pipeline stays baked into the compositional story.
  - Designers and tests can stand up any transition by providing `initialConfig` + scripted actions.

---

## 2. Core Principles

1. **Single Variant Surface**
   - `Variant = (StateMachine) => StateMachine`
   - State machine = `getValidActions(state): GameAction[]` with metadata.
   - Variants never mutate `GameState` or bypass replay; they only transform the action stream.

2. **Event Sourcing First**
   - `GameState.initialConfig` + `GameState.actionHistory` define the entire state.
   - Replay uses the same composed variant pipeline to rebuild derived fields.
   - Special behaviors (e.g., one hand endgame) show up as actual actions in history.

3. **Capability Pipeline**
   - Variants add metadata or scripted actions.
   - Authorization enforces executable actions (`canExecute`).
   - Visibility removes metadata the viewer lacks capability to see.
   - Every layer is a pure transform, mirroring Section 4.5 of the original vision.

4. **Scripted Actions Instead of Hooks**
   - Fast-forward or cleanup flows emit deterministic `autoExecute` actions.
   - Executor remains generic—no variant awareness or branching.
   - Replay naturally replays those scripted steps.

### 2.1 Why a Single Transformer Surface?

| Concern | Three-hook system | Single transformer | Outcome |
| --- | --- | --- | --- |
| Variant authoring | Decide which of three functions to wrap, risk partial coverage | Always wrap the action state machine | Lower cognitive load once learned |
| Composition conflicts | Different variants may override different hooks, hard to reason about order | One composition chain (`f(g(h(base)))`) | Deterministic layering |
| Core engine purity | Hooks tempt shortcut mutations (e.g., mutate initial state) | Engine exposed only via `getValidActions`/replay | Event sourcing untouched |
| Debuggability | Bugs hide in custom executors/initializers | Everything visible as actions in history | Easier to inspect and replay |

The cost: variants must encode “stateful” flows (like one-hand setup) by inspecting history and emitting scripted actions. We mitigate that by documenting the contract (Section 3.4) and supplying helper utilities during implementation.

---

## 3. Variant System Design

### 3.1 Core Types

```typescript
type StateMachine = (state: GameState) => GameAction[]

type Variant = (base: StateMachine) => StateMachine

interface VariantConfig {
  type: string
  params?: Record<string, unknown>
}
```

### 3.2 Composition

```typescript
const applyVariants = (base: StateMachine, variants: VariantConfig[]): StateMachine => {
  return variants
    .map(resolveVariant) // Registry lookup, pure
    .reduce((machine, variant) => variant(machine), base)
}
```

### 3.3 Patterns

- **Tournament:** Filter bids (`nello`, `splash`, `plunge`) out of the action list.
- **Forced Bid Minimum:** Filter early `pass` or low bids until threshold satisfied.
- **One-Hand (reframed):**
  - Variant injects scripted bidding/trump actions with `autoExecute: true`.
  - Replay processes them, landing in playing phase without custom executor logic.
  - End-of-hand logic emits a scripted `{ type: 'end-game' }` follow-up when appropriate.
- **Speed Mode:** Annotate single-option plays with `{ autoExecute: true, delay: 300 }`.
- **Daily Challenge:** Replace `score-hand` with `{ type: 'end-game', stars, shareText }` so the client can present completion UI.

### 3.4 Scripted Action Contract

- Use `action.autoExecute === true` to signal the client/runner to immediately enqueue the action after authorization.
- Scripted actions must be deterministic functions of `(state, history, params)` to guarantee replay.
- Keep metadata on the action itself so capability filtering can decide who sees it (`showHints`, `see-ai-intent` etc.).
- Variants should tag scripted actions (e.g. `meta.scriptId`, `meta.step`) so debugging tools can report which scripts fired and tests can assert ordering.
- Provide helper utilities during implementation:
  - `withScriptedSequence(scriptId, steps)` – yields a higher-order variant wrapper that handles history counting.
  - `isScriptComplete(scriptId, history)` – guard to avoid re-running scripts during replay.

---

## 4. Capability & Visibility Integration

1. **Variants Annotate Actions**
   - Add fields like `hint`, `aiIntent`, `autoExecute`, `shareText`.
   - Never remove base fields; only decorate or replace.

2. **Authorization Filters**
   - `getExecutableActions(playerCapabilities, actions)` returns only truthy actions for that player, preserving metadata.

3. **Visibility Filters**
   - `getVisibleActions(playerCapabilities, actions)` strips metadata the player cannot see.
   - Extend existing capability definitions (`see-hints`, `observe-all-hands`, etc.).

4. **State Exposure**
   - Multiplayer state snapshots include the action list already tailored per player.
   - Spectators, coaches, and AI share the same pipeline, differing only by capabilities.

---

## 5. Event Sourcing & Replay

- **State Shape:**
  ```typescript
  interface GameState {
    initialConfig: GameConfig
    actionHistory: GameAction[]
    derived: DerivedSnapshot // recomputed via replay
  }
  ```
- **Replay Algorithm:**
  1. Resolves variants from `initialConfig`.
  2. Composes state machine once.
  3. Reduces over `actionHistory` to derive the current snapshot.
- **Testing & Tooling:**
  - Unit tests feed scripted histories to hit each state transition.
  - Scenario loader (for designers) encodes seed + action list, replays to freeze UI at any step.
  - Undo/redo/time-travel become linear operations on `actionHistory`.

---

## 6. Implementation Roadmap

### Phase 1 – Foundation
- Extract new variant directory (`types.ts`, `registry.ts`, etc.).
- Implement tournament, forced bid minimum, speed mode as pure state-machine transformers.
- Rework one-hand variant to emit scripted actions rather than transform state.
- Update `GameConfig` to store `variantConfigs: VariantConfig[]`.
- Build a pure `applyVariants` helper and wrap `baseGetValidActions`.

### Phase 2 – Event Sourcing Integration
- Introduce `replayActions(config, actions)` utility that composes variants and reduces history.
- Refactor game stores/controllers to create state by replay rather than in-place mutation.
- Remove legacy `transformInitialState`/`transformExecuteAction` hooks.
- Ensure executor simply appends the chosen action (including autoExecute ones) and recomputes via replay if needed for derived data.

### Phase 3 – Capability Pipeline
- Connect variant-emitted metadata to capability-aware filters.
- Ensure multiplayer server/client uses `getExecutableActions` and `getVisibleActions` prior to delivery.
- Add coverage for spectator/coach/tutorial capability sets matching the original architecture doc.

### Phase 4 – Tooling & Tests
- Add unit tests for each variant’s composed state machine (base vs variant vs composed).
- Provide fixture helpers to produce scripted game histories for scenario coverage.
- Update E2E tests to rely on serialized history rather than ad-hoc state patches.
- Optional: expose CLI/UI harness for designers to load scenario JSON and inspect states.

---

## 7. Success Criteria

- Base engine contains no variant-specific flags or branches.
- Variants exist only as state-machine transformers; no custom initial-state/executor code remains.
- Replay from `initialConfig` + `actionHistory` deterministically reproduces gameplay (including variant effects).
- Capability filters strip or expose metadata exactly as defined in the vision doc.
- Tests and tools can recreate any transition via scripted action histories.

---

## 8. Notes & Follow-Ups

- Designer scenario scripts can rely on existing replay fixtures; no shared schema required right now.
- Auto-executed actions surface through the normal client subscription flow—UI components react to emitted actions rather than bespoke hooks.
- Replay performance currently needs no extra caching beyond the base reducer; revisit only if telemetry signals pressure.

---

## 9. Operational Guidance

- **Host auto-execute loop:** `GameHost` inspects the composed action list. While the head action has `autoExecute: true`, it dispatches it immediately through `executeAction`, verifying the script terminates. Guard against runaway scripts by enforcing a max auto-exec count per view refresh; log the script ID if exceeded.
- **Timer coordination:** When scheduling delayed system actions (AI thinking, shot clock), record the current `actionHistory.length`. On callback, re-check the history length; drop the timer if state advanced. This avoids nearly all race conditions without introducing locks.
- **Testing strategy:** 
  - Unit tests cover individual variants by feeding mocked base actions and asserting output.
  - Host tests simulate scripted sequences and ensure the auto-exec loop drains the script and exposes the expected final action list.
  - Replay tests serialize `initialConfig + actionHistory` and rebuild identical state snapshots.
- **Onboarding checklist for new variants:** 
  1. Identify which phases the variant touches.
  2. Decide whether it filters, annotates, or scripts actions.
  3. Use the helper utilities for scripts to avoid ad-hoc history parsing.
  4. Add capability gating metadata if visibility differs per player.
  5. Add tests asserting both base and variant behavior via composed state machine.

---

---

**Next Actions:** Socialize the streamlined plan, confirm variant inventory, and schedule Phase 1 implementation. Once agreed, retire the older multi-hook plan in `VARIANT_COMPOSITION_REFACTOR.md`.  
**Reference:** Aligns directly with Sections 5 and 4.5 of `docs/remixed-855ccfd5.md`.
