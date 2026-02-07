# ADR: One-Hand Mode Terminal Phase via GameRules Pattern

**Date**: 2025-01-12
**Status**: Implemented
**Decision Makers**: Architecture Team

---

## Context

One-hand mode is a game variant where only a single hand is played, primarily used for:
- Quick practice/training
- Hand replay and analysis
- Challenge sharing via URLs

The game needs to **end after the first hand** instead of dealing a new hand and continuing to next round.

### Initial Problem

When implementing one-hand mode, we faced a phase transition challenge:
- Normal games: After scoring, transition to `'bidding'` phase and deal new hand
- One-hand games: After scoring, should transition to terminal state and show completion modal

The naive approach would be to add conditionals in the executor:

```typescript
// ❌ ANTI-PATTERN - Violates architectural invariants
if (state.config.oneHandMode) {
  return { ...state, phase: 'one-hand-complete' };
} else {
  return { ...state, phase: 'bidding' };
}
```

This violates **Parametric Polymorphism** - executors should delegate to rules, never inspect state or config.

---

## Decision

We use the **GameRules Pattern** to handle mode-specific phase transitions:

### 1. Add New GameRules Method

**Added**: `getPhaseAfterHandComplete(state: GameState): GamePhase`

This rule method determines which phase to transition to after hand scoring completes.

**Base Implementation** (`src/game/layers/base.ts`):
```typescript
getPhaseAfterHandComplete: (_state: GameState) => {
  return 'bidding';  // Continue to next hand (standard behavior)
}
```

**OneHand Override** (`src/game/layers/oneHand.ts`):
```typescript
getPhaseAfterHandComplete: (_state: GameState) => {
  return 'one-hand-complete';  // Terminal state (don't deal new hand)
}
```

### 2. Update Executor to Delegate

**Modified**: `executeScoreHand` in `src/game/core/actions.ts`:

```typescript
// BEFORE: Hardcoded phase
phase: 'bidding',

// AFTER: Delegate to rule
const nextPhase = rules.getPhaseAfterHandComplete(state);
phase: nextPhase,
```

### 3. Create Terminal Phase

**Added**: `'one-hand-complete'` to `GamePhase` type

This is a proper terminal state that:
- Generates no actions (empty array from `generateStructuralActions`)
- Triggers client modal display
- Persists in URL state for replay

### 4. Two-Layer Composition

One-hand mode requires coordination between two layers:

**ActionTransformer Layer** (`oneHandActionTransformer`):
- Automates bidding/trump selection
- Auto-executes score-hand action
- **Responsibility**: Action automation and UX

**RuleSet Layer** (`oneHandRuleSet`):
- Overrides phase transition logic
- Returns terminal phase instead of bidding
- **Responsibility**: Execution semantics

These layers compose independently but work together for one-hand mode.

---

## Alternatives Considered

### Alt 1: ActionTransformer Replaces score-hand Action

**Approach**: Have ActionTransformer replace `score-hand` with `end-game` action.

**Rejected because**:
- ActionTransformers shouldn't control phase transitions
- Violates separation of concerns (automation vs execution)
- Creates coupling between layers

### Alt 2: Client-Side Detection

**Approach**: Client checks config and shows modal based on that.

**Rejected because**:
- Violates server authority principle
- Client shouldn't inspect config for behavior decisions
- Makes client "smarter" than it should be
- Breaks with URL replay (config may not be available)

### Alt 3: Add `oneHandMode` Flag to GameState

**Approach**: Store boolean flag in state, check in executor.

**Rejected because**:
- Violates Parametric Polymorphism
- Adds mode-specific checks to executor
- Doesn't scale (every mode needs its own flag and checks)
- Creates coupling between executor and modes

---

## Consequences

### ✅ Benefits

1. **Zero Coupling**: Executor has no knowledge of one-hand mode
2. **Parametric Polymorphism**: Executor delegates to `rules.getPhaseAfterHandComplete()`, never inspects state
3. **Server Authority**: Server determines phase, client just renders
4. **Composable**: OneHand RuleSet composes cleanly with other rulesets
5. **Extensible**: Pattern works for any mode needing custom phase transitions
6. **Type-Safe**: All phase transitions type-checked
7. **Testable**: Can test oneHandRuleSet in isolation

### ⚠️ Trade-offs

1. **Two-Layer Coordination**: OneHand requires both ActionTransformer AND RuleSet
   - **Mitigation**: Documented in code comments, validated at config time

2. **Additional GameRules Method**: Adds to interface surface area
   - **Acceptable**: Method is general-purpose (any mode can use it)
   - **Follows Pattern**: Same as existing rules like `checkHandOutcome`

3. **Terminal Phase Addition**: Adds new phase to type
   - **Acceptable**: Clean extension of phase state machine
   - **Properly Handled**: Action generation, UI, serialization all updated

### 📋 Implementation Checklist

- [x] Add `getPhaseAfterHandComplete` to `GameRules` interface
- [x] Implement in `baseRuleSet` (returns `'bidding'`)
- [x] Create `oneHandRuleSet` (returns `'one-hand-complete'`)
- [x] Add composition in `composeRules`
- [x] Update `executeScoreHand` to use rule method
- [x] Add `'one-hand-complete'` to `GamePhase` type
- [x] Handle in `generateStructuralActions` (return empty array)
- [x] Update client modal detection
- [x] Update UI components (GameInfoBar, PlayingArea)
- [x] Add to ruleset registry
- [x] Document separation of concerns

---

## Related Decisions

- **ADR-20251110-single-composition-point.md**: Enforces composition only in Room/HeadlessRoom
- **ORIENTATION.md lines 142-161**: Documents when to add new GameRules methods
- **ORIENTATION.md lines 527-612**: Step-by-step guide for adding mode-specific behavior

---

## Examples

### Normal Game Flow
```
scoring → getPhaseAfterHandComplete() → 'bidding' → deal new hand
```

### One-Hand Game Flow
```
scoring → getPhaseAfterHandComplete() → 'one-hand-complete' → show modal
```

### Code Pattern
```typescript
// Executor (mode-agnostic)
const nextPhase = rules.getPhaseAfterHandComplete(state);

// Base RuleSet
getPhaseAfterHandComplete: () => 'bidding'

// OneHand RuleSet (overrides)
getPhaseAfterHandComplete: () => 'one-hand-complete'
```

---

## Validation

This decision upholds all **Architectural Invariants**:

1. ✅ **Pure State Storage**: Room stores unfiltered state
2. ✅ **Server Authority**: Client trusts server's phase
3. ✅ **Capability-Based Access**: No identity checks
4. ✅ **Single Composition Point**: Composed in Room/HeadlessRoom only
5. ✅ **Zero Coupling**: Executor has no mode knowledge
6. ✅ **Parametric Polymorphism**: Executor calls `rules.method()`
7. ✅ **Event Sourcing**: Phase derivable from actions
8. ✅ **Clean Separation**: RuleSet handles logic, ActionTransformer handles automation

---

## Future Considerations

This pattern is reusable for other modes needing custom phase transitions:

- **Tournament modes**: Could have custom end conditions
- **Training modes**: Could pause at specific phases
- **Replay modes**: Could have read-only phases

The `getPhaseAfterHandComplete` method is general-purpose and can be overridden by any RuleSet.

---

**Conclusion**: This decision demonstrates textbook application of the GameRules pattern, maintaining zero coupling while enabling mode-specific behavior through composition.
