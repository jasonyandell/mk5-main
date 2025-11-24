# ADR: Unified Layer Architecture - Eliminating ActionTransformer Redundancy

**Date**: 2025-01-24
**Status**: Implemented
**Decision Makers**: Architecture Team

---

## Context

The codebase evolved with two parallel composition systems:

1. **RuleSets** - Composable game execution rules (WHO/WHEN/HOW/VALIDATION/SCORING)
2. **ActionTransformers** - Composable action filtering/annotation pipelines (UX/hints/automation)

Both systems solved the same architectural problem - **composable behavior without executor coupling** - but with redundant infrastructure.

### The Redundancy

**RuleSets had:**
- `Layer` interface with `rules` property
- Composition via `composeRules()`
- Registry pattern for lookup
- Applied in `ExecutionContext`

**ActionTransformers had:**
- `ActionTransformer` interface with factory functions
- Composition via function wrapping `(StateMachine) => StateMachine`
- Registry pattern for lookup
- Applied in `ExecutionContext`

Both lived in separate directories, had parallel test structures, and required duplicate maintenance.

### The Insight

Looking at actual use cases revealed the split was artificial:

- **oneHand**: Needed BOTH phase transition rule (`getPhaseAfterHandComplete`) AND action automation (`getValidActions`)
- **speed**: Only needed action transformation (auto-execute forced moves)
- **hints**: Only needed action transformation (annotate with educational text)
- **nello**: Only needed execution rules (trick completion, winner calculation)

The split implementation for oneHand was a code smell - it had:
1. `src/game/layers/oneHand.ts` - Phase transition rule
2. `src/game/action-transformers/oneHand.ts` - Action automation

**Why were these separate?** They're both part of the "oneHand" feature!

---

## Decision

**Unify RuleSets and ActionTransformers into a single `Layer` concept with two orthogonal composition surfaces:**

### The Unified Layer Interface

```typescript
interface Layer {
  name: string;

  // Surface 1: Action generation (what's possible)
  getValidActions?: (state: GameState, prev: GameAction[]) => GameAction[];

  // Surface 2: Execution rules (how things execute)
  rules?: {
    getTrumpSelector?: (state: GameState, bid: Bid, prev: number) => number;
    getFirstLeader?: (state: GameState, selector: number, trump: TrumpSelection, prev: number) => number;
    // ... 11 other rule methods
  };
}
```

**Key properties:**
- **Single composition point**: One registry, one lookup, one composition function
- **Orthogonal surfaces**: Layers can implement 0, 1, or 2 surfaces as needed
- **Uniform patterns**: Same reduce-based composition for both surfaces
- **Zero redundancy**: No parallel infrastructure

### Migration Path

1. **Phase 1-2**: Type system updates (GameRuleSet → Layer)
2. **Phase 3**: Migrate ActionTransformers to Layer.getValidActions
   - Update `oneHandRuleSet` to include automation logic
   - Create `speedRuleSet` and `hintsRuleSet` as Layers
   - Delete `src/game/action-transformers/` directory (6 files)
3. **Phase 4+**: Documentation and cleanup

---

## Implementation

### Before: Split Implementation

```typescript
// execution.ts (BEFORE)
const ruleSets = [baseRuleSet, ...enabledLayers];
const rules = composeRules(ruleSets);

const base = (state) => generateStructuralActions(state, rules);
const withRuleSets = composeActionGenerators(ruleSets, base);
const getValidActions = applyActionTransformers(withRuleSets, actionTransformerConfigs);
```

**THREE composition steps**, two separate systems.

### After: Unified Composition

```typescript
// execution.ts (AFTER)
const actionTransformerLayers = getLayersByNames(actionTransformerConfigs.map(t => t.type));
const ruleSets = [baseRuleSet, ...enabledLayers, ...actionTransformerLayers];
const rules = composeRules(ruleSets);

const base = (state) => generateStructuralActions(state, rules);
const getValidActions = composeActionGenerators(ruleSets, base);
```

**TWO composition steps**, one unified system.

### Layer Examples

**oneHand Layer** (uses BOTH surfaces):
```typescript
export const oneHandRuleSet: Layer = {
  name: 'oneHand',

  getValidActions: (state, prev) => {
    // Automate bidding/trump, auto-execute score-hand
    if (state.phase === 'bidding' && state.bids.length === 0) {
      return [{ type: 'pass', player: state.currentPlayer, autoExecute: true, ... }];
    }
    // ...
    return prev;
  },

  rules: {
    getPhaseAfterHandComplete: () => 'one-hand-complete'  // Terminal phase
  }
};
```

**speed Layer** (uses only getValidActions):
```typescript
export const speedRuleSet: Layer = {
  name: 'speed',

  getValidActions: (state, prev) => {
    // Auto-execute when player has only one legal action
    // ...
  }
};
```

**nello Layer** (uses only rules):
```typescript
export const nelloRuleSet: Layer = {
  name: 'nello',

  rules: {
    isTrickComplete: (state, prev) => state.currentTrick.length === 3,  // 3 plays, not 4
    getNextPlayer: (state, current, prev) => (current + 2) % 4,  // Skip partner
    checkHandOutcome: (state, prev) => {
      // Terminate if bidder wins any trick
      // ...
    }
  }
};
```

---

## Consequences

### Positive

1. **Zero Redundancy**: Single registry, single composition, single test structure
2. **Feature Cohesion**: oneHand is ONE layer, not split across two systems
3. **Simpler Mental Model**: "Layers compose to create game behavior"
4. **Reduced Coupling**: One fewer concept to understand
5. **Easier Extension**: Adding new behaviors requires learning one pattern
6. **Backward Compatible**: Config format unchanged (`actionTransformers: [{ type: 'speed' }]` still works)

### Trade-offs

1. **Larger Interface**: Layer has 15 optional methods vs previous focused interfaces
   - *Mitigation*: TypeScript makes unused methods zero-cost; most layers only use 1-3 methods
2. **Terminology Shift**: "ActionTransformer" → "Layer" requires documentation updates
   - *Mitigation*: Clear migration guide; aliases in registry for backward compatibility
3. **Registry Growth**: 10 entries vs previous 7 (game rules) + 3 (transformers)
   - *Mitigation*: Negligible - it's a Map lookup

### Validation

**Tests passing:**
- ✅ 1,124 unit tests
- ✅ 16 e2e tests
- ✅ TypeScript compilation
- ✅ All linting

**Files changed:**
- Created: `src/game/layers/speed.ts`, `src/game/layers/hints.ts`
- Modified: `src/game/layers/oneHand.ts`, `src/game/layers/registry.ts`, `src/game/layers/types.ts`
- Modified: `src/game/types/execution.ts`, `src/kernel/kernel.ts`
- Deleted: `src/game/action-transformers/` (6 files)
- Updated: Test expectations for 10-entry registry

---

## Alternatives Considered

### Alternative 1: Keep Separate Systems

**Rejected because:**
- Redundant infrastructure maintenance burden
- Split implementations (oneHand) violate feature cohesion
- Duplicate concepts increase cognitive load
- No architectural benefit to separation

### Alternative 2: Merge into ActionTransformers

**Rejected because:**
- ActionTransformer function wrapping `(StateMachine) => StateMachine` less flexible than Layer's reduce pattern
- Rules need access to previous rule results (threading), not full wrapping
- Would require inverting the existing RuleSet architecture (larger refactor)

### Alternative 3: Make ActionTransformers a Layer Property

```typescript
interface Layer {
  rules?: GameRules;
  actionTransformer?: ActionTransformerFactory;
}
```

**Rejected because:**
- Still maintains two composition mechanisms
- Doesn't unify the mental model
- Keeps ActionTransformer infrastructure around

---

## Related Decisions

- **ADR-20251112-onehand-terminal-phase**: Established the GameRules pattern that enabled this unification
- **ADR-20251111-url-replay-complete-config**: Config format preserved for backward compatibility

---

## Notes

This refactor exemplifies the "North Star" principle: **willing to do significant rework to achieve elegance and correctness**. The split system worked, but the unified architecture is:
- Simpler to understand
- Easier to extend
- Eliminates technical debt
- Better represents the actual problem domain

> "Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away." - Antoine de Saint-Exupéry

We removed an entire parallel composition system. The result is stronger for its absence.
