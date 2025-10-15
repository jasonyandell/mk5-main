# Variant Composition Refactor - Implementation Handoff

**Date:** 2025-01-15
**Status:** Ready to Begin Implementation
**Context Window:** Fresh start needed

---

## 🎯 Mission

Replace imperative variant system (418 lines) with pure functional composition + event sourcing.

---

## 📚 Required Reading (In Order)

### 1. **VARIANT_COMPOSITION_CONTEXT.md** (Read FIRST)
Your grounding document. Explains:
- What exists (WRONG patterns - don't copy)
- What we're building (CORRECT patterns - follow these)
- Common pitfalls and how to avoid them

**Critical sections:**
- Architecture Comparison (visual side-by-side)
- One-Hand Clarifications (what it IS and ISN'T)
- File-by-File Guidance (create/update/delete/ignore)
- Success Signals (good signs vs red flags)

### 2. **VARIANT_COMPOSITION_REFACTOR.md** (Implementation Plan)
Complete step-by-step plan with:
- 5 implementation phases
- Code examples for every variant
- Migration strategy for 49 files
- Success criteria

### 3. **docs/remixed-855ccfd5.md** (Vision - Reference Only)
High-level architectural vision. Don't implement everything - just what's in scope.

---

## 🧭 Mental Model (Memorize This)

```typescript
// The ONE fundamental truth
GameState = replayActions(initialConfig, actionHistory)

// Variants are function transformers (optional)
type Variant = {
  transformInitialState?: (base) => (config) => GameState
  transformGetValidActions?: (base) => (state) => GameAction[]
  transformExecuteAction?: (base) => (state, action) => GameState
}

// Composition
let f = base
for (variant of variants) {
  if (variant.transform) f = variant.transform(f)
}
```

**Key insights:**
1. Event sourcing is foundation (not bolt-on)
2. Variants are pure function wrappers (not hooks)
3. Base functions maximally permissive (variants filter)
4. Optional transforms (not required hooks)

---

## ✅ In Scope

1. **Variant system** - Pure functional composition
2. **Event sourcing** - `replayActions(config, actions)`
3. **Three variants:**
   - Tournament: Filter special bids (~10 lines)
   - One-hand: Hardcoded bid, skip to playing, end after one (~40 lines)
   - Speed-mode: Annotate single plays (~15 lines)
4. **Delete old code** - VariantRegistry.ts (418 lines)
5. **Remove tournamentMode** - From GameState (affects 49 files)
6. **Update tests** - Use composition + event sourcing patterns

---

## ❌ Out of Scope (Don't Touch)

1. **Seed finder** - Deferred entirely (future enhancement)
2. **E2E tests** - Broken, fix separately
3. **URL encoding** - Exists but misaligned, fix separately
4. **Capability system** - Future feature
5. **AI integration** - One-hand uses hardcoded bid (no AI yet)

---

## 🗺️ Implementation Roadmap

### Phase 1: Foundation (No Breaking Changes)
**Goal:** Build new system alongside old

**Steps:**
1. Create `src/game/variants/` directory structure
2. Implement 5 new files:
   - `types.ts` - Type definitions
   - `registry.ts` - Variant lookup (pure)
   - `tournament.ts` - Filter special bids
   - `oneHand.ts` - Skip bidding, end after one
   - `speedMode.ts` - Annotate single plays
3. Create `src/game/core/replay.ts` - Event sourcing utilities
4. Update `GameConfig` in protocol to add `variants` array

**Verify:** TypeScript compiles, no breaking changes yet

### Phase 2: Integration (Wire Into GameHost)
**Goal:** Make GameHost use new system

**Steps:**
1. Update GameHost constructor:
   - Compose variants
   - Store composed functions
   - Use `replayActions` for initial state
2. Update `getView()` - use composed `getValidActions`
3. Update `executeAction()` - remove all VariantRegistry calls
4. Delete private variant methods from GameHost

**Verify:** GameHost works with new variant system

### Phase 3: Destruction (Big Bang)
**Goal:** Delete old system completely

**Steps:**
1. Delete `src/server/game/VariantRegistry.ts` (418 lines)
2. Remove `tournamentMode` from GameState type (src/game/types.ts)
3. Remove `gameTarget` from GameState
4. Fix `src/game/core/rules.ts` - remove tournamentMode params
5. Fix `src/game/core/state.ts` - remove tournamentMode option
6. Fix all 49 files referencing tournamentMode

**Strategy:** Big bang commit. Fix all compilation errors at once.

**Verify:** TypeScript compiles (after fixing all 49 files)

### Phase 4: Test Migration
**Goal:** Update tests to use new patterns

**Steps:**
1. Create test helper utilities
2. Update tests to use `replayActions`
3. Update tests to compose variants manually
4. Add new variant composition tests

**Verify:** All unit tests pass (`npm test`)

### Phase 5: Store Integration
**Goal:** Update Svelte stores

**Steps:**
1. Update `gameStore.ts` - use variants array
2. Update `InProcessAdapter` - remove seed finder
3. Update UI components if needed

**Verify:** Game creation works in browser

---

## 🔑 Key Files Reference

### Files to CREATE (new, safe)
```
src/game/variants/
  ├── types.ts           # Type definitions
  ├── registry.ts        # Variant lookup
  ├── tournament.ts      # ~10 lines
  ├── oneHand.ts         # ~40 lines
  └── speedMode.ts       # ~15 lines

src/game/core/
  └── replay.ts          # Event sourcing utilities
```

### Files to UPDATE (modify carefully)
```
src/game/types.ts                    # Remove tournamentMode
src/shared/multiplayer/protocol.ts  # Add variants array
src/server/game/GameHost.ts         # Compose variants
src/game/core/rules.ts               # Remove tournamentMode params
src/game/core/state.ts               # Remove tournamentMode option
src/stores/gameStore.ts              # Use variants array
```

### Files to DELETE (remove completely)
```
src/server/game/VariantRegistry.ts  # 418 lines - DELETE ENTIRELY
```

### Files to IGNORE (leave alone)
```
src/game/core/ai-scheduler.ts       # Future use
src/game/core/gameSimulator.ts      # Future use
src/game/core/seedFinder.ts         # Future migration
src/game/core/url-compression.ts    # Future migration
src/tests/e2e/**                     # Separate effort
```

---

## 🎨 Code Examples (Copy These Patterns)

### Tournament Variant (Complete)
```typescript
// src/game/variants/tournament.ts
import type { VariantFactory } from './types';

export const tournamentVariant: VariantFactory = () => ({
  type: 'tournament',
  transformGetValidActions: (base) => (state) => {
    const actions = base(state);
    return actions.filter(action =>
      action.type !== 'bid' ||
      !['nello', 'splash', 'plunge'].includes(action.bid)
    );
  }
});
```

### One-Hand Variant (Complete)
```typescript
// src/game/variants/oneHand.ts
import type { VariantFactory } from './types';
import { executeAction } from '../core/actions';

export const oneHandVariant: VariantFactory = () => ({
  type: 'one-hand',

  transformInitialState: (base) => (config) => {
    let state = base(config);

    // Hardcoded: Player 3 bids 30, suit 4 trump
    state = executeAction(state, { type: 'bid', player: 3, bid: 'points', value: 30 });
    state = executeAction(state, { type: 'pass', player: 0 });
    state = executeAction(state, { type: 'pass', player: 1 });
    state = executeAction(state, { type: 'pass', player: 2 });
    state = executeAction(state, { type: 'select-trump', player: 3, trump: { type: 'suit', suit: 4 } });

    return state;  // Now at 'playing' phase
  },

  transformExecuteAction: (base) => (state, action) => {
    const newState = base(state, action);
    if (action.type === 'score-hand' && newState.phase === 'bidding') {
      return { ...newState, phase: 'game_end' };
    }
    return newState;
  }
});
```

### Event Sourcing Utility (Complete)
```typescript
// src/game/core/replay.ts
import type { GameState, GameAction, GameConfig } from '../types';
import { createInitialState } from './state';
import { executeAction as baseExecuteAction } from './actions';
import { getVariant } from '../variants/registry';

export function replayActions(
  config: GameConfig,
  actions: GameAction[]
): GameState {
  let createState = createInitialState;
  let executeAction = baseExecuteAction;

  for (const variantConfig of config.variants || []) {
    const variant = getVariant(variantConfig.type, variantConfig.config);
    if (variant.transformInitialState) {
      createState = variant.transformInitialState(createState);
    }
    if (variant.transformExecuteAction) {
      executeAction = variant.transformExecuteAction(executeAction);
    }
  }

  let state = createState(config);
  for (const action of actions) {
    state = executeAction(state, action);
  }
  return state;
}
```

---

## 🚦 Success Checklist

### Phase 1 Complete When:
- [ ] `src/game/variants/` directory exists with 5 files
- [ ] `src/game/core/replay.ts` exists
- [ ] TypeScript compiles with no errors
- [ ] No breaking changes to existing code

### Phase 2 Complete When:
- [ ] GameHost composes variants at construction
- [ ] GameHost uses composed functions
- [ ] All VariantRegistry calls removed from GameHost
- [ ] Game creation works

### Phase 3 Complete When:
- [ ] VariantRegistry.ts deleted (418 lines)
- [ ] tournamentMode removed from GameState
- [ ] All 49 files fixed
- [ ] TypeScript compiles

### Phase 4 Complete When:
- [ ] Test helpers created
- [ ] Tests use replayActions pattern
- [ ] Tests compose variants manually
- [ ] All unit tests pass (`npm test`)

### Phase 5 Complete When:
- [ ] gameStore.ts updated
- [ ] InProcessAdapter updated
- [ ] Game creation works in browser

### Final Success When:
- [ ] `npm run typecheck` passes
- [ ] `npm test` passes
- [ ] Tournament mode works (no special bids)
- [ ] One-hand mode works (starts at playing, ends after one)
- [ ] Can create state via `replayActions(config, actions)`

---

## ⚠️ Red Flags (Stop If You See These)

1. **Importing VariantRegistry** in new code → Reference context doc
2. **Adding variant flags to GameState** → Use composition instead
3. **Core engine checking variants** → Base should be permissive
4. **Mutating state in variants** → Return new objects
5. **Trying to integrate seed finder** → Out of scope
6. **Fixing E2E tests** → Out of scope
7. **Implementing URL encoding** → Out of scope

---

## 🆘 When You're Stuck

### "I don't know if I should modify this file"
→ Check "File-by-File Guidance" in VARIANT_COMPOSITION_CONTEXT.md

### "I don't understand the architecture"
→ Read "Architecture Comparison" section in VARIANT_COMPOSITION_CONTEXT.md

### "I'm confused about one-hand"
→ Read "One-Hand Variant: Critical Clarifications" in VARIANT_COMPOSITION_CONTEXT.md

### "Should I integrate seed finder?"
→ NO. Read "Scope Boundaries" in VARIANT_COMPOSITION_CONTEXT.md

### "Is this pattern correct?"
→ Compare against "Code Examples" in VARIANT_COMPOSITION_REFACTOR.md

### "The current code does X, should I copy that?"
→ NO. Current code is WRONG. Read "What Currently Exists (DO NOT COPY)" in VARIANT_COMPOSITION_CONTEXT.md

---

## 🎬 Starting Point

1. **Read VARIANT_COMPOSITION_CONTEXT.md** (15 min)
2. **Skim VARIANT_COMPOSITION_REFACTOR.md** (10 min)
3. **Start Phase 1, Step 1:** Create directory structure
   ```bash
   mkdir -p src/game/variants
   touch src/game/variants/types.ts
   ```
4. **Copy code from "Code Examples" section** above
5. **Verify TypeScript compiles:** `npm run typecheck`
6. **Proceed to next step**

---

## 📊 Progress Tracking

Mark phases as you complete them:

- [ ] **Phase 1:** Foundation (new files, no breaking changes)
- [ ] **Phase 2:** Integration (wire into GameHost)
- [ ] **Phase 3:** Destruction (delete old system, fix 49 files)
- [ ] **Phase 4:** Test Migration (update all tests)
- [ ] **Phase 5:** Store Integration (Svelte stores)

**Current Phase:** Phase 1 (not started)

---

## 💡 Remember

- **Don't trust current implementation** - it's wrong
- **Event sourcing is core** - not a bolt-on feature
- **Variants are wrappers** - pure functions, not hooks
- **Base is permissive** - generates ALL actions, variants filter
- **Stay in scope** - no seed finder, no E2E fixes, no URL encoding

**You have all the information you need in the three documents. Reference them constantly. Good luck!** 🚀
