# Pure Functional Composition Migration Plan

## Executive Summary

**Goal:** Migrate variant system from imperative hooks to pure functional composition.

**Why:** Current system uses 418-line `VariantRegistry` with lifecycle hooks, variant-specific state tracking, and baked-in `tournamentMode` flag in `GameState`. This creates coupling, complexity, and bugs.

**Solution:** Three pure functions (`createInitialState`, `getValidActions`, `executeAction`) that variants wrap. Composition is `f(g(h(x)))` all the way down.

**Impact:** Delete 418 lines of imperative code. Remove variant fields from `GameState`. Each variant is ~20 lines of pure transforms.

---

## Current State (What's Broken)

### Bad Pattern 1: Imperative Hooks
```typescript
// VariantRegistry.ts - 418 lines of imperative mess
VariantRegistry.register({
  type: 'one-hand',
  initialize: (state) => { /* mutate */ },
  afterAction: (state, action) => { /* check and mutate */ },
  checkGameEnd: (state) => { /* inspect */ },
  updateVariantState: (state) => { /* track separate state */ }
});
```

**Problems:**
- Not composable (can't chain variants)
- Separate `VariantState` tracking (breaks event sourcing)
- Imperative lifecycle (hard to reason about)
- 418 lines of ceremony

### Bad Pattern 2: Variant Flags in GameState
```typescript
interface GameState {
  tournamentMode: boolean;  // ❌ Variant shouldn't be in core state
  gameTarget: number;       // ❌ Unclear purpose, variant-specific
}

// In gameEngine.ts
if (!state.tournamentMode) {
  // Generate special bids...
}
```

**Problems:**
- Core engine knows about variants (coupling)
- Can't compose multiple variants
- Adding new variant = modify core types

---

## Target Architecture (Pure Composition)

### The Three Pure Functions

```typescript
// Everything composes over these three
type InitialStateFactory = (config: GameConfig) => GameState
type GetValidActions = (state: GameState) => GameAction[]
type ExecuteAction = (state: GameState, action: GameAction) => GameState
```

### Variants are Transform Factories

```typescript
type VariantFactory = (config?: any) => {
  transformInitialState?: (base: InitialStateFactory) => InitialStateFactory
  transformGetValidActions?: (base: GetValidActions) => GetValidActions
  transformExecuteAction?: (base: ExecuteAction) => ExecuteAction
}
```

### Composition Pattern

```typescript
// GameHost.constructor
let createInitialState = baseCreateInitialState;
let getValidActions = baseGetValidActions;
let executeAction = baseExecuteAction;

// Apply variants in order (last one wins)
for (const variantConfig of config.variants || []) {
  const factory = getVariant(variantConfig.type);
  const variant = factory(variantConfig.config);

  if (variant.transformInitialState) {
    createInitialState = variant.transformInitialState(createInitialState);
  }
  if (variant.transformGetValidActions) {
    getValidActions = variant.transformGetValidActions(getValidActions);
  }
  if (variant.transformExecuteAction) {
    executeAction = variant.transformExecuteAction(executeAction);
  }
}

// Store composed functions
this.getValidActions = getValidActions;
this.executeAction = executeAction;

// Use composed factory to create state
this.state = createInitialState(config);
```

**Key insight:** Each variant wraps the previous function. Base knows nothing about variants. Variants know nothing about each other.

---

## Key Principles (North Star)

### 1. Pure Function Composition
```typescript
const game = oneHand(tournament(base))
//           ^^^^^^^ wraps tournament
//                   ^^^^^^^^^^ wraps base
//                              ^^^^ knows nothing about variants
```

### 2. Optional Transforms
A variant specifies ONLY what it needs:
- Tournament: only `transformGetValidActions` (filters bids)
- One-hand: `transformInitialState` (fast-forward) + `transformExecuteAction` (intercept end)

### 3. No Variant State in GameState
Variants derive everything from `state.actionHistory` or existing fields. No `variantState`, no `tournamentMode`, no special fields.

### 4. Last One Wins
```typescript
variants: [
  { type: 'tournament' },   // First: filters special bids
  { type: 'one-hand' }      // Last: if it says "game over", that's final
]
```

### 5. Base Functions Stay Pure
`baseExecuteAction` doesn't check for variants. `baseGetValidActions` generates all actions. Variants wrap and filter.

---

## Implementation Plan

### Step 1: Create Type Definitions

**New File: `src/game/variants/types.ts`**
```typescript
import type { GameState, GameAction } from '../types';

export type InitialStateFactory = (config: GameConfig) => GameState;
export type GetValidActions = (state: GameState) => GameAction[];
export type ExecuteAction = (state: GameState, action: GameAction) => GameState;

export type InitialStateTransform = (base: InitialStateFactory) => InitialStateFactory;
export type ActionValidatorTransform = (base: GetValidActions) => GetValidActions;
export type ActionExecutorTransform = (base: ExecuteAction) => ExecuteAction;

export type VariantFactory = (config?: any) => {
  transformInitialState?: InitialStateTransform;
  transformGetValidActions?: ActionValidatorTransform;
  transformExecuteAction?: ActionExecutorTransform;
};
```

---

### Step 2: Export Base Functions

**New File: `src/game/core/rules.ts`**
```typescript
import { createInitialState } from './state';
import { getValidActions } from './gameEngine';
import { executeAction } from './actions';

export const baseCreateInitialState = createInitialState;
export const baseGetValidActions = getValidActions;
export const baseExecuteAction = executeAction;
```

---

### Step 3: Implement Tournament Variant

**New File: `src/game/variants/tournament.ts`**
```typescript
import type { VariantFactory } from './types';

export const tournamentVariant: VariantFactory = () => ({
  transformGetValidActions: (base) => (state) => {
    const actions = base(state);

    // Filter out special bids (nello, splash, plunge)
    return actions.filter(action =>
      action.type !== 'bid' ||
      !['nello', 'splash', 'plunge'].includes(action.bid)
    );
  }
});
```

**That's it. ~10 lines. No lifecycle hooks. Pure function wrapper.**

---

### Step 4: Implement One-Hand Variant

**New File: `src/game/variants/oneHand.ts`**
```typescript
import type { VariantFactory } from './types';
import { executeAction as pureExecuteAction } from '../core/actions';

export const oneHandVariant: VariantFactory = (config) => {
  const { bidder = 0, bid, trump } = config || {};

  return {
    // Transform 1: Fast-forward to 'playing' phase
    transformInitialState: (base) => (gameConfig) => {
      let state = base(gameConfig);

      // Execute bidding using pure executeAction
      state = pureExecuteAction(state, {
        type: 'bid',
        player: bidder,
        bid: bid.type,
        value: bid.value
      });

      // Other players pass
      for (let i = 0; i < 4; i++) {
        if (i !== bidder) {
          state = pureExecuteAction(state, { type: 'pass', player: i });
        }
      }

      // Execute trump selection
      state = pureExecuteAction(state, {
        type: 'select-trump',
        player: bidder,
        trump
      });

      return state;  // Now at 'playing' phase
    },

    // Transform 2: Intercept score-hand to end game
    transformExecuteAction: (base) => (state, action) => {
      const newState = base(state, action);

      // After scoring, if we're starting a new hand, end instead
      if (action.type === 'score-hand' && newState.phase === 'bidding') {
        return {
          ...newState,
          phase: 'game_end'
        };
      }

      return newState;
    }
  };
};
```

**~40 lines. Two transforms. Base knows nothing about one-hand.**

---

### Step 5: Create Variant Registry

**New File: `src/game/variants/index.ts`**
```typescript
import { tournamentVariant } from './tournament';
import { oneHandVariant } from './oneHand';
import type { VariantFactory } from './types';

const VARIANTS: Record<string, VariantFactory> = {
  'tournament': tournamentVariant,
  'one-hand': oneHandVariant
};

export function getVariant(type: string): VariantFactory {
  const variant = VARIANTS[type];
  if (!variant) {
    throw new Error(`Unknown variant: ${type}`);
  }
  return variant;
}

export { tournamentVariant, oneHandVariant };
export type { VariantFactory } from './types';
```

---

### Step 6: Update GameHost to Compose

**File: `src/server/game/GameHost.ts`**

**Change:**
```typescript
import { baseCreateInitialState, baseGetValidActions, baseExecuteAction } from '../../game/core/rules';
import { getVariant } from '../../game/variants';
import type { InitialStateFactory, GetValidActions, ExecuteAction } from '../../game/variants/types';

export class GameHost {
  private state: GameState;
  private getValidActions: GetValidActions;  // Composed function
  private executeAction: ExecuteAction;      // Composed function

  constructor(gameId: string, config: GameConfig, players: PlayerSession[]) {
    // Start with base functions
    let createInitialState: InitialStateFactory = baseCreateInitialState;
    let getValidActions: GetValidActions = baseGetValidActions;
    let executeAction: ExecuteAction = baseExecuteAction;

    // Compose all variants
    for (const variantConfig of config.variants || []) {
      const variantFactory = getVariant(variantConfig.type);
      const variant = variantFactory(variantConfig.config);

      if (variant.transformInitialState) {
        createInitialState = variant.transformInitialState(createInitialState);
      }
      if (variant.transformGetValidActions) {
        getValidActions = variant.transformGetValidActions(getValidActions);
      }
      if (variant.transformExecuteAction) {
        executeAction = variant.transformExecuteAction(executeAction);
      }
    }

    // Store composed functions
    this.getValidActions = getValidActions;
    this.executeAction = executeAction;

    // Create initial state with composed factory
    this.state = createInitialState(config);
  }

  // Use composed functions
  executeAction(playerId: string, action: GameAction) {
    const result = authorizeAndExecute(this.mpState, { playerId, action });
    // ... rest unchanged
  }

  private createView(forPlayerId?: string): GameView {
    const allValidActions = this.getValidActions(this.state);
    // ... rest unchanged
  }
}
```

---

### Step 7: Clean Up GameState

**File: `src/game/types.ts`**

**Remove:**
```typescript
export interface GameState {
  // DELETE THESE:
  tournamentMode: boolean;  // ❌ Variant, not core state
  gameTarget: number;       // ❌ Variant-specific
  hands?: ...;              // ❌ Legacy test compat
  bidWinner?: ...;          // ❌ Legacy test compat
  isComplete?: ...;         // ❌ Legacy test compat
  winner?: ...;             // ❌ Legacy test compat
}
```

**Optionally Add:**
```typescript
export interface GameState {
  // Optional: for debugging/replay
  variants?: string[];  // e.g., ['tournament', 'one-hand']
}
```

---

### Step 8: Update GameConfig

**File: `src/shared/multiplayer/protocol.ts`**

**Change:**
```typescript
export interface GameConfig {
  playerTypes: ('human' | 'ai')[];
  shuffleSeed?: number;

  // NEW: Variants to compose (in order)
  variants?: Array<{
    type: string;
    config?: any;
  }>;

  // DELETE:
  variant?: GameVariant;  // ❌ Old system
}

// DELETE entire GameVariant interface
```

---

### Step 9: Clean Up Core Functions

**File: `src/game/core/actions.ts`**

In `executeScoreHand`:
```typescript
function executeScoreHand(state: GameState): GameState {
  // ... scoring logic ...

  // Check if game complete (hardcode 7 marks)
  const DEFAULT_MARKS_TO_WIN = 7;
  if (isGameComplete(newMarks, DEFAULT_MARKS_TO_WIN)) {
    return { ...state, phase: 'game_end', teamMarks: newMarks };
  }

  // Start new hand
  // ...
  return { ...state, phase: 'bidding', ... };
}

// DELETE any checks for:
// - state.tournamentMode
// - state.gameTarget
```

**File: `src/game/core/gameEngine.ts`**

In `getBiddingActions`:
```typescript
function getBiddingActions(state: GameState): GameAction[] {
  // ... existing logic ...

  // DELETE this conditional:
  // if (!state.tournamentMode) {
  //   // Add nello, splash, plunge
  // }

  // Always generate ALL bid types
  // Let variants filter them out

  return actions;
}
```

---

### Step 10: Delete Old System

**DELETE ENTIRE FILE:**
- `src/server/game/VariantRegistry.ts` (418 lines)

**SEARCH AND DESTROY:**
- All references to `VariantState`
- All references to old `GameVariant` type (not the new one in protocol)
- All references to `state.tournamentMode`
- All references to `state.gameTarget` (except constant `DEFAULT_MARKS_TO_WIN`)

---

## Illustrative Examples

### Example 1: Standard Game (No Variants)
```typescript
const config = {
  playerTypes: ['human', 'ai', 'ai', 'ai'],
  shuffleSeed: 12345,
  variants: []  // Empty = base game
};

// In GameHost:
// createInitialState = baseCreateInitialState (unchanged)
// getValidActions = baseGetValidActions (unchanged)
// executeAction = baseExecuteAction (unchanged)

// Game starts at 'bidding', plays to 7 marks
```

### Example 2: Tournament Game
```typescript
const config = {
  variants: [{ type: 'tournament' }]
};

// In GameHost:
let getValidActions = baseGetValidActions;

const variant = tournamentVariant();
getValidActions = variant.transformGetValidActions!(getValidActions);

// Now getValidActions filters out nello/splash/plunge
// Everything else unchanged
```

### Example 3: One-Hand Challenge
```typescript
const config = {
  shuffleSeed: 12345,
  variants: [{
    type: 'one-hand',
    config: {
      bidder: 0,
      bid: { type: 'points', value: 32 },
      trump: { type: 'suit', suit: 4 }
    }
  }]
};

// In GameHost:
const variant = oneHandVariant(config.variants[0].config);
createInitialState = variant.transformInitialState!(createInitialState);
executeAction = variant.transformExecuteAction!(executeAction);

// createInitialState returns state at 'playing' phase (fast-forwarded)
// executeAction intercepts score-hand to change 'bidding' → 'game_end'
```

### Example 4: Composed Variants
```typescript
const config = {
  variants: [
    { type: 'tournament' },     // First
    {                            // Last
      type: 'one-hand',
      config: { /* ... */ }
    }
  ]
};

// Composition chain:
let f = baseGetValidActions;
f = tournamentVariant().transformGetValidActions!(f);  // Wraps base
// f now filters special bids

let g = baseExecuteAction;
g = oneHandVariant(config).transformExecuteAction!(g);  // Wraps base
// g now intercepts score-hand

// Result:
// - Starts at 'playing' (one-hand initial state)
// - No special bids (tournament filters)
// - Ends after one hand (one-hand execution)
```

---

## Testing Strategy

### Unit Tests for Variants
```typescript
test('tournament variant filters special bids', () => {
  const variant = tournamentVariant();
  const transform = variant.transformGetValidActions!;

  // Mock base that returns all bid types
  const base = () => [
    { type: 'bid', bid: 'points', value: 30 },
    { type: 'bid', bid: 'nello' },
    { type: 'bid', bid: 'splash' }
  ];

  const wrapped = transform(base);
  const actions = wrapped({} as GameState);

  expect(actions).toHaveLength(1);
  expect(actions[0].bid).toBe('points');
});

test('one-hand variant fast-forwards to playing', () => {
  const variant = oneHandVariant({
    bidder: 0,
    bid: { type: 'points', value: 32 },
    trump: { type: 'suit', suit: 4 }
  });

  const transform = variant.transformInitialState!;
  const wrapped = transform(baseCreateInitialState);

  const state = wrapped({ shuffleSeed: 123, playerTypes: [...] });

  expect(state.phase).toBe('playing');
  expect(state.winningBidder).toBe(0);
  expect(state.trump.type).toBe('suit');
});

test('one-hand variant ends after scoring', () => {
  const variant = oneHandVariant({});
  const transform = variant.transformExecuteAction!;

  // Mock base that transitions to 'bidding'
  const base = () => ({ phase: 'bidding' } as GameState);

  const wrapped = transform(base);
  const newState = wrapped({} as GameState, { type: 'score-hand' });

  expect(newState.phase).toBe('game_end');
});
```

### Integration Tests
```typescript
test('composed tournament + one-hand works', () => {
  const config = {
    playerTypes: ['human', 'ai', 'ai', 'ai'],
    shuffleSeed: 12345,
    variants: [
      { type: 'tournament' },
      { type: 'one-hand', config: { /* ... */ } }
    ]
  };

  const host = new GameHost('test', config, players);
  const view = host.getView();

  // Should start at 'playing'
  expect(view.state.phase).toBe('playing');

  // Should not have special bids
  const bidActions = view.validActions.filter(a => a.action.type === 'bid');
  expect(bidActions.every(a => !['nello', 'splash', 'plunge'].includes(a.action.bid))).toBe(true);

  // Play through hand...
  // After scoring, should be game_end
});
```

### Manual Testing Checklist
- [ ] Standard game plays to 7 marks
- [ ] Tournament game has no special bids
- [ ] One-hand game starts at playing phase
- [ ] One-hand game ends after one hand
- [ ] Tournament + one-hand composition works
- [ ] URL serialization still works (if implemented)

---

## Migration Execution Order

1. **Create new files** (no breaking changes yet):
   - `src/game/variants/types.ts`
   - `src/game/core/rules.ts`
   - `src/game/variants/tournament.ts`
   - `src/game/variants/oneHand.ts`
   - `src/game/variants/index.ts`

2. **Update GameHost** (breaking change):
   - Import base functions and variants
   - Compose in constructor
   - Store composed functions

3. **Update types** (breaking change):
   - Remove fields from GameState
   - Update GameConfig

4. **Clean up core functions** (breaking change):
   - Remove variant checks from actions.ts
   - Remove variant checks from gameEngine.ts

5. **Delete old system**:
   - Delete VariantRegistry.ts
   - Remove all references to old types

6. **Fix TypeScript errors**:
   - Search for `tournamentMode`
   - Search for `gameTarget`
   - Fix all usages

7. **Update tests**:
   - Rewrite tests to use new variant system
   - Add new unit tests for variants

8. **Run and verify**:
   - `npm run typecheck` (must pass)
   - `npm test` (update until all pass)
   - Manual testing

---

## Success Criteria

✅ **Code Quality:**
- TypeScript compiles with no errors
- No references to `tournamentMode`, `gameTarget`, `VariantState`
- All variants are pure function transforms (~20 lines each)
- Old `VariantRegistry.ts` (418 lines) deleted

✅ **Functional Correctness:**
- Standard game: plays to 7 marks normally
- Tournament game: no nello/splash/plunge bids available
- One-hand game: starts at 'playing', ends after one hand
- Composed variants: both effects apply correctly

✅ **Architecture:**
- All composition happens in GameHost constructor
- Base functions (in `core/`) have zero variant awareness
- Variants (in `variants/`) compose over base functions
- No variant-specific state in GameState

✅ **Tests:**
- All existing tests updated and passing
- New unit tests for tournament variant
- New unit tests for one-hand variant
- Integration test for composed variants

---

## Key Insights (Remember These)

### 1. "One-Hand" is NOT About Counting Hands
One-hand doesn't play N hands and stop. It fast-forwards PAST bidding to start at the playing phase, then ends immediately after scoring. It's about **skipping bidding** for new players and **creating shareable challenges**.

### 2. Three Separate Composition Chains
Don't think of "a variant" as a single thing. Think of three independent chains:
- Initial state chain (how game starts)
- Action validation chain (what's legal)
- Action execution chain (what happens)

Each variant wraps 0-3 of these chains.

### 3. Base Functions are Maximally Permissive
`baseGetValidActions` generates ALL possible actions (including nello/splash/plunge). Variants filter them out. Don't put variant logic in base functions.

### 4. Last One Wins
If tournament says "no nello" and one-hand says "game over", both happen. But if two variants conflict (e.g., both transform executeAction differently), last one wins because it's the outermost wrapper.

### 5. Event Sourcing Just Works
Because variants are applied at GameHost construction time, replaying from `actionHistory` just means: reconstruct the same composed functions, replay the actions. No variant state to serialize.

---

## Onboarding for Next Session

### Context
You're migrating a Texas 42 game from an imperative variant system to pure functional composition. Current system has 418-line VariantRegistry with lifecycle hooks. New system: three pure functions that variants wrap.

### What Exists
- Core game engine: `createInitialState`, `getValidActions`, `executeAction`
- Two variants to support: tournament (filter special bids), one-hand (skip bidding, end after one hand)
- GameHost that currently uses old VariantRegistry

### What to Build
See "Implementation Plan" section above. Key files:
- `src/game/variants/types.ts` - Type definitions
- `src/game/variants/tournament.ts` - ~10 lines
- `src/game/variants/oneHand.ts` - ~40 lines
- Update `GameHost.ts` to compose

### Start Here
1. Read "Key Principles" section (understand f(g(h(x))) pattern)
2. Read "Illustrative Examples" (see the pattern in action)
3. Create `src/game/variants/types.ts` (foundation)
4. Follow "Implementation Plan" in order

### Red Flags
- If you find yourself adding fields to GameState → STOP, variants should derive everything
- If base functions check for variants → STOP, variants wrap base, not vice versa
- If a variant is >50 lines → STOP, probably doing too much, split it up
- If composition is unclear → STOP, re-read "Target Architecture" section

### Questions to Ask
- Does this variant need to transform initialState, getValidActions, or executeAction?
- Is the variant wrapping the previous function (good) or calling it (bad)?
- Can I test this variant in isolation by mocking the base function?
- If I add a third variant, does composition still work?

---

## Quick Reference

### Variant Template
```typescript
export const myVariant: VariantFactory = (config) => ({
  transformInitialState: (base) => (gameConfig) => {
    let state = base(gameConfig);
    // ... modify initial state ...
    return state;
  },

  transformGetValidActions: (base) => (state) => {
    const actions = base(state);
    // ... filter/modify actions ...
    return actions;
  },

  transformExecuteAction: (base) => (state, action) => {
    const newState = base(state, action);
    // ... intercept transitions ...
    return newState;
  }
});
```

### Composition Pattern
```typescript
let f = base;
for (const v of variants) {
  if (v.transform) f = v.transform(f);
}
// f is now base wrapped by all variants
```

### Testing Pattern
```typescript
const variant = myVariant();
const wrapped = variant.transformX!(baseX);
const result = wrapped(input);
expect(result).toBe(expected);
```

---

**GO BUILD IT.** 🚀
