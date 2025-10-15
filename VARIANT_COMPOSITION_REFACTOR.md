# Variant Composition Refactor - Implementation Plan

**Last Updated:** 2025-01-15
**Status:** Ready for Implementation
**Estimated Effort:** Large (49 files impacted)

---

## Executive Summary

### Goal
Migrate from imperative variant system to pure functional composition with event sourcing as the foundation.

### Current Problems
- **418-line VariantRegistry** with imperative lifecycle hooks
- **Variant state tracked separately** from GameState (breaks event sourcing)
- **tournamentMode flag baked into GameState** and referenced in 49 files
- **Core engine knows about variants** (coupling)

### Solution
- **Pure function composition** over three transform points
- **Event sourcing as core state** (initialConfig + actionHistory = complete state)
- **Variants as composable transformers** (~20-40 lines each)
- **Base engine maximally permissive** (variants filter/transform)

### Impact
- Delete 418 lines of imperative code
- Remove variant fields from GameState
- Enable perfect state reconstruction via action replay
- Make variants composable, testable, and serializable

---

## 1. Architecture Principles

### 1.1 Event Sourcing First

**Core Truth:**
```typescript
GameState = replayActions(initialConfig, actionHistory)
```

**State Structure:**
```typescript
interface GameState {
  // Immutable source of truth
  initialConfig: GameConfig
  actionHistory: GameAction[]

  // Derived snapshot (cache for performance)
  // Everything here can be recomputed from config + history
  phase: GamePhase
  players: Player[]
  currentPlayer: number
  // ... all other game data
}
```

**Key Properties:**
- State is ALWAYS reproducible from history
- URL encoding = serialize config + actions
- Tests create state via action replay
- Time-travel debugging built-in
- Undo/redo for free

### 1.2 Pure Function Composition

**Variants are function transformers:**
```typescript
type Variant = {
  type: string  // For serialization
  transformInitialState?: (base: InitialStateFactory) => InitialStateFactory
  transformGetValidActions?: (base: StateMachine) => StateMachine
  transformExecuteAction?: (base: ExecuteAction) => ExecuteAction
}
```

**Optional Transforms:**
- Tournament needs only `transformGetValidActions` (filter bids)
- One-hand needs `transformInitialState` (skip bidding) + `transformExecuteAction` (end after one)
- Speed-mode needs only `transformGetValidActions` (annotate actions)

**Composition:**
```typescript
let createState = baseCreateInitialState
let getActions = baseGetValidActions
let executeAction = baseExecuteAction

for (const variant of config.variants) {
  if (variant.transformInitialState) {
    createState = variant.transformInitialState(createState)
  }
  if (variant.transformGetValidActions) {
    getActions = variant.transformGetValidActions(getActions)
  }
  if (variant.transformExecuteAction) {
    executeAction = variant.transformExecuteAction(executeAction)
  }
}

// Use composed functions
const state = createState(config)
const actions = getActions(state)
const newState = executeAction(state, action)
```

### 1.3 Base Engine is Maximally Permissive

**Base functions generate ALL structurally possible actions:**
- `baseGetValidActions` returns nello/splash/plunge bids
- `baseIsValidBid` allows special contracts
- No variant awareness in core engine

**Variants filter/transform at composition time:**
- Tournament variant filters out special bids
- Tests compose variants manually to test behavior

---

## 2. Event Sourcing Integration

### 2.1 Core Replay Function

```typescript
/**
 * Replay actions from initial config to reconstruct state.
 * This is the FUNDAMENTAL operation - all state derives from this.
 */
export function replayActions(
  config: GameConfig,
  actions: GameAction[]
): GameState {
  // Apply variants to get composed functions
  let createState = baseCreateInitialState
  let executeAction = baseExecuteAction

  for (const variantConfig of config.variants || []) {
    const variant = getVariant(variantConfig.type, variantConfig.config)
    if (variant.transformInitialState) {
      createState = variant.transformInitialState(createState)
    }
    if (variant.transformExecuteAction) {
      executeAction = variant.transformExecuteAction(executeAction)
    }
  }

  // Create initial state with composed factory
  let state = createState(config)

  // Replay all actions with composed executor
  for (const action of actions) {
    state = executeAction(state, action)
  }

  return state
}
```

### 2.2 State Structure Update

**Before (snapshot only):**
```typescript
interface GameState {
  phase: GamePhase
  players: Player[]
  tournamentMode: boolean  // ❌ variant flag
  // ... all derived state
}
```

**After (event sourcing):**
```typescript
interface GameState {
  // Source of truth (immutable)
  initialConfig: GameConfig
  actionHistory: GameAction[]

  // Derived state (everything below can be recomputed)
  phase: GamePhase
  players: Player[]
  currentPlayer: number
  dealer: number
  bids: Bid[]
  // ... all other game data

  // Theme (first-class, not derived)
  theme: string
  colorOverrides: Record<string, string>
}
```

### 2.3 Usage Patterns

**Create state at arbitrary point:**
```typescript
// Tests
const state = replayActions(
  { shuffleSeed: 42069, variants: [{ type: 'one-hand' }] },
  [
    { type: 'play', player: 0, dominoId: '0-2' },
    { type: 'play', player: 1, dominoId: '1-5' },
    // ... more actions
  ]
)

// URL replay (future)
const state = replayActions(
  parseConfigFromURL(params.s, params.h),
  parseActionsFromURL(params.a)
)
```

**Execute action:**
```typescript
function executeAction(state: GameState, action: GameAction): GameState {
  return {
    ...state,
    actionHistory: [...state.actionHistory, action],
    // Apply composed executor to derive new snapshot
    ...applyActionToSnapshot(state, action)
  }
}
```

---

## 3. Variant System Design

### 3.1 Type Definitions

**File: `src/game/variants/types.ts`**
```typescript
import type { GameState, GameAction } from '../types';

// State machine produces actions from state
export type StateMachine = (state: GameState) => GameAction[]

// Initial state factory
export type InitialStateFactory = (config: GameConfig) => GameState

// Action executor
export type ExecuteAction = (state: GameState, action: GameAction) => GameState

// Transform functions (optional, variants specify what they need)
export type InitialStateTransform = (base: InitialStateFactory) => InitialStateFactory
export type ActionValidatorTransform = (base: StateMachine) => StateMachine
export type ActionExecutorTransform = (base: ExecuteAction) => ExecuteAction

// Variant definition (function transformer)
export interface Variant {
  type: string  // For serialization
  transformInitialState?: InitialStateTransform
  transformGetValidActions?: ActionValidatorTransform
  transformExecuteAction?: ActionExecutorTransform
}

// Factory pattern for parameterized variants
export type VariantFactory = (config?: any) => Variant

// Serializable config (stored in GameState)
export interface VariantConfig {
  type: string
  config?: Record<string, any>
}
```

### 3.2 Variant Registry

**File: `src/game/variants/registry.ts`**
```typescript
import type { VariantFactory } from './types';
import { tournamentVariant } from './tournament';
import { oneHandVariant } from './oneHand';
import { speedModeVariant } from './speedMode';

const VARIANT_REGISTRY: Record<string, VariantFactory> = {
  'tournament': tournamentVariant,
  'one-hand': oneHandVariant,
  'speed-mode': speedModeVariant
};

/**
 * Get variant factory by type.
 * Pure lookup - no side effects, no registration.
 */
export function getVariant(type: string, config?: any): Variant {
  const factory = VARIANT_REGISTRY[type];
  if (!factory) {
    throw new Error(`Unknown variant: ${type}`);
  }
  return factory(config);
}

export { tournamentVariant, oneHandVariant, speedModeVariant };
export type { Variant, VariantFactory, VariantConfig } from './types';
```

### 3.3 Tournament Variant

**File: `src/game/variants/tournament.ts`**
```typescript
import type { VariantFactory } from './types';

/**
 * Tournament mode: Disable special contracts (nello, splash, plunge).
 * Only standard point and mark bids allowed.
 */
export const tournamentVariant: VariantFactory = () => ({
  type: 'tournament',

  transformGetValidActions: (base) => (state) => {
    const actions = base(state);

    // Filter out special bid types
    return actions.filter(action =>
      action.type !== 'bid' ||
      !['nello', 'splash', 'plunge'].includes(action.bid)
    );
  }
});
```

**~10 lines. Pure function wrapper. No lifecycle hooks.**

### 3.4 One-Hand Variant

**File: `src/game/variants/oneHand.ts`**
```typescript
import type { VariantFactory } from './types';
import { executeAction } from '../core/actions';

/**
 * One-hand mode: Skip bidding phase, play single hand, end immediately.
 *
 * Use case:
 * - New players: Don't understand bidding yet
 * - Expert challenges: Share competitive scenarios by seed
 *
 * Mechanics:
 * - Seed determines dealt hands (via base createInitialState)
 * - Hardcoded bid: Player 3 bids 30 points, suit 4 trump
 * - Game starts at 'playing' phase (bidding/trump already done)
 * - After scoring, end game instead of starting new hand
 */
export const oneHandVariant: VariantFactory = () => ({
  type: 'one-hand',

  // Transform 1: Fast-forward through bidding to 'playing' phase
  transformInitialState: (base) => (config) => {
    // Base creates state at 'bidding' phase with dealt hands
    let state = base(config);

    // Fast-forward through bidding (hardcoded: player 3 bids 30)
    state = executeAction(state, {
      type: 'bid',
      player: 3,
      bid: 'points',
      value: 30
    });

    // Other players pass
    state = executeAction(state, { type: 'pass', player: 0 });
    state = executeAction(state, { type: 'pass', player: 1 });
    state = executeAction(state, { type: 'pass', player: 2 });

    // Player 3 selects trump (hardcoded: suit 4)
    state = executeAction(state, {
      type: 'select-trump',
      player: 3,
      trump: { type: 'suit', suit: 4 }
    });

    // Now at 'playing' phase - ready to play!
    return state;
  },

  // Transform 2: Intercept scoring to end game instead of new hand
  transformExecuteAction: (base) => (state, action) => {
    const newState = base(state, action);

    // After scoring, base would transition to 'bidding' for new hand
    // We intercept to end game instead
    if (action.type === 'score-hand' && newState.phase === 'bidding') {
      return {
        ...newState,
        phase: 'game_end'
      };
    }

    return newState;
  }
});
```

**~40 lines. Two transforms. Base knows nothing about one-hand.**

### 3.5 Speed Mode Variant

**File: `src/game/variants/speedMode.ts`**
```typescript
import type { VariantFactory } from './types';

/**
 * Speed mode: Auto-play when only one valid option.
 * Adds 'autoExecute' flag to single-option plays for client UX.
 */
export const speedModeVariant: VariantFactory = () => ({
  type: 'speed-mode',

  transformGetValidActions: (base) => (state) => {
    const actions = base(state);

    if (state.phase !== 'playing') return actions;

    const playActions = actions.filter(a => a.type === 'play');

    // If only one valid play, mark for auto-execution
    if (playActions.length === 1) {
      return [{
        ...playActions[0],
        autoExecute: true,
        delay: 300  // ms delay for visual feedback
      }];
    }

    return actions;
  }
});
```

**~15 lines. Single transform. Annotates actions.**

---

## 4. Implementation Plan

### Phase 1: Foundation (No Breaking Changes)

**Goal:** Build new architecture alongside old system.

#### Step 1.1: Create Variant Type System
```bash
# Create directory structure
mkdir -p src/game/variants

# Create files
touch src/game/variants/types.ts
touch src/game/variants/registry.ts
touch src/game/variants/tournament.ts
touch src/game/variants/oneHand.ts
touch src/game/variants/speedMode.ts
```

**Files to create:**
1. `src/game/variants/types.ts` - Type definitions (see section 3.1)
2. `src/game/variants/registry.ts` - Variant lookup (see section 3.2)
3. `src/game/variants/tournament.ts` - Tournament implementation (see section 3.3)
4. `src/game/variants/oneHand.ts` - One-hand implementation (see section 3.4)
5. `src/game/variants/speedMode.ts` - Speed mode implementation (see section 3.5)

#### Step 1.2: Create Event Sourcing Utilities
```bash
touch src/game/core/replay.ts
```

**File: `src/game/core/replay.ts`**
```typescript
import type { GameState, GameAction, GameConfig } from '../types';
import { createInitialState } from './state';
import { executeAction as baseExecuteAction } from './actions';
import { getVariant } from '../variants/registry';

/**
 * Replay actions from initial config to reconstruct state.
 * This is the FUNDAMENTAL operation for event sourcing.
 */
export function replayActions(
  config: GameConfig,
  actions: GameAction[]
): GameState {
  // Compose variant transforms
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

  // Create initial state with composed factory
  let state = createState(config);

  // Replay all actions with composed executor
  for (const action of actions) {
    state = executeAction(state, action);
  }

  return state;
}

/**
 * Create initial state with variants applied.
 * Convenience wrapper for replayActions with empty history.
 */
export function createInitialStateWithVariants(config: GameConfig): GameState {
  return replayActions(config, []);
}
```

#### Step 1.3: Update GameConfig Type

**File: `src/shared/multiplayer/protocol.ts`**

**Change:**
```typescript
export interface GameConfig {
  playerTypes: ('human' | 'ai')[];
  shuffleSeed?: number;

  // NEW: Variants to compose (replaces singular variant)
  variants?: VariantConfig[];

  // OLD: Remove in Phase 2
  variant?: GameVariant;  // DEPRECATED
}

// NEW: Simple serializable config
export interface VariantConfig {
  type: string;
  config?: Record<string, any>;
}

// OLD: Delete in Phase 2
export interface GameVariant {
  type: 'standard' | 'one-hand' | 'tournament';
  config?: any;
}
```

### Phase 2: Integration (Wire Into GameHost)

**Goal:** Make GameHost use new variant system while keeping old one working.

#### Step 2.1: Update GameHost Constructor

**File: `src/server/game/GameHost.ts`**

**Add at top:**
```typescript
import { getVariant } from '../../game/variants/registry';
import type { VariantConfig } from '../../shared/multiplayer/protocol';
import { getValidActions as baseGetValidActions } from '../../game/core/gameEngine';
import { executeAction as baseExecuteAction } from '../../game/core/actions';
import { replayActions } from '../../game/core/replay';
```

**Modify constructor:**
```typescript
constructor(gameId: string, config: GameConfig, players: PlayerSession[]) {
  this.gameId = gameId;
  this.created = Date.now();
  this.lastUpdate = this.created;

  // Validate players
  if (players.length !== 4) {
    throw new Error(`Expected 4 players, got ${players.length}`);
  }

  this.players = new Map(players.map(p => [p.playerId, p]));

  // NEW: Compose variants
  const variantConfigs = config.variants || [];

  let getValidActions = baseGetValidActions;
  let executeAction = baseExecuteAction;

  for (const variantConfig of variantConfigs) {
    const variant = getVariant(variantConfig.type, variantConfig.config);

    if (variant.transformGetValidActions) {
      getValidActions = variant.transformGetValidActions(getValidActions);
    }
    if (variant.transformExecuteAction) {
      executeAction = variant.transformExecuteAction(executeAction);
    }
  }

  // Store composed functions
  this.getValidActionsComposed = getValidActions;
  this.executeActionComposed = executeAction;

  // Create initial state using event sourcing
  const initialState = replayActions(config, []);

  this.mpState = {
    state: initialState,
    sessions: players
  };
}
```

**Add instance variables:**
```typescript
private getValidActionsComposed: (state: GameState) => GameAction[];
private executeActionComposed: (state: GameState, action: GameAction) => GameState;
```

#### Step 2.2: Update getView() Method

**File: `src/server/game/GameHost.ts`**

**Change:**
```typescript
private createView(forPlayerId?: string): GameView {
  const { state, sessions } = this.mpState;

  // Use composed getValidActions (not base)
  const allValidActions = this.getValidActionsComposed(state);

  // ... rest unchanged
}
```

#### Step 2.3: Update executeAction() Method

**File: `src/server/game/GameHost.ts`**

**Remove all VariantRegistry calls:**
```typescript
executeAction(playerId: string, action: GameAction): { ok: boolean; error?: string } {
  const request = { playerId, action };

  const result = authorizeAndExecute(this.mpState, request);

  if (!result.ok) {
    return { ok: false, error: result.error };
  }

  // Update state (VariantRegistry calls REMOVED)
  this.mpState = result.value;
  this.lastUpdate = Date.now();

  // Notify all listeners
  this.notifyListeners();

  return { ok: true };
}
```

**Delete private methods:**
- `createInitialStateForVariant()` - replaced by replayActions
- `applyVariantRules()` - replaced by composed functions
- All VariantRegistry interactions

### Phase 3: Destruction (Big Bang Deletion)

**Goal:** Remove old variant system completely.

#### Step 3.1: Delete Old Variant System

```bash
# Delete imperative registry
rm src/server/game/VariantRegistry.ts

# Remove all imports
grep -r "VariantRegistry" src/ --files-with-matches | xargs sed -i '/VariantRegistry/d'
```

**Files affected:**
- `src/server/game/GameHost.ts` (already updated in Phase 2)
- Any test files importing VariantRegistry

#### Step 3.2: Remove tournamentMode from GameState

**File: `src/game/types.ts`**

**Remove these lines:**
```typescript
export interface GameState {
  // DELETE:
  gameTarget: number;        // Line 169
  tournamentMode: boolean;   // Line 170

  // DELETE (test compatibility):
  hands?: { [playerId: number]: Domino[] };
  bidWinner?: number;
  isComplete?: boolean;
  winner?: number;
}
```

**This will break 49 files.** Fix them all in the next step.

#### Step 3.3: Fix All tournamentMode References

**Strategy:** Replace all `tournamentMode` usages with variant composition.

**Files to update (49 total):**

**Core Engine Files (remove parameter):**
```typescript
// src/game/core/rules.ts
// BEFORE:
export function isValidBid(state: GameState, bid: Bid, playerHand?: Domino[]): boolean {
  return isValidOpeningBid(bid, playerHand, state.tournamentMode);
}

// AFTER:
export function isValidBid(state: GameState, bid: Bid, playerHand?: Domino[]): boolean {
  return isValidOpeningBid(bid, playerHand, false);  // Always permissive
}

// BEFORE:
export function isValidOpeningBid(bid: Bid, playerHand?: Domino[], tournamentMode: boolean = true): boolean

// AFTER:
export function isValidOpeningBid(bid: Bid, playerHand?: Domino[]): boolean
// Remove all `if (tournamentMode)` checks - always allow all bid types
```

**State Creation Files:**
```typescript
// src/game/core/state.ts
// BEFORE:
export function createInitialState(options?: {
  shuffleSeed?: number,
  tournamentMode?: boolean,
  playerTypes?: ('human' | 'ai')[]
}): GameState

// AFTER:
export function createInitialState(options?: {
  shuffleSeed?: number,
  playerTypes?: ('human' | 'ai')[]
}): GameState
// Remove tournamentMode from options and state initialization
```

**Test Files (use variants):**
```typescript
// BEFORE:
const state = createInitialState({
  shuffleSeed: 123,
  tournamentMode: true
});

// AFTER:
const state = replayActions({
  shuffleSeed: 123,
  variants: [{ type: 'tournament' }]
}, []);
```

**Automated replacement:**
```bash
# Find all files with tournamentMode
grep -r "tournamentMode" src/ --files-with-matches > files_to_fix.txt

# Review each file and update according to patterns above
# Cannot automate this - requires understanding context
```

#### Step 3.4: Update Protocol

**File: `src/shared/multiplayer/protocol.ts`**

**Remove old GameVariant:**
```typescript
// DELETE:
export interface GameVariant {
  type: 'standard' | 'one-hand' | 'tournament';
  config?: any;
}

// DELETE from GameConfig:
variant?: GameVariant;
```

**Keep only new VariantConfig:**
```typescript
export interface VariantConfig {
  type: string;  // Open-ended, not enum
  config?: Record<string, any>;
}

export interface GameConfig {
  playerTypes: ('human' | 'ai')[];
  shuffleSeed?: number;
  variants?: VariantConfig[];  // Array of composable variants
}
```

### Phase 4: Test Migration

**Goal:** Update all tests to use new variant system.

#### Step 4.1: Create Test Helpers

**File: `src/tests/helpers/variantTestHelper.ts`**
```typescript
import { replayActions } from '../../game/core/replay';
import type { GameConfig, GameAction } from '../../game/types';

/**
 * Create game state with variants for testing.
 * Uses event sourcing - can include action history.
 */
export function createTestGameState(
  seed: number,
  variantTypes: string[],
  actions: GameAction[] = []
): GameState {
  const config: GameConfig = {
    shuffleSeed: seed,
    playerTypes: ['human', 'ai', 'ai', 'ai'],
    variants: variantTypes.map(type => ({ type }))
  };

  return replayActions(config, actions);
}

/**
 * Test variant behavior by composing manually.
 */
export function testVariantFilter(
  variantType: string,
  baseActions: GameAction[]
): GameAction[] {
  const variant = getVariant(variantType);
  if (!variant.transformGetValidActions) {
    return baseActions;
  }

  const composed = variant.transformGetValidActions(
    (_state) => baseActions
  );

  return composed({} as GameState);
}
```

#### Step 4.2: Update Unit Tests

**Pattern for tournament mode tests:**
```typescript
// BEFORE:
test('tournament mode disallows nello', () => {
  const state = createInitialState({
    tournamentMode: true,
    shuffleSeed: 123
  });
  const actions = getValidActions(state);
  expect(actions.some(a => a.bid === 'nello')).toBe(false);
});

// AFTER:
test('tournament variant filters nello', () => {
  const state = createTestGameState(123, ['tournament']);
  const actions = getValidActions(state);  // Base actions

  // Apply tournament variant
  const variant = getVariant('tournament');
  const composed = variant.transformGetValidActions!(getValidActions);
  const filtered = composed(state);

  expect(filtered.some(a => a.bid === 'nello')).toBe(false);
  expect(actions.some(a => a.bid === 'nello')).toBe(true);  // Base allows it
});
```

**Pattern for one-hand tests:**
```typescript
test('one-hand starts at playing phase', () => {
  const state = createTestGameState(123, ['one-hand']);

  expect(state.phase).toBe('playing');
  expect(state.winningBidder).toBe(3);
  expect(state.currentBid.value).toBe(30);
});

test('one-hand ends after scoring', () => {
  const state = createTestGameState(123, ['one-hand']);

  // Play through hand to scoring
  let currentState = state;
  // ... play all tricks ...

  // Score hand
  currentState = executeAction(currentState, { type: 'score-hand' });

  expect(currentState.phase).toBe('game_end');
});
```

#### Step 4.3: Update Integration Tests

**Pattern for action replay:**
```typescript
test('can replay game from action history', () => {
  const config = {
    shuffleSeed: 42069,
    variants: [{ type: 'one-hand' }]
  };

  const actions = [
    { type: 'play', player: 0, dominoId: '0-2' },
    { type: 'play', player: 1, dominoId: '1-5' },
    { type: 'play', player: 2, dominoId: '2-3' },
    { type: 'play', player: 3, dominoId: '3-6' }
  ];

  const state = replayActions(config, actions);

  expect(state.currentTrick.length).toBe(4);
  expect(state.actionHistory).toEqual(actions);
});
```

### Phase 5: Store Integration

**Goal:** Update Svelte stores to use new variant system.

#### Step 5.1: Update gameStore.ts

**File: `src/stores/gameStore.ts`**

**Change variant creation:**
```typescript
// BEFORE:
export const gameVariants = {
  startOneHand: async (seed?: number) => {
    const newConfig: GameConfig = {
      playerTypes,
      ...(seed ? { shuffleSeed: seed } : {}),
      variant: {
        type: 'one-hand',
        config: seed ? { targetHand: 1, originalSeed: seed } : {}
      }
    };
    // ...
  }
};

// AFTER:
export const gameVariants = {
  startOneHand: async (seed?: number) => {
    const newConfig: GameConfig = {
      playerTypes,
      shuffleSeed: seed || Math.floor(Math.random() * 1000000),
      variants: [{ type: 'one-hand' }]
    };
    // ...
  }
};
```

#### Step 5.2: Update InProcessAdapter

**File: `src/server/offline/InProcessAdapter.ts`**

**Remove variant-specific logic:**
```typescript
// BEFORE:
private async handleCreateGame(config: GameConfig, _clientId: string): Promise<void> {
  // Check if we need to find a competitive seed for one-hand mode
  if (config.variant?.type === 'one-hand' && !config.shuffleSeed) {
    config = await this.findCompetitiveSeed(config);
  }
  // ...
}

// AFTER:
private async handleCreateGame(config: GameConfig, _clientId: string): Promise<void> {
  // Seed finding removed - out of scope for this refactor
  // Game creation is fast and pure
  const instance = this.registry.createGame(config);
  // ...
}
```

**Note:** Seed finding is explicitly OUT OF SCOPE. Remove it for now, add back later as separate feature.

---

## 5. Success Criteria

### 5.1 Code Quality
- ✅ TypeScript compiles with no errors (`npm run typecheck`)
- ✅ No references to `tournamentMode`, `gameTarget`, `VariantState`
- ✅ All variants are pure function transforms (~20-40 lines each)
- ✅ Old `VariantRegistry.ts` (418 lines) deleted
- ✅ Event sourcing utilities (`replayActions`) working

### 5.2 Functional Correctness
- ✅ Standard game: plays to 7 marks normally
- ✅ Tournament game: no nello/splash/plunge bids available
- ✅ One-hand game: starts at 'playing', ends after one hand
- ✅ Can create game state via action replay (`replayActions`)
- ✅ Composed variants: both effects apply correctly

### 5.3 Architecture
- ✅ All composition happens in GameHost constructor
- ✅ Base functions (in `core/`) have zero variant awareness
- ✅ Variants (in `variants/`) compose over base functions
- ✅ No variant-specific state in GameState
- ✅ Event sourcing as core state representation

### 5.4 Tests
- ✅ All unit tests passing (`npm test`)
- ✅ All 49 files updated for tournamentMode removal
- ✅ New tests for variant composition
- ✅ Tests use `replayActions` for state creation

---

## 6. Out of Scope (Future Enhancements)

### 6.1 Seed Finder (Deferred)
**Current:** Removed from InProcessAdapter
**Future:** Implement as compositional system:
- Pure `findCompetitiveSeed` function
- Variant-aware evaluators
- Client-side or server-side execution
- Progress reporting via existing PROGRESS messages

**Why deferred:** Seed finding is orthogonal to variant composition. Variants work fine with any seed (even random). Add seed finding after composition system is stable.

### 6.2 E2E Test Refactoring (Separate Effort)
**Current:** E2E tests are broken (URL replay issues)
**Future:** Fix URL replay system to work with event sourcing
**Why separate:** E2E tests need comprehensive URL encoding/decoding work that's independent of variant system

### 6.3 URL Encoding/Compression (Separate Effort)
**Current:** Beautiful URL compression exists but misaligned with new architecture
**Future:** Migrate URL system to serialize `initialConfig + actionHistory`
**Why separate:** Event sourcing architecture supports this perfectly, but implementation is large

**Critical constraint:** Must maintain URL isomorphism (state = deserialize(URL)). Event sourcing makes this trivial - just serialize config + actions.

### 6.4 Capability System (Separate Effort)
**Current:** Simple session-based authorization
**Future:** Fine-grained capability tokens (see vision doc section 4)
**Why separate:** Capability system is a large feature independent of variants

### 6.5 AI Integration for One-Hand (Future)
**Current:** Hardcoded bid (player 3 bids 30, suit 4 trump)
**Future:** Deterministic AI selects bid/trump based on hand strength
**Why deferred:** Hardcoded version is sufficient for initial implementation. AI integration can be added without changing variant structure.

---

## 7. Migration Checklist

### Pre-Migration
- [ ] Review this document completely
- [ ] Understand event sourcing architecture
- [ ] Understand variant composition pattern
- [ ] Identify any missing concerns

### Phase 1: Foundation
- [ ] Create `src/game/variants/` directory
- [ ] Implement `types.ts` (type definitions)
- [ ] Implement `registry.ts` (variant lookup)
- [ ] Implement `tournament.ts` (filter special bids)
- [ ] Implement `oneHand.ts` (skip bidding, end after one)
- [ ] Implement `speedMode.ts` (annotate single plays)
- [ ] Create `src/game/core/replay.ts` (event sourcing)
- [ ] Update `GameConfig` to add `variants` array
- [ ] Verify TypeScript compiles

### Phase 2: Integration
- [ ] Update GameHost constructor (compose variants)
- [ ] Add composed function storage
- [ ] Update `getView()` to use composed functions
- [ ] Update `executeAction()` to remove VariantRegistry
- [ ] Remove variant-specific methods from GameHost
- [ ] Verify GameHost works with variants

### Phase 3: Destruction
- [ ] Delete `VariantRegistry.ts` (418 lines)
- [ ] Remove `tournamentMode` from GameState type
- [ ] Remove `gameTarget` from GameState type
- [ ] Remove test compatibility fields
- [ ] Fix `src/game/core/rules.ts` (remove tournamentMode params)
- [ ] Fix `src/game/core/state.ts` (remove tournamentMode option)
- [ ] Fix `src/game/core/actions.ts` (remove variant checks)
- [ ] Fix all 49 files referencing tournamentMode
- [ ] Remove old `GameVariant` from protocol
- [ ] Verify TypeScript compiles (will have errors until all fixed)

### Phase 4: Test Migration
- [ ] Create test helper utilities
- [ ] Update tournament mode tests (use variant composition)
- [ ] Update one-hand tests (use replayActions)
- [ ] Add variant composition tests
- [ ] Add event sourcing tests (replayActions)
- [ ] Verify all unit tests pass (`npm test`)

### Phase 5: Store Integration
- [ ] Update `gameStore.ts` (use variants array)
- [ ] Update `InProcessAdapter` (remove seed finder)
- [ ] Update any UI components checking variant types
- [ ] Verify game creation works in browser

### Final Verification
- [ ] `npm run typecheck` passes
- [ ] `npm test` passes (all unit tests)
- [ ] Manual testing: standard game works
- [ ] Manual testing: tournament mode works
- [ ] Manual testing: one-hand mode works
- [ ] Manual testing: action replay works
- [ ] Review code for any remaining variant coupling

---

## 8. Key Insights (Remember These)

### 8.1 Event Sourcing is the Foundation
State = `replayActions(config, history)`. Everything else is derived. This gives us:
- Perfect state reconstruction
- URL serialization for free
- Time-travel debugging
- Test state creation via action replay

### 8.2 Variants Transform, They Don't Know
Variants wrap base functions. They don't know about:
- Other variants
- How they're composed
- What order they're applied
- The base implementation

Pure function composition: `f(g(h(x)))` all the way down.

### 8.3 Base Functions are Maximally Permissive
`baseGetValidActions` generates ALL structurally possible actions. Variants filter/transform. This means:
- Core engine is simpler (no variant logic)
- Tests can verify base behavior independently
- Variants can be tested by composing manually

### 8.4 Optional Transforms, Not Required Hooks
Variants specify ONLY what they need:
- Tournament: only `transformGetValidActions`
- One-hand: `transformInitialState` + `transformExecuteAction`
- Speed-mode: only `transformGetValidActions`

Not all variants need all transforms.

### 8.5 Composition Order Matters
Variants compose left-to-right. Last one wins if there's conflict:
```typescript
variants: [
  { type: 'tournament' },  // First: filters special bids
  { type: 'one-hand' }     // Last: if it changes phase, that's final
]
```

### 8.6 One-Hand is Simple Now
No AI integration, no seed finding complexity. Just:
1. Use seed to deal hands
2. Hardcode bid/trump
3. Fast-forward to playing
4. End after scoring

Clean, deterministic, testable. AI integration comes later.

---

## 9. FAQ

**Q: Why remove tournamentMode flag instead of just adding variants?**
A: Flags in state create coupling. Variants should be external transforms, not internal flags. Event sourcing + composition gives us the same functionality with zero coupling.

**Q: Why big-bang delete instead of gradual migration?**
A: Gradual creates half-broken state. Big-bang forces us to fix everything, ensures consistency, and is done in one PR. The code isn't that deeply coupled - 49 files is manageable.

**Q: Why defer seed finder?**
A: Seed finding is orthogonal. Variants work with any seed. Adding seed finding later doesn't require changing variant structure. Simplify now, enhance later.

**Q: How do I test variant behavior?**
A: Manually compose variants in tests:
```typescript
const variant = getVariant('tournament');
const composed = variant.transformGetValidActions!(base);
const filtered = composed(state);
```

**Q: Can variants have state?**
A: No. Variants are pure functions. They can close over config parameters, but all state derives from `GameState.actionHistory`. If a variant needs to "remember" something, it computes it from action history.

**Q: What if two variants conflict?**
A: Last one wins. Composition is `f(g(h(x)))` - rightmost is outermost, executes last, has final say.

**Q: How do I create game state at arbitrary point?**
A: Use `replayActions(config, actions)`. This is THE way to create state. Tests, URL replay, everything uses this.

**Q: Why remove test compatibility fields from GameState?**
A: They were hacks for old tests. Event sourcing gives us better ways to create test state. Clean slate.

---

## 10. Next Steps After Completion

Once variant composition is working:

1. **Add Seed Finder** (new feature)
   - Implement compositional evaluators
   - Client-side or server-side execution
   - Progress reporting UI

2. **Fix URL Encoding** (separate effort)
   - Serialize initialConfig + actionHistory
   - Migrate URL compression system
   - Fix E2E tests

3. **Enhance One-Hand** (enhancement)
   - Add AI bid/trump selection
   - Make bid/trump configurable
   - Add difficulty levels

4. **Add More Variants** (new features)
   - Daily challenge mode
   - Tutorial mode with hints
   - Custom scoring variants

5. **Capability System** (new feature)
   - Fine-grained access control
   - Spectator mode
   - Coach mode

---

**This document is the complete plan. Follow it sequentially. Each phase builds on the previous. Success criteria ensure nothing is missed.**

**GO BUILD IT.** 🚀
