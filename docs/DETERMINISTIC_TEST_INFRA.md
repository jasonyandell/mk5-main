# Deterministic Test Infrastructure

## Vision

Instead of hunting for the right seed or manually constructing complex game states, tests should use **named hand configurations** that are:
- **Minimal**: Small set of well-designed deals that cover many scenarios
- **Readable**: You can see exactly what each player has
- **Reusable**: One configuration serves multiple test cases
- **Type-safe**: Defined in code with autocomplete and refactoring support
- **Event-sourced**: Replay actions from known initial state

## Architecture

### Core Components

```
src/tests/fixtures/hands.ts          // Hand configuration definitions
src/tests/helpers/testGames.ts       // Helper functions for using handsets
src/game/core/state.ts               // createInitialStateWithHands()
```

### Hand Configuration Format

Each handset is a complete, valid deal (28 dominoes, 7 per player):

```typescript
export interface HandSet {
  name: string;
  description: string;
  dealer: number;
  hands: [Domino[], Domino[], Domino[], Domino[]];
}

export const BALANCED_STANDARD: HandSet = {
  name: 'BALANCED_STANDARD',
  description: 'Balanced hands for standard gameplay',
  dealer: 3,
  hands: [
    [d('6-6'), d('6-5'), d('5-4'), d('4-3'), d('3-2'), d('2-1'), d('1-0')],
    [d('6-4'), d('6-3'), d('5-5'), d('4-4'), d('3-3'), d('2-2'), d('1-1')],
    [d('6-2'), d('6-1'), d('5-3'), d('5-2'), d('4-2'), d('3-1'), d('2-0')],
    [d('6-0'), d('5-1'), d('5-0'), d('4-1'), d('4-0'), d('3-0'), d('0-0')],
  ]
};
```

## Initial Handsets

### 1. BALANCED_STANDARD

**Purpose**: Normal distribution with no extreme hands

**Characteristics**:
- Each player has mix of suits, doubles, and points
- No obvious special contract opportunities
- Represents typical game distribution

**Use for**:
- Authorization tests (no special rules needed)
- Basic bidding scenarios
- Standard trump selection
- Play validation
- Any test that doesn't need special hands

**Player 0** (left of dealer, bids first):
```
6-6, 6-5, 5-4, 4-3, 3-2, 2-1, 1-0
```

### 2. NELLO_FAVORABLE

**Purpose**: Player 0 has ideal nello hand

**Characteristics**:
- Player 0: All doubles (perfect for nello - will lose every trick)
- Player 1: High sixes (will dominate tricks)
- Player 2: Partner (sits out in nello)
- Player 3: Mixed mid-range

**Use for**:
- Nello success scenarios
- Testing 3-player tricks
- Partner sit-out mechanics
- Early termination when bidder wins a trick

**Player 0** (bidder):
```
6-6, 5-5, 4-4, 3-3, 2-2, 1-1, 0-0  // All doubles - loses all tricks
```

### 3. PLUNGE_READY

**Purpose**: Player 0 can bid plunge (4+ doubles)

**Characteristics**:
- Player 0: 4 doubles + strong sixes
- Player 2: Partner with solid support
- Players 1/3: Opponents with scattered strength

**Use for**:
- Plunge bidding scenarios
- Testing 4-double requirement
- Partner cooperation
- Must-win-all-tricks scenarios

**Player 0** (bidder):
```
6-6, 5-5, 4-4, 3-3, 6-5, 6-4, 6-3  // 4 doubles + high sixes
```

### 4. SPLASH_READY

**Purpose**: Player 0 can bid splash (3 doubles)

**Characteristics**:
- Player 0: 3 doubles + high sixes
- Similar to PLUNGE_READY but one less double

**Use for**:
- Splash bidding scenarios
- Testing 3-double requirement
- Comparing splash vs plunge

**Player 0** (bidder):
```
6-6, 5-5, 4-4, 6-5, 6-4, 6-3, 6-2  // 3 doubles + strong sixes
```

## Usage Examples

### Creating Initial State

```typescript
import { createStateFromHandSet } from '../helpers/testGames';

// Basic usage - no special rules
const { state, ctx } = createStateFromHandSet('BALANCED_STANDARD');

// With special rules enabled
const { state, ctx } = createStateFromHandSet('NELLO_FAVORABLE', {
  enabledRuleSets: ['nello']
});
```

### Playing Action Sequences

```typescript
import { playActionsFromHandSet } from '../helpers/testGames';

const { state, ctx } = playActionsFromHandSet(
  'NELLO_FAVORABLE',
  [
    { type: 'bid', player: 0, bid: 'marks', value: 1 },
    { type: 'pass', player: 1 },
    { type: 'pass', player: 2 },
    { type: 'pass', player: 3 },
    { type: 'select-trump', player: 0, trump: { type: 'nello' } },
  ],
  { enabledRuleSets: ['nello'] }
);

expect(state.phase).toBe('playing');
expect(state.trump.type).toBe('nello');
```

### Complete Test Example

```typescript
describe('Authorization', () => {
  it('correctly handles bid actions with capabilities', () => {
    // Use BALANCED_STANDARD - player 0's turn to bid
    const { state, ctx } = createStateFromHandSet('BALANCED_STANDARD');
    const sessions = createTestSessions();

    const bidAction: GameAction = { type: 'bid', player: 0, bid: 'points', value: 30 };

    // Player 0 CAN bid (current player + has capability)
    expect(canPlayerExecuteAction(sessions[0]!, bidAction, state, ctx)).toBe(true);

    // Others CANNOT (not their turn, even with capability)
    expect(canPlayerExecuteAction(sessions[1]!, bidAction, state, ctx)).toBe(false);
  });
});
```

## Design Principles

### Keep the Set Minimal

Only add new handsets when existing ones don't cover the scenario. Consider:
- Can you use BALANCED_STANDARD and play a few actions?
- Can you reuse NELLO_FAVORABLE for your authorization test?
- Is this truly a new scenario or just a different phase of an existing handset?

### Document Intent

Each handset should have:
1. **Clear name** - Describes the key characteristic
2. **Purpose statement** - Why this handset exists
3. **Characteristics** - What makes hands special
4. **Use cases** - Concrete test scenarios

Example:
```typescript
/**
 * AUTHORIZATION_PHASES
 *
 * Designed for testing authorization across ALL phases.
 * Balanced hands that allow clean progression through bidding → trump → playing.
 *
 * Characteristics:
 * - Player 1 can outbid player 0 (enables bidding tests)
 * - Player 1 has trump options (enables trump selection tests)
 * - All players have valid plays (enables play tests)
 * - Tricks complete normally (enables consensus tests)
 *
 * Use for:
 * - Phase transition authorization
 * - Multi-phase test scenarios
 * - Testing authorization persistence across phases
 */
```

### Valid by Construction

Each handset MUST:
- Contain exactly 28 dominoes (7 per player)
- Not duplicate any domino
- Specify a valid dealer (0-3)
- Have internally consistent hands (player 0 leads after dealer 3, etc.)

### Type Safety

Use the helper function `d()` to parse domino IDs:
```typescript
function d(id: string): Domino {
  const [a, b] = id.split('-').map(Number);
  const high = Math.max(a!, b!);
  const low = Math.min(a!, b!);

  // Calculate points (5s and 10s)
  let points = 0;
  if (high === 5 && low === 5) points = 10;
  if (high === 6 && low === 4) points = 10;
  if (high === 5 && low === 0) points = 5;
  if (high === 4 && low === 1) points = 5;
  if (high === 3 && low === 2) points = 5;

  return { id, high, low, points };
}
```

## Implementation Details

### State Creation Function

```typescript
// src/game/core/state.ts

export function createInitialStateWithHands(options: {
  hands: [Domino[], Domino[], Domino[], Domino[]],
  dealer?: number,
  playerTypes?: ('human' | 'ai')[],
  theme?: string,
  colorOverrides?: Record<string, string>
}): GameState {
  const dealer = options.dealer ?? 3;
  const currentPlayer = getPlayerLeftOfDealer(dealer);
  const playerTypes = options.playerTypes ?? ['human', 'ai', 'ai', 'ai'];

  return {
    initialConfig: {
      playerTypes,
      // Store hands as IDs for serialization
      hands: options.hands.map(h => h.map(d => d.id)),
      theme: options.theme ?? 'business',
      colorOverrides: options.colorOverrides ?? {}
    },

    phase: 'bidding' as const,
    players: [
      {
        id: 0,
        name: 'Player 1',
        hand: options.hands[0],
        teamId: 0,
        marks: 0,
        suitAnalysis: analyzeSuits(options.hands[0])
      },
      // ... players 1-3 similar
    ],
    currentPlayer,
    dealer,
    // ... rest of standard state initialization
  };
}
```

### Test Helper Functions

```typescript
// src/tests/helpers/testGames.ts

import { ALL_HANDSETS, type HandSetName, type HandSet } from '../fixtures/hands';

/**
 * Create initial state from a named handset.
 */
export function createStateFromHandSet(
  handSetName: HandSetName,
  config?: { enabledRuleSets?: string[] }
): { state: GameState; ctx: ExecutionContext; handSet: HandSet } {
  const handSet = ALL_HANDSETS[handSetName];
  const ctx = createExecutionContext({
    playerTypes: ['human', 'human', 'human', 'human'],
    enabledRuleSets: config?.enabledRuleSets ?? []
  });

  const state = createInitialStateWithHands({
    hands: handSet.hands,
    dealer: handSet.dealer
  });

  return { state, ctx, handSet };
}

/**
 * Play a sequence of actions from a handset.
 */
export function playActionsFromHandSet(
  handSetName: HandSetName,
  actions: GameAction[],
  config?: { enabledRuleSets?: string[] }
): { state: GameState; ctx: ExecutionContext } {
  const { state: initialState, ctx } = createStateFromHandSet(handSetName, config);

  const finalState = actions.reduce(
    (s, action) => executeAction(s, action, ctx.rules),
    initialState
  );

  return { state: finalState, ctx };
}
```

## Benefits

### For Test Authors

1. **No seed hunting** - Just pick the handset that fits your scenario
2. **Readable tests** - `createStateFromHandSet('NELLO_FAVORABLE')` is self-documenting
3. **Quick setup** - 1 line instead of 50 lines of state construction
4. **Type safety** - Autocomplete shows available handsets
5. **Shared vocabulary** - "Use the nello-favorable hands" is clear communication

### For Test Maintenance

1. **Easy to update** - Change handset once, all tests update
2. **Easy to verify** - Handset definition is the source of truth
3. **Easy to debug** - Print `handSet.hands[0]` to see exact dominoes
4. **Easy to extend** - Add new handset when needed, doesn't affect existing tests

### For Architecture

1. **Event-sourced** - Aligns with core architecture principle
2. **Deterministic** - Same handset = same initial state always
3. **Composable** - Combine handsets with action sequences
4. **Future-ready** - Foundation for E2E fixtures and UI-based test generation

## Future Extensions

### E2E Integration (when URL system is fixed)

```typescript
test('nello authorization', async ({ page }) => {
  // Load handset directly in browser
  await page.goto('/game?handset=NELLO_FAVORABLE&enabledRuleSets=nello');

  // Play actions to specific point
  await page.goto('/game?handset=NELLO_FAVORABLE&playTo=trump-selected');

  // Now test UI behavior at that exact game state
});
```

### UI Export (future)

```typescript
// After playing a game in UI, export as handset
function exportAsHandSet() {
  return {
    name: prompt("Handset name:"),
    description: prompt("Description:"),
    dealer: currentState.dealer,
    hands: currentState.players.map(p => p.hand.map(d => d.id))
  };
}
```

### Markers (future enhancement)

Add named markers to handsets for common points in gameplay:

```typescript
export const NELLO_FAVORABLE_WITH_MARKERS: HandSet = {
  name: 'NELLO_FAVORABLE',
  description: 'Player 0 has all doubles - ideal nello hand',
  dealer: 3,
  hands: [...],
  markers: {
    'post-bidding': [
      { type: 'bid', player: 0, bid: 'marks', value: 1 },
      { type: 'pass', player: 1 },
      { type: 'pass', player: 2 },
      { type: 'pass', player: 3 },
    ],
    'trump-selected': [
      /* post-bidding actions */,
      { type: 'select-trump', player: 0, trump: { type: 'nello' } },
    ],
  }
};

// Usage
const { state } = playActionsFromHandSet('NELLO_FAVORABLE',
  NELLO_FAVORABLE_WITH_MARKERS.markers['trump-selected']
);
```

## Migration Guide

### Before (manual state construction)

```typescript
it('should complete when bidder loses all 7 tricks', async () => {
  const ctx = createTestContext();
  let state = createInitialState();

  // 100 lines of manual setup
  state.players[0].hand = [/* manually construct hand */];
  state.phase = 'bidding';
  state.currentPlayer = 0;
  // ... 97 more lines

  // Play through entire game
  const bidTransitions = getNextStates(state, ctx);
  // ... complex test logic
});
```

### After (using handsets)

```typescript
it('should complete when bidder loses all 7 tricks', () => {
  const { state: finalState } = playActionsFromHandSet(
    'NELLO_FAVORABLE',
    NELLO_SUCCESS_ACTIONS,  // Pre-defined action sequence
    { enabledRuleSets: ['nello'] }
  );

  expect(finalState.phase).toBe('scoring');
  expect(finalState.tricks).toHaveLength(7);
});
```

## Adding New Handsets

Only add when:
1. ✅ No existing handset covers the scenario
2. ✅ You've tried replaying actions from existing handsets
3. ✅ The new scenario is reusable (not one-off)
4. ✅ You can clearly document the purpose

Process:
1. Create the handset in `hands.ts`
2. Document purpose, characteristics, and use cases
3. Add to `ALL_HANDSETS` registry
4. Verify all 28 dominoes are present and unique
5. Write at least one test using the new handset

Example PR description:
```
Add AUTHORIZATION_PHASES handset

Why: Need to test authorization across all game phases.
     BALANCED_STANDARD progresses unpredictably (players might all pass).

Characteristics: Hands designed so player 1 outbids player 0,
                 enabling clean progression through all phases.

Use cases: Phase transition tests, multi-phase authorization
```
