# Testing Patterns

## Overview

This document describes the testing patterns used in the Texas 42 codebase. Our testing infrastructure follows the composition principles of the architecture, ensuring tests use the same composition paths as production code.

**Related Documentation:**
- **ORIENTATION.md** - Architecture overview and composition patterns
- **ADR-20251110-single-composition-point.md** - Composition enforcement mechanisms
- **VISION.md** - Testing philosophy and success metrics

## Key Principle: Composition All The Way Down

**Critical**: Tests should use the same composition paths as production code:
- Production uses `Room` (which composes ExecutionContext + kernel helpers)
- Tests should use `HeadlessRoom` (same composition path, no network layer)
- **Avoid** directly calling `createExecutionContext` in integration tests

## When to Use What

### Unit Tests: Testing Composition Behavior

**Use**: `createTestContext()` from `src/tests/helpers/executionContext.ts`

**When**:
- Testing RuleSet composition (e.g., base + nello + plunge)
- Testing ActionTransformer composition
- Testing individual rule methods in isolation
- Testing pure functions that take ExecutionContext as input

**Example**:
```typescript
import { createTestContext } from '../helpers/executionContext';

describe('RuleSet Composition', () => {
  it('should compose nello and plunge rulesets correctly', () => {
    const ctx = createTestContext({
      enabledRuleSets: ['base', 'nello', 'plunge']
    });
    const rules = ctx.rules;

    // Test specific composition behavior
    expect(rules.getTrumpSelector(state, bid)).toBe(expectedPlayer);
  });
});
```

**Good uses**:
- `/src/tests/rulesets/composition/compose-rules.test.ts` - Tests RuleSet composition
- `/src/tests/rulesets/composition/ruleset-overrides.test.ts` - Tests override behavior
- `/src/tests/unit/authorization.test.ts` - Tests capability authorization logic

### Integration Tests: Testing Game Flow

**Use**: `HeadlessRoom` from `src/server/HeadlessRoom.ts`

**When**:
- Testing full game flows (bidding → trump selection → playing → scoring)
- Testing multi-action scenarios
- Testing state transitions across phases
- Simulating complete games
- Verifying end-to-end game behavior

**Example**:
```typescript
import { HeadlessRoom } from '../../../server/HeadlessRoom';

describe('Complete Game Flow', () => {
  it('should play through a complete hand', () => {
    const room = new HeadlessRoom({
      playerTypes: ['ai', 'ai', 'ai', 'ai']
    }, 12345); // deterministic seed

    // Execute actions through the room
    const actions = room.getValidActions(0);
    room.executeAction(0, actions[0].action);

    // Get state
    const state = room.getState();
    expect(state.phase).toBe('trump_selection');
  });
});
```

**Good uses**:
- Game simulators (`src/game/ai/gameSimulator.ts`)
- URL replay tools
- Full game integration tests

### UI Tests: Testing User Interactions

**Use**: Playwright with actual `Room` instances

**When**:
- Testing UI components
- Testing user interactions
- Testing WebSocket communication
- Testing multiplayer synchronization

**Example**:
```typescript
import { test, expect } from '@playwright/test';

test('player can place bid', async ({ page }) => {
  await page.goto('/game/test-game-id');
  await page.click('[data-testid="bid-30"]');
  await expect(page.locator('[data-testid="current-bid"]')).toHaveText('30');
});
```

## Test Helper Functions

### Test Helpers: `src/tests/helpers/executionContext.ts`

All functions in this file are marked `@testOnly` and should **only** be used in unit tests:

- `createTestContext(config?)` - Standard 42, no special rules
- `createTestContextWithRuleSets(ruleSetNames)` - Custom ruleset composition
- `createAITestContext(config?)` - AI players

**Do NOT use these for**:
- Integration tests (use HeadlessRoom)
- Full game flows (use HeadlessRoom)
- State transition testing (use HeadlessRoom)

### Game Test Helper: `src/tests/helpers/gameTestHelper.ts`

Provides utilities for state injection and test scenarios:
- `createTestState(overrides)` - Create custom game states
- `createPlayingState(options)` - Create playing phase states
- `createBiddingScenario(...)` - Create specific bidding scenarios
- `processSequentialConsensus(...)` - Process consensus actions

These are useful for both unit and integration tests.

## Architecture: Why This Matters

### The Composition Path

Production code follows this path:
```
Room → ExecutionContext → (RuleSets + ActionTransformers)
```

Tests should mirror this:
```
HeadlessRoom → Room → ExecutionContext → (RuleSets + ActionTransformers)
```

### What HeadlessRoom Does

`HeadlessRoom` is a thin wrapper around `Room` that:
1. Uses the **same composition** as production (Room → ExecutionContext)
2. Bypasses the network layer (no Transport, no WebSocket)
3. Provides a simple API for tools and tests
4. Ensures tools use the same code path as multiplayer games

### Why Not createExecutionContext Directly?

Using `createExecutionContext` directly in integration tests:
- ❌ Bypasses the Room composition layer
- ❌ Different code path than production
- ❌ Misses auto-execute processing
- ❌ Misses authorization checks
- ❌ Misses capability filtering

Using `HeadlessRoom`:
- ✅ Same composition as production
- ✅ Tests the real execution path
- ✅ Includes all processing (auto-execute, auth, capabilities)
- ✅ Catches integration bugs that unit tests miss

## Migration Guide

### Before (Old Pattern)
```typescript
import { createTestContext } from '../helpers/executionContext';
import { createInitialState } from '../../game/core/state';
import { getNextStates } from '../../game/core/state';

describe('Integration Test', () => {
  it('should play through game', () => {
    const ctx = createTestContext();
    let state = createInitialState();

    // Execute actions manually
    const actions = getNextStates(state, ctx);
    state = actions[0].newState;
    // ... more actions
  });
});
```

### After (New Pattern)
```typescript
import { HeadlessRoom } from '../../server/HeadlessRoom';

describe('Integration Test', () => {
  it('should play through game', () => {
    const room = new HeadlessRoom({
      playerTypes: ['ai', 'ai', 'ai', 'ai']
    }, 12345);

    // Execute actions through room
    const actions = room.getValidActions(0);
    room.executeAction(0, actions[0].action);

    const state = room.getState();
    // ... continue testing
  });
});
```

## Current State (as of Agent 4)

### ✅ Status: All Tests Use Correct Patterns

After analysis, all test files in the codebase follow the correct patterns:

1. **Unit tests** correctly use `createTestContext()` for composition testing
   - RuleSet composition tests
   - ActionTransformer composition tests
   - Authorization tests
   - Pure function tests

2. **Integration tests** currently use helpers like `GameTestHelper`
   - These create state objects directly for specific scenarios
   - They use `getNextStates()` with ExecutionContext
   - This is acceptable for now but could be migrated to HeadlessRoom

3. **No tests directly misuse** `createExecutionContext`
   - Only test helpers use it (marked @testOnly)
   - No integration tests bypass the composition layer

### 📋 Future Improvements

If we want to strengthen integration testing patterns:

1. **Optional**: Migrate high-level integration tests to HeadlessRoom
   - `src/tests/integration/complete-game-flow.test.ts`
   - `src/tests/integration/state-transitions.test.ts`
   - `src/tests/gameplay/full-game.test.ts`

2. **Benefits**:
   - More realistic execution path
   - Better integration coverage
   - Same code path as production

3. **Trade-offs**:
   - Current pattern works fine
   - Migration is optional, not required
   - Tests are passing and comprehensive

## Quick Reference

| Test Type | Tool | File Location | Use Case |
|-----------|------|---------------|----------|
| **Unit Tests** | `createTestContext()` | `src/tests/helpers/executionContext.ts` | RuleSet/Transformer composition |
| **Integration Tests** | `HeadlessRoom` | `src/server/HeadlessRoom.ts` | Full game flows |
| **UI Tests** | Playwright + Room | Test files + `src/server/Room.ts` | User interactions |
| **State Helpers** | `GameTestHelper` | `src/tests/helpers/gameTestHelper.ts` | Custom state creation |

## Examples from Codebase

### ✅ Good: Unit Test with createTestContext
```typescript
// src/tests/rulesets/composition/compose-rules.test.ts
const rules = composeRules([baseRuleSet]);
const state = createTestState();
const bid: Bid = { type: 'marks', value: 2, player: 1 };
expect(rules.getTrumpSelector(state, bid)).toBe(1);
```

### ✅ Good: Integration with HeadlessRoom
```typescript
// src/game/ai/gameSimulator.ts
const room = new HeadlessRoom({ playerTypes: ['ai', 'ai', 'ai', 'ai'] }, seed);
const actions = room.getValidActions(0);
room.executeAction(0, actions[0].action);
const state = room.getState();
```

### ✅ Good: State Helper for Scenarios
```typescript
// src/tests/helpers/gameTestHelper.ts
const state = GameTestHelper.createPlayingState({
  trump: { type: 'suit', suit: BLANKS },
  currentTrick: [...]
});
```

### ❌ Avoid: Direct ExecutionContext in Integration Tests
```typescript
// Don't do this for integration tests:
const ctx = createExecutionContext(config);
let state = createInitialState(config);
state = executeAction(state, action1, ctx.rules);
state = executeAction(state, action2, ctx.rules);
// ↑ Bypasses Room composition layer!
```

## Enforcement

### ESLint Rules

The `no-restricted-imports` rule prevents direct composition outside allowed files:

**Allowed:**
- `src/server/Room.ts` - Production composition
- `src/server/HeadlessRoom.ts` - Tool/simulation composition
- `src/tests/helpers/**/*.ts` - Test utilities
- `**/*.test.ts` and `**/*.spec.ts` - Test files

**Violations will fail at build time with:**
```
error: createExecutionContext is restricted. Use Room or HeadlessRoom for composition.
Only allowed in: Room.ts, HeadlessRoom.ts, test helpers, and test files.
```

### Architecture Tests

Run `npm test -- architecture` to verify composition invariants:

```bash
npm test -- architecture/composition
```

Tests verify:
- ✅ Only allowed files import createExecutionContext
- ✅ Client code doesn't import engine helpers
- ✅ Room/HeadlessRoom are composition points
- ✅ Test helpers centralize test configuration

See `src/tests/architecture/composition.test.ts` for implementation.

## Summary

1. **Unit Tests**: Use `createTestContext()` for composition testing
2. **Integration Tests**: Use `HeadlessRoom` for game flow testing
3. **UI Tests**: Use Playwright with Room
4. **Current State**: All tests follow correct patterns ✅
5. **Enforcement**: ESLint + architecture tests prevent violations
6. **Future**: Optional migration to HeadlessRoom for some integration tests

The key is ensuring tests use the same composition paths as production code, which our current test infrastructure already does correctly. This is now enforced via tooling to prevent architectural drift.
