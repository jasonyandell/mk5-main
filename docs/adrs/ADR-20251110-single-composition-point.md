# ADR-20251110: Single Composition Point for Execution Contexts

**Status**: Implemented
**Date**: 2025-11-10
**Deciders**: Architecture Review

## Context

The Texas 42 architecture uses a two-level composition system where RuleSets (execution semantics) and ActionTransformers (action transformation) combine to create game configurations. This composition creates an ExecutionContext that contains:

1. **rulesets**: Array of enabled GameRuleSets
2. **rules**: Composed GameRules interface (18 methods)
3. **getValidActions**: Composed state machine with applied ActionTransformers

### The Problem

As the codebase evolved, ExecutionContext composition spread across multiple files:

**Violations Found:**
- `src/game/multiplayer/AIClient.ts` - Created own execution context
- `src/game/core/playerView.ts` - Created own execution context
- `src/game/ai/gameSimulator.ts` - Created own execution context
- `src/game/utils/urlReplay.ts` - Created own execution context

This violated architectural invariant #4: **"Single Composition Point: Room constructor is ONLY place layers/action-transformers compose"**

### Why This Matters

**1. Configuration Drift**
When composition happens in multiple places, different parts of the system can have inconsistent configurations:
```typescript
// Room composes with tournament rules
const roomCtx = createExecutionContext({ actionTransformers: [{ type: 'tournament' }] });

// But AIClient creates its own context with different rules
const aiCtx = createExecutionContext({ playerTypes: ['ai', 'ai', 'ai', 'ai'] });
```

**2. Testing Complexity**
Multiple composition points make it unclear which configuration is being tested. Tests might pass with one composition but fail with another.

**3. Maintenance Burden**
When adding new RuleSets or ActionTransformers, changes must be coordinated across multiple composition points.

**4. Architectural Integrity**
The architecture assumes pure helpers receive ExecutionContext - they don't create it. When helpers compose their own contexts, they become impure.

## Decision

**Enforce single composition point via ESLint rules and architecture tests.**

### Legitimate Uses of createExecutionContext

Only THREE composition points are allowed:

1. **`src/server/Room.ts`** - Production server composition
   - Creates ExecutionContext from GameConfig
   - Composes rulesets and action transformers
   - Used by all pure helpers throughout request lifecycle

2. **`src/server/HeadlessRoom.ts`** - Tool/simulation composition (future)
   - Lightweight wrapper for deterministic execution
   - Used by tools, scripts, and simulations
   - No transport layer, direct API

3. **`src/tests/helpers/executionContext.ts`** - Test utilities
   - Factory functions for test execution contexts
   - Provides `createTestContext()`, `createTestContextWithRuleSets()`, etc.
   - Centralizes test configuration

4. **Test files directly** - For composition testing only
   - Tests that verify composition behavior
   - Example: Testing that RuleSets override correctly
   - Should be minimal and focused

### Files Fixed

All violations were refactored to use composition points:

**AIClient**: Now receives filtered actions from server via GameView.transitions
- **Before**: Created ExecutionContext, called `getNextStates()`, filtered locally
- **After**: Trusts server's filtered actions completely

**playerView**: Deprecated in favor of multiplayer system
- **Before**: Created ExecutionContext to compute transitions
- **After**: Multiplayer system uses Room's ExecutionContext

**gameSimulator**: Should use HeadlessRoom (pending Agent 3)
- **Before**: Created ExecutionContext directly
- **After**: Will use HeadlessRoom API with proper composition

**urlReplay**: Should use HeadlessRoom (pending Agent 3)
- **Before**: Created ExecutionContext directly
- **After**: Will use HeadlessRoom API with proper composition

## Consequences

### Positive

✅ **Configuration consistency** - All game logic uses same composed rules
✅ **Clear ownership** - Room (or HeadlessRoom) owns composition
✅ **Simpler helpers** - Pure helpers receive context, never create it
✅ **Enforced at build time** - ESLint catches new violations immediately
✅ **Enforced at test time** - Architecture tests verify compliance
✅ **Better testability** - Tests use centralized test helpers
✅ **Clearer documentation** - Explicit list of allowed composition points

### Neutral

- HeadlessRoom API provides composition for tools/scripts (future improvement)
- Test helpers centralize test configuration (improved organization)
- Some code needs migration (one-time refactor cost)

### Trade-offs

**Breaking Changes:**
- Files that created ExecutionContext directly must refactor
- Tools/scripts need to migrate to HeadlessRoom API (when available)

**Mitigation:**
- ESLint provides clear error messages pointing to allowed alternatives
- Architecture tests document the invariant
- This ADR explains the rationale and migration path

## Enforcement

### ESLint Rule

Added to `eslint.config.js`:

```javascript
{
  files: ['**/*.ts'],
  rules: {
    'no-restricted-imports': ['error', {
      patterns: [{
        group: ['*/types/execution', '../types/execution', '../../types/execution', '../../../types/execution'],
        importNames: ['createExecutionContext'],
        message: 'createExecutionContext is restricted. Use Room or HeadlessRoom for composition. Only allowed in: Room.ts, HeadlessRoom.ts, test helpers (src/tests/helpers/), and test files (*.test.ts, *.spec.ts).'
      }]
    }]
  }
},
{
  files: [
    'src/server/Room.ts',
    'src/server/HeadlessRoom.ts',
    'src/tests/helpers/**/*.ts',
    '**/*.test.ts',
    '**/*.spec.ts'
  ],
  rules: {
    'no-restricted-imports': 'off'
  }
}
```

### Architecture Tests

Created `src/tests/architecture/composition.test.ts`:

Tests verify:
1. Only allowed files import `createExecutionContext`
2. Client code doesn't import engine helpers
3. Composition happens only at designated points

### Documentation

- This ADR documents the invariant and rationale
- ORIENTATION.md explains the single composition point pattern
- VISION.md includes enforcement as a success metric
- TESTING_PATTERNS.md guides when to use each approach

## Migration Path

### For New Code

Use the appropriate composition point:

**Production game instances:**
```typescript
import { Room } from '@/server/Room';
const room = new Room(gameId, config, players);
```

**Tools, scripts, simulations:**
```typescript
import { HeadlessRoom } from '@/server/HeadlessRoom';
const room = new HeadlessRoom(config, seed);
room.executeAction(playerId, action);
```

**Unit tests:**
```typescript
import { createTestContext } from '@/tests/helpers/executionContext';
const ctx = createTestContext({ enableNello: true });
```

### For Existing Violations

**AIClient** - Already fixed, uses server's filtered actions

**playerView** - Deprecated, use multiplayer system

**gameSimulator** - Migrate to HeadlessRoom:
```typescript
// Before
const ctx = createExecutionContext({ playerTypes: ['ai', 'ai', 'ai', 'ai'] });
const actions = getNextStates(state, ctx);

// After
const room = new HeadlessRoom(config, seed);
const view = room.getView(playerId);
const actions = view.transitions;
```

**urlReplay** - Migrate to HeadlessRoom:
```typescript
// Before
const ctx = createExecutionContext({ playerTypes: ['human', 'human', 'human', 'human'] });
let state = createInitialState({ shuffleSeed: seed });
for (const action of actions) {
  state = executeAction(state, action, ctx.rules);
}

// After
const room = new HeadlessRoom(config, seed);
for (const action of actions) {
  room.executeAction(playerId, action);
}
const state = room.getState();
```

## Alternatives Considered

### Alternative 1: Allow Multiple Composition Points

**Rejected**: Leads to configuration drift and testing complexity. The cost of maintaining consistency across multiple compositions outweighs convenience.

### Alternative 2: Global ExecutionContext Singleton

**Rejected**: Prevents testing with different configurations. Makes it impossible to run multiple games with different rules simultaneously.

### Alternative 3: Pass Config Everywhere Instead of Context

**Rejected**: Every helper would need to recompose from config on every call. ExecutionContext is immutable and can be safely shared across all pure functions.

## Success Metrics

- ✅ ESLint catches violations at build time
- ✅ Architecture tests verify compliance
- ✅ Zero TypeScript compilation errors
- ✅ All existing tests pass
- ✅ gameSimulator migrated to HeadlessRoom (verified compliant 2025-01-11)
- ✅ urlReplay migrated to HeadlessRoom (completed 2025-01-11, see ADR-20251111)

## Implementation Notes

### Why Room AND HeadlessRoom?

**Room** is full-featured orchestrator:
- Manages multiplayer sessions
- Handles AI lifecycle
- Routes protocol messages
- Broadcasts filtered views
- Coordinates transport layer

**HeadlessRoom** is lightweight execution engine:
- No transport layer
- No session management
- Direct API for tools/scripts
- Deterministic execution only
- Used by simulations and replays

Both compose ExecutionContext the same way, but serve different use cases.

### Test Helper Pattern

Tests should use centralized helpers:

```typescript
// Good - uses test helper
import { createTestContext } from '@/tests/helpers/executionContext';
const ctx = createTestContext({ enableNello: true });

// Bad - creates context directly (ESLint error)
import { createExecutionContext } from '@/game/types/execution';
const ctx = createExecutionContext({ enableNello: true });
```

This centralizes test configuration and makes tests more maintainable.

## Future Work

1. **Complete HeadlessRoom implementation** - Agent 3's task
2. **Migrate gameSimulator** - Use HeadlessRoom instead of direct composition
3. **Migrate urlReplay** - Use HeadlessRoom instead of direct composition
4. **Stricter enforcement** - Consider making createExecutionContext private
5. **Documentation** - Add HeadlessRoom usage guide to ORIENTATION.md

## References

- **ORIENTATION.md** - "The Composition Point (Room Constructor)" section
- **VISION.md** - Non-Negotiable Principles (Single composition point)
- **ADR-20251110-remove-protocol-leaks.md** - Related architectural cleanup
- **src/server/Room.ts** - Primary composition point implementation
- **src/tests/helpers/executionContext.ts** - Test utility composition point
- **eslint.config.js** - ESLint enforcement rules

## Decision Rationale

The single composition point invariant is fundamental to maintaining architectural integrity. It ensures:

1. **Consistency**: All parts of the system use the same game rules
2. **Testability**: Clear boundaries for what configuration is being tested
3. **Maintainability**: Changes to composition logic happen in one place
4. **Correctness**: Pure helpers receive context, they never create it

Enforcing this via tooling (ESLint + tests) makes violations impossible to merge, preventing architectural drift.

**Result**: The architecture is now correct by construction - multiple conflicting compositions CAN'T exist because the tooling prevents them.

---

**Status Update**: Enforcement mechanisms implemented, waiting on HeadlessRoom (Agent 3) to complete migration of gameSimulator and urlReplay.
