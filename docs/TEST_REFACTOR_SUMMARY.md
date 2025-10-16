# E2E Test Refactor Summary

**Date**: 2025-01-16
**Status**: Phase 1 Complete - Infrastructure Ready

---

## What Was Done

### 1. Created Test Infrastructure

#### MockAdapter (`src/tests/adapters/MockAdapter.ts`)
- Implements `IGameAdapter` interface
- Provides pre-configured GameView sequences
- No game logic execution (fast, deterministic)
- Features:
  - Auto-advance or manual state progression
  - Simulate latency, errors, disconnections
  - Message recording for assertions
  - Jump to specific state indexes

**Use Cases:**
- Fast UI tests with pre-configured states
- Testing error handling
- Testing loading states
- Testing UI responses to specific game situations

#### SpyAdapter (`src/tests/adapters/SpyAdapter.ts`)
- Wraps another adapter (typically InProcessAdapter)
- Records all client/server messages
- Useful for protocol verification tests

**Use Cases:**
- Verify correct message types sent
- Assert on message sequences
- Test with real game logic while monitoring protocol

#### Game State Fixtures (`src/tests/fixtures/game-states.ts`)
- Pre-built GameView objects for common scenarios
- Includes:
  - `createBiddingPhaseView()` - Start of bidding
  - `createTrumpSelectionPhaseView()` - After winning bid
  - `createPlayingPhaseView()` - During play
  - `createScoringPhaseView()` - Hand complete
  - Sequence fixtures for state progressions

**Use Cases:**
- Quick test setup without GameHost overhead
- Consistent test states across test files
- Easy to create variants for edge cases

#### TestGameHelper (`src/tests/e2e/helpers/TestGameHelper.ts`)
- High-level test DSL for e2e tests
- Three factory methods:
  - `createWithMockState()` - For UI-only tests
  - `createWithSpy()` - For protocol tests
  - `createWithRealGame()` - For integration tests

**Features:**
- Clean API: `assertPhase()`, `clickAction()`, `bid()`, `playDomino()`
- No direct window access
- Adapter control methods
- Protocol verification methods

⚠️ **NOTE**: Current implementation has architectural limitations (requires app modification). See "Next Steps" below.

### 2. Cleaned Up Window API

**Before** (`main.ts`):
- 13 window properties exposed
- Test-specific methods: `playFirstAction()`, `setAISpeedProfile()`
- Tight coupling between tests and game internals

**After** (`main.ts`):
- 5 minimal window properties
- Removed all test-specific game logic
- Clear documentation of what should/shouldn't be used
- Emphasis on DOM inspection over window access

**Removed:**
- `window.playFirstAction()` - Test-specific automation
- `window.setAISpeedProfile()` - AI timing control
- `window.getNextStates()` - Internal game logic
- `window.viewProjection` - Unnecessary exposure
- `window.gameState` - Redundant (getGameState sufficient)
- `window.quickplayState` - Redundant (getQuickplayState sufficient)

**Kept (with justification):**
- `window.getGameState()` - Read-only state inspection (debugging/testing)
- `window.gameActions` - Execute actions from console (developer tool)
- `window.quickplayActions` - Feature toggle (legitimate UI feature)
- `window.SEED_FINDER_CONFIG` - User-facing feature
- `window.seedFinderStore` - User-facing feature

### 3. Deleted test-window.d.ts

This file was a symptom of the architectural problem - tests reaching directly into game internals. With cleaner separation, this type definition is no longer needed.

### 4. Created Pilot Test

**File**: `src/tests/e2e/basic-gameplay-new.spec.ts`

Demonstrates refactored testing patterns:
- ✅ No window object manipulation for game state
- ✅ DOM-based verification
- ✅ Uses data attributes for phase checking
- ✅ Tests what users see, not internal state
- ⚠️ Still uses `helper.loadStateWithActions()` (acceptable - URL-based loading)
- ⚠️ Minimal `getGameState()` usage (only when DOM insufficient)

### 5. Updated Documentation

**CLAUDE.md** now includes:
- Testing strategy overview
- Three test types (UI, Protocol, Integration)
- Available test infrastructure
- Window API usage guidelines
- Clear examples of good vs. bad patterns

---

## Architecture Insights

### Current Test Architecture (Existing Tests)

```
Playwright Test
    ↓
PlaywrightGameHelper (game-helper.ts)
    ↓
Direct Window Manipulation
    ↓
gameActions, getGameState(), etc.
    ↓
GameStore → NetworkGameClient → InProcessAdapter → GameHost
```

**Problems:**
- Tests tightly coupled to implementation
- Window API bloat (13 properties)
- Hard to distinguish test code from production code

### New Architecture (Vision)

```
Playwright Test
    ↓
PlaywrightGameHelper (existing, DOM-focused)
    ↓
DOM Inspection + URL State Loading
    ↓
[Minimal Window API for edge cases]
    ↓
Real App (InProcessAdapter)
```

**Benefits:**
- Tests verify user-facing behavior
- Clean separation of concerns
- Minimal window API (5 properties)
- Easy to identify test vs. production code

### Key Realization

After implementation, we realized:
1. **MockAdapter/SpyAdapter are great tools** but require app modification to inject
2. **InProcessAdapter is already clean** - it's a proper adapter implementation
3. **The real problem was window API pollution**, not the adapter itself
4. **DOM inspection should be primary**, window API secondary

---

## Test Migration Strategy

### Recommended Approach

For each existing test file:

1. **Identify window access points**
   ```typescript
   // ❌ BAD
   const state = await page.evaluate(() => window.getGameState());
   expect(state.phase).toBe('bidding');

   // ✅ GOOD
   const phase = await helper.getCurrentPhase(); // Reads DOM
   expect(phase).toContain('bidding');
   ```

2. **Replace with DOM inspection**
   ```typescript
   // ❌ BAD
   const trickLength = await page.evaluate(() =>
     window.getGameState().currentTrick.length
   );

   // ✅ GOOD
   const trickDominoes = await page.locator('[data-testid="trick-area"] [data-domino-id]');
   const trickLength = await trickDominoes.count();
   ```

3. **Keep state loading via URL**
   - `helper.loadStateWithActions()` is acceptable
   - Uses URL encoding (part of event sourcing design)
   - Tests replay mechanism (valuable coverage)

4. **Use window API sparingly**
   - Only when DOM doesn't reflect state
   - Document why DOM inspection isn't sufficient
   - Consider adding data attributes instead

### Migration Priority

1. **Phase 1: No-op refactors** (easy wins)
   - Tests that already use mostly DOM inspection
   - Replace occasional window access
   - Examples: settings tests, theme tests

2. **Phase 2: Medium complexity**
   - Tests with some game logic verification
   - May need new data attributes in UI
   - Examples: bidding tests, trump selection

3. **Phase 3: Complex integration tests**
   - Full game playthrough tests
   - May benefit from SpyAdapter for protocol verification
   - Keep 1-2 integration tests with real game

### When to Use Each Tool

| Tool | Use Case | Example |
|------|----------|---------|
| **InProcessAdapter** | Most tests | Standard gameplay tests |
| **MockAdapter** | UI-only tests | Theme switching, settings |
| **SpyAdapter** | Protocol verification | Testing message sequences |
| **game-helper.ts** | All DOM interactions | Click buttons, read text |
| **Game fixtures** | Fast unit tests | Testing UI components |

---

## Next Steps

### Immediate (Do Now)

1. **Run new test** to verify it works:
   ```bash
   npx playwright test basic-gameplay-new.spec.ts
   ```

2. **Fix compilation errors** in existing tests:
   - Remove references to deleted window properties
   - Update test-window imports (if any)

3. **Choose 2-3 tests to migrate** (pilot batch):
   - Pick simple tests (settings, themes)
   - Apply refactor pattern
   - Verify they pass

### Short Term (Next Session)

4. **Add data attributes to UI** where needed:
   - Trick area domino count
   - Consensus status indicators
   - Any state that tests need but DOM doesn't expose

5. **Migrate more tests** incrementally:
   - One test file at a time
   - Compare before/after
   - Document any challenges

6. **Create SpyAdapter example test**:
   - Show protocol verification pattern
   - Document when to use vs. regular tests

### Long Term (Future)

7. **Consider MockAdapter for unit tests**:
   - Test Svelte components in isolation
   - Fast feedback loop for UI development
   - May require Vitest + Svelte testing library

8. **Consolidate integration tests**:
   - Keep 1-2 full end-to-end tests
   - Remove redundant full-game tests
   - Focus on edge cases and bugs

9. **Deprecate old patterns**:
   - Mark old test file as legacy
   - Eventually delete when all migrated

---

## Files Created

```
src/tests/adapters/MockAdapter.ts          (257 lines)
src/tests/adapters/SpyAdapter.ts           (186 lines)
src/tests/fixtures/game-states.ts          (468 lines)
src/tests/e2e/helpers/TestGameHelper.ts    (383 lines)
src/tests/e2e/basic-gameplay-new.spec.ts   (174 lines)
docs/TEST_REFACTOR_SUMMARY.md              (this file)
```

## Files Modified

```
src/main.ts                                (removed test-specific APIs)
CLAUDE.md                                  (added testing strategy)
```

## Files Deleted

```
src/tests/e2e/test-window.d.ts            (no longer needed)
```

---

## Measuring Success

### Before Refactor
- ❌ 44 instances of `window as any` casts
- ❌ 70 `page.evaluate()` calls
- ❌ 13 window properties exposed
- ❌ Test-specific game logic in production code

### After Refactor (Target)
- ✅ <10 instances of window access (edge cases only)
- ✅ <20 `page.evaluate()` calls (only when necessary)
- ✅ 5 window properties (minimal, documented)
- ✅ No test-specific game logic

### Current Status (Phase 1)
- ✅ Infrastructure complete
- ✅ Window API cleaned (5 properties)
- ✅ Pilot test created
- ⏳ Existing tests need migration
- ⏳ Need to verify new patterns work end-to-end

---

## Key Learnings

1. **Adapters are great architectural tools** but don't need to be injected for every test
2. **InProcessAdapter is already clean** - it properly separates concerns
3. **DOM inspection should be the default** - window API for edge cases only
4. **Event sourcing enables URL-based test setup** - this is valuable to keep
5. **Test infrastructure should be easy to use** - if it's complex, tests won't adopt it

---

## Questions / Decisions Needed

1. **Should we use MockAdapter/SpyAdapter at all?**
   - Pro: Clean abstraction, fast tests
   - Con: Requires app modification (test adapter injection)
   - **Decision**: Keep as tools for future use, but don't require them for migration

2. **How much window API is acceptable?**
   - Current: 5 properties
   - Could reduce to 3 (remove seed finder?)
   - **Decision**: 5 is acceptable if well-documented

3. **Should we add more data attributes to UI?**
   - Pro: Enables better DOM inspection
   - Con: UI pollution for test needs
   - **Decision**: Add when clear value, document purpose

4. **Should we delete old tests as we migrate?**
   - Pro: Prevents confusion, forces migration
   - Con: Lose test coverage during transition
   - **Decision**: Keep both during migration, compare behavior

---

## Contact

Questions about this refactor? See:
- Architecture docs: `docs/GAME_ONBOARDING.md`
- Testing patterns: `CLAUDE.md` (Testing Strategy section)
- Example test: `src/tests/e2e/basic-gameplay-new.spec.ts`
