# E2E Test Modernization Plan (REVISED)

**Status**: Ready for execution (validated against actual codebase)
**Last Updated**: 2025-11-11
**Goal**: Fix all E2E tests properly - not just working, but GOOD tests aligned with architecture

---

## Problem Statement

### The Situation
After a major architecture refactor to event-sourced, pure functional design:
- **Current E2E status**: 7 passing, 8 failing, 69 skipped (84 total)
- **Target**: ~46 passing, 0 failing, 0 skipped
- **Issue**: Tests were written for a `window.gameActions` API that never existed in new architecture
- **Goal**: Modernize tests to use URL-based state + DOM verification + UI interaction patterns

### Why This Matters
Our E2E tests have been broken since the refactor. We need them not just working, but GOOD:
- Aligned with architectural principles (event sourcing, server authority, clean separation)
- Testing real user workflows (not internal APIs)
- Maintainable and clear patterns
- Zero technical debt or compatibility layers

---

## Architecture Context

### What Tests Expect (OLD - Doesn't Exist)
```typescript
window.gameActions = {
  updateTheme(theme, overrides) {...}  // NEVER EXISTED
  skipAIDelays() {...}                 // NEVER EXISTED
}
```

### What Actually Exists (NEW)
```typescript
// In main.ts (lines 36-77)
window.getGameView = () => GameView    // Read-only inspection (to be minimized)
window.game = game;                    // Dev console only (to be removed)

// In App.svelte
<div class="app-container" data-phase={$gameState.phase}>  // Line 138
```

### Core Principle (From VISION.md)
**Tests should use:**
1. **URL for state setup** - Event sourcing via replay
2. **DOM for verification** - Inspect rendered output
3. **UI interaction for actions** - Click buttons, not call functions

**Reference docs**:
- `docs/VISION.md` - Server authority, event sourcing, pure functions
- `docs/ORIENTATION.md` - Two-level composition, zero coupling, clean separation

---

## Testing Philosophy

### ✅ DO (Modern Pattern)

```typescript
// Setup: Load game state via URL
const url = helper.encodeGameUrl({ seed: 12345, actions: ['bid-30', 'pass', 'pass', 'pass'] });
await page.goto(url);

// Verify: Inspect DOM
await expect(page.locator('.badge')).toContainText('Trump');
const phase = await page.locator('.app-container').getAttribute('data-phase');
expect(phase).toBe('trump-selection');

// Act: User interaction
await page.click('[data-testid="trump-doubles"]');

// Wait: For DOM updates
await page.waitForSelector('[data-phase="playing"]');
```

### ❌ DON'T (Anti-patterns)

```typescript
// NO window API manipulation
window.gameActions.updateTheme(...);  // Doesn't exist, defeats architecture

// NO compat layers
window.gameActions = { wrapper... };  // Hides the problem, creates tech debt

// NO internal state inspection for game state
const state = window.getGameView();   // Wrong layer (use DOM instead)
```

---

## Current Test Status (Ground Truth)

### Test Files Inventory

| File | Status | Tests | Issue | Lines |
|------|--------|-------|-------|-------|
| `settings-color-panel.spec.ts` | 0/7 FAIL | 7 | `window.gameActions.updateTheme` | 302 |
| `history-navigation.spec.ts` | 0/1 FAIL | 1 | Timeout on debug panel | 87 |
| `perfects-page.spec.ts` | 2/2 PASS | 2 | None ✓ | 46 |
| `settings-color-basic.spec.ts` | 5/5 PASS | 5 | None ✓ | 136 |
| `basic-gameplay.spec.ts` | SKIP | 12 | `.skip` line 4 | 218 |
| `basic-gameplay-new.spec.ts` | SKIP | 11 | `.skip` line 16 | 218 |
| `url-state-management.spec.ts` | SKIP | 27 | `.skip` line 10 | 745 |
| `back-button-comprehensive.spec.ts` | SKIP | 10 | `.skip` line 9 | 494 |
| `ai-skip-functionality.spec.ts` | SKIP | 6 | `.skip` line 8 | 135 |
| `ai-back-button.spec.ts` | SKIP | 3 | `.skip` line 4 | 147 |

**Total**: 84 tests across 2,528 lines

### Available DOM Test Hooks (Verified in Code)

**Phase tracking** (App.svelte:138):
- `data-phase={$gameState.phase}` on `.app-container` ✓

**Theme tracking**:
- `data-theme` attribute on `<html>` element ✓

**Action buttons** (verified in components):
- `data-testid="settings-panel"` ✓
- `data-testid="action-panel"` ✓
- `data-testid="domino-{high}-{low}"` ✓ (generated)
- `data-testid="pass"` ✓
- `data-testid="trump-display"` ✓
- `data-testid="trick-table"` ✓ (dynamic)
- `data-testid="app-header"` ✓
- `data-testid="playing-area"` ✓

**Missing (needs to be added)**:
- `data-processing` attribute for isProcessing state

### URL Infrastructure (Verified Working)

**URL encoding** (url-compression.ts):
- `encodeGameUrl()` / `decodeGameUrl()` exist and work ✓
- Theme: `t=<theme-name>` (e.g., `t=forest`)
- Colors: `v=<compact-format>` (e.g., `v=op71.9,123,62` for OKLCH)
- Seed: `s=12345`
- Actions: `a=<compressed>` (e.g., `a=CAAA`)
- Test mode: `testMode=true` (disables AI delays)

**Helper utilities** (game-helper.ts, 856 lines):
- ✓ `encodeGameUrl()` - URL encoding
- ✓ `getCurrentPhase()` - Read phase from DOM
- ✓ `waitForGameReady()` - Wait for stable state
- ✓ `goto()` - Navigate with testMode
- ✓ `loadStateWithActions()` - Replay from URL
- ✓ DOM selector library (SELECTORS constant)
- ⚠️ Still uses `window.getGameView()` in 6+ locations (needs migration)

---

## Critical Findings from Investigation

### Finding 1: Theme Test Strategy in Original Plan is WRONG ❌

**Original plan suggested** (lines 96-109):
```typescript
const url = `/?theme=forest&colors=${encodeURIComponent(JSON.stringify({'--p': '100% 0 0'}))}`;
await page.goto(url);
```

**Why this is fundamentally flawed**:
1. Theme/color params use different URL format: `t=forest&v=op71.9,123,62` (not `theme=` and not JSON)
2. Navigating to a URL **loads** that theme, it doesn't **change** theme mid-session
3. Tests are verifying **theme change workflow** (UI interaction), not just URL loading
4. This approach tests nothing useful (just URL parsing, not user behavior)

**CORRECT approach** (DOM-driven):
```typescript
// Click theme button in UI (actual user workflow)
await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
await page.locator('.theme-colors-btn').click();
await page.locator('[data-theme="forest"]').click();

// Wait for URL to update (reactive effect)
await page.waitForFunction(() => window.location.href.includes('t=forest'));

// Verify URL updated correctly
expect(page.url()).toContain('t=forest');

// Verify DOM reflects change
const theme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
expect(theme).toBe('forest');
```

**Pattern**: UI interaction → reactive state update → URL update → DOM verification

### Finding 2: Consolidation Claims Are Overstated ⚠️

**Original plan claimed**: "100% redundant coverage between files"

**Reality after investigation**:
- `url-state-management.spec.ts` (27 tests): Comprehensive URL encoding/decoding, compression, error handling, deterministic replay, multi-hand games
- `back-button-comprehensive.spec.ts` (10 tests): Focused on browser navigation UX, history preservation, divergent timelines

**Overlap** (~30%):
- Both test "back button works"
- Both test "URL updates after actions"

**Unique to url-state-management** (~70%):
- URL compression efficiency
- Invalid action handling
- Deterministic replay verification
- Multi-hand game tracking
- Config persistence (playerTypes, dealer, target)

**Unique to back-button-comprehensive** (~70%):
- History preservation across divergent paths
- AI interaction with navigation
- Scoring phase navigation
- Multiple back/forward sequences

**Verdict**: ~30% redundant (not 100%). Should consolidate to ~18-20 tests (not 15).

### Finding 3: History Navigation Timeout - Unknown Root Cause ⚠️

**Test**: `history-navigation.spec.ts:5`
**Symptom**: `locator.click: Test timeout of 10000ms exceeded`
**Suspected line**: Line 22 (`helper.openDebugPanel()`)

**Hypotheses**:
1. Debug panel UI changed (selector `.dropdown-end .btn-ghost.btn-circle` may be stale)
2. Test timing issue (needs `waitForSelector` before click)
3. Network/rendering delay

**Recommended investigation**:
```typescript
// Add before opening debug panel
await page.pause(); // Visual inspection
await page.screenshot({ path: 'scratch/debug-panel-issue.png' });
```

### Finding 4: Skipped Tests May Have Hidden Issues ⚠️

**Original plan assumes**: Skipped tests will "just work" after unskipping.

**Concern**: If `basic-gameplay-new.spec.ts` was "designed for new architecture" and working, why skip it?

**Recommendation**: Unskip tests ONE FILE AT A TIME and fix issues incrementally (not in parallel).

### Finding 5: Task Sequencing Has Conflicts ❌

**Original plan**: Execute Task 4 (unskip tests using window API) and Task 5 (remove window API) in parallel.

**Problem**: Task 4 tests use `window.getGameView()` which Task 5 is removing → conflict.

**Solution**: Task 5 must come AFTER Task 4 (sequential, not parallel).

---

## Revised Execution Plan

### Phase 1: Quick Wins & Foundation (30 min)

**1A: Delete Obsolete Tests**
```bash
rm src/tests/e2e/ai-skip-functionality.spec.ts
```
- Deletes 6 tests testing non-existent `skipAIDelays` feature
- Remove `skipAIDelays` declaration from `src/types/global.d.ts`
- Result: 84 → 78 tests

**1B: Add DOM Test Hook (REQUIRED)**

Add to `App.svelte` (around line 138):
```svelte
<div
  class="app-container"
  data-phase={$gameState.phase}
  data-processing={$viewProjection.ui.isProcessing}
>
```

**Why critical**: Enables DOM-only testing of processing state (eliminates need for window API).

---

### Phase 2: Investigation & Fixes (4-5 hours) [SEQUENTIAL]

**2A: Debug History Navigation Timeout (30-45 min)**

File: `src/tests/e2e/history-navigation.spec.ts`

Steps:
1. Add screenshot/trace on failure (playwright.config.ts)
2. Run test with `page.pause()` before suspected line
3. Identify root cause (selector, timing, or UI change)
4. Fix and verify test passes

**2B: Investigate Theme Architecture (30 min) [RESEARCH]**

**Critical question**: How do theme changes work without `window.gameActions.updateTheme()`?

Research steps:
1. Find theme UI buttons in components (Settings panel, theme selector)
2. Trace event handlers (what happens on click?)
3. Follow state flow: UI click → state update → URL update
4. Document mechanism for fixing tests

Expected findings:
- Theme changes are handled by Svelte reactive statements in App.svelte (lines 58-129)
- UI clicks update gameState directly (not through kernel)
- Reactive effects apply theme and update URL
- Tests should click UI, not manipulate state

**2C: Fix Theme Tests - DOM-Driven Approach (2-3 hours)**

File: `src/tests/e2e/settings-color-panel.spec.ts` (7 FAILING tests)

**Pattern for ALL 7 tests**:

```typescript
// REMOVE all window.gameActions.updateTheme() calls

// REPLACE with DOM interactions:
test('should change color and update URL', async ({ page }) => {
  await helper.goto(12345);

  // Open color editor via UI
  await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
  await page.locator('.theme-colors-btn').click();

  // Interact with color picker
  const primaryPicker = page.locator('.color-picker[data-var="--p"]');
  await primaryPicker.click();
  await page.locator('.picker input[type="range"]').first().fill('100');

  // Close picker (triggers URL update)
  await page.keyboard.press('Escape');

  // Wait for URL update
  await page.waitForFunction(() => window.location.href.includes('v='));

  // Verify URL contains color override
  expect(page.url()).toContain('v=');

  // Verify DOM reflects change
  const newColor = await page.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue('--p').trim()
  );
  expect(newColor).not.toBe(''); // Non-empty means custom color applied
});
```

**Tests to fix** (all 7):
- Line 14: Change color → verify URL update
- Line 55: Theme change → clear custom colors
- Line 101/122: Copy URL → navigate back → verify persistence
- Line 155: Share theme URL → verify shareable
- Line 196: Reset colors → verify URL clears
- Line 234: Export CSS → verify values
- Line 259: Preserve colors across actions → verify URL maintained

**Success criteria**: All 7 tests pass, 0 references to `window.gameActions`.

Result: 8 failures → 0 failures, 70 skipped

---

### Phase 3: Smart Consolidation (3-4 hours)

**3A: Empirical Analysis (1 hour)**

Before consolidating, gather data:

```bash
# Unskip both files temporarily
# Edit url-state-management.spec.ts line 10: remove .skip
# Edit back-button-comprehensive.spec.ts line 9: remove .skip

# Run both files
npm run test:e2e -- url-state-management.spec.ts
npm run test:e2e -- back-button-comprehensive.spec.ts

# Document results:
# - Which tests pass/fail?
# - Which tests are truly redundant?
# - Which tests have unique coverage?
```

**3B: Create Consolidated File (2-3 hours)**

Create: `src/tests/e2e/url-and-navigation.spec.ts`

**Structure** (~18-20 tests, based on empirical findings):

```
url-and-navigation.spec.ts
├── URL Encoding (4 tests)
│   ├── Encodes actions in URL
│   ├── Handles compression
│   ├── Restores state from URL
│   └── Handles invalid action sequences
├── Browser History (6 tests)
│   ├── Back button works
│   ├── Forward button works
│   ├── Multiple back/forward sequences
│   ├── History branching on divergent path
│   ├── No history loss after back+continue
│   └── AI continues after navigation
├── Phase Transitions (3 tests)
│   ├── Tracks URL through bidding
│   ├── Tracks URL through trump
│   └── Tracks URL through playing
├── Config Persistence (3 tests)
│   ├── Player types preserved
│   ├── Game settings preserved
│   └── Multi-hand game tracking
└── Deterministic Replay (2 tests)
    ├── Same seed + actions = same state
    └── Compression/decompression is lossless
```

**Pattern**:
```typescript
test('should preserve state across back button', async ({ page }) => {
  // Load state via URL
  const url = helper.encodeGameUrl({ seed: 12345, actions: ['bid-30'] });
  await page.goto(url);

  // Verify phase
  const phase1 = await page.locator('.app-container').getAttribute('data-phase');
  expect(phase1).toBe('bidding');

  // Take action
  await page.click('[data-testid="pass"]');
  await page.waitForSelector('[data-phase="bidding"]');

  // Go back
  await page.goBack();

  // Verify URL and state restored
  expect(page.url()).toContain('a=CAAA'); // Original action encoding
  const phase2 = await page.locator('.app-container').getAttribute('data-phase');
  expect(phase2).toBe('bidding');
});
```

**3C: Delete Original Files**

```bash
rm src/tests/e2e/url-state-management.spec.ts
rm src/tests/e2e/back-button-comprehensive.spec.ts
```

Result: ~50 skipped tests remaining

---

### Phase 4: Incremental Unskipping (3-4 hours) [ONE FILE AT A TIME]

**4A: basic-gameplay-new.spec.ts (1-2 hours)**

File: `src/tests/e2e/basic-gameplay-new.spec.ts` (11 tests)

Steps:
1. Remove `.skip` from line 16
2. Run tests: `npm run test:e2e -- basic-gameplay-new.spec.ts`
3. If failures occur:
   - Debug each failure
   - Fix using DOM-driven patterns
   - Re-run until all pass
4. Only proceed to 4B when ALL tests pass

Expected: Should pass (designed for new architecture), but verify empirically.

Result: +11 passing

**4B: ai-back-button.spec.ts (1 hour)**

File: `src/tests/e2e/ai-back-button.spec.ts` (3 tests)

Steps:
1. Remove `.skip` from line 4
2. Run tests: `npm run test:e2e -- ai-back-button.spec.ts`
3. Fix any issues using DOM patterns
4. Verify all 3 pass

Note: Tests use `window.getGameView()` which still exists (will be refactored in Phase 5).

Result: +3 passing

**4C: Evaluate basic-gameplay.spec.ts (30 min)**

File: `src/tests/e2e/basic-gameplay.spec.ts` (12 tests)

Question: Is this redundant with `basic-gameplay-new.spec.ts`?

Steps:
1. Compare both files side-by-side
2. If >80% redundant: DELETE basic-gameplay.spec.ts
3. If unique coverage exists: Modernize and keep OR merge unique tests into basic-gameplay-new.spec.ts
4. Decision based on empirical comparison

Expected: Likely redundant (original plan suggests deletion).

Result: -12 skipped (deleted) OR +12 passing (fixed)

**4D: Evaluate settings-color-basic.spec.ts (15 min)**

File: `src/tests/e2e/settings-color-basic.spec.ts` (5 tests, currently PASSING)

Question: Is this redundant with settings-color-panel.spec.ts (which we fixed in Phase 2C)?

Steps:
1. Compare both files
2. If redundant: DELETE settings-color-basic.spec.ts
3. If unique: Keep both

Note: Original plan suggests deletion (line 78).

Expected: Likely redundant.

Result: -5 tests (deleted)

---

### Phase 5: Architecture Cleanup (2-3 hours)

**5A: Add data-processing to Helper (30 min)**

Update `src/tests/e2e/helpers/game-helper.ts`:

```typescript
// Replace window.getGameView() calls with DOM inspection

// OLD:
async isProcessing(): Promise<boolean> {
  return await this.page.evaluate(() => window.getGameView().isProcessing);
}

// NEW:
async isProcessing(): Promise<boolean> {
  const processing = await this.page.locator('.app-container').getAttribute('data-processing');
  return processing === 'true';
}

// OLD:
async getCurrentPhase(): Promise<GamePhase> {
  return await this.page.evaluate(() => window.getGameView().state.phase);
}

// NEW (already correct):
async getCurrentPhase(): Promise<GamePhase> {
  return await this.page.locator('.app-container').getAttribute('data-phase') as GamePhase;
}
```

**5B: Minimize Window API Usage (1-2 hours)**

Update all test files:

**Mapping** (window API → DOM):
- `view.state.phase` → `[data-phase]` attribute ✓
- `view.isProcessing` → `[data-processing]` attribute ✓
- `view.state.currentPlayer` → Parse `.turn-player` text content
- `view.state.tricks.length` → Count `.trick-spot .played-domino` elements
- `view.playerTypes` → URL param (reload if needed)

**Acceptable window API usage** (if needed):
- Synchronous blocking operations that have no DOM representation
- Should be minimal/rare

**5C: Remove Dev Helpers (15 min)**

Clean up `src/main.ts`:

```typescript
// DELETE this (line ~36):
window.game = game;

// KEEP (but minimize usage):
window.getGameView = () => { ... };  // Only if absolutely necessary
```

Clean up `src/types/global.d.ts`:
- Remove any unused declarations
- Ensure types match actual implementation

**5D: Final Verification (30 min)**

```bash
# Run full E2E suite
npm run test:e2e

# Expected: ~46 passing, 0 failing, 0 skipped

# Run full unit test suite (ensure no regressions)
npm test

# Expected: 1003 passing
```

---

## Success Criteria

### Quantitative Targets
- ✅ ~46 E2E tests passing (exact count depends on consolidation decisions)
- ✅ 0 E2E tests failing
- ✅ 0 E2E tests skipped
- ✅ 0 references to `window.gameActions` anywhere
- ✅ 1003 unit tests still passing (no regressions)
- ✅ Minimal window API usage (prefer DOM inspection)

### Qualitative Standards
- ✅ Tests use URL for state setup (event sourcing)
- ✅ Tests use DOM for verification (rendered output)
- ✅ Tests use UI interaction for actions (click buttons)
- ✅ No compatibility layers or workarounds
- ✅ Clear, maintainable test patterns
- ✅ Tests verify actual user workflows (not internal APIs)

### Files Changed Summary
- **Delete**: 3-4 test files (~23-28 tests removed as redundant/obsolete)
- **Create**: 1 consolidated file (~18-20 tests)
- **Update**: 3-4 files (modernized patterns)
- **Infrastructure**: Add data-processing attribute, minimize window API

---

## Common Patterns Reference

### Setup Game State

```typescript
// Via helper (recommended)
await helper.loadStateWithActions(12345, ['bid-30', 'pass', 'pass', 'pass']);

// Direct URL
await page.goto('/?s=12345&a=CAAA&testMode=true');

// With config
const url = helper.encodeGameUrl({
  seed: 12345,
  actions: ['bid-30'],
  playerTypes: ['human', 'ai', 'ai', 'ai']
});
await page.goto(url);
```

### Verify State via DOM

```typescript
// Phase
const phase = await page.locator('.app-container').getAttribute('data-phase');
expect(phase).toBe('playing');

// Processing
await page.waitForSelector('[data-processing="false"]');

// Score
await expect(page.locator('.score-us')).toContainText('2');

// Trump
await expect(page.locator('[data-testid="trump-display"]')).toContainText('Doubles');

// Current player
const playerText = await page.locator('.turn-player').textContent();
expect(playerText).toContain('Player 1');

// Theme
const theme = await page.evaluate(() =>
  document.documentElement.getAttribute('data-theme')
);
expect(theme).toBe('forest');
```

### Execute Actions via UI

```typescript
// Bid
await page.click('[data-testid="bid-30"]');

// Pass
await page.click('[data-testid="pass"]');

// Trump selection
await page.click('[data-testid="trump-doubles"]');

// Play domino
await page.click('[data-testid="domino-6-6"]');

// Change theme
await page.locator('[data-theme="forest"]').click();
```

### Wait for State Changes

```typescript
// Wait for phase transition
await page.waitForSelector('[data-phase="playing"]');

// Wait for processing to complete
await page.waitForSelector('[data-processing="false"]');

// Wait for URL update
await page.waitForFunction(() => window.location.href.includes('t=forest'));

// Wait for specific element
await page.waitForSelector('[data-testid="trump-doubles"]');

// Wait for element to disappear
await page.waitForSelector('[data-testid="bid-30"]', { state: 'hidden' });
```

### Verify URL Updates

```typescript
// Check seed
expect(page.url()).toContain('s=12345');

// Check actions encoded
expect(page.url()).toContain('a=');

// Check theme
expect(page.url()).toContain('t=forest');

// Check color overrides
expect(page.url()).toContain('v=');

// Full URL match
await page.waitForFunction(() =>
  window.location.search.includes('s=12345&a=CAAA')
);
```

---

## Troubleshooting Guide

### "Test times out waiting for element"

**Causes**:
1. `testMode=true` not set in URL (AI delays enabled)
2. URL encoding incorrect (state didn't load)
3. Selector is stale (UI changed)
4. Element rendered but not visible

**Debug**:
```typescript
// Add screenshot
await page.screenshot({ path: 'scratch/timeout-debug.png' });

// Check URL
console.log(page.url());

// Check phase
const phase = await page.locator('.app-container').getAttribute('data-phase');
console.log('Current phase:', phase);

// Pause test
await page.pause();
```

### "Element not found"

**Causes**:
1. `data-testid` doesn't match current implementation
2. Element not rendered yet (wait for phase transition)
3. Element hidden/not visible
4. Wrong selector

**Debug**:
```typescript
// List all testids
const testids = await page.evaluate(() =>
  Array.from(document.querySelectorAll('[data-testid]'))
    .map(el => el.getAttribute('data-testid'))
);
console.log('Available testids:', testids);

// Check if element exists (even if hidden)
const exists = await page.locator('[data-testid="bid-30"]').count();
console.log('Element count:', exists);
```

### "State doesn't match expected"

**Causes**:
1. URL encoding doesn't match intended actions
2. Action wasn't actually executed (network failed)
3. Wrong phase for expected state
4. Incorrect test assumptions

**Debug**:
```typescript
// Verify URL matches expectations
console.log('URL:', page.url());

// Check current phase
const phase = await page.locator('.app-container').getAttribute('data-phase');
console.log('Phase:', phase);

// Check processing state
const processing = await page.locator('.app-container').getAttribute('data-processing');
console.log('Processing:', processing);

// Screenshot current state
await page.screenshot({ path: 'scratch/state-mismatch.png' });
```

### "Theme/color not updating"

**Causes**:
1. Not interacting with UI (trying to manipulate state directly)
2. Not waiting for reactive effects to complete
3. Selector wrong for theme/color picker
4. Missing `Escape` keypress to close picker

**Debug**:
```typescript
// Verify theme changed in DOM
const theme = await page.evaluate(() =>
  document.documentElement.getAttribute('data-theme')
);
console.log('Current theme:', theme);

// Verify color override in DOM
const color = await page.evaluate(() =>
  getComputedStyle(document.documentElement).getPropertyValue('--p').trim()
);
console.log('Current --p color:', color);

// Check URL
console.log('URL:', page.url());
await page.screenshot({ path: 'scratch/theme-debug.png' });
```

---

## Key Differences from Original Plan

### What Changed
1. **Theme test strategy**: DOM-driven UI interaction (not URL navigation)
2. **Consolidation scope**: ~18-20 tests (not 15), based on ~30% redundancy (not 100%)
3. **Task sequencing**: Sequential phases (not parallel), especially Phase 4 before Phase 5
4. **Investigation added**: Phase 2B researches theme architecture before fixing
5. **data-processing required**: Not optional, needed for DOM-only testing
6. **Incremental unskipping**: One file at a time with fixes (not batch unskip)

### Why Changed
- **Original plan's theme approach was fundamentally flawed** (tested URL loading, not user workflow)
- **Redundancy claims were overstated** (empirical analysis shows 30%, not 100%)
- **Parallel tasks had hidden conflicts** (Task 4 uses API that Task 5 removes)
- **Need to understand architecture** before fixing (how do theme changes actually work?)
- **Incremental approach is safer** (fix issues as they arise, not batch at end)

---

## Risk Assessment & Mitigation

### High Risk ⚠️

**Risk 1: Theme Architecture Unknown**
- **Impact**: Could block all 7 theme tests indefinitely
- **Mitigation**: Phase 2B investigates first (30 min), then decide approach
- **Fallback**: Skip theme tests, document as tech debt, fix in separate session

**Risk 2: Consolidation Loses Coverage**
- **Impact**: Silent regression in URL handling or navigation
- **Mitigation**: Phase 3A runs both files first, maps coverage empirically
- **Fallback**: Keep both files if >50% unique coverage found

### Medium Risk ⚠️

**Risk 3: Hidden Issues in Skipped Tests**
- **Impact**: Tests fail when unskipped, require debugging
- **Mitigation**: Phase 4 unskips incrementally, fixes before moving to next file
- **Budget**: Add 50% time buffer to Phase 4 estimates

**Risk 4: Window API Removal Breaks Tests**
- **Impact**: Working tests start failing during Phase 5
- **Mitigation**: Phase 5 happens LAST, after all tests pass
- **Fallback**: Keep minimal `window.__testSync` API if needed

### Low Risk ℹ️

**Risk 5: Playwright Selectors Stale**
- **Impact**: Selectors in helper may not match current UI
- **Mitigation**: Helper uses `data-testid` attributes (stable by design)
- **Verification**: Visual inspection during debugging (screenshots/pause)

---

## Resources & References

### Documentation
- **Vision**: `docs/VISION.md` - North star outcomes, architectural invariants
- **Orientation**: `docs/ORIENTATION.md` - Mental models, composition, request flow
- **Principles**: `docs/ARCHITECTURE_PRINCIPLES.md` - Design philosophy
- **Concepts**: `docs/CONCEPTS.md` - Implementation reference

### Code References
- **Test Helper**: `src/tests/e2e/helpers/game-helper.ts` (856 lines, comprehensive)
- **Modern Example**: `src/tests/e2e/basic-gameplay-new.spec.ts` (designed for new architecture)
- **Working Examples**: `src/tests/e2e/perfects-page.spec.ts`, `settings-color-basic.spec.ts`
- **Window API**: `src/main.ts` (lines 36-77)
- **URL Encoding**: `src/game/core/url-compression.ts` (lines 204-251 for theme/colors)
- **App Component**: `src/App.svelte` (line 138 for data-phase, lines 58-129 for theme reactivity)

### Commands
```bash
# Run all E2E tests
npm run test:e2e

# Run specific test file
npm run test:e2e -- settings-color-panel.spec.ts

# Run with UI (debugging)
npm run test:e2e -- --ui

# Run with headed browser
npm run test:e2e -- --headed

# Run unit tests (check for regressions)
npm test

# Type check
npm run typecheck
```

---

## Execution Timeline

**Total estimated time**: 13-17 hours

| Phase | Time | Tasks |
|-------|------|-------|
| Phase 1: Foundation | 30 min | Delete obsolete, add data-processing |
| Phase 2: Investigation & Fixes | 4-5 hours | Debug timeout, research theme, fix 7 tests |
| Phase 3: Consolidation | 3-4 hours | Analyze, merge, verify |
| Phase 4: Unskipping | 3-4 hours | basic-gameplay-new, ai-back-button, evaluate others |
| Phase 5: Cleanup | 2-3 hours | Minimize window API, verify |

---

## Progress Tracking Template

Copy to `docs/E2E_MIGRATION_PROGRESS.md` and update as you work:

```markdown
# E2E Migration Progress

**Started**: [DATE]
**Status**: In Progress

## Phase 1: Foundation ✓ | 🚧 | ❌
- [ ] Delete ai-skip-functionality.spec.ts
- [ ] Remove skipAIDelays from global.d.ts
- [ ] Add data-processing to App.svelte

## Phase 2: Investigation & Fixes ✓ | 🚧 | ❌
- [ ] 2A: Fix history-navigation timeout
- [ ] 2B: Investigate theme architecture
- [ ] 2C: Fix theme test 1/7 (line 14)
- [ ] 2C: Fix theme test 2/7 (line 55)
- [ ] 2C: Fix theme test 3/7 (line 101)
- [ ] 2C: Fix theme test 4/7 (line 122)
- [ ] 2C: Fix theme test 5/7 (line 155)
- [ ] 2C: Fix theme test 6/7 (line 196)
- [ ] 2C: Fix theme test 7/7 (line 259)

## Phase 3: Consolidation ✓ | 🚧 | ❌
- [ ] 3A: Run url-state-management.spec.ts
- [ ] 3A: Run back-button-comprehensive.spec.ts
- [ ] 3A: Document coverage analysis
- [ ] 3B: Create url-and-navigation.spec.ts
- [ ] 3C: Delete original files

## Phase 4: Unskipping ✓ | 🚧 | ❌
- [ ] 4A: Unskip basic-gameplay-new.spec.ts (0/11 passing)
- [ ] 4B: Unskip ai-back-button.spec.ts (0/3 passing)
- [ ] 4C: Evaluate basic-gameplay.spec.ts
- [ ] 4D: Evaluate settings-color-basic.spec.ts

## Phase 5: Cleanup ✓ | 🚧 | ❌
- [ ] 5A: Update game-helper.ts for data-processing
- [ ] 5B: Minimize window API in tests
- [ ] 5C: Remove window.game from main.ts
- [ ] 5D: Final verification (E2E + unit tests)

## Current Status
**Passing**: 7 / ~46
**Failing**: 8 / 0
**Skipped**: 69 / 0

**Last updated**: [DATE]
**Notes**: [Any blockers, decisions, or findings]
```

---

**Document Status**: REVISED AND VALIDATED
**Ready to Execute**: YES
**Next Step**: Begin Phase 1 (delete obsolete tests + add data-processing attribute)
