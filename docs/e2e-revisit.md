# E2E Test Suite Fix Plan

## Overview
Plan to fix the e2e test suite based on audit findings. Out of 47 test files, we need to delete 24 debug files, update 8 AI quickplay tests, and maintain 15 legitimate tests. Final result: 47 → 23 test files.

## Phase 1: Cleanup (Delete Debug Files)

### Files to Delete Immediately (24 files)
```bash
# Pure debug files
rm src/tests/e2e/test-debug.spec.ts
rm src/tests/e2e/screenshot-test.spec.ts
rm src/tests/e2e/check-styles-test.spec.ts
rm src/tests/e2e/check-layout-with-url.spec.ts
rm src/tests/e2e/final-screenshot.spec.ts
rm src/tests/e2e/check-exact-url.spec.ts

# Hover debug files
rm src/tests/e2e/hover-precision-test.spec.ts
rm src/tests/e2e/hover-test-with-known-hand.spec.ts
rm src/tests/e2e/hover-test-detailed.spec.ts
rm src/tests/e2e/hover-precision-final.spec.ts
rm src/tests/e2e/hover-double-test.spec.ts
rm src/tests/e2e/check-hover-state.spec.ts
rm src/tests/e2e/final-hover-test.spec.ts
rm src/tests/e2e/test-hover-final.spec.ts

# Highlighting debug files
rm src/tests/e2e/debug-highlight-test.spec.ts
rm src/tests/e2e/simple-highlight-test.spec.ts
rm src/tests/e2e/screenshot-highlighting.spec.ts
rm src/tests/e2e/debug-highlighting.spec.ts
rm src/tests/e2e/test-highlighting.spec.ts
rm src/tests/e2e/test-highlighting-debug.spec.ts
rm src/tests/e2e/domino-highlighting-simple.spec.ts

# Layout debug files
rm src/tests/e2e/trick-layout-test.spec.ts
rm src/tests/e2e/compact-tricks-test.spec.ts

# Deprecated UI tests
rm src/tests/e2e/bidding-ui-test.spec.ts
```

## Phase 2: Fix AI Quickplay Tests (8 files)

### Current Issues
- Tests expect dedicated AI quickplay UI elements with specific data-testid attributes
- Actual AI controls are in DebugPanel QuickPlay tab and QuickAccessToolbar
- Need to update selectors and interaction patterns

### Files to Update

#### 1. `ai-quickplay-controls.spec.ts`
**Issues:**
- Uses `[data-testid="quickplay-run"]` - doesn't exist
- Uses `[data-testid="quickplay-stop"]` - doesn't exist  
- Uses `[data-testid="quickplay-pause"]` - doesn't exist
- Uses `[data-testid="quickplay-resume"]` - doesn't exist
- Uses `[data-testid="quickplay-speed"]` - doesn't exist
- Uses `[data-testid="ai-player-${i}"]` - doesn't exist

**Fix Strategy:**
```typescript
// Replace with DebugPanel interactions
await helper.openDebugPanel();
await helper.locator('button:has-text("QuickPlay")').click(); // Switch to QuickPlay tab
await helper.locator('input[type="checkbox"]:near(:text("QuickPlay Active"))').check();
await helper.locator('select').selectOption('instant'); // Speed selector
```

#### 2. `ai-quickplay-default-speed.spec.ts`
**Issues:**
- Uses `[data-testid="quickplay-speed"]` selector

**Fix Strategy:**
```typescript
await helper.openDebugPanel();
await helper.locator('button:has-text("QuickPlay")').click();
const speedSelector = helper.locator('select'); // Speed dropdown in QuickPlay tab
await expect(speedSelector).toHaveValue('instant');
```

#### 3. `ai-quickplay-game-completion.spec.ts`
**Issues:**
- Uses wrong selectors for AI controls
- Expects dedicated AI player checkboxes

**Fix Strategy:**
```typescript
// Use QuickAccessToolbar toggle instead
await helper.locator('[data-testid="quick-autoplay"]').click();
// Or use DebugPanel for full control
await helper.openDebugPanel();
await helper.locator('button:has-text("QuickPlay")').click();
await helper.locator('button:has-text("Play to End of Game")').click();
```

#### 4. `ai-quickplay-game.spec.ts`
**Issues:**
- Tests AI decision making with wrong element targeting
- Uses non-existent AI player controls

**Fix Strategy:**
```typescript
// Focus on testing AI behavior through store actions
// Use DebugPanel QuickPlay tab for AI configuration
await helper.openDebugPanel();
await helper.locator('button:has-text("QuickPlay")').click();
await helper.locator('button:has-text("Step")').click(); // Single step execution
```

#### 5. `ai-quickplay-new-game.spec.ts`
**Issues:**
- Tests game reset with wrong approach

**Fix Strategy:**
```typescript
// Use actual new game functionality
await helper.newGame(); // Uses [data-testid="new-game-button"]
// Or use DebugPanel reset
await helper.openDebugPanel();
await helper.locator('button:has-text("History")').click();
await helper.locator('button:has-text("Reset Game")').click();
```

#### 6. `ai-quickplay-reset.spec.ts`
**Issues:**
- Tests reset behavior incorrectly

**Fix Strategy:**
```typescript
// Use DebugPanel history reset
await helper.openDebugPanel();
await helper.locator('button:has-text("History")').click();
await helper.locator('button:has-text("Reset Game")').click();
```

#### 7. `ai-quickplay-step.spec.ts`
**Issues:**
- Uses wrong selectors for step functionality

**Fix Strategy:**
```typescript
await helper.openDebugPanel();
await helper.locator('button:has-text("QuickPlay")').click();
await helper.locator('button:has-text("Step")').click();
```

#### 8. `ai-quickplay-visibility.spec.ts`
**Issues:**
- Tests UI visibility with wrong assumptions

**Fix Strategy:**
```typescript
// Test QuickAccessToolbar visibility
await expect(helper.locator('[data-testid="quick-autoplay"]')).toBeVisible();
// Test DebugPanel QuickPlay tab
await helper.openDebugPanel();
await expect(helper.locator('button:has-text("QuickPlay")')).toBeVisible();
```

## Phase 3: Update Legitimate Tests (15 files)

### Files Needing Minor Updates

#### 1. `basic-gameplay.spec.ts`
**Status:** Mostly correct, uses PlaywrightGameHelper properly
**Minor fixes needed:**
- Update any hardcoded selectors to use helper methods
- Ensure all assertions use current UI structure

#### 2. `complete-game.spec.ts`
**Issues:**
- Uses `[data-generic-testid*="bid-button"]` - should use helper methods
- References old debug interface elements

**Fix Strategy:**
```typescript
// Replace direct selector usage with helper methods
const biddingOptions = await helper.getBiddingOptions();
await helper.selectActionByType('bid_points', 30);
```

#### 3. `tournament-compliance.spec.ts`
**Status:** Good structure, may need selector updates
**Minor fixes:**
- Replace direct button selectors with helper methods
- Update any hardcoded data-testid references

#### 4. `debug-ui-validation.spec.ts`
**Issues:**
- References old debug interface title
- Uses outdated element selectors

**Fix Strategy:**
```typescript
// Update to current mobile UI structure
await expect(page.locator('.app-container')).toBeVisible();
await expect(helper.locator('[data-testid="game-phase"]')).toBeVisible();
```

### Files That Are Correct
- `trump-suit-display.spec.ts`
- `page-reload-suit-consistency.spec.ts`
- `complete-trick-in-play-area.spec.ts`
- `debug-snapshot.spec.ts`
- `debug-snapshot-replay.spec.ts`
- `debug-ui-imports.spec.ts`
- `debug-ui-imports-test.spec.ts`
- `domino-highlighting.spec.ts`
- `highlighting-test.spec.ts`
- `test-highlighting-clears.test.ts`
- `test-actions-panel-scroll.spec.ts`

## Phase 4: Implementation Timeline

### Week 1: Cleanup
- [ ] Delete all 24 debug files
- [ ] Run test suite to ensure no dependencies broken
- [ ] Update any imports or references

### Week 2: AI Quickplay Fixes
- [ ] Fix `ai-quickplay-controls.spec.ts`
- [ ] Fix `ai-quickplay-default-speed.spec.ts`
- [ ] Fix `ai-quickplay-game-completion.spec.ts`
- [ ] Fix `ai-quickplay-game.spec.ts`

### Week 3: AI Quickplay Fixes (continued)
- [ ] Fix `ai-quickplay-new-game.spec.ts`
- [ ] Fix `ai-quickplay-reset.spec.ts`
- [ ] Fix `ai-quickplay-step.spec.ts`
- [ ] Fix `ai-quickplay-visibility.spec.ts`

### Week 4: Legitimate Test Updates & Final Validation
- [ ] Update `complete-game.spec.ts`
- [ ] Update `debug-ui-validation.spec.ts`
- [ ] Update `tournament-compliance.spec.ts`
- [ ] Update `basic-gameplay.spec.ts`
- [ ] Final test suite validation

## Success Criteria

### Quantitative Goals
- Reduce test files from 47 to 23 (24 deleted, 0 new)
- Achieve 100% test pass rate
- Reduce test execution time by ~50% (removing debug files)

### Qualitative Goals
- All tests use PlaywrightGameHelper consistently
- No hardcoded selectors or console.log statements
- Tests reflect actual UI structure and functionality
- Clear test names and descriptions
- Proper assertions with meaningful error messages

## Risk Mitigation

### Potential Issues
1. **Breaking existing CI/CD** - Delete files gradually, test after each batch
2. **Missing UI elements** - Verify all selectors exist before updating tests
3. **Test interdependencies** - Check for shared state or setup between tests

### Rollback Plan
- Keep deleted files in a separate branch for 30 days
- Maintain current test suite in parallel during transition

## Final Validation

### Pre-deployment Checklist
- [ ] All tests pass locally
- [ ] All tests pass in CI environment
- [ ] Test coverage reports show maintained coverage
- [ ] No console errors or warnings during test execution
- [ ] Test execution time within acceptable limits
- [ ] All updated tests follow established patterns and conventions
