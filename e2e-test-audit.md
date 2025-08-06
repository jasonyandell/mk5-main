# E2E Test Audit Report

## Executive Summary
Out of 47 total e2e test files, the audit reveals:
- **12 legitimate tests** that should be kept and maintained
- **8 tests for non-existent features** (AI quickplay UI)
- **27 debug/temporary files** that should be removed
- Most debug files contain console.log statements, screenshots, and lack proper assertions

## Categories of Tests

### ✅ KEEP: Legitimate Tests (12 files)
These tests follow proper structure, use PlaywrightGameHelper, and test current functionality:

#### Core Gameplay
1. `basic-gameplay.spec.ts` - Core gameplay validation including bidding and trump selection
2. `complete-game.spec.ts` - End-to-end integration test
3. `tournament-compliance.spec.ts` - Critical Texas 42 tournament rules validation

#### UI Features
4. `trump-suit-display.spec.ts` - Trump display in tricks
5. `page-reload-suit-consistency.spec.ts` - State persistence across reloads
6. `complete-trick-in-play-area.spec.ts` - Complete trick button functionality

#### Debug Panel
7. `debug-ui-validation.spec.ts` - Debug panel functionality
8. `debug-snapshot.spec.ts` - Snapshot creation and tracking
9. `debug-snapshot-replay.spec.ts` - Snapshot replay validation
10. `debug-ui-imports.spec.ts` - Debug UI component loading
11. `debug-ui-imports-test.spec.ts` - Basic UI and keyboard shortcuts

#### Highlighting & Actions
12. `domino-highlighting.spec.ts` - Comprehensive highlighting during bidding/trump selection
13. `domino-highlighting-simple.spec.ts` - Simple highlighting test
14. `highlighting-test.spec.ts` - Highlighting system test
15. `test-highlighting-debug.spec.ts` - Console events and highlighting
16. `test-highlighting-clears.test.ts` - Highlighting clears after actions
17. `test-actions-panel-scroll.spec.ts` - Actions panel scroll behavior

### ❌ REMOVE: Tests for Non-Existent Features (8 files)
These test AI quickplay UI elements that don't exist in the current interface:

1. `ai-quickplay-step.spec.ts`
2. `ai-quickplay-visibility.spec.ts`
3. `ai-quickplay-default-speed.spec.ts`
4. `ai-quickplay-controls.spec.ts`
5. `ai-quickplay-game-completion.spec.ts`
6. `ai-quickplay-new-game.spec.ts`
7. `ai-quickplay-reset.spec.ts`
8. `ai-quickplay-game.spec.ts`

### 🗑️ REMOVE: Debug/Temporary Files (27 files)
These files contain debugging artifacts and should be removed:

#### Pure Debug Files
1. `test-debug.spec.ts` - Simple button exploration with console.log
2. `screenshot-test.spec.ts` - Only takes screenshots
3. `check-styles-test.spec.ts` - CSS style debugging
4. `check-layout-with-url.spec.ts` - Hardcoded URL debugging
5. `final-screenshot.spec.ts` - Simple screenshot tool
6. `check-exact-url.spec.ts` - Hardcoded localhost URL test

#### Hover Debug Files
7. `hover-precision-test.spec.ts` - Manual hover exploration
8. `hover-test-with-known-hand.spec.ts` - Specific state hover testing
9. `hover-test-detailed.spec.ts` - Detailed hover debugging
10. `hover-precision-final.spec.ts` - Precise hover positioning debug
11. `hover-double-test.spec.ts` - Double domino hover debug
12. `check-hover-state.spec.ts` - Svelte component state inspection
13. `final-hover-test.spec.ts` - Visual hover verification
14. `test-hover-final.spec.ts` - Hardcoded state hover test

#### Highlighting Debug Files
15. `debug-highlight-test.spec.ts` - Class inspection debugging
16. `simple-highlight-test.spec.ts` - Has some legitimate tests but mixed with debug
17. `screenshot-highlighting.spec.ts` - Screenshot-only test
18. `debug-highlighting.spec.ts` - DOM inspection debugging
19. `test-highlighting.spec.ts` - Event system investigation

#### Layout Debug Files
20. `trick-layout-test.spec.ts` - Trick display layout debugging
21. `compact-tricks-test.spec.ts` - AI play and tricks debugging

#### Legacy/Deprecated
22. `bidding-ui-test.spec.ts` - Tests deprecated UI elements (action-button.bid)

## Recommendations

### Immediate Actions
1. **Delete all 27 debug/temporary files** - They provide no test coverage and clutter the test suite
2. **Delete or update the 8 AI quickplay tests** - The UI they test doesn't exist
3. **Keep and maintain the 12-17 legitimate tests** - These provide actual test coverage

### Best Practices Going Forward
1. **No console.log statements** in production tests
2. **No hardcoded URLs or game states** unless for specific regression tests
3. **Use PlaywrightGameHelper** for all interactions
4. **Include proper assertions** - not just screenshots
5. **Follow the 5-second timeout rule** per CLAUDE.md
6. **Name tests descriptively** - avoid "test", "final", "debug" in filenames

### Test Coverage Gaps
Based on the audit, consider adding tests for:
- Game scoring logic
- Hand completion detection
- Mark accumulation
- Player turn management
- Error states and edge cases

## Files to Keep (Clean List)
```
src/tests/e2e/basic-gameplay.spec.ts
src/tests/e2e/complete-game.spec.ts
src/tests/e2e/tournament-compliance.spec.ts
src/tests/e2e/trump-suit-display.spec.ts
src/tests/e2e/page-reload-suit-consistency.spec.ts
src/tests/e2e/complete-trick-in-play-area.spec.ts
src/tests/e2e/debug-ui-validation.spec.ts
src/tests/e2e/debug-snapshot.spec.ts
src/tests/e2e/debug-snapshot-replay.spec.ts
src/tests/e2e/debug-ui-imports.spec.ts
src/tests/e2e/debug-ui-imports-test.spec.ts
src/tests/e2e/domino-highlighting.spec.ts
src/tests/e2e/domino-highlighting-simple.spec.ts
src/tests/e2e/highlighting-test.spec.ts
src/tests/e2e/test-highlighting-debug.spec.ts
src/tests/e2e/test-highlighting-clears.test.ts
src/tests/e2e/test-actions-panel-scroll.spec.ts
```

Total: 17 legitimate tests out of 47 files (36% retention rate)