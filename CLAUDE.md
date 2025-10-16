# Texas 42

## Overview
Web implementation of Texas 42 dominoes game
- Complete rule enforcement and scoring
- Svelte/TypeScript/Vite/Tailwind CSS SPA with real-time gameplay
  - GameState type: src/game/types.ts:93-133 — Single source of truth
  - GameAction type: src/game/types.ts:143-151 — Event sourcing primitives
  - State store: src/stores/gameStore.ts:136 — Reactive state container
  - Action handlers: src/stores/gameStore.ts:373-476 — Pure state transitions
  - UI root: src/App.svelte — View layer entry point
Official rules are in docs/rules.md

## Philosophy
- Immutable state transitions
- Every line of code is a liability
- Strive for correct by construction

## Temporary files
- All temporary files, test artifacts, and scratch work should be placed in the scratch/ directory, which is gitignored
  - Playwright tests in scratch/ must use `.test.ts` extension (not `.spec.ts`)
  - Example: `scratch/debug-issue.test.ts` 
  - These won't run with `npm run test:e2e` (production tests only)
  - Run scratch tests explicitly: `npx playwright test --config=playwright.scratch.config.ts`

## CRITICAL: URL HANDLING - AUTOMATED TEST GENERATION
  User provides localhost URL with a bug report? Follow this workflow:

  1. **Generate test automatically**: `node scripts/replay-from-url.js "<url>" --generate-test`
      - Creates test file in scratch/test-{timestamp}.js
      - Test includes all replay logic and state logging

  2. **Debug with focused options**:
      - `--action-range 87 92` - Show only actions 87-92 (no grep needed!)
      - `--hand 4` - Focus on just hand 4
      - `--show-tricks` - Display trick winners and points
      - `--compact` - One line per action with score changes
      - `--stop-at N` - Stop replay at action N

## Testing Strategy

### Unit Tests
- For pure game logic and pasted URLs
- Test core functions in isolation

### E2E Tests (Playwright)

**Architecture Principles:**
- **Clean separation**: Tests interact with UI, not game internals
- **Minimal window API**: Prefer DOM inspection over `window.getGameState()`
- **Use game-helper.ts**: All DOM interactions through PlaywrightGameHelper
- **No setTimeout()**: BANNED - use proper waits instead

**Three Test Types:**

1. **UI Tests** (most tests)
   - Test user interactions and DOM behavior
   - Use `helper.goto(seed)` for setup
   - Use `helper.loadStateWithActions()` for specific scenarios
   - Verify via DOM inspection (phase attributes, visible buttons, etc.)
   - Example: `basic-gameplay-new.spec.ts`

2. **Protocol Tests** (when needed)
   - Verify client-server message protocol
   - Use SpyAdapter to wrap InProcessAdapter
   - Assert on message sequences
   - Example: Testing EXECUTE_ACTION messages

3. **Integration Tests** (1-2 tests)
   - Full end-to-end game flow
   - Real InProcessAdapter + GameHost
   - Complete game from bidding to scoring

**Available Test Infrastructure:**

- **MockAdapter** (`src/tests/adapters/MockAdapter.ts`)
  - Pre-configured state sequences
  - Fast, deterministic tests
  - No game logic execution

- **SpyAdapter** (`src/tests/adapters/SpyAdapter.ts`)
  - Wraps real adapter, records messages
  - Protocol verification
  - Message assertions

- **Game State Fixtures** (`src/tests/fixtures/game-states.ts`)
  - Pre-built GameView objects
  - Common scenarios (bidding, playing, scoring)
  - Reusable across tests

**Window API Usage:**

Minimal exposure in `main.ts`:
- `window.getGameState()` - Read-only state inspection (use sparingly)
- `window.quickplayActions` - Feature toggle (legitimate)
- `window.gameActions` - Console debugging

**IMPORTANT**: Prefer DOM inspection over window access:
```typescript
// ✅ GOOD - Test what user sees
const phase = await helper.getCurrentPhase(); // Reads data-phase attribute
await expect(page.locator('[data-testid="pass"]')).toBeVisible();

// ⚠️ USE SPARINGLY - Only when DOM doesn't reflect state
const state = await page.evaluate(() => window.getGameState());
expect(state.currentTrick.length).toBe(1);
```

** No legacy ** - CRITICAL. this is a greenfield project.  everything should be unified, even if it takes significant extra work

** No skipped tests ** - this is a greenfield project.  all tests should pass and be valuable, even if it takes significant extra work

## Running TypeScript scripts
- This project uses `"type": "module"` in package.json - ES modules only!
- When creating test scripts:
  - Use `.ts` extension for TypeScript files
  - Use ES module imports: `import { thing } from './path'`
  - NOT CommonJS: ~~`const { thing } = require('./path')`~~
- Common pitfall: npm run build outputs to dist/ which may not exist yet
