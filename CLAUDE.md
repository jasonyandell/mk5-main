** Texas 42 **
Web implementation of Texas 42 dominoes game
- Svelte/TypeScript SPA with real-time gameplay
- AI opponent with quickplay mode
- Complete rule enforcement and scoring
Official rules are in docs/rules.md

** Every line of code is a liability **
** Strive for correct by construction **

** Game logic **
Code is src/game/index.ts
Pure functions and states. Strictly. Fix any issues if you discover them

** Mobile UI ** - Prioritize mobile: wrap dominoes, no horizontal scroll, maximize vertical space

** Temporary files ** - All temporary files, test artifacts, and scratch work should be placed in the scratch/ directory, which is gitignored
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

** Unit Tests ** - for pure game logic and pasted URLs.

** Playwright ** - 
Use src/tests/e2e/helpers/game-helper.ts for all interactions
Locators go in game-helper.ts, NOT in tests
CRITICAL: setTimeout() is BANNED and can NEVER be used in playwright tests

** No legacy ** - CRITICAL. this is a greenfield project.  everything should be unified, even if it takes significant extra work

** No skipped tests ** - this is a greenfield project.  all tests should pass and be valuable, even if it takes significant extra work
