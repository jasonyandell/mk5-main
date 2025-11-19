# Texas 42

**Note**: This project uses [bd (beads)]. Use `bd` commands instead of markdown TODOs. Use the bd tool for all task tracking. Before starting work, run `bd ready` to see what's available. File issues for discovered bugs/work as you go.

**IMPORTANT**: This project uses git worktrees. Always use `bd --no-daemon` to avoid committing/pushing to the wrong branch:
- ✅ `bd --no-daemon ready`
- ✅ `bd --no-daemon show <issue-id>`
- ✅ `bd --no-daemon create --title "Title" --description "Details" --priority 1 --type task`
- ❌ `bd ready` (daemon can commit to wrong branch in worktree setup)

# North Star
The authors are building a crystal palace in the sky with this project.  We want this to be beautiful and correct above all. If the authors were mechanics, this project would be their "project car".  They work on it on weekends and free time for the love of the building and with no external time pressure, only pride in a job well done and the enjoyment of the process itself.  They enjoy elegance, simlicity and correctness.  They are MORE THAN HAPPY to spend extra time making every little thing perfect.  We are on the 8th major overhaul and if we get to 100 major overhaul, that just means we had fun.

## Quick Start

**New to the codebase?** Read [docs/ORIENTATION.md](docs/ORIENTATION.md) first for architecture overview.

**Detailed references:**
- [docs/remixed-855ccfd5.md](docs/remixed-855ccfd5.md) - Full multiplayer architecture specification
- [docs/archive/pure-layers-threaded-rules.md](docs/archive/pure-layers-threaded-rules.md) - RuleSet system deep-dive (historical)
- [docs/rules.md](docs/rules.md) - Official Texas 42 game rules

## Overview
Web implementation of Texas 42 dominoes game with pure functional architecture:
- Event sourcing: `state = replayActions(config, history)`
- Two-level composition: RuleSets (execution) + ActionTransformers (actions)
- Capability-based multiplayer with filtered views
- Zero coupling between core engine and action transformers/multiplayer

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

** No legacy ** - CRITICAL. this is a greenfield project.  everything should be unified, even if it takes significant extra work

** No skipped tests ** - this is a greenfield project.  all tests should pass and be valuable, even if it takes significant extra work

## Running TypeScript scripts
- This project uses `"type": "module"` in package.json - ES modules only!
- When creating test scripts:
  - Use `.ts` extension for TypeScript files
  - Use ES module imports: `import { thing } from './path'`
  - NOT CommonJS: ~~`const { thing } = require('./path')`~~
- Common pitfall: npm run build outputs to dist/ which may not exist yet
