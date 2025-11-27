# Development Workflows

## URL Debugging Workflow (Critical)

When a user provides a localhost URL with a bug report:

### Step 1: Generate Test Automatically
```bash
node scripts/replay-from-url.js "<url>" --generate-test
```
Creates `scratch/test-{timestamp}.js` with all replay logic and state logging.

### Step 2: Debug with Focused Options

| Option | Purpose |
|--------|---------|
| `--action-range 87 92` | Show only actions 87-92 |
| `--hand 4` | Focus on just hand 4 |
| `--show-tricks` | Display trick winners and points |
| `--compact` | One line per action with score changes |
| `--stop-at N` | Stop replay at action N |

### Example Debugging Session
```bash
# First, see overview
node scripts/replay-from-url.js "<url>" --compact

# Found issue around action 45, zoom in
node scripts/replay-from-url.js "<url>" --action-range 40 50

# Check specific hand
node scripts/replay-from-url.js "<url>" --hand 3 --show-tricks
```

## Testing Patterns

### Test Type Selection

| Test Type | Tool | Use When |
|-----------|------|----------|
| **Unit** | `createTestContext()` | Layer composition, pure function isolation |
| **Integration** | `HeadlessRoom` | Full game flows, multi-action scenarios |
| **E2E** | Playwright | UI interactions, user-visible behavior |

### Unit Test Pattern
```typescript
import { createTestContext } from '../helpers/executionContext';

describe('Layer Composition', () => {
  it('should compose nello and plunge correctly', () => {
    const ctx = createTestContext({ layers: ['base', 'nello', 'plunge'] });
    expect(rules.getTrumpSelector(state, bid)).toBe(expectedPlayer);
  });
});
```

### Integration Test Pattern
```typescript
import { HeadlessRoom } from '../../server/HeadlessRoom';

describe('Complete Game Flow', () => {
  it('should play through complete hand', () => {
    const room = new HeadlessRoom({
      playerTypes: ['ai', 'ai', 'ai', 'ai']
    }, 12345); // deterministic seed

    const actions = room.getValidActions(0);
    room.executeAction(0, actions[0].action);
    expect(room.getState().phase).toBe('trump_selection');
  });
});
```

### E2E Test Principles
- **Clean separation**: Tests interact with UI, not game internals
- **Minimal window API**: Prefer DOM inspection over `window.getGameState()`
- **Use PlaywrightGameHelper**: All DOM interactions through helper
- **No setTimeout()**: BANNED - use proper waits

### Key Testing Rule
Tests must use same composition paths as production:
- Production: `Room → ExecutionContext`
- Tests: `HeadlessRoom → Room → ExecutionContext`

**DON'T** directly call `createExecutionContext` in integration tests.

## Scratch Directory

All temporary files, test artifacts, and scratch work go in `scratch/` (gitignored).

### Scratch Playwright Tests
- Must use `.test.ts` extension (not `.spec.ts`)
- Example: `scratch/debug-issue.test.ts`
- Won't run with `npm run test:e2e` (production only)
- Run explicitly: `npx playwright test --config=playwright.scratch.config.ts`

## Debugging Tips by Issue Type

| Issue | Where to Look |
|-------|---------------|
| Action not available | Room constructor (composition), `authorizeAndExecute` |
| Wrong game behavior | Layer implementation, GameRules methods |
| State not updating | `executeAction`, check authorization |
| UI not reactive | Svelte stores, ViewProjection |
| Message not routing | `Room.handleMessage()`, Transport |

## ES Modules Gotchas

Project uses `"type": "module"` - ES modules only!

```typescript
// CORRECT
import { thing } from './path'

// WRONG - CommonJS won't work
const { thing } = require('./path')
```

- Use `.ts` extension for TypeScript files
- Common pitfall: `npm run build` outputs to `dist/` which may not exist yet

## Task Management (Beads)

### Before Starting Work
1. Check ready issues: Use beads MCP tools
2. Claim issue: `bd update <issue-id> --status in_progress`

### During Work
- File discovered bugs as you go
- Use `bd create` for new issues

### Before Closing
1. Run `npm run test:all` - must pass
2. Close issue: `bd close <issue-id>`

## Common Commands

```bash
# Development
npm run dev           # Start dev server
npm run build         # Production build
npm run preview       # Preview production build

# Quality
npm run typecheck     # TypeScript (run often!)
npm run lint          # ESLint
npm run format        # Prettier

# Testing
npm test              # Unit tests
npm run test:watch    # Watch mode
npm run test:e2e      # Playwright E2E
npm run test:all      # FULL SUITE (required before closing issues)

# Utilities
npm run generate:strength-table  # AI hand evaluation
npm run generate:perfects        # Find perfect hands
```

## Request Flow (For Debugging)

```
User clicks button
  → gameStore.executeAction(action)
  → GameClient.send({ type: 'EXECUTE_ACTION', action })
  → Socket.send() → Room.handleMessage()
  → executeKernelAction()
    - authorizeAndExecute()
    - Check action authority
    - Get all valid actions via composed state machine
    - Filter by session capabilities
    - Execute: executeAction(coreState, action, ctx.rules)
    - Process auto-execute actions
  → Room.broadcastState()
    - buildKernelView() per client
    - Build GameView { state, validActions, metadata }
    - send() to each client
  → GameClient.handleMessage()
    - Update cached view
    - Notify subscribers
  → gameStore subscription
  → Derived stores recompute
  → UI reactively updates
```

Room filters once per client perspective, broadcasts only GameView (never unfiltered state).
