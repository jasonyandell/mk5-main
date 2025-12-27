# Development Workflows

## URL Debugging Workflow (Critical)

When a user provides a localhost URL with a bug report:

### Step 1: Generate Test Automatically
```bash
npx tsx scripts/replay-from-url.ts "<url>" --generate-test
```
Creates `scratch/test-{timestamp}.test.ts` with Vitest assertions and replay logic.

Run the generated test:
```bash
npx vitest --config vitest.scratch.config.ts run scratch/test-*.test.ts
```

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
npx tsx scripts/replay-from-url.ts "<url>" --compact

# Found issue around action 45, zoom in
npx tsx scripts/replay-from-url.ts "<url>" --action-range 40 50

# Check specific hand
npx tsx scripts/replay-from-url.ts "<url>" --hand 3 --show-tricks
```

### Other URL Scripts
```bash
# Encode seed + actions to URL
npx tsx scripts/encode-url.ts 12345 pass pass pass bid-30

# Decode URL to see components
npx tsx scripts/decode-url.ts "?s=9ix&a=AAACS"
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
import { StateBuilder } from '../helpers/stateBuilder';

describe('Layer Composition', () => {
  it('should compose nello and plunge correctly', () => {
    const ctx = createTestContext({ layers: ['base', 'nello', 'plunge'] });
    const state = new StateBuilder()
      .withPhase('playing')
      .withTrump({ type: 'suit', suit: 'fives' })
      .build();
    expect(ctx.rules.getTrumpSelector(state, bid)).toBe(expectedPlayer);
  });
});
```

### StateBuilder (Primary Test State API)

Located in `src/tests/helpers/stateBuilder.ts`. Fluent API for constructing test states.

#### Factory Methods (Entry Points)
```typescript
// Create state at specific phase
StateBuilder.inBiddingPhase(dealer?)           // Dealt hands, ready to bid
StateBuilder.inTrumpSelection(bidder?, value?) // After winning bid
StateBuilder.inPlayingPhase(trump?)            // Trump selected, ready to play
StateBuilder.withTricksPlayed(count, trump?)   // Mid-hand with N tricks done
StateBuilder.inScoringPhase(scores)            // All tricks played
StateBuilder.gameEnded(winningTeam)            // Terminal state

// Special contracts
StateBuilder.nelloContract(bidder?)            // Marks bid, 3-player tricks
StateBuilder.splashContract(bidder?, value?)   // Partner selects trump
StateBuilder.plungeContract(bidder?, value?)   // 4+ marks, partner trump
StateBuilder.sevensContract(bidder?)           // High card must lead
```

#### Chainable Modifiers
```typescript
.withDealer(player)                 // Set dealer position (0-3)
.withCurrentPlayer(player)          // Set current player
.withTrump(trump)                   // Set trump selection
.withWinningBid(player, bid)        // Set winning bidder and bid
.withPlayerHand(index, dominoes)    // Set specific hand (strings or Domino[])
.withHands([hand0, hand1, ...])     // Set all 4 hands at once
.withCurrentTrick([plays])          // Set current trick in progress
.withTricks(tricks)                 // Set completed tricks
.withTeamScores(team0, team1)       // Set hand scores
.withTeamMarks(team0, team1)        // Set game marks
.withSeed(seed)                     // Set shuffle seed
.withConfig(partialConfig)          // Merge partial config
.with(overrides)                    // Escape hatch for arbitrary state
```

#### Deal Constraints (Generate Specific Hands)
```typescript
// Ensure player has minimum doubles (for plunge testing)
.withPlayerDoubles(0, 4)

// Full constraint specification
.withPlayerConstraint(0, {
  minDoubles: 4,
  exactDominoes: ['6-6'],
  voidInSuit: [6]
})

// Deterministic fill for remaining slots
.withFillSeed(12345)
```

#### Example Usage
```typescript
// Simple: playing phase with aces as trump
const state = StateBuilder
  .inPlayingPhase({ type: 'suit', suit: ACES })
  .withSeed(12345)
  .build();

// Complex: mid-hand scenario with specific setup
const state = StateBuilder
  .inPlayingPhase({ type: 'suit', suit: SIXES })
  .withTricksPlayed(3)
  .withTeamScores(15, 8)
  .withCurrentTrick([
    { player: 0, domino: '6-5' },
    { player: 1, domino: '5-4' }
  ])
  .withPlayerHand(2, ['6-6', '6-4', '5-5', '4-4'])
  .build();

// Special contract: plunge with 4 doubles
const state = StateBuilder
  .plungeContract(0, 4)
  .withPlayerDoubles(0, 4)
  .withFillSeed(99999)
  .build();
```

### Related Helpers
- `HandBuilder.fromStrings(['6-6', '5-5'])` - Parse hand from string IDs
- `HandBuilder.withDoubles(count)` - Generate hand with N doubles
- `DominoBuilder.from('6-5')` - Parse single domino from string
- `DominoBuilder.doubles(6)` - Create double-six

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
