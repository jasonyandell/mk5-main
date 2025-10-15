# URL-to-State Isomorphism Fix - Session Handoff

## Current State
The URL-to-state isomorphism is completely broken after the multiplayer refactoring. URLs contain encoded actions but these are never replayed when loading a game.

### What's Broken
- E2E tests failing: 4 tests in `basic-gameplay.spec.ts` expect specific states from URLs
- URL sharing doesn't work - always loads initial state
- Browser navigation (back/forward) doesn't restore game state
- Debug URLs are useless - can't reproduce issues

### Root Cause
```typescript
// Old system (working):
loadFromURL() → parse seed + actions → replay to reach state

// New system (broken):
GameConfig { seed } → creates fresh game → actions in URL ignored
```

## Architecture Context

### Current Flow
```
URL (?s=abc&a=AAACDE...)
    ↓
gameStore.ts creates GameConfig { seed } only
    ↓
NetworkGameClient → InProcessAdapter → GameHost
    ↓
Fresh game created (actions lost!)
```

### Key Files
- `src/stores/gameStore.ts` - Creates GameClient, needs URL parsing
- `src/game/core/url-compression.ts` - Has `decodeGameUrl()` to parse URLs
- `src/game/utils/replay.ts` - Has action replay logic (unused currently)
- `src/shared/multiplayer/protocol.ts` - GameConfig type (no actions field)

## The Fix: Two-Phase Initialization

### Phase 1: Parse and Create
```typescript
// gameStore.ts line ~30
const urlData = decodeGameUrl(window.location.search);
const config: GameConfig = {
  playerTypes: urlData.playerTypes,
  shuffleSeed: urlData.seed
};
gameClient = new NetworkGameClient(adapter, config);
```

### Phase 2: Replay Actions
```typescript
// After client ready, replay actions from URL
if (urlData.actions.length > 0) {
  for (const actionId of urlData.actions) {
    const action = parseActionId(actionId);
    await gameClient.requestAction(playerId, action);
  }
}
```

## Implementation Needs

### 1. Action ID Parser
Map URL action IDs to GameActions:
- `'pass'` → `{ type: 'pass', player: 0 }`
- `'bid-30'` → `{ type: 'bid', player: 0, value: 30 }`
- `'trump-blanks'` → `{ type: 'select-trump', suit: 0 }`
- `'play-3-2'` → `{ type: 'play', domino: [3, 2], player: 0 }`

### 2. Player Assignment
Determine which player executes each action:
- Track current player from game state
- Handle consensus actions (any player)
- Match action player field

### 3. URL Update Hook
Keep URL in sync as game progresses:
```typescript
gameClient.subscribe(state => {
  updateURLWithState(initialState, executedActions);
});
```

## Test Verification

### E2E Test Pattern
```typescript
// This should work after fix:
await helper.loadStateWithActions(12345, ['pass', 'pass', 'pass', 'pass']);
// Expects: redeal action available
// Currently: still in initial bidding
```

### Quick Test Commands
```bash
# Test specific failing test
npm run test:e2e -- src/tests/e2e/basic-gameplay.spec.ts -g "should progress through bidding round"

# All basic gameplay tests
npm run test:e2e -- src/tests/e2e/basic-gameplay.spec.ts
```

## Critical Implementation Notes

1. **Must replay through protocol** - Don't bypass GameHost validation
2. **Actions must execute sequentially** - Wait for each to complete
3. **Handle invalid actions gracefully** - URL might have stale actions
4. **Preserve session filtering** - Main client sessionId still applies

## Success Criteria
- [ ] URLs with actions load correct game state
- [ ] E2E tests pass (especially `loadStateWithActions` tests)
- [ ] Browser back/forward works
- [ ] URL updates as game progresses
- [ ] Share URLs work across sessions

## Current Session Work
- Implemented session filtering in InProcessAdapter ✓
- Documented current architecture ✓
- Identified URL replay as root cause of E2E failures ✓
- Created this handoff document ✓

## Next Steps
1. Implement action ID parser
2. Add replay logic to gameStore initialization
3. Test with failing E2E tests
4. Add URL update subscription
5. Verify all E2E tests pass

---
*Last updated: October 2024 - URL replay is THE critical blocker for E2E tests*