# Multiplayer Implementation Status

## Completed: Phase 1 & 2 (Authorization + Local GameClient)

### What Works Now ✅

**Core Primitives** (`src/game/multiplayer/`)
- `types.ts` - PlayerSession, MultiplayerGameState, ActionRequest, Result types
- `authorization.ts` - Pure authorization layer
  - `canPlayerExecuteAction()` - checks if player can execute action
  - `authorizeAndExecute()` - composable auth + execution
- `GameClient.ts` - Interface defining single source of truth
- `LocalGameClient.ts` - In-memory implementation with AI scheduler

**Integration**
- `gameStore.ts` - Simplified from 982→312 lines, wraps GameClient
- UI works unchanged via compatibility shims
- Build succeeds, 18/18 tests passing
- AI runs automatically via LocalGameClient polling

### Key Architectural Win

Everything now flows through:
```typescript
await gameClient.requestAction(playerId, action)
```

Authorization, validation, execution, and AI scheduling happen inside this single call.

## What's Next: Phase 3 (Network Transport)

### To Implement

**Server-side** (new files needed)
- `server/game-room.ts` - Durable Object running `authorizeAndExecute`
- `server/api/games.ts` - REST endpoints (create game, join, leave)
- Server-side AI worker management (spawn when seat empty, kill when player joins)

**Client-side** (new files needed)
- `src/game/multiplayer/NetworkGameClient.ts` - HTTP/WebSocket implementation
  - POST actions → server runs `authorizeAndExecute`
  - WebSocket receives state broadcasts
  - Same interface as LocalGameClient (swap in place)

**Progressive Enhancement**
- `upgradeToOnline(localState)` - Export state, create server game, return NetworkGameClient
- `downgradeToOffline(networkClient)` - Fetch state, create LocalGameClient

### Critical Implementation Notes

1. **Don't use modulus for player math** - Use `getNextPlayer()` from `src/game/core/players.ts`
2. **Authorization already works** - Server just needs to call `authorizeAndExecute(mpState, request)`
3. **State is already serializable** - GameState + actionHistory in URL proves this
4. **AI strategies are pure** - They work identically in worker processes

### What NOT to Build

- ❌ Don't rebuild authorization (already done)
- ❌ Don't change GameClient interface (it's perfect)
- ❌ Don't modify core game engine (it's pure)
- ❌ Don't add multiplayer logic to UI (stays unchanged)

### Estimated Effort

- Server implementation: ~4-6 hours
- NetworkGameClient: ~2-3 hours
- Progressive enhancement: ~1 hour
- **Total: ~8-10 hours to full multiplayer**

### Testing Strategy

Start with NetworkGameClient connecting to local server, then:
1. Two browser tabs, same device (both human players)
2. Cross-device multiplayer (find a friend)
3. Mixed human/AI seats (server spawns AI workers)
4. Drop-in/drop-out (AI fills vacated seat)

## Session Handoff

**Context for next session:**
1. Read `docs/multiplayer-architecture.md` (original vision)
2. Read this file (current status)
3. Review `src/game/multiplayer/` directory (what's built)
4. Start with: "Implement NetworkGameClient for online multiplayer"

**Files to reference:**
- `LocalGameClient.ts` - Template for NetworkGameClient
- `authorization.ts` - Server will call `authorizeAndExecute()`
- `GameClient.ts` - Interface both implementations must match
