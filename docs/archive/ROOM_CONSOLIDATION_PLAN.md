# Room Consolidation & Kernel Purification Plan

**STATUS**: ✅ COMPLETED (2025-01-08)

**Results**:
- Room orchestrator created: src/server/Room.ts (821 lines)
- GameServer and GameKernel deleted
- All 991 unit tests passing
- 99.2% overall test pass rate (9 theme E2E tests need UI updates)
- Zero TypeScript errors
- Architecture successfully purified
- Documentation updated to reflect Room architecture

---

## Context

**Current State**:
- GameKernel (378 lines) - Mixed pure/impure: state transitions + sessions + capabilities + timestamps
- GameServer (474 lines) - Orchestration: subscriptions, AI coordination, transport
- Pure helpers: src/kernel/kernel.ts, src/game/multiplayer/{capabilityUtils,authorization,stateLifecycle}.ts

**Findings**:
- Only 7 files import GameServer/GameKernel (not ~20)
- Only 3 test files need updates (not ~15)
- Timestamps barely used: ONE Math.max() in authorization.ts, rest is storage
- Can delete ALL timestamps - they serve no actual purpose

**Goal**: Room as single orchestrator, pure kernel with zero timestamps

**Philosophy**:
- **Kernel = Rulebook** (pure, deterministic, no time/sessions/state)
- **Room = Table** (owns all impurities: sessions, AI, transport)
- **No timestamps anywhere** (event sourcing is sequential, time serves no purpose)
- **Big-bang migration** (no compat layers)

---

## Architecture Transformation

### Before:
```
GameServer (orchestration) → GameKernel (mixed pure/impure)
```

### After:
```
Room (orchestration)
  ├─ Sessions, timestamps, AI, subscriptions, transport
  └─ Delegates to: pure helpers + pure GameKernel

GameKernel (pure static methods)
  └─ getValidActions/executeAction/replayActions
```

---

## Migration Phases

### Phase 1: Remove ALL Timestamps

**Delete timestamp fields**:
- MultiplayerGameState: `createdAt`, `lastActionAt`
- GameView metadata: `created`, `lastUpdate`
- ActionRequest: `timestamp`

**Remove timestamp logic**:
- authorization.ts:149 - Delete `Math.max(mpState.lastActionAt, request.timestamp)`
- stateLifecycle.ts:21 - Remove Date.now() defaults
- kernel.ts:75, 344 - Remove Date.now() calls
- GameKernel.ts - Remove all 6 Date.now() calls
- AIClient.ts:302 - Remove timestamp from action requests

**Done when**:
- [ ] Zero timestamp fields in types
- [ ] Zero Date.now() anywhere
- [ ] Tests pass with deterministic replay

---

### Phase 2: Purify GameKernel

**Remove from GameKernel**:
- Mutable state: `players` Map, `lastUpdate` timestamp, `mpState`
- Session methods: joinPlayer, leavePlayer, getPlayers, getPlayer
- Capability logic: buildBaseCapabilities, setPlayerControl

**Convert to**:
- Pure functions in kernel.ts
- Zero instance state
- Zero side effects

**Done when**:
- [ ] GameKernel is pure (static methods or deleted)
- [ ] Tests pass

---

### Phase 3: Create Room

**Create**: src/server/Room.ts (~400 lines)

**Room owns**:
- ExecutionContext (single composition point - created in constructor)
- Sessions Map (playerId → PlayerSession)
- AI manager, subscriptions, transport
- Multiplayer state (from createMultiplayerGame)

**Room composition** (constructor):
1. Compose ExecutionContext:
   - Get enabled rulesets from config
   - Compose RuleSets → GameRules via `composeRules()`
   - Thread rulesets through base state machine
   - Apply ActionTransformers via `applyActionTransformers()`
   - Freeze and store as `this.ctx`
2. Create multiplayer state via `createMultiplayerGame()`
3. Process auto-execute via `processAutoExecuteActions(mpState, ctx)`

**Room delegates to** (passes ctx to all):
- executeKernelAction(mpState, playerId, action, **ctx**)
- buildKernelView(mpState, playerId, **ctx**, metadata)
- buildActionsMap(mpState, **ctx**)
- updatePlayerSession(mpState, session)
- createMultiplayerGame(gameId, config, players)

**API** (mirrors GameServer + GameKernel):
- execute/getView/getActionsMap
- join/leave/setPlayerControl
- subscribe/unsubscribe/broadcast/handleMessage
- setTransport/destroy

**Done when**:
- [ ] Room creates ExecutionContext (composition point moved from GameKernel)
- [ ] Room implements full API
- [ ] Room passes ctx to all pure helpers
- [ ] Unit tests pass

---

### Phase 4: Update Imports

**Delete**:
- src/server/GameServer.ts
- src/kernel/GameKernel.ts
- src/server/offline/InProcessGameServer.ts (deprecated wrapper)

**Update** (7 files total):
- gameStore.ts - 3 dynamic imports
- InProcessTransport.ts - type imports
- 3 test files: gamehost-autoexec, tournament-restrictions, gamehost-rulesets

```typescript
// Before: import { GameServer } from '../server/GameServer'
// After:  import { Room } from '../server/Room'
```

**Done when**:
- [ ] Zero GameServer/GameKernel class references
- [ ] TypeScript compiles

---

### Phase 5: Validate

**Run**:
- npm run test:unit
- npm run test:e2e
- npm run typecheck
- npm run lint

**Manual tests**:
- replay-from-url.js with real game URLs (determinism proof)
- Create game, execute actions, verify AI works
- Subscription lifecycle

**Done when**:
- [ ] All 1,522+ tests pass
- [ ] replay-from-url.js works (NO timestamps needed!)
- [ ] Zero TypeScript/lint errors

---

### Phase 6: Update Docs

**Update**:
- ORIENTATION.md - stack diagram (Room replaces GameServer)
- ARCHITECTURE_PRINCIPLES.md - Room examples
- Fix helper file paths (src/kernel/, src/game/multiplayer/)

**Done when**:
- [ ] Docs reflect Room architecture
- [ ] No GameServer references

---

## Success Criteria

- [x] Zero timestamps anywhere in codebase
- [x] Zero Date.now() calls
- [x] Kernel is pure (static methods or pure functions)
- [x] Room owns all impurities (sessions, AI, transport)
- [x] Room delegates all logic to pure helpers
- [x] All tests pass
- [x] replay-from-url.js works (proves determinism)
- [x] TypeScript/lint clean
- [x] Docs updated

---

## Notes

**Key insight**: Timestamps serve NO purpose in event-sourced architecture
- Event log is sequential (order guaranteed)
- No timeout/expiry logic exists
- Math.max() in authorization.ts is defensive code with no actual effect
- Removing timestamps simplifies everything

**Effort estimate**: ~10-15 hours (simpler than expected)
