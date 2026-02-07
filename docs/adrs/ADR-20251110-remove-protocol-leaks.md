# ADR-20251110: Remove Protocol State Leaks

**Status**: Implemented
**Date**: 2025-11-10
**Deciders**: Architecture Review

## Context

The multiplayer protocol was leaking unfiltered state across the trust boundary, violating the core architectural principle of "server authority and filter on demand."

### The Problem

Protocol messages (`GameCreatedMessage` and `StateUpdateMessage`) contained THREE representations of game data:

1. **view: GameView** - Correctly filtered per-perspective (✅)
2. **state: MultiplayerGameState** - UNFILTERED with all hands visible (❌)
3. **actions: Record<string, ValidAction[]>** - ALL player actions indexed by playerId (❌)

This meant clients received:
- All players' hands regardless of capabilities
- All players' valid actions (including AI moves)
- Internal session details and capabilities
- Full action metadata including hints meant for specific players

**Example of the leak:**
```typescript
// Browser console revealed everything:
window.getGameState().coreState.players[2].hand  // AI's hand visible!
window.getActionsMap()['ai-3']  // See AI's available moves
```

### Violations

This broke multiple architectural invariants:

1. **Server Authority** - Clients had unfiltered data, enabling client-side validation/cheating
2. **Filter on Demand** - State was sent unfiltered and cached unfiltered
3. **Capability-Based Access** - Permissions were ignored when caching full state
4. **Zero Coupling** - Clients recomputed transitions instead of trusting server
5. **Trust Boundary** - Security perimeter was completely broken

## Decision

**Remove `state` and `actions` fields from all protocol messages. Send ONLY filtered `GameView`.**

### Changes Made

#### 1. Extended GameView with Transitions

Added `transitions: ViewTransition[]` field to GameView:

```typescript
export interface ViewTransition {
  id: string;
  label: string;
  action: GameAction;
  group?: string;
  recommended?: boolean;
}
```

`ViewTransition` is GameView equivalent of `StateTransition` without the leaked `newState` field.

#### 2. Updated Protocol Messages

**Before:**
```typescript
export interface GameCreatedMessage {
  type: 'GAME_CREATED';
  gameId: string;
  view: GameView;
  state: MultiplayerGameState;  // ❌ LEAK
  actions: Record<string, ValidAction[]>;  // ❌ LEAK
}
```

**After:**
```typescript
export interface GameCreatedMessage {
  type: 'GAME_CREATED';
  gameId: string;
  view: GameView;  // ✅ Only filtered data
}
```

Same change applied to `StateUpdateMessage`.

#### 3. Server Changes (Room.ts)

Updated three broadcast methods to send only GameView:

- `notifyListeners()` - removed `getState()` and `getActionsMap()` calls
- `handleCreateGame()` - removed state/actions from GAME_CREATED message
- `handleSubscribe()` - removed state/actions from STATE_UPDATE message

**Before:**
```typescript
const view = this.getView(playerId);
const state = this.getState();  // ❌
const actions = this.getActionsMap();  // ❌
this.sendMessage(clientId, { type: 'STATE_UPDATE', gameId, view, state, actions });
```

**After:**
```typescript
const view = this.getView(playerId);
this.sendMessage(clientId, { type: 'STATE_UPDATE', gameId, view });
```

#### 4. Client Changes (NetworkGameClient.ts)

- Removed `cachedState` and `cachedActions` properties
- Cache only `cachedView: GameView`
- Added `synthesizeStateFromView()` for backwards compatibility with `getState()` interface
- Updated message handlers to process only `view` from messages
- `getCachedActionsMap()` now derives from cached views instead of separate cache

**Key insight:** NetworkGameClient now trusts server completely - no local recomputation of state.

#### 5. Store Changes (gameStore.ts)

Removed client-side execution context and transition recomputation:

**Before:**
```typescript
const ctx = createExecutionContext({ playerTypes });
const allTransitions = getNextStates($gameState, ctx);  // ❌ Recomputing!
```

**After:**
```typescript
const view = this.client?.getCachedView($sessionId);
const transitions = view?.transitions ?? [];  // ✅ Trust server
```

Removed unused `getNextStates` and `createExecutionContext` imports.

## Consequences

### Positive

✅ **Security fixed** - No unfiltered state crosses network boundary
✅ **Architecture aligned** - "Filter on demand" and "server authority" restored
✅ **Simpler clients** - No execution context, no recomputation, just render
✅ **Smaller payloads** - Eliminated redundant state/actions fields
✅ **Single source of truth** - GameView is the ONLY data clients receive
✅ **Better encapsulation** - Internal MultiplayerGameState never exposed

### Neutral

- NetworkGameClient still implements `getState()` via synthesis for backwards compatibility
- Test fixtures updated to include `transitions` field
- MockAdapter simplified (removed unused helper methods)

### Trade-offs

- **Breaking protocol change** - Not backwards compatible with old clients
  - **Mitigation**: Greenfield project, no production deployments
- **Type synthesis overhead** - `synthesizeStateFromView()` converts GameView → MultiplayerGameState
  - **Mitigation**: Temporary bridge, consumers will migrate to GameView directly
  - **Future**: Deprecate `getState()` interface method entirely

## Implementation Notes

### Backwards Compatibility Bridge

NetworkGameClient provides temporary synthesis:

```typescript
async getState(): Promise<MultiplayerGameState> {
  const view = this.cachedView;
  return this.synthesizeStateFromView(view);  // Convert for compatibility
}
```

This allows existing code to continue working while migration happens incrementally.

### Test Updates

All test fixtures now include `transitions` field:

```typescript
const transitions = validActions.map(valid => ({
  id: JSON.stringify(valid.action),
  label: valid.label,
  action: valid.action,
  ...(valid.group ? { group: valid.group } : {}),
  ...(valid.recommended ? { recommended: valid.recommended } : {})
}));
```

### Validation

- ✅ TypeScript compilation succeeds (zero errors)
- ✅ All architectural invariants restored
- ✅ No unfiltered state in protocol messages
- ✅ Clients cannot access other players' hands
- ✅ Network payloads reduced

## Future Work

1. **Deprecate getState()** - Remove `MultiplayerGameState` from client interface entirely
2. **Direct GameView consumption** - Update all consumers to use `getView()` instead
3. **Remove synthesis** - Delete `synthesizeStateFromView()` once migration complete
4. **Protocol versioning** - Add version field to support future migrations

## References

- **VISION.md** - Non-Negotiable Principles (filter on demand, server authority, trust boundary)
- **ORIENTATION.md** - Architectural Invariants #2 (Server Authority) and #1 (Pure State Storage)
- **protocol.ts** - Updated message definitions
- **kernel.ts** - `buildKernelView()` now includes transitions
- **Room.ts** - Server-side broadcast methods simplified

## Decision Rationale

This change directly addresses a **critical security vulnerability** where unfiltered game state crossed the trust boundary. The protocol was designed to support "dumb clients" but was sending unnecessary data that enabled client-side cheating and violated core architectural principles.

The fix aligns implementation with intent: server computes everything, clients render what they're shown. No validation, no recomputation, complete trust.

**Result**: The architecture is now correct by construction - clients CAN'T cheat because they never receive unfiltered data.

---

## Completion Update (2025-11-10)

**Status**: Fully implemented and backwards compatibility removed

### Final Implementation

All backwards compatibility layers have been removed from the codebase:

1. **Protocol cleaned**: `state` and `actions` fields removed from `GameCreatedMessage` and `StateUpdateMessage`
2. **Server simplified**: Room broadcasts only `GameView` with transitions
3. **Client modernized**: NetworkGameClient removed all synthesis code and uses GameView directly
4. **Store updated**: gameStore uses transitions from GameView instead of recomputing

### Removed Code

The following backwards compatibility mechanisms were removed:

- `NetworkGameClient.synthesizeStateFromView()` - No longer needed
- `NetworkGameClient.getState()` - Removed, consumers use `getView()` instead
- `NetworkGameClient.getActionsMap()` - Removed, consumers use `view.transitions` instead
- Client-side execution context creation - Removed from gameStore
- Client-side transition recomputation - Removed from gameStore

### Test Results

All tests pass (978 tests):
- ✅ Unit tests: All passing
- ✅ Integration tests: All passing
- ✅ E2E tests: All passing
- ✅ TypeScript compilation: Zero errors

### Architecture Verification

The refactor successfully achieves 100% architectural purity:

1. ✅ **No state leaks**: Only GameView crosses the network boundary
2. ✅ **Server authority**: Clients trust server completely, no local validation
3. ✅ **Filter on demand**: Server computes views per-perspective on every update
4. ✅ **Trust boundary**: Security perimeter is properly enforced
5. ✅ **Single source of truth**: GameView includes all client needs (state + transitions)
6. ✅ **Clean separation**: Room (server) and NetworkGameClient (client) have clear responsibilities

### Breaking Changes

This is a **breaking protocol change**:
- Old clients cannot connect to new servers (missing `state`/`actions` fields)
- New clients cannot connect to old servers (missing `transitions` field)

**Impact**: None - greenfield project with no production deployments.

### Future Work

No further cleanup needed - the architecture is now fully aligned with the vision:

- Server: Pure authority, computes everything, broadcasts filtered views
- Clients: Dumb terminals, render what they're shown, trust completely
- Protocol: Minimal surface area, single data structure (GameView)

The system is now **correct by construction** - impossible to cheat because unfiltered data never leaves the server.
