# Capability System Reference

The capability system controls what players can **do** (actions) and what they can **see** (information). It uses composable capability tokens instead of boolean flags.

---

## Capability Types

Two capability types exist:

```typescript
type Capability =
  | { type: 'act-as-player'; playerIndex: number }
  | { type: 'observe-hands'; playerIndices: number[] | 'all' };
```

### act-as-player

Grants permission to execute actions for a specific seat.

```typescript
{ type: 'act-as-player', playerIndex: 0 }  // Can play as seat 0
```

### observe-hands

Controls which hands are visible. Can specify:
- **Array**: See specific players' hands
- **'all'**: See all players' hands (spectator mode)

```typescript
{ type: 'observe-hands', playerIndices: [0] }      // See own hand only
{ type: 'observe-hands', playerIndices: [0, 2] }   // See player 0 and 2
{ type: 'observe-hands', playerIndices: 'all' }    // See all hands
```

---

## Standard Capability Builders

Located in `src/multiplayer/capabilities.ts`:

### humanCapabilities(playerIndex)

For human players - can act and see their own hand.

```typescript
humanCapabilities(0)
// → [
//   { type: 'act-as-player', playerIndex: 0 },
//   { type: 'observe-hands', playerIndices: [0] }
// ]
```

### aiCapabilities(playerIndex)

For AI players - identical to human capabilities.

```typescript
aiCapabilities(1)
// → [
//   { type: 'act-as-player', playerIndex: 1 },
//   { type: 'observe-hands', playerIndices: [1] }
// ]
```

### spectatorCapabilities()

For spectators - can see all hands but cannot act.

```typescript
spectatorCapabilities()
// → [
//   { type: 'observe-hands', playerIndices: 'all' }
// ]
```

---

## How Capabilities Are Used

### Action Authorization

When a player requests an action:

```typescript
// In authorization.ts
function canExecuteAction(session: PlayerSession, action: GameAction): boolean {
  // Check if action has player field
  if (!('player' in action)) return true;  // System actions

  // Check act-as-player capability
  return session.capabilities.some(
    cap => cap.type === 'act-as-player' && cap.playerIndex === action.player
  );
}
```

### State Visibility

When building a view for a client:

```typescript
// In capabilities.ts
function getVisibleStateForSession(state: GameState, session: PlayerSession): GameState {
  const observeCap = session.capabilities.find(c => c.type === 'observe-hands');

  const visibleIndices = observeCap?.playerIndices === 'all'
    ? [0, 1, 2, 3]
    : observeCap?.playerIndices ?? [];

  return {
    ...state,
    players: state.players.map((player, idx) => ({
      ...player,
      hand: visibleIndices.includes(idx) ? player.hand : []
    }))
  };
}
```

### Metadata Filtering

Actions with metadata (hints, AI intent) are filtered based on capabilities:

```typescript
// In capabilities.ts
function filterActionForSession(session: PlayerSession, action: ValidAction): ValidAction {
  // Strip metadata that requires specific capabilities
  const { hint, aiIntent, requiredCapabilities, ...rest } = action;
  return rest;
}
```

---

## Integration with Room

Room builds views for each connected client:

```typescript
// In Room.ts
private sendStateToClient(clientId: string): void {
  const session = this.getSession(clientId);

  // Filter state by capabilities
  const filteredState = getVisibleStateForSession(this.state, session);

  // Filter actions by capabilities
  const authorizedActions = filterActionsForSession(session, this.validActions);

  // Build view
  const view = { state: filteredState, validActions: authorizedActions };

  this.send(clientId, { type: 'STATE_UPDATE', view });
}
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/multiplayer/types.ts` | Capability type definition (lines 22-24) |
| `src/multiplayer/capabilities.ts` | Builders + filtering functions |
| `src/multiplayer/authorization.ts` | Action authorization logic |

---

## Design Principles

### Simplicity Over Flexibility

Only 2 capability types exist. This covers:
- Players seeing their own hand
- Spectators seeing all hands
- Action authorization per seat

### No Identity Checks

Never check `playerId === someId`. Always check capabilities:

```typescript
// ❌ Wrong
if (session.playerId === 'admin') { /* special access */ }

// ✅ Correct
if (hasCapability(session, { type: 'observe-hands', playerIndices: 'all' })) {
  /* has spectator access */
}
```

### Composable Tokens

Capabilities are data that can be:
- Stored in player sessions
- Serialized and transmitted
- Combined (multiple per player)
- Inspected for debugging

---

## Testing

```typescript
test('spectator sees all hands but cannot act', () => {
  const session: PlayerSession = {
    playerId: 'spectator-1',
    playerIndex: 0,
    controlType: 'human',
    capabilities: spectatorCapabilities()
  };

  // Can see all hands
  const view = getVisibleStateForSession(state, session);
  expect(view.players.every(p => p.hand.length > 0)).toBe(true);

  // Cannot execute actions (no act-as-player capability)
  const canAct = session.capabilities.some(c => c.type === 'act-as-player');
  expect(canAct).toBe(false);
});
```

---

## References

- Multiplayer Architecture: [MULTIPLAYER.md](MULTIPLAYER.md)
- Implementation: `src/multiplayer/capabilities.ts`
- Types: `src/multiplayer/types.ts`
