# Capability System Reference

**Last Updated**: 2025-01-09

The capability system controls what players can see (visibility) and do (authorization) through composable capability tokens.

> **Authority vs. client snapshots**: Room always operates on the full, unfiltered `GameState`. Before shipping a snapshot to a client it applies capability filtering (hands removed, metadata stripped) but keeps the `MultiplayerGameState` shape. Client code should treat any host-sent `coreState` as already redacted and never attempt to reconstitute hidden information.

## Overview

Instead of boolean flags (`isAI`, `isSpectator`), we use capability tokens that compose naturally.

## Standard Capability Builders

Located in `src/game/multiplayer/capabilities.ts`

### Human Player
```typescript
humanCapabilities(playerIndex: 0 | 1 | 2 | 3)
```
Returns:
- `act-as-player` - Can execute actions for this player
- `observe-own-hand` - Can see their own hand

### AI Player
```typescript
aiCapabilities(playerIndex: 0 | 1 | 2 | 3)
```
Returns:
- `act-as-player` - Can execute actions for this player
- `observe-own-hand` - Can see their own hand
- `replace-ai` - Can be hot-swapped with human

### Spectator
```typescript
spectatorCapabilities()
```
Returns:
- `observe-all-hands` - Can see all player hands
- `observe-full-state` - Can see complete game state

### Coach
```typescript
coachCapabilities(studentIndex: 0 | 1 | 2 | 3)
```
Returns:
- `observe-hand` (studentIndex) - Can see student's hand
- `see-hints` - Can see hint metadata

### Tutorial Student
```typescript
tutorialCapabilities(playerIndex: 0 | 1 | 2 | 3)
```
Returns:
- `act-as-player` - Can execute actions
- `observe-own-hand` - Can see their hand
- `see-hints` - Can see hints for learning
- `undo-actions` - Can undo mistakes

## Fluent Builder API

```typescript
import { buildCapabilities } from 'src/game/multiplayer/capabilities';

const customCaps = buildCapabilities()
  .actAsPlayer(0)
  .observeOwnHand()
  .seeHints()
  .seeAIIntent()
  .build();
```

## All Capability Types

```typescript
type Capability =
  // Action capabilities
  | { type: 'act-as-player'; playerIndex: number }
  | { type: 'replace-ai' }
  | { type: 'configure-variant' }

  // Visibility capabilities
  | { type: 'observe-own-hand' }
  | { type: 'observe-hand'; playerIndex: number }
  | { type: 'observe-all-hands' }
  | { type: 'observe-full-state' }
  | { type: 'see-hints' }
  | { type: 'see-ai-intent' }

  // Special capabilities
  | { type: 'undo-actions' };
```

## How It Works

### Visibility Filtering

`getVisibleStateForSession(state, session)` filters game state based on capabilities:

```typescript
// Human player sees only their hand
const humanView = getVisibleStateForSession(state, humanSession);
// players[1,2,3].hand = [] (hidden)

// Spectator sees all hands
const spectatorView = getVisibleStateForSession(state, spectatorSession);
// All players[].hand visible

// Coach sees student's hand (index 0)
const coachView = getVisibleStateForSession(state, coachSession);
// players[0].hand visible, others hidden
```

### Action Authorization

`filterActionsForSession(session, actions)` filters actions by capabilities:

```typescript
const allActions = getValidActions(state);
const allowed = filterActionsForSession(playerSession, allActions);

// Returns only actions player can execute based on:
// 1. act-as-player capability
// 2. Action metadata requiredCapabilities field
```

### Metadata Filtering

Actions can have metadata (hints, AI intent) that only authorized viewers see:

```typescript
const action = {
  type: 'bid',
  value: 30,
  meta: {
    hint: 'Safe bid with this hand',
    requiredCapabilities: [{ type: 'see-hints' }]
  }
};

// Player without see-hints capability
filterActionForSession(playerSession, action)
// → { type: 'bid', value: 30 } (meta stripped)

// Tutorial student with see-hints
filterActionForSession(tutorialSession, action)
// → { type: 'bid', value: 30, meta: { hint: '...' } } (hint visible)
```

### Server-Generated Action Maps

`Room` applies `filterActionsForSession()` for every connected session and ships those filtered lists to clients:

```typescript
// Room.notifyListeners()
const actionsByPlayer = this.buildActionsMap(...);
record.listener({
  view,
  state,
  actions: actionsByPlayer,
  perspective
});
```

`NetworkGameClient` caches the resulting `Record<string, ValidAction[]>`, so UI components can call `getActions(playerId)` or read the Svelte stores without re-running capability logic on the client.

## Usage Examples

### Creating Custom Player Types

```typescript
// Tournament organizer
const organizerCaps = buildCapabilities()
  .observeFullState()
  .configureVariant()
  .build();

// Debug viewer
const debugCaps = buildCapabilities()
  .observeFullState()
  .observeAllHands()
  .seeHints()
  .seeAIIntent()
  .build();

// Replay spectator (no actions)
const replayCaps = buildCapabilities()
  .observeAllHands()
  .build();
```

### Integration with Room

```typescript
// Room uses standard builders
private buildBaseCapabilities(playerIndex: number, controlType: 'human' | 'ai') {
  const idx = playerIndex as 0 | 1 | 2 | 3;
  return controlType === 'human'
    ? humanCapabilities(idx)
    : aiCapabilities(idx);
}

// Subscriptions deliver HostViewUpdate payloads
host.subscribe(sessionId, ({ view, state, actions }) => {
  // view: FilteredGameState for this session
  // state: authoritative MultiplayerGameState snapshot
  // actions: Record<string, ValidAction[]> filtered by capability
});
```

### Integration with Authorization

```typescript
// Pure function checks capabilities
getValidActionsForPlayer(mpState, playerId) {
  const session = mpState.players.find(p => p.playerId === playerId);

  // Composed state machine includes BOTH RuleSets AND action transformers:
  // 1. RuleSets define game mechanics (base, nello, splash, plunge, sevens)
  // 2. Action transformers filter/annotate actions (tournament, oneHand)
  // 3. Result: Actions that are mechanically valid + transformer-appropriate
  const allActions = composedStateMachine(mpState.coreState);

  // Capability filtering is the final step (authorization)
  return filterActionsForSession(session, allActions);
}

// NetworkGameClient consumers read server-filtered results
const allowed = await gameClient.getActions('player-0');
```

## Testing Capabilities

```typescript
test('spectator sees all hands but cannot act', () => {
  const session = {
    playerId: 'spectator-1',
    playerIndex: 0,
    controlType: 'human',
    capabilities: spectatorCapabilities()
  };

  // Can see all hands
  const view = getVisibleStateForSession(state, session);
  expect(view.players.every(p => p.hand.length > 0)).toBe(true);

  // Cannot execute actions
  const actions = filterActionsForSession(session, allActions);
  expect(actions.length).toBe(0);
});
```

## Interaction with Layer System

**Capabilities filter layer-generated actions:**

```typescript
// Room creates view for a session
createView(playerId) {
  // 1. Compose RuleSets (game mechanics)
  const rules = composeRules([baseRuleSet, nelloRuleSet, ...]);

  // 2. Generate actions with composed rules
  const composedActions = getValidActions(state, ruleSets, rules);
  // → Includes NELLO bids if nello RuleSet enabled

  // 3. Apply action transformers (availability filters)
  const transformedActions = applyActionTransformers(composedActions, actionTransformerConfigs);
  // → May remove NELLO if tournament action transformer active

  // 4. Filter by capabilities (authorization)
  const session = getSession(playerId);
  const authorizedActions = filterActionsForSession(session, transformedActions);
  // → Only actions this session can execute

  return { state: filteredState, validActions: authorizedActions };
}
```

**Separation of Concerns:**
- **Layers**: Define HOW special contracts work (mechanics)
- **Variants**: Define WHEN special contracts are allowed (availability)
- **Capabilities**: Define WHO can execute actions (authorization)

**Example:**
```typescript
// Nello layer adds nello mechanics
const nelloAction = { type: 'bid', bid: 'nello', value: 2, player: 0 };

// Tournament variant removes it (filtering)
const tournamentActions = actions.filter(a => a.bid !== 'nello');

// Capabilities check authorization (act-as-player for player 0)
const session = { capabilities: [{ type: 'act-as-player', playerIndex: 0 }] };
const allowed = filterActionsForSession(session, actions);
// → Only player 0's actions
```

---

## References

- Vision Document: `docs/remixed-855ccfd5.md` §4
- Layer System: `docs/GAME_ONBOARDING.md` "Layer System Architecture"
- Implementation: `src/game/multiplayer/capabilities.ts`
- Filtering: `src/game/multiplayer/capabilityUtils.ts`
- Authorization: `src/game/multiplayer/authorization.ts`
- RuleSet Types: `src/game/layers/types.ts`
