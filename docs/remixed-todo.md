# Remixed Architecture - Outstanding Work
 
**Last Updated**: 2025-01-15
**Status**: Post-refactor tracking document
**Context**: Variant composition refactor completed, event sourcing foundation in place

---

## Executive Summary

The variant composition refactor successfully migrated from imperative hooks to pure functional composition with event sourcing. However, several features from the original vision document (`remixed-855ccfd5.md`) remain unimplemented or partially complete.

### ✅ Completed
- Single transformer surface for variants
- Pure functional composition (`applyVariants`)
- Event sourcing foundation (`replayActions`)
- Tournament and one-hand variants
- Base engine made maximally permissive
- Zero TypeScript compilation errors

### ⚠️ Partially Implemented
- Scripted actions (defined, not auto-executed by host)
- Capability system (types defined, not enforced)

### ❌ Not Implemented
- Auto-execute handler in GameHost
- Capability-based visibility filtering
- URL encoding updates for event sourcing
- Speed mode variant
- Daily challenge variant
- Seed finder system

---

## 1. Auto-Execute Handler ⚠️ CRITICAL

### Status: Partially Implemented

**What Exists:**
- One-hand variant emits actions with `autoExecute: true`
- Actions include `meta.scriptId` for debugging
- GameHost receives these actions in valid action list

**What's Missing:**
The GameHost loop that automatically executes scripted actions.

### Required Implementation

**File**: `src/server/game/GameHost.ts`

Add method:
```typescript
/**
 * Auto-execute scripted actions until none remain.
 * Guards against runaway scripts with max iteration limit.
 */
private async processAutoExecuteActions(): Promise<void> {
  const MAX_AUTO_EXEC = 50; // Prevent infinite loops
  let iterations = 0;

  while (iterations < MAX_AUTO_EXEC) {
    const actions = this.getValidActionsComposed(this.mpState.state);

    // Find first auto-execute action
    const autoAction = actions.find(a => a.autoExecute === true);
    if (!autoAction) break; // No more scripted actions

    // Execute it (find which player can execute it)
    const player = 'player' in autoAction
      ? this.getPlayerByIndex(autoAction.player)
      : this.mpState.sessions[0]; // Neutral actions: any player

    const result = await this.executeAction(player.playerId, autoAction);

    if (!result.ok) {
      console.error(`Auto-execute failed at step ${iterations}:`, result.error);
      break;
    }

    iterations++;
  }

  if (iterations === MAX_AUTO_EXEC) {
    console.error('Auto-execute limit reached - possible infinite loop', {
      scriptId: autoAction?.meta?.scriptId
    });
  }
}
```

Call after state changes:
```typescript
executeAction(playerId: string, action: GameAction) {
  // ... existing code ...

  this.mpState = result.value;
  this.lastUpdate = Date.now();

  // NEW: Process any auto-execute actions
  await this.processAutoExecuteActions();

  this.notifyListeners();
  return { ok: true };
}
```

### Impact

**Without this:**
- One-hand variant doesn't actually skip bidding
- Users see scripted actions in UI (confusing)
- Manual execution required for each scripted step

**With this:**
- One-hand mode works as designed (instant skip to playing)
- Scripted sequences execute automatically
- Cleaner UX (no intermediate states shown)

### Testing Pattern

```typescript
test('auto-executes one-hand bidding sequence', async () => {
  const host = new GameHost(gameId, {
    playerTypes: ['human', 'ai', 'ai', 'ai'],
    variant: { type: 'one-hand' }
  }, players);

  const view = host.getView();

  // Should be at playing phase after auto-exec
  expect(view.state.phase).toBe('playing');
  expect(view.state.bids).toHaveLength(4);
  expect(view.state.trump.type).not.toBe('not-selected');
});
```

---

## 2. Capability System ❌ NOT IMPLEMENTED

### Status: Types Defined, Not Enforced

**What Exists:**
- `PlayerSession` has capabilities concept
- Authorization checks player index
- Protocol supports metadata on actions

**What's Missing:**
Fine-grained capability-based action visibility and execution control.

### Design Overview

From `remixed-855ccfd5.md` Section 4.5:

```
Action Pipeline: Capability → Authorization → Visibility

1. Variant emits actions with metadata
2. Authorization filters by canExecute capability
3. Visibility filters metadata by viewing capability
```

### Required Types

**File**: `src/game/multiplayer/types.ts`

```typescript
// Capability tokens
export type Capability =
  | 'execute-action'      // Can execute this action
  | 'see-hidden-info'     // Can see opponent hands (spectator)
  | 'see-ai-intent'       // Can see AI hints (coach mode)
  | 'see-hints'           // Can see gameplay hints (tutorial)
  | 'observe-all-hands'   // Full visibility (spectator/replay)
  | 'control-player'      // Can control this player seat
  ;

export interface PlayerSession {
  playerId: string;
  playerIndex: 0 | 1 | 2 | 3;
  controlType: 'human' | 'ai';
  isConnected: boolean;
  name?: string;

  // NEW: Capability set
  capabilities: Set<Capability>;
}

export interface ActionMetadata {
  // Visibility control
  requiredCapability?: Capability;  // Need this to see metadata

  // Execution control
  canExecute?: Capability[];       // Need one of these to execute

  // Hints and annotations
  hint?: string;                   // Gameplay hint (requires see-hints)
  aiIntent?: string;              // AI reasoning (requires see-ai-intent)
  recommendation?: 'strong' | 'weak';
}
```

### Required Implementation

**File**: `src/game/multiplayer/authorization.ts`

```typescript
/**
 * Filter actions by execution capability.
 * Player can only execute actions they have capability for.
 */
export function getExecutableActions(
  session: PlayerSession,
  actions: GameAction[]
): GameAction[] {
  return actions.filter(action => {
    // Check execution capability
    if (action.canExecute && action.canExecute.length > 0) {
      return action.canExecute.some(cap => session.capabilities.has(cap));
    }

    // Default: player-specific actions require control-player
    if ('player' in action) {
      return session.capabilities.has('control-player')
        && action.player === session.playerIndex;
    }

    // Neutral actions: anyone can execute
    return true;
  });
}

/**
 * Filter action metadata by viewing capability.
 * Strips metadata the player cannot see.
 */
export function getVisibleActions(
  session: PlayerSession,
  actions: GameAction[]
): GameAction[] {
  return actions.map(action => {
    const metadata = action.metadata;
    if (!metadata) return action;

    // Filter metadata by capability
    const visibleMetadata: Partial<ActionMetadata> = {};

    if (metadata.hint && session.capabilities.has('see-hints')) {
      visibleMetadata.hint = metadata.hint;
    }

    if (metadata.aiIntent && session.capabilities.has('see-ai-intent')) {
      visibleMetadata.aiIntent = metadata.aiIntent;
    }

    // Always show recommendation if present
    if (metadata.recommendation) {
      visibleMetadata.recommendation = metadata.recommendation;
    }

    return {
      ...action,
      metadata: visibleMetadata
    };
  });
}
```

### Usage in GameHost

```typescript
private createView(forPlayerId?: string): GameView {
  const { state, sessions } = this.mpState;

  // Get all valid actions from composed variants
  const allValidActions = this.getValidActionsComposed(state);

  if (forPlayerId) {
    const session = this.players.get(forPlayerId);
    if (session) {
      // Apply capability filters
      let filtered = getExecutableActions(session, allValidActions);
      filtered = getVisibleActions(session, filtered);

      return { state, validActions: filtered, ... };
    }
  }

  // Unfiltered view (admin/debug)
  return { state, validActions: allValidActions, ... };
}
```

### Example Variant Using Capabilities

**File**: `src/game/variants/tutorial.ts`

```typescript
export const tutorialVariant: VariantFactory = () => (base) => (state) => {
  const actions = base(state);

  // Annotate actions with hints
  return actions.map(action => {
    if (action.type === 'bid') {
      return {
        ...action,
        metadata: {
          hint: getHintForBid(action, state),
          requiredCapability: 'see-hints',
          recommendation: evaluateBid(action, state) > 0.7 ? 'strong' : 'weak'
        }
      };
    }

    if (action.type === 'play') {
      return {
        ...action,
        metadata: {
          hint: getHintForPlay(action, state),
          aiIntent: getAIReasoning(action, state),
          requiredCapability: 'see-hints'
        }
      };
    }

    return action;
  });
};
```

### Session Creation with Capabilities

```typescript
function createPlayerSession(
  playerIndex: number,
  controlType: 'human' | 'ai',
  role: 'player' | 'spectator' | 'coach' = 'player'
): PlayerSession {
  const capabilities = new Set<Capability>();

  switch (role) {
    case 'player':
      capabilities.add('execute-action');
      capabilities.add('control-player');
      break;

    case 'spectator':
      capabilities.add('observe-all-hands');
      capabilities.add('see-hidden-info');
      break;

    case 'coach':
      capabilities.add('see-ai-intent');
      capabilities.add('see-hints');
      capabilities.add('observe-all-hands');
      break;
  }

  return {
    playerId: `${role}-${playerIndex}`,
    playerIndex,
    controlType,
    isConnected: true,
    capabilities
  };
}
```

---

## 3. URL Encoding for Event Sourcing ❌ NOT IMPLEMENTED

### Status: Old URL System Exists, Not Updated

**What Exists:**
- `src/game/core/url-compression.ts` with old encoding
- Encodes state snapshot (not config + actions)
- `tournamentMode` removed from encoding

**What's Missing:**
URL encoding that serializes `initialConfig + actionHistory` for perfect replay.

### Design Pattern

```
URL Format:
?c=<compressed-config>&a=<compressed-actions>

Where:
- c = base64(gzip(JSON.stringify(initialConfig)))
- a = base64(gzip(JSON.stringify(actionHistory)))
```

### Required Implementation

**File**: `src/game/core/url-encoding.ts` (NEW)

```typescript
import type { GameConfig } from '../../game/types/config';
import type { GameAction } from '../types';
import { replayActions } from './replay';

/**
 * Encode game state as URL parameters.
 * Uses initialConfig + actionHistory for perfect reproducibility.
 */
export function encodeGameUrl(state: GameState): string {
  const config = state.initialConfig;
  const actions = state.actionHistory;

  // Compress config
  const configJson = JSON.stringify(config);
  const configCompressed = gzipSync(configJson);
  const configEncoded = btoa(configCompressed);

  // Compress actions
  const actionsJson = JSON.stringify(actions);
  const actionsCompressed = gzipSync(actionsJson);
  const actionsEncoded = btoa(actionsCompressed);

  const params = new URLSearchParams();
  params.set('c', configEncoded);
  params.set('a', actionsEncoded);

  return `?${params.toString()}`;
}

/**
 * Decode URL parameters and reconstruct state.
 */
export function decodeGameUrl(url: string): GameState | null {
  const params = new URLSearchParams(url);

  const configEncoded = params.get('c');
  const actionsEncoded = params.get('a');

  if (!configEncoded) return null;

  try {
    // Decompress config
    const configCompressed = atob(configEncoded);
    const configJson = gunzipSync(configCompressed);
    const config: GameConfig = JSON.parse(configJson);

    // Decompress actions
    const actions: GameAction[] = actionsEncoded
      ? JSON.parse(gunzipSync(atob(actionsEncoded)))
      : [];

    // Replay to reconstruct state
    return replayActions(config, actions);
  } catch (error) {
    console.error('Failed to decode URL:', error);
    return null;
  }
}
```

### Integration

**Update stores to use URL state:**

```typescript
// src/stores/gameStore.ts - on page load
const urlState = decodeGameUrl(window.location.search);
if (urlState) {
  // Create client with replayed state
  gameClient = NetworkGameClient.fromState(adapter, urlState);
} else {
  // Normal initialization
  gameClient = new NetworkGameClient(adapter, config);
}
```

**Update URL on state changes:**

```typescript
// Watch for state changes and update URL
gameClient.subscribe(state => {
  clientState.set(state);

  // Update URL with current state
  const url = encodeGameUrl(state.state);
  window.history.replaceState(null, '', url);
});
```

---

## 4. Speed Mode Variant ❌ NOT IMPLEMENTED

### Status: Mentioned in Docs, Not Created

**What's Needed:**
A variant that auto-plays when only one legal move exists.

### Implementation

**File**: `src/game/variants/speedMode.ts` (NEW)

```typescript
import type { VariantFactory } from './types';

/**
 * Speed mode: Auto-execute when only one valid option.
 * Adds visual delay for clarity.
 */
export const speedModeVariant: VariantFactory = (config?) => (base) => (state) => {
  const actions = base(state);

  // Only affects playing phase
  if (state.phase !== 'playing') return actions;

  // Filter to play actions only
  const playActions = actions.filter(a => a.type === 'play');

  // If only one valid play, mark for auto-execution
  if (playActions.length === 1) {
    return [{
      ...playActions[0],
      autoExecute: true,
      delay: config?.delay || 300,  // ms delay for visual feedback
      meta: {
        scriptId: 'speed-mode-auto-play',
        reason: 'Only one legal play'
      }
    }];
  }

  // Multiple options: show all
  return actions;
};
```

**Register in `registry.ts`:**

```typescript
import { speedModeVariant } from './speedMode';

const VARIANT_REGISTRY: Record<string, VariantFactory> = {
  'tournament': tournamentVariant,
  'one-hand': oneHandVariant,
  'speed-mode': speedModeVariant,  // ADD
};
```

**Usage:**

```typescript
const config: GameConfig = {
  playerTypes: ['human', 'ai', 'ai', 'ai'],
  variant: {
    type: 'speed-mode',
    config: { delay: 500 }  // Half-second delay
  }
};
```

---

## 5. Daily Challenge Variant ❌ NOT IMPLEMENTED

### Status: Concept Only

**Design:**
- Fixed seed for daily challenge
- Single hand gameplay
- Star rating based on score
- Share button with URL

### Implementation Sketch

**File**: `src/game/variants/dailyChallenge.ts` (NEW)

```typescript
export const dailyChallengeVariant: VariantFactory = (config?) => (base) => (state) => {
  const actions = base(state);

  // Use one-hand variant as base behavior
  const oneHandActions = oneHandVariant()(base)(state);

  // After scoring, replace score-hand with end-game
  if (state.phase === 'scoring' && state.consensus.scoreHand.size === 4) {
    const score = state.teamScores[0];
    const stars = calculateStars(score, config?.targetScore || 42);

    return [{
      type: 'end-game',
      autoExecute: true,
      meta: {
        scriptId: 'daily-challenge-end',
        stars,
        shareText: `I scored ${score} points in today's Daily Challenge! ⭐`.repeat(stars)
      }
    }];
  }

  return oneHandActions;
};

function calculateStars(score: number, target: number): number {
  if (score >= target) return 3;      // Perfect or better
  if (score >= target * 0.8) return 2; // 80%+
  if (score >= target * 0.5) return 1; // 50%+
  return 0;
}
```

**Client handling:**

```typescript
// In UI component
if (action.type === 'end-game' && action.meta?.stars) {
  showChallengeComplete({
    stars: action.meta.stars,
    shareText: action.meta.shareText,
    shareUrl: window.location.href
  });
}
```

---

## 6. Seed Finder System ❌ NOT IMPLEMENTED

### Status: Removed During Refactor

**What Was Removed:**
- `InProcessAdapter.findCompetitiveSeed()` method
- Seed evaluation logic
- Progress reporting for seed search

**What's Needed:**
Compositional seed finder that works with variants.

### Design Pattern

```typescript
// Pure seed evaluator
type SeedEvaluator = (seed: number, config: GameConfig) => number;

// Compositional finder
async function findBestSeed(
  config: GameConfig,
  evaluator: SeedEvaluator,
  options: {
    maxAttempts?: number;
    minScore?: number;
    onProgress?: (progress: number) => void;
  }
): Promise<number> {
  let bestSeed = 0;
  let bestScore = -Infinity;

  for (let attempt = 0; attempt < options.maxAttempts; attempt++) {
    const seed = Math.floor(Math.random() * 1000000);
    const score = evaluator(seed, config);

    if (score > bestScore) {
      bestSeed = seed;
      bestScore = score;
    }

    if (score >= options.minScore) break;

    options.onProgress?.(attempt / options.maxAttempts);
  }

  return bestSeed;
}
```

**Evaluator Example:**

```typescript
// Evaluate one-hand seed for competitive balance
function evaluateOneHandSeed(seed: number, config: GameConfig): number {
  const state = replayActions({ ...config, shuffleSeed: seed }, []);

  // Criteria:
  // - Balanced hands (no player has overwhelming advantage)
  // - Interesting decisions (multiple viable bids)
  // - Not trivial (requires skill)

  const handScores = state.players.map(p => evaluateHand(p.hand));
  const variance = calculateVariance(handScores);
  const avgScore = average(handScores);

  // Want medium variance (not too even, not too lopsided)
  // Want decent average strength (interesting game)
  return (1 / (variance + 0.1)) * avgScore;
}
```

**Integration:**

```typescript
// In gameStore.ts
gameVariants.findAndStartOneHand = async () => {
  findingSeed.set(true);

  const seed = await findBestSeed(
    { playerTypes, variant: { type: 'one-hand' } },
    evaluateOneHandSeed,
    {
      maxAttempts: 100,
      minScore: 0.7,
      onProgress: (p) => console.log(`Finding seed: ${Math.round(p * 100)}%`)
    }
  );

  findingSeed.set(false);
  await gameVariants.startOneHand(seed);
};
```

---

## 7. Testing Gaps

### Missing Test Coverage

1. **Auto-execute handler tests**
   - Verify scripted sequences execute fully
   - Test max iteration limit guard
   - Test error handling mid-script

2. **Capability system tests**
   - Filter actions by execution capability
   - Strip metadata by viewing capability
   - Different roles (player/spectator/coach)

3. **URL encoding tests**
   - Encode/decode round-trip
   - Handle corrupt URLs gracefully
   - Verify exact state reconstruction

4. **Variant composition tests**
   - Multiple variants combined
   - Order of composition matters
   - Conflicting variants

5. **Event sourcing edge cases**
   - Very long action histories (performance)
   - Missing initialConfig
   - Invalid actions in history

### Test Patterns Needed

```typescript
// Auto-execute integration test
test('processes scripted action sequence', async () => {
  const host = new GameHost(gameId, config, players);

  // Should auto-exec until no more autoExecute actions
  const view = host.getView();

  expect(view.validActions.every(a => !a.autoExecute)).toBe(true);
});

// Capability filtering test
test('strips metadata for unauthorized viewers', () => {
  const spectator = createPlayerSession(0, 'human', 'spectator');
  const actions = [
    { type: 'bid', metadata: { hint: 'Bid 30!', requiredCapability: 'see-hints' } }
  ];

  const visible = getVisibleActions(spectator, actions);

  expect(visible[0].metadata?.hint).toBeUndefined();
});

// URL round-trip test
test('encodes and decodes state perfectly', () => {
  const original = createTestState();
  const url = encodeGameUrl(original);
  const decoded = decodeGameUrl(url);

  expect(decoded).toEqual(original);
  expect(decoded.actionHistory).toEqual(original.actionHistory);
});
```

---

## 8. Documentation Gaps

### Missing Docs

1. **Auto-execute protocol** - How variants signal auto-execution
2. **Capability reference** - All capabilities and their meanings
3. **Seed finder guide** - How to write evaluators
4. **URL format spec** - Encoding/compression details
5. **Migration guide** - Old variant system → new system

### Quick Docs Needed

**Auto-Execute Protocol:**
```markdown
# Auto-Execute Actions

Variants can mark actions for automatic execution:

```typescript
{
  type: 'bid',
  player: 3,
  bid: 'points',
  value: 30,
  autoExecute: true,      // Execute without user input
  delay: 300,             // Optional visual delay (ms)
  meta: {
    scriptId: 'setup',    // Debug identifier
    step: 1               // Sequence number
  }
}
```

GameHost processes these in `processAutoExecuteActions()` loop.
```

---

## Priority Ranking

### P0 - Critical for Core Functionality
1. **Auto-execute handler** - One-hand variant doesn't work without it
2. **Fix any failing tests** - Ensure refactor didn't break existing features

### P1 - High Value, Low Effort
3. **Speed mode variant** - Simple, high UX value
4. **URL encoding update** - Enables sharing/debugging

### P2 - Medium Priority
5. **Capability system** - Foundation for future features
6. **Daily challenge variant** - Nice-to-have

### P3 - Low Priority / Future
7. **Seed finder** - Useful but not essential
8. **Additional test coverage** - Incrementally improve

---

## Getting Started

### Implement Auto-Execute (P0)

1. Read this section: **#1. Auto-Execute Handler**
2. Add `processAutoExecuteActions()` to GameHost
3. Call it after `executeAction()`
4. Test with one-hand variant
5. Verify bidding sequence auto-runs

### Implement Speed Mode (P1)

1. Read this section: **#4. Speed Mode Variant**
2. Create `src/game/variants/speedMode.ts`
3. Register in `registry.ts`
4. Test in UI
5. Verify single-option plays auto-execute

### Test Everything

```bash
# Run tests after each feature
npm run typecheck  # Zero errors
npm test           # All passing
```

---

## Notes

- All deferred work is **compatible** with completed refactor
- No breaking changes required
- Incremental implementation possible
- Each feature can be tested in isolation

**Next Steps:** Pick from priority ranking, implement, test, repeat.
