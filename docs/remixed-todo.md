# Remixed Architecture - TODO List

**Last Updated**: 2025-01-18
**Status**: Remaining work after variant composition refactor
**Context**: Core refactor complete (composition, event sourcing, auto-execute, capabilities). This document tracks remaining features.

---

## Executive Summary

The variant composition refactor is complete with all core infrastructure in place. Architecture alignment with the vision document is now complete.

### Completed (2025-01-18)
- ✅ Result type standardization (`success` field everywhere)
- ✅ PlayerId type unification (string throughout)
- ✅ Standard capability builders (Vision §4.3)
- ✅ `getValidActionsForPlayer` pure function (Vision §3.2.3)
- ✅ Documentation updates (GAME_ONBOARDING.md, CAPABILITY_SYSTEM.md)

### Remaining Work
- URL encoding updates for event sourcing (config + actions format)
- Speed mode variant (infrastructure ready, needs implementation)
- Show hints variant (infrastructure ready, needs implementation)
- Daily challenge variant
- Seed finder system
- Test coverage for completed features

---

## 1. URL Encoding for Event Sourcing ❌ NOT IMPLEMENTED

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

## 2. Speed Mode Variant ❌ NOT IMPLEMENTED

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

## 3. Daily Challenge Variant ❌ NOT IMPLEMENTED

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

## 4. Seed Finder System ❌ NOT IMPLEMENTED

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

## 5. Architecture Alignment Completed ✅

### Status: COMPLETE (2025-01-18)

These items from the vision document (remixed-855ccfd5.md) have been fully implemented:

### 5.1 Result Type Standardization

**Problem**: Two incompatible `Result<T>` types existed
- Protocol layer used `{ ok: true/false }`
- Multiplayer layer used `{ success: true/false }`

**Solution**: Standardized on `{ success: true/false }` everywhere
- Updated `src/shared/multiplayer/protocol.ts:278-293`
- Updated `src/server/game/GameHost.ts:118`
- Fixed all tests to use `.success` instead of `.ok`

**Files Changed**:
- `src/shared/multiplayer/protocol.ts`
- `src/server/game/GameHost.ts`
- `src/tests/unit/authorization.test.ts`
- `src/tests/unit/protocol-architecture.test.ts`

### 5.2 PlayerId Type Unification

**Problem**: Inconsistent player identification
- `GameClient.requestAction()` took `number`
- But `PlayerSession.playerId` is `string`
- NetworkGameClient ignored the parameter entirely

**Solution**: Use `string` playerId consistently
- Changed `GameClient.requestAction(playerId: string, ...)`
- Updated `NetworkGameClient` to use provided playerId
- Updated all call sites to pass string IDs (e.g., "player-0", "ai-1")

**Files Changed**:
- `src/game/multiplayer/GameClient.ts:30`
- `src/game/multiplayer/NetworkGameClient.ts:79`
- `src/stores/gameStore.ts:198,220`
- `src/tests/unit/protocol-architecture.test.ts:163`

### 5.3 Standard Capability Builders (Vision §4.3)

**Created**: `src/game/multiplayer/capabilities.ts`

Provides standard builders:
- `humanCapabilities(playerIndex)` - Human player
- `aiCapabilities(playerIndex)` - AI with replace-ai
- `spectatorCapabilities()` - Observe all, no actions
- `coachCapabilities(studentIndex)` - See student hand + hints
- `tutorialCapabilities(playerIndex)` - Player + hints + undo
- `CapabilityBuilder` - Fluent API for custom composition

**Integrated in**:
- `src/server/game/GameHost.ts:389-393`
- `src/server/game/createGameAuthority.ts:34-37`

### 5.4 getValidActionsForPlayer Pure Function (Vision §3.2.3)

**Added**: `src/game/multiplayer/authorization.ts:71-92`

Pure function that:
- Takes `MultiplayerGameState` and `playerId`
- Composes variants with base state machine
- Filters by player's capabilities
- Returns array of executable actions

Signature:
```typescript
getValidActionsForPlayer(
  mpState: MultiplayerGameState,
  playerId: string,
  getValidActionsFn?: StateMachine
) → GameAction[]
```

### 5.5 Documentation Updates

**Updated**: `docs/GAME_ONBOARDING.md`
- Added capability builders section with examples
- Updated multiplayer section with `getValidActionsForPlayer`
- Fixed Result type documentation (`success` not `ok`)
- Updated action flow with string playerIds
- Added quick reference entries

**Created**: `docs/CAPABILITY_SYSTEM.md`
- Comprehensive capability reference
- All standard builders documented
- Usage examples for each player type
- Integration patterns
- Testing examples

### 5.6 Test Fixes

**Fixed**: All TypeScript compilation errors
- Added `timestamp` fields to `ActionRequest` objects
- Updated Result assertions to use `.success`
- Fixed `requestAction` signatures to use string playerIds

**Verification**: `npm run typecheck` passes with zero errors

---

## 6. Testing Gaps

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

## 7. Documentation Gaps

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

### P1 - High Value, Low Effort
1. **Speed mode variant** - Simple, high UX value
2. **URL encoding update** - Enables sharing/debugging (config + actions format)

### P2 - Medium Priority
3. **Daily challenge variant** - Nice-to-have

### P3 - Low Priority / Future
4. **Seed finder** - Useful but not essential
5. **Additional test coverage** - Incrementally improve

---

## Getting Started

### Implement Speed Mode (P1)

1. Read this section: **#2. Speed Mode Variant**
2. Create `src/game/variants/speedMode.ts`
3. Register in `registry.ts`
4. Test in UI
5. Verify single-option plays auto-execute

### Implement URL Encoding (P1)

1. Read this section: **#1. URL Encoding for Event Sourcing**
2. Update `src/game/core/url-compression.ts` to use config + actions format
3. Test encode/decode round-trip
4. Verify exact state reconstruction

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
