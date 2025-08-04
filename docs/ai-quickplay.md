# AI Quickplay Specification

## Purpose
Add AI players that make moves automatically, using the same action system as human players. Supports continuous play through multiple games with automatic reset.

## Core Design
- AI is just another action source, like UI clicks
- Integrates via store subscription with requestAnimationFrame loop
- Reuses all existing game logic
- Deterministic execution without setTimeout/setInterval

## Implementation

### 1. Control Store
```typescript
interface QuickplayState {
  enabled: boolean;
  speed: 'instant' | 'fast' | 'normal' | 'slow';
  aiPlayers: Set<number>;  // 0-3
  isPaused: boolean;
}
```

### 2. Execution Model
Use requestAnimationFrame for deterministic, continuous processing:

```typescript
function processAIMoves() {
  const $gameState = get(gameState);
  const $quickplayState = get(quickplayState);
  
  // Check if game is complete - start new game automatically
  if ($gameState.phase === 'game_end' && $gameState.isComplete) {
    gameActions.resetGame();
    // Continue processing after reset
    animationFrameId = requestAnimationFrame(() => {
      animationFrameId = requestAnimationFrame(processAIMoves);
    });
    return;
  }
  
  // Process AI move if it's an AI player's turn
  if ($quickplayState.aiPlayers.has($gameState.currentPlayer)) {
    const decision = makeAIDecision($gameState, $availableActions);
    if (decision) {
      gameActions.executeAction(decision);
    }
  }
  
  // Continue loop
  animationFrameId = requestAnimationFrame(processAIMoves);
}
```

**Critical**: Use requestAnimationFrame instead of setTimeout for deterministic execution.

### 3. Decision Logic

#### Basic Heuristic Strategy

**Bidding**:
- Count points in hand (5-5, 6-4 = 10 pts; 5-0, 4-1, 3-2 = 5 pts)
- Count doubles (adds strength)
- Bid 30 for weak hands, 32-35 for strong hands
- Pass if can't beat current bid

**Trump Selection**:
- Use `getStrongestSuits()` from existing suit analysis
- Pick suit with most dominoes
- Consider doubles trump if 3+ doubles

**Playing**:
- Use `getValidPlays()` to get legal moves
- When leading: play high-point dominoes (5+ points)
- When following: play high to win, low to duck
- If can't follow suit: trump if beneficial, else discard low points

### 4. UI Controls
- Run/Stop button
- Pause/Resume button  
- Step button (one move at a time)
- Speed selector
- Player toggles (which players are AI)

### 5. Key Requirements

**Must Reuse**:
- `player.suitAnalysis` - Already computed by game
- `getValidPlays()` - Legal move filtering
- `getNextStates()` - Available actions

**Must Preserve**:
- Action history (for replay)
- URL encoding
- Debug UI compatibility

**Error Handling**:
- If AI fails to decide, use first available action
- Disable AI on repeated failures
- Never block the UI

### 6. Game Completion and Reset

**Automatic New Game**:
When games complete, quickplay automatically starts a new game:
- Detect `game_end` phase AND `isComplete` flag
- Call `gameActions.resetGame()` to clear state and history
- Continue AI processing loop without interruption

**Game End State Integrity**:
When transitioning to `game_end` phase:
```typescript
if (isGameComplete(newMarks, state.gameTarget)) {
  newState.phase = 'game_end';
  newState.isComplete = true;
  newState.winner = newMarks[0] >= state.gameTarget ? 0 : 1;
  // Clear hands since game is over
  newState.players.forEach(player => {
    player.hand = [];
  });
}
```

**Critical Lessons Learned**:
1. **Early Scoring Optimization**: The game may transition to scoring phase before all tricks are played if the outcome is mathematically determined. When this happens and a team reaches 7 marks, hands must be cleared to prevent invalid states.

2. **Phase vs Complete**: Check both `phase === 'game_end'` AND `isComplete === true` before resetting. This prevents premature resets during state transitions.

3. **Action History Management**: The reset properly clears action history, preventing accumulation across games. Monitor with debug logging if needed.

### 7. Speed Control

**Instant Speed (Default)**:
- No delays between moves
- Processes as fast as browser allows
- Ideal for testing and rapid game completion

**Speed Implementation**:
- For instant speed: Execute immediately in requestAnimationFrame
- For other speeds: Track last move timestamp and delay appropriately
- Never use setTimeout for game logic

## Benefits
- Automated testing (run many games quickly)
- Demo mode for new players
- Development testing tool
- No changes to core game logic
- Continuous multi-game sessions
- Clean state management between games

## Files to Reference

**Core Implementation**:
- `/src/stores/quickplayStore.ts` - Main AI quickplay logic
- `/src/game/core/gameEngine.ts` - See `applyScoreHand` for game end handling
- `/src/stores/gameStore.ts` - See `resetGame` for state clearing

**UI Components**:
- `/src/debug/components/DebugQuickplay.svelte` - Control panel
- `/src/App.svelte` - Integration with main layout

**Test Examples**:
- `/src/tests/e2e/helpers/playwrightHelper.ts` - Test helper methods
- `/src/tests/e2e/ai-quickplay-*.spec.ts` - Feature tests