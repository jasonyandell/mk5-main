# Unified Transition System & Section-Based Gameplay

## Current Problems

### Timing Issues
The game currently has multiple transition execution paths that cause timing inconsistencies:

1. **Human Actions**: UI → ActionPanel → HumanController → gameActions.executeAction()
2. **AI Actions**: Game Loop → tickGame() → executeScheduledAIActions() → gameActions.executeAction()  
3. **Test Mode**: Direct execution bypassing controllers
4. **URL Replay**: Validates and replays transitions sequentially
5. **Skip AI Delays**: Separate path for skipping delays

This fragmentation causes:
- Skips and timing glitches that break immersion
- Difficulty debugging timing-related issues
- Complex code with multiple timing mechanisms (ticks, delays, schedules)
- Inability to easily compose game sections

### Section-Based Gameplay Needs
We need the ability to:
- Play arbitrary sections (one trick, one hand, one bid round)
- Intercept and control automatic transitions
- Show intro/outro screens for sections
- Create practice scenarios from any game state
- Step through games for tutorials

## Solution: Unified Transition Engine

### Core Architecture

```typescript
// 1. Single transition pipeline - ALL transitions flow through here
class TransitionEngine {
  private interceptors: TransitionInterceptor[] = [];
  private mode: ExecutionMode = 'normal';
  private delayStrategy: DelayStrategy;
  
  async executeTransition(transition: StateTransition): Promise<GameState> {
    // Pre-execution hooks (can block)
    for (const interceptor of this.interceptors) {
      if (!await interceptor.beforeTransition(transition)) {
        return null; // Blocked by interceptor
      }
    }
    
    // Apply delay if needed (replaces tick system)
    await this.delayStrategy.applyDelay(transition, this.mode);
    
    // Pure state transition
    const newState = applyPureTransition(transition);
    
    // Post-execution hooks
    for (const interceptor of this.interceptors) {
      await interceptor.afterTransition(transition, newState);
    }
    
    // Update stores/URL/history in one place
    this.commitState(newState, transition);
    
    return newState;
  }
  
  private commitState(state: GameState, transition: StateTransition) {
    // Single place for all state updates
    gameState.set(state);
    actionHistory.update(h => [...h, transition]);
    updateURL(state, transition);
  }
}

// 2. Execution modes for different contexts
enum ExecutionMode {
  NORMAL,        // Full game with delays
  INSTANT,       // No delays (testing)
  STEP_BY_STEP,  // Pause after each transition
  SECTION,       // Run specific section only
  REPLAY         // Replay from history
}

// 3. Transition interceptor interface
interface TransitionInterceptor {
  // Return false to block transition
  beforeTransition(transition: StateTransition): Promise<boolean>;
  afterTransition(transition: StateTransition, newState: GameState): Promise<void>;
}
```

### Simple Delay Strategy

Replace the complex tick-based system with inline delays:

```typescript
class DelayStrategy {
  async applyDelay(transition: StateTransition, mode: ExecutionMode) {
    // Skip delays in certain modes
    if (mode === 'instant' || mode === 'replay') return;
    
    const delayMs = this.getDelay(transition);
    if (delayMs > 0) {
      await sleep(delayMs);
    }
  }
  
  private getDelay(transition: StateTransition): number {
    // Human actions have no delay
    if (transition.action.player === 'human') return 0;
    
    // Simple, predictable delays by action type
    const delays = {
      'bid': 800,
      'pass': 600,
      'play': 500,
      'select-trump': 1000,
      'redeal': 1500
    };
    
    return delays[transition.action.type] || 500;
  }
}
```

## Section-Based Gameplay System

### Section Runner

```typescript
interface SectionConfig {
  id: 'trick' | 'hand' | 'bid-round' | 'game' | 'custom';
  initialState: GameState;
  endCondition: (state: GameState) => boolean;
  intro?: IntroConfig;
  outro?: (state: GameState) => OutroConfig;
  interceptors?: TransitionInterceptor[];
}

class SectionRunner {
  constructor(private engine: TransitionEngine) {}
  
  async runSection(config: SectionConfig): Promise<GameState> {
    let state = config.initialState;
    
    // Show intro if provided
    if (config.intro) {
      await this.showIntro(config.intro);
    }
    
    // Add section-specific interceptors
    config.interceptors?.forEach(i => this.engine.addInterceptor(i));
    
    // Run until end condition
    while (!config.endCondition(state)) {
      const transition = await this.getNextTransition(state);
      if (!transition) break; // No valid transitions
      
      state = await this.engine.executeTransition(transition);
      if (!state) break; // Interceptor blocked
    }
    
    // Remove section interceptors
    config.interceptors?.forEach(i => this.engine.removeInterceptor(i));
    
    // Show outro if provided
    if (config.outro) {
      await this.showOutro(config.outro(state));
    }
    
    return state;
  }
  
  private async getNextTransition(state: GameState): Promise<StateTransition> {
    const availableTransitions = getNextStates(state);
    if (availableTransitions.length === 0) return null;
    
    const currentPlayer = state.currentPlayer;
    
    if (state.playerTypes[currentPlayer] === 'human') {
      // Wait for UI click (via promise)
      return await waitForHumanAction(availableTransitions);
    } else {
      // AI decides immediately (no tick scheduling)
      return selectAIAction(state, availableTransitions);
    }
  }
}
```

### Game State Builder

Easy creation of arbitrary game states for sections:

```typescript
class GameStateBuilder {
  // Create a state in the middle of a trick
  static createTrickScenario(config: {
    trump: TrumpSelection,
    currentPlayer: number,
    cardsPlayed: Domino[],
    hands: Domino[][],
    scores?: [number, number]
  }): GameState {
    // Build valid state for mid-trick scenario
    const state = createInitialState();
    state.phase = 'playing';
    state.trump = config.trump;
    state.currentPlayer = config.currentPlayer;
    state.currentTrick.plays = config.cardsPlayed.map((d, i) => ({
      player: i,
      domino: d
    }));
    state.hands = config.hands;
    if (config.scores) state.teamScores = config.scores;
    return state;
  }
  
  // Create a bidding scenario
  static createBiddingScenario(config: {
    currentBidder: number,
    previousBids: Bid[],
    hands: Domino[][],
    dealer?: number
  }): GameState {
    const state = createInitialState();
    state.phase = 'bidding';
    state.currentPlayer = config.currentBidder;
    state.bids = config.previousBids;
    state.hands = config.hands;
    state.dealer = config.dealer ?? 0;
    state.currentBid = config.previousBids[config.previousBids.length - 1] || EMPTY_BID;
    return state;
  }
  
  // Create a "going set" scenario
  static createNearSetScenario(config: {
    bidAmount: number,
    currentScore: number,
    tricksLeft: number,
    bidder: number,
    hands: Domino[][]
  }): GameState {
    const state = createInitialState();
    state.phase = 'playing';
    state.winningBid = { type: 'points', value: config.bidAmount, player: config.bidder };
    state.winningBidder = config.bidder;
    state.teamScores = config.bidder % 2 === 0 
      ? [config.currentScore, 0] 
      : [0, config.currentScore];
    state.hands = config.hands;
    state.tricksRemaining = config.tricksLeft;
    return state;
  }
  
  // Validate state is legal/reachable
  static validate(state: GameState): ValidationResult {
    const errors: string[] = [];
    
    // Check hand sizes match phase
    const totalCards = state.hands.reduce((sum, hand) => sum + hand.length, 0);
    const expectedCards = state.phase === 'bidding' ? 28 : (28 - state.completedTricks.length * 4);
    if (totalCards !== expectedCards) {
      errors.push(`Invalid card count: ${totalCards} vs ${expectedCards} expected`);
    }
    
    // Check current player is valid
    if (state.currentPlayer < 0 || state.currentPlayer > 3) {
      errors.push(`Invalid current player: ${state.currentPlayer}`);
    }
    
    return { valid: errors.length === 0, errors };
  }
}
```

## Usage Examples

### Example 1: Normal Game (Fixes Timing)
```typescript
// Initialize engine for normal gameplay
const engine = new TransitionEngine();
engine.setMode('normal');

// All transitions now flow through one path
// No more timing glitches or skips
```

### Example 2: Play One Trick
```typescript
const trickSection: SectionConfig = {
  id: 'trick',
  initialState: GameStateBuilder.createTrickScenario({
    trump: { type: 'suit', suit: 'fives' },
    currentPlayer: 2,
    cardsPlayed: [
      { high: 5, low: 2 }, // P0 led
      { high: 5, low: 5 }  // P1 played
    ],
    hands: [
      [],
      [],
      [{ high: 5, low: 3 }, { high: 3, low: 1 }], // P2 must follow suit
      [{ high: 6, low: 6 }] // P3
    ]
  }),
  endCondition: (state) => state.currentTrick.plays.length === 4,
  intro: {
    title: "Complete the Trick",
    message: "Player 2 must follow the led suit (fives)"
  },
  outro: (state) => ({
    title: `Player ${state.currentTrick.winner} Wins!`,
    message: `Trick worth ${state.currentTrick.points} points`
  })
};

const runner = new SectionRunner(engine);
await runner.runSection(trickSection);
```

### Example 3: Practice Going Set
```typescript
const goingSetPractice: SectionConfig = {
  id: 'hand',
  initialState: GameStateBuilder.createNearSetScenario({
    bidAmount: 35,
    currentScore: 20,
    tricksLeft: 1,
    bidder: 0,
    hands: [
      [{ high: 2, low: 0 }], // Weak hand for bidder
      [{ high: 6, low: 6 }], // Strong for opponent
      [{ high: 1, low: 0 }], // Weak for partner
      [{ high: 5, low: 5 }]  // Trump for opponent
    ]
  }),
  endCondition: (state) => state.phase === 'scoring',
  intro: {
    title: "Avoid Going Set!",
    message: "You bid 35 but only have 20 points. This last trick is worth 22!"
  },
  outro: (state) => {
    const wentSet = state.teamScores[0] < 35;
    return {
      title: wentSet ? "WENT SET!" : "Made It!",
      message: wentSet 
        ? `Only scored ${state.teamScores[0]}, missing bid by ${35 - state.teamScores[0]}`
        : "You made your bid!",
      showReplay: true
    };
  }
};

await runner.runSection(goingSetPractice);
```

### Example 4: Step-by-Step Tutorial
```typescript
// Configure engine for tutorial mode
engine.setMode('step-by-step');

// Add educational interceptor
const tutorialInterceptor: TransitionInterceptor = {
  async afterTransition(transition, state) {
    // Explain what happened
    const explanation = explainTransition(transition, state);
    await ui.showExplanation({
      title: `Player ${transition.action.player} Action`,
      message: explanation,
      highlights: getRelevantCards(transition, state)
    });
    
    // Wait for user to continue
    await ui.waitForContinue();
  }
};

engine.addInterceptor(tutorialInterceptor);
```

### Example 5: Instant Testing
```typescript
// Run game instantly for tests
engine.setMode('instant');

// No delays - runs as fast as possible
const finalState = await runner.runSection({
  id: 'game',
  initialState: createInitialState(),
  endCondition: (state) => state.phase === 'game_end'
});

// Validate results
expect(finalState.teamScores).toEqual([42, 0]);
```

## Migration Strategy

### Phase 1: Unified Engine (Fix Timing)
1. Create `TransitionEngine` class in `src/game/core/transition-engine.ts`
2. Create `DelayStrategy` class to replace tick system
3. Route all transitions through engine in `gameStore.ts`
4. Remove game loop and tick-based AI scheduling
5. Remove multiple execution paths

### Phase 2: Section Runner (Enable Sections)
1. Create `SectionRunner` class in `src/game/sections/section-runner.ts`
2. Create `GameStateBuilder` helpers in `src/game/sections/state-builder.ts`
3. Implement section boundary detection
4. Add intro/outro UI components

### Phase 3: Enhanced Features
1. Create practice scenarios
2. Add replay controls
3. Build tutorial system
4. Create scenario library

## Benefits

### Immediate
- **Fixes all timing issues** - Single execution path with predictable delays
- **Simpler codebase** - Remove complex tick system and multiple paths
- **Better debugging** - All transitions logged in one place

### Future
- **Section-based play** - Easy to implement practice modes
- **Tutorials** - Step-by-step explanations
- **Testing** - Instant mode for fast tests
- **Replay** - Full replay system with controls
- **Composability** - Mix and match game sections

## Implementation Notes

### Key Changes to Existing Code

```typescript
// gameStore.ts - Simplify to single execution
export const gameActions = {
  async executeAction(transition: StateTransition) {
    await engine.executeTransition(transition);
  }
  // Remove: skipAIDelays, complex tick logic
};

// ActionPanel.svelte - Direct call
async function executeAction(action: StateTransition) {
  await gameActions.executeAction(action);
}

// Remove entirely:
// - Game loop (runGameLoop)
// - Tick-based AI scheduling
// - Controller callbacks for execution
// - Multiple execution paths
```

### Testing Strategy

1. Create engine with `instant` mode for tests
2. Use `GameStateBuilder` for test scenarios
3. Run sections instead of full games where appropriate
4. Validate state after each section

## Summary

This unified system solves current timing problems while enabling powerful new features. The migration can be incremental but should prioritize fixing the foundation (Phase 1) before adding new capabilities. The result will be a cleaner, more maintainable, and more feature-rich codebase.