# Unified Transition System - Best Approach

This document combines the best aspects of both architectural plans to create a pragmatic, low-risk path to fixing timing issues and enabling section-based gameplay.

## Executive Summary

**Approach**: Wrap existing system with a dispatcher (safer) while planning for future simplification.
**Priority**: Fix timing issues first, then enable sections, then optimize.
**Key Insight**: Don't replace working code - augment and control it.

## Core Problems We're Solving

1. **Multiple transition paths** causing timing inconsistencies and skips
2. **Competing AI drivers** (game loop vs quickplay) causing races
3. **No way to intercept/control** automatic transitions (consensus actions)
4. **Can't play bounded sections** (one trick, one hand, etc.)
5. **Difficult to compose** pre/post UI states around game segments

## Architecture Overview

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
│   UI/User   │────▶│  Dispatcher      │────▶│ gameActions    │
└─────────────┘     │  - Gate/Queue    │     │ .executeAction │
                    │  - Source Track   │     └────────────────┘
┌─────────────┐     │  - Events        │              │
│  AI Loop    │────▶│                  │              ▼
└─────────────┘     └──────────────────┘     ┌────────────────┐
                            ▲                 │  Game State    │
┌─────────────┐             │                 └────────────────┘
│  Section    │─────────────┘
│  Runner     │  (sets gate & listeners)
└─────────────┘
```

## Phase 1: Transition Dispatcher (Fix Timing)

### Core Dispatcher

Single entry point for ALL transitions. Wraps existing `gameActions.executeAction` to preserve working code.

```typescript
// src/game/core/transition-dispatcher.ts
import type { GameState, StateTransition } from '../types';

export type TransitionSource = 'ui' | 'ai' | 'replay' | 'system' | 'test';

export interface TransitionContext {
  prev: GameState;
  next: GameState;
  transition: StateTransition;
  source: TransitionSource;
}

export type TransitionListener = (ctx: TransitionContext) => void | Promise<void>;
export type GateFunction = (t: StateTransition, state: GameState) => boolean;

export class TransitionDispatcher {
  private beforeListeners: TransitionListener[] = [];
  private afterListeners: TransitionListener[] = [];
  private gate: GateFunction = () => true;
  private queue: Array<{ transition: StateTransition; source: TransitionSource }> = [];
  private executing = false;

  constructor(
    private executeAction: (t: StateTransition) => void,
    private getState: () => GameState
  ) {}

  // Main entry point - ALL transitions go through here
  async requestTransition(transition: StateTransition, source: TransitionSource): Promise<void> {
    const state = this.getState();
    
    // Check gate
    if (!this.gate(transition, state)) {
      this.queue.push({ transition, source });
      console.log(`[Dispatcher] Queued ${transition.id} from ${source}`);
      return;
    }

    await this.execute(transition, source);
  }

  private async execute(transition: StateTransition, source: TransitionSource): Promise<void> {
    // Prevent re-entrance
    if (this.executing) {
      this.queue.push({ transition, source });
      return;
    }

    this.executing = true;
    const prev = this.getState();

    try {
      // Before listeners (can inspect but not block)
      for (const listener of this.beforeListeners) {
        await listener({ prev, next: prev, transition, source });
      }

      // Execute through existing system
      this.executeAction(transition);
      
      const next = this.getState();

      // After listeners
      for (const listener of this.afterListeners) {
        await listener({ prev, next, transition, source });
      }
    } finally {
      this.executing = false;
      // Process queue after execution
      this.processQueue();
    }
  }

  // Gate management
  setGate(gate: GateFunction): void {
    this.gate = gate;
    this.processQueue(); // Re-check queued items
  }

  clearGate(): void {
    this.gate = () => true;
    this.processQueue();
  }

  // Queue management
  private processQueue(): void {
    if (this.executing) return;
    
    const pending = [...this.queue];
    this.queue = [];
    
    for (const { transition, source } of pending) {
      const state = this.getState();
      if (this.gate(transition, state)) {
        // Don't await - let them process in sequence
        this.execute(transition, source);
      } else {
        this.queue.push({ transition, source });
      }
    }
  }

  flushQueue(): void {
    this.queue = [];
  }

  // Event management
  onBefore(listener: TransitionListener): () => void {
    this.beforeListeners.push(listener);
    return () => {
      const idx = this.beforeListeners.indexOf(listener);
      if (idx >= 0) this.beforeListeners.splice(idx, 1);
    };
  }

  onAfter(listener: TransitionListener): () => void {
    this.afterListeners.push(listener);
    return () => {
      const idx = this.afterListeners.indexOf(listener);
      if (idx >= 0) this.afterListeners.splice(idx, 1);
    };
  }

  // Debugging
  getQueueSize(): number {
    return this.queue.length;
  }

  getQueuedTransitions(): ReadonlyArray<StateTransition> {
    return this.queue.map(q => q.transition);
  }
}
```

### Wire Into Existing System

Minimal changes to route through dispatcher:

```typescript
// src/stores/gameStore.ts
import { TransitionDispatcher } from '../game/core/transition-dispatcher';

// Create dispatcher wrapping existing execution
export const dispatcher = new TransitionDispatcher(
  (t) => gameActions.executeAction(t),
  () => get(gameState)
);

// Update game loop to use dispatcher
const runGameLoop = () => {
  const currentState = get(gameState);
  const result = tickGame(currentState);
  
  if (result.state.currentTick !== currentState.currentTick) {
    gameState.set(result.state);
  }
  
  // Route AI actions through dispatcher
  if (result.action) {
    dispatcher.requestTransition(result.action, 'ai');
  }
  
  if (gameLoopRunning) {
    animationFrame = requestAnimationFrame(runGameLoop);
  }
};
```

```typescript
// src/lib/components/ActionPanel.svelte
// Change from: gameActions.executeAction(action)
// To:
import { dispatcher } from '../../stores/gameStore';
await dispatcher.requestTransition(action, 'ui');
```

```typescript
// src/game/controllers/HumanController.ts
// Change executeTransition callback to use dispatcher
dispatcher.requestTransition(transition, 'ui');
```

## Phase 2: Stop Conditions DSL

Composable predicates for defining section boundaries:

```typescript
// src/game/core/stop-conditions.ts
import type { GameState, StateTransition } from '../types';

export interface StopContext {
  prev: GameState;
  next: GameState;
  transition: StateTransition;
  count: number; // Number of transitions executed in section
}

export type StopCondition = (ctx: StopContext) => boolean;

// Basic conditions
export const after = (n: number): StopCondition => 
  ({ count }) => count >= n;

export const afterFirst = (): StopCondition => 
  after(1);

export const whenType = (...types: string[]): StopCondition =>
  ({ transition }) => types.includes(transition.action.type);

export const whenId = (pattern: RegExp | string): StopCondition =>
  ({ transition }) => typeof pattern === 'string' 
    ? transition.id === pattern
    : pattern.test(transition.id);

// Game-specific conditions
export const afterPlays = (n: number): StopCondition => {
  let playCount = 0;
  return ({ transition }) => {
    if (transition.action.type === 'play') playCount++;
    return playCount >= n;
  };
};

export const afterTrickComplete = (): StopCondition =>
  ({ prev, next }) => 
    prev.currentTrick.plays.length > 0 && 
    next.currentTrick.plays.length === 0;

export const whenPhase = (...phases: GameState['phase'][]): StopCondition =>
  ({ next }) => phases.includes(next.phase);

export const whenHandComplete = (): StopCondition =>
  whenPhase('scoring', 'game_end');

export const whenGameComplete = (): StopCondition =>
  ({ next }) => next.phase === 'game_end';

// Combinators
export const or = (...conditions: StopCondition[]): StopCondition =>
  (ctx) => conditions.some(c => c(ctx));

export const and = (...conditions: StopCondition[]): StopCondition =>
  (ctx) => conditions.every(c => c(ctx));

export const not = (condition: StopCondition): StopCondition =>
  (ctx) => !condition(ctx);

// Special conditions for consensus
export const isConsensusAction = (t: StateTransition): boolean =>
  t.id.startsWith('agree-') || 
  t.id === 'complete-trick' || 
  t.id === 'complete-hand';
```

## Phase 3: Section Runner

Manages bounded gameplay sections with pre/post UI hooks:

```typescript
// src/game/core/section-runner.ts
import type { GameState, StateTransition } from '../types';
import { dispatcher } from '../../stores/gameStore';
import type { GateFunction } from './transition-dispatcher';
import type { StopCondition } from './stop-conditions';

export type ConsensusPolicy = 'allow' | 'hold' | 'inject';
export type AISpeed = 'instant' | 'fast' | 'normal' | 'slow';

export interface SectionConfig {
  name: string;
  
  // What transitions to allow
  allow?: GateFunction;
  
  // When to stop
  stopWhen: StopCondition;
  
  // How to handle consensus actions
  consensus?: ConsensusPolicy;
  
  // AI timing for this section
  aiSpeed?: AISpeed;
  
  // UI hooks
  beforeSection?: (state: GameState) => Promise<void>;
  afterSection?: (state: GameState, transitions: StateTransition[]) => Promise<void>;
}

export interface SectionHandle {
  done: Promise<SectionResult>;
  cancel: () => void;
  pause: () => void;
  resume: () => void;
  getProgress: () => { count: number; transitions: StateTransition[] };
}

export interface SectionResult {
  finalState: GameState;
  transitions: StateTransition[];
  cancelled: boolean;
}

export class SectionRunner {
  private activeSection: SectionHandle | null = null;

  async runSection(config: SectionConfig): Promise<SectionResult> {
    // Cancel any active section
    if (this.activeSection) {
      this.activeSection.cancel();
    }

    const transitions: StateTransition[] = [];
    let count = 0;
    let cancelled = false;
    let resolver: (result: SectionResult) => void;
    
    const done = new Promise<SectionResult>(r => { resolver = r; });

    // Show before UI if provided
    if (config.beforeSection) {
      const currentState = get(gameState);
      await config.beforeSection(currentState);
    }

    // Set up gate based on allow and consensus policy
    const baseAllow = config.allow || (() => true);
    const gateFunction: GateFunction = (t, state) => {
      // Handle consensus policy
      if (config.consensus === 'hold' && isConsensusAction(t)) {
        return false; // Queue consensus actions
      }
      return baseAllow(t, state);
    };

    dispatcher.setGate(gateFunction);

    // Listen for transitions
    const unsubscribe = dispatcher.onAfter(async ({ prev, next, transition }) => {
      transitions.push(transition);
      count++;

      // Check stop condition
      const shouldStop = config.stopWhen({ 
        prev, 
        next, 
        transition, 
        count 
      });

      if (shouldStop) {
        // Clean up
        unsubscribe();
        dispatcher.clearGate();
        
        // Show after UI if provided
        if (config.afterSection) {
          await config.afterSection(next, transitions);
        }

        // Resolve promise
        resolver({
          finalState: next,
          transitions,
          cancelled: false
        });
      }
    });

    // Set AI speed if specified
    if (config.aiSpeed) {
      setAISpeedProfile(config.aiSpeed);
    }

    // Handle for control
    const handle: SectionHandle = {
      done,
      cancel: () => {
        if (!cancelled) {
          cancelled = true;
          unsubscribe();
          dispatcher.clearGate();
          const currentState = get(gameState);
          resolver({
            finalState: currentState,
            transitions,
            cancelled: true
          });
        }
      },
      pause: () => dispatcher.setGate(() => false),
      resume: () => dispatcher.setGate(gateFunction),
      getProgress: () => ({ count, transitions: [...transitions] })
    };

    this.activeSection = handle;
    return done;
  }

  // Preset sections for common use cases
  static oneTransition(): SectionConfig {
    return {
      name: 'oneTransition',
      stopWhen: afterFirst(),
      consensus: 'hold'
    };
  }

  static onePlay(): SectionConfig {
    return {
      name: 'onePlay',
      allow: (t) => t.action.type === 'play',
      stopWhen: afterPlays(1),
      consensus: 'hold'
    };
  }

  static oneTrick(): SectionConfig {
    return {
      name: 'oneTrick',
      allow: (t) => 
        t.action.type === 'play' || 
        t.id === 'complete-trick' ||
        t.id.startsWith('agree-complete-trick'),
      stopWhen: afterTrickComplete(),
      consensus: 'allow' // Need consensus to complete trick
    };
  }

  static oneHand(): SectionConfig {
    return {
      name: 'oneHand',
      stopWhen: whenHandComplete(),
      consensus: 'allow'
    };
  }

  static fullGame(): SectionConfig {
    return {
      name: 'fullGame',
      stopWhen: whenGameComplete(),
      consensus: 'allow'
    };
  }
}
```

## Phase 4: AI Speed Control

Centralized speed control that both game loop and sections can use:

```typescript
// src/game/core/ai-scheduler.ts (additions)
export type AISpeedProfile = 'instant' | 'fast' | 'normal' | 'slow';

let currentSpeedProfile: AISpeedProfile = 'normal';

export function setAISpeedProfile(profile: AISpeedProfile): void {
  currentSpeedProfile = profile;
}

export function getAIDelayTicks(action: StateTransition): number {
  if (currentSpeedProfile === 'instant') return 0;
  
  const baseDelays = {
    'bid': { instant: 0, fast: 300, normal: 800, slow: 1500 },
    'pass': { instant: 0, fast: 200, normal: 600, slow: 1000 },
    'play': { instant: 0, fast: 200, normal: 500, slow: 1000 },
    'select-trump': { instant: 0, fast: 500, normal: 1000, slow: 2000 }
  };
  
  const delays = baseDelays[action.action.type] || 
    { instant: 0, fast: 200, normal: 500, slow: 1000 };
  
  const baseDelay = delays[currentSpeedProfile];
  
  // Add some randomness for realism (except instant)
  if (currentSpeedProfile !== 'instant') {
    return Math.ceil((baseDelay + Math.random() * baseDelay * 0.5) / 16.67);
  }
  
  return 0;
}

// Also export for UI/testing
export function skipAllAIDelays(): void {
  const state = get(gameState);
  const newState = skipAIDelays(state);
  if (newState !== state) {
    gameState.set(newState);
  }
}
```

## Usage Examples

### Example 1: Fix Current Game (Immediate)
```typescript
// Just routing through dispatcher fixes timing issues
// No other changes needed initially
```

### Example 2: Practice One Trick
```typescript
import { SectionRunner } from '../game/core/section-runner';
import { GameStateBuilder } from '../game/core/state-builder';

// Set up a trick scenario
const trickState = GameStateBuilder.createTrickScenario({
  trump: { type: 'suit', suit: 'fives' },
  currentPlayer: 2,
  cardsPlayed: [
    { high: 5, low: 2 }, // P0 led fives
    { high: 5, low: 5 }  // P1 played double-five
  ],
  hands: [
    [],
    [],
    [{ high: 5, low: 3 }, { high: 3, low: 1 }], // P2's hand
    [{ high: 6, low: 6 }] // P3's hand
  ]
});

// Load the state
gameActions.loadState(trickState);

// Run the section
const runner = new SectionRunner();
const result = await runner.runSection({
  ...SectionRunner.oneTrick(),
  beforeSection: async (state) => {
    await showOverlay({
      title: "Practice: Complete the Trick",
      message: "Fives are trump. P1 played double-five. Can you win?"
    });
  },
  afterSection: async (state, transitions) => {
    const winner = state.completedTricks[state.completedTricks.length - 1].winner;
    await showOverlay({
      title: winner === 2 ? "Well Done!" : "Try Again",
      message: `Player ${winner} won the trick`
    });
  }
});
```

### Example 3: Speed Run Mode
```typescript
// Quickplay refactored to use sections
export function startQuickplay() {
  const runner = new SectionRunner();
  
  return runner.runSection({
    ...SectionRunner.fullGame(),
    aiSpeed: 'fast',
    beforeSection: async () => {
      await showNotification("Starting quick game...");
    },
    afterSection: async (state) => {
      const winner = state.winningTeam;
      await showNotification(`Team ${winner} wins!`);
    }
  });
}
```

### Example 4: Tutorial - Step by Step
```typescript
const tutorialSection: SectionConfig = {
  name: 'tutorial',
  stopWhen: or(
    afterPlays(4), // One trick
    whenType('bid', 'pass') // Or one bid action
  ),
  consensus: 'hold',
  aiSpeed: 'slow'
};

// Add explanation after each action
dispatcher.onAfter(async ({ transition, next }) => {
  const explanation = explainTransition(transition, next);
  await showTutorial(explanation);
  await waitForUserContinue();
});

await runner.runSection(tutorialSection);
```

### Example 5: Test Mode - Instant Execution
```typescript
// For tests - run instantly
const runner = new SectionRunner();
const result = await runner.runSection({
  ...SectionRunner.fullGame(),
  aiSpeed: 'instant'
});

expect(result.finalState.winningTeam).toBe(0);
```

## Migration Plan

### Week 1: Stabilize Timing
1. Add `TransitionDispatcher` class
2. Wire UI, controllers, and game loop through dispatcher
3. Test that game still works normally
4. Verify timing issues are fixed

### Week 2: Enable Sections
1. Add `StopConditions` DSL
2. Implement `SectionRunner` with basic presets
3. Create simple UI to play one trick/hand
4. Test section boundaries work correctly

### Week 3: Polish & Optimize
1. Implement consensus policy properly
2. Add AI speed profiles to settings
3. Refactor Quickplay to use sections
4. Add more preset sections

### Week 4: Advanced Features
1. Add state builder helpers
2. Create practice scenarios
3. Add replay integration
4. Build tutorial system

## Benefits of This Approach

1. **Low Risk**: Wraps existing code, doesn't replace it
2. **Incremental**: Can be done in phases without breaking the game
3. **Immediate Value**: Fixes timing issues in week 1
4. **Future Proof**: Clean path to eventual simplification
5. **Testable**: Each phase can be tested independently
6. **Debuggable**: Central dispatcher logs all transitions
7. **Extensible**: Easy to add new features via listeners/gates

## Future Optimizations

Once stable, consider:
1. Replacing tick system with inline delays (cleaner)
2. Simplifying `gameActions.executeAction` (remove redundancy)
3. Adding network play support (dispatcher handles sync)
4. Adding undo/redo (dispatcher tracks history)
5. Adding dev tools (dispatcher provides introspection)

## Conclusion

This unified approach combines the pragmatism of the Codex plan (wrap don't replace) with the cleaner architecture goals of the original plan. It provides an immediate fix for timing issues while setting up a solid foundation for section-based gameplay and future enhancements.