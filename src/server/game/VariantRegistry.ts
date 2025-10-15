/**
 * VariantRegistry - Declarative game variant configurations.
 *
 * Variants modify game behavior without special client code.
 * Variant-specific state is tracked separately from core GameState.
 */

import type { GameState, GameAction } from '../../game/types';
import type { GameVariant } from '../../shared/multiplayer/protocol';

/**
 * Variant-specific state (separate from GameState)
 */
export interface VariantState {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
}

/**
 * Variant definition with lifecycle hooks
 */
export interface VariantDefinition {
  /** Unique variant identifier */
  type: string;

  /** Display name for UI */
  name: string;

  /** Description for UI */
  description: string;

  /** Default configuration */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  defaultConfig: any;

  /** Initialize variant state (separate from game state) */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  initializeVariantState?: (config: any) => VariantState;

  /** Initialize game state for this variant (optional modifications) */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  initialize?: (state: GameState, config: any) => GameState;

  /** Update variant state after action */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  updateVariantState?: (state: GameState, variantState: VariantState, action: GameAction, config: any) => VariantState;

  /** Check if game should end based on variant rules */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  checkGameEnd?: (state: GameState, variantState: VariantState, config: any) => boolean;

  /** Modify game state after action (using variant state) */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  afterAction?: (state: GameState, variantState: VariantState, action: GameAction, config: any) => GameState;

  /** Calculate final scores */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  calculateScores?: (state: GameState, variantState: VariantState, config: any) => { team0: number; team1: number };

  /** Validate configuration */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  validateConfig?: (config: any) => { valid: boolean; error?: string };
}

/**
 * Registry of all game variants
 */
export class VariantRegistry {
  private static variants = new Map<string, VariantDefinition>();

  /**
   * Register a new variant
   */
  static register(variant: VariantDefinition): void {
    this.variants.set(variant.type, variant);
  }

  /**
   * Get variant definition
   */
  static get(type: string): VariantDefinition | undefined {
    return this.variants.get(type);
  }

  /**
   * Get all registered variants
   */
  static getAll(): VariantDefinition[] {
    return Array.from(this.variants.values());
  }

  /**
   * Initialize variant state
   */
  static initializeVariantState(variant: GameVariant): VariantState {
    const def = this.get(variant.type);
    if (!def || !def.initializeVariantState) {
      return {};
    }

    return def.initializeVariantState(variant.config || def.defaultConfig);
  }

  /**
   * Apply variant initialization to game state
   */
  static initialize(state: GameState, variant: GameVariant): GameState {
    const def = this.get(variant.type);
    if (!def || !def.initialize) {
      return state;
    }

    return def.initialize(state, variant.config || def.defaultConfig);
  }

  /**
   * Update variant state after action
   */
  static updateVariantState(
    state: GameState,
    variantState: VariantState,
    action: GameAction,
    variant: GameVariant
  ): VariantState {
    const def = this.get(variant.type);
    if (!def || !def.updateVariantState) {
      return variantState;
    }

    return def.updateVariantState(state, variantState, action, variant.config || def.defaultConfig);
  }

  /**
   * Check if game should end based on variant rules
   */
  static checkGameEnd(state: GameState, variantState: VariantState, variant: GameVariant): boolean {
    const def = this.get(variant.type);
    if (!def || !def.checkGameEnd) {
      return false; // Use normal game end logic
    }

    return def.checkGameEnd(state, variantState, variant.config || def.defaultConfig);
  }

  /**
   * Apply variant-specific state modifications after action
   */
  static afterAction(
    state: GameState,
    variantState: VariantState,
    action: GameAction,
    variant: GameVariant
  ): GameState {
    const def = this.get(variant.type);
    if (!def || !def.afterAction) {
      return state;
    }

    return def.afterAction(state, variantState, action, variant.config || def.defaultConfig);
  }
}

// ============================================================================
// Built-in Variants
// ============================================================================

/**
 * Standard game - no modifications
 */
VariantRegistry.register({
  type: 'standard',
  name: 'Standard Game',
  description: 'Play to 250 points with marks',
  defaultConfig: {},
});

/**
 * One-hand mode - stop after N hands
 */
VariantRegistry.register({
  type: 'one-hand',
  name: 'One Hand Challenge',
  description: 'Play a single hand for the best score',
  defaultConfig: {
    targetHand: 1,
    maxAttempts: 3
  },

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  initializeVariantState(config: any): VariantState {
    return {
      handNumber: 1,
      attempts: config.attempts || 1,
      originalSeed: config.originalSeed
    };
  },

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  updateVariantState(_state: GameState, variantState: VariantState, action: GameAction, _config: any): VariantState {
    // Detect hand transitions (scoring → bidding means new hand)
    if (action.type === 'score-hand' || action.type === 'agree-score-hand') {
      // After scoring, we're about to start a new hand
      return {
        ...variantState,
        handNumber: variantState.handNumber + 1
      };
    }
    return variantState;
  },

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  checkGameEnd(state: GameState, variantState: VariantState, config: any): boolean {
    // End after target hand is complete
    const targetHand = config.targetHand || 1;

    // We've completed the target hand if we're about to start a hand beyond it
    if (state.phase === 'bidding' && variantState.handNumber > targetHand) {
      return true;
    }

    // Or if we've just finished scoring the target hand
    if (state.phase === 'scoring' && variantState.handNumber >= targetHand) {
      return true;
    }

    return false;
  },

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  afterAction(state: GameState, variantState: VariantState, _action: GameAction, config: any): GameState {
    // Force game end if we've completed target hand
    if (this.checkGameEnd!(state, variantState, config)) {
      return {
        ...state,
        phase: 'game_end'
      };
    }
    return state;
  },

  calculateScores(state: GameState, _variantState: VariantState): { team0: number; team1: number } {
    return {
      team0: state.teamScores[0],
      team1: state.teamScores[1]
    };
  }
});

/**
 * Tournament mode - best of N games
 */
VariantRegistry.register({
  type: 'tournament',
  name: 'Tournament',
  description: 'Best of N games format',
  defaultConfig: {
    bestOf: 3,
    pointLimit: 250
  },

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  initializeVariantState(config: any): VariantState {
    return {
      currentGame: 1,
      gameScores: [0, 0],
      bestOf: config.bestOf || 3
    };
  },

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  updateVariantState(state: GameState, variantState: VariantState, _action: GameAction, _config: any): VariantState {
    // Check if a game just ended
    if (state.phase === 'game_end') {
      const winner = state.teamScores[0] > state.teamScores[1] ? 0 : 1;
      const newScores = [...variantState.gameScores];
      newScores[winner]++;

      return {
        ...variantState,
        gameScores: newScores,
        currentGame: variantState.currentGame + 1
      };
    }
    return variantState;
  },

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  checkGameEnd(_state: GameState, variantState: VariantState, _config: any): boolean {
    // Check if a team has won the majority of games
    const needed = Math.ceil(variantState.bestOf / 2);
    return variantState.gameScores[0] >= needed || variantState.gameScores[1] >= needed;
  },

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  afterAction(state: GameState, variantState: VariantState, _action: GameAction, config: any): GameState {
    // If a game ended but tournament isn't over, reset for next game
    if (state.phase === 'game_end' && !this.checkGameEnd!(state, variantState, config)) {
      return {
        ...state,
        phase: 'bidding',
        teamScores: [0, 0],
        teamMarks: [0, 0]
      };
    }
    return state;
  }
});

/**
 * Speed mode - time limits per action
 */
VariantRegistry.register({
  type: 'speed',
  name: 'Speed 42',
  description: 'Fast-paced with time limits',
  defaultConfig: {
    actionTimeLimit: 10000, // 10 seconds per action
    handTimeLimit: 300000  // 5 minutes per hand
  },

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  initializeVariantState(_config: any): VariantState {
    return {
      handStartTime: Date.now(),
      playerTimeBanks: [30000, 30000, 30000, 30000], // 30 second time banks
      actionStartTime: Date.now()
    };
  },

  updateVariantState(_state: GameState, variantState: VariantState, _action: GameAction): VariantState {
    // Track timing for each action
    return {
      ...variantState,
      actionStartTime: Date.now()
    };
  }
});

/**
 * Nello variant - special bidding rules
 */
VariantRegistry.register({
  type: 'nello',
  name: 'Nello 42',
  description: 'Includes Nello and Plunge bids',
  defaultConfig: {
    allowNello: true,
    allowPlunge: true,
    nelloValue: 42,
    plungeValue: 84
  }
});

/**
 * Moon shooting variant
 */
VariantRegistry.register({
  type: 'moon',
  name: 'Moon Shooting',
  description: 'Bonus points for shooting the moon',
  defaultConfig: {
    moonBonus: 42,
    requireAllTricks: true
  },

  initializeVariantState(): VariantState {
    return {
      moonShotInProgress: false,
      trickWinners: []
    };
  },

  updateVariantState(state: GameState, variantState: VariantState, action: GameAction): VariantState {
    if (action.type === 'complete-trick') {
      // Track trick winners for moon shot detection
      const lastTrick = state.tricks[state.tricks.length - 1];
      if (lastTrick?.winner !== undefined) {
        return {
          ...variantState,
          trickWinners: [...variantState.trickWinners, lastTrick.winner]
        };
      }
    }

    // Reset on new hand
    if (action.type === 'score-hand') {
      return {
        ...variantState,
        trickWinners: []
      };
    }

    return variantState;
  },

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  afterAction(state: GameState, variantState: VariantState, _action: GameAction, config: any): GameState {
    // Check for moon shot after hand completes
    if (state.phase === 'scoring' && variantState.trickWinners.length === 7) {
      // Check if one team won all tricks
      const team0WonAll = variantState.trickWinners.every((w: number) => w % 2 === 0);
      const team1WonAll = variantState.trickWinners.every((w: number) => w % 2 === 1);

      if ((team0WonAll || team1WonAll) && config.requireAllTricks) {
        // Add moon bonus
        const moonTeam = team0WonAll ? 0 : 1;
        return {
          ...state,
          teamScores: [
            state.teamScores[0] + (moonTeam === 0 ? config.moonBonus : 0),
            state.teamScores[1] + (moonTeam === 1 ? config.moonBonus : 0)
          ] as [number, number]
        };
      }
    }
    return state;
  }
});