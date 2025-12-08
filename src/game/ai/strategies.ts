import type { AIStrategy } from './types';
import type { GameState, Player } from '../types';
import type { ValidAction } from '../../multiplayer/types';
import type { ExecutionContext } from '../types/execution';
import { determineBestTrump } from './hand-strength';
import { evaluateBidActions, selectBestPlay, type MonteCarloConfig } from './monte-carlo';
import { buildConstraints } from './constraint-tracker';
import { createExecutionContext } from '../types/execution';
import type { RandomGenerator } from './hand-sampler';
import { defaultRng } from './hand-sampler';


/**
 * Random AI strategy - picks random action
 */
export class RandomAIStrategy implements AIStrategy {
  private readonly rng: RandomGenerator;

  constructor(rng: RandomGenerator = defaultRng) {
    this.rng = rng;
  }

  chooseAction(_state: GameState, validActions: ValidAction[]): ValidAction {
    if (validActions.length === 0) {
      throw new Error('RandomAIStrategy: No valid actions available to choose from');
    }

    // Random choice from available actions
    const index = Math.floor(this.rng.random() * validActions.length);
    const chosen = validActions[index];
    if (!chosen) {
      throw new Error(`RandomAIStrategy: Failed to select action at index ${index} from ${validActions.length} actions`);
    }
    return chosen;
  }
}

/** Default configuration for Monte Carlo evaluation */
const DEFAULT_CONFIG: MonteCarloConfig = {
  simulations: 100
};

/** Threshold for bidding - bid if make rate >= this value */
const BID_THRESHOLD = 0.50;

/**
 * Beginner AI strategy - the standard playable AI
 *
 * Uses Monte Carlo simulation for both bidding and play decisions.
 * Trump selection uses simple heuristics (strongest suit).
 */
export class BeginnerAIStrategy implements AIStrategy {
  private ctx: ExecutionContext | null = null;
  private readonly config: MonteCarloConfig;

  constructor(config: Partial<MonteCarloConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  chooseAction(state: GameState, validActions: ValidAction[]): ValidAction {
    if (validActions.length === 0) {
      throw new Error('BeginnerAIStrategy: No valid actions available to choose from');
    }

    // Get current player for context
    const currentPlayer = state.players[state.currentPlayer];
    if (!currentPlayer) {
      throw new Error(`BeginnerAIStrategy: Current player ${state.currentPlayer} not found in state`);
    }

    // Use phase-specific logic
    switch (state.phase) {
      case 'bidding':
        return this.makeBidDecision(state, validActions);
      case 'trump_selection':
        return this.makeTrumpDecision(currentPlayer, validActions);
      case 'playing':
        return this.makePlayDecision(state, validActions);
      default: {
        const fallback = validActions[0];
        if (!fallback) {
          throw new Error(`BeginnerAIStrategy: No fallback action available for phase ${state.phase}`);
        }
        return fallback;
      }
    }
  }

  /**
   * Get or create execution context for MCTS evaluation.
   */
  private getExecutionContext(state: GameState): ExecutionContext {
    if (!this.ctx) {
      this.ctx = createExecutionContext(state.initialConfig);
    }
    return this.ctx;
  }

  /**
   * Use Monte Carlo simulation to decide whether and what to bid.
   *
   * Evaluates each possible bid by simulating many games and tracking
   * the make rate. Bids the highest value with make rate >= 50%.
   */
  private makeBidDecision(state: GameState, validActions: ValidAction[]): ValidAction {
    // Find pass action (always available during bidding)
    const passAction = validActions.find(va => va.action.type === 'pass');

    // Get bid actions (exclude pass)
    const bidActions = validActions.filter(va => va.action.type === 'bid');

    // If no bid actions available, must pass
    if (bidActions.length === 0) {
      if (passAction) return passAction;
      const fallback = validActions[0];
      if (!fallback) {
        throw new Error('BeginnerAIStrategy.makeBidDecision: No pass or fallback available');
      }
      return fallback;
    }

    // Evaluate bid actions using Monte Carlo
    const ctx = this.getExecutionContext(state);
    const evaluations = evaluateBidActions(
      state,
      bidActions,
      state.currentPlayer,
      ctx,
      this.config
    );

    // Find highest bid with make rate >= threshold
    const viableBids = evaluations.filter(e => e.makeBidRate >= BID_THRESHOLD);

    if (viableBids.length > 0) {
      // Sort by bid value descending and take highest
      viableBids.sort((a, b) => b.bidValue - a.bidValue);
      const bestBid = viableBids[0];
      if (bestBid) {
        return bestBid.action;
      }
    }

    // No bid meets threshold - pass
    if (passAction) return passAction;

    // Fallback - shouldn't reach here in normal play
    const fallback = validActions[0];
    if (!fallback) {
      throw new Error('BeginnerAIStrategy.makeBidDecision: No viable bid or pass available');
    }
    return fallback;
  }

  /**
   * Pick the strongest suit based on hand composition.
   */
  private makeTrumpDecision(player: Player, validActions: ValidAction[]): ValidAction {
    const bestTrump = determineBestTrump(player.hand, player.suitAnalysis);

    if (bestTrump.type === 'doubles') {
      const doublesAction = validActions.find(va =>
        va.action.type === 'select-trump' &&
        'trump' in va.action &&
        va.action.trump.type === 'doubles'
      );
      if (doublesAction) return doublesAction;
    } else if (bestTrump.type === 'suit' && typeof bestTrump.suit === 'number') {
      const trumpAction = validActions.find(va =>
        va.action.type === 'select-trump' &&
        'trump' in va.action &&
        va.action.trump.type === 'suit' &&
        va.action.trump.suit === bestTrump.suit
      );
      if (trumpAction) return trumpAction;
    }

    // Fallback
    const fallback = validActions[0];
    if (!fallback) {
      throw new Error('BeginnerAIStrategy.makeTrumpDecision: No trump selection available');
    }
    return fallback;
  }

  /**
   * Use Monte Carlo simulation to choose the best play.
   */
  private makePlayDecision(state: GameState, validActions: ValidAction[]): ValidAction {
    // Handle complete-trick action
    const completeTrickAction = validActions.find(va => va.action.type === 'complete-trick');
    if (completeTrickAction) return completeTrickAction;

    // Filter to play actions only
    const playActions = validActions.filter(va => va.action.type === 'play');

    if (playActions.length === 0) {
      // No play actions - take first available
      const fallback = validActions[0];
      if (!fallback) {
        throw new Error('BeginnerAIStrategy.makePlayDecision: No play actions available');
      }
      return fallback;
    }

    if (playActions.length === 1) {
      // Only one option - no need for simulation
      return playActions[0]!;
    }

    // Get or create execution context
    const ctx = this.getExecutionContext(state);

    // Build constraints from game history
    const myPlayerIndex = state.currentPlayer;
    const constraints = buildConstraints(state, myPlayerIndex, ctx.rules);

    // Evaluate and select best play using Monte Carlo
    const bestPlay = selectBestPlay(
      state,
      playActions,
      myPlayerIndex,
      constraints,
      ctx,
      this.config
    );

    if (bestPlay) return bestPlay;

    const fallback = playActions[0];
    if (!fallback) {
      throw new Error('BeginnerAIStrategy.makePlayDecision: No fallback available');
    }
    return fallback;
  }
}
