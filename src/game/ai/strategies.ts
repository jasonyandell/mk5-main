import type { AIStrategy } from './types';
import type { GameState, Player, Domino } from '../types';
import type { ValidAction } from '../../multiplayer/types';
import type { ExecutionContext } from '../types/execution';
import { determineBestTrump } from './hand-strength';
import {
  selectBestPlay,
  selectBestBid,
  evaluateTrumpOptions,
  type MonteCarloConfig
} from './monte-carlo';
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
  biddingSimulations: 5,
  playingSimulations: 10
};

/**
 * Beginner AI strategy - the standard playable AI
 *
 * Uses Monte Carlo simulation for both bidding and play decisions.
 * Trump selection now uses simulation-based evaluation to discover
 * the best trump from all available options (including special contracts
 * like nello and sevens when enabled).
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
        return this.makeTrumpDecision(state, currentPlayer, validActions);
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
   * Use trump-first Monte Carlo evaluation to decide whether and what to bid.
   *
   * New architecture:
   * 1. Discover all trump options from layers (suits, doubles, nello, sevens, etc.)
   * 2. For EACH trump option, simulate to get EV and point distribution
   * 3. Use the EV table to decide optimal bid and trump selection
   *
   * This correctly handles special contracts like nello (all-or-nothing)
   * and compares "aces at 34" vs "nello at 100% success" properly.
   */
  private makeBidDecision(state: GameState, validActions: ValidAction[]): ValidAction {
    const ctx = this.getExecutionContext(state);

    return selectBestBid(
      state,
      validActions,
      state.currentPlayer,
      ctx,
      this.config,
      'balanced' // Use balanced risk tolerance for beginner AI
    );
  }

  /**
   * Select the best trump using Monte Carlo evaluation.
   *
   * Uses the same evaluateTrumpOptions() as bidding to find the trump
   * with highest expected value. Falls back to heuristic if evaluation fails.
   */
  private makeTrumpDecision(state: GameState, player: Player, validActions: ValidAction[]): ValidAction {
    const ctx = this.getExecutionContext(state);

    // Evaluate all available trump options
    const evaluations = evaluateTrumpOptions(
      state,
      state.currentPlayer,
      ctx,
      this.config
    );

    // Find the trump with highest expected value
    let bestEval: { trump: import('../types').TrumpSelection; ev: number } | null = null;

    for (const evaluation of evaluations.values()) {
      if (!bestEval || evaluation.expectedValue > bestEval.ev) {
        bestEval = { trump: evaluation.trump, ev: evaluation.expectedValue };
      }
    }

    // Find matching action
    if (bestEval) {
      const trumpAction = validActions.find(va => {
        if (va.action.type !== 'select-trump' || !('trump' in va.action)) return false;
        const actionTrump = va.action.trump;
        if (actionTrump.type !== bestEval!.trump.type) return false;
        if (actionTrump.type === 'suit' && bestEval!.trump.type === 'suit') {
          return actionTrump.suit === bestEval!.trump.suit;
        }
        return true;
      });
      if (trumpAction) return trumpAction;
    }

    // Fallback to heuristic if simulation failed
    const bestTrump = determineBestTrump(player.hand);

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

    // Final fallback
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

// ============================================================================
// MinimaxStrategy - Phase 1: Ship Now AI
// ============================================================================

/** Configuration for MinimaxStrategy */
export interface MinimaxStrategyConfig {
  /** Number of simulations for bidding (default: 10) */
  biddingSimulations?: number;

  /** Number of simulations for playing (default: 30) */
  playingSimulations?: number;

  /** Use heuristic leads for first N tricks (default: 3) */
  heuristicLeadTricks?: number;
}

const DEFAULT_MINIMAX_CONFIG: Required<MinimaxStrategyConfig> = {
  biddingSimulations: 10,
  playingSimulations: 30,
  heuristicLeadTricks: 3
};

/**
 * Minimax AI Strategy (Phase 1: Ship Now)
 *
 * A stronger AI that combines:
 * - Heuristic opening leads (tricks 1-3): Lead highest trump
 * - PIMC with minimax rollout for mid-game
 * - More aggressive simulation count for accuracy
 *
 * This is the "beats most humans" strategy that ships now while
 * we build the perfect solver for Phase 2+.
 */
export class MinimaxStrategy implements AIStrategy {
  private ctx: ExecutionContext | null = null;
  private readonly config: Required<MinimaxStrategyConfig>;
  private readonly mcConfig: MonteCarloConfig;

  constructor(config: Partial<MinimaxStrategyConfig> = {}) {
    this.config = { ...DEFAULT_MINIMAX_CONFIG, ...config };
    this.mcConfig = {
      biddingSimulations: this.config.biddingSimulations,
      playingSimulations: this.config.playingSimulations
    };
  }

  chooseAction(state: GameState, validActions: ValidAction[]): ValidAction {
    if (validActions.length === 0) {
      throw new Error('MinimaxStrategy: No valid actions available');
    }

    const currentPlayer = state.players[state.currentPlayer];
    if (!currentPlayer) {
      throw new Error(`MinimaxStrategy: Current player ${state.currentPlayer} not found`);
    }

    switch (state.phase) {
      case 'bidding':
        return this.makeBidDecision(state, validActions);
      case 'trump_selection':
        return this.makeTrumpDecision(state, currentPlayer, validActions);
      case 'playing':
        return this.makePlayDecision(state, validActions, currentPlayer);
      default: {
        const fallback = validActions[0];
        if (!fallback) {
          throw new Error(`MinimaxStrategy: No fallback for phase ${state.phase}`);
        }
        return fallback;
      }
    }
  }

  private getExecutionContext(state: GameState): ExecutionContext {
    if (!this.ctx) {
      this.ctx = createExecutionContext(state.initialConfig);
    }
    return this.ctx;
  }

  /**
   * Bidding: Use trump-first Monte Carlo evaluation with more simulations.
   *
   * Uses the new architecture that:
   * 1. Discovers all trump options from layers
   * 2. Evaluates each trump option's EV and distribution
   * 3. Makes bid decision based on the EV table
   */
  private makeBidDecision(state: GameState, validActions: ValidAction[]): ValidAction {
    const ctx = this.getExecutionContext(state);

    return selectBestBid(
      state,
      validActions,
      state.currentPlayer,
      ctx,
      this.mcConfig,
      'balanced' // MinimaxStrategy uses balanced risk (could be made configurable)
    );
  }

  /**
   * Trump selection: Use Monte Carlo evaluation to pick optimal trump.
   *
   * Evaluates all available trump options and picks the one with
   * highest expected value. Falls back to heuristic if needed.
   */
  private makeTrumpDecision(state: GameState, player: Player, validActions: ValidAction[]): ValidAction {
    const ctx = this.getExecutionContext(state);

    // Evaluate all available trump options
    const evaluations = evaluateTrumpOptions(
      state,
      state.currentPlayer,
      ctx,
      this.mcConfig
    );

    // Find the trump with highest expected value
    let bestEval: { trump: import('../types').TrumpSelection; ev: number } | null = null;

    for (const evaluation of evaluations.values()) {
      if (!bestEval || evaluation.expectedValue > bestEval.ev) {
        bestEval = { trump: evaluation.trump, ev: evaluation.expectedValue };
      }
    }

    // Find matching action
    if (bestEval) {
      const trumpAction = validActions.find(va => {
        if (va.action.type !== 'select-trump' || !('trump' in va.action)) return false;
        const actionTrump = va.action.trump;
        if (actionTrump.type !== bestEval!.trump.type) return false;
        if (actionTrump.type === 'suit' && bestEval!.trump.type === 'suit') {
          return actionTrump.suit === bestEval!.trump.suit;
        }
        return true;
      });
      if (trumpAction) return trumpAction;
    }

    // Fallback to heuristic if simulation failed
    const bestTrump = determineBestTrump(player.hand);

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

    const fallback = validActions[0];
    if (!fallback) {
      throw new Error('MinimaxStrategy.makeTrumpDecision: No trump action available');
    }
    return fallback;
  }

  /**
   * Play decision: Heuristic leads early, PIMC+minimax later
   */
  private makePlayDecision(
    state: GameState,
    validActions: ValidAction[],
    currentPlayer: Player
  ): ValidAction {
    // Handle complete-trick action
    const completeTrickAction = validActions.find(va => va.action.type === 'complete-trick');
    if (completeTrickAction) return completeTrickAction;

    // Filter to play actions
    const playActions = validActions.filter(va => va.action.type === 'play');

    if (playActions.length === 0) {
      const fallback = validActions[0];
      if (!fallback) {
        throw new Error('MinimaxStrategy.makePlayDecision: No play actions');
      }
      return fallback;
    }

    // Only one option - no decision needed
    if (playActions.length === 1) {
      return playActions[0]!;
    }

    const ctx = this.getExecutionContext(state);
    const isLeading = state.currentTrick.length === 0;
    const trickNumber = state.tricks.length; // 0-indexed

    // Heuristic lead for early tricks
    if (isLeading && trickNumber < this.config.heuristicLeadTricks) {
      const heuristicPlay = this.selectHeuristicLead(
        state,
        playActions,
        currentPlayer.hand,
        ctx
      );
      if (heuristicPlay) return heuristicPlay;
    }

    // PIMC + minimax for everything else
    const myPlayerIndex = state.currentPlayer;
    const constraints = buildConstraints(state, myPlayerIndex, ctx.rules);

    const bestPlay = selectBestPlay(
      state,
      playActions,
      myPlayerIndex,
      constraints,
      ctx,
      this.mcConfig
    );

    if (bestPlay) return bestPlay;

    const fallback = playActions[0];
    if (!fallback) {
      throw new Error('MinimaxStrategy.makePlayDecision: No fallback');
    }
    return fallback;
  }

  /**
   * Heuristic opening lead: Lead highest trump
   *
   * Classic strong play: cash your trump winners early while you
   * have control. This extracts opponent trumps and establishes
   * your remaining trumps as winners.
   */
  private selectHeuristicLead(
    state: GameState,
    playActions: ValidAction[],
    hand: Domino[],
    ctx: ExecutionContext
  ): ValidAction | null {
    // Find all trump dominoes in hand with their ranks
    const trumpPlays: Array<{ action: ValidAction; domino: Domino; rank: number }> = [];

    for (const action of playActions) {
      if (action.action.type !== 'play') continue;

      const playAction = action.action;
      if (!('dominoId' in playAction)) continue;

      const domino = hand.find(d => String(d.id) === playAction.dominoId);
      if (!domino) continue;

      if (ctx.rules.isTrump(state, domino)) {
        // Rank trumps by their trick-taking power
        const rank = this.getTrumpRank(state, domino);
        trumpPlays.push({ action, domino, rank });
      }
    }

    // If we have trumps, lead the highest
    if (trumpPlays.length > 0) {
      trumpPlays.sort((a, b) => b.rank - a.rank);
      return trumpPlays[0]!.action;
    }

    // No trump - fall through to PIMC
    return null;
  }

  /**
   * Get the rank of a trump domino for lead selection.
   *
   * Higher rank = stronger trump = lead first.
   */
  private getTrumpRank(state: GameState, domino: Domino): number {
    // For pip trump: double of trump suit is highest, then 6-X, 5-X, etc.
    // For doubles trump: 6-6 highest, then 5-5, etc.
    const trump = state.trump;

    if (trump.type === 'doubles') {
      // Doubles trump: rank by pip value
      // Double-6 = 12, Double-5 = 10, etc.
      return domino.high + domino.low;
    }

    // Pip trump
    const trumpSuit = trump.suit;
    if (trumpSuit === undefined) return 0;

    // Double of trump suit is highest
    if (domino.high === trumpSuit && domino.low === trumpSuit) {
      return 100; // Highest possible
    }

    // Other trumps ranked by pip sum (6-X beats 5-X)
    return domino.high + domino.low;
  }
}
