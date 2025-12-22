/**
 * Rollout Strategy for Monte Carlo Simulations
 *
 * Provides fast, good-enough decision making for MCTS rollouts.
 * NOT a playable AI tier - internal to Monte Carlo only.
 *
 * This is a seam for future neural network integration (AlphaGo-style).
 */

import type { GameState, Player } from '../types';
import type { ValidAction } from '../../multiplayer/types';
import { analyzeHand } from './utilities';
import { determineBestTrump } from './hand-strength';
import { composeRules } from '../layers/compose';
import { baseLayer } from '../layers';

/**
 * Interface for rollout strategies used in Monte Carlo simulations.
 *
 * Future: NeuralRolloutStrategy could implement this for AlphaGo-style
 * value network evaluation.
 */
export interface RolloutStrategy {
  chooseAction(state: GameState, validActions: ValidAction[]): ValidAction;
}

/**
 * Heuristic-based rollout strategy.
 *
 * Uses simple rules for fast decision making during MCTS rollouts:
 * - Bidding: Always pass (rollouts assume bid already made)
 * - Trump: Pick strongest suit
 * - Playing: Basic trick-taking heuristics
 */
export class HeuristicRolloutStrategy implements RolloutStrategy {
  chooseAction(state: GameState, validActions: ValidAction[]): ValidAction {
    if (validActions.length === 0) {
      throw new Error('HeuristicRolloutStrategy: No valid actions available');
    }

    // Get current player for context
    const currentPlayer = state.players[state.currentPlayer];
    if (!currentPlayer) {
      throw new Error(`HeuristicRolloutStrategy: Current player ${state.currentPlayer} not found`);
    }

    // Phase-specific logic
    switch (state.phase) {
      case 'bidding':
        return this.makeBidDecision(validActions);
      case 'trump_selection':
        return this.makeTrumpDecision(currentPlayer, validActions);
      case 'playing':
        return this.makePlayDecision(state, currentPlayer, validActions);
      default: {
        // For other phases (scoring, etc), just pick first action
        const fallback = validActions[0];
        if (!fallback) {
          throw new Error(`HeuristicRolloutStrategy: No fallback for phase ${state.phase}`);
        }
        return fallback;
      }
    }
  }

  /**
   * During rollouts, bidding is usually already complete.
   * If we do need to bid, just pass - we're simulating post-bid play.
   */
  private makeBidDecision(validActions: ValidAction[]): ValidAction {
    const passAction = validActions.find(va => va.action.type === 'pass');
    if (passAction) return passAction;

    // Fallback to first action if pass not available
    const fallback = validActions[0];
    if (!fallback) {
      throw new Error('HeuristicRolloutStrategy: No bid action available');
    }
    return fallback;
  }

  /**
   * Pick the strongest suit based on hand composition.
   */
  private makeTrumpDecision(player: Player, validActions: ValidAction[]): ValidAction {
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

    // Fallback
    const fallback = validActions[0];
    if (!fallback) {
      throw new Error('HeuristicRolloutStrategy: No trump action available');
    }
    return fallback;
  }

  /**
   * Basic trick-taking heuristics:
   * - Complete tricks when possible
   * - When leading: play strongest domino
   * - When following: play count if partner winning, try to win or dump low otherwise
   */
  private makePlayDecision(state: GameState, player: Player, validActions: ValidAction[]): ValidAction {
    // Complete trick if that's an option
    const completeTrickAction = validActions.find(va => va.action.type === 'complete-trick');
    if (completeTrickAction) return completeTrickAction;

    // Get play actions
    const playActions = validActions.filter(va => va.action.type === 'play');
    if (playActions.length === 0) {
      const fallback = validActions[0];
      if (!fallback) {
        throw new Error('HeuristicRolloutStrategy: No play action available');
      }
      return fallback;
    }

    // Analyze hand
    const analysis = analyzeHand(state, player.id);

    // Leading a trick
    if (state.currentTrick.length === 0) {
      // Lead with strongest playable domino
      const playable = analysis.dominoes.filter(d => d.beatenBy !== undefined);
      if (playable.length > 0) {
        const bestLead = playable[0];
        if (bestLead) {
          const action = playActions.find(va =>
            va.action.type === 'play' &&
            'dominoId' in va.action &&
            va.action.dominoId === bestLead.domino.id
          );
          if (action) return action;
        }
      }

      // Fallback to first play
      const fallback = playActions[0];
      if (!fallback) {
        throw new Error('HeuristicRolloutStrategy: No lead play available');
      }
      return fallback;
    }

    // Following in a trick
    const myTeam = player.teamId;
    const rules = composeRules([baseLayer]);
    const currentWinnerPlayerId = rules.calculateTrickWinner(state, state.currentTrick);

    if (currentWinnerPlayerId === -1) {
      const fallback = playActions[0];
      if (!fallback) {
        throw new Error('HeuristicRolloutStrategy: Cannot determine winner');
      }
      return fallback;
    }

    const winnerPlayer = state.players[currentWinnerPlayerId];
    if (!winnerPlayer) {
      throw new Error(`HeuristicRolloutStrategy: Winner ${currentWinnerPlayerId} not found`);
    }

    const partnerCurrentlyWinning = winnerPlayer.teamId === myTeam;

    // Score each play action
    const scoredActions = playActions.map(va => {
      if (va.action.type !== 'play' || !('dominoId' in va.action)) return null;

      const dominoId = va.action.dominoId;
      const dominoAnalysis = analysis.dominoes.find(d => d.domino.id === dominoId);
      if (!dominoAnalysis) return null;

      return {
        action: va,
        points: dominoAnalysis.points,
        canBeat: dominoAnalysis.wouldBeatTrick,
      };
    }).filter((item): item is { action: ValidAction; points: number; canBeat: boolean } => item !== null);

    if (partnerCurrentlyWinning) {
      // Partner winning - play high count
      scoredActions.sort((a, b) => b.points - a.points);
    } else {
      // Opponent winning
      const winningPlays = scoredActions.filter(a => a.canBeat);

      if (winningPlays.length > 0) {
        // Can win - use lowest count to win
        winningPlays.sort((a, b) => a.points - b.points);
        const chosen = winningPlays[0];
        if (chosen?.action) return chosen.action;
      } else {
        // Can't win - dump lowest count
        scoredActions.sort((a, b) => a.points - b.points);
      }
    }

    const chosen = scoredActions[0];
    if (chosen?.action) return chosen.action;

    // Ultimate fallback
    const fallback = playActions[0];
    if (!fallback) {
      throw new Error('HeuristicRolloutStrategy: No scored action available');
    }
    return fallback;
  }
}

// Singleton instance for efficiency
let rolloutStrategyInstance: HeuristicRolloutStrategy | null = null;

/**
 * Get the rollout strategy instance.
 * Uses singleton pattern for efficiency during simulations.
 */
export function getRolloutStrategy(): RolloutStrategy {
  if (!rolloutStrategyInstance) {
    rolloutStrategyInstance = new HeuristicRolloutStrategy();
  }
  return rolloutStrategyInstance;
}
