/**
 * Minimax Evaluator for Perfect Information Game Trees
 *
 * Searches the game tree to terminal state (hand completion) using
 * alpha-beta pruning. Used by PIMC to evaluate each sampled world.
 *
 * Key features:
 * - Partnership minimax: Players 0,2 (Team 0) = MAX, Players 1,3 (Team 1) = MIN
 * - Searches to actual terminal state (no heuristic evaluation)
 * - Alpha-beta pruning with move ordering for efficiency
 * - Handles auto-execute actions (complete-trick, etc.) transparently
 * - Respects special contracts via ctx.rules
 */

import type { GameState, GameAction } from '../types';
import type { ExecutionContext } from '../types/execution';
import { executeAction } from '../core/actions';
import { getDominoPoints } from '../core/dominoes';

/**
 * Configuration for minimax search
 */
export interface MinimaxConfig {
  /** Enable alpha-beta pruning (default: true) */
  alphaBeta?: boolean;

  /** Move ordering strategy (default: 'heuristic') */
  moveOrdering?: 'none' | 'heuristic';

  /** Enable debug logging (default: false) */
  debug?: boolean;
}

/**
 * Result of minimax evaluation
 */
export interface MinimaxResult {
  /** Final points for team 0 */
  team0Points: number;

  /** Final points for team 1 */
  team1Points: number;

  /** Number of nodes explored during search */
  nodesExplored: number;
}

const DEFAULT_CONFIG: Required<MinimaxConfig> = {
  alphaBeta: true,
  moveOrdering: 'heuristic',
  debug: false
};

/**
 * Evaluate a game state using minimax to hand completion.
 *
 * Assumes all hands are visible (perfect information within the sampled world).
 * Returns the game-theoretic optimal outcome for both teams.
 *
 * @param state Game state with all hands visible
 * @param ctx Execution context with composed rules
 * @param config Optional configuration
 * @returns Terminal scores and search statistics
 */
export function minimaxEvaluate(
  state: GameState,
  ctx: ExecutionContext,
  config?: MinimaxConfig
): MinimaxResult {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  let nodesExplored = 0;

  /**
   * Core minimax search with alpha-beta pruning.
   *
   * @param currentState Current game state
   * @param alpha Best score MAX can guarantee
   * @param beta Best score MIN can guarantee
   * @returns Score from Team 0's perspective
   */
  function search(
    currentState: GameState,
    alpha: number,
    beta: number
  ): number {
    nodesExplored++;

    // Check for hand completion (terminal state)
    if (isHandComplete(currentState)) {
      return currentState.teamScores[0];
    }

    // Check for early termination (nello wins trick, plunge loses trick, etc.)
    const outcome = ctx.rules.checkHandOutcome(currentState);
    if (outcome.isDetermined) {
      // Outcome is determined early - return current team scores
      // (special contracts like nello/plunge determine winner early)
      return currentState.teamScores[0];
    }

    // Get valid actions for current position
    const actions = ctx.getValidActions(currentState);

    if (actions.length === 0) {
      // No actions available - return current score
      return currentState.teamScores[0];
    }

    // Handle auto-execute actions (complete-trick, score-hand, etc.)
    const autoAction = actions.find(a => a.autoExecute === true);
    if (autoAction) {
      const nextState = executeAction(currentState, autoAction, ctx.rules);
      return search(nextState, alpha, beta);
    }

    // Get play actions for current player
    const playActions = actions.filter(a => a.type === 'play');

    if (playActions.length === 0) {
      // No play actions - check for other action types (shouldn't happen in playing phase)
      return currentState.teamScores[0];
    }

    // Order moves for better pruning
    const orderedActions = cfg.moveOrdering === 'heuristic'
      ? orderMoves(playActions, currentState, ctx)
      : playActions;

    // Determine if current player is maximizing (Team 0) or minimizing (Team 1)
    const currentTeam = currentState.currentPlayer % 2;
    const isMaximizing = currentTeam === 0;

    if (isMaximizing) {
      let maxEval = -Infinity;
      for (const action of orderedActions) {
        const nextState = executeAction(currentState, action, ctx.rules);
        const evalScore = search(nextState, alpha, beta);
        maxEval = Math.max(maxEval, evalScore);

        if (cfg.alphaBeta) {
          alpha = Math.max(alpha, evalScore);
          if (beta <= alpha) break; // Beta cutoff
        }
      }
      return maxEval;
    } else {
      let minEval = Infinity;
      for (const action of orderedActions) {
        const nextState = executeAction(currentState, action, ctx.rules);
        const evalScore = search(nextState, alpha, beta);
        minEval = Math.min(minEval, evalScore);

        if (cfg.alphaBeta) {
          beta = Math.min(beta, evalScore);
          if (beta <= alpha) break; // Alpha cutoff
        }
      }
      return minEval;
    }
  }

  // Run the search
  const team0Points = search(state, -Infinity, Infinity);

  // Calculate team1Points from total (42 points total in a hand)
  const totalPointsInHand = 42;
  const team1Points = totalPointsInHand - team0Points;

  return {
    team0Points,
    team1Points,
    nodesExplored
  };
}

/**
 * Check if the hand is complete.
 *
 * Hand is complete when we're in 'scoring' phase or beyond.
 */
function isHandComplete(state: GameState): boolean {
  // Check phase directly
  if (state.phase === 'scoring' || state.phase === 'game_end' || state.phase === 'one-hand-complete') {
    return true;
  }

  // Also check if all tricks have been played
  if (state.tricks.length === 7 && state.currentTrick.length === 0) {
    return true;
  }

  return false;
}

/**
 * Order moves for better alpha-beta pruning efficiency.
 *
 * Heuristics:
 * - When leading: prefer non-count dominoes (save count for when partner winning)
 * - When following partner winning: prefer high-count plays
 * - When following opponent winning: prefer winning plays, then low-count dumps
 */
function orderMoves(
  actions: GameAction[],
  state: GameState,
  ctx: ExecutionContext
): GameAction[] {
  const currentPlayer = state.currentPlayer;
  const myTeam = currentPlayer % 2;
  const isLeading = state.currentTrick.length === 0;

  // Score each action for sorting
  const scored = actions.map(action => {
    if (action.type !== 'play' || !('dominoId' in action)) {
      return { action, score: 0 };
    }

    const player = state.players[currentPlayer];
    if (!player) return { action, score: 0 };

    const domino = player.hand.find(d => String(d.id) === action.dominoId);
    if (!domino) return { action, score: 0 };

    const points = getDominoPoints(domino);
    const isTrump = ctx.rules.isTrump(state, domino);

    if (isLeading) {
      // When leading: prefer non-count dominoes first
      // Score: lower points = higher priority
      // Trump as tiebreaker (prefer leading non-trump to save trump)
      return { action, score: -points - (isTrump ? 0 : 10) };
    } else {
      // When following: check who's winning
      const currentWinner = ctx.rules.calculateTrickWinner(state, state.currentTrick);
      const winnerTeam = currentWinner >= 0 ? state.players[currentWinner]?.teamId : undefined;
      const partnerWinning = winnerTeam === myTeam;

      // Check if this play would win
      const simulatedTrick = [...state.currentTrick, { player: currentPlayer, domino }];
      const tempState = { ...state, currentTrick: simulatedTrick };
      const newWinner = ctx.rules.calculateTrickWinner(tempState, simulatedTrick);
      const wouldWin = newWinner === currentPlayer;

      if (partnerWinning) {
        // Partner winning: prefer high count plays
        return { action, score: points };
      } else {
        // Opponent winning
        if (wouldWin) {
          // Can win: prefer winning with low count
          return { action, score: 100 - points };
        } else {
          // Can't win: prefer low count dumps
          return { action, score: -points };
        }
      }
    }
  });

  // Sort by score descending (higher score = better move = try first)
  scored.sort((a, b) => b.score - a.score);

  return scored.map(s => s.action);
}

/**
 * Create a terminal state from minimax result.
 *
 * Used by monte-carlo.ts to create a final state after minimax evaluation.
 * This maintains compatibility with the existing rollout interface.
 *
 * @param initialState Starting state that was evaluated
 * @param result Minimax evaluation result
 * @returns State representing the hand end
 */
export function createTerminalState(
  initialState: GameState,
  result: MinimaxResult
): GameState {
  return {
    ...initialState,
    phase: 'scoring',
    teamScores: [result.team0Points, result.team1Points],
    currentTrick: [],
    // Clear all hands to indicate hand is complete
    players: initialState.players.map(p => ({ ...p, hand: [] }))
  };
}
