/**
 * Consensus Helper Functions for Integration Tests
 *
 * These helpers automate the consensus process in tests where multiple players
 * need to agree on game state transitions (completing tricks, scoring hands).
 *
 * **Usage Context:**
 * - Use in integration tests that need to simulate full game flows
 * - Automates the tedious process of having all 4 players agree
 * - Supports skipping human players during AI simulation tests
 *
 * **NOT for:**
 * - Unit tests (use StateBuilder instead)
 * - Tests focused on single actions (use direct action execution)
 */

import type { GameState } from '../../game/types';
import { getNextStates } from '../../game';
import { createTestContext } from './executionContext';

/**
 * Process consensus sequentially for all players in turn order
 *
 * Automatically cycles through players and executes agree actions until
 * all players have reached consensus or a human player is encountered.
 *
 * @param initialState The initial game state
 * @param consensusType The type of consensus to process
 * @param humanPlayers Optional set of player IDs to skip (for human players)
 * @returns The state after processing consensus
 *
 * @example
 * ```typescript
 * // After a trick completes, get all AI players to agree
 * const stateAfterConsensus = await processSequentialConsensus(
 *   state,
 *   'completeTrick',
 *   new Set([0]) // Skip player 0 (human)
 * );
 * ```
 */
export async function processSequentialConsensus(
  initialState: GameState,
  consensusType: 'completeTrick' | 'scoreHand',
  humanPlayers: Set<number> = new Set()
): Promise<GameState> {
  const { executeAction } = await import('../../game/core/actions');
  const ctx = createTestContext();
  let state = initialState;
  const actionType = consensusType === 'completeTrick'
    ? 'agree-trick'
    : 'agree-score';

  // Process agrees sequentially until all players have agreed
  // Count agreements from action history
  let agreedPlayers = new Set<number>();
  for (let i = state.actionHistory.length - 1; i >= 0; i--) {
    const action = state.actionHistory[i];
    if (!action) continue;
    // Stop at the last progress action
    if ((consensusType === 'completeTrick' && action.type === 'complete-trick') ||
        (consensusType === 'scoreHand' && action.type === 'score-hand')) {
      break;
    }
    if (action.type === actionType && 'player' in action) {
      agreedPlayers.add(action.player);
    }
  }

  while (agreedPlayers.size < 4) {
    const transitions = getNextStates(state, ctx);
    const agreeAction = transitions.find(t =>
      t.action.type === actionType &&
      'player' in t.action &&
      !agreedPlayers.has(t.action.player) &&
      !humanPlayers.has(t.action.player)
    );

    if (!agreeAction) {
      break; // No more agrees available
    }

    state = executeAction(state, agreeAction.action);
    if ('player' in agreeAction.action) {
      agreedPlayers.add(agreeAction.action.player);
    }
  }

  return state;
}

/**
 * Process a complete trick including all plays and consensus
 *
 * Simulates a full trick by:
 * 1. Playing cards for all 4 players
 * 2. Processing consensus for trick completion
 * 3. Executing the trick completion action
 *
 * @param initialState The initial game state
 * @param humanPlayers Optional set of player IDs to skip during consensus
 * @returns The state after the trick is complete
 *
 * @example
 * ```typescript
 * // Simulate a full trick with AI players
 * let state = StateBuilder.inPlayingPhase().build();
 * state = await processCompleteTrick(state);
 * // Now state has one completed trick
 * ```
 */
export async function processCompleteTrick(
  initialState: GameState,
  humanPlayers: Set<number> = new Set()
): Promise<GameState> {
  const { executeAction } = await import('../../game/core/actions');
  const ctx = createTestContext();
  let state = initialState;

  // Play cards until trick is complete
  while (state.currentTrick.length < 4 && state.phase === 'playing') {
    const transitions = getNextStates(state, ctx);
    const playAction = transitions.find(t => t.action.type === 'play');
    if (playAction) {
      state = executeAction(state, playAction.action);
    } else {
      break;
    }
  }

  // Process consensus
  state = await processSequentialConsensus(state, 'completeTrick', humanPlayers);

  // Complete the trick if all agreed (check if complete-trick action is available)
  const transitions = getNextStates(state, ctx);
  const completeAction = transitions.find(t => t.action.type === 'complete-trick');
  if (completeAction) {
    state = executeAction(state, completeAction.action);
  }

  return state;
}

/**
 * Process hand scoring including consensus and execution
 *
 * Handles the end-of-hand scoring by:
 * 1. Processing consensus for hand scoring
 * 2. Executing the score-hand action
 *
 * @param initialState The initial game state (should be at end of hand)
 * @param humanPlayers Optional set of player IDs to skip during consensus
 * @returns The state after hand scoring is complete
 *
 * @example
 * ```typescript
 * // After 7 tricks are played
 * let state = await processHandScoring(state);
 * // Now marks have been awarded and hand is scored
 * ```
 */
export async function processHandScoring(
  initialState: GameState,
  humanPlayers: Set<number> = new Set()
): Promise<GameState> {
  const { executeAction } = await import('../../game/core/actions');
  const ctx = createTestContext();
  let state = initialState;

  // Process scoring consensus
  state = await processSequentialConsensus(state, 'scoreHand', humanPlayers);

  // Execute score-hand if all agreed (check if score-hand action is available)
  const transitions = getNextStates(state, ctx);
  const scoreAction = transitions.find(t => t.action.type === 'score-hand');
  if (scoreAction) {
    state = executeAction(state, scoreAction.action);
  }

  return state;
}
