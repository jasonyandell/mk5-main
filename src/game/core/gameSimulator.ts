/**
 * Game simulator for testing seeds and AI strategies.
 * Runs games to completion using AI players.
 */

import type { GameState, GameAction } from '../types';
import { createInitialState } from './state';
import { getValidActions } from './gameEngine';
import { executeAction } from './actions';
import { selectAIAction } from './ai-scheduler';
import { getNextStates } from './gameEngine';

/**
 * Options for game simulation
 */
export interface SimulateGameOptions {
  /** Maximum number of hands to simulate (safety limit) */
  maxHands?: number;
  /** Maximum actions per game (safety limit) */
  maxActions?: number;
  /** All players are AI */
  allAI?: boolean;
  /** AI difficulty level */
  aiDifficulty?: 'beginner' | 'intermediate' | 'expert';
  /** Log game progress */
  verbose?: boolean;
}

/**
 * Result of game simulation
 */
export interface SimulationResult {
  /** Winning team (0 or 1) */
  winner: 0 | 1;
  /** Final scores [team0, team1] */
  finalScores: [number, number];
  /** Number of hands played */
  handsPlayed: number;
  /** Number of actions executed */
  actionsExecuted: number;
  /** Final game state */
  finalState: GameState;
}

/**
 * Simulate a complete game with AI players
 */
export async function simulateGame(
  initialState: GameState,
  options: SimulateGameOptions = {}
): Promise<SimulationResult> {
  const {
    maxHands = 10,
    maxActions = 1000,
    allAI = true,
    verbose = false
  } = options;

  let state = { ...initialState };
  let handsPlayed = 0;
  let actionsExecuted = 0;
  let lastPhase = state.phase;

  // Override player types if all AI
  if (allAI) {
    state.playerTypes = ['ai', 'ai', 'ai', 'ai'];
  }

  // Run game until completion or limits reached
  while (state.phase !== 'game_end' && actionsExecuted < maxActions) {
    // Track hand transitions
    if (state.phase === 'bidding' && lastPhase === 'scoring') {
      handsPlayed++;
      if (handsPlayed >= maxHands) {
        if (verbose) console.log(`Reached max hands limit: ${maxHands}`);
        break;
      }
    }
    lastPhase = state.phase;

    // Get valid actions
    const validActions = getValidActions(state);

    if (validActions.length === 0) {
      if (verbose) console.log('No valid actions available');
      break;
    }

    // Determine current player
    const currentPlayer = state.currentPlayer;

    // Check for consensus actions (available to all)
    const consensusAction = validActions.find(action =>
      action.type === 'agree-score-hand' ||
      action.type === 'complete-trick' ||
      action.type === 'score-hand'
    );

    let selectedAction: GameAction | undefined;

    if (consensusAction) {
      // Immediate consensus for AI players
      selectedAction = consensusAction;
    } else {
      // Get player-specific actions
      const playerActions = validActions.filter(action => {
        if (!('player' in action)) return true;
        return action.player === currentPlayer;
      });

      if (playerActions.length === 0) {
        if (verbose) console.log(`No actions for player ${currentPlayer}`);
        break;
      }

      // For AI players, select action
      if (state.playerTypes[currentPlayer] === 'ai') {
        // Get transitions for AI selector
        const transitions = getNextStates(state);
        const playerTransitions = transitions.filter(t => {
          const action = t.action;
          if (!('player' in action)) return true;
          return action.player === currentPlayer;
        });

        const selected = selectAIAction(state, currentPlayer, playerTransitions);
        selectedAction = selected?.action;
      } else {
        // For human players in simulation, pick first action
        selectedAction = playerActions[0];
      }
    }

    if (!selectedAction) {
      if (verbose) console.log('No action selected');
      break;
    }

    // Execute action
    const newState = executeAction(state, selectedAction);
    if (!newState) {
      if (verbose) console.log('Action execution failed');
      break;
    }

    state = newState;
    actionsExecuted++;

    // Yield to event loop periodically
    if (actionsExecuted % 100 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  // Count hands if we started mid-hand
  if (handsPlayed === 0 && state.phase !== 'bidding') {
    handsPlayed = 1;
  }

  // Determine winner
  const winner = state.teamScores[0] > state.teamScores[1] ? 0 : 1;

  return {
    winner,
    finalScores: [state.teamScores[0], state.teamScores[1]],
    handsPlayed,
    actionsExecuted,
    finalState: state
  };
}

/**
 * Simulate multiple games with the same seed to get win rate
 */
export async function testSeedWinRate(
  seed: number,
  simulations: number = 10,
  options?: SimulateGameOptions
): Promise<{ winRate: number; avgScore: number }> {
  let wins = 0;
  let totalScore = 0;

  for (let i = 0; i < simulations; i++) {
    const initialState = createInitialState({
      shuffleSeed: seed,
      playerTypes: ['ai', 'ai', 'ai', 'ai']
    });

    const result = await simulateGame(initialState, {
      ...options,
      allAI: true,
      maxHands: 1 // For one-hand testing
    });

    if (result.winner === 0) {
      wins++;
    }
    totalScore += result.finalScores[0];

    // Yield periodically
    if (i % 10 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  return {
    winRate: wins / simulations,
    avgScore: totalScore / simulations
  };
}

/**
 * Find a competitive seed for one-hand mode
 */
export async function findCompetitiveSeed(options: {
  targetWinRate: number;
  tolerance: number;
  maxAttempts: number;
  onProgress?: (attempt: number) => void;
  simulationsPerSeed?: number;
}): Promise<number> {
  const {
    targetWinRate,
    tolerance,
    maxAttempts,
    onProgress,
    simulationsPerSeed = 5
  } = options;

  const minWinRate = targetWinRate - tolerance;
  const maxWinRate = targetWinRate + tolerance;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    // Generate random seed
    const seed = Math.floor(Math.random() * 1000000);

    // Report progress
    if (onProgress) {
      onProgress(attempt);
    }

    // Test this seed
    const { winRate } = await testSeedWinRate(seed, simulationsPerSeed, {
      maxHands: 1,
      verbose: false
    });

    // Check if within target range
    if (winRate >= minWinRate && winRate <= maxWinRate) {
      return seed;
    }

    // Yield to event loop
    if (attempt % 5 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  // If no perfect seed found, return a random one
  return Math.floor(Math.random() * 1000000);
}