/**
 * Game simulator for testing seeds and AI strategies.
 * Runs games to completion using AI players.
 */

import type { GameState } from '../types';
import { createInitialState } from '../core/state';
import { selectAIAction } from './actionSelector';
import { HeadlessRoom } from '../../server/HeadlessRoom';
import type { ValidAction } from '../../shared/multiplayer/protocol';

// Default target win rate constants for UI
export const TARGET_WIN_RATE_MIN = 0.4;  // 40% win rate minimum
export const TARGET_WIN_RATE_MAX = 0.6;  // 60% win rate maximum

// Seed finder configuration (mutable for tests)
export const SEED_FINDER_CONFIG = {
  TARGET_WIN_RATE: 0.5,  // 50% win rate target
  TOLERANCE: 0.1,        // ±10% tolerance
  MAX_ATTEMPTS: 100,     // Maximum seeds to test
  SIMULATIONS_PER_SEED: 5, // Number of games per seed
  // Legacy fields for test compatibility
  MAX_SEEDS_TO_TRY: 100,
  GAMES_PER_SEED: 10,
  SEARCH_TIMEOUT_MS: 5000,
  PROGRESS_REPORT_INTERVAL: 1
};

// Progress tracking for seed finding
export interface SeedProgress {
  currentSeed: number;
  seedsTried: number;
  gamesPlayed: number;
  totalGames: number;
  currentWinRate: number;
  bestSeed?: number;
  bestWinRate?: number;
}

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

  // Override player types if all AI
  const playerTypes: ('human' | 'ai')[] = allAI ? ['ai', 'ai', 'ai', 'ai'] : initialState.playerTypes;

  // Create HeadlessRoom with the game configuration
  const room = new HeadlessRoom(
    {
      playerTypes,
      shuffleSeed: initialState.shuffleSeed,
      theme: initialState.theme,
      colorOverrides: initialState.colorOverrides
    },
    initialState.shuffleSeed
  );

  let actionsExecuted = 0;
  let handsPlayed = 0;
  let lastPhase = room.getState().phase;

  // Run game until completion or limits reached
  while (room.getState().phase !== 'game_end' && actionsExecuted < maxActions) {
    const state = room.getState();

    // Track hand transitions
    if (state.phase === 'bidding' && lastPhase === 'scoring') {
      handsPlayed++;
      if (handsPlayed >= maxHands) {
        if (verbose) console.log(`Reached max hands limit: ${maxHands}`);
        break;
      }
    }
    lastPhase = state.phase;

    // Get all valid actions
    const actionsMap = room.getAllActions();
    const allActions: ValidAction[] = Object.values(actionsMap).flat();

    if (allActions.length === 0) {
      if (verbose) console.log('No valid actions available');
      break;
    }

    // Determine current player
    const currentPlayer = state.currentPlayer;

    // Check for consensus actions (autoExecute will handle these)
    const consensusAction = allActions.find(va =>
      va.action.type === 'complete-trick' ||
      va.action.type === 'score-hand' ||
      va.action.autoExecute === true
    );

    let selectedAction: ValidAction | undefined;

    if (consensusAction) {
      // Use consensus action (will be auto-executed)
      selectedAction = consensusAction;
    } else {
      // Get player-specific actions
      const playerActions = room.getValidActions(currentPlayer);

      if (playerActions.length === 0) {
        if (verbose) console.log(`No actions for player ${currentPlayer}`);
        break;
      }

      // For AI players, select action using AI selector
      if (playerTypes[currentPlayer] === 'ai') {
        const selected = selectAIAction(state, currentPlayer, playerActions);
        selectedAction = selected ?? undefined;
      } else {
        // For human players in simulation, pick first action
        selectedAction = playerActions[0];
      }
    }

    if (!selectedAction) {
      if (verbose) console.log('No action selected');
      break;
    }

    // Execute action through HeadlessRoom
    try {
      // Determine which player should execute this action
      const executingPlayer = 'player' in selectedAction.action
        ? selectedAction.action.player
        : currentPlayer;

      room.executeAction(executingPlayer, selectedAction.action);
      actionsExecuted++;
    } catch (error) {
      if (verbose) console.log('Action execution failed:', error);
      break;
    }

    // Yield to event loop periodically
    if (actionsExecuted % 100 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  const finalState = room.getState();

  // Count hands if we started mid-hand
  if (handsPlayed === 0 && finalState.phase !== 'bidding') {
    handsPlayed = 1;
  }

  // Determine winner
  const winner = finalState.teamScores[0] > finalState.teamScores[1] ? 0 : 1;

  return {
    winner,
    finalScores: [finalState.teamScores[0], finalState.teamScores[1]],
    handsPlayed,
    actionsExecuted,
    finalState
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