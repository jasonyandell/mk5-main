/**
 * Game simulator for testing seeds and AI strategies.
 * Runs games to completion using AI players.
 *
 * CRITICAL DEPENDENCIES:
 * - Uses beginner AI strategy (deterministic and fast)
 * - selectAIAction() is hardcoded to beginner strategy in actionSelector.ts:17
 * - Beginner AI provides deterministic, reproducible game outcomes
 * - Same seed + same actions = identical game state (event sourcing guarantee)
 *
 * FAIL-FAST DESIGN:
 * - Throws immediately on any error (no graceful degradation)
 * - No verbose logging (crashes expose bugs via stack traces)
 * - Deterministic system means crashes are reproducible
 * - If simulation fails, it's a bug that must be fixed
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
 *
 * NOTE: aiDifficulty is ignored - beginner AI is hardcoded in actionSelector.ts
 */
export interface SimulateGameOptions {
  /** Maximum number of hands to simulate (safety limit) */
  maxHands?: number;
  /** Maximum actions per game (safety limit) */
  maxActions?: number;
  /** All players are AI */
  allAI?: boolean;
  /** AI difficulty level (CURRENTLY IGNORED - beginner is hardcoded) */
  aiDifficulty?: 'beginner' | 'intermediate' | 'expert';
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
 * Simulate a complete game with AI players.
 *
 * FAIL-FAST: Throws immediately on any error condition:
 * - No valid actions available
 * - AI fails to select action
 * - Action execution fails
 * - Game doesn't reach completion
 *
 * All errors indicate bugs that must be fixed (deterministic system).
 */
export async function simulateGame(
  initialState: GameState,
  options: SimulateGameOptions = {}
): Promise<SimulationResult> {
  const {
    maxHands = 100,
    maxActions = 5000,
    allAI = true
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
        throw new Error(
          `[simulateGame] Reached max hands limit: ${maxHands}\n` +
          `Actions executed: ${actionsExecuted}\n` +
          `Phase: ${state.phase}\n` +
          `This indicates the game is not progressing to completion within expected bounds.`
        );
      }
    }
    lastPhase = state.phase;

    // Get all valid actions
    const actionsMap = room.getAllActions();
    const allActions: ValidAction[] = Object.values(actionsMap).flat();

    if (allActions.length === 0) {
      throw new Error(
        `[simulateGame] No valid actions available\n` +
        `Phase: ${state.phase}\n` +
        `Actions executed: ${actionsExecuted}\n` +
        `Hands played: ${handsPlayed}\n` +
        `Current player: ${state.currentPlayer}\n` +
        `This indicates a game state with no legal moves (stuck state).`
      );
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
    let playerActions: ValidAction[] = [];

    if (consensusAction) {
      // Use consensus action (will be auto-executed)
      selectedAction = consensusAction;
    } else {
      // Get player-specific actions
      playerActions = room.getValidActions(currentPlayer);

      if (playerActions.length === 0) {
        throw new Error(
          `[simulateGame] No actions available for current player\n` +
          `Current player: ${currentPlayer}\n` +
          `Phase: ${state.phase}\n` +
          `Actions executed: ${actionsExecuted}\n` +
          `Total actions available: ${allActions.length}\n` +
          `This indicates action filtering is removing all player actions.`
        );
      }

      // For AI players, select action using AI selector (ALWAYS beginner strategy)
      if (playerTypes[currentPlayer] === 'ai') {
        const selected = selectAIAction(state, currentPlayer, playerActions);
        selectedAction = selected ?? undefined;
      } else {
        // For human players in simulation, pick first action
        selectedAction = playerActions[0];
      }
    }

    if (!selectedAction) {
      throw new Error(
        `[simulateGame] AI failed to select action\n` +
        `Current player: ${currentPlayer}\n` +
        `Phase: ${state.phase}\n` +
        `Available actions: ${playerActions.length}\n` +
        `Actions executed: ${actionsExecuted}\n` +
        `This indicates a bug in selectAIAction (should never return null with valid actions).`
      );
    }

    // Execute action through HeadlessRoom (fail-fast on errors)
    // Determine which player should execute this action
    const executingPlayer = 'player' in selectedAction.action
      ? selectedAction.action.player
      : currentPlayer;

    try {
      room.executeAction(executingPlayer, selectedAction.action);
      actionsExecuted++;
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      throw new Error(
        `[simulateGame] Action execution failed\n` +
        `Action type: ${selectedAction.action.type}\n` +
        `Executing player: ${executingPlayer}\n` +
        `Current player: ${currentPlayer}\n` +
        `Phase: ${state.phase}\n` +
        `Actions executed: ${actionsExecuted}\n` +
        `Original error: ${errorMsg}\n` +
        `This indicates the action was invalid or violated game rules.`
      );
    }

    // Yield to event loop periodically
    if (actionsExecuted % 100 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  const finalState = room.getState();

  // FAIL-FAST: Game must complete or hit explicit limit
  if (finalState.phase !== 'game_end') {
    if (actionsExecuted >= maxActions) {
      throw new Error(
        `[simulateGame] Hit maxActions limit without completing game\n` +
        `Max actions: ${maxActions}\n` +
        `Actions executed: ${actionsExecuted}\n` +
        `Final phase: ${finalState.phase}\n` +
        `Hands played: ${handsPlayed}\n` +
        `Team marks: [${finalState.teamMarks[0]}, ${finalState.teamMarks[1]}]\n` +
        `Game target: ${finalState.gameTarget}\n` +
        `This indicates the game is taking too many actions or stuck in a loop.\n` +
        `Either increase maxActions or fix the game logic causing excessive actions.`
      );
    } else {
      throw new Error(
        `[simulateGame] Game ended without reaching game_end phase\n` +
        `Final phase: ${finalState.phase}\n` +
        `Actions executed: ${actionsExecuted}\n` +
        `This should never happen - indicates a logic bug in the game loop.`
      );
    }
  }

  // Count hands if we started mid-hand
  // If game ended and we never counted a hand transition, count 1 hand
  if (handsPlayed === 0) {
    handsPlayed = 1;
  }

  // Determine winner based on MARKS (not scores)
  // teamMarks is the actual game score, teamScores is just the current hand
  const winner = finalState.teamMarks[0] > finalState.teamMarks[1] ? 0 : 1;

  return {
    winner,
    finalScores: finalState.teamMarks, // Return marks (game score), not hand scores
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
      allAI: true
      // No maxHands override - use default or options value to allow redeals
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

    // Test this seed (no maxHands override - allow redeals)
    const { winRate } = await testSeedWinRate(seed, simulationsPerSeed);

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