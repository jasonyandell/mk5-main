import { createInitialState, getNextStates, type GameState } from '../index';
import { RandomAIStrategy, BeginnerAIStrategy } from '../ai/strategies';

// Helper to yield control to UI thread
const yieldToUI = () => new Promise(resolve => setTimeout(resolve, 0));

// Helper to log seed results with consistent formatting
const logSeedResult = (message: string, seed: number, winRate: number) => {
  console.log(`${message} ${seed} with win rate ${(winRate * 100).toFixed(1)}%`);
};

// Stateless AI strategies (can be reused)
const RANDOM_STRATEGY = new RandomAIStrategy();
const BEGINNER_STRATEGY = new BeginnerAIStrategy();

// Configuration - can be overridden for testing
export const SEED_FINDER_CONFIG = {
  MAX_SEEDS_TO_TRY: 10000,      // Maximum seeds to test before giving up
  GAMES_PER_SEED: 100,          // Games to play per seed for evaluation
  SEARCH_TIMEOUT_MS: 300000,    // Maximum time to search for a seed
  PROGRESS_REPORT_INTERVAL: 1, // Report progress every N games
  MAX_ACTIONS_PER_GAME: 500,    // Safety limit for game simulation
  TARGET_WIN_RATE_MIN: 0.02,    // Minimum acceptable win rate (2%)
  TARGET_WIN_RATE_MAX: 0.2,     // Maximum acceptable win rate (20%)
  FALLBACK_SEED: 424242,        // Default seed if search fails
};

// Export for components that need these values
export const TARGET_WIN_RATE_MIN = SEED_FINDER_CONFIG.TARGET_WIN_RATE_MIN;
export const TARGET_WIN_RATE_MAX = SEED_FINDER_CONFIG.TARGET_WIN_RATE_MAX;

export interface SeedProgress {
  currentSeed: number;
  seedsTried: number;
  gamesPlayed: number;
  totalGames: number;
  currentWinRate: number;
  bestSeed?: number;
  bestWinRate?: number;
}

export interface SeedResult {
  seed: number;
  winRate: number;
  gamesPlayed: number;
}

function apply(state:GameState, actionId:string): GameState {
  const transitions = getNextStates(state);
  const action = transitions.find(t => t.id === actionId);
  if (!action) throw Error(`No action`);
  return action?.newState
}

/**
 * Simulates a single game with Random AI playing as human (player 0)
 * Returns true if team 0 (human's team) wins
 */
function simulateSingleGame(seed: number): boolean {
  // Create initial state with Random AI as player 0, Beginner AIs as players 1-3
  let state = createInitialState({
    shuffleSeed: seed,
    playerTypes: ['ai', 'ai', 'ai', 'ai']
  });

  let actionCount = 0;

  state = apply(state, 'pass');
  state = apply(state, 'pass');
  state = apply(state, 'pass');
  state = apply(state, 'bid-30');
  state = BEGINNER_STRATEGY.chooseAction(state,getNextStates(state)).newState; // choose trumps

  // Play one hand
  while (state.phase !== 'bidding' && actionCount < SEED_FINDER_CONFIG.MAX_ACTIONS_PER_GAME) {
    const transitions = getNextStates(state);
    if (transitions.length === 0) break;

    // Use Random strategy for player 0, Beginner strategy for players 1-3
    const strategy = state.currentPlayer === 0 ? RANDOM_STRATEGY : BEGINNER_STRATEGY;
    const action = strategy.chooseAction(state, transitions);
    state = action.newState;
    actionCount++;
  }

  const team0Score = state.teamMarks[0];
 
  return team0Score === 1;
}

/**
 * Evaluates a seed by playing multiple games
 * Returns the win rate for team 0 (human's team)
 */
export async function evaluateSeed(
  baseSeed: number,
  gamesCount: number = SEED_FINDER_CONFIG.GAMES_PER_SEED,
  onProgress?: (gamesPlayed: number, wins: number) => void
): Promise<SeedResult> {
  let wins = 0;
  
  for (let i = 0; i < gamesCount; i++) {
    // Use different seed for each game to get variety
    // but deterministic based on base seed
    const gameSeed = baseSeed;
    
    if (simulateSingleGame(gameSeed)) {
      wins++;
    }
    
    // Report progress at configured intervals
    if (onProgress && (i + 1) % SEED_FINDER_CONFIG.PROGRESS_REPORT_INTERVAL === 0) {
      onProgress(i + 1, wins);
      // Yield to UI
      await yieldToUI();
    }
  }
  
  return {
    seed: baseSeed,
    winRate: wins / gamesCount,
    gamesPlayed: gamesCount
  };
}

/**
 * Finds a balanced seed where random AI wins between 10% and 50% of games
 */
export async function findBalancedSeed(
  onProgress?: (progress: SeedProgress) => void,
  shouldUseBest?: () => boolean
): Promise<{ seed: number; winRate: number }> {
  const startTime = Date.now();

  // Track the best (lowest win rate) seed found so far
  let bestSeed: number | null = null;
  let bestWinRate = 1.0; // Start with worst possible

  // Helper to build progress report
  const reportProgress = (seed: number, attempt: number, gamesPlayed: number, currentWinRate: number) => {
    if (onProgress) {
      const progress: SeedProgress = {
        currentSeed: seed,
        seedsTried: attempt + 1,
        gamesPlayed,
        totalGames: SEED_FINDER_CONFIG.GAMES_PER_SEED,
        currentWinRate
      };
      if (bestSeed !== null) {
        progress.bestSeed = bestSeed;
        progress.bestWinRate = bestWinRate;
      }
      onProgress(progress);
    }
  };

  // Helper to evaluate and return fallback seed
  const useFallbackSeed = async (): Promise<{ seed: number; winRate: number }> => {
    console.warn('Could not find balanced seed - using fallback');
    const fallbackResult = await evaluateSeed(SEED_FINDER_CONFIG.FALLBACK_SEED, SEED_FINDER_CONFIG.GAMES_PER_SEED);
    logSeedResult('Fallback seed', SEED_FINDER_CONFIG.FALLBACK_SEED, fallbackResult.winRate);
    return { seed: SEED_FINDER_CONFIG.FALLBACK_SEED, winRate: fallbackResult.winRate };
  };

  for (let attempt = 0; attempt < SEED_FINDER_CONFIG.MAX_SEEDS_TO_TRY; attempt++) {
    // Check timeout
    if (Date.now() - startTime > SEED_FINDER_CONFIG.SEARCH_TIMEOUT_MS) {
      console.warn('Seed finder timeout');
      return useFallbackSeed();
    }
    
    // Generate random seed
    const seed = Math.floor(Math.random() * 1000000);

    // Report initial progress for this seed
    reportProgress(seed, attempt, 0, 0);

    // Yield to UI to show the seed number immediately
    await yieldToUI();

    // Evaluate this seed
    const result = await evaluateSeed(
      seed, 
      SEED_FINDER_CONFIG.GAMES_PER_SEED,
      (games, wins) => {
        // Report progress with running win rate
        reportProgress(seed, attempt, games, wins / games);
      }
    );
    
    // Update best seed if this one is better (lower win rate)
    if (result.winRate < bestWinRate) {
      bestSeed = seed;
      bestWinRate = result.winRate;
    }

    // Report final win rate
    reportProgress(seed, attempt, SEED_FINDER_CONFIG.GAMES_PER_SEED, result.winRate);

    // Check if this seed is balanced (within target win rate range)
    if (result.winRate >= SEED_FINDER_CONFIG.TARGET_WIN_RATE_MIN && result.winRate <= SEED_FINDER_CONFIG.TARGET_WIN_RATE_MAX) {
      logSeedResult('Found balanced seed', seed, result.winRate);
      return { seed, winRate: result.winRate };
    }

    // Check if user wants to use best seed found so far
    if (shouldUseBest && shouldUseBest() && bestSeed !== null) {
      logSeedResult('Using best seed found:', bestSeed, bestWinRate);
      return { seed: bestSeed, winRate: bestWinRate };
    }

    // Yield to UI between seeds to keep the interface responsive
    await yieldToUI();
  }

  // If we found any seed, use the best one
  if (bestSeed !== null) {
    logSeedResult('No balanced seed found, using best:', bestSeed, bestWinRate);
    return { seed: bestSeed, winRate: bestWinRate };
  }

  return useFallbackSeed();
}

