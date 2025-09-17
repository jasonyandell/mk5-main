import { createInitialState, getNextStates } from '../index';
import { RandomAIStrategy, BeginnerAIStrategy } from '../controllers/strategies';

// Configuration - can be overridden for testing
export const SEED_FINDER_CONFIG = {
  MAX_SEEDS_TO_TRY: 10000,         // Maximum seeds to test before giving up
  GAMES_PER_SEED: 100,              // Games to play per seed for evaluation
  SEARCH_TIMEOUT_MS: 30000,         // Maximum time to search for a seed (30 seconds)
  PROGRESS_REPORT_INTERVAL: 10,    // Report progress every N games
  MAX_ACTIONS_PER_GAME: 500,       // Safety limit for game simulation
  TARGET_WIN_RATE_MIN: 0.02,       // Minimum acceptable win rate (2%)
  TARGET_WIN_RATE_MAX: 0.25,       // Maximum acceptable win rate (25%)
  FALLBACK_SEED: 424242,            // Default seed if search fails
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
}

export interface SeedResult {
  seed: number;
  winRate: number;
  gamesPlayed: number;
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

  const randomStrategy = new RandomAIStrategy();
  const beginnerStrategy = new BeginnerAIStrategy();
  let actionCount = 0;

  // Play until game ends
  while (state.phase !== 'game_end' && actionCount < SEED_FINDER_CONFIG.MAX_ACTIONS_PER_GAME) {
    const transitions = getNextStates(state);
    if (transitions.length === 0) break;

    // Use Random strategy for player 0, Beginner strategy for players 1-3
    const strategy = state.currentPlayer === 0 ? randomStrategy : beginnerStrategy;
    const action = strategy.chooseAction(state, transitions);
    state = action.newState;
    actionCount++;
  }

  // Check if team 0 won (higher score)
  if (state.phase === 'game_end') {
    const team0Score = state.teamScores[0] || 0;
    const team1Score = state.teamScores[1] || 0;
    return team0Score > team1Score;
  }

  // If game didn't end properly, consider it a loss
  return false;
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
    const gameSeed = baseSeed + i * 1000;
    
    if (simulateSingleGame(gameSeed)) {
      wins++;
    }
    
    // Report progress at configured intervals
    if (onProgress && (i + 1) % SEED_FINDER_CONFIG.PROGRESS_REPORT_INTERVAL === 0) {
      onProgress(i + 1, wins);
      // Yield to UI
      await new Promise(resolve => setTimeout(resolve, 0));
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
  onProgress?: (progress: SeedProgress) => void
): Promise<number> {
  const startTime = Date.now();
  
  for (let attempt = 0; attempt < SEED_FINDER_CONFIG.MAX_SEEDS_TO_TRY; attempt++) {
    // Check timeout
    if (Date.now() - startTime > SEED_FINDER_CONFIG.SEARCH_TIMEOUT_MS) {
      console.warn('Seed finder timeout - using fallback seed');
      return SEED_FINDER_CONFIG.FALLBACK_SEED;
    }
    
    // Generate random seed
    const seed = Math.floor(Math.random() * 1000000);
    
    // Evaluate this seed
    const result = await evaluateSeed(
      seed, 
      SEED_FINDER_CONFIG.GAMES_PER_SEED,
      (games, wins) => {
        // Report progress with running win rate
        if (onProgress) {
          onProgress({
            currentSeed: seed,
            seedsTried: attempt + 1,
            gamesPlayed: games,
            totalGames: SEED_FINDER_CONFIG.GAMES_PER_SEED,
            currentWinRate: wins / games // Calculate running win rate
          });
        }
      }
    );
    
    // Report final win rate
    if (onProgress) {
      onProgress({
        currentSeed: seed,
        seedsTried: attempt + 1,
        gamesPlayed: SEED_FINDER_CONFIG.GAMES_PER_SEED,
        totalGames: SEED_FINDER_CONFIG.GAMES_PER_SEED,
        currentWinRate: result.winRate
      });
    }
    
    // Check if this seed is balanced (within target win rate range)
    if (result.winRate >= SEED_FINDER_CONFIG.TARGET_WIN_RATE_MIN && result.winRate <= SEED_FINDER_CONFIG.TARGET_WIN_RATE_MAX) {
      console.log(`Found balanced seed ${seed} with win rate ${(result.winRate * 100).toFixed(1)}%`);
      return seed;
    }
  }
  
  console.warn('Could not find balanced seed - using fallback');
  return SEED_FINDER_CONFIG.FALLBACK_SEED;
}

