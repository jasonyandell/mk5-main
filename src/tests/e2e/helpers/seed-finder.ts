import type { Page } from '@playwright/test';
import type { TestWindow } from '../test-window';

/**
 * Helper function to play one hand to completion
 */
async function playOneHandToCompletion(page: Page): Promise<void> {
  // Set AI to instant speed
  await page.evaluate(() => {
    const w = window as unknown as TestWindow;
    if (w.setAISpeedProfile) w.setAISpeedProfile('instant');
  });
  
  // Play loop: nudge human turns until overlay appears
  const start = Date.now();
  while (Date.now() - start < 20000) {
    const done = await page.evaluate(() => {
      const w = window as unknown as TestWindow;
      const overlay = w.getSectionOverlay?.();
      if (overlay && overlay.type === 'oneHand') return true;
      if (w.playFirstAction) w.playFirstAction();
      return false;
    });
    if (done) break;
    await page.waitForTimeout(10);
  }
}

/**
 * Helper function to determine game outcome
 */
async function getGameOutcome(page: Page): Promise<'won' | 'lost'> {
  return await page.evaluate(() => {
    const w = window as unknown as TestWindow;
    const overlay = w.getSectionOverlay?.();
    const state = w.getGameState?.();
    
    // Check the overlay for outcome information
    if (overlay && 'weWon' in overlay) {
      return overlay.weWon ? 'won' : 'lost';
    }
    
    // Fallback: check team scores (for one hand, higher score wins)
    if (state?.teamScores && Array.isArray(state.teamScores)) {
      return state.teamScores[0] > state.teamScores[1] ? 'won' : 'lost';
    }
    
    // Default to lost if we can't determine
    return 'lost';
  });
}

/**
 * Finds a seed that produces the desired outcome by playing real games
 * @param page - Playwright page to run the game in
 * @param desiredOutcome - Whether we want team 0 to win or lose
 * @param maxAttempts - Maximum number of seeds to try (default 10)
 */
export async function findSeedWithOutcome(
  page: Page, 
  desiredOutcome: 'won' | 'lost',
  maxAttempts: number = 10
): Promise<number> {
  const startTime = Date.now();
  const timeout = 30000; // 30 second timeout
  
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    // Check timeout
    if (Date.now() - startTime > timeout) {
      throw new Error(`Timeout: Could not find seed for ${desiredOutcome} outcome in ${timeout}ms`);
    }
    
    // Generate a random seed
    const seed = Math.floor(Math.random() * 1000000);
    
    try {
      // Load the game with this seed
      await page.goto(`/?h=one_hand&s=${seed}`, { waitUntil: 'networkidle' });
      await page.waitForSelector('.app-container', { timeout: 5000 });
      
      // Play the game to completion
      await playOneHandToCompletion(page);
      
      // Check the outcome
      const outcome = await getGameOutcome(page);
      if (outcome === desiredOutcome) {
        return seed;
      }
    } catch (error) {
      // Game failed to load or play, try another seed
      console.warn(`Seed ${seed} failed:`, error);
    }
  }
  
  throw new Error(`Could not find seed for ${desiredOutcome} outcome in ${maxAttempts} attempts`);
}

/**
 * Cache for seeds found during the current test run
 * Not persisted between runs to handle AI changes
 */
class SeedCache {
  private winningSeeds: number[] = [];
  private losingSeeds: number[] = [];
  
  async getWinningSeed(page: Page): Promise<number> {
    if (this.winningSeeds.length > 0) {
      return this.winningSeeds[0]!;
    }
    const seed = await findSeedWithOutcome(page, 'won');
    this.winningSeeds.push(seed);
    return seed;
  }
  
  async getLosingSeed(page: Page): Promise<number> {
    if (this.losingSeeds.length > 0) {
      return this.losingSeeds[0]!;
    }
    const seed = await findSeedWithOutcome(page, 'lost');
    this.losingSeeds.push(seed);
    return seed;
  }
  
  // Clear cache (useful between test files)
  clear(): void {
    this.winningSeeds = [];
    this.losingSeeds = [];
  }
}

// Single instance per test run
export const seedCache = new SeedCache();