import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';
import type { TestWindow } from './test-window';
import { findSeedWithOutcome } from './helpers/seed-finder';

test.setTimeout(30000);

// These helpers are now imported from seed-finder.ts
// Re-export them here for use in this test file
const playOneHandToCompletion = async (page: Page): Promise<void> => {
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
};

const getGameOutcome = async (page: Page): Promise<'won' | 'lost'> => {
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
};

// Helper to wait for modal to be visible
const waitForModal = async (page: Page): Promise<void> => {
  const modal = page.locator('.modal-box');
  await expect(modal).toBeVisible({ timeout: 5000 });
  
  // Also verify title is visible (We Won! or We Lost)
  const title = page.locator('.modal-box h3');
  await expect(title).toBeVisible({ timeout: 5000 });
};

test.describe('Sections: One Hand from URL shows completion modal', () => {
  test('Retry button replays same seed after loss', async ({ page }) => {
    // Find a seed that produces a loss (allow up to 10 attempts)
    const losingSeed = await findSeedWithOutcome(page, 'lost', 10);
    
    // Now test with that seed
    await page.goto(`/?h=one_hand&s=${losingSeed}`, { waitUntil: 'networkidle' });
    await page.waitForSelector('.app-container');
    await playOneHandToCompletion(page);
    await waitForModal(page);
    
    // Verify we lost
    const outcome = await getGameOutcome(page);
    expect(outcome).toBe('lost');
    
    // Verify URL was minimized (no h= param)
    const urlAfterCompletion = page.url();
    expect(urlAfterCompletion).toMatch(/\?s=/);
    expect(urlAfterCompletion).not.toMatch(/h=one_hand/);
    
    // Click Retry button (only appears when lost)
    const retryBtn = page.getByRole('button', { name: /Retry \(/ });
    await expect(retryBtn).toBeVisible();
    await retryBtn.click();
    
    // Play second hand to completion
    await playOneHandToCompletion(page);
    await waitForModal(page);
    
    // Verify same seed was used by checking the URL
    const urlAfterRetry = page.url();
    const urlParams = new URLSearchParams(urlAfterRetry.split('?')[1]);
    const seedFromUrl = urlParams.get('s');
    expect(seedFromUrl).toBe(String(losingSeed));
  });

  test('New button generates new seed after win', async ({ page }) => {
    // Find a seed that produces a win (allow up to 10 attempts)
    const winningSeed = await findSeedWithOutcome(page, 'won', 10);
    
    // Now test with that seed
    await page.goto(`/?h=one_hand&s=${winningSeed}`, { waitUntil: 'networkidle' });
    await page.waitForSelector('.app-container');
    await playOneHandToCompletion(page);
    await waitForModal(page);
    
    // Verify we won
    const outcome = await getGameOutcome(page);
    expect(outcome).toBe('won');
    
    // Verify URL was minimized
    const urlAfterCompletion = page.url();
    expect(urlAfterCompletion).toMatch(/\?s=/);
    expect(urlAfterCompletion).not.toMatch(/h=one_hand/);
    
    // Click New button (appears for both win and loss)
    const newBtn = page.getByRole('button', { name: 'New' });
    await expect(newBtn).toBeVisible();
    await newBtn.click();
    
    // Play second hand to completion
    await playOneHandToCompletion(page);
    await waitForModal(page);
    
    // Verify different seed was generated
    const seedAfterNew = await page.evaluate(() => 
      (window as unknown as TestWindow).getGameState?.()?.shuffleSeed as number
    );
    expect(seedAfterNew).not.toBe(winningSeed);
    expect(seedAfterNew).toBeTruthy(); // Should have a valid seed
  });

  test('Share results button appears after win', async ({ page }) => {
    // Find a seed that produces a win (allow up to 10 attempts)
    const winningSeed = await findSeedWithOutcome(page, 'won', 10);

    // Now test with that seed
    await page.goto(`/?h=one_hand&s=${winningSeed}`, { waitUntil: 'networkidle' });
    await page.waitForSelector('.app-container');
    await playOneHandToCompletion(page);
    await waitForModal(page);

    // Verify we won
    const outcome = await getGameOutcome(page);
    expect(outcome).toBe('won');

    // Share results button should be visible (only appears when won)
    // Use text locator since button has an icon
    const shareBtn = page.getByRole('button').filter({ hasText: 'Share results' });
    await expect(shareBtn).toBeVisible();

    // Click it to verify it works
    await shareBtn.click();

    // Wait a bit for the clipboard operation and state change
    await page.waitForTimeout(100);

    // Should change to "Copied!" after clicking - button changes to btn-success class
    // Try both selectors to see which one works
    const copiedBtn = page.locator('button').filter({ has: page.locator('text=Copied!') });
    await expect(copiedBtn).toBeVisible();
  });
});