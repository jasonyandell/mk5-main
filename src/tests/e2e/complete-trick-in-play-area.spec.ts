import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe('Complete Trick in Play Area', () => {
  test('shows complete trick button in play area', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    const locators = helper.getLocators();
    
    // Load state with trump already selected (with AI players for automated play)
    await helper.loadStateWithActions(12345, ['bid-30', 'pass', 'pass', 'pass', 'trump-blanks'], ['human', 'ai', 'ai', 'ai']);
    
    // Playing area should be visible automatically in playing phase (auto-waits)
    await expect(locators.playingArea()).toBeVisible();
    
    // Play a complete trick (4 dominoes)
    await helper.playFullTrick();
    
    // Check that trick table is tappable (auto-waits for element)
    await expect(locators.trickTableTappable()).toBeVisible();
    
    // Verify tap indicator shows (will be agree action for player 0)
    await expect(locators.tapIndicator()).toBeVisible();
    
    // Click the tappable trick table
    await locators.trickTableTappable().click();
    
    // Wait for phase to remain in playing (next trick)
    await helper.waitForPhase('playing');
    
    // Verify we're still in playing phase
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('playing');
  });

  test('button DOM structure is correct', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    const locators = helper.getLocators();
    
    // Load state where we're in playing phase (with AI for automated play)
    await helper.loadStateWithActions(12345, ['bid-30', 'pass', 'pass', 'pass', 'trump-blanks'], ['human', 'ai', 'ai', 'ai']);
    
    // In playing phase, playing area should be visible (auto-waits)
    await expect(locators.playingArea()).toBeVisible();
    
    // Verify the DOM structure exists
    const playingAreaExists = await page.evaluate(() => {
      const area = document.querySelector('[data-testid="playing-area"]');
      return area ? 'PlayingArea exists' : 'PlayingArea not found';
    });
    
    expect(playingAreaExists).toBe('PlayingArea exists');
  });

  test('button appears for complete trick action', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    const locators = helper.getLocators();
    
    // Use the same approach as the working test - load state with actions (with AI players)
    await helper.loadStateWithActions(12345, [
      'bid-30', 'pass', 'pass', 'pass', 'trump-blanks'
    ], ['human', 'ai', 'ai', 'ai']);
    
    // Playing area should be visible automatically
    await expect(locators.playingArea()).toBeVisible();
    
    // Play complete trick
    await helper.playFullTrick();
    
    // Verify trick table becomes tappable (auto-waits)
    await expect(locators.trickTableTappable()).toBeVisible();
    
    // Verify tap indicator appears (will be agree action for player 0)
    await expect(locators.tapIndicator()).toBeVisible();
    
    // Test passes - action button appears correctly
  });
});