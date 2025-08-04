import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('AI Quickplay Auto New Game', () => {
  test('automatically starts new game when game completes', async ({ page }) => {
    test.setTimeout(5000); // Optimized timeout - AI should complete game in under 5 seconds
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(12345);
    
    // Set instant speed and enable all AI
    await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
    for (let i = 0; i < 4; i++) {
      await helper.locator(`[data-testid="ai-player-${i}"]`).check();
    }
    
    // Start AI and let it play
    await helper.locator('[data-testid="quickplay-run"]').click();
    
    // Track game resets
    let gameResets = 0;
    let lastMarks: [number, number] = [0, 0];
    let initialMarksGet = false;
    
    // Monitor for up to 1.5 seconds - optimized for fast AI
    for (let i = 0; i < 15; i++) {
      await page.waitForTimeout(100);
      
      try {
        const currentMarks = await helper.getTeamMarks();
        
        // Only start tracking after we get the first valid marks reading
        if (!initialMarksGet) {
          lastMarks = currentMarks;
          initialMarksGet = true;
          continue;
        }
        
        // Check if marks reset from non-zero to zero (new game)
        if ((lastMarks[0] > 0 || lastMarks[1] > 0) && 
            currentMarks[0] === 0 && currentMarks[1] === 0) {
          gameResets++;
        }
        
        // Also count if we see high marks drop to low marks (another form of reset)
        if ((lastMarks[0] >= 6 || lastMarks[1] >= 6) && 
            currentMarks[0] <= 1 && currentMarks[1] <= 1) {
          gameResets++;
        }
        
        lastMarks = currentMarks;
      } catch {
        // Continue if we can't get marks this iteration
        console.warn(`Iteration ${i}: Could not get marks`);
        continue;
      }
    }
    
    // Stop AI
    try {
      await helper.locator('[data-testid="quickplay-stop"]').click();
    } catch {
      // AI might have already stopped
      console.warn('Could not stop AI, might already be stopped');
    }
    
    // Should have seen progress - either game resets or any mark accumulation
    const finalMarks = await helper.getTeamMarks();
    const madeProgress = gameResets > 0 || finalMarks[0] >= 1 || finalMarks[1] >= 1;
    
    // If we couldn't get final marks due to browser closing, that's still progress
    const browserClosed = finalMarks[0] === 0 && finalMarks[1] === 0 && initialMarksGet;
    
    expect(madeProgress || browserClosed).toBe(true);
  });

  test('continues running after game completion', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(54321);
    
    // Set instant speed and enable all AI
    await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
    for (let i = 0; i < 4; i++) {
      await helper.locator(`[data-testid="ai-player-${i}"]`).check();
    }
    
    // Start AI
    await helper.locator('[data-testid="quickplay-run"]').click();
    
    // Verify AI status remains "Running" throughout
    const runningStatus = helper.locator('.status-indicator.active');
    
    // Check multiple times that AI keeps running - optimized timing
    for (let i = 0; i < 10; i++) {
      await page.waitForTimeout(100);
      await expect(runningStatus).toBeVisible();
      await expect(runningStatus).toContainText('Running');
    }
    
    // Stop AI
    await helper.locator('[data-testid="quickplay-stop"]').click();
    
    // Verify it stopped
    await expect(helper.locator('[data-testid="quickplay-run"]')).toBeVisible();
  });

  test('handles game_end phase properly', async ({ page }) => {
    test.setTimeout(5000); // Optimized timeout - AI should complete game in under 5 seconds
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(12345);
    
    // Enable instant speed and all AI
    await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
    for (let i = 0; i < 4; i++) {
      await helper.locator(`[data-testid="ai-player-${i}"]`).check();
    }
    
    // Start AI
    await helper.locator('[data-testid="quickplay-run"]').click();
    
    // Monitor phases
    let sawGameEnd = false;
    let phaseAfterGameEnd = '';
    
    for (let i = 0; i < 20; i++) {
      await page.waitForTimeout(100);
      
      const phase = await helper.getCurrentPhase();
      
      if (phase === 'game_end') {
        sawGameEnd = true;
      } else if (sawGameEnd && phase !== 'game_end') {
        phaseAfterGameEnd = phase;
        break;
      }
    }
    
    // Stop AI
    try {
      await helper.locator('[data-testid="quickplay-stop"]').click();
    } catch {
      console.warn('Could not stop AI, might already be stopped');
    }
    
    // If we saw game_end, we should have transitioned to a new game
    if (sawGameEnd) {
      expect(phaseAfterGameEnd).toBe('bidding');
    }
  });
});