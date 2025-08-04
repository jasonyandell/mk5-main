import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('AI Quickplay Step', () => {
  test('step button executes single AI action at instant speed', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Start game with deterministic seed
    await helper.gotoWithSeed(12345);
    
    // Set quickplay to instant speed
    await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
    
    // Get initial game state
    const phaseBefore = await helper.getCurrentPhase();
    const currentPlayerBefore = await helper.getCurrentPlayer();
    const actionsBefore = await helper.getAvailableActions();
    
    // Extract player number from "Current Player: P0" format
    const playerNum = parseInt(currentPlayerBefore.replace('Current Player: P', ''));
    
    // Enable AI only for current player
    await helper.locator(`[data-testid="ai-player-${playerNum}"]`).check();
    
    // Verify step button is enabled
    const stepButton = helper.locator('[data-testid="quickplay-step"]');
    await expect(stepButton).toBeEnabled();
    
    // Execute single step
    await stepButton.click();
    
    // Wait minimal time for instant speed
    await page.waitForTimeout(50);
    
    // Verify exactly one action was taken
    if (phaseBefore.includes('bidding')) {
      // In bidding phase, current player should have changed
      const currentPlayerAfter = await helper.getCurrentPlayer();
      expect(currentPlayerAfter).not.toBe(currentPlayerBefore);
      
      // Verify one of the available actions was selected
      expect(actionsBefore.length).toBeGreaterThan(0);
      expect(actionsBefore.some(a => 
        a.type === 'bid_points' || 
        a.type === 'bid_marks' || 
        a.type === 'pass'
      )).toBe(true);
    } else if (phaseBefore.includes('trump_selection')) {
      // After trump selection, should be in playing phase
      const phaseAfter = await helper.getCurrentPhase();
      expect(phaseAfter).toContain('playing');
      
      // Verify trump actions were available
      expect(actionsBefore.some(a => a.type === 'trump_selection')).toBe(true);
    } else if (phaseBefore.includes('playing')) {
      // In playing phase, either domino was played or trick completed
      const actionsAfter = await helper.getAvailableActions();
      
      // Verify play or complete_trick action was available
      expect(actionsBefore.some(a => 
        a.type === 'play_domino' || 
        a.type === 'complete_trick'
      )).toBe(true);
      
      // Game state should have advanced
      expect(actionsAfter).not.toEqual(actionsBefore);
    }
  });

  test('step button disabled when current player is not AI', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(12345);
    
    // Ensure all AI players are disabled
    for (let i = 0; i < 4; i++) {
      await helper.locator(`[data-testid="ai-player-${i}"]`).uncheck();
    }
    
    // Step button should be disabled
    const stepButton = helper.locator('[data-testid="quickplay-step"]');
    await expect(stepButton).toBeDisabled();
  });

  test('step button enabled only when current player is AI', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(12345);
    
    // Get current player
    const currentPlayer = await helper.getCurrentPlayer();
    const currentPlayerNum = parseInt(currentPlayer.replace('Current Player: P', ''));
    
    // Disable all players first
    for (let i = 0; i < 4; i++) {
      await helper.locator(`[data-testid="ai-player-${i}"]`).uncheck();
    }
    
    // Enable a different player as AI
    const otherPlayerNum = (currentPlayerNum + 1) % 4;
    await helper.locator(`[data-testid="ai-player-${otherPlayerNum}"]`).check();
    
    // Step button should still be disabled
    const stepButton = helper.locator('[data-testid="quickplay-step"]');
    await expect(stepButton).toBeDisabled();
    
    // Enable current player as AI
    await helper.locator(`[data-testid="ai-player-${currentPlayerNum}"]`).check();
    
    // Step button should now be enabled
    await expect(stepButton).toBeEnabled();
  });
});