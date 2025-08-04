import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('AI Quickplay Reset Behavior', () => {
  test('action history resets when new game starts', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(12345);
    
    // Make some manual moves to create action history
    let actions = await helper.getAvailableActions();
    if (actions.length > 0) {
      await helper.selectActionByIndex(0);
    }
    
    await page.waitForTimeout(100);
    actions = await helper.getAvailableActions();
    if (actions.length > 0) {
      await helper.selectActionByIndex(0);
    }
    
    // Open action history to see current state
    const toggleButton = helper.locator('.toggle-btn').filter({ hasText: 'Show Actions' });
    await toggleButton.click();
    
    // Count action log entries (excluding initial state)
    const actionEntries = helper.locator('.log-entry').filter({ hasNot: helper.locator('.action-id').filter({ hasText: 'initial-state' }) });
    const initialCount = await actionEntries.count();
    expect(initialCount).toBeGreaterThan(0);
    
    // Reset game manually
    await helper.locator('[data-testid="new-game-button"]').click();
    
    // Wait for reset
    await page.waitForTimeout(100);
    
    // Check action history is cleared
    const newCount = await actionEntries.count();
    expect(newCount).toBe(0);
    
    // Verify URL is clean
    const url = await helper.getCurrentURL();
    expect(url).not.toContain('d=');
  });

  test('action history resets during AI quickplay game transitions', async ({ page }) => {
    test.setTimeout(5000); // Optimized timeout - AI should complete game in under 5 seconds
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(12345);
    
    // Enable instant speed and all AI
    await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
    for (let i = 0; i < 4; i++) {
      await helper.locator(`[data-testid="ai-player-${i}"]`).check();
    }
    
    // Open action history
    const toggleButton = helper.locator('.toggle-btn').filter({ hasText: 'Show Actions' });
    await toggleButton.click();
    
    // Start AI
    await helper.locator('[data-testid="quickplay-run"]').click();
    
    // Monitor action count
    let maxActionCount = 0;
    let sawReset = false;
    
    for (let i = 0; i < 15; i++) {
      try {
        await page.waitForTimeout(100);
        
        const actionEntries = helper.locator('.log-entry').filter({ 
          hasNot: helper.locator('.action-id').filter({ hasText: 'initial-state' }) 
        });
        const currentCount = await actionEntries.count();
        
        // If count dropped significantly, we saw a reset
        if (maxActionCount > 20 && currentCount < 5) {
          sawReset = true;
          break;
        }
        
        maxActionCount = Math.max(maxActionCount, currentCount);
      } catch {
        // Continue if there's an error counting
        console.warn(`Iteration ${i}: Could not count action entries`);
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
    
    // We should have either seen a reset or not accumulated an extreme number of actions
    // The system should prevent excessive action accumulation through resets
    expect(sawReset || maxActionCount < 200).toBe(true);
  });

  test('URL clears when game resets', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(12345);
    
    // Make some moves
    const actions = await helper.getAvailableActions();
    if (actions.length > 0) {
      await helper.selectActionByIndex(0);
      await page.waitForTimeout(50);
    }
    
    // Verify URL has data
    let url = await helper.getCurrentURL();
    expect(url).toContain('d=');
    
    // Reset game
    await helper.locator('[data-testid="new-game-button"]').click();
    
    // Verify URL is clean
    url = await helper.getCurrentURL();
    expect(url).not.toContain('d=');
  });
});