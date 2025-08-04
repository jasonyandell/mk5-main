import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('AI Quickplay Visibility', () => {
  test('quickplay panel remains visible when action history is expanded', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(12345);
    
    // Make some moves to create action history
    const actions = await helper.getAvailableActions();
    if (actions.length > 0) {
      await helper.selectActionByIndex(0);
      await page.waitForTimeout(100);
      
      const actions2 = await helper.getAvailableActions();
      if (actions2.length > 0) {
        await helper.selectActionByIndex(0);
        await page.waitForTimeout(100);
      }
    }
    
    // Verify quickplay panel is visible
    const quickplayPanel = helper.locator('.quickplay-panel');
    await expect(quickplayPanel).toBeVisible();
    
    // Expand action history
    const toggleButton = helper.locator('.toggle-btn').filter({ hasText: 'Show Actions' });
    if (await toggleButton.isVisible()) {
      await toggleButton.click();
      
      // Verify action history is expanded
      const actionLog = helper.locator('.action-log');
      await expect(actionLog).toBeVisible();
    }
    
    // Verify quickplay panel is still visible after expanding action history
    await expect(quickplayPanel).toBeVisible();
    
    // Verify we can interact with quickplay controls
    const runButton = helper.locator('[data-testid="quickplay-run"]');
    await expect(runButton).toBeVisible();
    await expect(runButton).toBeInViewport();
    
    // Verify the speed selector is accessible
    const speedSelector = helper.locator('[data-testid="quickplay-speed"]');
    await expect(speedSelector).toBeVisible();
    
    // Verify AI player checkboxes are accessible
    const aiPlayerCheckbox = helper.locator('[data-testid="ai-player-0"]');
    await expect(aiPlayerCheckbox).toBeVisible();
  });

  test('right panel scrolls when content exceeds viewport', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(12345);
    
    // Get the right panel element
    const rightPanel = helper.locator('.debug-right');
    
    // Check if it has scrollable overflow
    const hasScroll = await rightPanel.evaluate((el) => {
      return el.scrollHeight > el.clientHeight;
    });
    
    // If content exceeds viewport, scrolling should be available
    if (hasScroll) {
      // Verify overflow-y is set to auto or scroll
      const overflowY = await rightPanel.evaluate((el) => {
        return window.getComputedStyle(el).overflowY;
      });
      expect(['auto', 'scroll']).toContain(overflowY);
    }
    
    // Verify quickplay panel is at the top of right panel
    const quickplayPanel = helper.locator('.quickplay-panel');
    const quickplayBox = await quickplayPanel.boundingBox();
    const rightPanelBox = await rightPanel.boundingBox();
    
    if (quickplayBox && rightPanelBox) {
      // Quickplay should be near the top of the right panel
      expect(quickplayBox.y).toBeLessThanOrEqual(rightPanelBox.y + 50);
    }
  });
});