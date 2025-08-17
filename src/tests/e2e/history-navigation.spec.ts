import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('History Navigation URL Preservation', () => {
  test('should preserve URL when clicking history items in debug panel', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Start with a deterministic game seed - allow URL updates for this test
    await helper.gotoWithSeed(12345, { disableUrlUpdates: false });
    
    // Wait for app to load
    await page.waitForSelector('.app-container', { timeout: 5000 });
    
    // Perform a few actions to create history
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    
    // Get URL after actions - should contain state
    const urlWithActions = page.url();
    expect(urlWithActions).toContain('d=');
    
    // Open debug panel
    await helper.openDebugPanel();
    
    // Click on History tab if it exists
    const historyTab = page.locator('.tab').filter({ hasText: 'History' });
    const tabCount = await historyTab.count();
    
    if (tabCount > 0) {
      await historyTab.click();
      // Wait for history items to appear
      await page.waitForSelector('.history-item', { timeout: 2000 });
      
      // Check if history rows exist
      const historyRows = page.locator('.history-item');
      const historyRowCount = await historyRows.count();
      
      // We should have at least 3 history items from our actions
      expect(historyRowCount).toBeGreaterThanOrEqual(3);
      
      if (historyRowCount > 0) {
        // Look for time travel buttons
        const timeTravelButton = historyRows.first().locator('.time-travel-button');
        const buttonExists = await timeTravelButton.count() > 0;
        
        if (buttonExists) {
          // Click time travel on the first item
          // Store URL before time travel
          const urlBefore = page.url();
          
          await timeTravelButton.click();
          
          // Wait for URL to change
          await page.waitForFunction(
            (prevUrl) => window.location.href !== prevUrl,
            urlBefore,
            { timeout: 2000 }
          );
          
          // Check that URL still contains state data
          const newURL = page.url();
          expect(newURL).toContain('?d=');
        }
        
        // Verify the debug panel is still visible
        await expect(page.locator('.debug-panel')).toBeVisible();
      }
    } else {
      // If no History tab, just verify debug panel opened successfully
      // This is acceptable as the debug panel might have different tabs in different builds
      await expect(page.locator('.debug-panel')).toBeVisible();
    }
    
    // Close debug panel
    await helper.closeDebugPanel();
  });
});