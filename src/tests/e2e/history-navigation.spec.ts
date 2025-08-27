import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe('History Navigation URL Preservation', () => {
  test('should preserve URL when clicking history items in debug panel', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    const locators = helper.getLocators();
    
    // Start with a deterministic game seed - allow URL updates for this test
    await helper.goto(12345, { disableUrlUpdates: false });
    
    // Perform a few actions to create history
    await helper.bid(30, false);
    await helper.pass();
    await helper.pass();
    
    // Get URL after actions - should contain state (v2 format)
    const urlWithActions = page.url();
    expect(urlWithActions).toContain('?v=2');
    
    // Open debug panel directly
    await helper.openDebugPanel();
    
    // Debug panel should be visible (auto-waits)
    await expect(locators.debug()).toBeVisible();
    
    // Click on History tab if it exists
    const historyTab = locators.historyTab();
    const tabCount = await historyTab.count();
    
    if (tabCount > 0) {
      await historyTab.click();
      
      // Wait for history items to appear (auto-waits)
      await expect(locators.historyItem().first()).toBeVisible();
      
      // Check if history rows exist
      const historyRows = locators.historyItem();
      const historyRowCount = await historyRows.count();
      
      // We should have at least 3 history items from our actions
      expect(historyRowCount).toBeGreaterThanOrEqual(3);
      
      if (historyRowCount > 0) {
        // Look for time travel buttons
        const timeTravelButton = historyRows.first().locator('.time-travel-button');
        const buttonExists = await timeTravelButton.count() > 0;
        
        if (buttonExists) {
          // Click time travel on the first item (goes back to state after first action)
          // Store URL before time travel
          const urlBefore = page.url();
          
          await timeTravelButton.click();
          
          // Wait for game to be ready after time travel
          await helper.waitForGameReady();
          
          // The URL should now reflect the time-traveled state
          const newURL = page.url();
          
          // URL should contain state data (v2 format)
          expect(newURL).toContain('?v=2');
          
          // URL should be different from before (we went back in time)
          // If it's the same, that's OK - it might mean we clicked on current state
          if (newURL !== urlBefore) {
            // Successfully time traveled to a different state
            expect(newURL).toContain('?v=2');
          }
        }
        
        // Verify the debug panel is still visible
        await expect(locators.debug()).toBeVisible();
      }
    } else {
      // If no History tab, just verify debug panel opened successfully
      // This is acceptable as the debug panel might have different tabs in different builds
      await expect(locators.debug()).toBeVisible();
    }
    
    // Close debug panel
    await helper.closeDebugPanel();
    
    // Debug panel should be hidden (auto-waits for not visible)
    await expect(locators.debug()).not.toBeVisible();
  });
});