import { test, expect } from '@playwright/test';

test.describe('History Navigation URL Preservation', () => {
  test('should preserve URL when clicking history items in debug panel', async ({ page }) => {
    // Start with a fresh game and perform a few actions to create history
    await page.goto('/');
    
    // Wait for app to load
    await page.waitForSelector('.app-container', { timeout: 5000 });
    
    // Perform a few actions to create history
    await page.locator('[data-testid="bid-30"]').click();
    await page.waitForTimeout(200);
    await page.locator('[data-testid="pass"]').click();
    await page.waitForTimeout(200);
    
    // Get URL after actions
    const urlWithActions = page.url();
    expect(urlWithActions).toContain('d=');
    
    // Open debug panel using the debug nav button
    const debugButton = page.locator('[data-testid="nav-debug"]');
    await debugButton.click();
    
    // Wait for debug panel to open
    await page.waitForSelector('.debug-panel', { timeout: 5000 });
    
    // Click on History tab
    await page.locator('.tab').filter({ hasText: 'History' }).click();
    
    // ISSUE: Test assumes debug panel UI exists - may not be present
    throw new Error('BRITTLE TEST: Relies on specific debug panel UI that may not exist');
    // Check if history rows exist
    const historyRowCount = await page.locator('.history-row').count();
    
    if (historyRowCount > 0) {
      // Click time travel on the first item
      await page.click('.history-row:first-child .time-travel-button');
      
      // Wait a moment for URL to update
      await page.waitForTimeout(1000);
      
      // Check that URL still contains state data
      const newURL = page.url();
      expect(newURL).toContain('?d=');
      
      // Verify the debug panel is still visible
      await expect(page.locator('.debug-panel')).toBeVisible();
    } else {
      // ISSUE: Test passes even when feature doesn't work
      throw new Error('FALSE POSITIVE: Test passes with console.log when no history found');
      // If no history, just verify the debug panel works
      console.log('No history items found, test passed by opening debug panel');
    }
  });
});