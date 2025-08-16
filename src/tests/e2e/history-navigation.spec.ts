import { test, expect } from '@playwright/test';

test.describe('History Navigation URL Preservation', () => {
  test('should preserve URL when clicking history items in debug panel', async ({ page }) => {
    // Navigate to a URL with multiple actions
    const testURL = '/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ4NzIzNjEzOTN9LCJhIjpbeyJpIjoicCJ9LHsiaSI6IjMwIn0seyJpIjoiMzEifSx7ImkiOiIzMiJ9LHsiaSI6InRydW1wLWJsYW5rcyJ9LHsiaSI6IjExIn0seyJpIjoiMjEifSx7ImkiOiIzMSJ9LHsiaSI6IjYxIn0seyJpIjoiY29tcGxldGUtdHJpY2sifSx7ImkiOiIzMyJ9LHsiaSI6IjAwIn0seyJpIjoiNTMifSx7ImkiOiI2MyJ9LHsiaSI6ImNvbXBsZXRlLXRyaWNrIn1dfQ';
    await page.goto(testURL);
    
    // Wait for app to load
    await page.waitForSelector('.app-container', { timeout: 5000 });
    
    // Open debug panel using the mobile debug button
    const debugButton = page.locator('.nav-button.debug').first();
    await debugButton.click();
    
    // Wait for debug panel to open
    await page.waitForSelector('.debug-panel', { timeout: 5000 });
    
    // Click on History tab
    await page.click('button:has-text("History")');
    
    // Wait for history items to load
    await page.waitForSelector('.history-item', { timeout: 5000 });
    
    // Get initial URL
    const initialURL = page.url();
    expect(initialURL).toContain('?d=');
    
    // Count history items
    const historyItems = await page.$$('.history-item');
    console.log(`Found ${historyItems.length} history items`);
    expect(historyItems.length).toBeGreaterThan(0);
    
    // Click on a history item's time travel button
    if (historyItems.length >= 5) {
      // Click time travel on the 5th item
      await page.click('.history-item:nth-child(5) .time-travel-button');
      
      // Wait a moment for URL to update
      await page.waitForTimeout(1000);
      
      // Check that URL still contains state data
      const newURL = page.url();
      console.log('Initial URL:', initialURL);
      console.log('New URL:', newURL);
      
      expect(newURL).toContain('?d=');
      
      // The URL should be different from the initial one
      expect(newURL).not.toBe(initialURL);
      
      // Verify the debug panel is still visible
      await expect(page.locator('.debug-panel')).toBeVisible();
    }
  });
});