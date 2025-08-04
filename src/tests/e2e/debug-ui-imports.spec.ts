import { test, expect } from '@playwright/test';

test.describe('Debug UI Import Verification', () => {
  test('should load app without errors and open Debug UI', async ({ page }) => {
    // Navigate to the app
    await page.goto('/');
    
    // Wait for app to load
    await expect(page.locator('h1')).toContainText('Texas 42');
    
    // Verify debug toggle button exists
    const debugToggle = page.getByTestId('debug-toggle');
    await expect(debugToggle).toBeVisible();
    
    // Click to open Debug UI
    await debugToggle.click();
    
    // Verify Debug UI opens
    const debugPanel = page.getByTestId('debug-panel');
    await expect(debugPanel).toBeVisible();
    
    // Check all tabs are present
    await expect(page.getByTestId('tab-state')).toBeVisible();
    await expect(page.getByTestId('tab-actions')).toBeVisible();
    await expect(page.getByTestId('tab-history')).toBeVisible();
    await expect(page.getByTestId('tab-json')).toBeVisible();
    await expect(page.getByTestId('tab-tools')).toBeVisible();
    
    // Verify state tab content loads (default tab)
    await expect(page.getByTestId('phase')).toBeVisible();
    await expect(page.getByTestId('current-player')).toBeVisible();
    await expect(page.getByTestId('team-scores')).toBeVisible();
    
    // Click through each tab to verify components load
    await page.getByTestId('tab-actions').click();
    await page.waitForTimeout(100); // Brief wait for tab switch
    
    await page.getByTestId('tab-history').click();
    await page.waitForTimeout(100);
    
    await page.getByTestId('tab-json').click();
    await expect(page.getByTestId('copy-json')).toBeVisible();
    await expect(page.getByTestId('download-json')).toBeVisible();
    
    await page.getByTestId('tab-tools').click();
    await expect(page.getByTestId('generate-bug-report')).toBeVisible();
    await expect(page.getByTestId('generate-test')).toBeVisible();
    await expect(page.getByTestId('autoplay-toggle')).toBeVisible();
    
    // Close Debug UI
    await page.getByTestId('debug-close').click();
    await expect(debugPanel).not.toBeVisible();
    
    // Test keyboard shortcut
    await page.keyboard.press('Control+Shift+D');
    await expect(debugPanel).toBeVisible();
    
    // No console errors should occur
    const consoleErrors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });
    
    await page.waitForTimeout(500); // Wait a bit to catch any delayed errors
    expect(consoleErrors).toHaveLength(0);
  });
  
  test('should handle basic interactions without errors', async ({ page }) => {
    await page.goto('/');
    
    // Open Debug UI
    await page.getByTestId('debug-toggle').click();
    
    // Try some basic interactions
    const firstAction = page.locator('[data-testid^="bid-"]').first();
    const actionCount = await firstAction.count();
    
    if (actionCount > 0) {
      // Click an action if available
      await firstAction.click();
      
      // Verify state updated
      await page.waitForTimeout(200);
      
      // Check undo button becomes enabled
      const undoButton = page.getByTestId('undo-button');
      await expect(undoButton).toBeEnabled();
      
      // Test undo
      await undoButton.click();
      await page.waitForTimeout(200);
    }
    
    // Switch tabs again to ensure no errors
    await page.getByTestId('tab-state').click();
    await page.getByTestId('tab-json').click();
    
    // Copy JSON should work
    await page.getByTestId('copy-json').click();
    
    // Generate bug report
    await page.getByTestId('tab-tools').click();
    await page.fill('textarea', 'Test bug description');
    await page.getByTestId('generate-bug-report').click();
    await expect(page.getByTestId('copy-bug-report')).toBeVisible();
  });
});