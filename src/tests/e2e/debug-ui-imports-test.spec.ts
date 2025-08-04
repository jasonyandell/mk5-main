import { test, expect } from '@playwright/test';

test.describe('Debug UI Import Test', () => {
  test('should load UI without errors and interact with basic elements', async ({ page }) => {
    // Navigate to the app
    await page.goto('/');
    
    // Wait for the app to load
    await page.waitForSelector('header', { timeout: 5000 });
    
    // Check that main components are visible
    await expect(page.locator('header')).toBeVisible();
    await expect(page.locator('.game-progress')).toBeVisible();
    await expect(page.locator('.playing-area')).toBeVisible();
    await expect(page.locator('.action-panel')).toBeVisible();
    
    // Check phase badge is visible
    await expect(page.locator('.phase-badge')).toBeVisible();
    
    // Try clicking on AI controls
    const playAllButton = page.locator('button:has-text("Play All")');
    if (await playAllButton.isVisible()) {
      await playAllButton.click();
      
      // Verify AI status appears
      await expect(page.locator('.ai-status')).toBeVisible();
      
      // Click pause button
      await page.locator('button:has-text("Pause")').click();
    }
    
    // Open debug panel with keyboard shortcut
    await page.keyboard.press('Control+Shift+D');
    await expect(page.locator('.debug-panel')).toBeVisible();
    
    // Try switching tabs in debug panel
    await page.locator('.tab:has-text("Actions & History")').click();
    await expect(page.locator('.actions-tab')).toBeVisible();
    
    // Close debug panel
    await page.keyboard.press('Escape');
    await expect(page.locator('.debug-panel')).not.toBeVisible();
    
    // Click on some actions if available
    const bidButton = page.locator('[data-testid="bid-30"]');
    if (await bidButton.isVisible()) {
      await bidButton.click();
    }
    
    // Verify no console errors
    const consoleErrors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });
    
    // Wait a bit to catch any delayed errors
    await page.waitForTimeout(1000);
    
    expect(consoleErrors).toHaveLength(0);
  });

  test('should handle domino interactions', async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('header', { timeout: 5000 });
    
    // Look for any playable dominoes
    const playableDomino = page.locator('.domino.playable').first();
    if (await playableDomino.count() > 0) {
      await playableDomino.click();
    }
    
    // Check that dominoes render with pips
    await expect(page.locator('.domino .pip').first()).toBeVisible();
  });
});