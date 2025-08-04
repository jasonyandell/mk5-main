import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('AI Quickplay Default Speed', () => {
  test('default speed is instant', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(12345);
    
    // Check that speed selector defaults to instant
    const speedSelector = helper.locator('[data-testid="quickplay-speed"]');
    await expect(speedSelector).toHaveValue('instant');
    
    // Start AI without changing speed
    await helper.locator('[data-testid="quickplay-run"]').click();
    
    // Verify status shows instant speed
    const statusIndicator = helper.locator('.status-indicator.active');
    await expect(statusIndicator).toBeVisible();
    await expect(statusIndicator).toContainText('Running (instant)');
    
    // Stop AI
    await helper.locator('[data-testid="quickplay-stop"]').click();
  });
});