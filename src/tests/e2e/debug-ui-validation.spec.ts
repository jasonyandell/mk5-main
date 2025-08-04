import { test, expect } from '@playwright/test';

test.describe('Debug UI Validation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load debug interface correctly', async ({ page }) => {
    // Check the title is correct
    await expect(page.locator('h1')).toContainText('Texas 42 Debug Interface');
    
    // Check the phase indicator is present
    await expect(page.locator('[data-testid="game-phase"]')).toHaveText('bidding');
    
    // Check if actions count is showing
    const actionsCount = page.locator('[data-testid="actions-count"]');
    await expect(actionsCount).toBeVisible();
    
    const count = await actionsCount.textContent();
    console.log('Actions count shown:', count);
    
    // Count should be greater than 0
    expect(parseInt(count || '0')).toBeGreaterThan(0);
  });

  test('should show available actions', async ({ page }) => {
    // Listen to console logs
    page.on('console', msg => console.log('Browser console:', msg.text()));
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    
    // Wait a bit for reactive updates - optimized for offline testing
    await page.waitForTimeout(100);
    
    // Check the actions count first
    const actionsCount = page.locator('[data-testid="actions-count"]');
    const countText = await actionsCount.textContent();
    console.log('Actions count displayed:', countText);
    
    // Look for any action buttons using the generic test ID attribute  
    const actionButtons = page.locator('[data-generic-testid*="action-button"]');
    const bidButtons = page.locator('[data-generic-testid*="bid-button"]');
    
    // Log what we find
    const actionCount = await actionButtons.count();
    const bidCount = await bidButtons.count();
    
    console.log('Action buttons found:', actionCount);
    console.log('Bid buttons found:', bidCount);
    
    // Take a screenshot to see what's happening
    await page.screenshot({ path: 'test-results/debug-actions.png' });
    
    // Get all text from buttons
    if (actionCount > 0) {
      const actionTexts = await actionButtons.allTextContents();
      console.log('Action button texts:', actionTexts);
    }
    
    if (bidCount > 0) {
      const bidTexts = await bidButtons.allTextContents();
      console.log('Bid button texts:', bidTexts);
    }
    
    // Should have some actions available
    expect(actionCount + bidCount).toBeGreaterThan(0);
  });

  test('should show game state information', async ({ page }) => {
    // Check that debug panels are visible
    await expect(page.locator('[data-testid="debug-panel"]')).toBeVisible();
    
    // Check phase is bidding
    await expect(page.locator('[data-testid="game-phase"]')).toHaveText('bidding');
  });
});