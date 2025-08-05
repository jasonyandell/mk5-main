import { test, expect } from '@playwright/test';

test.describe('Domino Highlighting - Simple', () => {
  test('should highlight dominoes during bidding', async ({ page }) => {
    // Navigate to the app
    await page.goto('/');
    
    // Wait for app to load
    await page.waitForTimeout(2000);
    
    // Click on Actions tab
    await page.click('[data-testid="nav-actions"]');
    
    // Wait for hand to be visible
    await expect(page.locator('.hand-section')).toBeVisible({ timeout: 5000 });
    
    // Find a domino to hover over (try to find 3-5 or any domino)
    const dominoes = page.locator('.hand-section .domino');
    const count = await dominoes.count();
    
    if (count > 0) {
      // Hover over first domino
      await dominoes.first().hover();
      
      // Check for highlighting
      const highlighted = page.locator('.domino.highlight-suit-0, .domino.highlight-suit-1, .domino.highlight-suit-2, .domino.highlight-suit-3, .domino.highlight-suit-4, .domino.highlight-suit-5, .domino.highlight-suit-6');
      await expect(highlighted).toHaveCount({ min: 1 });
      
      // Check for badge
      const badge = page.locator('.suit-highlight-badge');
      await expect(badge).toBeVisible();
      
      // Screenshot for debugging
      await page.screenshot({ path: 'test-results/bidding-highlight.png', fullPage: true });
    }
  });

  test('should show different highlighting during trump selection', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Make a bid to get to trump selection
    await page.click('[data-testid="nav-actions"]');
    await page.waitForTimeout(500);
    
    // Click bid-30
    const bid30 = page.locator('[data-testid="bid-30"]');
    if (await bid30.isVisible()) {
      await bid30.click();
      
      // Wait for trump selection
      await expect(page.locator('text=Select Trump')).toBeVisible({ timeout: 5000 });
      
      // Hover over a trump button
      const trumpButton = page.locator('.trump-button').first();
      await trumpButton.hover();
      
      // Check for trump button hover effect
      await expect(trumpButton).toHaveClass(/trump-hovering/);
      
      // Check for domino highlighting (should be primary only, no suit colors)
      const highlighted = page.locator('.domino.highlight-primary');
      const suitHighlighted = page.locator('[class*="highlight-suit-"]');
      
      await expect(highlighted.first()).toBeVisible();
      await expect(suitHighlighted).toHaveCount(0);
      
      // Screenshot
      await page.screenshot({ path: 'test-results/trump-selection-highlight.png', fullPage: true });
    }
  });
});