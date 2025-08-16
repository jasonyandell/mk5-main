import { test, expect } from '@playwright/test';

test.describe('Complete Game End-to-End', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load game interface correctly', async ({ page }) => {
    // Verify page loads
    await page.waitForSelector('.app-container', { timeout: 5000 });
    
    // Verify game phase is displayed
    await expect(page.locator('.phase-name')).toContainText('Bidding');
    
    // Verify navigation is present
    await expect(page.locator('nav')).toBeVisible();
    
    // Verify header is present
    await expect(page.locator('.app-header')).toBeVisible();
    
    // Verify main content area is present
    await expect(page.locator('.game-container')).toBeVisible();
  });

  test('should handle basic bidding flow', async ({ page }) => {
    // Wait for page to load
    await page.waitForSelector('.app-container', { timeout: 5000 });
    
    // Navigate to actions panel
    await page.locator('[data-testid="nav-actions"]').click();
    await page.waitForTimeout(200);
    
    // Check if bid buttons are available
    const bidButtons = page.locator('[data-testid^="bid-"]');
    const passButton = page.locator('[data-testid="pass"]');
    const buttonCount = await bidButtons.count() + await passButton.count();
    
    expect(buttonCount).toBeGreaterThan(0);
    
    // Click the first available action button
    if (await bidButtons.count() > 0) {
      await bidButtons.first().click();
    } else {
      await passButton.click();
    }
    
    // Verify the bid was registered
    await page.waitForTimeout(100); // Small delay for state update
    
    // Check game phase - should still be in bidding or advanced
    const phaseElement = page.locator('.phase-name');
    const phaseText = await phaseElement.textContent();
    expect(['Bidding', 'Trump Selection']).toContain(phaseText);
  });

  test('should display game controls correctly', async ({ page }) => {
    // Verify navigation controls are present
    await expect(page.locator('[data-testid="nav-game"]')).toBeVisible();
    await expect(page.locator('[data-testid="nav-actions"]')).toBeVisible();
    await expect(page.locator('[data-testid="nav-debug"]')).toBeVisible();
  });

  test('should reset game correctly', async ({ page }) => {
    // Reload page to reset game
    await page.reload();
    await page.waitForSelector('.app-container', { timeout: 5000 });
    
    // Verify game resets to bidding phase
    await expect(page.locator('.phase-name')).toContainText('Bidding');
    
    // Verify scores reset
    await expect(page.locator('.score-card.us .score-value')).toContainText('0');
    await expect(page.locator('.score-card.them .score-value')).toContainText('0');
  });

  test('should display responsive layout', async ({ page }) => {
    // Test desktop layout
    await page.setViewportSize({ width: 1200, height: 800 });
    await expect(page.locator('.app-container')).toBeVisible();
    
    // Test tablet layout
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(page.locator('.app-container')).toBeVisible();
    
    // Test mobile layout
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator('.app-container')).toBeVisible();
  });

  test('should show correct team assignments', async ({ page }) => {
    // Check that team scores are displayed in header
    await expect(page.locator('.score-card.us')).toBeVisible();
    await expect(page.locator('.score-card.them')).toBeVisible();
    
    // Navigate to game view to see playing area
    await page.locator('[data-testid="nav-game"]').click();
    await page.waitForTimeout(200);
    
    // Check that playing area is displayed
    await expect(page.locator('.playing-area')).toBeVisible();
  });

  test('should handle accessibility features', async ({ page }) => {
    // Navigate to actions panel
    await page.locator('[data-testid="nav-actions"]').click();
    await page.waitForTimeout(200);
    
    // Check for proper button structure
    const actionButtons = page.locator('.action-button');
    const buttonCount = await actionButtons.count();
    expect(buttonCount).toBeGreaterThan(0);
    
    // Verify buttons are keyboard accessible
    await page.keyboard.press('Tab');
    const focusedElement = page.locator(':focus');
    await expect(focusedElement).toHaveCount(1);
    
    // Verify buttons have title attributes for tooltips
    const firstButton = actionButtons.first();
    await expect(firstButton).toHaveAttribute('title');
  });

  test('should maintain game state consistency', async ({ page }) => {
    // Verify game phase is shown in header
    await expect(page.locator('.phase-name')).toContainText('Bidding');
    
    // Navigate to debug panel to see game state
    await page.locator('[data-testid="nav-debug"]').click();
    await page.waitForTimeout(200);
    
    // Check that debug panel is visible
    await expect(page.locator('.debug-panel')).toBeVisible();
    
    // Close debug panel first to avoid backdrop issues
    await page.locator('.close-button').click();
    await page.waitForTimeout(200);
    
    // Verify actions are available by navigating to actions panel
    await page.locator('[data-testid="nav-actions"]').click();
    await page.waitForTimeout(200);
    
    const actionButtons = page.locator('.action-button');
    const count = await actionButtons.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should handle game flow transitions', async ({ page }) => {
    // Start in bidding phase
    await expect(page.locator('.phase-name')).toContainText('Bidding');
    
    // Navigate to actions panel
    await page.locator('[data-testid="nav-actions"]').click();
    await page.waitForTimeout(200);
    
    // Verify bid actions are available
    const bidButtons = page.locator('[data-testid^="bid-"]');
    await expect(bidButtons.first()).toBeVisible();
    
    // Verify that actions are available
    const buttonCount = await bidButtons.count() + await page.locator('[data-testid="pass"]').count();
    expect(buttonCount).toBeGreaterThan(0);
  });
});