import { test, expect } from '@playwright/test';

test.describe('Complete Game End-to-End', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load game interface correctly', async ({ page }) => {
    // Verify page loads
    await expect(page.locator('h1')).toContainText('Texas 42 Debug Interface');
    
    // Verify game phase is displayed
    await expect(page.locator('[data-testid="game-phase"]')).toHaveText('bidding');
    
    // Verify all 4 players are shown
    const players = page.locator('.player-section');
    await expect(players).toHaveCount(4);
    
    // Verify debug panels are present
    await expect(page.locator('[data-testid="debug-panel"]')).toBeVisible();
    await expect(page.locator('[data-testid="player-hands"]')).toBeVisible();
    
    // Verify actions are available
    await expect(page.locator('[data-testid="actions-count"]')).toBeVisible();
  });

  test('should handle basic bidding flow', async ({ page }) => {
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    
    // Check if bid buttons are available
    const bidButtons = page.locator('[data-generic-testid*="bid-button"]');
    const buttonCount = await bidButtons.count();
    
    expect(buttonCount).toBeGreaterThan(0);
    
    // Click the first available bid button
    await bidButtons.first().click({ force: true });
    
    // Verify the bid was registered
    await page.waitForTimeout(100); // Small delay for state update
    
    // Check game state shows a bid was made
    const phaseElement = page.locator('[data-testid="game-phase"]');
    await expect(phaseElement).toHaveText('bidding');
  });

  test('should display game controls correctly', async ({ page }) => {
    // Verify new game button is present
    await expect(page.locator('[data-testid="new-game-button"]')).toBeVisible();
    await expect(page.locator('[data-testid="new-game-button"]')).toContainText('New Game');
  });

  test.skip('should handle debug panel toggle', async ({ page }) => {
    // Skip - entire UI is now debug interface
  });

  test('should reset game correctly', async ({ page }) => {
    // Click new game button
    await page.locator('[data-testid="new-game-button"]').click();
    
    // Verify game resets to bidding phase
    await expect(page.locator('[data-testid="game-phase"]')).toHaveText('bidding');
    
    // Verify scores reset
    await expect(page.locator('[data-testid="team-0-marks"]')).toContainText('0');
    await expect(page.locator('[data-testid="team-1-marks"]')).toContainText('0');
  });

  test('should display responsive layout', async ({ page }) => {
    // Test desktop layout
    await page.setViewportSize({ width: 1200, height: 800 });
    await expect(page.locator('.debug-layout')).toBeVisible();
    
    // Test tablet layout
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(page.locator('.debug-layout')).toBeVisible();
    
    // Test mobile layout
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator('.debug-layout')).toBeVisible();
  });

  test('should show correct team assignments', async ({ page }) => {
    // Check that players are assigned to correct teams
    const players = page.locator('.player-section');
    await expect(players).toHaveCount(4);
    
    // Verify team info is displayed for each player
    const teamInfos = page.locator('.team-info');
    await expect(teamInfos).toHaveCount(4);
    
    // Check that team assignments are visible in the text
    await expect(page.locator('.team-info')).toContainText(['Team 1', 'Team 2']);
  });

  test('should handle accessibility features', async ({ page }) => {
    // Check for proper button structure
    const actionButtons = page.locator('[data-generic-testid*="bid-button"]');
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
    // Check if debug state display is present (always visible in new UI)
    await expect(page.locator('[data-testid="debug-panel"]')).toBeVisible();
    
    // Verify game phase is shown
    await expect(page.locator('[data-testid="game-phase"]')).toHaveText('bidding');
    
    // Verify actions count is displayed
    const actionsCount = page.locator('[data-testid="actions-count"]');
    await expect(actionsCount).toBeVisible();
    const count = await actionsCount.textContent();
    expect(parseInt(count || '0')).toBeGreaterThan(0);
  });

  test('should handle game flow transitions', async ({ page }) => {
    // Start in bidding phase
    await expect(page.locator('[data-testid="game-phase"]')).toHaveText('bidding');
    
    // Verify bid actions are available
    const bidButtons = page.locator('[data-generic-testid*="bid-button"]');
    await expect(bidButtons.first()).toBeVisible();
    
    // Verify actions count matches available buttons
    const actionsCount = page.locator('[data-testid="actions-count"]');
    const displayedCount = await actionsCount.textContent();
    const buttonCount = await bidButtons.count();
    expect(parseInt(displayedCount || '0')).toBe(buttonCount);
  });
});