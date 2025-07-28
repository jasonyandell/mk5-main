import { test, expect } from '@playwright/test';

test.describe('Complete Game End-to-End', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load game interface correctly', async ({ page }) => {
    // Verify page loads
    await expect(page.locator('h1')).toContainText('Texas 42 - mk5');
    
    // Verify game phase is displayed
    await expect(page.locator('[data-testid="game-phase"]')).toHaveText('bidding');
    
    // Verify all 4 players are shown
    const players = page.locator('.player-section');
    await expect(players).toHaveCount(4);
    
    // Verify scoreboard is present
    await expect(page.locator('.scoreboard')).toBeVisible();
    
    // Verify bidding panel is present
    await expect(page.locator('.bidding-panel')).toBeVisible();
  });

  test('should handle basic bidding flow', async ({ page }) => {
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    
    // Check if bid buttons are available
    const bidButtons = page.locator('.bid-btn');
    const buttonCount = await bidButtons.count();
    
    if (buttonCount > 0) {
      // Click the first available bid button
      await bidButtons.first().click();
      
      // Verify the bid was registered
      await page.waitForTimeout(100); // Small delay for state update
      
      // Check if bid history updated
      const bidHistory = page.locator('.bids-list');
      await expect(bidHistory).not.toContainText('No bids yet');
    }
  });

  test('should display game controls correctly', async ({ page }) => {
    // Verify reset button is present
    await expect(page.locator('.reset-btn')).toBeVisible();
    await expect(page.locator('.reset-btn')).toContainText('New Game');
    
    // Verify debug toggle is present
    await expect(page.locator('[data-testid="debug-toggle"]')).toBeVisible();
  });

  test('should handle debug panel toggle', async ({ page }) => {
    // Click debug toggle
    await page.locator('[data-testid="debug-toggle"]').click();
    
    // Verify debug panel opens
    await expect(page.locator('.debug-overlay')).toBeVisible();
    await expect(page.locator('.debug-panel')).toBeVisible();
    
    // Verify debug panel contains expected content
    await expect(page.locator('.debug-panel')).toContainText('Debug Panel');
    
    // Close debug panel
    await page.locator('.close-btn').click();
    
    // Verify debug panel closes
    await expect(page.locator('.debug-overlay')).not.toBeVisible();
  });

  test('should reset game correctly', async ({ page }) => {
    // Click reset button
    await page.locator('.reset-btn').click();
    
    // Verify game resets to bidding phase
    await expect(page.locator('[data-testid="game-phase"]')).toHaveText('bidding');
    
    // Verify scores reset
    const teamScores = page.locator('.score-item.marks .value');
    await expect(teamScores.first()).toContainText('0');
  });

  test('should display responsive layout', async ({ page }) => {
    // Test desktop layout
    await page.setViewportSize({ width: 1200, height: 800 });
    await expect(page.locator('.game-layout')).toBeVisible();
    
    // Test tablet layout
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(page.locator('.game-layout')).toBeVisible();
    
    // Test mobile layout
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator('.game-layout')).toBeVisible();
  });

  test('should show correct team assignments', async ({ page }) => {
    // Check that players are assigned to correct teams
    const teamBadges = page.locator('.team-badge');
    await expect(teamBadges).toHaveCount(4);
    
    // Verify team colors/identifiers are present
    await expect(page.locator('.team-badge.team-0')).toHaveCount(2);
    await expect(page.locator('.team-badge.team-1')).toHaveCount(2);
  });

  test('should handle accessibility features', async ({ page }) => {
    // Check for screen reader content
    await expect(page.locator('.sr-only')).toHaveCount(0); // We don't have sr-only content yet
    
    // Check for proper ARIA labels (if any)
    const buttons = page.locator('button');
    const buttonCount = await buttons.count();
    expect(buttonCount).toBeGreaterThan(0);
    
    // Verify buttons are keyboard accessible
    await page.keyboard.press('Tab');
    const focusedElement = page.locator(':focus');
    await expect(focusedElement).toHaveCount(1);
  });

  test('should maintain game state consistency', async ({ page }) => {
    // Open debug panel to inspect state
    await page.locator('[data-testid="debug-toggle"]').click();
    
    // Check if state display is present
    await expect(page.locator('.state-summary')).toBeVisible();
    
    // Verify initial state values
    const stateItems = page.locator('.summary-item');
    await expect(stateItems).toHaveCount(8); // Expected number of state summary items
    
    // Close debug panel
    await page.locator('.close-btn').click();
  });

  test('should handle game flow transitions', async ({ page }) => {
    // Start in bidding phase
    await expect(page.locator('[data-testid="game-phase"]')).toHaveText('bidding');
    
    // Verify bidding panel is visible
    await expect(page.locator('.bidding-panel')).toBeVisible();
    
    // Note: Full game flow testing would require more complex automation
    // since it depends on the specific game state and available actions
  });
});