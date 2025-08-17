import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Complete Game End-to-End', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    await helper.goto();
  });

  test('should load game interface correctly', async ({ page }) => {
    // Verify page loads
    await page.waitForSelector('.app-container', { timeout: 5000 });
    
    // Verify game phase is displayed
    await expect(helper.getPhaseNameLocator()).toContainText('Bidding');
    
    // Verify navigation is present
    await expect(page.locator('nav')).toBeVisible();
    
    // Verify header is present
    await expect(helper.getAppHeaderLocator()).toBeVisible();
    
    // Verify main content area is present
    await expect(helper.getGameContainerLocator()).toBeVisible();
  });

  test('should handle basic bidding flow', async ({ page }) => {
    // Wait for page to load
    await page.waitForSelector('.app-container', { timeout: 5000 });
    
    // Navigate to actions panel
    await helper.getNavLocator('actions').click();
    await page.waitForTimeout(200);
    
    // Check if bid buttons are available
    const bidButtons = helper.getBidButtonsLocator();
    const passButton = helper.getPassButtonLocator();
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
    const phaseElement = helper.getPhaseNameLocator();
    const phaseText = await phaseElement.textContent();
    expect(['Bidding', 'Trump Selection']).toContain(phaseText);
  });

  test('should display game controls correctly', async ({ page: _page }) => {
    // Verify navigation controls are present
    await expect(helper.getNavLocator('game')).toBeVisible();
    await expect(helper.getNavLocator('actions')).toBeVisible();
    await expect(helper.getNavLocator('debug')).toBeVisible();
  });

  test('should reset game correctly', async ({ page }) => {
    // Reload page to reset game
    await page.reload();
    await page.waitForSelector('.app-container', { timeout: 5000 });
    
    // Verify game resets to bidding phase
    await expect(helper.getPhaseNameLocator()).toContainText('Bidding');
    
    // Verify scores reset
    await expect(helper.getScoreValueLocator('us')).toContainText('0');
    await expect(helper.getScoreValueLocator('them')).toContainText('0');
  });

  test('should display responsive layout', async ({ page }) => {
    // Test desktop layout
    await page.setViewportSize({ width: 1200, height: 800 });
    await expect(helper.getAppContainerLocator()).toBeVisible();
    
    // Test tablet layout
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(helper.getAppContainerLocator()).toBeVisible();
    
    // Test mobile layout
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(helper.getAppContainerLocator()).toBeVisible();
  });

  test('should show correct team assignments', async ({ page }) => {
    // Check that team scores are displayed in header
    await expect(helper.getScoreCardLocator('us')).toBeVisible();
    await expect(helper.getScoreCardLocator('them')).toBeVisible();
    
    // Navigate to game view to see playing area
    await helper.getNavLocator('game').click();
    await page.waitForTimeout(200);
    
    // Check that playing area is displayed
    await expect(helper.getPlayingAreaLocator()).toBeVisible();
  });

  test('should handle accessibility features', async ({ page }) => {
    // Navigate to actions panel
    await helper.getNavLocator('actions').click();
    await page.waitForTimeout(200);
    
    // Check for proper button structure
    const actionButtons = helper.getActionButtonsLocator();
    const buttonCount = await actionButtons.count();
    expect(buttonCount).toBeGreaterThan(0);
    
    // Verify buttons are keyboard accessible
    await page.keyboard.press('Tab');
    const focusedElement = helper.getFocusedElementLocator();
    await expect(focusedElement).toHaveCount(1);
    
    // Verify buttons have title attributes for tooltips
    const firstButton = actionButtons.first();
    await expect(firstButton).toHaveAttribute('title');
  });

  test('should maintain game state consistency', async ({ page }) => {
    // Verify game phase is shown in header
    await expect(helper.getPhaseNameLocator()).toContainText('Bidding');
    
    // Navigate to debug panel to see game state
    await helper.getNavLocator('debug').click();
    await page.waitForTimeout(200);
    
    // Check that debug panel is visible
    await expect(helper.getDebugPanelLocator()).toBeVisible();
    
    // Close debug panel first to avoid backdrop issues
    await helper.getCloseButtonLocator().click();
    await page.waitForTimeout(200);
    
    // Verify actions are available by navigating to actions panel
    await helper.getNavLocator('actions').click();
    await page.waitForTimeout(200);
    
    const actionButtons = helper.getActionButtonsLocator();
    const count = await actionButtons.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should handle game flow transitions', async ({ page }) => {
    // Start in bidding phase
    await expect(helper.getPhaseNameLocator()).toContainText('Bidding');
    
    // Navigate to actions panel
    await helper.getNavLocator('actions').click();
    await page.waitForTimeout(200);
    
    // Verify bid actions are available
    const bidButtons = helper.getBidButtonsLocator();
    await expect(bidButtons.first()).toBeVisible();
    
    // Verify that actions are available
    const buttonCount = await bidButtons.count() + await helper.getPassButtonLocator().count();
    expect(buttonCount).toBeGreaterThan(0);
  });
});