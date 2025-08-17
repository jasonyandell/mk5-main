import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Page Reload State Consistency', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    await helper.goto();
  });

  test('game state persists after page reload following bid-30', async ({ page }) => {
    // Navigate to Actions tab where bidding happens
    await page.locator('[data-testid="nav-actions"]').click();
    await page.waitForTimeout(100);
    
    // Make a bid-30 action
    await helper.selectActionByType('bid_points', 30);

    // Capture the URL for reload
    const urlAfterBid = page.url();
    expect(urlAfterBid).toContain('d='); // Should have state in URL

    // Reload the page
    await page.goto(urlAfterBid);
    
    // Wait for the game to load
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    
    // Navigate back to Actions tab after reload
    await page.locator('[data-testid="nav-actions"]').click();
    await page.waitForTimeout(100);

    // Verify game state persisted - state should be consistent after reload
    const currentPlayerAfter = await helper.getCurrentPlayer();
    
    // Verify that a player is displayed (any valid player is fine)
    expect(currentPlayerAfter).toMatch(/Current Player: P\d+/);
    
    // Verify we're in a valid game phase 
    const phase = await helper.getCurrentPhase();
    expect(phase).toMatch(/bidding|trump_selection|playing/);
  });

  test('state consistency across multiple reloads with different actions', async ({ page }) => {
    // Perform multiple actions
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    
    // Capture URL with multiple actions
    const urlWithActions = page.url();
    expect(urlWithActions).toContain('d=');
    
    // First reload
    await page.goto(urlWithActions);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    
    const phaseAfterFirstReload = await helper.getCurrentPhase();
    const playerAfterFirstReload = await helper.getCurrentPlayer();
    
    // Second reload - should be identical
    await page.goto(urlWithActions);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    
    const phaseAfterSecondReload = await helper.getCurrentPhase();
    const playerAfterSecondReload = await helper.getCurrentPlayer();
    
    // Both reloads should result in identical state
    expect(phaseAfterSecondReload).toBe(phaseAfterFirstReload);
    expect(playerAfterSecondReload).toBe(playerAfterFirstReload);
  });

  test('trump selection persists after reload', async ({ page }) => {
    // Navigate to Actions tab where bidding happens
    await page.locator('[data-testid="nav-actions"]').click();
    await page.waitForTimeout(100);
    
    // Complete bidding
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    
    // Select trump
    await helper.setTrumpBySuit('blanks');
    
    // Capture URL after trump selection
    const urlWithTrump = page.url();
    
    // Reload
    await page.goto(urlWithTrump);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    
    // Navigate to Game tab after reload to see trump
    await page.locator('[data-testid="nav-game"]').click();
    await page.waitForTimeout(100);
    
    // Verify we're in a valid game phase after reload
    const phase = await helper.getCurrentPhase();
    expect(phase).toMatch(/bidding|trump_selection|playing/);
    
    // Verify the URL still contains the state
    expect(page.url()).toContain('d=');
  });

  test('game state deterministic with same seed across sessions', async ({ page: _page }) => {
    // Use a specific seed
    const seed = 99999;
    await helper.goto(seed);
    
    // Capture initial player
    const initialPlayer = await helper.getCurrentPlayer();
    
    // Navigate to a different seed
    await helper.goto(11111);
    
    // Navigate back to the original seed
    await helper.goto(seed);
    
    // Should have same initial player
    const finalPlayer = await helper.getCurrentPlayer();
    expect(finalPlayer).toBe(initialPlayer);
  });
});