import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Basic Gameplay', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    await helper.goto();
  });

  test('should load game interface', async ({ page }) => {
    // Check main UI elements are present
    await expect(page.locator('h1')).toContainText('Texas 42 - mk5');
    await expect(page.locator('[data-testid="score-board"]')).toBeVisible();
    await expect(page.locator('[data-testid="player-hands"]')).toBeVisible();
  });

  test('should start in bidding phase', async ({ page }) => {
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('bidding');
    
    // Should have bidding options available
    const biddingOptions = await helper.getBiddingOptions();
    expect(biddingOptions.length).toBeGreaterThan(0);
    expect(biddingOptions).toContain('Pass');
  });

  test('should allow valid opening bids', async ({ page }) => {
    const biddingOptions = await helper.getBiddingOptions();
    
    // Should include point bids 30-41
    expect(biddingOptions.some(option => option.includes('30'))).toBe(true);
    expect(biddingOptions.some(option => option.includes('35'))).toBe(true);
    expect(biddingOptions.some(option => option.includes('41'))).toBe(true);
    
    // Should include mark bids 1-2 in tournament mode
    expect(biddingOptions.some(option => option.includes('1 mark'))).toBe(true);
    expect(biddingOptions.some(option => option.includes('2 mark'))).toBe(true);
  });

  test('should not allow invalid opening bids', async ({ page }) => {
    const biddingOptions = await helper.getBiddingOptions();
    
    // Should not include point bids below 30 or above 41
    expect(biddingOptions.some(option => option.includes('29'))).toBe(false);
    expect(biddingOptions.some(option => option.includes('42'))).toBe(false);
    
    // Should not include 3+ marks in opening bid
    expect(biddingOptions.some(option => option.includes('3 mark'))).toBe(false);
  });

  test('should progress through bidding round', async ({ page }) => {
    // Pass for all players
    for (let i = 0; i < 4; i++) {
      await helper.placeBid('Pass');
      await page.waitForTimeout(100);
    }
    
    // Should redeal after all passes
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('bidding');
  });

  test('should handle winning bid and trump selection', async ({ page }) => {
    // Place a winning bid
    await helper.placeBid('Bid 30 points');
    
    // Other players pass
    for (let i = 0; i < 3; i++) {
      await helper.placeBid('Pass');
      await page.waitForTimeout(100);
    }
    
    // Should now be in trump selection
    await page.waitForTimeout(500);
    
    // Should have trump options
    const actions = await helper.getAvailableActions();
    expect(actions.some(action => action.includes('trump'))).toBe(true);
  });

  test('should transition to playing phase after trump selection', async ({ page }) => {
    // Quick bidding round
    await helper.placeBid('Bid 30 points');
    for (let i = 0; i < 3; i++) {
      await helper.placeBid('Pass');
      await page.waitForTimeout(100);
    }
    
    // Select trump
    await helper.selectTrump('Declare Blanks trump');
    await page.waitForTimeout(200);
    
    // Should be in playing phase
    await helper.waitForPhaseChange('playing');
    
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('playing');
    
    // Should show trump
    const trump = await helper.getTrump();
    expect(trump).toContain('Blanks');
  });

  test('should track scores correctly', async ({ page }) => {
    const initialScores = await helper.getTeamScores();
    expect(initialScores).toEqual([0, 0]);
    
    const initialMarks = await helper.getTeamMarks();
    expect(initialMarks).toEqual([0, 0]);
  });

  test('should allow domino plays in correct order', async ({ page }) => {
    // Set up playing phase (this would need state injection)
    // For now, test the basic interface
    
    const currentTrick = await helper.getCurrentTrick();
    expect(Array.isArray(currentTrick)).toBe(true);
  });

  test('should complete full game flow', async ({ page }) => {
    test.setTimeout(30000); // Longer timeout for full game
    
    try {
      await helper.performCompleteGame('random');
      
      // Game should eventually complete
      const finalPhase = await helper.getCurrentPhase();
      expect(finalPhase).toContain('game_end');
      
    } catch (error) {
      // Generate bug report if game gets stuck
      const report = await helper.generateBugReport(
        'Full game flow test failed: ' + error
      );
      console.log('Bug report generated:', report);
      throw error;
    }
  });

  test('should handle new game correctly', async ({ page }) => {
    await helper.newGame();
    
    // Should reset to initial state
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('bidding');
    
    const scores = await helper.getTeamScores();
    expect(scores).toEqual([0, 0]);
    
    const marks = await helper.getTeamMarks();
    expect(marks).toEqual([0, 0]);
  });

  test('should validate game rules throughout play', async ({ page }) => {
    // Check initial state validation
    let errors = await helper.validateGameRules();
    expect(errors).toHaveLength(0);
    
    // Make a few moves and validate
    await helper.placeBid('Bid 30 points');
    
    errors = await helper.validateGameRules();
    expect(errors).toHaveLength(0);
  });

  test('should be responsive on mobile', async ({ page }) => {
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    // Should still be functional
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('[data-testid="score-board"]')).toBeVisible();
    
    // Should be able to interact with bidding
    const biddingOptions = await helper.getBiddingOptions();
    expect(biddingOptions.length).toBeGreaterThan(0);
  });

  test('should handle debug panel', async ({ page }) => {
    // Open debug panel
    await helper.openDebugPanel();
    
    // Should show game state
    await expect(page.locator('[data-testid="debug-panel"]')).toBeVisible();
    await expect(page.locator('[data-testid="state-display"]')).toBeVisible();
    
    // Close debug panel
    await helper.closeDebugPanel();
    await expect(page.locator('[data-testid="debug-panel"]')).not.toBeVisible();
  });
});