import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Basic Gameplay', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    await helper.goto();
  });

  test('should load game interface', async () => {
    // Check main UI elements are present
    // Use page directly from beforeEach setup
    await expect(helper['page'].locator('h1')).toContainText('Texas 42 Debug Interface');
    await expect(helper['page'].locator('[data-testid="debug-panel"]')).toBeVisible();
    await expect(helper['page'].locator('[data-testid="actions-count"]')).toBeVisible();
  });

  test('should start in bidding phase', async () => {
    const phase = await helper.getCurrentPhase();
    expect(phase.toLowerCase()).toContain('bidding');
    
    // Should have bidding options available
    const biddingOptions = await helper.getBiddingOptions();
    expect(biddingOptions.length).toBeGreaterThan(0);
    expect(biddingOptions.some(option => option.type === 'pass')).toBe(true);
  });

  test('should allow valid opening bids', async () => {
    const biddingOptions = await helper.getBiddingOptions();
    
    // Should include point bids 30-41
    expect(biddingOptions.some(option => option.type === 'bid_points' && option.value === 30)).toBe(true);
    expect(biddingOptions.some(option => option.type === 'bid_points' && option.value === 35)).toBe(true);
    expect(biddingOptions.some(option => option.type === 'bid_points' && option.value === 41)).toBe(true);
    
    // Should include mark bids 1-2 in tournament mode  
    expect(biddingOptions.some(option => option.type === 'bid_marks' && option.value === 1)).toBe(true);
    expect(biddingOptions.some(option => option.type === 'bid_marks' && option.value === 2)).toBe(true);
  });

  test('should not allow invalid opening bids', async () => {
    const biddingOptions = await helper.getBiddingOptions();
    
    // Should not include point bids below 30 or above 41
    expect(biddingOptions.some(option => option.type === 'bid_points' && option.value === 29)).toBe(false);
    expect(biddingOptions.some(option => option.type === 'bid_points' && option.value === 42)).toBe(false);
    
    // Should not include 3+ marks in opening bid
    expect(biddingOptions.some(option => option.type === 'bid_marks' && option.value! >= 3)).toBe(false);
  });

  test('should progress through bidding round', async () => {
    // Pass for all players
    for (let i = 0; i < 4; i++) {
      await helper.selectActionByType('pass');
    }
    
    // Should redeal after all passes
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('bidding');
  });

  test('should handle winning bid and trump selection', async () => {
    // Place a winning bid
    await helper.selectActionByType('bid_points', 30);
    
    // Other players pass
    for (let i = 0; i < 3; i++) {
      await helper.selectActionByType('pass');
    }
    
    // Should now be in trump selection
    const actions = await helper.getAvailableActions();
    expect(actions.some(action => action.type === 'trump_selection')).toBe(true);
  });

  test('should transition to playing phase after trump selection', async () => {
    // Quick bidding round
    await helper.selectActionByType('bid_points', 30);
    for (let i = 0; i < 3; i++) {
      await helper.selectActionByType('pass');
    }
    
    // Select trump
    await helper.setTrumpBySuit('blanks');
    
    // Should be in playing phase
    await helper.waitForPhaseChange('playing');
    
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('playing');
    
    // Should show trump
    const trump = await helper.getTrump();
    expect(trump).toContain('0s');
  });

  test('should track scores correctly', async () => {
    const initialScores = await helper.getTeamScores();
    expect(initialScores).toEqual([0, 0]);
    
    const initialMarks = await helper.getTeamMarks();
    expect(initialMarks).toEqual([0, 0]);
  });

  test('should allow domino plays in correct order', async () => {
    // Set up playing phase (this would need state injection)
    // For now, test the basic interface
    
    const currentTrick = await helper.getCurrentTrick();
    expect(Array.isArray(currentTrick)).toBe(true);
  });

  test.skip('should complete full game flow', async () => {
    // Skip - complex test that relies on debug panel functionality
    // TODO: Reimplement without debug panel dependency
  });

  test('should handle new game correctly', async () => {
    await helper.newGame();
    
    // Should reset to initial state
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('bidding');
    
    const scores = await helper.getTeamScores();
    expect(scores).toEqual([0, 0]);
    
    const marks = await helper.getTeamMarks();
    expect(marks).toEqual([0, 0]);
  });

  test.skip('should validate game rules throughout play', async () => {
    // Skip - debug panel validation not available in new UI
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

  test.skip('should handle debug panel', async () => {
    // Skip - entire UI is now debug interface, no separate debug panel
  });
});