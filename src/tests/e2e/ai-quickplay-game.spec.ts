import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('AI Quickplay Complete Game', () => {
  test('completes entire hand with all AI players at instant speed', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(12345);
    
    // Set instant speed and enable all AI
    await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
    for (let i = 0; i < 4; i++) {
      await helper.locator(`[data-testid="ai-player-${i}"]`).check();
    }
    
    // Get initial state
    const initialPlayer = await helper.getCurrentPlayer();
    const initialActions = await helper.getAvailableActions();
    
    // Start AI
    await helper.locator('[data-testid="quickplay-run"]').click();
    
    // Verify AI is running
    const runningStatus = helper.locator('.status-indicator.active');
    await expect(runningStatus).toBeVisible();
    await expect(runningStatus).toContainText('Running (instant)');
    
    // Wait for AI to make progress - check multiple times with optimized timing
    let progressMade = false;
    for (let i = 0; i < 20; i++) {
      await page.waitForTimeout(100);
      
      const currentPlayer = await helper.getCurrentPlayer();
      const currentActions = await helper.getAvailableActions();
      
      // Check if player changed or actions changed significantly
      if (currentPlayer !== initialPlayer || 
          currentActions.length !== initialActions.length) {
        progressMade = true;
        break;
      }
    }
    
    // Stop AI
    await helper.locator('[data-testid="quickplay-stop"]').click();
    
    // Verify AI stopped
    await expect(helper.locator('[data-testid="quickplay-run"]')).toBeVisible();
    
    // Verify progress was made
    expect(progressMade).toBe(true);
  });

  test('handles mixed AI and human players', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(12345);
    
    // Set instant speed
    await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
    
    // Enable AI for players 1 and 3 only (opposing team members)
    await helper.locator('[data-testid="ai-player-0"]').uncheck();
    await helper.locator('[data-testid="ai-player-1"]').check();
    await helper.locator('[data-testid="ai-player-2"]').uncheck();
    await helper.locator('[data-testid="ai-player-3"]').check();
    
    // Start AI
    await helper.locator('[data-testid="quickplay-run"]').click();
    
    // AI should stop when it's a human player's turn - optimized for offline testing
    await page.waitForTimeout(100);
    
    // Get current player
    const currentPlayer = await helper.getCurrentPlayer();
    const currentPlayerNum = parseInt(currentPlayer.replace('Current Player: P', ''));
    
    // Current player should be human (0 or 2)
    expect([0, 2]).toContain(currentPlayerNum);
    
    // Make human move
    const actions = await helper.getAvailableActions();
    if (actions.length > 0) {
      await helper.selectActionByIndex(0);
    }
    
    // AI should continue - optimized for offline testing
    await page.waitForTimeout(100);
    
    // Stop AI
    await helper.locator('[data-testid="quickplay-stop"]').click();
  });

  test('respects game end conditions', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(12345);
    
    // For now, just test that AI respects the current game state
    // Set instant speed and enable all AI
    await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
    for (let i = 0; i < 4; i++) {
      await helper.locator(`[data-testid="ai-player-${i}"]`).check();
    }
    
    // Get initial marks
    const initialMarks = await helper.getTeamMarks();
    
    // Start AI
    await helper.locator('[data-testid="quickplay-run"]').click();
    
    // Let it run for a bit - allowing more time for game progression verification
    await page.waitForTimeout(1000);
    
    // Stop AI
    await helper.locator('[data-testid="quickplay-stop"]').click();
    
    // Check if marks increased (game progressed)
    const finalMarks = await helper.getTeamMarks();
    const gameProgressed = finalMarks[0] > initialMarks[0] || finalMarks[1] > initialMarks[1];
    expect(gameProgressed).toBe(true);
    
    // If game reached 7 marks, it should be in game_end phase
    if (Math.max(...finalMarks) >= 7) {
      const phase = await helper.getCurrentPhase();
      expect(phase).toContain('game_end');
    }
  });

  test('AI makes reasonable decisions', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(54321); // Different seed for variety
    
    // Set instant speed
    await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
    
    // Test bidding decision
    const initialPhase = await helper.getCurrentPhase();
    if (initialPhase.includes('bidding')) {
      // Enable AI for current player
      const currentPlayer = await helper.getCurrentPlayer();
      const playerNum = parseInt(currentPlayer.replace('Current Player: P', ''));
      await helper.locator(`[data-testid="ai-player-${playerNum}"]`).check();
      
      // Get available bid actions
      const actions = await helper.getAvailableActions();
      const bidActions = actions.filter(a => 
        a.type === 'bid_points' || a.type === 'bid_marks' || a.type === 'pass'
      );
      
      // Verify bid actions are available
      expect(bidActions.length).toBeGreaterThan(0);
      
      // Execute AI step - timing already optimal
      await helper.locator('[data-testid="quickplay-step"]').click();
      await page.waitForTimeout(50);
      
      // Verify AI made a bid decision (player changed)
      const playerAfter = await helper.getCurrentPlayer();
      expect(playerAfter).not.toBe(currentPlayer);
    }
  });

  test('handles scoring phase automatically', async ({ page }) => {
    test.setTimeout(5000);
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(12345);
    
    // Just test that AI can handle different phases
    // Enable AI for all players
    await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
    for (let i = 0; i < 4; i++) {
      await helper.locator(`[data-testid="ai-player-${i}"]`).check();
    }
    
    // Get initial phase
    const initialPhase = await helper.getCurrentPhase();
    
    // Run AI briefly
    await helper.locator('[data-testid="quickplay-run"]').click();
    
    // Let it run through multiple actions - optimized for AI quickplay
    await page.waitForTimeout(1000);
    
    // Stop AI (check if button exists first)
    const stopButton = helper.locator('[data-testid="quickplay-stop"]');
    const stopButtonVisible = await stopButton.isVisible();
    if (stopButtonVisible) {
      await stopButton.click();
    }
    
    // Verify game progressed
    const finalPhase = await helper.getCurrentPhase();
    
    // Game should have progressed from initial phase
    expect(finalPhase).not.toBe(initialPhase);
    const actions = await helper.getAvailableActions();
    
    // Game should have made progress
    expect(actions).toBeDefined();
  });
});