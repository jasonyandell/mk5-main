import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe('AI Controller After Navigation', () => {
  test('AI should play after browser back button', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Start game with URL updates enabled
    await helper.goto(12345, { disableUrlUpdates: false });
    
    // Enable AI for other players
    await helper.enableAIForOtherPlayers();
    
    // Player 0 bids 30
    await helper.bid(30, false);
    
    // Wait for AI to make their moves (passes)
    await helper.waitForAIMove();
    
    // Should be in trump selection (player 0 won)
    let phase = await helper.getCurrentPhase();
    expect(phase).toBe('trump_selection');
    
    // Select trump
    await helper.setTrump('blanks');
    
    // Now in playing phase - player 1 should play
    phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');
    const player1 = await helper.getCurrentPlayer();
    expect(player1).toBe('P1'); // Left of dealer (P0)
    
    // Store current URL
    
    // Go back to trump selection
    await page.goBack();
    await helper.waitForGameReady();
    
    // Should be back in trump selection
    phase = await helper.getCurrentPhase();
    expect(phase).toBe('trump_selection');
    
    // Go forward to playing phase
    await page.goForward();
    await helper.waitForGameReady();
    
    // Should be back in playing phase
    phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');
    
    // CRITICAL: AI (P1) should now play automatically
    // Wait for AI to take action
    await helper.waitForAIMove();
    
    // Check that a domino was played (trick should have at least 1 domino)
    const trick = await helper.getCurrentTrick();
    expect(trick.length).toBeGreaterThan(0);
    
    // Verify we're not stuck on "P1 is thinking..."
    const turnPlayer = await helper.getCurrentPlayer();
    expect(turnPlayer).not.toBe('P1'); // Should have moved to next player
  });

  test('AI should play after loading from URL', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Create a state where it's AI's turn
    // P0 bids, others pass, P0 selects trump -> P1's turn to play
    await helper.loadStateWithActions(12345, ['30', 'p', 'p', 'p', 't0']);
    
    // Enable AI for other players
    await helper.enableAIForOtherPlayers();
    
    // Should be P1's turn (left of dealer P0)
    const phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');
    
    // Wait for AI to play
    await helper.waitForAIMove();
    
    // Check that AI played
    const trick = await helper.getCurrentTrick();
    expect(trick.length).toBeGreaterThan(0);
  });

  test('AI should continue playing after popstate event', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Start with URL updates enabled
    await helper.goto(12345, { disableUrlUpdates: false });
    
    // Enable AI
    await helper.enableAIForOtherPlayers();
    
    // Make some moves to create history
    await helper.bid(30, false);
    await helper.waitForAIMove(); // Let AI pass
    
    // Store URL after bidding
    
    // Continue to trump selection
    const phase1 = await helper.getCurrentPhase();
    expect(phase1).toBe('trump_selection');
    
    await helper.setTrump('blanks');
    
    // Now use browser history API directly to trigger popstate
    await page.evaluate(() => {
      window.history.back();
    });
    
    // Wait for popstate to be handled
    await helper.waitForNavigationRestore();
    
    // Should be back at trump selection
    const phase2 = await helper.getCurrentPhase();
    expect(phase2).toBe('trump_selection');
    
    // Go forward again
    await page.evaluate(() => {
      window.history.forward();
    });
    
    await helper.waitForNavigationRestore();
    
    // Should be in playing phase with AI taking action
    const phase3 = await helper.getCurrentPhase();
    expect(phase3).toBe('playing');
    
    // Wait for AI to play
    await helper.waitForAIMove();
    
    // Verify AI played
    const trick = await helper.getCurrentTrick();
    expect(trick.length).toBeGreaterThan(0);
  });
});