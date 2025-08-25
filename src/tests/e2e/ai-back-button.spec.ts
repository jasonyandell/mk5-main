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
    
    // Wait for all 3 AI players to pass
    await helper.waitForAIMove(); // P1 passes
    await helper.waitForAIMove(); // P2 passes
    await helper.waitForAIMove(); // P3 passes
    
    // Should be in trump selection (player 0 won)
    let phase = await helper.getCurrentPhase();
    expect(phase).toBe('trump_selection');
    
    // Select trump
    await helper.setTrump('blanks');
    
    // Now in playing phase - player 0 should play (bid winner leads)
    phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');
    
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
    
    // P0 (human) needs to play first since bid winner leads
    await helper.playAnyDomino();
    
    // Now all AI players should play automatically
    await helper.waitForAIMove();
    
    // Check that all AI played (trick should have 4 dominoes total)
    const trickLength = await page.evaluate(() => {
      const state = (window as any).getGameState();
      return state.currentTrick.length;
    });
    expect(trickLength).toBe(4); // P0, P1, P2, P3 all played
  });

  test('AI should play after loading from URL', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Load state where P1 won bid - when we enable AI, they will play automatically
    // P0 passes, P1 bids 30, P2/P3 pass, P1 selects trump
    await helper.loadStateWithActions(12345, ['p', '30', 'p', 'p', 't0']);
    
    // Enable AI for other players - this triggers AI to play immediately
    await helper.enableAIForOtherPlayers();
    
    // Should be in playing phase
    const phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');
    
    // AI should have already played (P1, P2, P3 all play automatically)
    const trickLength = await page.evaluate(() => {
      const state = (window as any).getGameState();
      return state.currentTrick.length;
    });
    expect(trickLength).toBe(3); // P1 leads (bid winner), then P2, P3 play, waiting for P0
  });

  test('AI should continue playing after popstate event', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Start with URL updates enabled
    await helper.goto(12345, { disableUrlUpdates: false });
    
    // Enable AI
    await helper.enableAIForOtherPlayers();
    
    // Make some moves to create history
    await helper.bid(30, false);
    
    // Wait for all 3 AI players to respond
    await helper.waitForAIMove(); // P1 passes
    await helper.waitForAIMove(); // P2 passes
    await helper.waitForAIMove(); // P3 passes
    
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
    
    // Should be in playing phase with P0 (bid winner) to play
    const phase3 = await helper.getCurrentPhase();
    expect(phase3).toBe('playing');
    
    // P0 (human) plays first since they won the bid
    await helper.playAnyDomino();
    
    // Now AI players should all play automatically
    await helper.waitForAIMove();
    
    // Verify all AI players played (trick should have 4 dominoes total)
    const trickLength2 = await page.evaluate(() => {
      const state = (window as any).getGameState();
      return state.currentTrick.length;
    });
    expect(trickLength2).toBe(4); // P0 + P1, P2, P3 all played
  });
});