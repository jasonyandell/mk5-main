import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe.skip('AI Controller After Navigation', () => {
  test('AI should play after browser back button', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Load game state with AI players and P0 winning bid
    // This approach is more reliable than trying to manually bid
    await helper.loadStateWithActions(12345, 
      ['bid-30', 'pass', 'pass', 'pass', 'trump-blanks'], 
      ['human', 'ai', 'ai', 'ai']
    );
    
    // Should be in playing phase
    let phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');
    
    // Store current URL
    const urlBefore = await helper.getCurrentURL();
    
    // Enable AI for other players to ensure they play
    await helper.enableAIForOtherPlayers();
    
    // Play one domino as P0
    await helper.playAnyDomino();
    
    // AI should execute synchronously in test mode after human action
    // Verify trick is complete
    let trickLength = await page.evaluate(() => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const state = (window as any).getGameState();
      return state.currentTrick.length;
    });
    expect(trickLength).toBe(4); // Should be complete immediately in test mode
    
    // Store URL after playing (includes all actions)
    const urlAfter = await helper.getCurrentURL();
    
    // Go back to before we played
    await page.goto(urlBefore);
    await helper.waitForGameReady();
    
    // Should be back at start of playing phase
    phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');
    
    // Check trick is empty
    trickLength = await page.evaluate(() => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const state = (window as any).getGameState();
      return state.currentTrick.length;
    });
    expect(trickLength).toBe(0);
    
    // Go forward to after playing
    await page.goto(urlAfter);
    await helper.waitForGameReady();
    
    // Should have the completed trick
    trickLength = await page.evaluate(() => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const state = (window as any).getGameState();
      return state.currentTrick.length;
    });
    expect(trickLength).toBe(4); // P0, P1, P2, P3 all played
  });

  test('AI should play after loading from URL', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Load state where P1 won bid with AI players
    // P0 passes, P1 bids 30, P2/P3 pass, P1 selects trump
    // Set players 1-3 as AI from the start
    await helper.loadStateWithActions(12345, ['pass', 'bid-30', 'pass', 'pass', 'trump-blanks'], ['human', 'ai', 'ai', 'ai']);
    
    // Should be in playing phase (AI executes synchronously in test mode)
    const phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');
    
    // AI should have already played (P1, P2, P3 all play automatically)
    const trickLength = await page.evaluate(() => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const state = (window as any).getGameState();
      return state.currentTrick.length;
    });
    expect(trickLength).toBe(3); // P1 leads (bid winner), then P2, P3 play, waiting for P0
  });

  test('AI should continue playing after popstate event', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Load state at trump selection with AI players
    await helper.loadStateWithActions(12345, 
      ['bid-30', 'pass', 'pass', 'pass'], 
      ['human', 'ai', 'ai', 'ai']
    );
    
    // Should be at trump selection
    const phase1 = await helper.getCurrentPhase();
    expect(phase1).toBe('trump_selection');
    
    // Select trump
    await helper.setTrump('blanks');
    
    // Should be in playing phase
    const phase2 = await helper.getCurrentPhase();
    expect(phase2).toBe('playing');
    
    // Now use browser history API directly to trigger popstate
    await page.evaluate(() => {
      window.history.back();
    });
    
    // Wait for popstate to be handled
    await helper.waitForNavigationRestore();
    
    // Should be back at trump selection
    const phase3 = await helper.getCurrentPhase();
    expect(phase3).toBe('trump_selection');
    
    // Go forward again
    await page.evaluate(() => {
      window.history.forward();
    });
    
    await helper.waitForNavigationRestore();
    
    // Should be in playing phase with P0 (bid winner) to play
    const phase4 = await helper.getCurrentPhase();
    expect(phase4).toBe('playing');
    
    // Enable AI for other players
    await helper.enableAIForOtherPlayers();
    
    // P0 (human) plays first since they won the bid
    await helper.playAnyDomino();
    
    // AI should execute synchronously in test mode
    // Verify all AI players played (trick should have 4 dominoes total)
    const trickLength = await page.evaluate(() => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const state = (window as any).getGameState();
      return state.currentTrick.length;
    });
    expect(trickLength).toBe(4); // P0 + P1, P2, P3 all played
  });
});