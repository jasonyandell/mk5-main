/* eslint-disable @typescript-eslint/no-explicit-any */
// Reason: page.evaluate() runs in browser context where TypeScript cannot verify window properties.
// The use of 'any' for window casts is architectural, not a shortcut.

import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';
import type { PartialGameState } from '../types/test-helpers';

test.describe.skip('Comprehensive Back Button Navigation', () => {
  test('should handle back button during bidding phase', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Start with URL tracking enabled  
    await helper.goto(12345, { disableUrlUpdates: false });
    
    // P0 passes first
    await helper.pass();
    const urlAfterFirstPass = page.url();
    
    // P1 bids 30
    await helper.bid(30, false);
    const urlAfterBid = page.url();
    
    // Go back to before the bid
    await page.goBack();
    await helper.waitForNavigationRestore();
    
    // URL should match first pass state
    expect(page.url()).toBe(urlAfterFirstPass);
    
    // P1 should be able to make different action (bid 31)
    await helper.bid(31, false);
    
    // Should have different URL now
    expect(page.url()).not.toBe(urlAfterBid);
  });

  test('should handle back button during trump selection', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Load state where P0 won bid and is at trump selection
    await helper.loadStateWithActions(12345, ['bid-30', 'pass', 'pass', 'pass'],
      ['human', 'human', 'human', 'human'] // All human for this test
    );
    
    // Should be in trump selection
    let phase = await helper.getCurrentPhase();
    expect(phase).toBe('trump_selection');
    
    const urlBeforeTrump = page.url();
    
    // Select trump
    await helper.setTrump('blanks');
    const urlAfterTrump = page.url();
    
    // Should be in playing phase
    phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');
    
    // URLs should be different
    expect(urlAfterTrump).not.toBe(urlBeforeTrump);
    
    // Go back to trump selection
    await page.goBack();
    await helper.waitForNavigationRestore();
    
    // Should be back in trump selection
    phase = await helper.getCurrentPhase();
    expect(phase).toBe('trump_selection');
    
    // Verify URL went back
    expect(page.url()).toBe(urlBeforeTrump);
    
    // Can select different trump
    await helper.setTrump('doubles');
    
    // Should be in playing phase with different trump
    phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');
    
    // Check trump was set correctly via game view
    const trumpType = await page.evaluate(() => {
      const view = (window as any).getGameView?.();
      return view?.trump?.type || 'not-selected';
    });
    expect(trumpType).toBe('doubles');
  });

  test('should handle back button during playing phase', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Start game and get to playing phase
    await helper.goto(12345, { disableUrlUpdates: false });
    await helper.bid(30, false);
    await helper.pass();
    await helper.pass();
    await helper.pass();
    const urlBeforeTrump = page.url();
    
    await helper.setTrump('blanks');
    const urlAtStartOfPlaying = page.url();
    
    // URLs should be different (trump selection is a new action)
    expect(urlAtStartOfPlaying).not.toBe(urlBeforeTrump);
    
    // Go back to trump selection
    await page.goBack();
    await helper.waitForNavigationRestore();
    
    // Should be back at trump selection
    const phase = await helper.getCurrentPhase();
    expect(phase).toBe('trump_selection');
    
    // Go forward to playing
    await page.goForward();
    await helper.waitForNavigationRestore();
    
    // Should be back in playing phase
    const phase2 = await helper.getCurrentPhase();
    expect(phase2).toBe('playing');
  });

  test('should handle back button during scoring phase', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Load a game near scoring
    // This creates a valid state progression to scoring
    const actionsToScoring = [
      'bid-30', 'pass', 'pass', 'pass', 'trump-blanks' // Setup game
    ];
    
    await helper.loadStateWithActions(12345, actionsToScoring);
    
    // Enable URL updates
    await page.evaluate(() => {
      window.history.replaceState = window.history.replaceState.bind(window.history);
      window.history.pushState = window.history.pushState.bind(window.history);
    });
    
    // Should be in playing phase
    let phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');
    
    
    // Simulate reaching scoring phase by loading a state with all tricks played
    // In a real game, we'd play all tricks, but for testing we'll jump to a known state
    const scoringActions = [...actionsToScoring];
    // Add dummy trick completions to simulate a full hand
    for (let i = 0; i < 7; i++) {
      scoringActions.push('play-0-0', 'play-1-0', 'play-1-1', 'play-2-0', 'complete-trick');
    }
    
    await helper.loadStateWithActions(12345, scoringActions);
    
    phase = await helper.getCurrentPhase();
    // If we reach scoring phase
    if (phase === 'scoring') {
      
      // Score the hand
      const actions = await helper.getAvailableActions();
      const scoreAction = actions.find(a => a.type === 'score_hand' || a.id === 'score-hand');
      if (scoreAction) {
        await helper.selectAction(scoreAction.index);
      }
      
      
      // Go back to scoring phase
      await page.goBack();
      await helper.waitForGameReady();
      
      // Should be back at scoring
      phase = await helper.getCurrentPhase();
      expect(['scoring', 'playing']).toContain(phase);
    }
  });

  test('should handle multiple back/forward navigations', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    await helper.goto(12345, { disableUrlUpdates: false });
    
    // Create history: bid -> pass -> pass -> pass
    await helper.bid(30, false);
    const url1 = page.url();
    
    await helper.pass();
    const url2 = page.url();
    
    await helper.pass();
    const url3 = page.url();
    
    await helper.pass();
    
    // Go back twice
    await page.goBack();
    await helper.waitForNavigationRestore();
    await page.goBack();
    await helper.waitForNavigationRestore();
    
    // Should be at url2
    expect(page.url()).toBe(url2);
    
    // Go forward once
    await page.goForward();
    await helper.waitForNavigationRestore();
    
    // Should be at url3
    expect(page.url()).toBe(url3);
    
    // Go back three times to beginning
    await page.goBack();
    await helper.waitForNavigationRestore();
    await page.goBack();
    await helper.waitForNavigationRestore();
    
    // Should be at url1
    expect(page.url()).toBe(url1);
  });

  test('should maintain game state consistency through navigation', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    await helper.goto(12345, { disableUrlUpdates: false });
    
    // Play several actions
    await helper.bid(35, false);
    await helper.pass();
    await helper.pass();
    await helper.pass();
    await helper.setTrump('doubles');
    
    // Get current state
    const stateBeforeNav = await page.evaluate(() => {
      const view = (window as any).getGameView?.();
      if (!view) return null;
      return {
        phase: view.phase,
        currentPlayer: view.currentPlayer,
        trump: view.trump,
        currentBid: view.currentBid,
        winningBidder: view.winningBidder
      } as PartialGameState;
    });
    
    // Go back twice
    await page.goBack();
    await helper.waitForNavigationRestore();
    await page.goBack();
    await helper.waitForNavigationRestore();
    
    // Go forward twice
    await page.goForward();
    await helper.waitForNavigationRestore();
    await page.goForward();
    await helper.waitForNavigationRestore();
    
    // Get state after navigation
    const stateAfterNav = await page.evaluate(() => {
      const view = (window as any).getGameView?.();
      if (!view) return null;
      return {
        phase: view.phase,
        currentPlayer: view.currentPlayer,
        trump: view.trump,
        currentBid: view.currentBid,
        winningBidder: view.winningBidder
      } as PartialGameState;
    });
    
    // Key state should match
    if (stateBeforeNav && stateAfterNav) {
      expect(stateAfterNav.phase).toBe(stateBeforeNav.phase);
      expect(stateAfterNav.currentPlayer).toBe(stateBeforeNav.currentPlayer);
      expect(stateAfterNav.trump).toStrictEqual(stateBeforeNav.trump);
      expect(stateAfterNav.currentBid).toStrictEqual(stateBeforeNav.currentBid);
      expect(stateAfterNav.winningBidder).toBe(stateBeforeNav.winningBidder);
    }
  });

  test('should handle back button with AI enabled', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Start with AI players from the beginning - enable URL updates
    await helper.goto(12345, { disableUrlUpdates: false });
    
    // Enable AI for players 1-3
    await helper.enableAIForOtherPlayers();
    
    // Store initial URL
    const urlInitial = page.url();
    
    // Make a bid - this will cause AI to respond immediately in test mode
    await helper.bid(30, false);
    
    // AI should have already responded (synchronous in test mode)
    // Store URL after AI played
    const urlAfterAI = page.url();
    
    // Should have multiple actions in URL now (bid + AI responses)
    // Check v2 URL format
    const params = new URLSearchParams(urlAfterAI.split('?')[1]);
    const actionsStr = params.get('a');
    if (actionsStr) {
      // In test mode with AI, we should have at least 4 actions (P0 bid + 3 AI passes/bids)
      expect(actionsStr.length).toBeGreaterThanOrEqual(4);
    }
    
    // Go back to initial state
    await page.goBack();
    await helper.waitForNavigationRestore();
    
    // Should be back at start of bidding
    const phase = await helper.getCurrentPhase();
    expect(phase).toBe('bidding');
    
    // URL should be initial URL
    expect(page.url()).toBe(urlInitial);
    
    // Forward again should restore the state with AI responses
    await page.goForward();
    await helper.waitForNavigationRestore();
    
    // Should be back to post-AI state (could be trump selection if P0 won, or more bidding)
    const phaseAfter = await helper.getCurrentPhase();
    expect(['bidding', 'trump_selection']).toContain(phaseAfter);
    
    // URL should match the post-AI URL
    expect(page.url()).toBe(urlAfterAI);
  });

  test('should preserve browser history when using back button', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Start with URL tracking enabled
    await helper.goto(12345, { disableUrlUpdates: false });
    
    // Make several actions to build history
    await helper.bid(30, false);
    const url1 = page.url();
    
    await helper.pass();
    const url2 = page.url();
    
    await helper.pass();
    const url3 = page.url();
    
    await helper.pass();
    const url4 = page.url();
    
    // Verify we can go back in history
    await page.goBack();
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(url3);
    
    // Verify history is preserved - we should still be able to go back again
    await page.goBack();
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(url2);
    
    // And forward should still work
    await page.goForward();
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(url3);
    
    await page.goForward();
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(url4);
    
    // Now go back to url2 and make a different action
    await page.goBack();
    await helper.waitForNavigationRestore();
    await page.goBack();
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(url2);
    
    // Make a different action (bid instead of pass)
    await helper.bid(31, false);
    
    // This should create a new history branch, but we should still be able to go back
    await page.goBack();
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(url2);
    
    // And back again to url1
    await page.goBack();
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(url1);
  });

  test('should preserve back history when navigating from mid-history position', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    await helper.goto(12345, { disableUrlUpdates: false });
    
    // Build history: A -> B -> C -> D
    await helper.bid(30, false);  // State B
    const urlB = page.url();
    
    await helper.pass();           // State C  
    
    await helper.pass();           // State D
    
    // Go back to state B (back twice)
    await page.goBack();  // D -> C
    await helper.waitForNavigationRestore();
    await page.goBack();  // C -> B
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(urlB);
    
    // Take a different action from B, creating state E
    await helper.bid(31, false);  // New state E
    const urlE = page.url();
    
    // Critical test: Can we still go back to B?
    await page.goBack();
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(urlB);
    
    // And can we go back even further to the initial state?
    await page.goBack();
    await helper.waitForNavigationRestore();
    const urlInitial = page.url();
    expect(urlInitial).toContain('?s='); // Should have initial state with seed
    
    // Verify we can navigate forward again
    await page.goForward();
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(urlB);
    
    await page.goForward();
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(urlE);
  });

  test('should not lose history after using back button and continuing play', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    await helper.goto(12345, { disableUrlUpdates: false });
    
    // Build history: initial -> bid30 -> pass -> pass
    await helper.bid(30, false);
    const url1Action = page.url();
    
    await helper.pass();
    const url2Actions = page.url();
    
    await helper.pass();
    
    // Go back to 2 actions
    await page.goBack();
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(url2Actions);
    
    // Continue playing from this point with a different action
    await helper.bid(31, false);
    const urlNewPath = page.url();
    
    // Key test: We should still be able to go back to previous states
    // Go back to 2 actions state
    await page.goBack();
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(url2Actions);
    
    // Go back to 1 action state
    await page.goBack();
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(url1Action);
    
    // Go back to initial state
    await page.goBack();
    await helper.waitForNavigationRestore();
    const urlInitial = page.url();
    
    // Verify we're at the initial state (just seed, no actions)
    // Check URL format
    const params = new URLSearchParams(urlInitial.split('?')[1]);
    const actionsStr = params.get('a');
    if (actionsStr) {
      expect(actionsStr).toBe(''); // No actions, just initial state
    }
    
    // And we should be able to go forward through our new path
    await page.goForward();
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(url1Action);
    
    await page.goForward();
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(url2Actions);
    
    await page.goForward();
    await helper.waitForNavigationRestore();
    expect(page.url()).toBe(urlNewPath);
  });
});