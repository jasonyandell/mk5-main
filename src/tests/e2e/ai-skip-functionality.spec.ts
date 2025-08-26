/* eslint-disable @typescript-eslint/no-explicit-any */
// Reason: page.evaluate() runs in browser context where TypeScript cannot verify window properties.
// The use of 'any' for window casts is architectural, not a shortcut.

import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe('AI Skip Functionality', () => {
  test('table should be clickable in playing phase', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Load deterministic state in playing phase
    await helper.loadStateWithActions(12345, ['30', 'p', 'p', 'p', 'trump-blanks']);
    
    const table = page.locator('[data-testid="trick-table"]');
    await expect(table).toBeVisible();
    await expect(table).not.toBeDisabled();
  });

  test('clicking table calls skipAIDelays', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Load deterministic state
    await helper.loadStateWithActions(12345, ['30', 'p', 'p', 'p', 'trump-blanks']);
    
    // Inject spy to verify skipAIDelays is called
    await page.evaluate(() => {
      let called = false;
      if ((window as any).gameActions) {
        const original = (window as any).gameActions.skipAIDelays;
        (window as any).gameActions.skipAIDelays = function() {
          called = true;
          return original.call(this);
        };
      }
      (window as any).checkSkipCalled = () => called;
    });
    
    // Click the table
    await page.locator('[data-testid="trick-table"]').click();
    
    // Verify skipAIDelays was called
    const wasCalled = await page.evaluate(() => {
      return (window as any).checkSkipCalled ? (window as any).checkSkipCalled() : false;
    });
    
    expect(wasCalled).toBe(true);
  });

  test('table click works with complete trick', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Load state with a full trick played (deterministic)
    await helper.loadStateWithActions(12345, [
      '30', 'p', 'p', 'p',
      'trump-blanks',
      'play-6-6', 'play-6-5', 'play-6-4', 'play-6-3',
      'complete-trick'  // Complete the trick
    ]);
    
    // Verify we're in playing phase with new trick
    const phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');
    
    // Table should still be clickable
    const table = page.locator('[data-testid="trick-table"]');
    await expect(table).toBeVisible();
    
    // Click table - should not cause errors
    await table.click();
    
    // Game should still be valid
    const phaseAfter = await helper.getCurrentPhase();
    expect(phaseAfter).toBe('playing');
  });

  test('table hover shows pointer cursor', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    await helper.loadStateWithActions(12345, ['30', 'p', 'p', 'p', 'trump-blanks']);
    
    const table = page.locator('[data-testid="trick-table"]');
    await table.hover();
    
    const cursor = await table.evaluate(el => 
      window.getComputedStyle(el).cursor
    );
    expect(cursor).toBe('pointer');
  });

  test('multiple clicks work without issues', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    await helper.loadStateWithActions(12345, ['30', 'p', 'p', 'p', 'trump-blanks']);
    
    const table = page.locator('[data-testid="trick-table"]');
    
    // Click multiple times - should not cause errors
    for (let i = 0; i < 3; i++) {
      await table.click();
    }
    
    // Game should still be valid
    const phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');
  });

  test('table click works after player action', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Load state where player 0 can play - with AI players
    await helper.loadStateWithActions(12345, ['30', 'p', 'p', 'p', 'trump-blanks'], 
      ['human', 'ai', 'ai', 'ai']);
    
    // AI is already enabled from loadStateWithActions, no need to enable again
    
    // Play a domino
    await helper.playAnyDomino();
    
    // In test mode, AI should execute synchronously, so trick should be complete
    const trickLength = await page.evaluate(() => {
      const state = (window as any).getGameState();
      return state.currentTrick.length;
    });
    expect(trickLength).toBe(4); // Full trick after P0 plays and AI follows
    
    // Now click the trick table button (not the dominoes)
    // Use the stable data-trick-button attribute to find the button
    const trickButton = page.locator('[data-trick-button="true"]').first();
    await trickButton.click();
    
    // Game should still be valid - could be playing (more tricks) or scoring (hand complete)
    const phase = await helper.getCurrentPhase();
    expect(['playing', 'scoring']).toContain(phase);
  });
});