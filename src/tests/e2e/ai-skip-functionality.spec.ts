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
    
    const table = page.locator('.trick-table');
    await expect(table).toBeVisible();
    await expect(table).toHaveClass(/tappable/);
    await expect(table).not.toBeDisabled();
  });

  test('clicking table calls skipAIDelays', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Load deterministic state
    await helper.loadStateWithActions(12345, ['30', 'p', 'p', 'p', 'trump-blanks']);
    
    // Inject spy to verify skipAIDelays is called
    await page.evaluate(() => {
      let called = false;
      if ((window as any).controllerManager) {
        const original = (window as any).controllerManager.skipAIDelays;
        (window as any).controllerManager.skipAIDelays = function() {
          called = true;
          return original.call(this);
        };
      }
      (window as any).checkSkipCalled = () => called;
    });
    
    // Click the table
    await page.locator('.trick-table').click();
    
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
    const table = page.locator('.trick-table');
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
    
    const table = page.locator('.trick-table');
    await table.hover();
    
    const cursor = await table.evaluate(el => 
      window.getComputedStyle(el).cursor
    );
    expect(cursor).toBe('pointer');
  });

  test('multiple clicks work without issues', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    await helper.loadStateWithActions(12345, ['30', 'p', 'p', 'p', 'trump-blanks']);
    
    const table = page.locator('.trick-table');
    
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
    
    // Load state where player 0 can play
    await helper.loadStateWithActions(12345, ['30', 'p', 'p', 'p', 'trump-blanks']);
    
    // Play a domino
    await helper.playAnyDomino();
    
    // Click table - should trigger skipAIDelays
    await page.locator('.trick-table').click();
    
    // Game should still be valid
    const phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');
    
    // Trick should exist and have at least one domino
    const trick = await helper.getCurrentTrick();
    expect(trick.length).toBeGreaterThan(0);
  });
});