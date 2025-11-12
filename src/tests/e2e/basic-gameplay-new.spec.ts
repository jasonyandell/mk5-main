/**
 * Basic Gameplay Tests - Refactored Version
 *
 * This is a pilot test demonstrating the new testing pattern:
 * - No direct window object manipulation
 * - Tests only DOM/UI behavior
 * - Uses URL state loading for setup
 * - Minimal test API exposure
 *
 * Compare with basic-gameplay.spec.ts to see improvements.
 */

import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe('Basic Gameplay (Refactored)', () => {
  test('should load game interface', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    const locators = helper.getLocators();

    await helper.goto(12345);

    // Check main UI elements are present (auto-waits)
    await expect(locators.app()).toBeVisible();
    await expect(locators.appHeader()).toBeVisible();
    await expect(locators.scoreDisplay()).toBeVisible();
  });

  test('should start in bidding phase', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    await helper.goto(12345);

    // Verify phase from DOM (no window access)
    const phase = await helper.getCurrentPhase();
    expect(phase?.toLowerCase()).toContain('bidding');

    // Should have action buttons visible
    const actionPanel = page.locator('[data-testid="action-panel"]');
    await expect(actionPanel).toBeVisible();

    // Should have pass button
    await expect(page.locator('[data-testid="pass"]')).toBeVisible();

    // Should have bid buttons
    const bidButtons = page.locator('button[data-testid^="bid-"]');
    expect(await bidButtons.count()).toBeGreaterThan(0);
  });

  test('should show proper UI during bidding phase', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    await helper.goto(12345);

    // The ActionPanel should be visible since we're in bidding phase
    const actionPanel = page.locator('[data-testid="action-panel"]');
    await expect(actionPanel).toBeVisible();

    // Should show hand with 7 dominoes
    const handDominoes = actionPanel.locator('button[data-testid^="domino-"]');
    const dominoCount = await handDominoes.count();
    expect(dominoCount).toBe(7);

    // Should have pass button
    const passButton = page.locator('[data-testid="pass"]');
    await expect(passButton).toBeVisible();

    // Should have bid buttons
    const bidButtons = page.locator('button[data-testid^="bid-"]');
    const bidCount = await bidButtons.count();
    expect(bidCount).toBeGreaterThan(0);
  });

  test('should handle winning bid and trump selection', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    // Load state with player 0 bidding 30 and others passing
    // Players 1-3 are AI so they execute their pass actions automatically
    await helper.loadStateWithActions(12345, ['bid-30', 'pass', 'pass', 'pass'],
      ['human', 'ai', 'ai', 'ai']);

    // Should now be in trump selection phase (check DOM)
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('trump');

    // Should have trump selection buttons visible
    const trumpButtons = page.locator('button[data-testid^="trump-"]');
    expect(await trumpButtons.count()).toBeGreaterThan(0);
  });

  test('should transition to playing phase after trump selection', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    // Load state with bidding done and trump selected
    // Players 1-3 are AI so they execute their pass actions automatically
    await helper.loadStateWithActions(12345, ['bid-30', 'pass', 'pass', 'pass', 'trump-blanks'],
      ['human', 'ai', 'ai', 'ai']);

    // Verify we're in playing phase (from DOM)
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('playing');

    // Playing area should be visible
    await expect(helper.getLocators().playingArea()).toBeVisible();
  });

  test('should track scores correctly', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    const locators = helper.getLocators();

    await helper.goto(12345);

    // Check initial scores from UI elements
    const usScore = await locators.scoreUs().textContent();
    const themScore = await locators.scoreThem().textContent();

    // Extract numbers from score text
    const usScoreNum = parseInt(usScore?.match(/\d+/)?.[0] || '0');
    const themScoreNum = parseInt(themScore?.match(/\d+/)?.[0] || '0');

    expect(usScoreNum).toBe(0);
    expect(themScoreNum).toBe(0);
  });

  test('should allow domino plays in correct order', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    // Load state in playing phase - use all human players for control
    await helper.loadStateWithActions(12345, ['bid-30', 'pass', 'pass', 'pass', 'trump-blanks'],
      ['human', 'human', 'human', 'human']);

    // Playing area should be visible
    await expect(helper.getLocators().playingArea()).toBeVisible();

    // Check trick is initially empty by looking at DOM
    // Dominoes in the trick have data-testid="domino-{high}-{low}" attribute
    const trickArea = page.locator('[data-testid="trick-area"]');
    const initialDominoesInTrick = await trickArea.locator('button[data-testid^="domino-"]').count();
    expect(initialDominoesInTrick).toBe(0);

    // Play a domino
    await helper.playAnyDomino();

    // Trick should now have one domino (check DOM)
    // Wait for the domino to appear in the trick area using proper selector
    await trickArea.locator('button[data-testid^="domino-"]').first().waitFor({ state: 'visible', timeout: 2000 });
    const updatedDominoesInTrick = await trickArea.locator('button[data-testid^="domino-"]').count();
    expect(updatedDominoesInTrick).toBe(1);
  });

  test('should handle new game correctly', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    const locators = helper.getLocators();

    // Start with a game in progress
    // Players 1-3 are AI so they execute their pass actions automatically
    await helper.loadStateWithActions(12345, ['bid-30', 'pass', 'pass', 'pass', 'trump-blanks'],
      ['human', 'ai', 'ai', 'ai']);

    // Verify we're in playing
    expect(await helper.getCurrentPhase()).toContain('playing');

    // New game by loading fresh seed
    await helper.goto(54321);

    // Should reset to bidding phase
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('bidding');

    // Check scores are reset (from UI)
    const usScore = await locators.scoreUs().textContent();
    const themScore = await locators.scoreThem().textContent();

    const usScoreNum = parseInt(usScore?.match(/\d+/)?.[0] || '0');
    const themScoreNum = parseInt(themScore?.match(/\d+/)?.[0] || '0');

    expect(usScoreNum).toBe(0);
    expect(themScoreNum).toBe(0);
  });

  test('should be responsive on mobile', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    const locators = helper.getLocators();

    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });

    await helper.goto(12345);

    // Should still be functional
    await expect(locators.appHeader()).toBeVisible();
    await expect(locators.scoreDisplay()).toBeVisible();

    // Should have bidding UI visible
    await expect(page.locator('[data-testid="action-panel"]')).toBeVisible();
    await expect(page.locator('[data-testid="pass"]')).toBeVisible();
  });

  test('should display trump indicator during playing phase', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    // Players 1-3 are AI so they execute their pass actions automatically
    await helper.loadStateWithActions(12345, ['bid-30', 'pass', 'pass', 'pass', 'trump-blanks'],
      ['human', 'ai', 'ai', 'ai']);

    // During playing phase, trump is shown in the game info bar with class text-secondary
    // It's displayed as "{trump-name} trump" (e.g., "blanks trump")
    const trumpDisplay = page.locator('.game-info-bar .text-secondary');
    await expect(trumpDisplay).toBeVisible();

    // Should show "blanks trump"
    const trumpText = await trumpDisplay.textContent();
    expect(trumpText?.toLowerCase()).toContain('blank');
    expect(trumpText?.toLowerCase()).toContain('trump');
  });

  test('should handle passes correctly', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    await helper.goto(12345);

    // Click pass button
    await helper.pass();

    // After AI plays, we should still be in bidding
    await page.waitForTimeout(100);
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('bidding');
  });

  test('should allow valid opening bids', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    await helper.goto(12345);

    const actions = await helper.getAvailableActions();

    // Should include point bids 30-41
    expect(actions.some(action => action.type === 'bid_points' && action.value === 30)).toBe(true);
    expect(actions.some(action => action.type === 'bid_points' && action.value === 35)).toBe(true);
    expect(actions.some(action => action.type === 'bid_points' && action.value === 41)).toBe(true);

    // Should include mark bids 1-2 in tournament mode
    expect(actions.some(action => action.type === 'bid_marks' && action.value === 1)).toBe(true);
    expect(actions.some(action => action.type === 'bid_marks' && action.value === 2)).toBe(true);
  });

  test('should not allow invalid opening bids', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    await helper.goto(12345);

    const actions = await helper.getAvailableActions();

    // Should not include point bids below 30 or above 41
    expect(actions.some(action => action.type === 'bid_points' && action.value === 29)).toBe(false);
    expect(actions.some(action => action.type === 'bid_points' && action.value === 42)).toBe(false);

    // Should not include 3+ marks in opening bid
    expect(actions.some(action => action.type === 'bid_marks' && typeof action.value === 'number' && action.value >= 3)).toBe(false);
  });

  test('should progress through bidding round and allow redeal', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    // Load state after all 4 players have passed
    await helper.loadStateWithActions(12345, ['pass', 'pass', 'pass', 'pass'],
      ['human', 'human', 'human', 'human']);

    // Should have redeal action available after all passes
    const actions = await helper.getAvailableActions();
    expect(actions.some(action => action.type === 'redeal')).toBe(true);

    // Execute redeal
    await helper.selectAction({ type: 'redeal' });

    // Should be back in bidding phase after redeal
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('bidding');
  });
});
