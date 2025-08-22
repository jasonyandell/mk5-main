import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe('Basic Gameplay', () => {
  test('should load game interface', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    const locators = helper.getLocators();
    
    await helper.goto(12345);
    
    // Check main UI elements are present (auto-waits)
    await expect(locators.app()).toBeVisible();
    await expect(locators.appHeader()).toBeVisible();
    // Score display should be visible
    await expect(locators.scoreDisplay()).toBeVisible();
  });

  test('should start in bidding phase', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    await helper.goto(12345);
    
    // Verify phase
    const phase = await helper.getCurrentPhase();
    expect(phase.toLowerCase()).toContain('bidding');
    
    // Should have bidding options available
    const actions = await helper.getAvailableActions();
    expect(actions.length).toBeGreaterThan(0);
    expect(actions.some(action => action.type === 'pass')).toBe(true);
  });

  test('should show proper UI during bidding phase', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    await helper.goto(12345);
    
    // The ActionPanel should be visible since we're in bidding phase
    const actionPanel = page.locator('.action-panel');
    await expect(actionPanel).toBeVisible();
    
    // Should show hand and bidding actions
    const handSection = actionPanel.locator('.hand-section');
    await expect(handSection).toBeVisible();
    
    // Should have bidding section with actions
    const biddingSection = actionPanel.locator('.action-group').filter({ hasText: 'Bidding' });
    await expect(biddingSection).toBeVisible();
    
    // Should have pass button
    const passButton = actionPanel.locator('button').filter({ hasText: 'Pass' });
    await expect(passButton).toBeVisible();
    
    // Bidding table should NOT be visible when player has actions
    const biddingTable = page.locator('.bidding-table');
    await expect(biddingTable).not.toBeVisible();
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

  test('should progress through bidding round', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Load state after all 4 players have passed
    await helper.loadStateWithActions(12345, ['p', 'p', 'p', 'p']);
    
    // Should have redeal action available after all passes
    const actions = await helper.getAvailableActions();
    expect(actions.some(action => action.type === 'redeal')).toBe(true);
    
    // Execute redeal
    await helper.selectAction({ type: 'redeal' });
    
    // Should be back in bidding phase after redeal
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('bidding');
  });

  test('should handle winning bid and trump selection', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Load state with player 0 bidding 30 and others passing
    await helper.loadStateWithActions(12345, ['30', 'p', 'p', 'p']);
    
    // Should now be in trump selection
    const actions = await helper.getAvailableActions();
    expect(actions.some(action => action.type === 'trump_selection')).toBe(true);
  });

  test('should transition to playing phase after trump selection', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    
    // Load state with bidding done and trump selected (t0 = blanks)
    await helper.loadStateWithActions(12345, ['30', 'p', 'p', 'p', 't0']);
    
    // Verify we're in playing phase
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('playing');
  });

  test('should track scores correctly', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    const locators = helper.getLocators();
    
    await helper.goto(12345);
    
    // Check initial scores using locators
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
    
    // Load state in playing phase with trump selected
    await helper.loadStateWithActions(12345, ['30', 'p', 'p', 'p', 'trump-blanks']);
    
    // Playing area should be visible
    await expect(helper.getLocators().playingArea()).toBeVisible();
    
    // Get current trick (should be empty initially)
    const currentTrick = await helper.getCurrentTrick();
    expect(Array.isArray(currentTrick)).toBe(true);
    expect(currentTrick.length).toBe(0);
    
    // Play a domino
    await helper.playAnyDomino();
    
    // Trick should now have one domino
    const updatedTrick = await helper.getCurrentTrick();
    expect(updatedTrick.length).toBeGreaterThan(0);
  });

  test('should handle new game correctly', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    const locators = helper.getLocators();
    
    // Start with a game in progress
    await helper.loadStateWithActions(12345, ['30', 'p', 'p', 'p', 'trump-blanks']);
    
    // New game by loading fresh state
    await helper.goto(54321);
    
    // Should reset to initial state
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('bidding');
    
    // Check scores directly from UI (no trump in bidding phase)
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
    
    // Should be able to interact with bidding
    const actions = await helper.getAvailableActions();
    expect(actions.length).toBeGreaterThan(0);
  });
});