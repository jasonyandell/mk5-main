/**
 * URL Navigation and State Restoration Tests
 *
 * CURRENTLY SKIPPED: These tests require URL updates to be enabled, but the
 * PlaywrightGameHelper disables URL updates (history.pushState/replaceState) for
 * deterministic testing. To properly test URL-based navigation:
 * 1. Would need a separate test mode that preserves URL updates
 * 2. Would need to handle async state changes from URL changes
 * 3. May require refactoring helper to support URL-update-enabled mode
 *
 * The underlying functionality (URL encoding/decoding and state restoration)
 * is tested in unit tests. These E2E tests would add coverage for:
 * - Browser back/forward button behavior
 * - Direct URL navigation
 * - Popstate event handling
 *
 * Decision: Skip for now - the core functionality is covered by unit tests,
 * and proper E2E coverage would require significant test infrastructure changes.
 */
import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe.skip('URL Navigation and State Restoration', () => {
  test('AI should play after browser back button', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    // Load game state with human players and P0 winning bid (start of playing phase)
    await helper.loadStateWithActions(12345,
      ['bid-30', 'pass', 'pass', 'pass', 'trump-blanks'],
      ['human', 'human', 'human', 'human']
    );

    // Should be in playing phase
    let phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');

    // Store current URL
    const urlBefore = await helper.getCurrentURL();

    // Play one domino as P0 - this will add the action to URL
    const playingArea = page.locator('[data-testid="playing-area"]');
    await expect(playingArea).toBeVisible();

    // Get initial hand count
    const initialHandCount = await page.locator('[data-testid="playing-area"] [data-playable="true"]').count();
    expect(initialHandCount).toBeGreaterThan(0);

    // Play a domino
    await helper.playAnyDomino();

    // Verify domino appears in trick by checking DOM
    const trickArea = page.locator('[data-testid="trick-area"]');
    await trickArea.locator('button[data-testid^="domino-"]').first().waitFor({ state: 'visible', timeout: 2000 });
    let trickLength = await trickArea.locator('button[data-testid^="domino-"]').count();
    expect(trickLength).toBe(1); // P0 played

    // Store URL after playing (includes the play action)
    const urlAfter = await helper.getCurrentURL();
    expect(urlAfter).not.toBe(urlBefore); // URL should have changed

    // Go back to before we played
    await page.goto(urlBefore);
    await helper.waitForGameReady();

    // Should be back at start of playing phase
    phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');

    // Check trick is empty by checking DOM
    const trickAreaBefore = page.locator('[data-testid="trick-area"]');
    trickLength = await trickAreaBefore.locator('button[data-testid^="domino-"]').count();
    expect(trickLength).toBe(0);

    // Go forward to after playing
    await page.goto(urlAfter);
    await helper.waitForGameReady();

    // Should have the played domino - check DOM
    await trickArea.locator('button[data-testid^="domino-"]').first().waitFor({ state: 'visible', timeout: 2000 });
    trickLength = await trickArea.locator('button[data-testid^="domino-"]').count();
    expect(trickLength).toBe(1); // P0 played (others are human and haven't played yet)
  });

  test('should restore state correctly after loading from URL', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    // Load state where P1 won bid with human players
    // P0 passes, P1 bids 30, P2/P3 pass, P1 selects trump, then P1 plays first domino
    await helper.loadStateWithActions(12345,
      ['pass', 'bid-30', 'pass', 'pass', 'trump-blanks', 'play-6-6'],
      ['human', 'human', 'human', 'human']
    );

    // Should be in playing phase with state restored
    const phase = await helper.getCurrentPhase();
    expect(phase).toBe('playing');

    // Should have P1's domino in the trick (from URL actions)
    // Check DOM for dominoes in trick area
    const trickArea = page.locator('[data-testid="trick-area"]');
    await trickArea.locator('button[data-testid^="domino-"]').first().waitFor({ state: 'visible', timeout: 2000 });
    const trickLength = await trickArea.locator('button[data-testid^="domino-"]').count();
    expect(trickLength).toBe(1); // P1 played (bid winner leads)

    // Current player should be P2 (next after P1)
    const currentPlayer = await helper.getCurrentPlayer();
    expect(currentPlayer).toContain('P2');
  });

  test('should handle state navigation correctly', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    // Load state at trump selection
    await helper.loadStateWithActions(12345,
      ['bid-30', 'pass', 'pass', 'pass'],
      ['human', 'human', 'human', 'human']
    );

    // Should be at trump selection
    const phase1 = await helper.getCurrentPhase();
    expect(phase1).toBe('trump_selection');

    // Store URL before trump selection
    const trumpSelectionUrl = await helper.getCurrentURL();

    // Select trump
    await helper.setTrump('blanks');

    // Should be in playing phase
    const phase2 = await helper.getCurrentPhase();
    expect(phase2).toBe('playing');

    // Store the URL after trump selection
    const playingUrl = await helper.getCurrentURL();
    expect(playingUrl).not.toBe(trumpSelectionUrl); // URLs should differ

    // Navigate back to trump selection state
    await page.goto(trumpSelectionUrl);
    await helper.waitForGameReady();

    // Should be back at trump selection
    const phase3 = await helper.getCurrentPhase();
    expect(phase3).toBe('trump_selection');

    // Navigate forward to playing state
    await page.goto(playingUrl);
    await helper.waitForGameReady();

    // Should be in playing phase
    const phase4 = await helper.getCurrentPhase();
    expect(phase4).toBe('playing');

    // Verify we can still play
    const playingArea = page.locator('[data-testid="playing-area"]');
    await expect(playingArea).toBeVisible();

    // Verify turn player is displayed (should be P0 since they won bid)
    const currentPlayer = await helper.getCurrentPlayer();
    expect(currentPlayer).toContain('P0');
  });
});