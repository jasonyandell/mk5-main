/**
 * Consensus Layer E2E Tests
 *
 * Verifies the human-only consensus behavior from player 0's perspective:
 * - Human player sees agree-trick action when trick is complete
 * - complete-trick is gated until human agrees
 * - Clicking agree unlocks the proceed action
 *
 * Note: E2E tests run from player 0's perspective. The UI only shows
 * actions available to the viewing player, not all players.
 */

import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe('Consensus Layer - Human Only', () => {
  /**
   * Test setup: Seed 12345 with 1 human + 3 AI
   *
   * Hands for seed 12345:
   * P0 (human): 4-3, 4-1, 3-1, 6-3, 1-1, 6-2, 2-1
   * P1 (ai):    6-0, 5-3, 6-6, 5-4, 4-2, 6-5, 2-2
   * P2 (ai):    5-1, 4-0, 3-0, 5-0, 3-2, 3-3, 6-1
   * P3 (ai):    5-2, 1-0, 5-5, 0-0, 6-4, 4-4, 2-0
   *
   * Actions to reach completed trick state:
   * - bid-30, pass, pass, pass (P0 wins bid)
   * - trump-blanks (P0 selects blanks as trump)
   * - play-6-3 (P0 leads 6-3)
   * - play-6-6 (P1 follows suit)
   * - play-6-1 (P2 follows suit)
   * - play-6-4 (P3 follows suit, 10 points)
   *
   * Result: Trick complete, P1 wins (6-6 highest), 10 points
   * Expected: agree-trick-p0 shown as proceed action (gating complete-trick)
   */
  test('completed trick shows agree action for human, not complete-trick', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    // Load state with completed trick (1 human + 3 AI)
    await helper.loadStateWithActions(12345, [
      'bid-30', 'pass', 'pass', 'pass',
      'trump-blanks',
      'play-6-3', 'play-6-6', 'play-6-1', 'play-6-4'
    ], ['human', 'ai', 'ai', 'ai']);

    // Get available actions - the trick table button should have agree-trick-p0
    const actions = await helper.getAvailableActions();
    const actionIds = actions.map(a => a.id);

    // The proceed button should be agree-trick-p0 (consensus gating)
    expect(actionIds).toContain('agree-trick-p0');

    // complete-trick should NOT be available yet (gated)
    expect(actionIds).not.toContain('complete-trick');
  });

  test('clicking agree-trick advances to next trick', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    // Load state with completed trick
    await helper.loadStateWithActions(12345, [
      'bid-30', 'pass', 'pass', 'pass',
      'trump-blanks',
      'play-6-3', 'play-6-6', 'play-6-1', 'play-6-4'
    ], ['human', 'ai', 'ai', 'ai']);

    // Verify we're showing the agree action
    let actions = await helper.getAvailableActions();
    expect(actions.map(a => a.id)).toContain('agree-trick-p0');

    // Click the agree action (which triggers complete-trick behind the scenes)
    await helper.selectAction('agree-trick-p0');

    // Wait for state to update - trick should complete and we move on
    // With 1 human + 3 AI, after human agrees, complete-trick executes automatically
    // Then the next trick begins (P1 won, so P1 leads, but P1 is AI so it plays)

    // Give time for the chain of events
    await page.waitForTimeout(300);

    // The current trick should now be empty or have new plays
    // Check that we're no longer showing agree-trick-p0
    actions = await helper.getAvailableActions();
    const actionIds = actions.map(a => a.id);

    // Should NOT have the same agree action (trick advanced)
    expect(actionIds).not.toContain('agree-trick-p0');

    // We should see either: new dominoes to play, another agree action (if AI completed another trick),
    // or the trick-table (neutral state)
    const hasDominoActions = actionIds.some(id => id.startsWith('domino-'));
    const hasNewAgree = actionIds.some(id => id.startsWith('agree-'));
    const hasTrickTable = actionIds.includes('trick-table');

    expect(hasDominoActions || hasNewAgree || hasTrickTable).toBe(true);
  });

  test('with all humans, each human agree action appears sequentially', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    // Load with all human players
    await helper.loadStateWithActions(12345, [
      'bid-30', 'pass', 'pass', 'pass',
      'trump-blanks',
      'play-6-3', 'play-6-6', 'play-6-1', 'play-6-4'
    ], ['human', 'human', 'human', 'human']);

    // Player 0's view should show agree-trick-p0
    let actions = await helper.getAvailableActions();
    let actionIds = actions.map(a => a.id);

    // Should see player 0's agree action (viewing as player 0)
    expect(actionIds).toContain('agree-trick-p0');

    // Should NOT see complete-trick (still gated - need all 4 humans to agree)
    expect(actionIds).not.toContain('complete-trick');

    // Click player 0's agree
    await helper.selectAction('agree-trick-p0');

    // After clicking, player 0 has agreed, but 3 more humans need to agree
    // Since we're viewing as player 0, we won't see their agree buttons
    // We'll see the trick table (waiting for others)
    actions = await helper.getAvailableActions();
    actionIds = actions.map(a => a.id);

    // Player 0's agree action should be gone
    expect(actionIds).not.toContain('agree-trick-p0');

    // Still no complete-trick (others haven't agreed)
    expect(actionIds).not.toContain('complete-trick');
  });

  test('with mixed human/AI, only human agree actions are needed', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);

    // Load with 2 humans (P0, P2) + 2 AI (P1, P3)
    await helper.loadStateWithActions(12345, [
      'bid-30', 'pass', 'pass', 'pass',
      'trump-blanks',
      'play-6-3', 'play-6-6', 'play-6-1', 'play-6-4'
    ], ['human', 'ai', 'human', 'ai']);

    // Player 0's view should show agree-trick-p0
    let actions = await helper.getAvailableActions();
    let actionIds = actions.map(a => a.id);

    expect(actionIds).toContain('agree-trick-p0');
    expect(actionIds).not.toContain('complete-trick');

    // AI players (P1, P3) should NOT have agree actions visible
    expect(actionIds).not.toContain('agree-trick-p1');
    expect(actionIds).not.toContain('agree-trick-p3');

    // Click player 0's agree
    await helper.selectAction('agree-trick-p0');

    // Now we need P2 to agree (also human), but P2's action won't be visible to P0
    // We'll see the trick table
    actions = await helper.getAvailableActions();
    actionIds = actions.map(a => a.id);

    expect(actionIds).not.toContain('agree-trick-p0');
    // Still waiting for P2
    expect(actionIds).not.toContain('complete-trick');
  });
});
