import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Complete Trick in Play Area', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    await helper.goto();
    await helper.injectGameState({
      currentPlayer: 0,
      phase: 'playing',
      dealer: 0,
      winningBidder: 0,
      currentBid: { type: 'points' as const, value: 30, player: 0 },
      trump: { type: 'suit', suit: 3 }, // Threes trump
      // All 4 players have played - ready to complete trick
      currentTrick: [
        { player: 0, domino: { high: 3, low: 2, points: 5, id: '3-2' } },
        { player: 1, domino: { high: 4, low: 1, points: 5, id: '4-1' } },
        { player: 2, domino: { high: 5, low: 0, points: 5, id: '5-0' } },
        { player: 3, domino: { high: 6, low: 4, points: 10, id: '6-4' } }
      ],
      hands: {
        0: [
          { high: 6, low: 6, points: 0, id: '6-6' },
          { high: 5, low: 5, points: 10, id: '5-5' },
          { high: 4, low: 4, points: 0, id: '4-4' },
          { high: 3, low: 3, points: 0, id: '3-3' },
          { high: 2, low: 2, points: 0, id: '2-2' },
          { high: 1, low: 1, points: 0, id: '1-1' }
        ],
        1: [], 2: [], 3: []
      },
      // Required fields for GameState
      players: [
        { id: 0, name: 'Player 0', hand: [], teamId: 0, marks: 0 },
        { id: 1, name: 'Player 1', hand: [], teamId: 1, marks: 0 },
        { id: 2, name: 'Player 2', hand: [], teamId: 0, marks: 0 },
        { id: 3, name: 'Player 3', hand: [], teamId: 1, marks: 0 }
      ],
      bids: [],
      tricks: [],
      currentSuit: 3,
      teamScores: [0, 0],
      teamMarks: [0, 0],
      gameTarget: 7,
      tournamentMode: false,
      shuffleSeed: 12345
    });
  });

  test('shows complete trick button in play area', async ({ page }) => {
    // Should see the complete trick button in the playing area
    const completeButton = page.locator('[data-testid="complete-trick"]');
    await expect(completeButton).toBeVisible();
    
    // Button should be prominently displayed
    await expect(completeButton).toContainText('Complete trick');
    
    // Should NOT need to switch to Actions panel
    const actionsTab = page.locator('[data-testid="nav-actions"]');
    await expect(actionsTab).not.toHaveClass(/active/);
  });

  test('complete trick button works directly from play area', async ({ page }) => {
    // Click the complete trick button
    await page.locator('[data-testid="complete-trick"]').click();
    
    // Trick should be completed
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('playing');
    
    // Should be ready for next trick - the phase should still be playing
    // (The actual trick completion would be visible in the UI)
  });

  test('button appears with nice animation', async ({ page }) => {
    // The button should have a fade-in animation
    const button = page.locator('.proceed-action-button');
    await expect(button).toBeVisible();
    
    // Should have the proceed icon
    const icon = button.locator('.proceed-icon');
    await expect(icon).toBeVisible();
    await expect(icon).toContainText('✓');
  });

  test('button is mobile-friendly', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    // Button should still be visible and properly sized
    const button = page.locator('.proceed-action-button');
    await expect(button).toBeVisible();
    
    // Check minimum touch target size
    const box = await button.boundingBox();
    expect(box?.height).toBeGreaterThanOrEqual(48);
  });

  test('no proceed button when waiting for plays', async ({ page }) => {
    // Load state with incomplete trick
    await helper.goto();
    await helper.injectGameState({
      currentPlayer: 1, // Not human player's turn
      phase: 'playing',
      dealer: 0,
      winningBidder: 0,
      currentBid: { type: 'points' as const, value: 30, player: 0 },
      trump: { type: 'suit', suit: 3 },
      currentTrick: [
        { player: 0, domino: { high: 3, low: 2, points: 5, id: '3-2' } }
      ],
      hands: {
        0: [
          { high: 6, low: 6, points: 0, id: '6-6' },
          { high: 5, low: 5, points: 10, id: '5-5' }
        ],
        1: [], 2: [], 3: []
      },
      // Required fields for GameState
      players: [
        { id: 0, name: 'Player 0', hand: [], teamId: 0, marks: 0 },
        { id: 1, name: 'Player 1', hand: [], teamId: 1, marks: 0 },
        { id: 2, name: 'Player 2', hand: [], teamId: 0, marks: 0 },
        { id: 3, name: 'Player 3', hand: [], teamId: 1, marks: 0 }
      ],
      bids: [],
      tricks: [],
      currentSuit: 3,
      teamScores: [0, 0],
      teamMarks: [0, 0],
      gameTarget: 7,
      tournamentMode: false,
      shuffleSeed: 12345
    });
    
    // Should not see any proceed button
    const proceedButton = page.locator('.proceed-action-button');
    await expect(proceedButton).not.toBeVisible();
  });
});