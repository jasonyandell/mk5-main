import { test, expect } from '@playwright/test';
import { PlaywrightHelper } from './src/tests/e2e/helpers/playwrightHelper';

test('domino highlighting works on hover during bidding phase', async ({ page }) => {
  const helper = new PlaywrightHelper(page);

  // Start game in bidding phase with known hand
  const gameState = {
    phase: 'bidding',
    currentPlayer: 0,
    hands: [
      [
        { high: 6, low: 6, points: 0 }, // double-6
        { high: 5, low: 3, points: 0 }, // 5-3
        { high: 5, low: 2, points: 0 }, // 5-2
        { high: 4, low: 1, points: 5 },  // 4-1
        { high: 3, low: 3, points: 0 }, // double-3
        { high: 2, low: 0, points: 0 }, // 2-0
        { high: 1, low: 0, points: 0 }, // 1-0
      ],
      // ... other hands
    ]
  };

  await helper.navigateToGame();
  await helper.loadGameFromState(gameState);

  // Verify we're in bidding phase
  await expect(page.getByTestId('phase-bidding')).toBeVisible();

  // Test 1: Hover over a non-double domino (5-3)
  const domino53 = page.getByTestId('domino-5-3');
  await domino53.hover();
  
  // Should show "Fives" or "Threes" indicator depending on which half
  const indicator = page.locator('.suit-highlight-indicator');
  await expect(indicator).toBeVisible();
  const text = await indicator.textContent();
  expect(['Fives', 'Threes']).toContain(text);

  // Check that dominoes with fives are highlighted
  await expect(page.getByTestId('domino-5-3')).toHaveClass(/highlight-primary/);
  await expect(page.getByTestId('domino-5-2')).toHaveClass(/highlight-primary/);

  // Test 2: Hover over a double (6-6)
  await domino53.hover({ position: { x: 0, y: 0 } }); // move away first
  const double66 = page.getByTestId('domino-6-6');
  await double66.hover();

  // Should show "Doubles & Sixes"
  await expect(indicator).toHaveText('Doubles & Sixes');

  // All doubles should be highlighted as primary
  await expect(page.getByTestId('domino-6-6')).toHaveClass(/highlight-primary/);
  await expect(page.getByTestId('domino-3-3')).toHaveClass(/highlight-primary/);

  // Test 3: Move mouse away - highlighting should disappear
  await page.mouse.move(0, 0);
  await expect(indicator).not.toBeVisible();
  await expect(page.getByTestId('domino-6-6')).not.toHaveClass(/highlight-primary/);
});

test('domino highlighting works during trump selection', async ({ page }) => {
  const helper = new PlaywrightHelper(page);

  // Start game in trump selection phase
  const gameState = {
    phase: 'trump_selection',
    currentPlayer: 0,
    bidWinner: 0,
    hands: [
      [
        { high: 4, low: 4, points: 0 }, // double-4
        { high: 4, low: 2, points: 0 }, // 4-2
        { high: 4, low: 0, points: 0 }, // 4-0
        // ... rest of hand
      ],
      // ... other hands
    ]
  };

  await helper.navigateToGame();
  await helper.loadGameFromState(gameState);

  // Verify we're in trump selection phase
  await expect(page.getByTestId('phase-trump_selection')).toBeVisible();

  // Hover over 4-2 domino
  const domino42 = page.getByTestId('domino-4-2');
  await domino42.hover({ position: { x: 10, y: 10 } }); // hover on top half (fours)

  // Should show "Fours" and highlight all fours
  const indicator = page.locator('.suit-highlight-indicator');
  await expect(indicator).toHaveText('Fours');
  await expect(page.getByTestId('domino-4-4')).toHaveClass(/highlight-primary/);
  await expect(page.getByTestId('domino-4-2')).toHaveClass(/highlight-primary/);
  await expect(page.getByTestId('domino-4-0')).toHaveClass(/highlight-primary/);
});

test('no highlighting during playing phase', async ({ page }) => {
  const helper = new PlaywrightHelper(page);

  // Start game in playing phase
  const gameState = {
    phase: 'playing',
    currentPlayer: 0,
    trump: { type: 'suit', suit: 5 },
    hands: [
      [
        { high: 5, low: 5, points: 10 },
        { high: 5, low: 3, points: 0 },
        // ... rest of hand
      ],
      // ... other hands
    ]
  };

  await helper.navigateToGame();
  await helper.loadGameFromState(gameState);

  // Verify we're in playing phase
  await expect(page.getByTestId('phase-playing')).toBeVisible();

  // Hover over a domino
  const domino55 = page.getByTestId('domino-5-5');
  await domino55.hover();

  // No highlight indicator should appear
  const indicator = page.locator('.suit-highlight-indicator');
  await expect(indicator).not.toBeVisible();

  // Domino should not have highlight class
  await expect(domino55).not.toHaveClass(/highlight-primary/);
});