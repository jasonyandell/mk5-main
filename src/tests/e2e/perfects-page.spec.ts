import { test, expect } from '@playwright/test';

test.describe('Perfects Page', () => {
  test('loads and displays perfect hand collections', async ({ page }) => {
    await page.goto('/perfects');

    // Wait for data to load by checking for dominoes
    await page.waitForSelector('[data-testid^="domino-"]', { timeout: 5000 });

    // Check that hands are displayed (now with "Hand 1", "Hand 2", etc.)
    const hands = page.locator('span:has-text("Hand 1")');
    await expect(hands.first()).toBeVisible();

    // Check for trump badges
    const trumpBadges = page.locator('.badge-info');
    await expect(trumpBadges.first()).toBeVisible();

    // Trump badges are the main indicator now (no type badges anymore)

    // Check that dominoes are rendered
    const dominoes = page.locator('[data-testid^="domino-"]');
    await expect(dominoes.first()).toBeVisible();

    // Test pagination if present
    const nextButton = page.locator('button:has-text("Next")').first();
    if (await nextButton.isVisible()) {
      await nextButton.click();
      // After clicking next, we should still see hands
      await expect(page.locator('span:has-text("Hand 1")').first()).toBeVisible();
    }
  });

  test('can navigate manually to main game', async ({ page }) => {
    await page.goto('/perfects');

    // Verify we're on perfects page
    await expect(page.locator('h1:has-text("42")')).toBeVisible();

    // Navigate manually to main game
    await page.goto('/');

    // Should be back on main game page
    await expect(page).toHaveURL('/');
    // Check for main game header with scoring display
    await expect(page.locator('[data-testid="app-header"]')).toBeVisible();
  });
});