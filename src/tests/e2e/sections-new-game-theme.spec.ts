// TODO: Rewrite for new variant system (old section-runner removed)

import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe('Sections: Header New Game preserves theme', () => {
  test('Header New Game keeps selected theme across multiple resets', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.goto(12345, { disableUrlUpdates: false });

    // Open settings and switch to Theme tab
    await page.getByRole('button', { name: 'Menu' }).click();
    await page.locator('.settings-btn').click();
    await expect(page.locator('[data-testid="settings-panel"]')).toBeVisible();
    await page.getByRole('button', { name: 'Theme' }).click();

    // Select Dark theme
    const targetTheme = 'dark';
    const themePreview = page.locator(`[data-theme="${targetTheme}"]`).first();
    await expect(themePreview).toBeVisible();
    await themePreview.locator('..').click();

    // Verify DOM theme and URL
    await page.waitForFunction((expected) => document.documentElement.getAttribute('data-theme') === expected, targetTheme);
    await expect(page).toHaveURL(/t=dark/);

    // Close settings
    await page.locator('[data-testid="settings-close-button"]').click();

    // Click New Game from header menu twice; verify theme/URL persist
    for (let i = 0; i < 2; i++) {
      await page.getByRole('button', { name: 'Menu' }).click();
      await page.getByRole('button', { name: 'New Game' }).click();
      await page.waitForSelector('.app-container');
      await page.waitForFunction((expected) => document.documentElement.getAttribute('data-theme') === expected, targetTheme);
      await expect(page).toHaveURL(/t=dark/);
    }
  });
});

