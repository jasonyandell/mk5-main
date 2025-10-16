// TODO: Rewrite for new variant system (old section-runner removed)
// TODO: Rewrite for new variant system (old section-runner removed)

import { test, expect } from '@playwright/test';

test.describe('Sections: Theme preservation', () => {
  test('h=one_hand with t=dark preserves theme on seed injection and into playing', async ({ page }) => {
    await page.goto('/?h=one_hand&t=dark&testMode=true', { waitUntil: 'networkidle' });

    // App should render and theme should apply
    await page.waitForSelector('.app-container');

    // Wait for theme attribute to reflect URL theme
    await page.waitForFunction(() => document.documentElement.getAttribute('data-theme') === 'dark');

    // Scenario should remain present initially and theme param should exist
    await expect(page).toHaveURL(/h=one_hand/);
    await expect(page).toHaveURL(/t=dark/);

    // Should reach playing phase
    await page.waitForFunction(() => document.querySelector('.app-container')?.getAttribute('data-phase') === 'playing');

    // Ensure URL contains a seed now (injected) and still has theme
    await expect(page).toHaveURL(/\?[^#]*s=/);
    await expect(page).toHaveURL(/t=dark/);
  });

  test('New Game from completion modal preserves theme parameter and DOM theme across multiple runs', async ({ page }) => {
    // Start section with explicit theme and testMode to skip seed finder
    await page.goto('/?h=one_hand&t=dark&testMode=true', { waitUntil: 'networkidle' });
    await page.waitForSelector('.app-container');

    // Speed up AI and drive to completion
    await page.evaluate(() => {
      const w = window as any;
      if (w.setAISpeedProfile) w.setAISpeedProfile('instant');
    });

    const start = Date.now();
    while (Date.now() - start < 20000) {
      const done = await page.evaluate(() => {
        const w = window as any;
        const overlay = w.getSectionOverlay?.();
        if (overlay && overlay.type === 'oneHand') return true;
        if (w.playFirstAction) w.playFirstAction();
        return false;
      });
      if (done) break;
      await page.waitForTimeout(10);
    }

    // Verify completion modal
    await expect(page.locator('.modal-box')).toBeVisible({ timeout: 10000 });

    // Click New Game twice, verifying theme and URL each time
    for (let i = 0; i < 2; i++) {
      await page.getByRole('button', { name: 'New Game' }).click();
      await page.waitForSelector('.app-container');
      await expect(page).toHaveURL(/h=one_hand/);
      await expect(page).toHaveURL(/t=dark/);
      await page.waitForFunction(() => document.documentElement.getAttribute('data-theme') === 'dark');

      // Drive to next completion modal to bring back the New Game button again
      const start = Date.now();
      while (Date.now() - start < 20000) {
        const done = await page.evaluate(() => {
          const w = window as any;
          const overlay = w.getSectionOverlay?.();
          if (overlay && overlay.type === 'oneHand') return true;
          if (w.playFirstAction) w.playFirstAction();
          return false;
        });
        if (done) break;
        await page.waitForTimeout(10);
      }
      await expect(page.locator('.modal-box')).toBeVisible({ timeout: 10000 });
    }
  });
});
