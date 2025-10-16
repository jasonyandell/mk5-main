// TODO: Rewrite for new variant system (old section-runner removed)

// TODO: Rewrite for new variant system (old section-runner removed)
import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe('URL/History compact-consensus encoding', () => {
  test('agree-* are omitted from history state actions; URL reload works', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.goto(999999); // deterministic seed

    // Open settings
    await page.getByRole('button', { name: 'Menu' }).click();
    await page.locator('.settings-btn').click();
    await expect(page.locator('[data-testid="settings-panel"]')).toBeVisible();

    // Start the one-hand section
    await page.getByRole('button', { name: 'Play One Hand' }).click();

    // Drive completion without quickplay
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
    const overlayType = await page.evaluate(() => (window as any).getSectionOverlay?.()?.type);
    expect(overlayType).toBe('oneHand');
    await expect(page.locator('.modal-box')).toBeVisible({ timeout: 5000 });
    // Title indicates outcome
    const title = await page.locator('.modal-box h3').textContent();
    expect(title || '').toMatch(/We Won|We Lost/);

    // Verify compact actions in history state
    const compactState = await page.evaluate(() => {
      const actions = (window.history.state?.actions || []) as string[];
      return {
        hasAgree: actions.some((id) => id.startsWith('agree-')),
        count: actions.length,
      };
    });
    expect(compactState.hasAgree).toBe(false);

    // Reload via current URL and ensure app loads to a terminal phase
    const url = page.url();
    await page.goto(url, { waitUntil: 'networkidle' });
    await expect(page.locator('.app-container')).toBeVisible();
    const phase = await page.locator('.app-container').getAttribute('data-phase');
    expect(['scoring', 'game_end', 'bidding']).toContain(phase || '');
  });
});
