import { test, expect } from '@playwright/test';
import type { TestWindow } from './test-window';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe('Sections: One Hand without Quickplay (main loop only)', () => {
  test('start one hand -> make one play -> reach terminal -> small history', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    // Do not use testMode, so the main loop runs
    await helper.goto(24680, { testMode: false });

    // Open settings panel and start one hand
    await page.getByRole('button', { name: 'Menu' }).click();
    await page.locator('.settings-btn').click();
    await page.getByRole('button', { name: 'Play One Hand' }).click();
    // Close settings (optional)
    await page.locator('[data-testid="settings-close-button"]').click();

    // Speed up AI to avoid long waits
    await page.evaluate(() => {
      const w = window as unknown as TestWindow;
      if (w.setAISpeedProfile) w.setAISpeedProfile('instant');
    });

    // Drive progression without quickplay: speed AI and keep attempting first play until complete
    await page.waitForFunction(() => {
      const w = window as unknown as TestWindow;
      if (w.setAISpeedProfile) w.setAISpeedProfile('instant');
      if (w.playFirstAction) w.playFirstAction();
      const m = Array.from(document.querySelectorAll('.modal .modal-box h3'))
        .some(el => /We Won|We Lost/.test(el.textContent || ''));
      return m;
    }, { timeout: 20000, polling: 200 });

    // Verify action history is small and terminal phase
    const count = await page.evaluate(() => (window as unknown as TestWindow).getGameState?.()?.actionHistory?.length as number);
    expect(count).toBeLessThan(100);
    const phase = await page.locator('.app-container').getAttribute('data-phase');
    expect(['scoring', 'game_end']).toContain(phase);
  });
});
