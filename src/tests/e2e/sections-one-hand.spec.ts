import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe('Sections: Play One Hand', () => {
  test('runs to scoring or game end when triggered from settings', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.goto(424242); // deterministic seed

    // Open settings panel (open menu, then settings)
    await page.getByRole('button', { name: 'Menu' }).click();
    await page.locator('.settings-btn').click();
    await expect(page.locator('[data-testid="settings-panel"]')).toBeVisible();

    // Click the Play One Hand button
    await page.getByRole('button', { name: 'Play One Hand' }).click();

    // Enable quickplay for all players to drive transitions automatically
    await page.evaluate(() => {
      const w = window as unknown as {
        quickplayActions?: { toggle: () => void; togglePlayer: (id: number) => void };
        getQuickplayState?: () => { enabled: boolean; aiPlayers: Set<number> };
      };
      if (!w.quickplayActions || !w.getQuickplayState) return;
      const state = w.getQuickplayState();
      // Ensure all players are AI-controlled in quickplay
      for (let i = 0; i < 4; i++) {
        if (!state.aiPlayers.has(i)) w.quickplayActions.togglePlayer(i);
      }
      if (!state.enabled) w.quickplayActions.toggle();
    });

    // Wait until the section reaches scoring or game_end
    await page.waitForFunction(() => {
      const el = document.querySelector('.app-container');
      if (!el) return false;
      const phase = el.getAttribute('data-phase');
      return phase === 'scoring' || phase === 'game_end';
    }, { timeout: 15000 });

    const phase = await page.locator('.app-container').getAttribute('data-phase');
    expect(['scoring', 'game_end']).toContain(phase);
  });
});
