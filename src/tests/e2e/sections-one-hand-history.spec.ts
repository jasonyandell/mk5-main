// TODO: Rewrite for new variant system (old section-runner removed)

import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe('Sections: One Hand history stays small and terminal', () => {
  test('start one hand -> take a random action -> reach terminal -> history < 100 and stable', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.goto(777777); // deterministic seed

    // Open settings panel
    await page.getByRole('button', { name: 'Menu' }).click();
    await page.locator('.settings-btn').click();
    await expect(page.locator('[data-testid="settings-panel"]')).toBeVisible();

    // Start the one-hand section
    await page.getByRole('button', { name: 'Play One Hand' }).click();

    // Enable quickplay for all players to drive the hand to completion
    await page.evaluate(() => {
      const w = window as unknown as {
        quickplayActions?: { toggle: () => void; togglePlayer: (id: number) => void };
        getQuickplayState?: () => { enabled: boolean; aiPlayers: Set<number> };
      };
      if (!w.quickplayActions || !w.getQuickplayState) return;
      const state = w.getQuickplayState();
      for (let i = 0; i < 4; i++) {
        if (!state.aiPlayers.has(i)) w.quickplayActions.togglePlayer(i);
      }
      if (!state.enabled) w.quickplayActions.toggle();
    });

    // Wait for completion modal (terminal state)
    await expect(page.locator('.modal-box')).toBeVisible({ timeout: 15000 });
    // Title indicates outcome
    const title = await page.locator('.modal-box h3').textContent();
    expect(title || '').toMatch(/We Won|We Lost/);

    // Verify action history directly via window state to avoid heavy UI rendering
    const count = await page.evaluate(() => {
      const w = window as any;
      return w.getGameState?.()?.actionHistory?.length as number;
    });
    expect(count).toBeLessThan(100);

    // Ensure terminal phase
    const phase = await page.locator('.app-container').getAttribute('data-phase');
    expect(['scoring', 'game_end']).toContain(phase);
  });
});
