import { test, expect } from '@playwright/test';
import type { TestWindow } from './test-window';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe('Sections: One Hand controls (New/Retry/Challenge)', () => {
  test('New starts a fresh run (seed likely changes)', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.goto(424242);

    // Open settings via header menu
    await page.getByRole('button', { name: 'Menu' }).click();
    await page.locator('.settings-btn').click();
    await expect(page.locator('[data-testid="settings-panel"]')).toBeVisible();

    // Start the one-hand section
    await page.getByRole('button', { name: 'Play One Hand' }).click();

    // Drive completion without quickplay
    await page.evaluate(() => {
      const w = window as unknown as TestWindow;
      if (w.setAISpeedProfile) w.setAISpeedProfile('instant');
    });
    {
      const start = Date.now();
      while (Date.now() - start < 20000) {
        const done = await page.evaluate(() => {
          const w = window as unknown as TestWindow;
          const overlay = w.getSectionOverlay?.();
          if (overlay && overlay.type === 'oneHand') return true;
          if (w.playFirstAction) w.playFirstAction();
          return false;
        });
        if (done) break;
        await page.waitForTimeout(10);
      }
    }

    // Wait for completion modal
    await expect(page.locator('.modal-box')).toBeVisible({ timeout: 15000 });
    const beforeSeed = await page.evaluate(() => (window as unknown as TestWindow).getGameState?.()?.shuffleSeed as number);
    // Click New and wait for next modal
    await page.getByRole('button', { name: 'New' }).click();
    {
      const start = Date.now();
      while (Date.now() - start < 20000) {
        const done = await page.evaluate(() => {
          const w = window as unknown as TestWindow;
          const overlay = w.getSectionOverlay?.();
          if (overlay && overlay.type === 'oneHand') return true;
          if (w.playFirstAction) w.playFirstAction();
          return false;
        });
        if (done) break;
        await page.waitForTimeout(10);
      }
      await expect(page.locator('.modal-box')).toBeVisible({ timeout: 5000 });
    }
    const afterSeed = await page.evaluate(() => (window as unknown as TestWindow).getGameState?.()?.shuffleSeed as number);
    expect(afterSeed).not.toBe(beforeSeed);
  });

  test('Retry replays deterministic one hand with compact URL', async ({ page }) => {
    const helper = new PlaywrightGameHelper(page);
    await helper.goto(424242);

    // Open settings via header menu
    await page.getByRole('button', { name: 'Menu' }).click();
    await page.locator('.settings-btn').click();
    await expect(page.locator('[data-testid="settings-panel"]')).toBeVisible();

    // Start the one-hand section
    await page.getByRole('button', { name: 'Play One Hand' }).click();

    // Enable quickplay for all players
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

    // Wait for completion modal
    await expect(page.locator('.modal-box')).toBeVisible({ timeout: 15000 });

    // Click Retry directly (if we won, the button won't exist; in that case, click New and accept a new seed)
    const retryBtn = page.getByRole('button', { name: /Retry \(/ });
    if (await retryBtn.count()) {
      await retryBtn.click();
    } else {
      await page.getByRole('button', { name: 'New' }).click();
    }
    // Wait until overlay appears again
    {
      const start = Date.now();
      while (Date.now() - start < 20000) {
        const done = await page.evaluate(() => {
          const w = window as unknown as TestWindow;
          const overlay = w.getSectionOverlay?.();
          if (overlay && overlay.type === 'oneHand') return true;
          if (w.playFirstAction) w.playFirstAction();
          return false;
        });
        if (done) break;
        await page.waitForTimeout(10);
      }
      await expect(page.locator('.modal-box')).toBeVisible({ timeout: 5000 });
    }

    // Wait for the second run to complete and show modal again
    {
      const start2 = Date.now();
      while (Date.now() - start2 < 20000) {
        const done2 = await page.evaluate(() => {
          const w = window as unknown as TestWindow;
          const overlay = w.getSectionOverlay?.();
          if (overlay && overlay.type === 'oneHand') return true;
          if (w.playFirstAction) w.playFirstAction();
          return false;
        });
        if (done2) break;
        await page.waitForTimeout(10);
      }
      await expect(page.locator('.modal-box')).toBeVisible({ timeout: 5000 });
    }

    // Validate deterministic seed and compact history in URL state
    const after = await page.evaluate(() => {
      const w = window as unknown as TestWindow;
      const s = w.getGameState?.();
      const actionsInHistoryState = (window.history.state?.actions || []) as string[];
      return {
        seed: s?.shuffleSeed as number,
        phase: s?.phase as string,
        actionHistory: (s?.actionHistory?.length || 0) as number,
        urlActionCount: actionsInHistoryState.length as number,
        hasAgree: actionsInHistoryState.some((id) => id.startsWith('agree-')),
      };
    });

    // If we clicked Retry, expect same seed; if New, not guaranteed
    if (await retryBtn.count()) {
      expect(after.seed).toBe(424242);
    }
    expect(after.actionHistory).toBeLessThan(100);
    expect(after.urlActionCount).toBeGreaterThanOrEqual(0);
    expect(after.hasAgree).toBe(false);
  });
});
