import { test, expect } from '@playwright/test';
import type { TestWindow } from './test-window';

test.setTimeout(30000);

test.describe('Sections: One Hand from URL shows completion modal', () => {
  test('Load /?h=one_hand → completes and shows modal; restart works', async ({ page }) => {
    await page.goto('/?h=one_hand', { waitUntil: 'networkidle' });

    // App should render
    await page.waitForSelector('.app-container');

    // Capture initial seed (after seed is supplied)
    const initialSeed = await page.evaluate(() => (window as unknown as TestWindow).getGameState?.()?.shuffleSeed as number);
    expect(initialSeed).toBeTruthy();

    // Drive completion without quickplay: make AI instant and nudge human turns
    await page.evaluate(() => {
      const w = window as unknown as TestWindow;
      if (w.setAISpeedProfile) w.setAISpeedProfile('instant');
    });
    // Nudge loop: if human turn, play first action; else wait
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
    // Verify overlay exists and modal renders
    const overlayType = await page.evaluate(() => (window as unknown as TestWindow).getSectionOverlay?.()?.type);
    expect(overlayType).toBe('oneHand');
    // Title should be either We Won! or We Lost
    const title = page.locator('.modal-box h3');
    await expect(title).toBeVisible({ timeout: 5000 });

    // Wait for the completion modal to stay visible
    await expect(page.locator('.modal-box')).toBeVisible({ timeout: 20000 });

    // URL should be minimized (no h=), but include seed
    const url = page.url();
    expect(url).toMatch(/\?s=/);
    expect(url).not.toMatch(/h=one_hand/);

    // Retry should re-run same seed and show modal again
    const retryBtn = page.getByRole('button', { name: /Retry \(/ });
    if (await retryBtn.count()) {
      await retryBtn.click();
    } else {
      // If we won, there is no Retry; click New to start again with a new seed
      await page.getByRole('button', { name: 'New' }).click();
    }
    // Nudge loop again until overlay appears
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

    const afterSeed = await page.evaluate(() => (window as unknown as TestWindow).getGameState?.()?.shuffleSeed as number);
    // If we clicked Retry, seed should be same; if New, seed likely different
    if (await retryBtn.count()) {
      expect(afterSeed).toBe(initialSeed);
    } else {
      expect(afterSeed).not.toBe(initialSeed);
    }

    // If Challenge button is present (we won), it should be clickable
    const challengeBtn = page.getByRole('button', { name: 'Challenge!' });
    if (await challengeBtn.count()) {
      await challengeBtn.click();
    }
  });
});
