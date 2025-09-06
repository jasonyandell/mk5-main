import { test, expect } from '@playwright/test';
import type { TestWindow } from './test-window';

test.describe('Sections: Load from URL slug', () => {
  test('h=one_hand starts section and persists in URL (no quickplay)', async ({ page }) => {
    // Case 1: No seed supplied, app should supply one and remain in playing
    const urlNoSeed = '/?h=one_hand&testMode=true';
    await page.goto(urlNoSeed, { waitUntil: 'networkidle' });
    await page.waitForSelector('.app-container');
    await expect(page).toHaveURL(/h=one_hand/);
    await page.waitForTimeout(300);
    await expect(page).toHaveURL(/h=one_hand/);
    const phaseNoSeed = await page.locator('.app-container').getAttribute('data-phase');
    expect(phaseNoSeed).toBe('playing');

    // Case 2: With seed supplied
    const url = '/?s=93ci&h=one_hand&testMode=true';
    await page.goto(url, { waitUntil: 'networkidle' });

    // App container should appear
    await page.waitForSelector('.app-container');

    // Immediately verify URL contains scenario
    await expect(page).toHaveURL(/h=one_hand/);

    // Verify we are at a valid game phase and no crash
    const phase = await page.locator('.app-container').getAttribute('data-phase');
    expect(phase).toBeTruthy();

    // Wait briefly to ensure section is not immediately cleared
    await page.waitForTimeout(300);
    await expect(page).toHaveURL(/h=one_hand/);
    // Phase should still be playing with the prepared one hand state
    const phaseAfter = await page.locator('.app-container').getAttribute('data-phase');
    expect(phaseAfter).toBe('playing');

    // Ensure we did not silently navigate away or reset
    const state = await page.evaluate(() => (window as unknown as TestWindow).getGameState?.());
    expect(state).toBeTruthy();
    expect(state?.shuffleSeed).toBe(parseInt('93ci', 36));
  });

  test('h=one_hand with t=dark applies theme and keeps it during setup', async ({ page }) => {
    await page.goto('/?h=one_hand&t=dark&testMode=true', { waitUntil: 'networkidle' });
    await page.waitForSelector('.app-container');

    // Should keep scenario param present for a bit and apply theme
    await expect(page).toHaveURL(/h=one_hand/);
    await expect(page).toHaveURL(/t=dark/);
    await page.waitForFunction(() => document.documentElement.getAttribute('data-theme') === 'dark');

    // Phase should settle to playing for prepared one-hand
    const phase = await page.locator('.app-container').getAttribute('data-phase');
    expect(phase).toBe('playing');
  });
});
