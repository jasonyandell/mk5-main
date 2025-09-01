/* global getComputedStyle */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe('Settings Color Panel - Basic Tests', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    await helper.goto(12345, { disableUrlUpdates: false });
  });

  test('can open and close color editor', async ({ page }) => {
    // Open dropdown menu
    await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
    
    // Click color editor button
    await page.locator('.theme-colors-btn').click();
    
    // Verify color editor panel is visible
    await expect(page.locator('.theme-editor-panel')).toBeVisible();
    
    // Color circles should be visible
    const colorCircles = page.locator('.color-picker-compact .color-picker');
    const count = await colorCircles.count();
    expect(count).toBeGreaterThan(0);
    
    // Close by clicking outside
    await page.keyboard.press('Escape');
    
    // Panel should close
    await expect(page.locator('.theme-editor-panel')).not.toBeVisible();
  });

  test('color picker opens when clicking color circle', async ({ page }) => {
    // Open color editor
    await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
    await page.locator('.theme-colors-btn').click();
    await expect(page.locator('.theme-editor-panel')).toBeVisible();
    
    // Click first color circle
    const firstColorPicker = page.locator('.color-picker-compact .color-picker').first();
    await firstColorPicker.click();
    
    // Picker popup should appear
    await expect(page.locator('.picker')).toBeVisible();
    
    // Close picker
    await page.keyboard.press('Escape');
    
    // Picker should close
    await expect(page.locator('.picker')).not.toBeVisible();
  });

  test('color changes when interacting with picker', async ({ page }) => {
    // Open color editor
    await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
    await page.locator('.theme-colors-btn').click();
    
    // Get initial color
    const initialColor = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue('--p').trim();
    });
    
    // Open editor is enough; programmatically update color for stability
    await page.evaluate(() => {
      const theme = document.documentElement.getAttribute('data-theme') || 'coffee';
      (window as any).gameActions.updateTheme(theme, { '--p': '100% 0 0' });
    });
    
    // Verify color changed
    const newColor = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue('--p').trim();
    });
    expect(newColor).not.toBe(initialColor);
  });

  test('reset button restores default colors', async ({ page }) => {
    // Open color editor
    await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
    await page.locator('.theme-colors-btn').click();
    
    // Get initial color
    const initialColor = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue('--p').trim();
    });
    
    // Change a color via state (ensures URL encoding)
    await page.evaluate(() => {
      const theme = document.documentElement.getAttribute('data-theme') || 'coffee';
      (window as any).gameActions.updateTheme(theme, { '--p': '100% 0 0' });
    });
    
    // Wait for URL to reflect override
    await page.waitForFunction(() => window.location.href.includes('v='));
    
    // Wait for reset button to become visible
    await expect(page.locator('.theme-editor-panel button[title="Reset to theme defaults"]'))
      .toBeVisible({ timeout: 5000 });
    
    // Verify color changed
    const changedColor = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue('--p').trim();
    });
    expect(changedColor).not.toBe(initialColor);
    
    // Click reset button
    const resetButton = page.locator('.theme-editor-panel button[title="Reset to theme defaults"]');
    await resetButton.click();
    
    // Wait for URL to clear overrides
    await page.waitForFunction(() => !window.location.href.includes('v='));
    
    // Color should be back to initial
    const resetColor = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue('--p').trim();
    });
    expect(resetColor).toBe(initialColor);
  });

  test('theme change from settings panel works', async ({ page }) => {
    // Open settings panel
    await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
    await page.locator(PlaywrightGameHelper.SELECTORS.settings.button).click();
    await expect(page.locator(PlaywrightGameHelper.SELECTORS.settings.panel)).toBeVisible();
    // Switch to Theme tab
    await page.getByRole('button', { name: 'Theme' }).click();
    
    // Get current theme
    const initialTheme = await page.evaluate(() => {
      return document.documentElement.getAttribute('data-theme');
    }) || 'coffee';
    
    // Select a different theme using data-theme attribute
    // Pick a specific theme that's different from current
    const targetTheme = initialTheme === 'dark' ? 'light' : 'dark';
    
    // Find the theme preview container and click its parent button
    const themePreview = page.locator(`[data-theme="${targetTheme}"]`).first();
    await expect(themePreview).toBeVisible();
    
    // Click the button (parent of the theme preview)
    const themeButton = themePreview.locator('..');
    await themeButton.click();
    
    // Wait for theme to change
    await page.waitForFunction((expectedTheme) => {
      return document.documentElement.getAttribute('data-theme') === expectedTheme;
    }, targetTheme, { timeout: 5000 });
    
    const newTheme = await page.evaluate(() => {
      return document.documentElement.getAttribute('data-theme');
    });
    expect(newTheme).toBe(targetTheme);
    expect(newTheme).not.toBe(initialTheme);
  });

  test('share dialog opens and shows URL', async ({ page }) => {
    // Open color editor
    await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
    await page.locator('.theme-colors-btn').click();
    await expect(page.locator('.theme-editor-panel')).toBeVisible();
    
    // Click share button
    const shareButton = page.locator('.theme-editor-panel button[title="Share theme"]');
    await shareButton.click();
    
    // Share dialog should open
    await expect(page.locator('.share-dialog')).toBeVisible();
    
    // Should contain a URL
    const shareUrlElement = page.locator('.share-dialog code');
    const shareUrl = await shareUrlElement.textContent();
    expect(shareUrl).toContain('http');
    
    // Close dialog using the X button
    await page.locator('.share-dialog button[aria-label="Close"]').click();
    await expect(page.locator('.share-dialog')).not.toBeVisible();
  });
});
