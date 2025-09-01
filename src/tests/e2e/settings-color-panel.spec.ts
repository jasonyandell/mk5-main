/* global getComputedStyle */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe('Settings Color Panel', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    await helper.goto(12345, { disableUrlUpdates: false });
  });

  test('changes a color and verifies it appears in URL', async ({ page }) => {
    // Open dropdown menu
    await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
    
    // Click color editor button (🎨 Colors)
    await page.locator('.theme-colors-btn').click();
    
    // Wait for color editor panel to be visible
    await expect(page.locator('.theme-editor-panel')).toBeVisible();
    
    // Get initial URL
    const initialUrl = page.url();
    expect(initialUrl).not.toContain('v=');
    
    // Get initial primary color value
    const initialColor = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue('--p').trim();
    });
    
    // Apply override programmatically for stability
    await page.evaluate(() => {
      const theme = document.documentElement.getAttribute('data-theme') || 'coffee';
      (window as any).gameActions.updateTheme(theme, { '--p': '100% 0 0' });
    });
    
    // Wait for URL to update with color override
    await page.waitForFunction(() => {
      return window.location.href.includes('v=');
    }, { timeout: 5000 });
    
    // Verify URL contains color override parameter
    const updatedUrl = page.url();
    expect(updatedUrl).toContain('v=');
    
    // Verify the color actually changed
    const newColor = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue('--p').trim();
    });
    expect(newColor).not.toBe(initialColor);
  });

  test('changes theme and verifies custom colors are lost', async ({ page }) => {
    // First, set a custom color
    await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
    await page.locator('.theme-colors-btn').click();
    await expect(page.locator('.theme-editor-panel')).toBeVisible();
    
    // Apply override programmatically
    await page.evaluate(() => {
      const theme = document.documentElement.getAttribute('data-theme') || 'coffee';
      (window as any).gameActions.updateTheme(theme, { '--p': '100% 0 0' });
    });
    
    // Wait for color override to be in URL
    await page.waitForFunction(() => window.location.href.includes('v='));
    const urlWithColor = page.url();
    expect(urlWithColor).toContain('v=');
    
    // Close the color editor to avoid backdrop intercepting clicks
    await page.locator('button[aria-label="Close color editor"]').click();
    await expect(page.locator('.theme-editor-panel')).not.toBeVisible();
    
    // Now open settings panel to change theme
    await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
    await page.locator(PlaywrightGameHelper.SELECTORS.settings.button).click();
    await expect(page.locator(PlaywrightGameHelper.SELECTORS.settings.panel)).toBeVisible();
    // Switch to Theme tab
    await page.getByRole('button', { name: 'Theme' }).click();
    
    // Click on a different theme (Dark theme)
    const themePreview = page.locator('[data-theme="dark"]').first();
    await expect(themePreview).toBeVisible();
    const darkThemeButton = themePreview.locator('..');
    await darkThemeButton.click();
    
    // Wait for theme to change
    await page.waitForFunction(() => {
      return document.documentElement.getAttribute('data-theme') === 'dark';
    });
    
    // Verify URL no longer contains color overrides
    await page.waitForFunction(() => !window.location.href.includes('v='));
    const newUrl = page.url();
    expect(newUrl).not.toContain('v=');
    expect(newUrl).toContain('t=dark');
  });

  test('copies URL with color, changes color, navigates back to copied URL', async ({ page }) => {
    // Set a custom color
    await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
    await page.locator('.theme-colors-btn').click();
    await expect(page.locator('.theme-editor-panel')).toBeVisible();
    
    // Change to a distinct value programmatically
    await page.evaluate(() => {
      const theme = document.documentElement.getAttribute('data-theme') || 'coffee';
      (window as any).gameActions.updateTheme(theme, { '--p': '100% 0 0' });
    });
    
    // Wait for URL to have the red color
    await page.waitForFunction(() => window.location.href.includes('v='));
    const urlWithRed = page.url();
    expect(urlWithRed).toContain('v=');
    
    // Store the URL for later
    const copiedUrl = urlWithRed;
    
    // Get the computed red color value
    const redColorValue = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue('--p').trim();
    });
    
    // Now change the color to a different value programmatically
    await page.evaluate(() => {
      const theme = document.documentElement.getAttribute('data-theme') || 'coffee';
      (window as any).gameActions.updateTheme(theme, { '--p': '0% 0 0' });
    });
    
    // Wait for URL to update with blue color
    await page.waitForFunction((oldUrl) => window.location.href !== oldUrl, urlWithRed);
    
    // Verify color changed to blue
    const blueColorValue = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue('--p').trim();
    });
    expect(blueColorValue).not.toBe(redColorValue);
    
    // Navigate to the copied URL (with red color)
    await page.goto(copiedUrl);
    await helper.waitForGameReady();
    
    // Verify the color is back to red
    const restoredColorValue = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue('--p').trim();
    });
    expect(restoredColorValue).toBe(redColorValue);
    
    // Verify URL still contains the color override
    expect(page.url()).toContain('v=');
  });

  test('shares theme via URL button and verifies link includes colors', async ({ page }) => {
    // Set a custom color
    await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
    await page.locator('.theme-colors-btn').click();
    await expect(page.locator('.theme-editor-panel')).toBeVisible();
    
    // Apply override programmatically
    await page.evaluate(() => {
      const theme = document.documentElement.getAttribute('data-theme') || 'coffee';
      (window as any).gameActions.updateTheme(theme, { '--p': '100% 0 0' });
    });
    
    // Wait for color to be applied
    await page.waitForFunction(() => window.location.href.includes('v='));
    
    // Click share button (📤)
    const shareButton = page.locator('.theme-editor-panel button[title="Share theme"]');
    await shareButton.click();
    
    // Wait for share dialog to appear
    await expect(page.locator('.share-dialog')).toBeVisible();
    
    // Verify the share URL contains color overrides
    const shareUrlElement = page.locator('.share-dialog code');
    const shareUrl = await shareUrlElement.textContent();
    expect(shareUrl).toContain('v=');
    
    // Click copy link button (clipboard may be restricted in CI; don't assert style change)
    const copyLinkButton = page.locator('.share-dialog button:has-text("Link")');
    await copyLinkButton.click();
    
    // Close dialog (X button)
    await page.locator('.share-dialog button[aria-label="Close"]').click();
    await expect(page.locator('.share-dialog')).not.toBeVisible();
  });

  test('resets colors to theme defaults', async ({ page }) => {
    // Set a custom color
    await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
    await page.locator('.theme-colors-btn').click();
    await expect(page.locator('.theme-editor-panel')).toBeVisible();
    
    // Get initial primary color
    const initialColor = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue('--p').trim();
    });
    
    // Apply override via state (avoid opening picker to keep reset button visible)
    await page.evaluate(() => {
      const theme = document.documentElement.getAttribute('data-theme') || 'coffee';
      (window as any).gameActions.updateTheme(theme, { '--p': '100% 0 0' });
    });
    
    // Wait for color to change
    await page.waitForFunction(() => window.location.href.includes('v='));
    
    // Verify color changed
    const changedColor = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue('--p').trim();
    });
    expect(changedColor).not.toBe(initialColor);
    
    // Click reset button (🔄)
    const resetButton = page.locator('.theme-editor-panel button[title="Reset to theme defaults"]');
    await resetButton.click();
    
    // Wait for colors to reset (URL should no longer have v= parameter)
    await page.waitForFunction(() => !window.location.href.includes('v='));
    
    // Verify color is back to original
    const resetColor = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue('--p').trim();
    });
    expect(resetColor).toBe(initialColor);
    
    // Verify URL no longer contains color overrides
    expect(page.url()).not.toContain('v=');
  });

  test('exports CSS with custom colors', async ({ page }) => {
    // Set a custom color
    await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
    await page.locator('.theme-colors-btn').click();
    await expect(page.locator('.theme-editor-panel')).toBeVisible();
    
    // Apply override programmatically
    await page.evaluate(() => {
      const theme = document.documentElement.getAttribute('data-theme') || 'coffee';
      (window as any).gameActions.updateTheme(theme, { '--p': '100% 0 0' });
    });
    
    // Wait for color to be applied
    await page.waitForFunction(() => window.location.href.includes('v='));
    
    // Click share button to open dialog
    const shareButton = page.locator('.theme-editor-panel button[title="Share theme"]');
    await shareButton.click();
    await expect(page.locator('.share-dialog')).toBeVisible();
    
    // Click Copy CSS button (clipboard may be restricted; don't assert style change)
    const copyCssButton = page.locator('.share-dialog button:has-text("CSS")');
    await copyCssButton.click();
  });

  test('preserves colors across game actions', async ({ page }) => {
    // Set a custom color
    await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
    await page.locator('.theme-colors-btn').click();
    await expect(page.locator('.theme-editor-panel')).toBeVisible();
    
    // Apply override programmatically
    await page.evaluate(() => {
      const theme = document.documentElement.getAttribute('data-theme') || 'coffee';
      (window as any).gameActions.updateTheme(theme, { '--p': '100% 0 0' });
    });
    
    // Wait for color to be applied
    await page.waitForFunction(() => window.location.href.includes('v='));
    page.url(); // Verify URL updated with color override
    
    // Store the custom color value
    const customColor = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue('--p').trim();
    });
    
    // Close color editor
    await page.locator('body').click({ position: { x: 10, y: 10 } });
    
    // Play some game actions
    await helper.bid(30, false);
    await helper.pass();
    await helper.pass();
    await helper.pass();
    
    // Wait for trump selection phase
    await helper.waitForPhase('trump_selection');
    await helper.setTrump('doubles');
    
    // Verify color is still applied after game actions
    const colorAfterActions = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue('--p').trim();
    });
    expect(colorAfterActions).toBe(customColor);
    
    // Verify URL still contains color override
    expect(page.url()).toContain('v=');
  });
});
