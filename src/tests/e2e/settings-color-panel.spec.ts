/* global getComputedStyle */
import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/game-helper';

test.describe('Settings Color Panel', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    await helper.goto(12345);
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
    
    // Click on the first color circle (Primary color) to open picker
    const firstColorPicker = page.locator('.color-picker-compact .color-picker').first();
    await firstColorPicker.click();
    
    // Wait for color picker popup to appear
    await expect(page.locator('.picker')).toBeVisible();
    
    // The color picker modifies the color immediately on interaction
    // Use keyboard to navigate and change the value
    await page.keyboard.press('Tab'); // Focus on first slider
    await page.keyboard.press('ArrowLeft'); // Change the value
    await page.keyboard.press('ArrowLeft');
    await page.keyboard.press('ArrowLeft');
    
    // Close the picker
    await page.keyboard.press('Escape');
    
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
    
    // Change primary color
    const firstColorPicker = page.locator('.color-picker-compact .color-picker').first();
    await firstColorPicker.click();
    await expect(page.locator('.picker')).toBeVisible();
    
    // Change color using keyboard
    await page.keyboard.press('Tab');
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('Escape');
    
    // Wait for color override to be in URL
    await page.waitForFunction(() => window.location.href.includes('v='));
    const urlWithColor = page.url();
    expect(urlWithColor).toContain('v=');
    
    // Now open settings panel to change theme
    await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
    await page.locator(PlaywrightGameHelper.SELECTORS.settings.button).click();
    await expect(page.locator(PlaywrightGameHelper.SELECTORS.settings.panel)).toBeVisible();
    
    // Click on a different theme (Dark theme)
    const darkThemeButton = page.locator('[onclick*="updateTheme"][onclick*="dark"]').first();
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
    
    // Change primary color to a distinct value
    const firstColorPicker = page.locator('.color-picker-compact .color-picker').first();
    await firstColorPicker.click();
    await expect(page.locator('.picker')).toBeVisible();
    
    // Change color using keyboard for a distinct value
    await page.keyboard.press('Tab');
    await page.keyboard.press('Home'); // Go to start
    await page.keyboard.press('Escape');
    
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
    
    // Now change the color to a different value
    await firstColorPicker.click();
    await expect(page.locator('.picker')).toBeVisible();
    // Change to different color using keyboard
    await page.keyboard.press('Tab');
    await page.keyboard.press('End'); // Go to end
    await page.keyboard.press('Escape');
    
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
    
    // Change primary color
    const firstColorPicker = page.locator('.color-picker-compact .color-picker').first();
    await firstColorPicker.click();
    await expect(page.locator('.picker')).toBeVisible();
    
    // Change color using keyboard
    await page.keyboard.press('Tab');
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('Escape');
    
    // Wait for color to be applied
    await page.waitForFunction(() => window.location.href.includes('v='));
    
    // Click share button (📤)
    const shareButton = page.locator('.theme-editor-panel button[title="Share theme link"]');
    await shareButton.click();
    
    // Wait for share dialog to appear
    await expect(page.locator('.share-dialog')).toBeVisible();
    
    // Verify the share URL contains color overrides
    const shareUrlElement = page.locator('.share-dialog code');
    const shareUrl = await shareUrlElement.textContent();
    expect(shareUrl).toContain('v=');
    
    // Click copy link button
    const copyLinkButton = page.locator('button:has-text("Copy Link")');
    await copyLinkButton.click();
    
    // Verify button changes to show success
    await expect(page.locator('button:has-text("✓ Copied!")')).toBeVisible();
    
    // Close dialog
    await page.locator('.share-dialog button:has-text("Done")').click();
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
    
    // Change primary color
    const firstColorPicker = page.locator('.color-picker-compact .color-picker').first();
    await firstColorPicker.click();
    await expect(page.locator('.picker')).toBeVisible();
    
    // Change color using keyboard
    await page.keyboard.press('Tab');
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('Escape');
    
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
    
    // Change primary color
    const firstColorPicker = page.locator('.color-picker-compact .color-picker').first();
    await firstColorPicker.click();
    await expect(page.locator('.picker')).toBeVisible();
    
    // Change color using keyboard
    await page.keyboard.press('Tab');
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('Escape');
    
    // Wait for color to be applied
    await page.waitForFunction(() => window.location.href.includes('v='));
    
    // Click share button to open dialog
    const shareButton = page.locator('.theme-editor-panel button[title="Share theme link"]');
    await shareButton.click();
    await expect(page.locator('.share-dialog')).toBeVisible();
    
    // Click Copy CSS button
    const copyCssButton = page.locator('button:has-text("Copy CSS")');
    await copyCssButton.click();
    
    // Verify button changes to show success
    await expect(page.locator('button:has-text("✓ Copied!")')).toBeVisible();
  });

  test('preserves colors across game actions', async ({ page }) => {
    // Set a custom color
    await page.locator('.dropdown-end .btn-ghost.btn-circle').click();
    await page.locator('.theme-colors-btn').click();
    await expect(page.locator('.theme-editor-panel')).toBeVisible();
    
    // Change primary color
    const firstColorPicker = page.locator('.color-picker-compact .color-picker').first();
    await firstColorPicker.click();
    await expect(page.locator('.picker')).toBeVisible();
    
    // Change color using keyboard
    await page.keyboard.press('Tab');
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('Escape');
    
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