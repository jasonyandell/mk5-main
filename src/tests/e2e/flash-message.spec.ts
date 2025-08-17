import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Flash Message on Complete Trick', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
  });

  test('shows flash message when completing a trick', async ({ page }) => {
    // Load state with trump selected and in playing phase
    await helper.loadStateWithActions(12345, [
      '30', 'p', 'p', 'p', 'trump-blanks'
    ]);
    
    // Wait for page to load and playing phase
    await page.waitForTimeout(200);
    
    // Play 4 dominoes to complete a trick
    for (let i = 0; i < 4; i++) {
      await helper.playAnyDomino();
      await page.waitForTimeout(50);
    }
    
    // Now complete the trick to see the flash message
    await helper.completeTrick();
    
    // Check if flash message appears
    const flashMessage = helper.getFlashMessageLocator();
    await expect(flashMessage).toBeVisible();
    
    // Get the text and verify it shows a player win
    const text = await flashMessage.textContent();
    expect(text).toMatch(/P[1-4] Wins!/);
    
    // Wait for auto-dismiss after 1 second
    await page.waitForTimeout(1100);
    await expect(flashMessage).not.toBeVisible();
  });
  
  test('flash message is clickable to dismiss', async ({ page }) => {
    // Load state with trump selected and in playing phase
    await helper.loadStateWithActions(12345, [
      '30', 'p', 'p', 'p', 'trump-blanks'
    ]);
    
    // Wait for page to load
    await page.waitForTimeout(200);
    
    // Play 4 dominoes then complete trick
    for (let i = 0; i < 4; i++) {
      await helper.playAnyDomino();
      await page.waitForTimeout(50);
    }
    
    // Complete the trick to show flash message
    await helper.completeTrick();
    
    const flashMessage = helper.getFlashMessageLocator();
    
    // Should be visible
    await expect(flashMessage).toBeVisible();
    
    // Click to dismiss
    await flashMessage.click();
    
    // Should disappear immediately
    await expect(flashMessage).not.toBeVisible();
  });
  
  test('clicking anywhere dismisses flash message', async ({ page }) => {
    // Load state with trump selected and in playing phase
    await helper.loadStateWithActions(12345, [
      '30', 'p', 'p', 'p', 'trump-blanks'
    ]);
    
    // Wait for page to load
    await page.waitForTimeout(200);
    
    // Play 4 dominoes then complete trick
    for (let i = 0; i < 4; i++) {
      await helper.playAnyDomino();
      await page.waitForTimeout(50);
    }
    
    // Complete the trick to show flash message
    await helper.completeTrick();
    
    const flashMessage = helper.getFlashMessageLocator();
    
    // Should be visible
    await expect(flashMessage).toBeVisible();
    
    // Click anywhere on the page (using the app container)
    await page.locator('.app-container').click({ position: { x: 100, y: 100 } });
    
    // Should disappear immediately
    await expect(flashMessage).not.toBeVisible();
  });
});