import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Flash Message on Complete Trick - Dynamic', () => {
  test('shows flash message when dynamically completing a trick', async ({ page }) => {
    // Start with a state where 4 dominoes have been played for the second trick
    // This state has: bid-30, 3 passes, trump-blanks, first trick (4 plays + complete), then 4 plays for second trick
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ4NzMxNTY2MjB9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0cnVtcC1ibGFua3MifSx7ImkiOiIzMiJ9LHsiaSI6IjYzIn0seyJpIjoiMzMifSx7ImkiOiIzMSJ9LHsiaSI6ImNvbXBsZXRlLXRyaWNrIn0seyJpIjoiMjIifSx7ImkiOiI2MiJ9LHsiaSI6IjIxIn0seyJpIjoiNDMifV19');
    
    // Wait for page to load
    await page.waitForTimeout(500);
    
    // The flash message should NOT be visible initially
    const flashMessage = page.locator('.flash-message');
    await expect(flashMessage).not.toBeVisible();
    
    // Now complete the trick - the button should be available since 4 dominoes have been played
    const helper = new PlaywrightGameHelper(page);
    
    // Check if complete-trick action is available
    const actions = await helper.getAvailableActions();
    const completeTrickAction = actions.find(a => a.type === 'complete_trick');
    
    if (!completeTrickAction) {
      // If not available, we might need to wait or the state is wrong
      console.log('Available actions:', actions);
      throw new Error('Complete trick action not available - check test state');
    }
    
    await helper.completeTrick();
    
    // Flash message should appear
    await expect(flashMessage).toBeVisible();
    
    // Get the text and verify it shows a player win
    const text = await flashMessage.textContent();
    expect(text).toMatch(/P[1-4] Wins!/);
    
    // Wait for auto-dismiss after 1 second
    await page.waitForTimeout(1100);
    await expect(flashMessage).not.toBeVisible();
  });
  
  test('flash message can be dismissed by clicking', async ({ page }) => {
    // Start with same state - 4 dominoes played for second trick
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ4NzMxNTY2MjB9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0cnVtcC1ibGFua3MifSx7ImkiOiIzMiJ9LHsiaSI6IjYzIn0seyJpIjoiMzMifSx7ImkiOiIzMSJ9LHsiaSI6ImNvbXBsZXRlLXRyaWNrIn0seyJpIjoiMjIifSx7ImkiOiI2MiJ9LHsiaSI6IjIxIn0seyJpIjoiNDMifV19');
    
    // Wait for page to load
    await page.waitForTimeout(500);
    
    const flashMessage = page.locator('.flash-message');
    
    // Complete the trick
    const helper = new PlaywrightGameHelper(page);
    
    // Check if complete-trick action is available
    const actions = await helper.getAvailableActions();
    const completeTrickAction = actions.find(a => a.type === 'complete_trick');
    
    if (!completeTrickAction) {
      console.log('Available actions:', actions);
      throw new Error('Complete trick action not available - check test state');
    }
    
    await helper.completeTrick();
    
    // Should be visible
    await expect(flashMessage).toBeVisible();
    
    // Click to dismiss
    await flashMessage.click();
    
    // Should disappear immediately
    await expect(flashMessage).not.toBeVisible();
  });
  
  test('clicking anywhere on page dismisses flash message', async ({ page }) => {
    // Start with same state - 4 dominoes played for second trick
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ4NzMxNTY2MjB9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0cnVtcC1ibGFua3MifSx7ImkiOiIzMiJ9LHsiaSI6IjYzIn0seyJpIjoiMzMifSx7ImkiOiIzMSJ9LHsiaSI6ImNvbXBsZXRlLXRyaWNrIn0seyJpIjoiMjIifSx7ImkiOiI2MiJ9LHsiaSI6IjIxIn0seyJpIjoiNDMifV19');
    
    // Wait for page to load
    await page.waitForTimeout(500);
    
    const flashMessage = page.locator('.flash-message');
    
    // Complete the trick
    const helper = new PlaywrightGameHelper(page);
    
    // Check if complete-trick action is available
    const actions = await helper.getAvailableActions();
    const completeTrickAction = actions.find(a => a.type === 'complete_trick');
    
    if (!completeTrickAction) {
      console.log('Available actions:', actions);
      throw new Error('Complete trick action not available - check test state');
    }
    
    await helper.completeTrick();
    
    // Should be visible
    await expect(flashMessage).toBeVisible();
    
    // Click on the header area (anywhere on the page)
    await page.locator('header').first().click();
    
    // Should disappear immediately
    await expect(flashMessage).not.toBeVisible();
  });
});