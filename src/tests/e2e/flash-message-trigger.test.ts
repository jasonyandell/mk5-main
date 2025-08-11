import { test, expect } from '@playwright/test';
import { completeTestTrick } from './helpers/playwrightHelper';

test.describe('Flash Message on Complete Trick - Dynamic', () => {
  test('shows flash message when dynamically completing a trick', async ({ page }) => {
    // Start with a state just before the second complete-trick
    // We'll remove the last complete-trick and the dominoes after it
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ4NzMxNTY2MjB9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0cnVtcC1ibGFua3MifSx7ImkiOiIzMmQifSx7ImkiOiI2MyJ9LHsiaSI6IjMzZCJ9LHsiaSI6IjMxZCJ9LHsiaSI6ImNvbXBsZXRlLXRyaWNrIn0seyJpIjoiMjIifSx7ImkiOiI2MiJ9LHsiaSI6IjIxIn0seyJpIjoiNDMifV19');
    
    // Wait for page to load
    await page.waitForTimeout(500);
    
    // The flash message should NOT be visible initially
    const flashMessage = page.locator('.flash-message');
    await expect(flashMessage).not.toBeVisible();
    
    // Now complete the trick
    await completeTestTrick(page);
    
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
    // Start with a state just before complete-trick
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ4NzMxNTY2MjB9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0cnVtcC1ibGFua3MifSx7ImkiOiIzMmQifSx7ImkiOiI2MyJ9LHsiaSI6IjMzZCJ9LHsiaSI6IjMxZCJ9LHsiaSI6ImNvbXBsZXRlLXRyaWNrIn0seyJpIjoiMjIifSx7ImkiOiI2MiJ9LHsiaSI6IjIxIn0seyJpIjoiNDMifV19');
    
    // Wait for page to load
    await page.waitForTimeout(500);
    
    const flashMessage = page.locator('.flash-message');
    
    // Complete the trick
    await completeTestTrick(page);
    
    // Should be visible
    await expect(flashMessage).toBeVisible();
    
    // Click to dismiss
    await flashMessage.click();
    
    // Should disappear immediately
    await expect(flashMessage).not.toBeVisible();
  });
  
  test('clicking anywhere on page dismisses flash message', async ({ page }) => {
    // Start with a state just before complete-trick
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ4NzMxNTY2MjB9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0cnVtcC1ibGFua3MifSx7ImkiOiIzMmQifSx7ImkiOiI2MyJ9LHsiaSI6IjMzZCJ9LHsiaSI6IjMxZCJ9LHsiaSI6ImNvbXBsZXRlLXRyaWNrIn0seyJpIjoiMjIifSx7ImkiOiI2MiJ9LHsiaSI6IjIxIn0seyJpIjoiNDMifV19');
    
    // Wait for page to load
    await page.waitForTimeout(500);
    
    const flashMessage = page.locator('.flash-message');
    
    // Complete the trick
    await completeTestTrick(page);
    
    // Should be visible
    await expect(flashMessage).toBeVisible();
    
    // Click on the header area (anywhere on the page)
    await page.locator('header').first().click();
    
    // Should disappear immediately
    await expect(flashMessage).not.toBeVisible();
  });
});