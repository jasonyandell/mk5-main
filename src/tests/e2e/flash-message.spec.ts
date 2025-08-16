import { test, expect } from '@playwright/test';

test.describe('Flash Message on Complete Trick', () => {
  test('shows flash message when completing a trick', async ({ page }) => {
    // Load the URL with the specific game state
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ4NzMxNTY2MjB9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0cnVtcC1ibGFua3MifSx7ImkiOiIzMmQifSx7ImkiOiI2MyJ9LHsiaSI6IjMzZCJ9LHsiaSI6IjMxZCJ9LHsiaSI6ImNvbXBsZXRlLXRyaWNrIn0seyJpIjoiMjIifSx7ImkiOiI2MiJ9LHsiaSI6IjIxIn0seyJpIjoiNDMifSx7ImkiOiJjb21wbGV0ZS10cmljayJ9LHsiaSI6IjY1In0seyJpIjoiNjEifSx7ImkiOiIxMCJ9LHsiaSI6IjY2In1dfQ');
    
    // Wait for page to load
    await page.waitForTimeout(500);
    
    // Check if flash message appears after last complete-trick
    const flashMessage = page.locator('.flash-message');
    
    // Since we're loading a state with 2 complete-trick actions already executed,
    // the second one should trigger the flash
    await expect(flashMessage).toBeVisible();
    
    // Get the text and verify it shows a player win
    const text = await flashMessage.textContent();
    expect(text).toMatch(/P[1-4] Wins!/);
    
    // Wait for auto-dismiss after 1 second
    await page.waitForTimeout(1100);
    await expect(flashMessage).not.toBeVisible();
  });
  
  test('flash message is clickable to dismiss', async ({ page }) => {
    // Load the URL with the specific game state
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ4NzMxNTY2MjB9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0cnVtcC1ibGFua3MifSx7ImkiOiIzMmQifSx7ImkiOiI2MyJ9LHsiaSI6IjMzZCJ9LHsiaSI6IjMxZCJ9LHsiaSI6ImNvbXBsZXRlLXRyaWNrIn0seyJpIjoiMjIifSx7ImkiOiI2MiJ9LHsiaSI6IjIxIn0seyJpIjoiNDMifSx7ImkiOiJjb21wbGV0ZS10cmljayJ9LHsiaSI6IjY1In0seyJpIjoiNjEifSx7ImkiOiIxMCJ9LHsiaSI6IjY2In1dfQ');
    
    // Wait for page to load
    await page.waitForTimeout(500);
    
    const flashMessage = page.locator('.flash-message');
    
    // Should be visible
    await expect(flashMessage).toBeVisible();
    
    // Click to dismiss
    await flashMessage.click();
    
    // Should disappear immediately
    await expect(flashMessage).not.toBeVisible();
  });
  
  test('clicking anywhere dismisses flash message', async ({ page }) => {
    // Load the URL with the specific game state
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ4NzMxNTY2MjB9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0cnVtcC1ibGFua3MifSx7ImkiOiIzMmQifSx7ImkiOiI2MyJ9LHsiaSI6IjMzZCJ9LHsiaSI6IjMxZCJ9LHsiaSI6ImNvbXBsZXRlLXRyaWNrIn0seyJpIjoiMjIifSx7ImkiOiI2MiJ9LHsiaSI6IjIxIn0seyJpIjoiNDMifSx7ImkiOiJjb21wbGV0ZS10cmljayJ9LHsiaSI6IjY1In0seyJpIjoiNjEifSx7ImkiOiIxMCJ9LHsiaSI6IjY2In1dfQ');
    
    // Wait for page to load
    await page.waitForTimeout(500);
    
    const flashMessage = page.locator('.flash-message');
    
    // Should be visible
    await expect(flashMessage).toBeVisible();
    
    // Click on the board area
    await page.locator('.game-container').click();
    
    // Should disappear immediately
    await expect(flashMessage).not.toBeVisible();
  });
});