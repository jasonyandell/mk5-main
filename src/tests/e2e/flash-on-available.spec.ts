import { test, expect } from '@playwright/test';

test.describe('Flash Message When Complete-Trick Becomes Available', () => {
  test('shows flash message when loading state where complete-trick is available', async ({ page }) => {
    // Load URL where 4 dominoes have been played and complete-trick is the next action
    // This state has played: 32d, 63, 33d, 31d (but NOT yet complete-trick)
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ4NzMxNTY2MjB9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0cnVtcC1ibGFua3MifSx7ImkiOiIzMmQifSx7ImkiOiI2MyJ9LHsiaSI6IjMzZCJ9LHsiaSI6IjMxZCJ9XX0');
    
    // Wait for page to load
    await page.waitForTimeout(500);
    
    // Flash message should be visible immediately
    const flashMessage = page.locator('.flash-message');
    await expect(flashMessage).toBeVisible();
    
    // Should show which player wins
    const text = await flashMessage.textContent();
    expect(text).toMatch(/P[1-4] Wins!/);
    
    // Should auto-dismiss after 1 second
    await page.waitForTimeout(1100);
    await expect(flashMessage).not.toBeVisible();
  });
  
  test('shows flash message when 4th domino is played', async ({ page }) => {
    // Load state with 3 dominoes played
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ4NzMxNTY2MjB9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0cnVtcC1ibGFua3MifSx7ImkiOiIzMmQifSx7ImkiOiI2MyJ9LHsiaSI6IjMzZCJ9XX0');
    
    await page.waitForTimeout(500);
    
    // Should not have flash message yet
    const flashMessage = page.locator('.flash-message');
    await expect(flashMessage).not.toBeVisible();
    
    // Switch to actions panel
    await page.locator('[data-testid="nav-actions"]').click();
    await page.waitForTimeout(200);
    
    // Play the 4th domino (31d - 3-1 double)
    const playButton = page.locator('.action-button').filter({ hasText: '3-1' });
    if (await playButton.count() > 0) {
      await playButton.first().click();
    } else {
      // Fallback: try to find any play action
      const anyPlayButton = page.locator('.action-button').first();
      await anyPlayButton.click();
    }
    
    // Flash message should appear now
    await expect(flashMessage).toBeVisible();
    
    // Should show which player wins
    const text = await flashMessage.textContent();
    expect(text).toMatch(/P[1-4] Wins!/);
  });
  
  test('flash message is dismissible by clicking', async ({ page }) => {
    // Load URL where complete-trick is available
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ4NzMxNTY2MjB9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0cnVtcC1ibGFua3MifSx7ImkiOiIzMmQifSx7ImkiOiI2MyJ9LHsiaSI6IjMzZCJ9LHsiaSI6IjMxZCJ9XX0');
    
    await page.waitForTimeout(500);
    
    const flashMessage = page.locator('.flash-message');
    await expect(flashMessage).toBeVisible();
    
    // Click to dismiss
    await flashMessage.click();
    
    // Should disappear immediately
    await expect(flashMessage).not.toBeVisible();
  });
});