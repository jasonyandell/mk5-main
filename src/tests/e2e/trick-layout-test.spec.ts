import { test, expect } from '@playwright/test';

test.describe('Trick Layout Test', () => {
  test('should display tricks horizontally', async ({ page }) => {
    // Navigate to the app
    await page.goto('/');
    
    // Wait for the app to load
    await page.waitForSelector('header', { timeout: 5000 });
    
    // Start the game to get to playing phase
    const bidButton = page.locator('[data-testid="bid-30"]');
    if (await bidButton.isVisible()) {
      // Make some bids to get past bidding phase
      for (let i = 0; i < 4; i++) {
        const passButton = page.locator('[data-testid="pass"]');
        const bid30 = page.locator('[data-testid="bid-30"]');
        
        if (await bid30.isVisible() && i === 0) {
          await bid30.click();
        } else if (await passButton.isVisible()) {
          await passButton.click();
        }
        await page.waitForTimeout(100);
      }
      
      // Select trump if needed
      const trumpButton = page.locator('[data-testid="trump-fives"]').or(page.locator('button:has-text("Fives")'));
      if (await trumpButton.isVisible()) {
        await trumpButton.click();
      }
    }
    
    // Check the trick container layout
    const trickContainer = page.locator('.trick-horizontal');
    await expect(trickContainer).toBeVisible();
    
    // Get all trick positions
    const trickPositions = page.locator('.trick-position');
    const count = await trickPositions.count();
    
    // Should have 4 positions
    expect(count).toBe(4);
    
    // Get the bounding boxes to verify horizontal layout
    if (count >= 2) {
      const firstBox = await trickPositions.nth(0).boundingBox();
      const secondBox = await trickPositions.nth(1).boundingBox();
      
      if (firstBox && secondBox) {
        // In horizontal layout, Y coordinates should be similar
        expect(Math.abs(firstBox.y - secondBox.y)).toBeLessThan(10);
        
        // X coordinates should be different (side by side)
        expect(secondBox.x).toBeGreaterThan(firstBox.x);
        
        console.log('First position:', firstBox);
        console.log('Second position:', secondBox);
      }
    }
    
    // Take a screenshot for visual verification
    await page.screenshot({ path: 'trick-layout.png', fullPage: true });
  });
});