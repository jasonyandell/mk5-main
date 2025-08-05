import { test, expect } from '@playwright/test';

test.describe('Domino Highlighting Test', () => {
  test('mouseover highlighting during bidding and trump selection', async ({ page }) => {
    // Navigate to the app
    await page.goto('/');
    
    // Wait for any element to ensure page loaded
    await page.waitForTimeout(1000);
    
    // Click on Actions tab - use force in case of overlay
    await page.click('[data-testid="nav-actions"]', { force: true });
    
    // Wait a bit for transition
    await page.waitForTimeout(500);
    
    // === TEST 1: Bidding Phase - Color Highlighting ===
    console.log('Testing bidding phase highlighting...');
    
    // Check that we have a hand section
    const handSection = page.locator('.hand-section');
    const handVisible = await handSection.isVisible().catch(() => false);
    
    if (handVisible) {
      // Find any domino in the hand
      const dominoes = page.locator('.hand-section .domino');
      const dominoCount = await dominoes.count();
      console.log(`Found ${dominoCount} dominoes in hand`);
      
      if (dominoCount > 0) {
        // Hover over first domino
        await dominoes.first().hover();
        await page.waitForTimeout(200);
        
        // Check for colored highlighting (should have suit-specific class)
        const coloredHighlights = await page.locator('.domino[class*="highlight-suit-"]').count();
        console.log(`Found ${coloredHighlights} dominoes with colored highlighting`);
        
        // Check for suit badge
        const badgeVisible = await page.locator('.suit-highlight-badge').isVisible().catch(() => false);
        console.log(`Suit badge visible: ${badgeVisible}`);
        
        // Move mouse away
        await page.mouse.move(0, 0);
        await page.waitForTimeout(200);
      }
    }
    
    // === TEST 2: Trump Selection - Generic Highlighting ===
    console.log('\nTesting trump selection highlighting...');
    
    // Try to make a bid
    const bid30 = page.locator('[data-testid="bid-30"]');
    const bidVisible = await bid30.isVisible().catch(() => false);
    
    if (bidVisible) {
      await bid30.click();
      await page.waitForTimeout(500);
      
      // Check if we're in trump selection
      const trumpTextVisible = await page.locator('text=Select Trump').isVisible().catch(() => false);
      console.log(`Trump selection visible: ${trumpTextVisible}`);
      
      if (trumpTextVisible) {
        // Find a trump button
        const trumpButtons = page.locator('.trump-button');
        const trumpButtonCount = await trumpButtons.count();
        console.log(`Found ${trumpButtonCount} trump buttons`);
        
        if (trumpButtonCount > 0) {
          // Hover over first trump button
          await trumpButtons.first().hover();
          await page.waitForTimeout(200);
          
          // Check for generic highlighting
          const genericHighlights = await page.locator('.domino.highlight-primary').count();
          const coloredHighlights = await page.locator('.domino[class*="highlight-suit-"]').count();
          
          console.log(`Generic highlights: ${genericHighlights}, Colored highlights: ${coloredHighlights}`);
          
          // Check trump button hover state
          const hoveringCount = await page.locator('.trump-button.trump-hovering').count();
          console.log(`Trump buttons with hover state: ${hoveringCount}`);
        }
      }
    }
    
    // Take screenshot for debugging
    await page.screenshot({ path: 'test-results/highlighting-final.png', fullPage: true });
    console.log('\nTest completed!');
  });
});