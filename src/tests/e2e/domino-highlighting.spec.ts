import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Domino Highlighting During Bidding and Trump Selection', () => {
  test.setTimeout(10000);
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    await helper.goto();
    await helper.startNewGame();
  });

  test('should highlight dominoes by suit color during bidding phase', async ({ page }) => {
    // Navigate to Actions panel during bidding
    await page.click('[data-testid="nav-actions"]');
    
    // Wait for hand section to be visible
    await expect(page.locator('.hand-section')).toBeVisible();
    
    // Get all dominoes in hand
    const dominoes = page.locator('.hand-section .domino');
    const dominoCount = await dominoes.count();
    
    // Test hovering over a specific domino (e.g., 3-5)
    const domino35 = page.locator('[data-testid="domino-5-3"]').first();
    if (await domino35.count() > 0) {
      await domino35.hover();
      
      // Check that dominoes with 3s are highlighted
      const highlightedDominoes = page.locator('.domino.highlight-suit-3, .domino.highlight-primary');
      await expect(highlightedDominoes).toHaveCount({ min: 1 });
      
      // Check that the suit badge appears with correct color
      const suitBadge = page.locator('.suit-highlight-badge');
      await expect(suitBadge).toBeVisible();
      await expect(suitBadge).toHaveClass(/suit-badge-3/);
      await expect(suitBadge).toContainText('Threes');
    }
    
    // Test hovering over a double (e.g., 4-4)
    const domino44 = page.locator('[data-testid="domino-4-4"]').first();
    if (await domino44.count() > 0) {
      await domino44.hover();
      
      // Check that all doubles are highlighted as primary
      const doubleHighlights = page.locator('.domino.highlight-doubles, .domino.highlight-primary');
      await expect(doubleHighlights).toHaveCount({ min: 1 });
      
      // Check that dominoes with 4s are highlighted as secondary
      const fourHighlights = page.locator('.domino.highlight-suit-4, .domino.highlight-secondary');
      await expect(fourHighlights).toHaveCount({ min: 1 });
      
      // Check badge shows "Doubles & Fours"
      const suitBadge = page.locator('.suit-highlight-badge');
      await expect(suitBadge).toBeVisible();
      await expect(suitBadge).toHaveClass(/suit-badge-doubles/);
      await expect(suitBadge).toContainText('Doubles & Fours');
    }
    
    // Move mouse away to clear highlighting
    await page.mouse.move(0, 0);
    
    // Verify all highlighting is removed
    await expect(page.locator('.domino.highlight-primary')).toHaveCount(0);
    await expect(page.locator('.domino.highlight-secondary')).toHaveCount(0);
    await expect(page.locator('.suit-highlight-badge')).not.toBeVisible();
  });

  test('should show generic highlighting during trump selection', async ({ page }) => {
    // First, make a bid to get to trump selection
    await helper.makeBid(30);
    
    // Wait for trump selection phase
    await expect(page.locator('text=Select Trump')).toBeVisible({ timeout: 5000 });
    
    // Hover over "Threes" trump button
    const threesButton = page.locator('button:has-text("Threes")');
    await threesButton.hover();
    
    // Check that dominoes with 3s are highlighted (but not with suit-specific color)
    const highlightedDominoes = page.locator('.domino.highlight-primary');
    await expect(highlightedDominoes).toHaveCount({ min: 1 });
    
    // Verify NO suit-specific coloring during trump selection
    await expect(page.locator('.domino.highlight-suit-3')).toHaveCount(0);
    
    // Check that the trump button has hover effect
    await expect(threesButton).toHaveClass(/trump-hovering/);
    
    // Hover over "Doubles" trump button
    const doublesButton = page.locator('button:has-text("Doubles")');
    await doublesButton.hover();
    
    // Check that only doubles are highlighted
    const doubleHighlights = page.locator('.domino.highlight-primary');
    const allDoubles = [];
    for (let i = 0; i <= 6; i++) {
      allDoubles.push(`[data-testid="domino-${i}-${i}"]`);
    }
    
    // Count how many doubles we have in hand
    let doubleCount = 0;
    for (const selector of allDoubles) {
      if (await page.locator(selector).count() > 0) {
        doubleCount++;
      }
    }
    
    if (doubleCount > 0) {
      await expect(doubleHighlights).toHaveCount(doubleCount);
    }
  });

  test('should not re-animate dominoes when hovering during bidding', async ({ page }) => {
    await page.click('[data-testid="nav-actions"]');
    await expect(page.locator('.hand-section')).toBeVisible();
    
    // Wait for initial animations to complete
    await page.waitForTimeout(1000);
    
    // Get a domino element
    const firstDomino = page.locator('.hand-section .domino').first();
    
    // Add a test attribute to track if animation replays
    await firstDomino.evaluate(el => {
      el.setAttribute('data-animation-complete', 'true');
    });
    
    // Hover over the domino
    await firstDomino.hover();
    
    // Check that the animation-complete attribute is still there
    // (would be removed if component re-rendered)
    await expect(firstDomino).toHaveAttribute('data-animation-complete', 'true');
    
    // Hover over a different domino
    const secondDomino = page.locator('.hand-section .domino').nth(1);
    await secondDomino.hover();
    
    // Original domino should still have the attribute
    await expect(firstDomino).toHaveAttribute('data-animation-complete', 'true');
  });

  test('should show appropriate badge colors for each suit', async ({ page }) => {
    await page.click('[data-testid="nav-actions"]');
    await expect(page.locator('.hand-section')).toBeVisible();
    
    const suitTests = [
      { suit: 0, name: 'Blanks', badgeClass: 'suit-badge-0' },
      { suit: 1, name: 'Ones', badgeClass: 'suit-badge-1' },
      { suit: 2, name: 'Twos', badgeClass: 'suit-badge-2' },
      { suit: 3, name: 'Threes', badgeClass: 'suit-badge-3' },
      { suit: 4, name: 'Fours', badgeClass: 'suit-badge-4' },
      { suit: 5, name: 'Fives', badgeClass: 'suit-badge-5' },
      { suit: 6, name: 'Sixes', badgeClass: 'suit-badge-6' },
    ];
    
    for (const { suit, name, badgeClass } of suitTests) {
      // Find any domino with this suit
      let found = false;
      for (let other = 0; other <= 6; other++) {
        const selector1 = `[data-testid="domino-${suit}-${other}"]`;
        const selector2 = `[data-testid="domino-${other}-${suit}"]`;
        
        if (await page.locator(selector1).count() > 0) {
          await page.locator(selector1).first().hover();
          found = true;
          break;
        } else if (await page.locator(selector2).count() > 0) {
          await page.locator(selector2).first().hover();
          found = true;
          break;
        }
      }
      
      if (found) {
        const badge = page.locator('.suit-highlight-badge');
        await expect(badge).toBeVisible();
        await expect(badge).toHaveClass(new RegExp(badgeClass));
        await expect(badge).toContainText(name);
        
        // Move mouse away
        await page.mouse.move(0, 0);
        await page.waitForTimeout(100);
      }
    }
  });

  test('should show pointing finger animation on trump button hover', async ({ page }) => {
    // Get to trump selection
    await helper.makeBid(30);
    await expect(page.locator('text=Select Trump')).toBeVisible();
    
    const trumpButton = page.locator('.trump-button').first();
    
    // Initially no pointing finger
    await expect(trumpButton).not.toHaveClass(/trump-hovering/);
    
    // Hover over button
    await trumpButton.hover();
    
    // Should have hovering class and pointing animation
    await expect(trumpButton).toHaveClass(/trump-hovering/);
    
    // The CSS adds a ::after pseudo-element with pointing finger
    // We can check the button has the class that triggers it
    const hasHoverClass = await trumpButton.evaluate(el => 
      el.classList.contains('trump-hovering')
    );
    expect(hasHoverClass).toBe(true);
    
    // Move away
    await page.mouse.move(0, 0);
    await expect(trumpButton).not.toHaveClass(/trump-hovering/);
  });
});