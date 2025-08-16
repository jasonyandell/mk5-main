import { test } from '@playwright/test';

test('test precise domino hover', async ({ page }) => {
  // Start fresh game
  await page.goto('/');
  await page.waitForSelector('.game-progress', { timeout: 5000 });
  
  // Should be in bidding phase
  await page.waitForSelector('.action-button.bid', { timeout: 5000 });
  
  // Find a non-double domino (like 3-2)
  const dominoes = page.locator('.domino');
  const count = await dominoes.count();
  
  for (let i = 0; i < count; i++) {
    const domino = dominoes.nth(i);
    const testId = await domino.getAttribute('data-testid');
    
    if (testId && testId.includes('3') && testId.includes('2') && !testId.includes('3-3')) {
      console.log('Found domino:', testId);
      
      const box = await domino.boundingBox();
      if (!box) continue;
      
      // Hover over top half (should highlight 3s)
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 4);
      await page.waitForTimeout(100);
      
      let suitText = await page.locator('.suit-indicator').textContent();
      console.log('Hovering top half:', suitText);
      
      await page.screenshot({ path: 'hover-top-half.png', fullPage: true });
      
      // Hover over bottom half (should highlight 2s)
      await page.mouse.move(box.x + box.width / 2, box.y + box.height * 3 / 4);
      await page.waitForTimeout(100);
      
      suitText = await page.locator('.suit-indicator').textContent();
      console.log('Hovering bottom half:', suitText);
      
      await page.screenshot({ path: 'hover-bottom-half.png', fullPage: true });
      
      break;
    }
  }
  
  // Now find a double (like 3-3)
  for (let i = 0; i < count; i++) {
    const domino = dominoes.nth(i);
    const testId = await domino.getAttribute('data-testid');
    
    if (testId && testId.includes('3-3')) {
      console.log('Found double:', testId);
      
      const box = await domino.boundingBox();
      if (!box) continue;
      
      // Hover over it
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.waitForTimeout(100);
      
      const suitText = await page.locator('.suit-indicator').textContent();
      console.log('Hovering double:', suitText);
      
      // Count highlighted dominoes
      const primaryCount = await page.locator('.domino.highlight-primary').count();
      const secondaryCount = await page.locator('.domino.highlight-secondary').count();
      console.log('Primary highlights:', primaryCount, 'Secondary highlights:', secondaryCount);
      
      await page.screenshot({ path: 'hover-double.png', fullPage: true });
      
      break;
    }
  }
});