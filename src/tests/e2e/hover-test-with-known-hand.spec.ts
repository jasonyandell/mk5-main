import { test } from '@playwright/test';

test('test hover with specific game state', async ({ page }) => {
  // Use URL with known hand containing 3-2 and 3-3
  await page.goto('/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQzMzIxNDYzNTh9LCJhIjpbXX0');
  await page.waitForSelector('.game-progress', { timeout: 5000 });
  
  // Log all dominoes in hand
  const dominoes = await page.locator('.domino').evaluateAll(elements => 
    elements.map(el => el.getAttribute('data-testid'))
  );
  console.log('Dominoes in hand:', dominoes);
  
  // Find 3-2 domino
  const domino32 = page.locator('[data-testid="domino-3-2"]');
  const has32 = await domino32.count() > 0;
  
  if (has32) {
    const box = await domino32.boundingBox();
    if (box) {
      // Hover over top half (3)
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 4);
      await page.waitForTimeout(200);
      
      const topSuit = await page.locator('.suit-indicator').textContent();
      console.log('Hovering 3-2 top half:', topSuit);
      
      // Count highlighted
      let highlighted = await page.locator('.domino.highlight-primary').count();
      console.log('Dominoes highlighted when hovering 3:', highlighted);
      
      await page.screenshot({ path: 'hover-3.png' });
      
      // Hover over bottom half (2)
      await page.mouse.move(box.x + box.width / 2, box.y + box.height * 3 / 4);
      await page.waitForTimeout(200);
      
      const bottomSuit = await page.locator('.suit-indicator').textContent();
      console.log('Hovering 3-2 bottom half:', bottomSuit);
      
      highlighted = await page.locator('.domino.highlight-primary').count();
      console.log('Dominoes highlighted when hovering 2:', highlighted);
      
      await page.screenshot({ path: 'hover-2.png' });
    }
  }
  
  // Find 3-3 domino
  const domino33 = page.locator('[data-testid="domino-3-3"]');
  const has33 = await domino33.count() > 0;
  
  if (has33) {
    const box = await domino33.boundingBox();
    if (box) {
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.waitForTimeout(200);
      
      const doubleSuit = await page.locator('.suit-indicator').textContent();
      console.log('Hovering 3-3:', doubleSuit);
      
      const primary = await page.locator('.domino.highlight-primary').count();
      const secondary = await page.locator('.domino.highlight-secondary').count();
      console.log('When hovering 3-3 - Primary:', primary, 'Secondary:', secondary);
      
      await page.screenshot({ path: 'hover-double-3.png' });
    }
  }
});