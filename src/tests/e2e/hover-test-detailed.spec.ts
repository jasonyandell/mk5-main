import { test } from '@playwright/test';

test('test detailed hover behavior', async ({ page }) => {
  // Start fresh game
  await page.goto('/');
  await page.waitForSelector('.game-progress', { timeout: 5000 });
  
  // Log all dominoes
  const dominoes = await page.locator('.domino').evaluateAll(elements => 
    elements.map(el => ({
      id: el.getAttribute('data-testid'),
      title: el.getAttribute('title')
    }))
  );
  console.log('Hand contains:', dominoes);
  
  // Find any non-double domino
  const nonDouble = dominoes.find(d => {
    const parts = d.title?.split('-');
    return parts && parts[0] !== parts[1];
  });
  
  if (nonDouble) {
    console.log('Testing with non-double:', nonDouble.title);
    const domino = page.locator(`[data-testid="${nonDouble.id}"]`);
    const box = await domino.boundingBox();
    
    if (box) {
      // Test top half
      await page.mouse.move(box.x + box.width / 2, box.y + 10);
      await page.waitForTimeout(200);
      
      let suit = await page.locator('.suit-indicator').textContent();
      console.log('Top half hover:', suit);
      await page.screenshot({ path: 'hover-test-top.png' });
      
      // Test bottom half
      await page.mouse.move(box.x + box.width / 2, box.y + box.height - 10);
      await page.waitForTimeout(200);
      
      suit = await page.locator('.suit-indicator').textContent();
      console.log('Bottom half hover:', suit);
      await page.screenshot({ path: 'hover-test-bottom.png' });
    }
  }
  
  // Find any double
  const double = dominoes.find(d => {
    const parts = d.title?.split('-');
    return parts && parts[0] === parts[1];
  });
  
  if (double) {
    console.log('Testing with double:', double.title);
    const domino = page.locator(`[data-testid="${double.id}"]`);
    const box = await domino.boundingBox();
    
    if (box) {
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.waitForTimeout(200);
      
      const suit = await page.locator('.suit-indicator').textContent();
      console.log('Double hover:', suit);
      
      const primary = await page.locator('.domino.highlight-primary').count();
      const secondary = await page.locator('.domino.highlight-secondary').count();
      console.log('Highlights - Primary:', primary, 'Secondary:', secondary);
      
      await page.screenshot({ path: 'hover-test-double.png' });
    }
  }
});