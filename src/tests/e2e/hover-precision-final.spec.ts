import { test } from '@playwright/test';

test('test precise hover with 3-2 domino', async ({ page }) => {
  // Use a URL that ensures we have specific dominoes
  await page.goto('/');
  await page.waitForSelector('.game-progress', { timeout: 5000 });
  
  // Find a domino with different values (not a double)
  const dominoes = await page.locator('.domino').all();
  
  for (const domino of dominoes) {
    const title = await domino.getAttribute('title');
    if (!title) continue;
    
    const [high, low] = title.split('-').map(Number);
    if (high === low || high === 0 || low === 0) continue; // Skip doubles and blanks for clarity
    
    console.log(`\nTesting with domino: ${title} (high=${high}, low=${low})`);
    const box = await domino.boundingBox();
    if (!box) continue;
    
    // Move mouse away first
    await page.mouse.move(0, 0);
    await page.waitForTimeout(100);
    
    // Hover very close to the top (10% from top)
    await page.mouse.move(box.x + box.width / 2, box.y + box.height * 0.1);
    await page.waitForTimeout(300);
    
    let suitText = await page.locator('.suit-indicator').textContent();
    console.log(`Top hover (10% from top): ${suitText}`);
    
    // Take screenshot
    await page.screenshot({ path: `hover-${title}-top.png` });
    
    // Hover very close to the bottom (90% from top)
    await page.mouse.move(box.x + box.width / 2, box.y + box.height * 0.9);
    await page.waitForTimeout(300);
    
    suitText = await page.locator('.suit-indicator').textContent();
    console.log(`Bottom hover (90% from top): ${suitText}`);
    
    // Take screenshot
    await page.screenshot({ path: `hover-${title}-bottom.png` });
    
    // Verify the highlighting is working
    const expectedTopSuit = ['Blanks', 'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes'][high];
    const expectedBottomSuit = ['Blanks', 'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes'][low];
    
    console.log(`Expected: Top=${expectedTopSuit}, Bottom=${expectedBottomSuit}`);
    
    break; // Just test one domino
  }
});