import { test } from '@playwright/test';

test('test hover on double domino', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.game-progress', { timeout: 5000 });
  
  // Find a double domino
  const dominoes = await page.locator('.domino').all();
  
  for (const domino of dominoes) {
    const title = await domino.getAttribute('title');
    if (!title) continue;
    
    const [high, low] = title.split('-').map(Number);
    if (high !== low) continue; // Skip non-doubles
    
    console.log(`\nTesting with double: ${title}`);
    const box = await domino.boundingBox();
    if (!box) continue;
    
    // Hover over the double
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.waitForTimeout(300);
    
    const suitText = await page.locator('.suit-indicator').textContent();
    console.log(`Hovering on ${title}: ${suitText}`);
    
    // Count highlights
    const primaryCount = await page.locator('.domino.highlight-primary').count();
    const secondaryCount = await page.locator('.domino.highlight-secondary').count();
    console.log(`Primary highlights: ${primaryCount}, Secondary highlights: ${secondaryCount}`);
    
    // Take screenshot
    await page.screenshot({ path: `hover-double-${title}.png` });
    
    // Also test hovering on a non-double with the same suit
    const nonDoubleWithSameSuit = await page.locator('.domino').all();
    for (const nd of nonDoubleWithSameSuit) {
      const ndTitle = await nd.getAttribute('title');
      if (!ndTitle) continue;
      
      const [ndHigh, ndLow] = ndTitle.split('-').map(Number);
      if (ndHigh === ndLow) continue; // Skip doubles
      
      if (ndHigh === high || ndLow === high) {
        console.log(`\nNow testing non-double ${ndTitle} that contains ${high}`);
        const ndBox = await nd.boundingBox();
        if (!ndBox) continue;
        
        // Hover on the matching number
        const isHighMatch = ndHigh === high;
        await page.mouse.move(
          ndBox.x + ndBox.width / 2, 
          ndBox.y + (isHighMatch ? ndBox.height * 0.1 : ndBox.height * 0.9)
        );
        await page.waitForTimeout(300);
        
        const ndSuitText = await page.locator('.suit-indicator').textContent();
        console.log(`Hovering on ${high} in ${ndTitle}: ${ndSuitText}`);
        
        break;
      }
    }
    
    break; // Just test one double
  }
});