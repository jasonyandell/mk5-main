import { test } from '@playwright/test';

test('final hover test with visual check', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.game-progress', { timeout: 5000 });
  
  // Find a double
  const double = page.locator('.domino').filter({ hasText: /^(\d)\s+\1$/ }).first();
  const doubleTitle = await double.getAttribute('title');
  console.log('Found double:', doubleTitle);
  
  if (doubleTitle) {
    await double.hover();
    await page.waitForTimeout(500);
    
    // Take screenshot
    await page.screenshot({ path: 'final-hover-check.png', fullPage: true });
    
    // Check actual DOM
    const dominoStates = await page.evaluate(() => {
      const dominoes = Array.from(document.querySelectorAll('.domino'));
      return dominoes.map(el => {
        const computedStyle = window.getComputedStyle(el);
        return {
          title: el.getAttribute('title'),
          backgroundColor: computedStyle.backgroundColor,
          borderColor: computedStyle.borderColor,
          boxShadow: computedStyle.boxShadow,
          classList: Array.from(el.classList)
        };
      });
    });
    
    console.log('Domino styles:', dominoStates);
    
    // Check if any have the highlight background colors
    const highlighted = dominoStates.filter(d => 
      d.backgroundColor.includes('254, 243, 199') || // highlight-primary color
      d.backgroundColor.includes('219, 234, 254')    // highlight-secondary color
    );
    
    console.log('Highlighted dominoes:', highlighted.length);
  }
});