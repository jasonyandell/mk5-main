import { test, expect } from '@playwright/test';

test('simple highlight check', async ({ page }) => {
  // Go to app
  await page.goto('/');
  
  // Wait a moment
  await page.waitForTimeout(1000);
  
  // Click Actions tab
  await page.click('[data-testid="nav-actions"]');
  await page.waitForTimeout(500);
  
  // Get first domino and hover
  const firstDomino = page.locator('.hand-section .domino').first();
  
  // Get initial classes
  const initialClasses = await firstDomino.getAttribute('class');
  console.log('Initial classes:', initialClasses);
  
  // Hover over it
  await firstDomino.hover();
  await page.waitForTimeout(500);
  
  // Get classes after hover
  const hoverClasses = await firstDomino.getAttribute('class');
  console.log('Hover classes:', hoverClasses);
  
  // Check all dominoes for highlight classes
  const allDominoes = page.locator('.hand-section .domino');
  const count = await allDominoes.count();
  
  for (let i = 0; i < count; i++) {
    const domino = allDominoes.nth(i);
    const classes = await domino.getAttribute('class');
    if (classes && classes.includes('highlight')) {
      console.log(`Domino ${i} classes:`, classes);
    }
  }
  
  // Take screenshot
  await page.screenshot({ path: 'test-results/simple-highlight.png', fullPage: true });
});