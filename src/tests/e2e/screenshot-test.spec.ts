import { test } from '@playwright/test';

test('take screenshot of current trick area', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('header', { timeout: 5000 });
  
  // Wait for the trick area to be visible
  await page.waitForSelector('.current-trick-area', { timeout: 5000 });
  
  // Take a screenshot of just the trick area
  const trickArea = page.locator('.current-trick-area');
  await trickArea.screenshot({ path: 'current-trick-area.png' });
  
  // Also take a full page screenshot
  await page.screenshot({ path: 'full-page.png', fullPage: true });
  
  // Log what we find
  const trickContainer = await page.locator('.trick-horizontal').count();
  console.log('Elements with .trick-horizontal class:', trickContainer);
  
  const trickGrid = await page.locator('.trick-grid').count();
  console.log('Elements with .trick-grid class:', trickGrid);
  
  // Check the actual HTML
  const trickAreaHTML = await trickArea.innerHTML();
  console.log('Trick area HTML:', trickAreaHTML.substring(0, 500));
});