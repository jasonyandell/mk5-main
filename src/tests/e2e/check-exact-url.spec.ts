import { test } from '@playwright/test';

test('check exact URL layout', async ({ page }) => {
  // Go to the exact URL
  await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQzMzIxNDYzNTh9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiIzNSJ9LHsiaSI6InAifSx7ImkiOiJwIn0seyJpIjoidHJ1bXAtYmxhbmtzIn0seyJpIjoiNjMifSx7ImkiOiI2MCJ9LHsiaSI6IjYyIn0seyJpIjoiNjQifSx7ImkiOiJjb21wbGV0ZS10cmljayJ9XX0');
  
  await page.waitForSelector('.game-progress', { timeout: 5000 });
  
  // Take a full screenshot
  await page.screenshot({ path: 'exact-url-screenshot.png', fullPage: true });
  
  // Also take a screenshot of just the game progress area
  const gameProgress = page.locator('.game-progress');
  await gameProgress.screenshot({ path: 'game-progress-only.png' });
  
  // Check the HTML structure
  const trickDominoesClass = await page.locator('.trick-dominoes-horizontal').count();
  console.log('Elements with .trick-dominoes-horizontal:', trickDominoesClass);
  
  const oldTrickDominoesClass = await page.locator('.trick-dominoes').count();
  console.log('Elements with .trick-dominoes (old class):', oldTrickDominoesClass);
  
  // Get the actual HTML of the first completed trick
  const firstTrick = page.locator('.trick-card').first();
  if (await firstTrick.count() > 0) {
    const html = await firstTrick.innerHTML();
    console.log('First trick HTML:', html.substring(0, 300));
  }
});