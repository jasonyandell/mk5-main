import { test } from '@playwright/test';

test('test compact tricks display', async ({ page }) => {
  // Start a game and play multiple tricks
  await page.goto('/');
  await page.waitForSelector('.game-progress', { timeout: 5000 });
  
  // Use AI to play several tricks quickly
  await page.click('button:has-text("Play All")');
  
  // Wait for several tricks to complete
  await page.waitForTimeout(3000);
  
  // Stop AI
  const pauseButton = page.locator('button:has-text("Pause")');
  if (await pauseButton.isVisible()) {
    await pauseButton.click();
  }
  
  // Take screenshot of game progress
  await page.screenshot({ path: 'compact-tricks.png', fullPage: true });
  
  // Check how many tricks fit
  const tricksCount = await page.locator('.trick-card').count();
  console.log('Number of tricks displayed:', tricksCount);
  
  // Check the height of the game progress area
  const progressArea = page.locator('.game-progress');
  const box = await progressArea.boundingBox();
  console.log('Game progress height:', box?.height);
  
  // Check if scrollbar is present
  const hasScroll = await progressArea.evaluate(el => {
    return el.scrollHeight > el.clientHeight;
  });
  console.log('Has scrollbar:', hasScroll);
});