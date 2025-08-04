import { test } from '@playwright/test';

test('check bidding UI improvements', async ({ page }) => {
  // Start fresh game
  await page.goto('/');
  await page.waitForSelector('.game-progress', { timeout: 5000 });
  
  // Should be in bidding phase
  await page.waitForSelector('.action-button.bid', { timeout: 5000 });
  
  // Take screenshot of bidding UI
  await page.screenshot({ path: 'bidding-ui.png', fullPage: true });
  
  // Count bid buttons
  const bidButtons = await page.locator('.action-button.bid').count();
  console.log('Number of bid buttons:', bidButtons);
  
  // Check pass button
  const passButton = await page.locator('.action-button.pass').count();
  console.log('Pass button present:', passButton > 0);
  
  // Try hovering over a domino
  const firstDomino = page.locator('.domino').first();
  if (await firstDomino.count() > 0) {
    await firstDomino.hover();
    await page.waitForTimeout(500);
    
    // Take screenshot while hovering
    await page.screenshot({ path: 'bidding-hover.png', fullPage: true });
    
    // Check for highlighted dominoes
    const highlightedPrimary = await page.locator('.domino.highlight-primary').count();
    const highlightedSecondary = await page.locator('.domino.highlight-secondary').count();
    console.log('Primary highlighted dominoes:', highlightedPrimary);
    console.log('Secondary highlighted dominoes:', highlightedSecondary);
    
    // Check suit indicator
    const suitIndicator = await page.locator('.suit-indicator').textContent();
    console.log('Suit indicator text:', suitIndicator);
  }
});