import { test, expect } from '@playwright/test';

test('simple highlight test', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.game-progress', { timeout: 5000 });
  
  // Check that we're in bidding phase
  const phaseIndicator = await page.locator('.phase-badge').textContent();
  console.log('Current phase:', phaseIndicator);
  
  // Find any domino and hover
  const firstDomino = page.locator('.domino').first();
  await firstDomino.hover();
  await page.waitForTimeout(500);
  
  // Check if suit indicator appears
  const suitIndicator = await page.locator('.suit-indicator');
  const isVisible = await suitIndicator.isVisible();
  console.log('Suit indicator visible:', isVisible);
  
  if (isVisible) {
    const text = await suitIndicator.textContent();
    console.log('Suit indicator text:', text);
  }
  
  // Take screenshot
  await page.screenshot({ path: 'simple-hover.png' });
  
  // Now check the actual highlight logic by injecting a test
  const result = await page.evaluate(() => {
    // Find the first domino element
    const domino = document.querySelector('.domino');
    if (!domino) return 'No domino found';
    
    // Check computed styles
    const styles = window.getComputedStyle(domino);
    return {
      backgroundColor: styles.backgroundColor,
      classList: Array.from(domino.classList),
      title: domino.getAttribute('title')
    };
  });
  
  console.log('First domino state:', result);
});