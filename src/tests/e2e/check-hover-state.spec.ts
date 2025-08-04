import { test } from '@playwright/test';

test('check hover state in component', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.game-progress', { timeout: 5000 });
  
  // Add console log listener
  page.on('console', msg => {
    if (msg.type() === 'log') {
      console.log('Browser console:', msg.text());
    }
  });
  
  // Inject a function to check the Svelte component state
  await page.evaluate(() => {
    // Override console.log to capture logs
    const originalLog = console.log;
    window.console.log = (...args) => {
      originalLog(...args);
      window.postMessage({ type: 'console-log', data: args }, '*');
    };
  });
  
  // Find and hover over 3-3
  const domino33 = page.locator('[data-testid="domino-3-3"]');
  if (await domino33.count() > 0) {
    console.log('Found 3-3, hovering...');
    const box = await domino33.boundingBox();
    if (box) {
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.waitForTimeout(500);
      
      // Check if suit indicator is showing
      const suitIndicator = await page.locator('.suit-indicator').textContent();
      console.log('Suit indicator shows:', suitIndicator);
      
      // Check the actual highlight prop values being passed
      const highlightValues = await page.evaluate(() => {
        const dominoElements = document.querySelectorAll('.domino');
        const results = [];
        dominoElements.forEach(el => {
          // Try to get the Svelte component instance
          const title = el.getAttribute('title');
          const classes = Array.from(el.classList);
          results.push({ title, classes });
        });
        return results;
      });
      
      console.log('Domino elements after hover:', highlightValues);
    }
  }
});