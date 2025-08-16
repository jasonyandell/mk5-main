import { test } from '@playwright/test';

test('final hover test with screenshot', async ({ page }) => {
  // Navigate directly to bidding phase
  await page.goto('http://localhost:3001/#eyJjdXJyZW50UGxheWVyIjowLCJwaGFzZSI6ImJpZGRpbmciLCJkZWFsZXIiOjAsImJpZHMiOltdLCJoYW5kcyI6W1t7ImhpZ2giOjYsImxvdyI6NiwicG9pbnRzIjowfSx7ImhpZ2giOjUsImxvdyI6MywicG9pbnRzIjowfSx7ImhpZ2giOjUsImxvdyI6MiwicG9pbnRzIjowfSx7ImhpZ2giOjQsImxvdyI6MSwicG9pbnRzIjo1fSx7ImhpZ2giOjMsImxvdyI6MywicG9pbnRzIjowfSx7ImhpZ2giOjIsImxvdyI6MCwicG9pbnRzIjowfSx7ImhpZ2giOjEsImxvdyI6MCwicG9pbnRzIjowfV0sW10sW10sW11dLCJ0cnVtcCI6eyJ0eXBlIjoibm9uZSJ9fQ==');
  
  await page.waitForTimeout(2000);
  
  // Go to Actions tab
  await page.click('[data-testid="nav-actions"]');
  await page.waitForTimeout(1000);
  
  // Take before screenshot
  await page.screenshot({ path: 'test-results/hover-test-before.png' });
  
  // Find a domino to hover - look for any in the hand-display
  const handDisplay = page.locator('.hand-display');
  const dominoes = handDisplay.locator('.domino');
  
  // Hover on the second domino (5-3)
  await dominoes.nth(1).hover();
  await page.waitForTimeout(1000);
  
  // Take after screenshot
  await page.screenshot({ path: 'test-results/hover-test-after.png' });
  
  // Also check the console for any errors
  const consoleMessages: string[] = [];
  page.on('console', msg => consoleMessages.push(msg.text()));
  
  console.log('Test complete. Check test-results/ for screenshots.');
});