import { test, expect } from '@playwright/test';

test('screenshot domino highlighting', async ({ page }) => {
  // Navigate to a game in bidding phase
  await page.goto('http://localhost:3001/#eyJjdXJyZW50UGxheWVyIjowLCJwaGFzZSI6ImJpZGRpbmciLCJkZWFsZXIiOjAsImJpZHMiOltdLCJoYW5kcyI6W1t7ImhpZ2giOjYsImxvdyI6NiwicG9pbnRzIjowfSx7ImhpZ2giOjUsImxvdyI6MywicG9pbnRzIjowfSx7ImhpZ2giOjUsImxvdyI6MiwicG9pbnRzIjowfSx7ImhpZ2giOjQsImxvdyI6MSwicG9pbnRzIjo1fSx7ImhpZ2giOjMsImxvdyI6MywicG9pbnRzIjowfSx7ImhpZ2giOjIsImxvdyI6MCwicG9pbnRzIjowfSx7ImhpZ2giOjEsImxvdyI6MCwicG9pbnRzIjowfV0sW10sW10sW11dLCJ0cnVtcCI6eyJ0eXBlIjoibm9uZSJ9fQ==');
  
  // Wait for page to load
  await page.waitForTimeout(1000);
  
  // Click on Actions tab to see "Your Hand"
  await page.click('[data-testid="nav-actions"]');
  await page.waitForTimeout(500);
  
  // Take screenshot before hovering
  await page.screenshot({ path: 'test-results/before-hover.png', fullPage: true });
  
  // Find the 5-3 domino and hover over it
  const domino53 = page.getByTestId('domino-5-3');
  await domino53.hover();
  
  // Wait a moment for the hover effect
  await page.waitForTimeout(500);
  
  // Take screenshot while hovering
  await page.screenshot({ path: 'test-results/while-hovering-5-3.png', fullPage: true });
  
  // Also test hovering over a double
  await page.mouse.move(0, 0); // Move mouse away
  await page.waitForTimeout(200);
  
  const double66 = page.getByTestId('domino-6-6');
  await double66.hover();
  await page.waitForTimeout(500);
  
  // Take screenshot while hovering over double
  await page.screenshot({ path: 'test-results/while-hovering-double-6-6.png', fullPage: true });
  
  console.log('Screenshots saved to test-results/');
});