import { test, expect } from '@playwright/test';

test('debug highlighting classes', async ({ page }) => {
  // Navigate to a game in bidding phase
  await page.goto('http://localhost:3001/#eyJjdXJyZW50UGxheWVyIjowLCJwaGFzZSI6ImJpZGRpbmciLCJkZWFsZXIiOjAsImJpZHMiOltdLCJoYW5kcyI6W1t7ImhpZ2giOjYsImxvdyI6NiwicG9pbnRzIjowfSx7ImhpZ2giOjUsImxvdyI6MywicG9pbnRzIjowfSx7ImhpZ2giOjUsImxvdyI6MiwicG9pbnRzIjowfSx7ImhpZ2giOjQsImxvdyI6MSwicG9pbnRzIjo1fSx7ImhpZ2giOjMsImxvdyI6MywicG9pbnRzIjowfSx7ImhpZ2giOjIsImxvdyI6MCwicG9pbnRzIjowfSx7ImhpZ2giOjEsImxvdyI6MCwicG9pbnRzIjowfV0sW10sW10sW11dLCJ0cnVtcCI6eyJ0eXBlIjoibm9uZSJ9fQ==');
  
  await page.waitForTimeout(1000);
  
  // Click on Actions tab
  await page.click('[data-testid="nav-actions"]');
  await page.waitForTimeout(500);
  
  // Find the 5-3 domino and hover
  const domino53 = page.getByTestId('domino-5-3');
  
  // Get classes before hover
  const classesBefore = await domino53.getAttribute('class');
  console.log('Classes before hover:', classesBefore);
  
  // Hover over the domino
  await domino53.hover();
  await page.waitForTimeout(500);
  
  // Get classes after hover
  const classesAfter = await domino53.getAttribute('class');
  console.log('Classes after hover:', classesAfter);
  
  // Also check the 5-2 domino which should also be highlighted
  const domino52 = page.getByTestId('domino-5-2');
  const classes52 = await domino52.getAttribute('class');
  console.log('Classes on 5-2 domino:', classes52);
  
  // Check if the highlight indicator is visible
  const indicator = page.locator('.suit-highlight-badge');
  const isIndicatorVisible = await indicator.isVisible();
  console.log('Indicator visible:', isIndicatorVisible);
  
  if (isIndicatorVisible) {
    const indicatorText = await indicator.textContent();
    console.log('Indicator text:', indicatorText);
  }
});