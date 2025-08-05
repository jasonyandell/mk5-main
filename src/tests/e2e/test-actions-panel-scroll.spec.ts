import { test, expect } from '@playwright/test';

test('Your Hand stays visible when scrolling Actions during trump selection', async ({ page }) => {
  // Navigate to trump selection phase
  await page.goto('http://localhost:3001/#eyJjdXJyZW50UGxheWVyIjowLCJwaGFzZSI6InRydW1wX3NlbGVjdGlvbiIsImRlYWxlciI6MCwiYmlkV2lubmVyIjowLCJiaWRBbW91bnQiOjMwLCJiaWRzIjpbeyJwbGF5ZXIiOjAsInR5cGUiOiJiaWQiLCJ2YWx1ZSI6MzB9XSwiaGFuZHMiOltbeyJoaWdoIjo2LCJsb3ciOjYsInBvaW50cyI6MH0seyJoaWdoIjo1LCJsb3ciOjMsInBvaW50cyI6MH0seyJoaWdoIjo1LCJsb3ciOjAsInBvaW50cyI6NX0seyJoaWdoIjo0LCJsb3ciOjEsInBvaW50cyI6NX0seyJoaWdoIjozLCJsb3ciOjMsInBvaW50cyI6MH0seyJoaWdoIjoyLCJsb3ciOjAsInBvaW50cyI6MH0seyJoaWdoIjoxLCJsb3ciOjAsInBvaW50cyI6MH1dLFtdLFtdLFtdXSwidHJ1bXAiOnsidHlwZSI6Im5vbmUifX0=');
  
  await page.waitForTimeout(1000);
  
  // Click on Actions tab
  await page.click('[data-testid="nav-actions"]');
  await page.waitForTimeout(500);
  
  // Check that "Your Hand" section is visible
  const yourHandSection = page.locator('.hand-section');
  await expect(yourHandSection).toBeVisible();
  
  // Check initial position of Your Hand
  const initialBounds = await yourHandSection.boundingBox();
  console.log('Initial Your Hand position:', initialBounds?.y);
  
  // Find the actions container and scroll it
  const actionsContainer = page.locator('.actions-container');
  
  // Scroll down in the actions container
  await actionsContainer.evaluate(el => el.scrollTop = 200);
  await page.waitForTimeout(500);
  
  // Check if Your Hand is still visible and in the same position
  await expect(yourHandSection).toBeVisible();
  const afterScrollBounds = await yourHandSection.boundingBox();
  console.log('After scroll Your Hand position:', afterScrollBounds?.y);
  
  // Your Hand should stay in the same position
  if (initialBounds && afterScrollBounds) {
    expect(afterScrollBounds.y).toBe(initialBounds.y);
  }
  
  // Also check that we can see trump buttons
  const trumpButtons = page.locator('.trump-button');
  const trumpButtonCount = await trumpButtons.count();
  console.log('Number of trump buttons visible:', trumpButtonCount);
  expect(trumpButtonCount).toBeGreaterThan(0);
  
  // Take screenshots
  await page.screenshot({ path: 'test-results/actions-panel-before-scroll.png' });
  
  // Scroll to bottom
  await actionsContainer.evaluate(el => el.scrollTop = el.scrollHeight);
  await page.waitForTimeout(500);
  
  await page.screenshot({ path: 'test-results/actions-panel-after-scroll.png' });
  
  // Your Hand should still be visible
  await expect(yourHandSection).toBeVisible();
});