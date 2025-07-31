import { test, expect } from '@playwright/test';

test('simple button test', async ({ page }) => {
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(1000);
  
  // Check for bid buttons
  const bidButtons = page.locator('[data-generic-testid*="bid-button"]');
  const count = await bidButtons.count();
  console.log('Bid buttons found:', count);
  
  if (count > 0) {
    const firstButton = bidButtons.first();
    const text = await firstButton.textContent();
    const testId = await firstButton.getAttribute('data-testid');
    console.log('First button text:', text);
    console.log('First button testid:', testId);
  }
  
  expect(count).toBeGreaterThan(0);
});