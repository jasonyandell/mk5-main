import { test, expect } from '@playwright/test';

test('manual hover test - keep browser open', async ({ page }) => {
  // Start with a game in bidding phase
  const gameUrl = '/#