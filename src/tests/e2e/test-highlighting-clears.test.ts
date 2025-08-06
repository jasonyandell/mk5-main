import { test, expect } from '@playwright/test';

test.describe('Highlighting clears on action', () => {
  test('highlighting clears when bidding action is performed', async ({ page }) => {
    // Load bidding phase
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ0Mjg2MjcwOTR9LCJhIjpbeyJpIjoicCJ9LHsiaSI6IjMwIn0seyJpIjoicCJ9XX0');
    
    // Wait for the action panel
    await page.waitForSelector('.action-panel', { timeout: 5000 });
    
    // Click on a domino to highlight it
    const domino = await page.locator('.action-panel .domino').first();
    await domino.click();
    
    // Check that some dominoes are highlighted
    let highlighted = await page.locator('.action-panel .domino.highlight-primary, .action-panel .domino.highlight-secondary').count();
    console.log(`Dominoes highlighted after click: ${highlighted}`);
    expect(highlighted).toBeGreaterThan(0);
    
    // Now perform a bid action (pass)
    const passButton = await page.locator('button:has-text("Pass")').first();
    await passButton.click();
    
    // Wait a bit for the action to process
    await page.waitForTimeout(200);
    
    // Check that highlighting is cleared
    highlighted = await page.locator('.action-panel .domino.highlight-primary, .action-panel .domino.highlight-secondary').count();
    console.log(`Dominoes highlighted after action: ${highlighted}`);
    expect(highlighted).toBe(0);
  });

  test('highlighting clears when trump is selected', async ({ page }) => {
    // Load trump selection phase
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ0Mjg2Mjg4OTh9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifV19');
    
    // Wait for the action panel
    await page.waitForSelector('.action-panel', { timeout: 5000 });
    
    // Click on a domino to highlight it
    const domino = await page.locator('.action-panel .domino').first();
    await domino.click();
    
    // Check that some dominoes are highlighted
    let highlighted = await page.locator('.action-panel .domino.highlight-primary, .action-panel .domino.highlight-secondary').count();
    console.log(`Dominoes highlighted after click: ${highlighted}`);
    expect(highlighted).toBeGreaterThan(0);
    
    // Now select a trump
    const trumpButton = await page.locator('button:has-text("Threes")').first();
    await trumpButton.click();
    
    // Wait a bit for the action to process
    await page.waitForTimeout(200);
    
    // Check that highlighting is cleared (might switch panels, so check both)
    highlighted = await page.locator('.domino.highlight-primary, .domino.highlight-secondary').count();
    console.log(`Dominoes highlighted after trump selection: ${highlighted}`);
    expect(highlighted).toBe(0);
  });

  test('highlighting clears when team status is toggled', async ({ page }) => {
    // Load bidding phase
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ0Mjg2MjcwOTR9LCJhIjpbeyJpIjoicCJ9LHsiaSI6IjMwIn0seyJpIjoicCJ9XX0');
    
    // Wait for the action panel
    await page.waitForSelector('.action-panel', { timeout: 5000 });
    
    // Click on a domino to highlight it
    const domino = await page.locator('.action-panel .domino').first();
    await domino.click();
    
    // Check that some dominoes are highlighted
    let highlighted = await page.locator('.action-panel .domino.highlight-primary, .action-panel .domino.highlight-secondary').count();
    console.log(`Dominoes highlighted after click: ${highlighted}`);
    expect(highlighted).toBeGreaterThan(0);
    
    // Click team status toggle
    const teamStatusToggle = await page.locator('.team-status-toggle').first();
    await teamStatusToggle.click();
    
    // Wait a bit for the action to process
    await page.waitForTimeout(200);
    
    // Check that highlighting is cleared
    highlighted = await page.locator('.action-panel .domino.highlight-primary, .action-panel .domino.highlight-secondary').count();
    console.log(`Dominoes highlighted after team status toggle: ${highlighted}`);
    expect(highlighted).toBe(0);
  });
});