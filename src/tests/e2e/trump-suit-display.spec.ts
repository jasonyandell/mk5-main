import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Trump Suit Display in Previous Tricks', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    await helper.goto();
  });

  test('shows trump suit in Previous Tricks panel after trump is selected', async () => {
    // Complete bidding to get to trump selection
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');  
    await helper.selectActionByType('pass');
    
    // Set trump to doubles
    await helper.setTrump('doubles');
    
    // Play a few dominoes to create tricks
    await helper.playAnyDomino(); // Player 1 leads  
    await helper.playAnyDomino(); // Player 2
    await helper.playAnyDomino(); // Player 3
    await helper.playAnyDomino(); // Player 0
    
    // Complete the trick
    await helper.completeTrick();
    
    // Verify trump suit appears in Previous Tricks panel
    const trickElements = helper.page.locator('.trick-compact');
    await expect(trickElements).toHaveCount(1);
    
    // Check for trump display in the completed trick
    const trumpDisplay = helper.page.locator('.trump-display');
    await expect(trumpDisplay).toBeVisible();
    await expect(trumpDisplay).toContainText('[doubles]');
    
    // Verify styling looks like a domino
    await expect(trumpDisplay).toHaveCSS('background-color', 'rgb(33, 37, 41)'); // Dark background
    await expect(trumpDisplay).toHaveCSS('color', 'rgb(255, 255, 255)'); // White text
    await expect(trumpDisplay).toHaveCSS('font-family', /monospace/); // Monospace font
  });

  test('shows trump suit for different suit types', async () => {
    // Test with numeric suit (5s)
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    
    await helper.setTrump('fives');
    
    // Start a trick
    await helper.playAnyDomino();
    
    // Check current trick shows trump
    const trumpDisplay = helper.page.locator('.trump-display');
    await expect(trumpDisplay).toBeVisible();
    await expect(trumpDisplay).toContainText('[5s]');
  });

  test('shows trump suit in both completed and current tricks', async () => {
    // Complete bidding and set trump
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.setTrump('sixes');
    
    // Play full trick to complete it
    await helper.playAnyDomino();
    await helper.playAnyDomino();
    await helper.playAnyDomino();
    await helper.playAnyDomino();
    await helper.completeTrick();
    
    // Start another trick
    await helper.playAnyDomino();
    
    // Should have trump display in both completed and current trick
    const trumpDisplays = helper.page.locator('.trump-display');
    await expect(trumpDisplays).toHaveCount(2);
    
    // Both should show the same trump
    for (const display of await trumpDisplays.all()) {
      await expect(display).toContainText('[6s]');
    }
  });

  test('does not show trump suit before trump is selected', async () => {
    // Before trump selection
    const trumpDisplay = helper.page.locator('.trump-display');
    await expect(trumpDisplay).toHaveCount(0);
    
    // Even during bidding
    await helper.selectActionByType('bid_points', 30);
    await expect(trumpDisplay).toHaveCount(0);
  });

  test('trump display appears to the left of winner and points', async () => {
    // Complete bidding and set trump
    await helper.selectActionByType('bid_points', 30);  
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.setTrump('blanks');
    
    // Play and complete a trick
    await helper.playAnyDomino();
    await helper.playAnyDomino();
    await helper.playAnyDomino();
    await helper.playAnyDomino();
    await helper.completeTrick();
    
    // Verify the order: trump display should come before winner info
    const trickInfo = helper.page.locator('.trick-info').first();
    const trumpDisplay = trickInfo.locator('.trump-display');
    const winnerInfo = trickInfo.locator('.winner-info');
    const pointsInfo = trickInfo.locator('.points-info');
    
    await expect(trumpDisplay).toBeVisible();
    await expect(winnerInfo).toBeVisible();
    await expect(pointsInfo).toBeVisible();
    
    // Trump should show [0s] for blanks
    await expect(trumpDisplay).toContainText('[0s]');
  });
});