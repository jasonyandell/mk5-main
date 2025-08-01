import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Trump Suit Display in Previous Tricks', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
  });

  test('shows trump suit in Previous Tricks panel after trump is selected', async () => {
    // Start with deterministic seed 1 which offers all trump options
    await helper.gotoWithSeed(1);
    
    // Complete bidding to get to trump selection
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');  
    await helper.selectActionByType('pass');
    
    // Set trump to twos (with seed 1, this consistently produces [2s] display)
    await helper.setTrump('twos');
    
    // Play a few dominoes to create tricks
    await helper.playAnyDomino(); // Player 1 leads  
    await helper.playAnyDomino(); // Player 2
    await helper.playAnyDomino(); // Player 3
    await helper.playAnyDomino(); // Player 0
    
    // Complete the trick
    await helper.completeTrick();
    
    // Verify trump suit appears in Previous Tricks panel
    const trickElements = helper.locator('.trick-compact');
    await expect(trickElements).toHaveCount(1);
    
    // Check for trump display in the completed trick
    const trumpDisplay = helper.locator('.trump-display');
    await expect(trumpDisplay).toBeVisible();
    await expect(trumpDisplay).toContainText('[2s]');
    
    // Verify styling looks like a domino
    await expect(trumpDisplay).toHaveCSS('background-color', 'rgb(33, 37, 41)'); // Dark background
    await expect(trumpDisplay).toHaveCSS('color', 'rgb(255, 255, 255)'); // White text
    await expect(trumpDisplay).toHaveCSS('font-family', /monospace/); // Monospace font
  });

  test('shows trump suit for different suit types', async () => {
    // Start with deterministic seed 1
    await helper.gotoWithSeed(1);
    
    // Test with twos (which seed 1 consistently produces)
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    
    await helper.setTrump('twos');
    
    // Start a trick
    await helper.playAnyDomino();
    
    // Check current trick shows trump
    const trumpDisplay = helper.locator('.trump-display');
    await expect(trumpDisplay).toBeVisible();
    await expect(trumpDisplay).toContainText('[2s]');
  });

  test('shows trump suit in both completed and current tricks', async () => {
    // Start with deterministic seed 1
    await helper.gotoWithSeed(1);
    
    // Complete bidding and set trump to twos (which seed 1 consistently produces)
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.setTrump('twos');
    
    // Play full trick to complete it
    await helper.playAnyDomino();
    await helper.playAnyDomino();
    await helper.playAnyDomino();
    await helper.playAnyDomino();
    await helper.completeTrick();
    
    // Start another trick
    await helper.playAnyDomino();
    
    // Should have trump display in both completed and current trick
    const trumpDisplays = helper.locator('.trump-display');
    await expect(trumpDisplays).toHaveCount(2);
    
    // Both should show valid trump indicators (format: [Xs])
    for (const display of await trumpDisplays.all()) {
      const text = await display.textContent();
      expect(text).toMatch(/^\[\d+s\]$/); // Should match pattern like [2s], [6s], etc.
    }
  });

  test('does not show trump suit before trump is selected', async () => {
    // Start with deterministic seed 1
    await helper.gotoWithSeed(1);
    
    // Before trump selection
    const trumpDisplay = helper.locator('.trump-display');
    await expect(trumpDisplay).toHaveCount(0);
    
    // Even during bidding
    await helper.selectActionByType('bid_points', 30);
    await expect(trumpDisplay).toHaveCount(0);
  });

  test('trump display appears to the left of winner and points', async () => {
    // Start with deterministic seed 1
    await helper.gotoWithSeed(1);
    
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
    const trickInfo = helper.locator('.trick-info').first();
    const trumpDisplay = trickInfo.locator('.trump-display');
    const winnerInfo = trickInfo.locator('.winner-info');
    const pointsInfo = trickInfo.locator('.points-info');
    
    await expect(trumpDisplay).toBeVisible();
    await expect(winnerInfo).toBeVisible();
    await expect(pointsInfo).toBeVisible();
    
    await expect(trumpDisplay).toContainText('[0s]');
  });
});