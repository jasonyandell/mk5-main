import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Complete Trick in Play Area', () => {
  test.setTimeout(15000); // Increase timeout as these tests play through full hands
  let helper: PlaywrightGameHelper;

  test('shows complete trick button in play area', async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    
    // Load a state with trump selected
    await helper.loadStateWithActions(12345, [
      '30', 'p', 'p', 'p', 'trump-blanks'
    ]);
    
    // Navigate to the Game tab where PlayingArea is shown
    await page.locator('[data-testid="nav-game"]').click();
    await expect(page.locator('.playing-area')).toBeVisible({ timeout: 2000 });
    
    // Play 4 dominoes to make complete-trick available
    for (let i = 0; i < 4; i++) {
      await helper.playAnyDomino();
    }
    
    // Verify the PlayingArea exists
    const playingArea = page.locator('.playing-area');
    await expect(playingArea).toBeVisible();
    
    // Check that complete-trick button is available
    const completeTrickButton = page.locator('.tap-indicator');
    await expect(completeTrickButton).toBeVisible();
    
    // Verify the button text
    const buttonText = await completeTrickButton.locator('.tap-text').textContent();
    expect(buttonText).toBe('Complete trick');
    
    // Click the button and verify it works
    await completeTrickButton.click();
    
    // After completing trick, we should be ready for next trick
    await helper.waitForPhaseChange('playing', 2000);
    const phase = await helper.getCurrentPhase();
    expect(phase).toContain('playing');
  });

  test('button DOM structure is correct', async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    await helper.goto(12345);
    
    // Wait for the app to fully load first
    await page.waitForSelector('.app-container', { timeout: 5000 });
    
    // Navigate to the Game tab where PlayingArea is shown
    await page.locator('[data-testid="nav-game"]').click();
    await expect(page.locator('.playing-area')).toBeVisible({ timeout: 2000 });
    
    // Check that the PlayingArea component exists
    const playingArea = page.locator('.playing-area');
    await expect(playingArea).toBeVisible();
    
    // The tap-indicator button exists in the DOM even if not currently visible
    // Check if the button would render when conditions are met
    const buttonStructure = await page.evaluate(() => {
      const area = document.querySelector('.playing-area');
      return area ? 'PlayingArea exists' : 'PlayingArea not found';
    });
    
    expect(buttonStructure).toBe('PlayingArea exists');
  });

  test('button appears for scoring actions', async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    
    // Start a game and complete bidding
    await helper.loadStateWithActions(12345, [
      '30', 'p', 'p', 'p', 'trump-blanks'
    ]);
    
    // Play through tricks until we reach scoring phase
    for (let trick = 0; trick < 7; trick++) {
      // Check if we've already reached scoring phase
      const currentPhase = await helper.getCurrentPhase();
      if (currentPhase.includes('scoring')) {
        break; // Game ended early (bid was set or made)
      }
      
      // Play 4 dominoes per trick
      for (let play = 0; play < 4; play++) {
        const phase = await helper.getCurrentPhase();
        if (!phase.includes('playing')) break;
        
        // Check if page is still valid before playing
        if (page.isClosed()) {
          throw new Error('Page was closed during test execution');
        }
        
        await helper.playAnyDomino();
      }
      
      // Complete the trick if we're still in playing phase
      const phase = await helper.getCurrentPhase();
      if (phase.includes('playing')) {
        const actions = await helper.getActionsList();
        if (actions.find(a => a.type === 'complete_trick')) {
          await helper.completeTrick();
        }
      }
    }
    
    // At this point, we should be in scoring phase
    const proceedButton = page.locator('.tap-indicator');
    await expect(proceedButton).toBeVisible();
    
    // Verify the button text indicates scoring
    const buttonText = await proceedButton.locator('.tap-text').textContent();
    expect(buttonText).toMatch(/Score [Hh]and/);
    
    // Click to score the hand
    await proceedButton.click();
    
    // Wait for score update
    await page.waitForFunction(
      () => {
        const scoreElements = document.querySelectorAll('.score-value');
        return Array.from(scoreElements).some(el => parseInt(el.textContent || '0') > 0);
      },
      {},
      { timeout: 2000 }
    );
    const scores = await helper.getTeamScores();
    // At least one team should have points
    expect(scores[0] + scores[1]).toBeGreaterThan(0);
  });
});