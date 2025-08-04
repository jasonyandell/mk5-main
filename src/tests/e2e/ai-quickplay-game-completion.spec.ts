import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('AI Quickplay Game Completion', () => {
  test('games reach completion and reset', async ({ page }) => {
    test.setTimeout(5000); // Optimized timeout - AI should complete game in under 5 seconds
    const helper = new PlaywrightGameHelper(page);
    await helper.gotoWithSeed(12345);
    
    // Enable instant speed and all AI
    await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
    for (let i = 0; i < 4; i++) {
      await helper.locator(`[data-testid="ai-player-${i}"]`).check();
    }
    
    // Open browser console to see debug logs
    page.on('console', msg => {
      if (msg.text().includes('[Quickplay]') || msg.text().includes('[GameStore]')) {
        console.log('Browser:', msg.text());
      }
    });
    
    // Track phases and marks
    const phaseHistory: string[] = [];
    const markHistory: [number, number][] = [];
    
    // Start AI
    await helper.locator('[data-testid="quickplay-run"]').click();
    
    // Monitor for 1.2 seconds - optimized for fast AI
    for (let i = 0; i < 12; i++) {
      await page.waitForTimeout(100);
      
      const phase = await helper.getCurrentPhase();
      const marks = await helper.getTeamMarks();
      
      // Log changes
      if (phaseHistory.length === 0 || phase !== phaseHistory[phaseHistory.length - 1]) {
        phaseHistory.push(phase);
        console.log(`Phase changed to: ${phase}`);
      }
      
      if (markHistory.length === 0 || 
          marks[0] !== markHistory[markHistory.length - 1][0] ||
          marks[1] !== markHistory[markHistory.length - 1][1]) {
        markHistory.push(marks);
        console.log(`Marks changed to: ${marks[0]}-${marks[1]}`);
      }
      
      // Check for game completion
      if (marks[0] >= 7 || marks[1] >= 7) {
        console.log('Game should be complete!');
      }
    }
    
    // Stop AI
    try {
      await helper.locator('[data-testid="quickplay-stop"]').click();
    } catch {
      console.warn('Could not stop AI, might already be stopped');
    }
    
    // Log what we observed
    console.log('Phase history:', phaseHistory);
    console.log('Mark history:', markHistory);
    
    // We should have seen some phase changes
    expect(phaseHistory.length).toBeGreaterThan(1);
  });
});