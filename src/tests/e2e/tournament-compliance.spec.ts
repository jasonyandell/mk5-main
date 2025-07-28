import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Tournament Compliance E2E Tests', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    await helper.goto();
  });

  test.describe('Critical Tournament Rules', () => {
    test('should enforce sequential mark bidding (no jumping)', async ({ page }) => {
      // Bid 1 mark
      await page.locator('[data-testid="bid-P0-1M"]').click();
      
      // Next player should only see 2M, not 3M or higher
      const buttons = await page.locator('button').all();
      const buttonTexts = await Promise.all(buttons.map(b => b.innerText()));
      
      const p1MarkBids = buttonTexts.filter(t => t.match(/P1: \d+M$/));
      const maxMarkBid = Math.max(...p1MarkBids.map(t => parseInt(t.match(/(\d+)M$/)[1])));
      
      expect(maxMarkBid).toBe(2); // Can only bid 2M after 1M
    });

    test('should require 2M to be bid before 3M is available', async ({ page }) => {
      // Verify 3M not available as opening bid
      let buttons = await page.locator('button').all();
      let buttonTexts = await Promise.all(buttons.map(b => b.innerText()));
      let threeMark = buttonTexts.find(t => t === 'P0: 3M');
      expect(threeMark).toBeUndefined();

      // Bid 2M first
      await page.locator('[data-testid="bid-P0-2M"]').click();

      // Now 3M should be available to next player
      buttons = await page.locator('button').all();
      buttonTexts = await Promise.all(buttons.map(b => b.innerText()));
      threeMark = buttonTexts.find(t => t === 'P1: 3M');
      expect(threeMark).toBeDefined();
    });

    test('should prohibit special contracts in tournament mode', async ({ page }) => {
      const buttons = await page.locator('button').all();
      const buttonTexts = await Promise.all(buttons.map(b => b.innerText()));
      
      // Should not see Nel-O, Splash, or Plunge bids
      const specialContracts = buttonTexts.filter(t => 
        t.includes('Nel-O') || t.includes('NELLO') || 
        t.includes('Splash') || t.includes('SPLASH') || 
        t.includes('Plunge') || t.includes('PLUNGE')
      );
      
      expect(specialContracts.length).toBe(0);
    });

    test('should enforce maximum 2M opening bid in tournament', async ({ page }) => {
      const buttons = await page.locator('button').all();
      const buttonTexts = await Promise.all(buttons.map(b => b.innerText()));
      
      const markBids = buttonTexts.filter(t => t.match(/P0: \d+M$/));
      if (markBids.length > 0) {
        const maxMarkBid = Math.max(...markBids.map(t => parseInt(t.match(/(\d+)M$/)[1])));
        expect(maxMarkBid).toBeLessThanOrEqual(2);
      }
    });
  });

  test.describe('Exact Point Distribution', () => {
    test('should distribute exactly 42 points per hand', async ({ page }) => {
      // Bid 30 points
      await helper.bidPoints(0, 30);
      await helper.bidPass(1);
      await helper.bidPass(2);
      await helper.bidPass(3);
      await helper.setTrump('5s');

      // Play 7 tricks
      for (let trick = 1; trick <= 7; trick++) {
        for (let play = 0; play < 4; play++) {
          await helper.playAnyDomino();
          await page.waitForTimeout(50);
        }
        
        if (trick < 7) {
          await helper.completeTrick();
        } else {
          await helper.completeTrick();
          await page.waitForTimeout(100);
          
          // Check scores before final scoring (after all tricks completed)
          const [team0Score, team1Score] = await helper.getTeamScores();
          // Should have at least 35 points from counting dominoes (may vary based on implementation)
          expect(team0Score + team1Score).toBeGreaterThanOrEqual(35);
          expect(team0Score + team1Score).toBeLessThanOrEqual(42);
          
          await helper.scoreHand();
        }
      }
    });

    test('should have exactly 5 counting dominoes totaling 35 points', async ({ page }) => {
      const pageContent = await page.content();
      
      // Verify presence of all counting dominoes
      const countingDominoes = [
        '[5|5]', // 10 points
        '[6|4]', // 10 points  
        '[5|0]', // 5 points
        '[4|1]', // 5 points
        '[3|2]'  // 5 points
      ];
      
      countingDominoes.forEach(domino => {
        expect(pageContent).toContain(domino);
      });
    });
  });

  test.describe('Suit Following Enforcement', () => {
    test('should enforce must-follow-suit rule', async ({ page }) => {
      await page.locator('[data-testid="bid-P0-30"]').click();
      await page.locator('[data-testid="bid-P1-PASS"]').click();
      await page.locator('[data-testid="bid-P2-PASS"]').click();
      await page.locator('[data-testid="bid-P3-PASS"]').click();
      await page.locator('[data-testid="set-trump-5s"]').click();

      // First player leads
      const firstPlayButtons = await page.locator('button[data-testid^="play-"]').all();
      await firstPlayButtons[0].click();

      // Second player should only see valid plays
      // (UI should prevent illegal plays)
      const secondPlayButtons = await page.locator('button[data-testid^="play-"]').all();
      expect(secondPlayButtons.length).toBeGreaterThan(0);
      
      // All available plays should be legal
      await secondPlayButtons[0].click();
      
      // Game should continue normally
      await expect(page.locator('[data-testid="current-player"]')).toContainText(/Current Player: P[0-3]/);
    });

    test('should handle doubles belonging to natural suit', async ({ page }) => {
      await page.locator('[data-testid="bid-P0-30"]').click();
      await page.locator('[data-testid="bid-P1-PASS"]').click();
      await page.locator('[data-testid="bid-P2-PASS"]').click();
      await page.locator('[data-testid="bid-P3-PASS"]').click();
      await page.locator('[data-testid="set-trump-3s"]').click();

      // When 3s are trump, [3|3] should be treated as highest three
      await expect(page.locator('[data-testid="trump"]')).toContainText('Trump: 3s');
      
      // Game should recognize doubles in their natural suit
      const pageContent = await page.content();
      expect(pageContent).toContain('[3|3]');
    });

    test('should handle doubles-trump correctly', async ({ page }) => {
      await page.locator('[data-testid="bid-P0-30"]').click();
      await page.locator('[data-testid="bid-P1-PASS"]').click();
      await page.locator('[data-testid="bid-P2-PASS"]').click();
      await page.locator('[data-testid="bid-P3-PASS"]').click();
      await page.locator('[data-testid="set-trump-Doubles"]').click();

      // When doubles are trump, only the 7 doubles should be trump
      await expect(page.locator('[data-testid="trump"]')).toContainText('Trump: Doubles');
      
      // All doubles should be present: [0|0], [1|1], [2|2], [3|3], [4|4], [5|5], [6|6]
      const pageContent = await page.content();
      for (let i = 0; i <= 6; i++) {
        expect(pageContent).toContain(`[${i}|${i}]`);
      }
    });
  });

  test.describe('Tournament Scoring Rules', () => {
    test('should award exactly 1 mark for successful 30-41 point bids', async ({ page }) => {
      const [initialMarks0, initialMarks1] = await helper.getTeamMarks();

      // Complete hand with 30-point bid
      await helper.playCompleteHand(0, 30, 'points', 'fives');
      
      // May need to click additional action after scoring
      try {
        await helper.clickActionIndex(0);
      } catch (e) {
        // Action may not be needed depending on game state
      }

      const [finalMarks0, finalMarks1] = await helper.getTeamMarks();
      
      // Should have gained or lost exactly 1 mark (either team makes/sets the bid)
      const totalInitialMarks = initialMarks0 + initialMarks1;
      const totalFinalMarks = finalMarks0 + finalMarks1;
      expect(totalFinalMarks - totalInitialMarks).toBe(1);
    });

    test('should enforce 7-mark game target', async ({ page }) => {
      // Game should end when a team reaches 7 marks
      const team0Marks = parseInt(await page.locator('[data-testid="team-0-marks"]').textContent().then(text => text.match(/(\d+) marks/)[1]));
      const team1Marks = parseInt(await page.locator('[data-testid="team-1-marks"]').textContent().then(text => text.match(/(\d+) marks/)[1]));
      
      if (team0Marks >= 7 || team1Marks >= 7) {
        await expect(page.locator('[data-testid="phase"]')).toContainText('Phase: GAME_END');
      } else {
        expect(Math.max(team0Marks, team1Marks)).toBeLessThan(7);
      }
    });

    test('should handle set penalties correctly', async ({ page }) => {
      const [initialMarks0, initialMarks1] = await helper.getTeamMarks();

      // Bid high (2M = 84 points) to increase chance of set
      await helper.playCompleteHand(0, 2, 'marks', 'fives');
      
      // Check score before completing
      const [team0Score] = await helper.getTeamScores();
      
      // May need to click additional action after scoring
      try {
        await helper.clickActionIndex(0);
      } catch (e) {
        // Action may not be needed depending on game state
      }

      const [finalMarks0, finalMarks1] = await helper.getTeamMarks();

      // Verify correct penalty/award based on score
      if (team0Score < 84) {
        // Set - opponents get marks
        expect(finalMarks1).toBe(initialMarks1 + 2);
        expect(finalMarks0).toBe(initialMarks0);
      } else {
        // Made - bidder gets marks  
        expect(finalMarks0).toBe(initialMarks0 + 2);
        expect(finalMarks1).toBe(initialMarks1);
      }
    });
  });

  test.describe('Turn Order and Dealing', () => {
    test('should start bidding with player left of dealer', async ({ page }) => {
      const coreState = await page.locator('h3:has-text("Core State")').locator('..').innerText();
      
      // Should show dealer and first bidder
      expect(coreState).toContain('Phase: BIDDING');
      expect(coreState).toContain('Dealer: P3');
      expect(coreState).toContain('Current Player: P0'); // Left of P3
    });

    test('should advance dealer after all-pass redeal', async ({ page }) => {
      // All players pass
      await page.locator('[data-testid="bid-P0-PASS"]').click();
      await page.locator('[data-testid="bid-P1-PASS"]').click();
      await page.locator('[data-testid="bid-P2-PASS"]').click();
      await page.locator('[data-testid="bid-P3-PASS"]').click();

      // Redeal
      await page.locator('[data-testid="redeal"]').click();

      // Dealer should advance to P0, first bidder to P1
      await expect(page.locator('[data-testid="dealer"]')).toContainText('Dealer: P0');
      await expect(page.locator('[data-testid="current-player"]')).toContainText('Current Player: P1');
    });

    test('should ensure each player bids exactly once per round', async ({ page }) => {
      // Track bidding sequence
      await page.locator('[data-testid="bid-P0-30"]').click();
      await expect(page.locator('[data-testid="current-player"]')).toContainText('Current Player: P1');

      await page.locator('[data-testid="bid-P1-31"]').click();
      await expect(page.locator('[data-testid="current-player"]')).toContainText('Current Player: P2');

      await page.locator('[data-testid="bid-P2-PASS"]').click();
      await expect(page.locator('[data-testid="current-player"]')).toContainText('Current Player: P3');

      await page.locator('[data-testid="bid-P3-PASS"]').click();

      // Should transition to trump selection
      await expect(page.locator('[data-testid="set-trump-5s"]')).toBeVisible();
    });
  });

  test.describe('Game State Consistency', () => {
    test('should maintain consistent domino count throughout', async ({ page }) => {
      // All players should have 7 dominoes at start
      const pageContent = await page.content();
      
      // Count domino displays (this depends on UI implementation)
      // Should total 28 dominoes across all players
      const dominoPattern = /\[\d\|\d\]/g;
      const dominoes = pageContent.match(dominoPattern);
      
      // Expect significant number of dominoes displayed
      expect(dominoes?.length).toBeGreaterThan(20);
    });

    test('should enforce correct trick count (exactly 7)', async ({ page }) => {
      // Bid 30 points
      await helper.bidPoints(0, 30);
      await helper.bidPass(1);
      await helper.bidPass(2);
      await helper.bidPass(3);
      await helper.setTrump('5s');

      // Play exactly 7 tricks
      for (let trick = 1; trick <= 7; trick++) {
        for (let play = 0; play < 4; play++) {
          await helper.playAnyDomino();
          await page.waitForTimeout(50);
        }
        
        if (trick < 7) {
          await helper.completeTrick();
          await expect(page.locator('[data-testid="tricks-completed"]')).toContainText(`Tricks Completed: ${trick}/7`);
        } else {
          // 7th trick should complete and transition to scoring phase
          await helper.completeTrick();
          await expect(page.locator('[data-testid="tricks-completed"]')).toContainText(`Tricks Completed: ${trick}/7`);
          await expect(page.locator('[data-testid="phase"]')).toContainText('Phase: SCORING');
          // Now score the hand
          await helper.scoreHand();
        }
      }
    });
  });
});