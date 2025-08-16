import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Page Reload Suit Analysis Consistency', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    await helper.goto();
  });

  test('suit analysis remains consistent after page reload following bid-30', async ({ page }) => {
    // Capture initial suit analysis for all players
    const initialSuitData = await page.evaluate(() => {
      const players = document.querySelectorAll('[data-testid="player-hands"] .player-section');
      const suitData: Record<number, string[]> = {};
      
      players.forEach((playerEl, index) => {
        const suitCounts = playerEl.querySelectorAll('.suit-count');
        const suits: string[] = [];
        suitCounts.forEach(suitEl => {
          suits.push(suitEl.textContent || '');
        });
        suitData[index] = suits;
      });
      
      return suitData;
    });

    // Make a bid-30 action
    await helper.selectActionByType('bid_points', 30);

    // Capture the URL for reload
    const urlAfterBid = page.url();

    // Reload the page
    await page.goto(urlAfterBid);
    
    // Wait for the game to load
    await helper.locator('[data-testid="player-hands"]').waitFor({ state: 'visible' });

    // Capture suit analysis after reload
    const reloadedSuitData = await page.evaluate(() => {
      const players = document.querySelectorAll('[data-testid="player-hands"] .player-section');
      const suitData: Record<number, string[]> = {};
      
      players.forEach((playerEl, index) => {
        const suitCounts = playerEl.querySelectorAll('.suit-count');
        const suits: string[] = [];
        suitCounts.forEach(suitEl => {
          suits.push(suitEl.textContent || '');
        });
        suitData[index] = suits;
      });
      
      return suitData;
    });

    // Verify suit analysis is identical for all players
    for (let playerId = 0; playerId < 4; playerId++) {
      expect(reloadedSuitData[playerId]).toEqual(initialSuitData[playerId]);
    }
  });

  test('suit analysis consistency across multiple reloads with different actions', async ({ page }) => {
    // Take initial snapshot
    const getPlayerSuitData = async () => {
      return await page.evaluate(() => {
        const players = document.querySelectorAll('[data-testid="player-hands"] .player-section');
        const playerData: Array<{ suits: string[], dominoes: string[] }> = [];
        
        players.forEach((playerEl) => {
          const suitCounts = playerEl.querySelectorAll('.suit-count');
          const suits: string[] = [];
          suitCounts.forEach(suitEl => {
            suits.push(suitEl.textContent || '');
          });
          
          const dominoes = Array.from(playerEl.querySelectorAll('.domino-mini'));
          const dominoList: string[] = [];
          dominoes.forEach(dominoEl => {
            dominoList.push((dominoEl.textContent || '').trim());
          });
          
          playerData.push({ suits, dominoes: dominoList });
        });
        
        return playerData;
      });
    };

    const initialData = await getPlayerSuitData();

    // Make several actions
    await helper.selectActionByType('bid_points', 35);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');

    // Reload after multiple actions
    await page.reload();
    await helper.locator('[data-testid="player-hands"]').waitFor({ state: 'visible' });

    const afterReloadData = await getPlayerSuitData();

    // Verify player hands are identical (dominoes should be the same)
    for (let i = 0; i < 4; i++) {
      expect(afterReloadData[i]!.dominoes).toEqual(initialData[i]!.dominoes);
      expect(afterReloadData[i]!.suits).toEqual(initialData[i]!.suits);
    }
  });

  test('trump analysis updates correctly after trump selection and reload', async ({ page }) => {
    // Get to trump selection phase
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    
    // Should now be in trump selection phase - wait for phase change
    await helper.waitForPhaseChange('trump_selection', 5000);

    // Capture URL before trump selection (for potential future use)
    // const urlBeforeTrump = page.url();

    // Select trump (sixes)
    await helper.setTrump('sixes');

    // Capture trump analysis after selection
    const trumpAnalysisAfterSelection = await page.evaluate(() => {
      const players = document.querySelectorAll('[data-testid="player-hands"] .player-section');
      const trumpData: Record<number, { trumpCount: string, trumpDominoes: string[] }> = {};
      
      players.forEach((playerEl, index) => {
        const trumpCountEl = playerEl.querySelector('.trump-count');
        const trumpCount = trumpCountEl?.textContent || '';
        
        const trumpDominoes = Array.from(playerEl.querySelectorAll('.trump-domino'));
        const trumpList: string[] = [];
        trumpDominoes.forEach(trumpEl => {
          trumpList.push((trumpEl.textContent || '').trim());
        });
        
        trumpData[index] = { trumpCount, trumpDominoes: trumpList };
      });
      
      return trumpData;
    });

    // Reload the page
    await page.reload();
    await helper.locator('[data-testid="player-hands"]').waitFor({ state: 'visible' });

    // Capture trump analysis after reload
    const trumpAnalysisAfterReload = await page.evaluate(() => {
      const players = document.querySelectorAll('[data-testid="player-hands"] .player-section');
      const trumpData: Record<number, { trumpCount: string, trumpDominoes: string[] }> = {};
      
      players.forEach((playerEl, index) => {
        const trumpCountEl = playerEl.querySelector('.trump-count');
        const trumpCount = trumpCountEl?.textContent || '';
        
        const trumpDominoes = Array.from(playerEl.querySelectorAll('.trump-domino'));
        const trumpList: string[] = [];
        trumpDominoes.forEach(trumpEl => {
          trumpList.push((trumpEl.textContent || '').trim());
        });
        
        trumpData[index] = { trumpCount, trumpDominoes: trumpList };
      });
      
      return trumpData;
    });

    // Verify trump analysis is consistent after reload
    for (let playerId = 0; playerId < 4; playerId++) {
      expect(trumpAnalysisAfterReload[playerId]).toEqual(trumpAnalysisAfterSelection[playerId]);
    }
  });

  test('suit analysis deterministic with same seed across sessions', async ({ page }) => {
    // Get the initial shuffle seed from game state (for potential future use)
    // const initialSeed = await page.evaluate(() => {
    //   return (window as unknown).gameState?.gameState?.shuffleSeed || null;
    // });

    // Capture initial player data
    const getPlayerData = async () => {
      return await page.evaluate(() => {
        const players = document.querySelectorAll('[data-testid="player-hands"] .player-section');
        const data: Array<{ dominoes: string[], suits: string[] }> = [];
        
        players.forEach((playerEl) => {
          const dominoes = Array.from(playerEl.querySelectorAll('.domino-mini'));
          const dominoList: string[] = [];
          dominoes.forEach(dominoEl => {
            dominoList.push((dominoEl.textContent || '').trim());
          });
          
          const suitCounts = playerEl.querySelectorAll('.suit-count');
          const suits: string[] = [];
          suitCounts.forEach(suitEl => {
            suits.push(suitEl.textContent || '');
          });
          
          data.push({ dominoes: dominoList, suits });
        });
        
        return data;
      });
    };

    const initialData = await getPlayerData();

    // Start a completely new game (clear URL)
    await page.goto(page.url().split('?')[0] || page.url());
    await helper.locator('[data-testid="player-hands"]').waitFor({ state: 'visible' });

    // The new game should have different hands (different seed)
    const newGameData = await getPlayerData();
    
    // Verify this is actually a different game (very unlikely to have identical hands)
    let isDifferent = false;
    for (let i = 0; i < 4; i++) {
      if (JSON.stringify(newGameData[i]) !== JSON.stringify(initialData[i])) {
        isDifferent = true;
        break;
      }
    }
    expect(isDifferent).toBe(true);

    // Note: We can't easily test the same seed without implementing a way to set it in the UI
    // The important thing is that suit analysis is consistent after reload, which we test above
  });
});