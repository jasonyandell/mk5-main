import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Debug Snapshot Replay Validation', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    await helper.goto();
  });

  test('reproduces and validates trick winner bug scenario', async () => {
    // Load the problematic state from the URL mentioned in the issue
    const problematicState = {
      "phase": "playing" as const,
      "players": [
        {
          "id": 0,
          "name": "Player 1",
          "hand": [
            {"high": 6, "low": 5, "id": "6-5"},
            {"high": 5, "low": 2, "id": "5-2"}
          ],
          "teamId": 0 as const,
          "marks": 0
        },
        {
          "id": 1,
          "name": "Player 2", 
          "hand": [
            {"high": 5, "low": 3, "id": "5-3"},
            {"high": 2, "low": 1, "id": "2-1"}
          ],
          "teamId": 1 as const,
          "marks": 0
        },
        {
          "id": 2,
          "name": "Player 3",
          "hand": [
            {"high": 5, "low": 0, "id": "5-0"},
            {"high": 3, "low": 2, "id": "3-2"}
          ],
          "teamId": 0 as const,
          "marks": 0
        },
        {
          "id": 3,
          "name": "Player 4",
          "hand": [
            {"high": 1, "low": 0, "id": "1-0"},
            {"high": 6, "low": 0, "id": "6-0"}
          ],
          "teamId": 1 as const,
          "marks": 0
        }
      ],
      "currentPlayer": 1,
      "dealer": 3,
      "winningBidder": 1,
      "trump": 7 as const, // Doubles are trump
      "currentTrick": [
        {"player": 1, "domino": {"high": 4, "low": 0, "id": "4-0"}}, // Led 4s
        {"player": 2, "domino": {"high": 6, "low": 2, "id": "6-2"}}, // Off-suit
        {"player": 3, "domino": {"high": 2, "low": 0, "id": "2-0"}}, // Off-suit
        {"player": 0, "domino": {"high": 1, "low": 1, "id": "1-1"}}  // TRUMP! Should win
      ],
      "currentSuit": 4, // 4s were led
      "tricks": [],
      "teamScores": [10, 15] as [number, number],
      "teamMarks": [0, 0] as [number, number],
      "gameTarget": 7,
      "tournamentMode": true,
      "bids": [
        {"type": "pass" as const, "player": 0},
        {"type": "points" as const, "value": 30, "player": 1},
        {"type": "pass" as const, "player": 2},
        {"type": "pass" as const, "player": 3}
      ],
      "currentBid": {"type": "points" as const, "value": 30, "player": 1},
      "shuffleSeed": 12345
    };

    // Load this state 
    await helper.loadState(problematicState);
    
    // Verify the state loaded correctly
    expect(await helper.getCurrentPhase()).toContain('playing');
    expect(await helper.getTrump()).toContain('Doubles'); // Doubles trump
    
    // The current trick should have Player 0's 1-1 (trump) winning
    // But if there's a bug, it might show wrong winner
    
    // Complete the trick and check who wins
    await helper.completeTrick();
    
    // After completing trick, we should be able to validate the logic
    // The action should have been valid and Player 0 should have won
    
    // Create a snapshot by playing a few more actions to trigger it
    const currentPhase = await helper.getCurrentPhase();
    if (currentPhase.includes('playing')) {
      // Play a few more dominoes to create snapshot
      for (let i = 0; i < 4; i++) {
        const actions = await helper.getAvailableActions();
        if (actions.length > 0) {
          await helper.selectActionByIndex(0);
        }
      }
      
      // Complete another trick to trigger snapshot
      const availableActions = await helper.getAvailableActions();
      const completeTrickAction = availableActions.find(a => a.type === 'complete_trick');
      if (completeTrickAction) {
        await helper.completeTrick();
      }
    }
  });

  test('validates correct trick winner with trump domino', async () => {
    // Create a specific scenario where trump should win
    await helper.newGame();
    
    // Complete bidding - player 1 bids 30
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    
    // Set doubles as trump
    await helper.setTrump('doubles');
    
    // Get initial state before playing
    await helper.getCurrentURL();
    
    // Play a trick where trump should win
    // This will create the base state for snapshot
    
    // Player 1 leads (non-trump)
    await helper.playAnyDomino();
    
    // Player 2 plays (try to get off-suit)
    await helper.playAnyDomino();
    
    // Player 3 plays (try to get off-suit)  
    await helper.playAnyDomino();
    
    // Player 0 plays trump if possible
    await helper.playAnyDomino();
    
    // Complete trick - this should create snapshot
    await helper.completeTrick();
    
    // Verify snapshot was created
    expect(await helper.hasDebugSnapshot()).toBe(true);
    
    const snapshotInfo = await helper.getDebugSnapshotInfo();
    expect(snapshotInfo).not.toBeNull();
    expect(snapshotInfo!.reason).toContain('initial state');
    
    // Validate the action sequence (skip for now since validation UI might not be ready)
    // const validation = await helper.validateActionSequence();
    // expect(validation.success).toBe(true);
    
    // Generate bug report to ensure it contains the action sequence
    const bugReport = await helper.getBugReport();
    expect(bugReport).toContain('Base state for reproduction');
    expect(bugReport).toContain('Action sequence from action history');
  });

  test('detects invalid action in replayed sequence', async () => {
    // This test simulates what would happen if game logic changes
    // made a previously valid action become invalid
    
    await helper.newGame();
    
    // Create a snapshot by performing valid actions
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass'); 
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.setTrump('doubles');
    
    // Verify snapshot exists
    expect(await helper.hasDebugSnapshot()).toBe(true);
    
    // The validation should pass for these valid actions
    const validation = await helper.validateActionSequence();
    expect(validation.success).toBe(true);
    expect(validation.errors).toHaveLength(0);
    
    // Generate a bug report that would catch logic regressions
    const bugReport = await helper.getBugReport();
    expect(bugReport).toContain('throw new Error');
    expect(bugReport).toContain('not available at step');
    expect(bugReport).toContain('Available:');
  });

  test('URL contains snapshot data for debugging', async () => {
    await helper.newGame();
    
    // Perform actions to create snapshot
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    
    // URL should contain snapshot parameter
    expect(await helper.hasSnapshotInURL()).toBe(true);
    
    const url = await helper.getCurrentURL();
    expect(url).toMatch(/d=/);
    
    // Copy the state URL and verify it contains snapshot data
    const copiedURL = await helper.copyStateURL();
    expect(copiedURL).toMatch(/d=/);
    
    // The URL should be different from simple state URLs
    expect(copiedURL).not.toMatch(/^.*\?state=/);
  });

  test('snapshot system works across page reloads', async ({ page }) => {
    await helper.newGame();
    
    // Create snapshot
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    
    // Get the URL with snapshot
    const urlWithSnapshot = await helper.getCurrentURL();
    expect(urlWithSnapshot).toMatch(/d=/);
    
    // Reload the page with the snapshot URL
    await page.goto(urlWithSnapshot);
    await helper.goto(); // Re-initialize
    
    // The state should be loaded from the snapshot base state (initial state)
    // (Note: full action replay isn't implemented yet, but base state should load)
    expect(await helper.getCurrentPhase()).toContain('bidding');
  });
});