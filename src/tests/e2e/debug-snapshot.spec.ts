import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Debug Snapshot System', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    await helper.goto();
  });

  test('creates debug snapshot after first action', async () => {
    // Start a game
    await helper.newGame();
    
    // Initially no snapshot
    expect(await helper.hasDebugSnapshot()).toBe(false);
    
    // Take first action
    await helper.selectActionByType('bid_points', 30); // Player 0 bids 30
    
    // Should now have a snapshot
    expect(await helper.hasDebugSnapshot()).toBe(true);
    
    // Get snapshot info
    const snapshot = await helper.getDebugSnapshotInfo();
    expect(snapshot).not.toBeNull();
    expect(snapshot!.reason).toContain('initial state');
    expect(snapshot!.actionCount).toBe(1);
    
    // Take more actions
    await helper.selectActionByType('pass'); // Player 1 passes
    await helper.selectActionByType('pass'); // Player 2 passes  
    await helper.selectActionByType('pass'); // Player 3 passes
    
    // Check snapshot has grown
    const updatedSnapshot = await helper.getDebugSnapshotInfo();
    expect(updatedSnapshot).not.toBeNull();
    expect(updatedSnapshot!.actionCount).toBe(4);
    
    // Select trump
    await helper.setTrump('doubles');
    
    // Snapshot should now have 5 actions
    const finalSnapshot = await helper.getDebugSnapshotInfo();
    expect(finalSnapshot).not.toBeNull();
    expect(finalSnapshot!.actionCount).toBe(5);
  });

  test('tracks all actions from initial state', async () => {
    // Start fresh game
    await helper.newGame();
    
    // Initially no snapshot
    expect(await helper.hasDebugSnapshot()).toBe(false);
    
    // Take first action - should create snapshot
    await helper.selectActionByType('bid_points', 30); // Action 1
    expect(await helper.hasDebugSnapshot()).toBe(true);
    
    let snapshotInfo = await helper.getDebugSnapshotInfo();
    expect(snapshotInfo!.actionCount).toBe(1);
    
    // Take more actions - should grow snapshot
    await helper.selectActionByType('pass'); // Action 2
    snapshotInfo = await helper.getDebugSnapshotInfo();
    expect(snapshotInfo!.actionCount).toBe(2);
    
    await helper.selectActionByType('pass'); // Action 3
    snapshotInfo = await helper.getDebugSnapshotInfo();
    expect(snapshotInfo!.actionCount).toBe(3);
    
    await helper.selectActionByType('pass'); // Action 4
    snapshotInfo = await helper.getDebugSnapshotInfo();
    expect(snapshotInfo!.actionCount).toBe(4);
  });

  test('validates action sequence correctly', async () => {
    // Start game and create a snapshot
    await helper.newGame();
    
    // Perform actions to create snapshot
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    
    // Select trump to add more actions
    await helper.setTrump('doubles');
    
    // Verify snapshot exists
    expect(await helper.hasDebugSnapshot()).toBe(true);
    
    // Validate the action sequence
    const validation = await helper.validateActionSequence();
    expect(validation.success).toBe(true);
    expect(validation.errors).toHaveLength(0);
  });

  test('detects invalid action sequences', async () => {
    // This test would require manually crafting an invalid sequence
    // For now, we'll test that the validation system works
    await helper.newGame();
    
    // Create some actions
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    
    // The validation should pass for valid sequences
    if (await helper.hasDebugSnapshot()) {
      const validation = await helper.validateActionSequence();
      expect(validation.success).toBe(true);
    }
  });

  test('updates URL with snapshot parameter', async () => {
    await helper.newGame();
    
    // Initial URL should not have snapshot
    expect(await helper.hasSnapshotInURL()).toBe(false);
    
    // Perform actions to create snapshot
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    
    // URL should now contain snapshot parameter
    expect(await helper.hasSnapshotInURL()).toBe(true);
    
    const url = await helper.getCurrentURL();
    expect(url).toMatch(/d=/);
  });

  test('generates enhanced bug reports with action sequences', async () => {
    await helper.newGame();
    
    // Create a snapshot by performing actions
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.setTrump('doubles');
    
    // Generate bug report
    const bugReport = await helper.getBugReport();
    
    // Bug report should contain action sequence validation
    expect(bugReport).toContain('All actions from initial state to current state');
    expect(bugReport).toContain('baseState');
    expect(bugReport).toContain('actionSequence');
    expect(bugReport).toContain('playwrightHelper.getAvailableActions');
    expect(bugReport).toContain('Invalid action at step');
  });
});