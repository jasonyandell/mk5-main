import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Bug Report Generation Tests', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
    await helper.goto();
  });

  test('generates bug report with efficient action array', async () => {
    // Perform some actions to create an action history
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.setTrump('doubles');
    
    // Play a few dominoes to create more history
    await helper.playAnyDomino(); // Player 1 leads
    await helper.playAnyDomino(); // Player 2
    await helper.playAnyDomino(); // Player 3
    await helper.playAnyDomino(); // Player 0
    
    // Generate bug report
    const bugReport = await helper.getBugReport();
    
    // Verify the bug report contains the action array
    expect(bugReport).toContain('const actionIds = [');
    expect(bugReport).toContain('"bid-30"');
    expect(bugReport).toContain('"pass"');
    expect(bugReport).toContain('"trump-doubles"');
    
    // Verify it's a unit test, not Playwright test
    expect(bugReport).toContain('import { test, expect } from \'vitest\'');
    expect(bugReport).toContain('import { getNextStates } from \'../game\'');
    expect(bugReport).toContain('import type { GameState }');
    
    // Verify it contains the base state
    expect(bugReport).toContain('const baseState: GameState = {');
    expect(bugReport).toContain('getNextStates(currentState)');
    
    // Verify it has proper test structure
    expect(bugReport).toContain('test(\'Bug report -');
    
    // Verify game object assertions are included
    expect(bugReport).toContain('expect(currentState.phase)');
    expect(bugReport).toContain('expect(currentState.currentPlayer)');
    expect(bugReport).toContain('expect(currentState.teamScores)');
    expect(bugReport).toContain('expect(currentState.teamMarks)');
  });

  test('bug report includes validation error information', async ({ page }) => {
    // Create a scenario that might trigger validation errors
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    
    // Inject a potential validation scenario by manipulating the state
    // This is just for testing - normally validation errors would occur naturally
    await page.evaluate(() => {
      // Access the game store and set a validation error
      const gameStore = (window as any).gameStore;
      if (gameStore && gameStore.stateValidationError) {
        gameStore.stateValidationError.set('Test validation error for bug report generation');
      }
    });
    
    const bugReport = await helper.getBugReport();
    
    // Verify validation error is included in the bug report (in comment format)
    expect(bugReport).toContain('VALIDATION ERROR DETECTED');
    expect(bugReport).toContain('Test validation error');
  });

  test('bug report provides step-by-step action replay using game logic', async () => {
    // Create action history
    await helper.selectActionByType('bid_points', 35);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.setTrump('sixes');
    
    const bugReport = await helper.getBugReport();
    
    // Verify step-by-step replay section uses game logic
    expect(bugReport).toContain('Replay actions step by step using game logic');
    expect(bugReport).toContain('for (let i = 0; i < actionIds.length; i++)');
    expect(bugReport).toContain('getNextStates(currentState)');
    expect(bugReport).toContain('matchingTransition.newState');
    expect(bugReport).toContain('console.log(`Step ${i + 1}:');
  });

  test('bug report includes comprehensive game state assertions', async () => {
    // Create a more complex game state
    await helper.selectActionByType('bid_marks', 1);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.setTrump('fives');
    
    const bugReport = await helper.getBugReport();
    
    // Verify comprehensive game state assertions are included
    expect(bugReport).toContain('expect(currentState.phase)');
    expect(bugReport).toContain('expect(currentState.currentPlayer)');
    expect(bugReport).toContain('expect(currentState.teamScores)');
    expect(bugReport).toContain('expect(currentState.teamMarks)');
    expect(bugReport).toContain('expect(currentState.trump)');
    expect(bugReport).toContain('expect(currentState.players).toHaveLength(4)');
    expect(bugReport).toContain('expect(currentState.bids).toHaveLength(');
    expect(bugReport).toContain('expect(currentState.tricks).toHaveLength(');
  });

  test('bug report uses unit test format with game objects', async () => {
    // Create action sequence
    await helper.selectActionByType('bid_points', 32);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    
    const bugReport = await helper.getBugReport();
    
    // Verify unit test format is used
    expect(bugReport).toContain('const baseState: GameState = {');
    expect(bugReport).toContain('const actionIds = [');
    expect(bugReport).toContain('"bid-32"');
    expect(bugReport).toContain('from \'vitest\'');
    expect(bugReport).not.toContain('page.'); // No browser interactions
    expect(bugReport).not.toContain('await'); // Synchronous unit test
  });

  test('bug report can be copied and includes proper timestamp', async () => {
    // Perform actions
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    
    // Generate bug report via UI
    await helper.page.locator('[data-testid="bug-report-button"]').click();
    
    // Verify bug panel is shown
    await expect(helper.page.locator('[data-testid="bug-content"]')).toBeVisible();
    
    // Verify generated test code area has content
    const testCodeArea = helper.page.locator('[data-testid="generated-test-code"]');
    const bugReportContent = await testCodeArea.inputValue();
    
    
    expect(bugReportContent).toContain('test(\'Bug report -');
    expect(bugReportContent).toMatch(/Bug report - \d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}/);
    
    // Verify copy button is available
    await expect(helper.page.locator('[data-testid="copy-test-code"]')).toBeVisible();
  });

  test('bug report works with different game phases', async () => {
    // Test initial state (bidding phase)
    let bugReport = await helper.getBugReport();
    expect(bugReport).toContain('"phase": "bidding"');
    
    // Make a single bid to change phase
    await helper.selectActionByType('bid_points', 30);
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    await helper.selectActionByType('pass');
    
    // Generate bug report and verify phase assertion
    bugReport = await helper.getBugReport();
    expect(bugReport).toContain('expect(currentState.phase).toBe(\'trump_selection\')');
    
    // Verify it contains unit test format
    expect(bugReport).toContain('from \'vitest\'');
    expect(bugReport).toContain('getNextStates');
  });
});