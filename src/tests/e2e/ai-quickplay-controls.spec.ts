import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('AI Quickplay Controls', () => {
  test.describe('Run/Stop functionality', () => {
    test('run button starts AI playing at instant speed', async ({ page }) => {
      const helper = new PlaywrightGameHelper(page);
      await helper.gotoWithSeed(12345);
      
      // Set to instant speed
      await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
      
      // Enable all players as AI
      for (let i = 0; i < 4; i++) {
        await helper.locator(`[data-testid="ai-player-${i}"]`).check();
      }
      
      // Get initial state
      const phaseBefore = await helper.getCurrentPhase();
      const actionsBefore = await helper.getAvailableActions();
      
      // Click run button
      const runButton = helper.locator('[data-testid="quickplay-run"]');
      await expect(runButton).toBeVisible();
      await runButton.click();
      
      // Stop button should appear
      const stopButton = helper.locator('[data-testid="quickplay-stop"]');
      await expect(stopButton).toBeVisible();
      
      // Wait for AI to make moves (instant speed) - optimized for offline testing
      await page.waitForTimeout(100);
      
      // Game should have progressed
      const phaseAfter = await helper.getCurrentPhase();
      const actionsAfter = await helper.getAvailableActions();
      
      // Either phase changed or actions changed (game progressed)
      const gameProgressed = phaseAfter !== phaseBefore || 
                           JSON.stringify(actionsAfter) !== JSON.stringify(actionsBefore);
      expect(gameProgressed).toBe(true);
      
      // Stop the AI
      await stopButton.click();
      
      // Run button should reappear
      await expect(runButton).toBeVisible();
    });

    test('stop button halts AI execution', async ({ page }) => {
      const helper = new PlaywrightGameHelper(page);
      await helper.gotoWithSeed(12345);
      
      // Set to instant speed and enable all AI
      await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
      for (let i = 0; i < 4; i++) {
        await helper.locator(`[data-testid="ai-player-${i}"]`).check();
      }
      
      // Start AI
      await helper.locator('[data-testid="quickplay-run"]').click();
      
      // Let it run briefly
      await page.waitForTimeout(100);
      
      // Stop AI
      await helper.locator('[data-testid="quickplay-stop"]').click();
      
      // Get current state
      const stateAfterStop = await helper.getAvailableActions();
      
      // Wait to ensure no more moves are made - optimized timing
      await page.waitForTimeout(100);
      
      // State should not have changed
      const stateAfterWait = await helper.getAvailableActions();
      expect(JSON.stringify(stateAfterWait)).toBe(JSON.stringify(stateAfterStop));
    });
  });

  test.describe('Pause/Resume functionality', () => {
    test('pause button temporarily stops AI execution', async ({ page }) => {
      const helper = new PlaywrightGameHelper(page);
      await helper.gotoWithSeed(12345);
      
      // Set to instant speed and enable all AI
      await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
      for (let i = 0; i < 4; i++) {
        await helper.locator(`[data-testid="ai-player-${i}"]`).check();
      }
      
      // Start AI
      await helper.locator('[data-testid="quickplay-run"]').click();
      
      // Pause button should be visible
      const pauseButton = helper.locator('[data-testid="quickplay-pause"]');
      await expect(pauseButton).toBeVisible();
      
      // Let it run briefly
      await page.waitForTimeout(100);
      
      // Pause AI
      await pauseButton.click();
      
      // Resume button should appear
      const resumeButton = helper.locator('[data-testid="quickplay-resume"]');
      await expect(resumeButton).toBeVisible();
      
      // Get current state
      const stateAfterPause = await helper.getAvailableActions();
      
      // Wait to ensure no moves while paused - optimized timing
      await page.waitForTimeout(100);
      
      // State should not have changed
      const stateAfterWait = await helper.getAvailableActions();
      expect(JSON.stringify(stateAfterWait)).toBe(JSON.stringify(stateAfterPause));
      
      // Stop to clean up
      await helper.locator('[data-testid="quickplay-stop"]').click();
    });

    test('resume button continues AI execution', async ({ page }) => {
      const helper = new PlaywrightGameHelper(page);
      await helper.gotoWithSeed(12345);
      
      // Set to instant speed and enable all AI
      await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
      for (let i = 0; i < 4; i++) {
        await helper.locator(`[data-testid="ai-player-${i}"]`).check();
      }
      
      // Start AI
      await helper.locator('[data-testid="quickplay-run"]').click();
      
      // Let it run a bit - optimized for AI quickplay
      await page.waitForTimeout(100);
      
      // Pause
      await helper.locator('[data-testid="quickplay-pause"]').click();
      
      // Verify paused status
      const pausedStatus = helper.locator('.status-indicator.active');
      await expect(pausedStatus).toContainText('Paused');
      
      // Verify game is paused (actions still available but not executing)
      
      // Resume
      await helper.locator('[data-testid="quickplay-resume"]').click();
      
      // Verify running status
      const runningStatus = helper.locator('.status-indicator.active');
      await expect(runningStatus).toContainText('Running');
      
      // Pause button should be visible again
      await expect(helper.locator('[data-testid="quickplay-pause"]')).toBeVisible();
      
      // Stop to clean up
      await helper.locator('[data-testid="quickplay-stop"]').click();
    });
  });

  test.describe('Player toggle functionality', () => {
    test('can toggle individual players between AI and human', async ({ page }) => {
      const helper = new PlaywrightGameHelper(page);
      await helper.gotoWithSeed(12345);
      
      // Initially all should be checked (default)
      for (let i = 0; i < 4; i++) {
        const checkbox = helper.locator(`[data-testid="ai-player-${i}"]`);
        await expect(checkbox).toBeChecked();
      }
      
      // Uncheck player 1
      await helper.locator('[data-testid="ai-player-1"]').uncheck();
      await expect(helper.locator('[data-testid="ai-player-1"]')).not.toBeChecked();
      
      // Check it again
      await helper.locator('[data-testid="ai-player-1"]').check();
      await expect(helper.locator('[data-testid="ai-player-1"]')).toBeChecked();
    });

    test('AI only acts for enabled players', async ({ page }) => {
      const helper = new PlaywrightGameHelper(page);
      await helper.gotoWithSeed(12345);
      
      // Set instant speed
      await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
      
      // Get current player
      const currentPlayer = await helper.getCurrentPlayer();
      const currentPlayerNum = parseInt(currentPlayer.replace('Current Player: P', ''));
      
      // Disable only current player
      for (let i = 0; i < 4; i++) {
        if (i === currentPlayerNum) {
          await helper.locator(`[data-testid="ai-player-${i}"]`).uncheck();
        } else {
          await helper.locator(`[data-testid="ai-player-${i}"]`).check();
        }
      }
      
      // Start AI
      await helper.locator('[data-testid="quickplay-run"]').click();
      
      // Wait briefly - optimized timing
      await page.waitForTimeout(100);
      
      // Should still be same player's turn (AI didn't act)
      const playerAfter = await helper.getCurrentPlayer();
      expect(playerAfter).toBe(currentPlayer);
      
      // Stop AI
      await helper.locator('[data-testid="quickplay-stop"]').click();
    });
  });

  test.describe('Status display', () => {
    test('shows correct status when running', async ({ page }) => {
      const helper = new PlaywrightGameHelper(page);
      await helper.gotoWithSeed(12345);
      
      // Set speed
      await helper.locator('[data-testid="quickplay-speed"]').selectOption('instant');
      
      // Start AI
      await helper.locator('[data-testid="quickplay-run"]').click();
      
      // Check status indicator
      const statusIndicator = helper.locator('.status-indicator.active');
      await expect(statusIndicator).toBeVisible();
      await expect(statusIndicator).toContainText('Running (instant)');
      
      // Stop
      await helper.locator('[data-testid="quickplay-stop"]').click();
    });

    test('shows AI badge for current player when enabled', async ({ page }) => {
      const helper = new PlaywrightGameHelper(page);
      await helper.gotoWithSeed(12345);
      
      // Get current player
      const currentPlayer = await helper.getCurrentPlayer();
      const currentPlayerNum = parseInt(currentPlayer.replace('Current Player: P', ''));
      
      // Enable current player as AI
      await helper.locator(`[data-testid="ai-player-${currentPlayerNum}"]`).check();
      
      // AI badge should be visible
      const aiBadge = helper.locator('.ai-badge');
      await expect(aiBadge).toBeVisible();
      await expect(aiBadge).toContainText('AI');
      
      // Disable current player as AI
      await helper.locator(`[data-testid="ai-player-${currentPlayerNum}"]`).uncheck();
      
      // AI badge should not be visible
      await expect(aiBadge).not.toBeVisible();
    });
  });
});