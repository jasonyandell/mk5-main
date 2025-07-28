import { Page, expect } from '@playwright/test';
import type { GameState } from '../../../game/types';

/**
 * Playwright helper for E2E testing of Texas 42 game
 * Provides page interaction utilities and state injection
 */
export class PlaywrightGameHelper {
  constructor(private page: Page) {}

  async goto() {
    await this.page.goto('/');
    await expect(this.page.locator('h1')).toContainText('Texas 42 - mk5');
  }

  async getCurrentPhase(): Promise<string> {
    // Implementation depends on data-testid attributes in components
    const phaseElement = this.page.locator('[data-testid="game-phase"]');
    return await phaseElement.textContent() || '';
  }

  async getCurrentPlayer(): Promise<string> {
    const playerElement = this.page.locator('[data-testid="current-player"]');
    return await playerElement.textContent() || '';
  }

  async openDebugPanel() {
    await this.page.locator('[data-testid="debug-toggle"]').click();
    await expect(this.page.locator('[data-testid="debug-panel"]')).toBeVisible();
  }

  async closeDebugPanel() {
    await this.page.locator('[data-testid="debug-close"]').click();
    await expect(this.page.locator('[data-testid="debug-panel"]')).not.toBeVisible();
  }

  async getAvailableActions(): Promise<string[]> {
    const actionButtons = this.page.locator('[data-testid="action-button"]');
    return await actionButtons.allTextContents();
  }

  async clickAction(actionText: string) {
    await this.page
      .locator(`[data-testid="action-button"]:has-text("${actionText}")`)
      .click();
  }

  async clickActionByIndex(index: number) {
    await this.page.locator('[data-testid="action-button"]').nth(index).click();
  }

  async getBiddingOptions(): Promise<string[]> {
    const biddingButtons = this.page.locator('[data-testid="bid-button"]');
    return await biddingButtons.allTextContents();
  }

  async placeBid(bidText: string) {
    await this.page
      .locator(`[data-testid="bid-button"]:has-text("${bidText}")`)
      .click();
  }

  async selectTrump(trumpText: string) {
    await this.page
      .locator(`[data-testid="trump-button"]:has-text("${trumpText}")`)
      .click();
  }

  async getPlayerHand(playerIndex: number): Promise<string[]> {
    const handDominoes = this.page.locator(
      `[data-testid="player-${playerIndex}-hand"] [data-testid="domino-card"]`
    );
    return await handDominoes.allTextContents();
  }

  async playDomino(dominoId: string) {
    await this.page
      .locator(`[data-testid="domino-${dominoId}"][data-playable="true"]`)
      .click();
  }

  async getTeamScores(): Promise<[number, number]> {
    const team0Score = await this.page
      .locator('[data-testid="team-0-score"]')
      .textContent();
    const team1Score = await this.page
      .locator('[data-testid="team-1-score"]')
      .textContent();
    
    return [
      parseInt(team0Score || '0'),
      parseInt(team1Score || '0')
    ];
  }

  async getTeamMarks(): Promise<[number, number]> {
    const team0Marks = await this.page
      .locator('[data-testid="team-0-marks"]')
      .textContent();
    const team1Marks = await this.page
      .locator('[data-testid="team-1-marks"]')
      .textContent();
    
    return [
      parseInt(team0Marks || '0'),
      parseInt(team1Marks || '0')
    ];
  }

  async getCurrentTrick(): Promise<string[]> {
    const trickPlays = this.page.locator('[data-testid="current-trick"] [data-testid="domino-card"]');
    return await trickPlays.allTextContents();
  }

  async getTrump(): Promise<string> {
    const trumpElement = this.page.locator('[data-testid="trump-display"]');
    return await trumpElement.textContent() || '';
  }

  async newGame() {
    await this.page.locator('[data-testid="new-game-button"]').click();
    await this.page.waitForTimeout(100); // Brief wait for state reset
  }

  async waitForPhaseChange(expectedPhase: string, timeout = 5000) {
    await this.page.waitForFunction(
      (phase) => {
        const phaseElement = document.querySelector('[data-testid="game-phase"]');
        return phaseElement?.textContent?.includes(phase);
      },
      expectedPhase,
      { timeout }
    );
  }

  async injectGameState(state: GameState) {
    // Inject state via debug panel
    await this.openDebugPanel();
    
    // Use debug panel's state injection functionality
    await this.page.locator('[data-testid="inject-state-button"]').click();
    
    // Fill in the state JSON
    await this.page
      .locator('[data-testid="state-json-input"]')
      .fill(JSON.stringify(state));
    
    await this.page.locator('[data-testid="load-state-button"]').click();
    await this.closeDebugPanel();
  }

  async generateBugReport(description: string): Promise<string> {
    await this.openDebugPanel();
    
    // Click bug report generator
    await this.page.locator('[data-testid="generate-bug-report"]').click();
    
    // Fill description
    await this.page.locator('[data-testid="bug-description"]').fill(description);
    
    // Get generated report
    const reportElement = this.page.locator('[data-testid="bug-report-output"]');
    const report = await reportElement.textContent();
    
    await this.closeDebugPanel();
    return report || '';
  }

  async takeGameScreenshot(name: string) {
    await this.page.screenshot({ 
      path: `test-results/screenshots/${name}.png`,
      fullPage: true 
    });
  }

  async validateGameRules(): Promise<string[]> {
    // Use debug panel to validate current state
    await this.openDebugPanel();
    
    const validationButton = this.page.locator('[data-testid="validate-rules"]');
    await validationButton.click();
    
    const errorsElement = this.page.locator('[data-testid="validation-errors"]');
    const errorsText = await errorsElement.textContent();
    
    await this.closeDebugPanel();
    
    return errorsText ? errorsText.split('\n').filter(e => e.trim()) : [];
  }

  async performCompleteGame(strategy: 'random' | 'aggressive' | 'conservative' = 'random') {
    while (true) {
      const phase = await this.getCurrentPhase();
      
      if (phase.includes('game_end') || phase.includes('complete')) {
        break;
      }
      
      const actions = await this.getAvailableActions();
      if (actions.length === 0) {
        break;
      }
      
      // Select action based on strategy
      let actionIndex = 0;
      
      switch (strategy) {
        case 'aggressive':
          // Choose highest bid or most aggressive play
          actionIndex = actions.length - 1;
          break;
        case 'conservative':
          // Choose pass or lowest bid
          actionIndex = actions.findIndex(a => a.toLowerCase().includes('pass')) || 0;
          break;
        default:
          // Random selection
          actionIndex = Math.floor(Math.random() * actions.length);
      }
      
      await this.clickActionByIndex(actionIndex);
      await this.page.waitForTimeout(100); // Brief pause for state updates
    }
  }
}