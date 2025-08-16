import { expect } from '@playwright/test';
import type { Page } from '@playwright/test';
import type { GameState } from '../../../game/types';
import { encodeURLData } from '../../../game/core/url-compression';

interface ActionOption {
  index: number;
  type: string;
  value?: string | number;
  id: string;
}

/**
 * Playwright helper for E2E testing of Texas 42 game
 * Provides page interaction utilities and state injection
 */
export class PlaywrightGameHelper {
  constructor(private page: Page) {}
  
  // Public method to access page for direct operations
  getPage(): Page {
    return this.page;
  }
  
  // Public method to get locator
  locator(selector: string) {
    return this.page.locator(selector);
  }

  async goto() {
    await this.page.goto('/');
    // Wait for the app to load - mobile UI uses .app-container
    await this.page.waitForSelector('.app-container', { timeout: 2000 });
  }

  async gotoWithSeed(seed: number) {
    // Create a deterministic URL with the specified seed
    const urlData = {
      v: 1 as const,
      s: { s: seed }, // MinimalGameState with shuffle seed
      a: [] // No actions initially
    };
    const encoded = encodeURLData(urlData);
    await this.page.goto(`/?d=${encoded}`);
    // Wait for mobile UI to load
    await this.page.waitForSelector('.app-container', { timeout: 2000 });
  }

  async getCurrentPhase(): Promise<string> {
    try {
      // Try to get phase from the header's phase indicator
      const phaseElement = this.page.locator('.phase-name');
      const phaseText = await phaseElement.textContent({ timeout: 1000 });
      if (phaseText) {
        return phaseText.toLowerCase();
      }
      
      // Fallback: try to infer phase from visible actions
      const bidButtons = await this.page.locator('[data-testid^="bid-"]').count();
      const trumpButtons = await this.page.locator('[data-testid^="trump-"]').count();
      
      if (bidButtons > 0) return 'bidding';
      if (trumpButtons > 0) return 'trump_selection';
      
      return 'playing'; // Default assumption
    } catch (error) {
      console.warn('Could not get current phase:', error);
      return 'bidding'; // Default to bidding for new games
    }
  }

  async getCurrentPlayer(): Promise<string> {
    try {
      // Try to get current player from header's turn display
      const turnElement = this.page.locator('.turn-player');
      const turnText = await turnElement.textContent({ timeout: 2000 });
      if (turnText) {
        return `Current Player: ${turnText}`;
      }
      
      return 'Current Player: P0'; // Default assumption
    } catch (error) {
      console.warn('Could not get current player:', error);
      return '';
    }
  }

  async openDebugPanel() {
    // Use the nav debug button
    await this.page.locator('[data-testid="nav-debug"]').click();
    await expect(this.page.locator('.debug-panel')).toBeVisible();
  }

  async closeDebugPanel() {
    await this.page.locator('.close-button').click();
    await expect(this.page.locator('.debug-panel')).not.toBeVisible();
  }

  async getAvailableActions(): Promise<ActionOption[]> {
    return await this.getActionsList();
  }

  async getBiddingOptions(): Promise<ActionOption[]> {
    const actions = await this.getActionsList();
    return actions.filter(a => a.type === 'pass' || a.type === 'bid_points' || a.type === 'bid_marks');
  }

  async selectActionByIndex(index: number) {
    const locator = this.page.locator('.action-button').nth(index);
    await locator.waitFor({ state: 'visible' });
    await locator.click({ force: true });
  }

  async getActionsList(): Promise<ActionOption[]> {
    // Wait for actions to be available with optimized timeout
    await this.page.waitForSelector('.action-button', { timeout: 1000 }).catch(() => {
      // If no actions available, return empty array
      return null;
    });
    
    const buttons = await this.page.locator('.action-button').all();
    const actions = [];
    
    for (let i = 0; i < buttons.length; i++) {
      const actionId = await buttons[i].getAttribute('data-testid');
      
      let type = 'unknown';
      let value: string | number | undefined = undefined;
      
      if (actionId === 'pass') {
        type = 'pass';
      } else if (actionId?.startsWith('bid-')) {
        const parts = actionId.split('-');
        if (parts.length === 3 && parts[2] === 'marks') {
          type = 'bid_marks';
          value = parseInt(parts[1]);
        } else if (parts.length === 2) {
          type = 'bid_points';
          value = parseInt(parts[1]);
        }
      } else if (actionId?.startsWith('trump-')) {
        type = 'trump_selection';
        value = actionId.split('-')[1];
      } else if (actionId?.startsWith('play-')) {
        type = 'play_domino';
        value = actionId.substring(5);
      } else if (actionId === 'complete-trick') {
        type = 'complete_trick';
      } else if (actionId === 'score-hand') {
        type = 'score_hand';
      } else if (actionId === 'redeal') {
        type = 'redeal';
      } else if (actionId === 'select-trump') {
        type = 'select_trump';
      }
      
      actions.push({ index: i, type, value, id: actionId || '' });
    }
    
    return actions;
  }

  async selectActionByType(type: string, value?: string | number) {
    // If looking for trump actions, ensure we're on actions panel
    if (type === 'trump_selection' || type.startsWith('trump')) {
      await this.page.locator('[data-testid="nav-actions"]').click();
      await this.page.waitForTimeout(200);
    }
    
    const actions = await this.getActionsList();
    const action = actions.find(a => a.type === type && (value === undefined || a.value === value));
    if (action) {
      await this.selectActionByIndex(action.index);
    } else {
      throw new Error(`No action found for type: ${type}, value: ${value}`);
    }
  }

  async selectActionById(id: string) {
    const actions = await this.getActionsList();
    const action = actions.find(a => a.id === id);
    if (action) {
      await this.selectActionByIndex(action.index);
    } else {
      console.log('Available actions:', actions.map(a => a.id));
      throw new Error(`No action found for id: ${id}. Available actions: ${actions.map(a => a.id).join(', ')}`);
    }
  }

  async setTrumpBySuit(suit: string) {
    // Navigate to actions panel first to ensure trump actions are visible
    await this.page.locator('[data-testid="nav-actions"]').click();
    await this.page.waitForTimeout(200); // Small delay for navigation
    
    // Use the correct trump action ID format
    const trumpActionId = `trump-${suit}`;
    await this.selectActionById(trumpActionId);
  }

  // Add back methods that tests expect, but using new index-based approach
  async bidPoints(_player: number, points: number) {
    await this.selectActionByType('bid_points', points);
  }

  async bidMarks(_player: number, marks: number) {
    await this.selectActionByType('bid_marks', marks);
  }

  async bidPass(_player: number) {
    await this.selectActionByType('pass');
  }

  async setTrump(suit: string) {
    // Map common suit names to internal names
    const suitMap: Record<string, string> = {
      '0s': 'blanks',
      '1s': 'ones',
      '2s': 'twos', 
      '3s': 'threes',
      '4s': 'fours',
      '5s': 'fives',
      '6s': 'sixes',
      'doubles': 'doubles',
      'Doubles': 'doubles',
      'blanks': 'blanks',
      'ones': 'ones',
      'twos': 'twos',
      'threes': 'threes',
      'fours': 'fours',
      'fives': 'fives',
      'sixes': 'sixes'
    };
    
    const internalSuit = suitMap[suit] || suit.toLowerCase();
    await this.setTrumpBySuit(internalSuit);
  }

  async clickActionIndex(index: number) {
    await this.selectActionByIndex(index);
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

  async playAnyDomino() {
    const actions = await this.getActionsList();
    const playAction = actions.find(a => a.type === 'play_domino');
    if (playAction) {
      await this.selectActionByIndex(playAction.index);
    } else {
      throw new Error('No play_domino action available');
    }
  }
  
  async playDominoByValue(dominoValue: string) {
    await this.selectActionByType('play_domino', dominoValue);
  }

  async getTeamScores(): Promise<[number, number]> {
    try {
      // Look for score values in the header score cards
      const team0Score = await this.page
        .locator('.score-card.us .score-value')
        .textContent({ timeout: 1000 });
      const team1Score = await this.page
        .locator('.score-card.them .score-value')
        .textContent({ timeout: 1000 });
      
      // Extract numbers from text
      const score0 = parseInt((team0Score || '0').match(/(\d+)/)?.[1] || '0');
      const score1 = parseInt((team1Score || '0').match(/(\d+)/)?.[1] || '0');
      
      return [score0, score1];
    } catch (error) {
      console.warn('Could not get team scores, returning [0, 0]:', error);
      return [0, 0];
    }
  }

  async getTeamMarks(): Promise<[number, number]> {
    try {
      // Use the same score card elements as scores since marks are displayed there
      const team0Marks = await this.page
        .locator('.score-card.us .score-value')
        .textContent({ timeout: 1000 });
      const team1Marks = await this.page
        .locator('.score-card.them .score-value')
        .textContent({ timeout: 1000 });
      
      // Extract numbers from text
      const marks0 = parseInt((team0Marks || '0').match(/(\d+)/)?.[1] || '0');
      const marks1 = parseInt((team1Marks || '0').match(/(\d+)/)?.[1] || '0');
      
      return [marks0, marks1];
    } catch (error) {
      // If elements not found, return [0, 0] as fallback
      console.warn('Could not get team marks, returning [0, 0]:', error);
      return [0, 0];
    }
  }

  async getCurrentTrick(): Promise<string[]> {
    try {
      const trickPlays = this.page.locator('.current-trick-area .domino-card, .trick-area .domino-card');
      return await trickPlays.allTextContents();
    } catch (error) {
      console.warn('Could not get current trick, returning empty array:', error);
      return [];
    }
  }

  async getTrump(): Promise<string> {
    try {
      // Look for trump display in the game interface
      const trumpElement = this.page.locator('.trump-display, .game-trump, [data-trump]');
      const trumpText = await trumpElement.textContent({ timeout: 1000 });
      return trumpText || '';
    } catch (error) {
      console.warn('Could not get trump, returning empty string:', error);
      return '';
    }
  }

  async completeTrick() {
    await this.selectActionByType('complete_trick');
  }

  async scoreHand() {
    await this.selectActionByType('score_hand');
  }

  async redeal() {
    await this.selectActionByType('redeal');
  }


  async newGame() {
    // Simply reload the page to start a new game
    await this.page.reload();
    await this.page.waitForSelector('.app-container', { timeout: 2000 });
  }

  async waitForPhaseChange(expectedPhase: string, timeout = 2000) {
    await this.page.waitForFunction(
      (phase) => {
        const phaseElement = document.querySelector('.phase-name');
        return phaseElement?.textContent?.toLowerCase().includes(phase);
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
      
      const actions = await this.getActionsList();
      if (actions.length === 0) {
        break;
      }
      
      let actionIndex = 0;
      
      switch (strategy) {
        case 'aggressive':
          actionIndex = actions.length - 1;
          break;
        case 'conservative':
          actionIndex = actions.findIndex(a => a.type === 'pass') || 0;
          break;
        default:
          actionIndex = Math.floor(Math.random() * actions.length);
      }
      
      await this.selectActionByIndex(actionIndex);
    }
  }

  async playCompleteHand(bidder: number, bidValue: number, bidType: 'points' | 'marks', trump: string) {
    // Bidding phase
    for (let i = 0; i < 4; i++) {
      const currentPlayerText = await this.getCurrentPlayer();
      const currentPlayerNum = parseInt(currentPlayerText.replace('Current Player: P', ''));
      
      if (currentPlayerNum === bidder) {
        if (bidType === 'marks') {
          await this.selectActionByType('bid_marks', bidValue);
        } else {
          await this.selectActionByType('bid_points', bidValue);
        }
      } else {
        await this.selectActionByType('pass');
      }
    }

    // Trump selection
    await this.setTrumpBySuit(trump);

    // Play tricks until hand is complete (may end early due to hand outcome detection)
    for (let trick = 1; trick <= 7; trick++) {
      // Check if we're already in scoring phase (hand ended early)
      const phase = await this.getCurrentPhase();
      if (phase.toLowerCase().includes('scoring')) {
        // Hand ended early - no more tricks to play
        break;
      }
      
      // Each player plays a domino
      for (let play = 0; play < 4; play++) {
        // Check again in case hand ends mid-trick
        const currentPhase = await this.getCurrentPhase();
        if (currentPhase.toLowerCase().includes('scoring')) {
          break;
        }
        
        await this.playAnyDomino();
      }
      
      // Only complete trick if we're still in playing phase
      const phaseAfterPlays = await this.getCurrentPhase();
      if (phaseAfterPlays.toLowerCase().includes('playing')) {
        await this.completeTrick();
      }
    }
    
    // Score hand if we're in scoring phase
    const finalPhase = await this.getCurrentPhase();
    if (finalPhase.toLowerCase().includes('scoring')) {
      try {
        await this.scoreHand();
      } catch {
        // Scoring might not be available if already scored
      }
    }
  }

  // Debug snapshot and replay methods
  async hasDebugSnapshot(): Promise<boolean> {
    const snapshotElement = this.page.locator('[data-testid="debug-snapshot"]');
    return await snapshotElement.count() > 0;
  }

  async getDebugSnapshotInfo(): Promise<{reason: string, actionCount: number} | null> {
    if (!(await this.hasDebugSnapshot())) {
      return null;
    }
    
    const reasonElement = this.page.locator('[data-testid="snapshot-reason"]');
    const actionCountElement = this.page.locator('[data-testid="snapshot-action-count"]');
    
    const reason = await reasonElement.textContent() || '';
    const actionCountText = await actionCountElement.textContent() || '0';
    const actionCount = parseInt(actionCountText.replace(/\D/g, ''));
    
    return { reason, actionCount };
  }

  async validateActionSequence(): Promise<{success: boolean, errors: string[]}> {
    const validateButton = this.page.locator('[data-testid="validate-sequence-button"]');
    await validateButton.click();
    
    // Wait for validation to complete - optimized for offline testing
    await this.page.waitForTimeout(100);
    
    const errorElements = this.page.locator('[data-testid="validation-error"]');
    const errorCount = await errorElements.count();
    
    if (errorCount === 0) {
      return { success: true, errors: [] };
    }
    
    const errors = [];
    for (let i = 0; i < errorCount; i++) {
      const errorText = await errorElements.nth(i).textContent();
      if (errorText) {
        errors.push(errorText);
      }
    }
    
    return { success: false, errors };
  }

  async loadState(state: GameState) {
    // Mock the prompt and then click the load state button
    await this.page.evaluate((stateJson) => {
      window.prompt = () => stateJson;
    }, JSON.stringify(state));
    
    // Find and click the Load State button (it's a secondary button in the header)
    await this.page.getByText('Load State').click();
  }

  async clickAction(actionId: string) {
    await this.page.locator(`[data-action-id="${actionId}"]`).click();
  }

  async getCurrentURL(): Promise<string> {
    return this.page.url();
  }

  async hasSnapshotInURL(): Promise<boolean> {
    const url = await this.getCurrentURL();
    return url.includes('d=');
  }

  async copyStateURL(): Promise<string> {
    // Just return the current URL since that's what would be copied
    return await this.getCurrentURL();
  }



  async getCurrentState(): Promise<unknown> {
    // This would need to be implemented to get the current game state
    // For now, return null - this is used in commented sections of bug reports
    return null;
  }
}

// Export a singleton instance for backward compatibility
export const playwrightHelper = {
  loadState: async (page: Page, state: GameState) => {
    const helper = new PlaywrightGameHelper(page);
    return helper.loadState(state);
  },
  clickAction: async (page: Page, actionId: string) => {
    const helper = new PlaywrightGameHelper(page);
    return helper.clickAction(actionId);
  },
  getAvailableActions: async (page: Page) => {
    const helper = new PlaywrightGameHelper(page);
    return helper.getAvailableActions();
  }
};