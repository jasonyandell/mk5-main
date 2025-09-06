/* eslint-disable @typescript-eslint/no-explicit-any */
// Reason: page.waitForFunction() and page.evaluate() run in browser context where TypeScript cannot
// verify window properties. The use of 'any' for window casts is architectural, not a shortcut.

import { expect } from '@playwright/test';
import type { Page, Locator } from '@playwright/test';
import { encodeGameUrl } from '../../../game/core/url-compression';

/**
 * Simplified, deterministic Playwright helper for Texas 42 E2E testing
 * Uses Playwright's built-in auto-waiting and deterministic patterns
 */
interface ActionInfo {
  index: number;
  type: string;
  value?: string | number | undefined;
  id: string;
}

interface GameMetrics {
  scores: [number, number];
  trump: string;
}

interface GotoOptions {
  disableUrlUpdates?: boolean;
  testMode?: boolean; // default true; when false, main loop runs
}

interface QuickplayState {
  enabled: boolean;
  aiPlayers: Set<number>;
}

interface QuickplayActions {
  toggle(): void;
  togglePlayer(playerId: number): void;
}

interface GameWindow {
  quickplayActions?: QuickplayActions;
  getQuickplayState?: () => QuickplayState;
}

export class PlaywrightGameHelper {
  private page: Page;
  // Timeout strategy for different operation types
  static TIMEOUTS = {
    quick: 1000,     // For immediate DOM checks
    normal: 5000,    // For standard operations
    slow: 10000,     // For page loads and complex operations
    test: 10000      // Overall test timeout
  };

  // Centralized selector definitions using data attributes and roles
  static SELECTORS = {
    app: '.app-container',
    appHeader: '.app-header',
    phase: '[data-phase]',
    actionButton: 'button[data-testid]',
    domino: {
      playable: '.domino-wrapper[data-playable="true"] button[data-testid^="domino-"]',
      any: 'button[data-testid^="domino-"]',
      byId: (id: string) => `[data-testid="domino-${id}"]`
    },
    score: {
      display: '.app-header',  // Scores are now in the header
      us: '.app-header .text-primary',
      them: '.app-header .text-secondary'
    },
    trump: '[data-testid="trump-display"]',
    trick: {
      table: '[data-testid="trick-table"], [data-testid*="complete-trick"], [data-testid*="score-hand"]',
      tappable: '[data-testid="trick-table"], [data-testid*="complete-trick"], [data-testid*="score-hand"]',
      spot: '.trick-spot',
      played: '.trick-spot .played-domino'
    },
    nav: (item: string) => `[data-testid="nav-${item}"]`,
    bid: (value: number, isMarks: boolean) => `[data-testid="bid-${value}${isMarks ? '-marks' : ''}"]`,
    pass: '[data-testid="pass"]',
    trumpSelect: (suit: string) => `[data-testid="trump-${suit}"]`,
    flash: '.flash-message',
    settings: {
      panel: '[data-testid="settings-panel"]',
      button: '.settings-btn',
      closeButton: '[data-testid="settings-close-button"]',
      historyTab: '[data-testid="history-tab"]',
      historyItem: '[data-testid="history-item"]',
      timeTravelButton: '[data-testid="time-travel-button"], .time-travel-button'
    },
    debug: {
      panel: '[data-testid="settings-panel"]',
      button: '.settings-btn',
      closeButton: '[data-testid="settings-close-button"]',
      historyTab: '[data-testid="history-tab"]',
      historyItem: '[data-testid="history-item"]',
      timeTravelButton: '[data-testid="time-travel-button"], .time-travel-button'
    },
    playingArea: '[data-testid="playing-area"]',
    tapIndicator: '[data-testid*="complete-trick"], [data-testid*="agree-complete-trick"]',
    tapText: '.tap-text',
    turnPlayer: '.turn-player'
  };

  constructor(page: Page) {
    this.page = page;
    // Set default timeout for all operations (5 seconds for reliability)
    this.page.setDefaultTimeout(PlaywrightGameHelper.TIMEOUTS.slow);
  }

  /**
   * Navigate to game with deterministic seed and wait for ready state
   */
  async goto(seed = 12345, options: GotoOptions = {}): Promise<void> {
    // Always use all human players for deterministic testing
    const playerTypes = ['human', 'human', 'human', 'human'] as ('human' | 'ai')[];
    const urlStr = encodeGameUrl(seed, [], playerTypes, undefined, undefined);
    
    // Add testMode for deterministic testing (disables main loop) unless overridden
    const tm = options.testMode ?? true;
    const url = tm ? `${urlStr}&testMode=true` : urlStr;
    
    // Navigate and wait for network idle for deterministic loading
    await this.page.goto(url, { 
      waitUntil: 'networkidle',
      timeout: PlaywrightGameHelper.TIMEOUTS.slow 
    });
    
    // Wait for game to be fully ready
    await this.waitForGameReady();
    
    // Disable animations and AI in a single evaluation
    await this.page.evaluate((opts: GotoOptions) => {
      // Disable all animations for deterministic testing
      const style = document.createElement('style');
      style.textContent = `
        *, *::before, *::after {
          animation-duration: 0s !important;
          animation-delay: 0s !important;
          transition-duration: 0s !important;
          transition-delay: 0s !important;
        }
      `;
      document.head.appendChild(style);
      
      // Disable AI for all players
      const gameWindow = window as GameWindow;
      if (gameWindow.quickplayActions && gameWindow.getQuickplayState) {
        const state = gameWindow.getQuickplayState();
        if (state.enabled) {
          gameWindow.quickplayActions.toggle();
        }
        for (const playerId of state.aiPlayers) {
          gameWindow.quickplayActions.togglePlayer(playerId);
        }
      }
      
      // Disable URL updates unless explicitly enabled
      if (opts?.disableUrlUpdates !== false) {
        window.history.pushState = () => {};
        window.history.replaceState = () => {};
      }
    }, options);
  }

  /**
   * Wait for game to be ready using deterministic conditions
   */
  async waitForGameReady(): Promise<void> {
    // Wait for app container and phase to be set
    await this.page.waitForSelector(PlaywrightGameHelper.SELECTORS.app);
    
    // Wait for phase attribute to be set and game state to be stable
    await this.page.waitForFunction(
      () => {
        const container = document.querySelector('.app-container');
        const phase = container?.getAttribute('data-phase');
        const gameState = (window as any).getGameState?.();
        
        // Check that we have both a phase and a game state
        // and that the game is not in a transitional state
        return !!(phase && gameState && !gameState.isProcessing);
      },
      { timeout: PlaywrightGameHelper.TIMEOUTS.normal }
    );
  }
  
  /**
   * Wait for navigation state to be restored after browser navigation
   * Simply waits for game state to be ready after navigation
   */
  async waitForNavigationRestore(): Promise<void> {
    // Wait for game state to exist and be stable
    await this.page.waitForFunction(
      () => {
        const gameState = (window as any).getGameState?.();
        return !!gameState && !('isProcessing' in gameState && gameState.isProcessing);
      },
      { timeout: PlaywrightGameHelper.TIMEOUTS.normal }
    );
    
    // Then wait for game to be ready
    await this.waitForGameReady();
  }

  /**
   * Get current game phase directly from data attribute
   */
  async getCurrentPhase(): Promise<string> {
    const container = this.page.locator(PlaywrightGameHelper.SELECTORS.app);
    return await container.getAttribute('data-phase') || 'unknown';
  }

  /**
   * Wait for specific phase change
   */
  async waitForPhase(expectedPhase: string, timeout = PlaywrightGameHelper.TIMEOUTS.normal): Promise<void> {
    await this.page.waitForFunction(
      (phase: string) => document.querySelector('.app-container')?.getAttribute('data-phase') === phase,
      expectedPhase,
      { timeout }
    );
  }

  /**
   * Get current player from UI
   */
  async getCurrentPlayer(): Promise<string> {
    const turnPlayer = this.page.locator(PlaywrightGameHelper.SELECTORS.turnPlayer);
    await turnPlayer.waitFor({ state: 'visible', timeout: PlaywrightGameHelper.TIMEOUTS.quick });
    const text = await turnPlayer.textContent();
    return text ? text.trim() : 'P0';
  }

  /**
   * Unified action selection method
   */
  async selectAction(criteria: number | string | { type: string; value?: string | number }): Promise<void> {
    // Wait for any action to be available
    await this.page.waitForSelector(PlaywrightGameHelper.SELECTORS.actionButton);
    
    let locator;
    
    if (typeof criteria === 'number') {
      // Select by index
      const buttons = await this.page.locator(PlaywrightGameHelper.SELECTORS.actionButton).all();
      if (criteria >= 0 && criteria < buttons.length) {
        locator = buttons[criteria];
      }
    } else if (typeof criteria === 'string') {
      // Select by test id or type
      locator = this.page.locator(`[data-testid="${criteria}"]`).or(
        this.page.locator(PlaywrightGameHelper.SELECTORS.actionButton).filter({ hasText: criteria })
      );
    } else if (criteria.type) {
      // Select by action type and optional value
      const actions = await this.getAvailableActions();
      const action = actions.find(a => 
        a.type === criteria.type && 
        (criteria.value === undefined || a.value == criteria.value)
      );
      if (action) {
        return this.selectAction(action.index);
      }
    }
    
    if (!locator) {
      throw new Error(`No action found matching criteria: ${JSON.stringify(criteria)}`);
    }
    
    // Click and wait for state change
    const preClickPhase = await this.getCurrentPhase();
    await locator.click();
    
    // Wait for any state change (phase or action availability)
    try {
      await this.page.waitForFunction(
        ({ prevPhase }: { prevPhase: string }) => {
          const currentPhase = document.querySelector('.app-container')?.getAttribute('data-phase');
          const hasActions = document.querySelectorAll('button[data-testid]').length > 0;
          return currentPhase !== prevPhase || hasActions;
        },
        { prevPhase: preClickPhase },
        { timeout: PlaywrightGameHelper.TIMEOUTS.quick }
      );
    } catch {
      // State might have already changed, continue
    }
  }

  /**
   * Get all available actions in current state
   */
  async getAvailableActions(): Promise<ActionInfo[]> {
    // First ensure there are action buttons or we're in a stable state
    await this.page.waitForFunction(
      () => {
        const buttons = document.querySelectorAll('button[data-testid]');
        // Either we have buttons, or we're in a phase that doesn't need them
        const phase = document.querySelector('.app-container')?.getAttribute('data-phase');
        return buttons.length > 0 || phase === 'game_over' || phase === 'waiting';
      },
      { timeout: PlaywrightGameHelper.TIMEOUTS.quick }
    );
    
    const buttons = await this.page.locator(PlaywrightGameHelper.SELECTORS.actionButton).all();
    const actions = [];
    
    for (let i = 0; i < buttons.length; i++) {
      const button = buttons[i];
      const testId = await button!.getAttribute('data-testid');
      const text = await button!.textContent();
      
      // Parse action type and value from testId or text
      let type = 'unknown';
      let value = undefined;
      
      if (testId) {
        if (testId === 'pass') {
          type = 'pass';
        } else if (testId.startsWith('bid-')) {
          const parts = testId.split('-');
          type = parts[2] === 'marks' ? 'bid_marks' : 'bid_points';
          value = parseInt(parts[1] || '0');
        } else if (testId.startsWith('trump-')) {
          type = 'trump_selection';
          value = testId.substring(6);
        } else if (testId.startsWith('domino-')) {
          type = 'play_domino';
          value = testId.substring(7);
        } else {
          // Map common action IDs
          const actionMap: Record<string, string> = {
            'complete-trick': 'complete_trick',
            'score-hand': 'score_hand',
            'next-trick': 'next_trick',
            'redeal': 'redeal',
            'start-hand': 'start_hand'
          };
          const mapped = actionMap[testId as keyof typeof actionMap];
          type = mapped || testId;
        }
      } else if (text) {
        // Fallback to text-based detection
        const textLower = text.toLowerCase();
        if (textLower.includes('complete')) type = 'complete_trick';
        else if (textLower.includes('score')) type = 'score_hand';
        else if (textLower.includes('next')) type = 'next_trick';
        else if (textLower.includes('deal')) type = 'redeal';
      }
      
      actions.push({ index: i, type, value, id: testId || `action-${i}` });
    }
    
    return actions;
  }

  /**
   * Perform bid action
   */
  async bid(value: number, isMarks = false): Promise<void> {
    const selector = PlaywrightGameHelper.SELECTORS.bid(value, isMarks);
    await this.page.locator(selector).click();
  }

  /**
   * Pass bid
   */
  async pass(): Promise<void> {
    await this.page.locator(PlaywrightGameHelper.SELECTORS.pass).click();
  }

  /**
   * Set trump suit
   */
  async setTrump(suit: string): Promise<void> {
    // Normalize suit name
    const suitMap: Record<string, string> = {
      '0s': 'blanks', '1s': 'ones', '2s': 'twos', '3s': 'threes',
      '4s': 'fours', '5s': 'fives', '6s': 'sixes', 'doubles': 'doubles'
    };
    const normalizedSuit = (suitMap[suit] || suit).toLowerCase();
    
    // Ensure we're in trump selection phase
    const currentPhase = await this.getCurrentPhase();
    if (currentPhase !== 'trump_selection') {
      throw new Error(`Cannot set trump: not in trump_selection phase (current: ${currentPhase})`);
    }
    
    // Wait for trump buttons to be available
    const trumpButton = this.page.locator(`[data-testid="trump-${normalizedSuit}"]`);
    await trumpButton.waitFor({ state: 'visible', timeout: PlaywrightGameHelper.TIMEOUTS.normal });
    
    // Click via JavaScript to bypass any overlay issues
    await trumpButton.evaluate((el: Element) => (el as HTMLElement).click());
    
    // Wait for phase to change from trump_selection
    await this.page.waitForFunction(
      () => {
        const phase = document.querySelector('.app-container')?.getAttribute('data-phase');
        return phase && phase !== 'trump_selection';
      },
      { timeout: PlaywrightGameHelper.TIMEOUTS.normal }
    );
  }

  /**
   * Play any available domino
   */
  async playAnyDomino(): Promise<void> {
    // Ensure page is still valid and game is ready
    await this.waitForGameReady();
    
    // Dismiss any flash messages first
    const flash = this.page.locator(PlaywrightGameHelper.SELECTORS.flash);
    if (await flash.isVisible()) {
      await flash.click();
    }
    
    // Click first playable domino using auto-waiting
    const playable = this.page.locator(PlaywrightGameHelper.SELECTORS.domino.playable)
      .or(this.page.locator(PlaywrightGameHelper.SELECTORS.domino.any + ':not([disabled])'))
      .first();
    
    // Try to find and click a playable domino with a longer timeout
    try {
      await playable.waitFor({ state: 'visible', timeout: 3000 });
      await playable.click({ force: true });
    } catch {
      // If no playable dominoes visible, try any domino
      try {
        const anyDomino = this.page.locator(PlaywrightGameHelper.SELECTORS.domino.any).first();
        const count = await anyDomino.count();
        if (count > 0) {
          await anyDomino.click({ force: true });
        } else {
          throw new Error('No dominoes found to play');
        }
      } catch (err: any) {
        // If page/context is closed, throw a more informative error
        if (err.message?.includes('Target page, context or browser has been closed')) {
          throw new Error('Page was closed while trying to play domino');
        }
        throw err;
      }
    }
    
    // Wait for any state changes after playing domino
    // Use waitForFunction to detect when game processes the move
    await this.page.waitForFunction(
      () => {
        const state = (window as any).getGameState?.();
        return state && !(state as any).isProcessing;
      },
      { timeout: PlaywrightGameHelper.TIMEOUTS.quick }
    );
  }

  /**
   * Play specific domino
   */
  async playDomino(dominoId: string): Promise<void> {
    await this.page.locator(PlaywrightGameHelper.SELECTORS.domino.byId(dominoId)).click();
  }

  /**
   * Get game metrics (scores, marks, trump, etc.)
   */
  async getGameMetrics(): Promise<GameMetrics> {
    const [usScore, themScore] = await Promise.all([
      this.page.locator(PlaywrightGameHelper.SELECTORS.score.us).textContent(),
      this.page.locator(PlaywrightGameHelper.SELECTORS.score.them).textContent()
    ]);
    
    // Try to get trump from multiple possible locations
    let trump = '';
    
    // First try the data-testid="trump-display" element (for non-playing phases)
    const trumpDisplay = this.page.locator('[data-testid="trump-display"]');
    if (await trumpDisplay.count() > 0) {
      const trumpText = await trumpDisplay.textContent();
      // Extract just the trump value from "Trump: XXX" format
      trump = trumpText?.replace('Trump: ', '') || '';
    } else {
      // During playing phase, look for trump in the game info bar
      const trumpElement = this.page.locator('.text-secondary:has-text("trump")');
      if (await trumpElement.count() > 0) {
        const trumpText = await trumpElement.textContent();
        // Extract trump from "XXX trump" format
        trump = trumpText?.replace(' trump', '') || '';
      }
    }
    
    return {
      scores: [
        parseInt(usScore?.match(/\d+/)?.[0] || '0'),
        parseInt(themScore?.match(/\d+/)?.[0] || '0')
      ],
      trump: trump
    };
  }

  /**
   * Get player's hand
   */
  async getPlayerHand(playerIndex = 0): Promise<string[]> {
    if (playerIndex !== 0) return []; // Only support current player
    
    const dominoes = await this.page.locator(PlaywrightGameHelper.SELECTORS.domino.any).all();
    const hand = [];
    
    for (const domino of dominoes) {
      const testId = await domino.getAttribute('data-testid');
      const match = testId?.match(/domino-(\d)-(\d)/);
      if (match) {
        hand.push(`${match[1]}-${match[2]}`);
      }
    }
    
    return hand;
  }

  /**
   * Get current trick
   */
  async getCurrentTrick(): Promise<string[]> {
    return await this.page.locator(PlaywrightGameHelper.SELECTORS.trick.played).allTextContents();
  }

  /**
   * Perform common game actions
   */
  async performGameAction(action: string): Promise<void> {
    const actionMap: Record<string, string | (() => Promise<void>)> = {
      'complete': 'complete-trick',
      'score': 'score-hand',
      'next': 'next-trick',
      'redeal': 'redeal',
      'new': async (): Promise<void> => { await this.page.reload(); }
    };
    
    const mapped = actionMap[action];
    if (typeof mapped === 'function') {
      await mapped();
    } else {
      await this.selectAction(mapped || action);
    }
  }

  /**
   * Complete a full trick (4 plays)
   * NOTE: This requires players 1-3 to be AI for automated play
   */
  async playFullTrick(): Promise<void> {
    // Check if AI is already enabled for players 1-3
    const needsAI = await this.page.evaluate(() => {
      const state = (window as any).getGameState?.();
      if (!state || !state.playerTypes) return true;
      // Check if players 1-3 are already AI
      return state.playerTypes[1] !== 'ai' || 
             state.playerTypes[2] !== 'ai' || 
             state.playerTypes[3] !== 'ai';
    });
    
    // Only enable AI if not already enabled
    if (needsAI) {
      await this.enableAIForOtherPlayers();
    }
    
    // Play one domino as player 0
    await this.playAnyDomino();
    
    // In test mode, AI should execute synchronously after human action
    // Just verify trick is complete (should have 4 dominoes)
    await this.page.waitForFunction(
      () => {
        const state = (window as any).getGameState?.();
        return state && state.currentTrick && state.currentTrick.length === 4;
      },
      { timeout: PlaywrightGameHelper.TIMEOUTS.quick } // Quick timeout since it's synchronous
    );
  }

  /**
   * Complete bidding round
   */
  async completeBidding(bids: Array<{ type: string; value?: number }>): Promise<void> {
    for (const bid of bids) {
      await this.page.waitForSelector(PlaywrightGameHelper.SELECTORS.actionButton);
      
      if (bid.type === 'pass') {
        await this.pass();
      } else if (bid.type === 'points') {
        await this.bid(bid.value || 30, false);
      } else if (bid.type === 'marks') {
        await this.bid(bid.value || 1, true);
      }
      
      // Wait for next player's turn
      await this.page.waitForFunction(
        () => document.querySelectorAll('button[data-testid]').length > 0,
        null,
        { timeout: PlaywrightGameHelper.TIMEOUTS.quick }
      );
    }
  }

  /**
   * Load game state from URL
   */
  async loadStateWithActions(
    seed: number, 
    actions: string[], 
    playerTypes: ('human' | 'ai')[] = ['human', 'human', 'human', 'human'],
    dealer?: number,
    tournamentMode?: boolean
  ): Promise<void> {
    const urlStr = encodeGameUrl(seed, actions, playerTypes, dealer, tournamentMode);
    
    // Include testMode=true to disable controllers, but AI will execute synchronously
    // when explicitly specified in the URL data
    await this.page.goto(`${urlStr}&testMode=true`, { 
      waitUntil: 'networkidle',
      timeout: PlaywrightGameHelper.TIMEOUTS.slow 
    });
    await this.waitForGameReady();
    
    // If AI players are specified, they execute synchronously in test mode
    // Just verify the state is stable after loading
    if (playerTypes.some(t => t === 'ai')) {
      await this.page.waitForFunction(
        () => {
          const state = (window as any).getGameState?.();
          if (!state) return false;
          // State is stable when current player is human or game ended
          return state.playerTypes[state.currentPlayer] === 'human' ||
                 state.phase === 'scoring' ||
                 state.phase === 'game_over';
        },
        { timeout: PlaywrightGameHelper.TIMEOUTS.quick }
      );
    }
  }

  /**
   * Navigate tabs - DEPRECATED
   * The game automatically shows the correct view based on game state
   * This method is kept only for backward compatibility but should not be used
   */
  async navigateTo(tab: string): Promise<void> {
    // Do nothing - the game handles tab switching automatically
    console.warn(`navigateTo('${tab}') called but is deprecated - game handles tab switching automatically`);
  }

  /**
   * Debug panel operations
   */
  async openDebugPanel(): Promise<void> {
    // First open the dropdown menu (three dots button)
    const dropdownButton = this.page.locator('.dropdown-end .btn-ghost.btn-circle');
    await dropdownButton.click();
    
    // Wait for dropdown menu to be visible
    await this.page.waitForSelector('.dropdown-content', { state: 'visible' });
    
    // Now click the settings button inside the dropdown
    await this.page.locator(PlaywrightGameHelper.SELECTORS.debug.button).click();
    
    // Wait for debug panel to be visible (it's a fixed fullscreen drawer)
    await expect(this.page.locator(PlaywrightGameHelper.SELECTORS.debug.panel)).toBeVisible();
  }

  async closeDebugPanel(): Promise<void> {
    await this.page.locator(PlaywrightGameHelper.SELECTORS.debug.closeButton).click();
    // Wait for debug panel to close
    await expect(this.page.locator(PlaywrightGameHelper.SELECTORS.debug.panel)).not.toBeVisible();
  }

  /**
   * Get current URL
   */
  async getCurrentURL(): Promise<string> {
    return this.page.url();
  }

  /**
   * Check if URL has snapshot
   */
  async hasSnapshotInURL(): Promise<boolean> {
    const url = await this.getCurrentURL();
    return url.includes('d=');
  }

  /**
   * Enable AI for other players
   */
  async enableAIForOtherPlayers(): Promise<void> {
    await this.page.evaluate(() => {
      const gameWindow = window as any;
      
      // Use the enableAI action if available
      if (gameWindow.gameActions && gameWindow.gameActions.enableAI) {
        gameWindow.gameActions.enableAI();
      } else {
        // Fallback to direct state update if action not available
        if (gameWindow.gameState && gameWindow.getGameState) {
          const currentState = gameWindow.getGameState();
          // Set players 1-3 as AI
          const newState = {
            ...currentState,
            playerTypes: ['human', 'ai', 'ai', 'ai'] as ('human' | 'ai')[]
          };
          // Update the game state directly - gameState now has methods exposed
          if (typeof gameWindow.gameState.set === 'function') {
            gameWindow.gameState.set(newState);
          } else if (typeof gameWindow.gameState === 'object' && gameWindow.gameState.set) {
            // Already an object with set method
            gameWindow.gameState.set(newState);
          }
        }
      }
      
      // Also enable quickplay if available (for backward compatibility)
      if (gameWindow.quickplayActions && gameWindow.getQuickplayState) {
        const state = gameWindow.getQuickplayState();
        for (let i = 1; i <= 3; i++) {
          if (!state.aiPlayers.has(i)) {
            gameWindow.quickplayActions.togglePlayer(i);
          }
        }
        if (!state.enabled) {
          gameWindow.quickplayActions.toggle();
        }
      }
    });
  }

  /**
   * Wait for AI to complete its move(s)
   * In test mode, AI executes synchronously, so this just verifies state
   */
  async waitForAIMove(_expectedActionCount?: number): Promise<void> {
    // In test mode, AI executes synchronously, so just verify state has changed
    await this.page.waitForFunction(
      () => {
        const state = (window as any).getGameState?.();
        if (!state) return false;
        
        // AI has finished if it's human's turn or game phase changed
        return state.playerTypes[state.currentPlayer] === 'human' ||
               state.phase === 'scoring' ||
               state.phase === 'game_over';
      },
      { timeout: PlaywrightGameHelper.TIMEOUTS.quick } // Quick timeout since it's synchronous
    );

    // Also ensure game state is stable
    await this.waitForGameReady();
  }

  /**
   * Perform complete game with strategy
   */
  async performCompleteGame(strategy = 'random'): Promise<void> {
    while (true) {
      const phase = await this.getCurrentPhase();
      if (phase.includes('game_end') || phase.includes('complete')) break;
      
      const actions = await this.getAvailableActions();
      if (actions.length === 0) break;
      
      let index = 0;
      switch (strategy) {
        case 'aggressive':
          index = actions.length - 1;
          break;
        case 'conservative':
          index = actions.findIndex(a => a.type === 'pass') || 0;
          break;
        default:
          index = Math.floor(Math.random() * actions.length);
      }
      
      await this.selectAction(index);
    }
  }

  /**
   * Simplified locator access
   */
  locator(selector: string): Locator {
    // Map old selectors to new ones for compatibility
    const mappings: Record<string, string> = {
      '.trick-horizontal': '.trick-table',
      '.trick-position': '.trick-spot',
      '.proceed-action-button': '[data-testid*="complete-trick"], [data-testid*="agree-complete-trick"]',
      '[data-testid="complete-trick"]': '[data-testid*="complete-trick"], [data-testid*="agree-complete-trick"]',
      '.history-row': '.history-item',
      '.trump-display': PlaywrightGameHelper.SELECTORS.trump
    };
    
    return this.page.locator(mappings[selector] || selector);
  }

  /**
   * Check if current player is human
   */
  async isCurrentPlayerHuman(): Promise<boolean> {
    return await this.page.evaluate(() => {
      const state = (window as any).getGameState?.();
      if (!state) return true; // Default to human if no state
      return state.playerTypes[state.currentPlayer] === 'human';
    });
  }

  /**
   * Check if it's currently an AI player's turn
   */
  async isAITurn(): Promise<boolean> {
    return await this.page.evaluate(() => {
      const state = (window as any).getGameState?.();
      if (!state) return false;
      return state.playerTypes[state.currentPlayer] === 'ai';
    });
  }

  /**
   * Wait until it's a human player's turn
   */
  async waitForHumanTurn(): Promise<void> {
    await this.page.waitForFunction(
      () => {
        const state = (window as any).getGameState?.();
        if (!state) return false;
        // Wait for human turn or game end
        return state.playerTypes[state.currentPlayer] === 'human' ||
               state.phase === 'scoring' ||
               state.phase === 'game_over';
      },
      { timeout: PlaywrightGameHelper.TIMEOUTS.normal }
    );
  }

  /**
   * Check if player 0 can currently take actions
   */
  async canPlayerAct(): Promise<boolean> {
    return await this.page.evaluate(() => {
      const state = (window as any).getGameState?.();
      if (!state) return false;
      // Player 0 can act if it's their turn or there are consensus actions
      return state.currentPlayer === 0 || 
             (window as any).getAvailableActions?.().some((a: any) => 
               a.id.includes('agree-') || a.id === 'complete-trick' || a.id === 'score-hand'
             );
    });
  }

  /**
   * Get page object for direct access
   */
  getPage(): Page {
    return this.page;
  }

  /**
   * Get locators for common elements
   */
  getLocators() {
    return {
      app: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.app),
      appHeader: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.appHeader),
      playingArea: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.playingArea),
      trickTable: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.trick.table),
      trickTableTappable: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.trick.tappable),
      tapIndicator: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.tapIndicator),
      tapText: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.tapText),
      nav: (item: string): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.nav(item)),
      flash: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.flash),
      debug: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.debug.panel),
      settingsButton: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.settings.button),
      settingsCloseButton: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.settings.closeButton),
      debugButton: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.settings.button),
      debugCloseButton: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.settings.closeButton),
      historyTab: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.debug.historyTab).filter({ hasText: 'History' }),
      historyItem: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.debug.historyItem),
      timeTravelButton: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.debug.timeTravelButton),
      scoreDisplay: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.score.display),
      scoreUs: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.score.us),
      scoreThem: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.score.them),
      trump: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.trump),
      turnPlayer: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.turnPlayer)
    };
  }
}

// Export singleton pattern for backward compatibility
export const playwrightHelper = {
  loadState: async (page: Page, state: { shuffleSeed: number }) => {
    const helper = new PlaywrightGameHelper(page);
    return helper.loadStateWithActions(state.shuffleSeed, []);
  },
  clickAction: async (page: Page, actionId: string | number) => {
    const helper = new PlaywrightGameHelper(page);
    return helper.selectAction(actionId);
  },
  getAvailableActions: async (page: Page) => {
    const helper = new PlaywrightGameHelper(page);
    return helper.getAvailableActions();
  }
};
