import './styles/app.css';
import { mount } from 'svelte';
import { get } from 'svelte/store';
import App from './App.svelte';
import PerfectsApp from './PerfectsApp.svelte';
import { game, gameState, gameClient } from './stores/gameStore';
import { SEED_FINDER_CONFIG } from './game/core/seedFinder';
import { seedFinderStore } from './stores/seedFinderStore';
import type { GameView } from './shared/multiplayer/protocol';
import { NetworkGameClient } from './game/multiplayer/NetworkGameClient';

// Route to appropriate app based on URL path
const pathname = window.location.pathname;
// Handle both local dev (/perfects) and GitHub Pages (/mk5-main/perfects)
const isPerfectsPage = pathname === '/perfects' ||
                      pathname === '/perfects/' ||
                      pathname.endsWith('/perfects') ||
                      pathname.endsWith('/perfects/');

const AppComponent = isPerfectsPage ? PerfectsApp : App;
const app = mount(AppComponent, {
  target: document.getElementById('app')!,
});

if (!isPerfectsPage) {
  window.addEventListener('popstate', () => {
    // History loading is deprecated in the new GameClient architecture
    // URL-based game state loading would need to be reimplemented if needed
    console.warn('History state loading not yet implemented in new architecture');
  });
}

// =============================================================================
// Window API for Development & Testing
// =============================================================================
// Minimal exposure for:
// 1. Browser console debugging (game developers)
// 2. E2E test verification (read-only state inspection)
//
// IMPORTANT: Tests should prefer DOM inspection over window access.
// Only use window API when DOM doesn't reflect the state you need to verify.
// =============================================================================

declare global {
  interface Window {
    // State inspection (read-only, for debugging/testing)
    getGameView: () => GameView;

    // Development/debug tools
    game: typeof game; // Execute actions from console

    // Seed finder (user-facing feature)
    SEED_FINDER_CONFIG: typeof SEED_FINDER_CONFIG;
    seedFinderStore: typeof seedFinderStore;
  }
}

if (typeof window !== 'undefined' && !isPerfectsPage) {
  // Read-only state inspection (minimal exposure for testing/debugging)
  window.getGameView = () => {
    const client = gameClient as NetworkGameClient;
    const cachedView = client.getCachedView();
    if (!cachedView) {
      // Return a minimal GameView structure
      const state = get(gameState);
      return {
        state,
        validActions: [],
        players: [],
        metadata: {
          gameId: 'unknown'
        }
      };
    }
    return cachedView;
  };

  // Development tools (execute actions from browser console)
  window.game = game;

  // Seed finder (user-facing feature)
  window.SEED_FINDER_CONFIG = SEED_FINDER_CONFIG;
  window.seedFinderStore = seedFinderStore;
}

export default app;
