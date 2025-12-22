import './styles/app.css';
import { mount } from 'svelte';
import App from './App.svelte';
import { SEED_FINDER_CONFIG } from './game/ai/gameSimulator';
import { seedFinderStore } from './stores/seedFinderStore';
import { getGameView } from './stores/gameStore';

const app = mount(App, {
  target: document.getElementById('app')!,
});

// =============================================================================
// Window API for User-Facing Features + Testing
// =============================================================================
// Seed finder is exposed for user access (documented feature)
// getGameView is exposed for E2E test support
// =============================================================================

declare global {
  interface Window {
    // Seed finder (user-facing feature)
    SEED_FINDER_CONFIG: typeof SEED_FINDER_CONFIG;
    seedFinderStore: typeof seedFinderStore;
    // Test support
    getGameView?: typeof getGameView;
  }
}

if (typeof window !== 'undefined') {
  // Seed finder (user-facing feature)
  window.SEED_FINDER_CONFIG = SEED_FINDER_CONFIG;
  window.seedFinderStore = seedFinderStore;

  // Test support - expose minimal API for E2E tests
  window.getGameView = getGameView;
}

export default app;
