import './styles/app.css';
import { mount } from 'svelte';
import App from './App.svelte';
import PerfectsApp from './PerfectsApp.svelte';
import { SEED_FINDER_CONFIG } from './game/ai/gameSimulator';
import { seedFinderStore } from './stores/seedFinderStore';
import { getGameView } from './stores/gameStore';

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

if (typeof window !== 'undefined' && !isPerfectsPage) {
  // Seed finder (user-facing feature)
  window.SEED_FINDER_CONFIG = SEED_FINDER_CONFIG;
  window.seedFinderStore = seedFinderStore;

  // Test support - expose minimal API for E2E tests
  window.getGameView = getGameView;
}

export default app;
