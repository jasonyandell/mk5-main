import './styles/app.css';
import { mount } from 'svelte';
import { get } from 'svelte/store';
import App from './App.svelte';
import PerfectsApp from './PerfectsApp.svelte';
import { gameActions, gameState, viewProjection } from './stores/gameStore';
import { setAISpeedProfile } from './game/core/ai-scheduler';
import { getNextStates } from './game';
import { quickplayActions, quickplayState } from './stores/quickplayStore';
import { SEED_FINDER_CONFIG } from './game/core/seedFinder';
import { seedFinderStore } from './stores/seedFinderStore';
import type { StateTransition } from './game/types';

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

declare global {
  interface Window {
    quickplayActions: typeof quickplayActions;
    quickplayState: typeof quickplayState;
    getQuickplayState: () => ReturnType<typeof get>;
    gameActions: typeof gameActions;
    gameState: typeof gameState;
    getGameState: () => ReturnType<typeof get>;
    setAISpeedProfile: typeof setAISpeedProfile;
    getNextStates: typeof getNextStates;
    viewProjection: typeof viewProjection;
    playFirstAction: () => void;
    SEED_FINDER_CONFIG: typeof SEED_FINDER_CONFIG;
    seedFinderStore: typeof seedFinderStore;
  }
}

if (typeof window !== 'undefined' && !isPerfectsPage) {
  window.quickplayActions = quickplayActions;
  window.quickplayState = quickplayState;
  window.getQuickplayState = () => get(quickplayState);
  window.gameActions = gameActions;
  window.SEED_FINDER_CONFIG = SEED_FINDER_CONFIG;
  window.seedFinderStore = seedFinderStore;
  // Expose gameState as a readable store (it's derived, not writable)
  window.gameState = gameState;
  window.getGameState = () => get(gameState);
  window.setAISpeedProfile = setAISpeedProfile;
  window.getNextStates = getNextStates;
  window.viewProjection = viewProjection;
  window.playFirstAction = () => {
    const state = get(gameState);
    const ts = getNextStates(state);
    // Priority: consensus/proceed for P0 → complete/score → agree-* → play → bid/pass → trump
    const isP0 = (t: StateTransition) => 'player' in t.action && t.action.player === 0;
    const byId = (id: string) => ts.find((t: StateTransition) => t?.id === id);
    const first =
      byId('complete-trick') ||
      byId('score-hand') ||
      ts.find((t: StateTransition) => t?.id === 'agree-complete-trick-0') ||
      ts.find((t: StateTransition) => t?.id === 'agree-score-hand-0') ||
      ts.find((t: StateTransition) => t?.action?.type === 'play' && isP0(t)) ||
      ts.find((t: StateTransition) => (t?.action?.type === 'bid' || t?.action?.type === 'pass') && isP0(t)) ||
      ts.find((t: StateTransition) => t?.action?.type === 'select-trump') ||
      ts[0];
    if (first) gameActions.executeAction(first);
  };
}

export default app;
