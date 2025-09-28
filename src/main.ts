import './styles/app.css';
import { mount } from 'svelte';
import { get } from 'svelte/store';
import type { Writable, Readable } from 'svelte/store';
import App from './App.svelte';
import PerfectsApp from './PerfectsApp.svelte';
import { gameActions, gameState, actionHistory, controllerManager, startGameLoop, stopGameLoop, sectionOverlay, viewProjection } from './stores/gameStore';
import { setAISpeedProfile } from './game/core/ai-scheduler';
import { getNextStates } from './game';
import { dispatcher } from './stores/gameStore';
import { quickplayActions, quickplayState } from './stores/quickplayStore';
import { SEED_FINDER_CONFIG } from './game/core/seedFinder';
import { seedFinderStore } from './stores/seedFinderStore';
import type { GameState, StateTransition } from './game/types';

// Route to appropriate app based on URL path
const pathname = window.location.pathname;
const isPerfectsPage = pathname === '/perfects' || pathname === '/perfects/';

const AppComponent = isPerfectsPage ? PerfectsApp : App;
const app = mount(AppComponent, {
  target: document.getElementById('app')!,
});

if (!isPerfectsPage) {
  window.addEventListener('popstate', (event) => {
    if (event.state) {
      gameActions.loadFromHistoryState(event.state);
    } else {
      gameActions.loadFromURL();
    }
  });
}

declare global {
  interface Window {
    __actionHistory?: typeof actionHistory;
    controllerManager?: typeof controllerManager;
    quickplayActions: typeof quickplayActions;
    quickplayState: typeof quickplayState;
    getQuickplayState: () => ReturnType<typeof get>;
    gameActions: typeof gameActions;
    gameState: typeof gameState;
    getGameState: () => ReturnType<typeof get>;
    getSectionOverlay: () => unknown;
    startGameLoop: typeof startGameLoop;
    stopGameLoop: typeof stopGameLoop;
    setAISpeedProfile: typeof setAISpeedProfile;
    getNextStates: typeof getNextStates;
    viewProjection: typeof viewProjection;
    playFirstAction: () => void;
    SEED_FINDER_CONFIG: typeof SEED_FINDER_CONFIG;
    seedFinderStore: typeof seedFinderStore;
  }
}

if (typeof window !== 'undefined' && !isPerfectsPage) {
  window.__actionHistory = actionHistory;
  window.controllerManager = controllerManager;
  window.quickplayActions = quickplayActions;
  window.quickplayState = quickplayState;
  window.getQuickplayState = () => get(quickplayState);
  window.gameActions = gameActions;
  window.SEED_FINDER_CONFIG = SEED_FINDER_CONFIG;
  window.seedFinderStore = seedFinderStore;
  // Properly expose the store with its methods
  window.gameState = {
    set: (state: GameState) => gameState.set(state),
    update: (fn: (state: GameState) => GameState) => gameState.update(fn),
    subscribe: gameState.subscribe,
    get: () => get(gameState)
  } as Writable<GameState> & { get: () => GameState };
  window.getGameState = () => get(gameState);
  window.getSectionOverlay = () => get(sectionOverlay as Readable<unknown>);
  window.startGameLoop = startGameLoop;
  window.stopGameLoop = stopGameLoop;
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
    if (first) dispatcher.requestTransition(first, 'ui');
  };
}

export default app;
