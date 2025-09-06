import './styles/app.css';
import { mount } from 'svelte';
import { get } from 'svelte/store';
import type { Writable, Readable } from 'svelte/store';
import App from './App.svelte';
import { gameActions, gameState, actionHistory, controllerManager, startGameLoop, stopGameLoop, sectionOverlay, viewProjection } from './stores/gameStore';
import { setAISpeedProfile } from './game/core/ai-scheduler';
import { getNextStates } from './game';
import { dispatcher } from './stores/gameStore';
import { quickplayActions, quickplayState } from './stores/quickplayStore';
import type { GameState, StateTransition } from './game/types';

const app = mount(App, {
  target: document.getElementById('app')!,
});

window.addEventListener('popstate', (event) => {
  if (event.state) {
    gameActions.loadFromHistoryState(event.state);
  } else {
    gameActions.loadFromURL();
  }
});

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
  }
}

if (typeof window !== 'undefined') {
  window.__actionHistory = actionHistory;
  window.controllerManager = controllerManager;
  window.quickplayActions = quickplayActions;
  window.quickplayState = quickplayState;
  window.getQuickplayState = () => get(quickplayState);
  window.gameActions = gameActions;
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
