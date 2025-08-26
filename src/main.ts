import './styles/app.css';
import { mount } from 'svelte';
import { get } from 'svelte/store';
import type { Writable } from 'svelte/store';
import App from './App.svelte';
import { gameActions, gameState, actionHistory, controllerManager } from './stores/gameStore';
import { quickplayActions, quickplayState } from './stores/quickplayStore';
import type { GameState } from './game/types';

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
}

export default app;