import './styles/app.css';
import { mount } from 'svelte';
import { get } from 'svelte/store';
import App from './App.svelte';
import { gameActions, gameState, actionHistory, controllerManager } from './stores/gameStore';
import { quickplayActions, quickplayState } from './stores/quickplayStore';

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
  window.gameState = gameState;
  window.getGameState = () => get(gameState);
}

export default app;