import './styles/app.css';
import App from './App.svelte';
import { mount } from 'svelte';
import { gameActions, gameState } from './stores/gameStore';
import { quickplayActions, quickplayState } from './stores/quickplayStore';
import { get } from 'svelte/store';

const app = mount(App, {
  target: document.getElementById('app')!,
});

// Handle browser back/forward navigation
window.addEventListener('popstate', (event) => {
  if (event.state) {
    gameActions.loadFromHistoryState(event.state);
  } else {
    // If no state, try to load from URL
    gameActions.loadFromURL();
  }
});

// Expose stores for testing and debugging
import { actionHistory, controllerManager } from './stores/gameStore';

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