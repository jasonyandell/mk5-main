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

// Expose quickplay and gameState for testing
declare global {
  interface Window {
    quickplayActions: typeof quickplayActions;
    quickplayState: typeof quickplayState;
    getQuickplayState: () => ReturnType<typeof get>;
    gameActions: typeof gameActions;
    getGameState: () => ReturnType<typeof get>;
  }
}

if (typeof window !== 'undefined') {
  window.quickplayActions = quickplayActions;
  window.quickplayState = quickplayState;
  window.getQuickplayState = () => get(quickplayState);
  window.gameActions = gameActions;
  window.getGameState = () => get(gameState);
}

export default app;