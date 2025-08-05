import './styles/app.css';
import App from './App.svelte';
import { mount } from 'svelte';
import { gameActions } from './stores/gameStore';

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

export default app;