import './styles/app.css';
import App from './App.svelte';
import { mount } from 'svelte';
import { gameActions } from './stores/gameStore';

// Check for state or snapshot parameter in URL and load it automatically
function loadStateFromURL() {
  const urlParams = new window.URLSearchParams(window.location.search);
  const snapshotParam = urlParams.get('snapshot');
  const stateParam = urlParams.get('state');
  
  if (snapshotParam) {
    try {
      const decodedSnapshot = decodeURIComponent(snapshotParam);
      const snapshotData = JSON.parse(decodedSnapshot);
      
      // Load the base state and replay all actions to get to current state
      gameActions.loadStateWithActionReplay(snapshotData.baseState, snapshotData.actions);
      
      console.log('Loaded game state from snapshot URL', {
        reason: snapshotData.reason,
        actionCount: snapshotData.actions.length
      });
    } catch (error) {
      console.error('Failed to load snapshot from URL:', error);
    }
  } else if (stateParam) {
    try {
      const decodedState = decodeURIComponent(stateParam);
      const gameState = JSON.parse(decodedState);
      gameActions.loadState(gameState);
      console.log('Loaded game state from URL');
    } catch (error) {
      console.error('Failed to load state from URL:', error);
    }
  }
}

// Load state from URL before mounting the app
loadStateFromURL();

const app = mount(App, {
  target: document.getElementById('app')!,
});

export default app;