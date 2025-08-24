import './styles/app.css';
import App from './AppEvent.svelte';
import { mount } from 'svelte';
import { eventGame } from './stores/eventGame';

const app = mount(App, {
  target: document.getElementById('app')!,
});

declare global {
  interface Window {
    eventGame: typeof eventGame;
  }
}

if (typeof window !== 'undefined') {
  window.eventGame = eventGame;
}

export default app;