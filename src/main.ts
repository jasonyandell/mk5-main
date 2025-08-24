import './styles/app.css';
import { mount } from 'svelte';
import { get } from 'svelte/store';

const urlParams = new URLSearchParams(window.location.search);
const useEventSystem = urlParams.has('event');

async function initApp() {
  let app: any;

  if (useEventSystem) {
    const AppEvent = await import('./AppEvent.svelte');
    const { eventGame } = await import('./stores/eventGame');
    
    app = mount(AppEvent.default, {
      target: document.getElementById('app')!,
    });
    
    if (typeof window !== 'undefined') {
      (window as any).eventGame = eventGame;
    }
  } else {
    const App = await import('./App.svelte');
    const { gameActions, gameState } = await import('./stores/gameStore');
    const { quickplayActions, quickplayState } = await import('./stores/quickplayStore');
    const { actionHistory, controllerManager } = await import('./stores/gameStore');

    app = mount(App.default, {
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
  }

  return app;
}

const app = await initApp();

export default app;