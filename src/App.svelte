<script lang="ts">
  import { gameState, availableActions, gameActions, gameHistory } from './stores/gameStore';
  import DebugGameState from './debug/components/DebugGameState.svelte';
  import DebugActions from './debug/components/DebugActions.svelte';
  import DebugPlayerHands from './debug/components/DebugPlayerHands.svelte';
  import DebugJsonView from './debug/components/DebugJsonView.svelte';
  import DebugBugReport from './debug/components/DebugBugReport.svelte';
  import DebugReplay from './debug/components/DebugReplay.svelte';
  import type { StateTransition } from './game/types';
  
  function handleAction(transition: StateTransition) {
    gameActions.executeAction(transition);
  }
  
  function resetGame() {
    gameActions.resetGame();
  }
  
  function loadState() {
    const input = prompt('Paste JSON state:');
    if (input) {
      try {
        const state = JSON.parse(input);
        gameActions.loadState(state);
      } catch (e) {
        alert('Invalid JSON state');
      }
    }
  }
  
  function copyStateURL() {
    const stateParam = encodeURIComponent(JSON.stringify($gameState));
    const url = `${window.location.origin}${window.location.pathname}?state=${stateParam}`;
    navigator.clipboard.writeText(url).then(() => {
      alert('State URL copied to clipboard!');
    }).catch(() => {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = url;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      alert('State URL copied to clipboard!');
    });
  }
  
  function undo() {
    gameActions.undo();
  }
  
  let showBugPanel = $state(false);
</script>

<main class="debug-container">
  <header class="debug-header">
    <div class="header-left">
      <h1>Texas 42 Debug Interface</h1>
      <div class="phase-indicator phase-{$gameState.phase}">
        {$gameState.phase.replace('_', ' ').toUpperCase()}
      </div>
    </div>
    <div class="header-controls">
      <button class="control-btn undo-btn" onclick={undo} disabled={$gameHistory.length === 0} data-testid="undo-button">
        Undo
      </button>
      <button class="control-btn" onclick={resetGame} data-testid="new-game-button">
        New Game
      </button>
      <button class="control-btn secondary" onclick={loadState}>
        Load State
      </button>
      <button class="control-btn secondary" onclick={copyStateURL} data-testid="copy-state-url-button">
        Copy State URL
      </button>
    </div>
  </header>
  
  <DebugBugReport 
    gameState={$gameState} 
    generateBugReport={gameActions.generateBugReport}
    bind:showBugPanel
  />
  
  <div class="debug-layout">
    <div class="debug-left">
      <DebugGameState gameState={$gameState} />
      <DebugReplay />
    </div>
    
    <div class="debug-center">
      <div class="center-top">
        <DebugPlayerHands gameState={$gameState} />
      </div>
      <div class="center-bottom">
        <DebugActions 
          availableActions={$availableActions}
          onAction={handleAction}
        />
      </div>
    </div>
    
    <div class="debug-right">
      <DebugJsonView gameState={$gameState} />
    </div>
  </div>
  
  <!-- Hidden test element for e2e tests -->
  <div data-testid="game-phase" style="display: none;">{$gameState.phase}</div>
</main>

<style>
  :global(body) {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f8f9fa;
  }
  
  .debug-container {
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  
  .debug-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: #343a40;
    color: white;
    border-bottom: 1px solid #495057;
    flex-shrink: 0;
  }
  
  .header-left {
    display: flex;
    align-items: center;
    gap: 16px;
  }
  
  .debug-header h1 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
  }
  
  .phase-indicator {
    padding: 4px 8px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  
  .phase-bidding { background: #007bff; }
  .phase-trump_selection { background: #fd7e14; }
  .phase-playing { background: #28a745; }
  .phase-scoring { background: #6f42c1; }
  .phase-game_end { background: #dc3545; }
  
  .header-controls {
    display: flex;
    gap: 8px;
  }
  
  .control-btn {
    background: #6c757d;
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 3px;
    font-size: 12px;
    cursor: pointer;
    transition: background 0.1s ease;
  }
  
  .control-btn:hover {
    background: #5a6268;
  }
  
  .control-btn.secondary {
    background: #495057;
  }
  
  .control-btn.secondary:hover {
    background: #383d43;
  }
  
  .control-btn.undo-btn {
    background: #007bff;
  }
  
  .control-btn.undo-btn:hover:not(:disabled) {
    background: #0056b3;
  }
  
  .control-btn.undo-btn:disabled {
    background: #495057;
    opacity: 0.6;
    cursor: not-allowed;
  }
  
  
  .debug-layout {
    flex: 1;
    display: grid;
    grid-template-columns: 300px 1fr 300px;
    gap: 1px;
    background: #dee2e6;
    min-height: 0;
  }
  
  .debug-left,
  .debug-right {
    background: white;
    overflow: hidden;
  }
  
  .debug-center {
    background: white;
    display: flex;
    flex-direction: column;
    gap: 1px;
    overflow: hidden;
  }
  
  .center-top {
    flex-shrink: 0;
    background: white;
    overflow: hidden;
  }
  
  .center-bottom {
    flex: 1;
    background: white;
    overflow: hidden;
    position: relative;
    z-index: 1;
  }
  
  @media (max-width: 1200px) {
    .debug-layout {
      grid-template-columns: 250px 1fr 250px;
    }
  }
  
  @media (max-width: 1024px) {
    .debug-layout {
      grid-template-columns: 1fr;
      grid-template-rows: auto auto auto;
    }
    
    .debug-right {
      flex-direction: row;
    }
  }
  
  @media (max-width: 768px) {
    .debug-header {
      flex-direction: column;
      gap: 8px;
      padding: 8px;
    }
    
    .header-left {
      flex-direction: column;
      gap: 8px;
      text-align: center;
    }
    
    .debug-header h1 {
      font-size: 16px;
    }
  }
  
  @media (prefers-reduced-motion: reduce) {
    .control-btn {
      transition: none;
    }
  }
</style>