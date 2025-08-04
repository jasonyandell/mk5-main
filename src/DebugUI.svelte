<script lang="ts">
  import { gameState, availableActions, actionHistory, stateValidationError } from './stores/gameStore';
  import DebugGameState from './debug/DebugGameState.svelte';
  import DebugPreviousTricks from './debug/DebugPreviousTricks.svelte';
  import DebugPlayerHands from './debug/DebugPlayerHands.svelte';
  import DebugActions from './debug/DebugActions.svelte';
  import DebugReplay from './debug/DebugReplay.svelte';
  import DebugJsonView from './debug/DebugJsonView.svelte';
  import DebugBugReport from './debug/DebugBugReport.svelte';
  import DebugTestGen from './debug/DebugTestGen.svelte';
  import DebugQuickPlay from './debug/DebugQuickPlay.svelte';
  
  let isOpen = false;
  let activeTab = 'state';
  
  function toggleDebugUI() {
    isOpen = !isOpen;
  }
  
  function handleKeydown(e: KeyboardEvent) {
    if (e.ctrlKey && e.shiftKey && e.key === 'D') {
      toggleDebugUI();
    }
  }
</script>

<svelte:window on:keydown={handleKeydown} />

<button 
  class="debug-toggle"
  on:click={toggleDebugUI}
  data-testid="debug-toggle"
  title="Toggle Debug UI (Ctrl+Shift+D)"
>
  🐛
</button>

{#if isOpen}
  <div class="debug-overlay" data-testid="debug-panel">
    <div class="debug-container">
      <div class="debug-header">
        <h2>Debug UI - Texas 42</h2>
        <div class="debug-controls">
          <div class="tab-bar">
            <button 
              class="tab" 
              class:active={activeTab === 'state'}
              on:click={() => activeTab = 'state'}
              data-testid="tab-state"
            >
              State
            </button>
            <button 
              class="tab" 
              class:active={activeTab === 'actions'}
              on:click={() => activeTab = 'actions'}
              data-testid="tab-actions"
            >
              Actions
            </button>
            <button 
              class="tab" 
              class:active={activeTab === 'history'}
              on:click={() => activeTab = 'history'}
              data-testid="tab-history"
            >
              History
            </button>
            <button 
              class="tab" 
              class:active={activeTab === 'json'}
              on:click={() => activeTab = 'json'}
              data-testid="tab-json"
            >
              JSON
            </button>
            <button 
              class="tab" 
              class:active={activeTab === 'tools'}
              on:click={() => activeTab = 'tools'}
              data-testid="tab-tools"
            >
              Tools
            </button>
          </div>
          <button 
            class="close-btn"
            on:click={toggleDebugUI}
            data-testid="debug-close"
          >
            ✕
          </button>
        </div>
      </div>
      
      {#if $stateValidationError}
        <div class="validation-error" data-testid="validation-error">
          <h3>⚠️ State Validation Error</h3>
          <pre>{$stateValidationError}</pre>
        </div>
      {/if}
      
      <div class="debug-content">
        {#if activeTab === 'state'}
          <div class="state-view">
            <div class="state-column">
              <DebugGameState gameState={$gameState} />
              <DebugPreviousTricks tricks={$gameState.tricks} />
            </div>
            <div class="hands-column">
              <DebugPlayerHands players={$gameState.players} currentPlayer={$gameState.currentPlayer} />
            </div>
          </div>
        {:else if activeTab === 'actions'}
          <DebugActions actions={$availableActions} />
        {:else if activeTab === 'history'}
          <DebugReplay history={$actionHistory} />
        {:else if activeTab === 'json'}
          <DebugJsonView state={$gameState} />
        {:else if activeTab === 'tools'}
          <div class="tools-grid">
            <DebugBugReport />
            <DebugTestGen />
            <DebugQuickPlay />
          </div>
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  .debug-toggle {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: #333;
    color: white;
    border: 2px solid #666;
    font-size: 24px;
    cursor: pointer;
    z-index: 1000;
    transition: all 0.3s;
  }
  
  .debug-toggle:hover {
    background: #555;
    transform: scale(1.1);
  }
  
  .debug-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.9);
    z-index: 2000;
    overflow: auto;
  }
  
  .debug-container {
    max-width: 1600px;
    margin: 20px auto;
    background: #1a1a1a;
    border-radius: 8px;
    color: #e0e0e0;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
  }
  
  .debug-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    border-bottom: 2px solid #333;
    background: #222;
    border-radius: 8px 8px 0 0;
  }
  
  .debug-header h2 {
    margin: 0;
    color: #4CAF50;
  }
  
  .debug-controls {
    display: flex;
    align-items: center;
    gap: 20px;
  }
  
  .tab-bar {
    display: flex;
    gap: 10px;
  }
  
  .tab {
    padding: 8px 16px;
    background: #333;
    border: 1px solid #555;
    border-radius: 4px;
    color: #aaa;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .tab:hover {
    background: #444;
    color: #fff;
  }
  
  .tab.active {
    background: #4CAF50;
    color: white;
    border-color: #4CAF50;
  }
  
  .close-btn {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: #ff5252;
    color: white;
    border: none;
    font-size: 20px;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .close-btn:hover {
    background: #ff1744;
    transform: scale(1.1);
  }
  
  .validation-error {
    margin: 20px;
    padding: 15px;
    background: #d32f2f;
    color: white;
    border-radius: 4px;
  }
  
  .validation-error h3 {
    margin: 0 0 10px 0;
  }
  
  .validation-error pre {
    margin: 0;
    font-family: 'Courier New', monospace;
    font-size: 14px;
    white-space: pre-wrap;
  }
  
  .debug-content {
    padding: 20px;
    min-height: 500px;
  }
  
  .state-view {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
  
  .state-column {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  
  .hands-column {
    display: flex;
    flex-direction: column;
  }
  
  .tools-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 20px;
  }
</style>