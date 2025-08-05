<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { gameState, actionHistory, stateValidationError, gameActions, initialState } from '../../stores/gameStore';
  import { quickplayState, quickplayActions } from '../../stores/quickplayStore';
  import StateTreeView from './StateTreeView.svelte';

  const dispatch = createEventDispatcher();

  let activeTab = 'state';
  let showDiff = false;
  let showTreeView = true;
  let showHistoricalTreeView = true;
  let previousState: any = null;
  let changedPaths = new Set<string>();

  // JSON stringify with indentation
  function prettyJson(obj: any): string {
    return JSON.stringify(obj, null, 2);
  }


  // Copy to clipboard
  function copyToClipboard(text: string) {
    navigator.clipboard.writeText(text);
  }

  // Generate shareable URL
  function copyShareableUrl() {
    const url = window.location.href;
    copyToClipboard(url);
  }

  // Time travel to a specific action
  function timeTravel(index: number) {
    // Reset to initial state
    gameActions.loadState($initialState);
    
    // Replay actions up to index
    const actionsToReplay = $actionHistory.slice(0, index + 1);
    for (const action of actionsToReplay) {
      gameActions.executeAction(action);
    }
  }

  // Find differences between two objects
  function findDifferences(obj1: any, obj2: any, path: string = 'root'): Set<string> {
    const differences = new Set<string>();
    
    if (obj1 === obj2) return differences;
    
    if (typeof obj1 !== typeof obj2) {
      differences.add(path);
      return differences;
    }
    
    if (typeof obj1 !== 'object' || obj1 === null || obj2 === null) {
      if (obj1 !== obj2) {
        differences.add(path);
      }
      return differences;
    }
    
    // For arrays
    if (Array.isArray(obj1) && Array.isArray(obj2)) {
      const maxLength = Math.max(obj1.length, obj2.length);
      for (let i = 0; i < maxLength; i++) {
        const childDiffs = findDifferences(obj1[i], obj2[i], `${path}[${i}]`);
        childDiffs.forEach(diff => differences.add(diff));
      }
      return differences;
    }
    
    // For objects
    const allKeys = new Set([...Object.keys(obj1), ...Object.keys(obj2)]);
    for (const key of allKeys) {
      const childDiffs = findDifferences(obj1[key], obj2[key], `${path}.${key}`);
      childDiffs.forEach(diff => differences.add(diff));
    }
    
    return differences;
  }

  // Track state changes
  $: {
    if (showDiff && previousState) {
      changedPaths = findDifferences(previousState, $gameState);
    } else {
      changedPaths = new Set();
    }
    previousState = JSON.parse(JSON.stringify($gameState));
  }
</script>

<div class="debug-backdrop" role="presentation" on:click={() => dispatch('close')} on:keydown={() => {}}>
  <div class="debug-panel" role="dialog" tabindex="-1" on:click|stopPropagation on:keydown={() => {}}>
    <div class="panel-header">
      <h2>Debug Panel</h2>
      <button class="close-button" on:click={() => dispatch('close')}>×</button>
    </div>

    <div class="panel-tabs">
      <button 
        class="tab" 
        class:active={activeTab === 'state'}
        on:click={() => activeTab = 'state'}
      >
        Game State
      </button>
      <button 
        class="tab" 
        class:active={activeTab === 'history'}
        on:click={() => activeTab = 'history'}
      >
        History
      </button>
      <button 
        class="tab" 
        class:active={activeTab === 'quickplay'}
        on:click={() => activeTab = 'quickplay'}
      >
        QuickPlay
      </button>
      <button 
        class="tab" 
        class:active={activeTab === 'historical'}
        on:click={() => activeTab = 'historical'}
      >
        Historical State
      </button>
    </div>

    <div class="panel-content">
      {#if activeTab === 'state'}
        <div class="state-tab">
          <div class="state-controls">
            <label class="toggle-label">
              <input 
                type="checkbox" 
                bind:checked={showTreeView}
              />
              Tree View
            </label>
            <label class="toggle-label">
              <input 
                type="checkbox" 
                bind:checked={showDiff}
              />
              Diff Mode
            </label>
            <button 
              class="control-button"
              on:click={() => copyToClipboard(prettyJson($gameState))}
            >
              Copy State
            </button>
            <button 
              class="control-button"
              on:click={copyShareableUrl}
            >
              Copy URL
            </button>
          </div>
          
          <div class="state-display">
            {#if showTreeView}
              <StateTreeView 
                data={$gameState} 
                searchQuery=""
                changedPaths={showDiff ? changedPaths : new Set()}
              />
            {:else}
              <pre>{prettyJson($gameState)}</pre>
            {/if}
          </div>
        </div>
      {/if}

      {#if activeTab === 'history'}
        <div class="history-tab">
          <div class="history-header">
            <h3>Action History ({$actionHistory.length})</h3>
            <div class="history-controls">
              <button 
                class="control-button"
                on:click={gameActions.undo}
                disabled={$actionHistory.length === 0}
              >
                Undo Last
              </button>
              <button 
                class="control-button"
                on:click={gameActions.resetGame}
              >
                Reset Game
              </button>
            </div>
          </div>
          <div class="action-history">
            {#if $actionHistory.length === 0}
              <div class="empty-history">
                No actions taken yet. Start playing to see history.
              </div>
            {:else}
              {#each $actionHistory as action, index}
                <div class="history-item">
                  <span class="history-index">#{index + 1}</span>
                  <span class="history-label">{action.label}</span>
                  <button 
                    class="time-travel-button"
                    on:click={() => timeTravel(index)}
                    title="Time travel to this point"
                  >
                    ⏪
                  </button>
                </div>
              {/each}
            {/if}
          </div>
        </div>
      {/if}

      {#if activeTab === 'quickplay'}
        <div class="quickplay-tab">
          <div class="quickplay-controls">
            <label>
              <input 
                type="checkbox" 
                bind:checked={$quickplayState.enabled}
                on:change={(e) => quickplayActions.toggle()}
              />
              QuickPlay Active
            </label>
            
            <label>
              Speed:
              <select 
                bind:value={$quickplayState.speed}
                on:change={(e) => quickplayActions.setSpeed(e.currentTarget.value as any)}
              >
                <option value="instant">Instant</option>
                <option value="fast">Fast</option>
                <option value="normal">Normal</option>
                <option value="slow">Slow</option>
              </select>
            </label>

            <button 
              class="control-button"
              on:click={quickplayActions.step}
              disabled={$quickplayState.enabled}
            >
              Step
            </button>

            <button 
              class="control-button"
              on:click={quickplayActions.playToEndOfHand}
            >
              Play to End of Hand
            </button>

            <button 
              class="control-button"
              on:click={quickplayActions.playToEndOfGame}
            >
              Play to End of Game
            </button>
          </div>

          <div class="quickplay-status">
            <h4>Status</h4>
            <div class="status-grid">
              <span>Active:</span> <span>{$quickplayState.enabled}</span>
              <span>Speed:</span> <span>{$quickplayState.speed}</span>
              <span>Phase:</span> <span>{$gameState.phase}</span>
              <span>Current Player:</span> <span>P{$gameState.currentPlayer}</span>
            </div>
          </div>
        </div>
      {/if}

      {#if activeTab === 'historical'}
        <div class="historical-tab">
          <div class="historical-controls">
            <h3>Event Sourcing State</h3>
            <label class="toggle-label">
              <input 
                type="checkbox" 
                bind:checked={showHistoricalTreeView}
              />
              Tree View
            </label>
            <button 
              class="control-button"
              on:click={() => {
                const historicalData = {
                  initialState: $initialState,
                  actions: $actionHistory
                };
                copyToClipboard(prettyJson(historicalData));
              }}
            >
              Copy Historical JSON
            </button>
          </div>
          
          <div class="historical-display">
            {#if showHistoricalTreeView}
              <StateTreeView 
                data={{
                  initialState: $initialState,
                  actions: $actionHistory
                }}
                searchQuery=""
                changedPaths={new Set()}
              />
            {:else}
              <div class="historical-section">
                <h4>Initial State</h4>
                <pre>{prettyJson($initialState)}</pre>
              </div>
              
              <div class="historical-section">
                <h4>Actions ({$actionHistory.length})</h4>
                <pre>{prettyJson($actionHistory)}</pre>
              </div>
            {/if}
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .debug-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .debug-panel {
    background-color: white;
    width: 95%;
    max-width: 600px;
    height: 90%;
    max-height: 90vh;
    border-radius: 12px 12px 0 0;
    display: flex;
    flex-direction: column;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    margin-top: auto;
  }

  @media (min-width: 768px) {
    .debug-panel {
      width: 80%;
      max-width: 1200px;
      height: 80%;
      max-height: 800px;
      border-radius: 12px;
      margin-top: 0;
    }
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    border-bottom: 1px solid #e5e7eb;
  }

  .panel-header h2 {
    margin: 0;
    font-size: 20px;
    color: #002868;
  }

  .close-button {
    background: none;
    border: none;
    font-size: 28px;
    cursor: pointer;
    color: #6b7280;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
  }

  .close-button:hover {
    background-color: #f3f4f6;
  }

  .panel-tabs {
    display: flex;
    gap: 2px;
    padding: 0 12px;
    background-color: #f9fafb;
    border-bottom: 1px solid #e5e7eb;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }

  .tab {
    padding: 10px 16px;
    background: none;
    border: none;
    cursor: pointer;
    font-weight: 500;
    color: #6b7280;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
    font-size: 14px;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .tab:hover {
    color: #374151;
  }

  .tab.active {
    color: #002868;
    border-bottom-color: #002868;
  }

  .panel-content {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  /* State Tab */
  .state-tab {
    padding: 20px;
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .state-controls {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
  }

  .toggle-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 14px;
    color: #374151;
    cursor: pointer;
  }

  .toggle-label input {
    cursor: pointer;
  }

  .control-button {
    padding: 8px 16px;
    background-color: #002868;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
  }

  .control-button:hover {
    background-color: #001a4d;
  }

  .control-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .state-display {
    flex: 1;
    overflow: auto;
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 16px;
  }

  .state-display pre {
    margin: 0;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 12px;
    line-height: 1.5;
  }

  /* History Tab */
  .history-tab {
    padding: 20px;
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .history-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .history-header h3 {
    margin: 0;
    font-size: 18px;
    color: #374151;
  }

  .history-controls {
    display: flex;
    gap: 8px;
  }

  .action-history {
    flex: 1;
    overflow-y: auto;
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px;
  }

  .empty-history {
    text-align: center;
    padding: 40px;
    color: #9ca3af;
    font-style: italic;
  }

  .history-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    margin-bottom: 8px;
    transition: background-color 0.2s;
  }

  .history-item:hover {
    background-color: #f9fafb;
  }

  .history-index {
    font-family: monospace;
    font-size: 12px;
    color: #6b7280;
    min-width: 40px;
  }

  .history-label {
    flex: 1;
    font-size: 14px;
  }

  .time-travel-button {
    padding: 4px 8px;
    background-color: #3b82f6;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
  }

  .time-travel-button:hover {
    background-color: #2563eb;
  }

  /* QuickPlay Tab */
  .quickplay-tab {
    padding: 20px;
  }

  .quickplay-controls {
    display: flex;
    gap: 16px;
    align-items: center;
    margin-bottom: 24px;
    flex-wrap: wrap;
  }

  .quickplay-controls label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
  }

  .quickplay-controls select {
    padding: 4px 8px;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    font-size: 14px;
  }

  .quickplay-status {
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 16px;
  }

  .quickplay-status h4 {
    margin: 0 0 12px 0;
    font-size: 16px;
    color: #374151;
  }

  .status-grid {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 8px 16px;
    font-size: 14px;
  }

  .status-grid span:nth-child(odd) {
    font-weight: 600;
    color: #6b7280;
  }

  /* Historical State Tab */
  .historical-tab {
    padding: 20px;
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .historical-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }

  .historical-controls h3 {
    margin: 0;
    font-size: 18px;
    color: #374151;
    margin-right: auto;
  }

  .historical-display {
    flex: 1;
    overflow: auto;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .historical-section {
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 16px;
  }

  .historical-section h4 {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
    color: #6b7280;
  }

  .historical-section pre {
    margin: 0;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 12px;
    line-height: 1.5;
    overflow: auto;
    max-height: 300px;
  }
</style>