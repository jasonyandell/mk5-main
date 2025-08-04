<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { quickplayState, quickplayActions, startQuickplay, stopQuickplay } from '../../stores/quickplayStore';
  import type { GameState } from '../../game/types';
  
  export let gameState: GameState;
  
  const speeds = ['instant', 'fast', 'normal', 'slow'] as const;
  const playerNames = ['Player 1', 'Player 2', 'Player 3', 'Player 4'];
  
  // Ensure quickplay is properly initialized when component mounts
  onMount(() => {
    if ($quickplayState.enabled) {
      startQuickplay();
    }
  });
  
  // Clean up on unmount
  onDestroy(() => {
    stopQuickplay();
  });
</script>

<div class="quickplay-panel">
  <div class="panel-header">
    <h3>AI Quickplay</h3>
    <div class="quickplay-controls">
      {#if $quickplayState.enabled}
        <button 
          class="control-btn stop-btn" 
          onclick={quickplayActions.toggle}
          data-testid="quickplay-stop"
        >
          Stop
        </button>
        {#if $quickplayState.isPaused}
          <button 
            class="control-btn resume-btn" 
            onclick={quickplayActions.resume}
            data-testid="quickplay-resume"
          >
            Resume
          </button>
        {:else}
          <button 
            class="control-btn pause-btn" 
            onclick={quickplayActions.pause}
            data-testid="quickplay-pause"
          >
            Pause
          </button>
        {/if}
      {:else}
        <button 
          class="control-btn run-btn" 
          onclick={quickplayActions.toggle}
          data-testid="quickplay-run"
        >
          Run
        </button>
      {/if}
      <button 
        class="control-btn step-btn" 
        onclick={quickplayActions.step}
        disabled={!$quickplayState.aiPlayers.has(gameState.currentPlayer)}
        data-testid="quickplay-step"
      >
        Step
      </button>
    </div>
  </div>
  
  <div class="quickplay-settings">
    <div class="setting-group">
      <label for="quickplay-speed-select">Speed:</label>
      <select
        id="quickplay-speed-select" 
        value={$quickplayState.speed} 
        onchange={(e) => quickplayActions.setSpeed(e.currentTarget.value as any)}
        data-testid="quickplay-speed"
      >
        {#each speeds as speed}
          <option value={speed}>{speed}</option>
        {/each}
      </select>
    </div>
    
    <div class="setting-group">
      <div>AI Players:</div>
      <div class="player-toggles">
        {#each [0, 1, 2, 3] as playerId}
          <label class="player-toggle">
            <input 
              type="checkbox" 
              checked={$quickplayState.aiPlayers.has(playerId)}
              onchange={() => quickplayActions.togglePlayer(playerId)}
              data-testid={`ai-player-${playerId}`}
            />
            <span class="player-name team-{gameState.players[playerId].teamId}">
              {playerNames[playerId]}
            </span>
          </label>
        {/each}
      </div>
    </div>
  </div>
  
  <div class="quickplay-status">
    {#if $quickplayState.enabled}
      <div class="status-indicator active">
        {#if $quickplayState.isPaused}
          <span class="status-icon paused">⏸</span>
          Paused
        {:else}
          <span class="status-icon running">▶</span>
          Running ({$quickplayState.speed})
        {/if}
      </div>
    {:else}
      <div class="status-indicator inactive">
        <span class="status-icon">⏹</span>
        Stopped
      </div>
    {/if}
    
    <div class="current-turn">
      Current: {playerNames[gameState.currentPlayer]}
      {#if $quickplayState.aiPlayers.has(gameState.currentPlayer)}
        <span class="ai-badge">AI</span>
      {/if}
    </div>
  </div>
</div>

<style>
  .quickplay-panel {
    background: white;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    margin-bottom: 8px;
    flex-shrink: 0;
  }
  
  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: #f8f9fa;
    border-bottom: 1px solid #dee2e6;
  }
  
  .panel-header h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: #495057;
  }
  
  .quickplay-controls {
    display: flex;
    gap: 8px;
  }
  
  .control-btn {
    background: #6c757d;
    color: white;
    border: none;
    padding: 4px 12px;
    border-radius: 3px;
    font-size: 12px;
    cursor: pointer;
    transition: background 0.1s ease;
  }
  
  .control-btn:hover:not(:disabled) {
    background: #5a6268;
  }
  
  .control-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .run-btn {
    background: #28a745;
  }
  
  .run-btn:hover {
    background: #218838;
  }
  
  .stop-btn {
    background: #dc3545;
  }
  
  .stop-btn:hover {
    background: #c82333;
  }
  
  .pause-btn {
    background: #ffc107;
    color: #212529;
  }
  
  .pause-btn:hover {
    background: #e0a800;
  }
  
  .resume-btn {
    background: #17a2b8;
  }
  
  .resume-btn:hover {
    background: #138496;
  }
  
  .step-btn {
    background: #6f42c1;
  }
  
  .step-btn:hover:not(:disabled) {
    background: #5a32a3;
  }
  
  .quickplay-settings {
    padding: 16px;
    border-bottom: 1px solid #dee2e6;
  }
  
  .setting-group {
    margin-bottom: 12px;
  }
  
  .setting-group:last-child {
    margin-bottom: 0;
  }
  
  .setting-group label {
    display: block;
    font-size: 12px;
    font-weight: 600;
    color: #495057;
    margin-bottom: 4px;
  }
  
  .setting-group select {
    width: 100%;
    padding: 4px 8px;
    border: 1px solid #ced4da;
    border-radius: 3px;
    font-size: 12px;
    background: white;
  }
  
  .player-toggles {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  
  .player-toggle {
    display: flex;
    align-items: center;
    font-size: 12px;
    cursor: pointer;
  }
  
  .player-toggle input {
    margin-right: 6px;
  }
  
  .player-name {
    font-weight: 500;
  }
  
  .team-0 {
    color: #007bff;
  }
  
  .team-1 {
    color: #dc3545;
  }
  
  .quickplay-status {
    padding: 12px 16px;
    background: #f8f9fa;
    font-size: 12px;
  }
  
  .status-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
    font-weight: 600;
  }
  
  .status-indicator.active {
    color: #28a745;
  }
  
  .status-indicator.inactive {
    color: #6c757d;
  }
  
  .status-icon {
    font-size: 14px;
  }
  
  .status-icon.running {
    animation: pulse 1s infinite;
  }
  
  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
  }
  
  .current-turn {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #495057;
  }
  
  .ai-badge {
    background: #6f42c1;
    color: white;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
</style>