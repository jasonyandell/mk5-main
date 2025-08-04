<script lang="ts">
  import type { StateTransition } from '../game/types';
  import { gameActions } from '../stores/gameStore';
  
  export let history: StateTransition[];
  
  let selectedEventIndex: number | null = null;
  
  function getActionIcon(id: string): string {
    if (id.startsWith('bid-')) return '💰';
    if (id.startsWith('pass-')) return '🚫';
    if (id.startsWith('set-trump-')) return '👑';
    if (id.startsWith('play-domino-')) return '🎯';
    if (id === 'complete-trick') return '✓';
    if (id === 'score-hand') return '📊';
    if (id === 'redeal') return '🔄';
    return '•';
  }
  
  function getActionPlayer(id: string): string | null {
    const match = id.match(/P(\d)/);
    return match ? `P${match[1]}` : null;
  }
  
  function undoLastAction() {
    gameActions.undo();
  }
  
  function jumpToEvent(index: number) {
    gameActions.jumpToEvent(index);
    selectedEventIndex = index;
  }
  
  function clearHistory() {
    if (confirm('Are you sure you want to clear all history and reset to initial state?')) {
      gameActions.reset();
      selectedEventIndex = null;
    }
  }
</script>

<div class="replay-panel">
  <div class="replay-header">
    <h3>Action History ({history.length})</h3>
    <div class="replay-controls">
      <button 
        class="control-btn undo"
        on:click={undoLastAction}
        disabled={history.length === 0}
        data-testid="undo-button"
        title="Undo last action"
      >
        ↶ Undo
      </button>
      <button 
        class="control-btn clear"
        on:click={clearHistory}
        disabled={history.length === 0}
        data-testid="clear-history"
        title="Clear all history"
      >
        🗑️ Clear
      </button>
    </div>
  </div>
  
  {#if history.length === 0}
    <div class="no-history">No actions taken yet</div>
  {:else}
    <div class="history-timeline">
      <div class="timeline-header">
        <span>Event</span>
        <span>Player</span>
        <span>Action</span>
        <span>Description</span>
      </div>
      
      <div class="timeline-events">
        {#each history as event, index}
          <div 
            class="timeline-event"
            class:selected={selectedEventIndex === index}
            on:click={() => jumpToEvent(index)}
            data-testid="event-{index}"
          >
            <span class="event-number">#{index + 1}</span>
            <span class="event-player">{getActionPlayer(event.id) || '-'}</span>
            <span class="event-icon">{getActionIcon(event.id)}</span>
            <span class="event-label">{event.label}</span>
            <span class="event-id" title={event.id}>{event.id}</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}
  
  <div class="replay-info">
    <h4>Time Travel</h4>
    <p>Click any event to restore the game to that state. Use Undo to revert the last action.</p>
    {#if selectedEventIndex !== null}
      <p class="current-position">Currently at event #{selectedEventIndex + 1}</p>
    {/if}
  </div>
</div>

<style>
  .replay-panel {
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    height: 100%;
  }
  
  .replay-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }
  
  .replay-header h3 {
    margin: 0;
    color: #9C27B0;
    font-size: 18px;
  }
  
  .replay-controls {
    display: flex;
    gap: 10px;
  }
  
  .control-btn {
    padding: 8px 16px;
    border-radius: 6px;
    border: none;
    color: white;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 5px;
  }
  
  .control-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .control-btn.undo {
    background: #2196F3;
  }
  
  .control-btn.undo:hover:not(:disabled) {
    background: #1976D2;
  }
  
  .control-btn.clear {
    background: #F44336;
  }
  
  .control-btn.clear:hover:not(:disabled) {
    background: #D32F2F;
  }
  
  .no-history {
    color: #666;
    font-style: italic;
    text-align: center;
    padding: 40px;
  }
  
  .history-timeline {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: #333;
    border-radius: 6px;
    overflow: hidden;
  }
  
  .timeline-header {
    display: grid;
    grid-template-columns: 60px 60px 40px 1fr auto;
    gap: 10px;
    padding: 10px 15px;
    background: #3a3a3a;
    font-size: 12px;
    color: #888;
    text-transform: uppercase;
    font-weight: bold;
  }
  
  .timeline-events {
    flex: 1;
    overflow-y: auto;
  }
  
  .timeline-event {
    display: grid;
    grid-template-columns: 60px 60px 40px 1fr auto;
    gap: 10px;
    padding: 12px 15px;
    border-bottom: 1px solid #444;
    cursor: pointer;
    transition: all 0.2s;
    align-items: center;
  }
  
  .timeline-event:hover {
    background: #3a3a3a;
  }
  
  .timeline-event.selected {
    background: #1B5E20;
    border-left: 4px solid #4CAF50;
    padding-left: 11px;
  }
  
  .event-number {
    font-family: monospace;
    color: #666;
    font-size: 12px;
  }
  
  .event-player {
    font-family: monospace;
    color: #4CAF50;
    font-weight: bold;
  }
  
  .event-icon {
    font-size: 18px;
    text-align: center;
  }
  
  .event-label {
    color: #fff;
  }
  
  .event-id {
    font-family: monospace;
    font-size: 11px;
    color: #666;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 150px;
  }
  
  .replay-info {
    margin-top: 20px;
    padding: 15px;
    background: #333;
    border-radius: 6px;
  }
  
  .replay-info h4 {
    margin: 0 0 10px 0;
    color: #888;
    font-size: 14px;
  }
  
  .replay-info p {
    margin: 5px 0;
    font-size: 13px;
    color: #aaa;
  }
  
  .current-position {
    color: #4CAF50 !important;
    font-weight: bold;
  }
  
  ::-webkit-scrollbar {
    width: 8px;
  }
  
  ::-webkit-scrollbar-track {
    background: #2a2a2a;
  }
  
  ::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 4px;
  }
  
  ::-webkit-scrollbar-thumb:hover {
    background: #666;
  }
</style>