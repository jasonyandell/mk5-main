<script lang="ts">
  import type { StateTransition } from '../game/types';
  import { gameActions } from '../stores/gameStore';
  
  export let actions: StateTransition[];
  
  function getActionCategory(id: string): string {
    if (id.startsWith('bid-')) return 'bid';
    if (id.startsWith('pass-')) return 'bid';
    if (id.startsWith('set-trump-')) return 'trump';
    if (id.startsWith('play-domino-')) return 'play';
    if (id === 'complete-trick') return 'system';
    if (id === 'score-hand') return 'system';
    if (id === 'redeal') return 'system';
    return 'other';
  }
  
  function getCategoryColor(category: string): string {
    const colors: Record<string, string> = {
      'bid': '#2196F3',
      'trump': '#FF9800',
      'play': '#4CAF50',
      'system': '#9C27B0',
      'other': '#607D8B'
    };
    return colors[category] || '#666';
  }
  
  function getCategoryIcon(category: string): string {
    const icons: Record<string, string> = {
      'bid': '💰',
      'trump': '👑',
      'play': '🎯',
      'system': '⚙️',
      'other': '❓'
    };
    return icons[category] || '•';
  }
  
  function executeAction(action: StateTransition) {
    gameActions.executeAction(action);
  }
  
  // Group actions by category
  $: groupedActions = actions.reduce((groups, action) => {
    const category = getActionCategory(action.id);
    if (!groups[category]) groups[category] = [];
    groups[category].push(action);
    return groups;
  }, {} as Record<string, StateTransition[]>);
  
  const categoryOrder = ['bid', 'trump', 'play', 'system', 'other'];
</script>

<div class="actions-panel">
  <h3>Available Actions ({actions.length})</h3>
  
  {#if actions.length === 0}
    <div class="no-actions">No actions available in current state</div>
  {:else}
    <div class="actions-grid">
      {#each categoryOrder as category}
        {#if groupedActions[category]}
          <div class="action-category">
            <div class="category-header" style="color: {getCategoryColor(category)}">
              <span class="category-icon">{getCategoryIcon(category)}</span>
              <span class="category-name">{category.toUpperCase()}</span>
              <span class="category-count">{groupedActions[category].length}</span>
            </div>
            
            <div class="action-buttons">
              {#each groupedActions[category] as action}
                <button
                  class="action-button"
                  on:click={() => executeAction(action)}
                  data-testid={action.id}
                  title={action.id}
                  style="border-color: {getCategoryColor(category)}"
                >
                  {action.label}
                </button>
              {/each}
            </div>
          </div>
        {/if}
      {/each}
    </div>
  {/if}
  
  <div class="action-help">
    <h4>Action Format</h4>
    <div class="format-examples">
      <div class="format-item">
        <code>bid-P0-31</code>
        <span>Player 0 bids 31 points</span>
      </div>
      <div class="format-item">
        <code>set-trump-5s</code>
        <span>Set trump to fives</span>
      </div>
      <div class="format-item">
        <code>play-domino-P2-5-3</code>
        <span>Player 2 plays 5-3 domino</span>
      </div>
    </div>
  </div>
</div>

<style>
  .actions-panel {
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 20px;
  }
  
  .actions-panel h3 {
    margin: 0 0 20px 0;
    color: #FF9800;
    font-size: 18px;
  }
  
  .no-actions {
    color: #666;
    font-style: italic;
    text-align: center;
    padding: 40px;
  }
  
  .actions-grid {
    display: flex;
    flex-direction: column;
    gap: 25px;
  }
  
  .action-category {
    background: #333;
    border-radius: 6px;
    padding: 15px;
  }
  
  .category-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 15px;
    font-weight: bold;
    font-size: 14px;
  }
  
  .category-icon {
    font-size: 18px;
  }
  
  .category-name {
    flex: 1;
  }
  
  .category-count {
    background: rgba(255, 255, 255, 0.1);
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
  }
  
  .action-buttons {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 10px;
  }
  
  .action-button {
    padding: 10px 15px;
    background: #3a3a3a;
    border: 2px solid;
    border-radius: 6px;
    color: #fff;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 14px;
    text-align: left;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  
  .action-button:hover {
    background: #444;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
  }
  
  .action-button:active {
    transform: translateY(0);
  }
  
  .action-help {
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid #444;
  }
  
  .action-help h4 {
    margin: 0 0 15px 0;
    color: #888;
    font-size: 14px;
  }
  
  .format-examples {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  
  .format-item {
    display: flex;
    align-items: center;
    gap: 15px;
    font-size: 13px;
  }
  
  .format-item code {
    background: #444;
    padding: 4px 8px;
    border-radius: 4px;
    font-family: monospace;
    color: #4CAF50;
    min-width: 180px;
  }
  
  .format-item span {
    color: #999;
  }
</style>