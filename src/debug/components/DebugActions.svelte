<script lang="ts">
  import type { StateTransition } from '../../game/types';
  
  interface Props {
    availableActions: StateTransition[];
    onAction: (transition: StateTransition) => void;
  }
  
  let { availableActions, onAction }: Props = $props();
  
  // Import the game state to get current player info
  import { gameState } from '../../stores/gameStore';
  
  const actionsByType = $derived(() => {
    const groups: Record<string, StateTransition[]> = {};
    
    availableActions.forEach(action => {
      let type = 'other';
      
      if (action.id === 'pass') type = 'bidding';
      else if (action.id.startsWith('bid-')) type = 'bidding';
      else if (action.id.startsWith('trump-')) type = 'trump';
      else if (action.id.startsWith('play-')) type = 'playing';
      else if (action.id === 'complete-trick') type = 'trick';
      else if (action.id === 'score-hand') type = 'scoring';
      else if (action.id === 'redeal') type = 'redeal';
      
      if (!groups[type]) groups[type] = [];
      groups[type].push(action);
    });
    
    return groups;
  });
  
  function getActionTypeLabel(type: string): string {
    switch (type) {
      case 'bidding': return 'Bids';
      case 'trump': return 'Trump Selection';
      case 'playing': return 'Play Domino';
      case 'trick': return 'Trick Complete';
      case 'scoring': return 'Scoring';
      case 'redeal': return 'Redeal';
      default: return 'Other';
    }
  }
  
  function getActionColor(type: string): string {
    switch (type) {
      case 'bidding': return '#007bff';
      case 'trump': return '#fd7e14';
      case 'playing': return '#28a745';
      case 'trick': return '#6f42c1';
      case 'scoring': return '#dc3545';
      case 'redeal': return '#6c757d';
      default: return '#495057';
    }
  }
  
  function getActionTestId(action: StateTransition): string {
    // The current player in the current game state is the one who can take these actions
    const currentPlayer = $gameState.currentPlayer;
    
    // Generate specific test IDs that match what tests expect
    if (action.id === 'pass') {
      // For bidding pass actions, create P0-PASS, P1-PASS etc
      return `bid-P${currentPlayer}-PASS`;
    } else if (action.id.startsWith('bid-')) {
      // For point bids and mark bids  
      const parts = action.id.split('-');
      if (parts.length >= 2) {
        if (parts.length === 3 && parts[2] === 'marks') {
          // Mark bids: bid-1-marks -> bid-P0-1M
          return `bid-P${currentPlayer}-${parts[1]}M`;
        } else {
          // Point bids: bid-30 -> bid-P0-30
          const points = parts[1];
          return `bid-P${currentPlayer}-${points}`;
        }
      }
    } else if (action.id.startsWith('trump-')) {
      // For trump selection, create set-trump-5s etc
      const suit = action.id.split('-')[1];
      // Map full suit names to abbreviated forms
      const suitMap: Record<string, string> = {
        'blanks': '0s',
        'ones': '1s', 
        'twos': '2s',
        'threes': '3s',
        'fours': '4s',
        'fives': '5s',
        'sixes': '6s',
        'doubles': 'Doubles'
      };
      const abbreviatedSuit = suitMap[suit] || suit;
      return `set-trump-${abbreviatedSuit}`;
    } else if (action.id === 'complete-trick') {
      // For completing tricks, append the trick number (which trick we're completing)
      // We have 'tricks.length' completed tricks, so we're completing trick number (tricks.length + 1)
      const trickNumber = $gameState.tricks.length + 1;
      return `complete-trick-${trickNumber}`;
    } else if (action.id === 'score-hand') {
      return 'score-hand';
    } else if (action.id === 'redeal') {
      return 'redeal';
    } else if (action.id.startsWith('play-')) {
      // For playing dominoes
      return action.id;
    }
      
    // Default fallback
    return action.id;
  }

  function getCombinedTestId(action: StateTransition): string {
    const specific = getActionTestId(action);
    const generic = getGenericTestId(action);
    
    // Combine both test IDs in the same attribute with space separation
    // This allows both exact matches and substring matches to work
    return `${specific} ${generic}`;
  }

  function getGenericTestId(action: StateTransition): string {
    // Add generic test IDs that the tests expect
    if (action.id === 'pass' || action.id.startsWith('bid-')) {
      return 'bid-button';
    } else if (action.id.startsWith('trump-')) {
      return 'trump-button';
    } else {
      return 'action-button';
    }
  }


  function getActionButtonLabel(action: StateTransition): string {
    const currentPlayer = $gameState.currentPlayer;
    
    // Format button labels to match test expectations
    if (action.id === 'pass') {
      return `P${currentPlayer}: Pass`;
    } else if (action.id.startsWith('bid-')) {
      const parts = action.id.split('-');
      if (parts.length === 3 && parts[2] === 'marks') {
        // Mark bids: bid-1-marks -> "P0: 1M"
        const marks = parseInt(parts[1]);
        return `P${currentPlayer}: ${marks}M`;
      } else if (parts.length === 2) {
        // Point bids: bid-30 -> "P0: 30"
        const points = parseInt(parts[1]);
        return `P${currentPlayer}: ${points}`;
      }
    } else if (action.id.startsWith('trump-')) {
      // For trump selection
      const suit = action.id.split('-')[1];
      const suitNames: Record<string, string> = {
        'blanks': 'Blanks',
        'ones': 'Ones', 
        'twos': 'Twos',
        'threes': 'Threes',
        'fours': 'Fours',
        'fives': 'Fives',
        'sixes': 'Sixes',
        'doubles': 'Doubles'
      };
      const suitName = suitNames[suit] || suit;
      return `Declare ${suitName} trump`;
    }
    
    // Default to original label for other actions
    return action.label;
  }
</script>

<div class="debug-actions">
  <div class="actions-header">
    <h3>Available Actions (<span data-testid="actions-count">{availableActions.length}</span>)</h3>
  </div>
  
  {#if availableActions.length === 0}
    <div class="no-actions">
      No actions available
    </div>
  {:else}
    <!-- Debug UI shows all actions with both specific and generic test IDs -->
    <div class="actions-simple">
      {#each availableActions as action, index}
        <button 
          class="action-btn"
          onclick={() => onAction(action)}
          title={`${action.id}: ${action.label}`}
          data-testid={getActionTestId(action)}
          data-generic-testid={getGenericTestId(action)}
          data-action-id={action.id}
        >
          {getActionButtonLabel(action)}
        </button>
      {/each}
    </div>
  {/if}
  
  <div class="actions-footer">
    <div class="action-count">
      {availableActions.length} possible transitions
    </div>
  </div>
</div>

<style>
  .debug-actions {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 12px;
    height: 100%;
    display: flex;
    flex-direction: column;
  }
  
  .actions-header h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: #212529;
    pointer-events: none;
  }
  
  .no-actions {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #6c757d;
    font-style: italic;
    font-size: 12px;
  }
  
  .actions-grid {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-top: 12px;
    overflow-y: auto;
  }
  
  .actions-simple {
    flex: 1;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 6px;
    margin-top: 12px;
    overflow-y: auto;
    position: relative;
    z-index: 10;
  }
  
  .action-group {
    border: 1px solid #e9ecef;
    border-radius: 3px;
    padding: 8px;
    background: #f8f9fa;
  }
  
  .group-header {
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .action-buttons {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 4px;
  }
  
  .action-btn {
    background: white;
    border: 1px solid;
    border-radius: 3px;
    padding: 6px;
    cursor: pointer;
    font-size: 10px;
    text-align: left;
    transition: none; /* Disable transitions for stability */
    display: flex;
    flex-direction: column;
    gap: 2px;
    position: relative;
    z-index: 20;
  }
  
  .action-btn:hover {
    background: #f8f9fa;
    /* Remove transform for stability */
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  
  .action-id {
    font-family: monospace;
    font-weight: 600;
    font-size: 9px;
    opacity: 0.8;
  }
  
  .action-label {
    font-size: 10px;
    line-height: 1.2;
  }
  
  .actions-footer {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #e9ecef;
    font-size: 10px;
    color: #6c757d;
    text-align: center;
    pointer-events: none;
  }
  
  @media (prefers-reduced-motion: reduce) {
    .action-btn {
      transition: none;
    }
    
    .action-btn:hover {
      /* Already no transform */
    }
  }
</style>