<script lang="ts">
  import type { StateTransition } from '../../game/types';
  import { valuesToGlyph, supportsDominoGlyphs } from '../../game/core/domino-glyphs';
  
  interface Props {
    availableActions: StateTransition[];
    onAction: (transition: StateTransition) => void;
  }
  
  let { availableActions, onAction }: Props = $props();
  
  // Import the game state to get current player info
  import { gameState } from '../../stores/gameStore';
  
  const useGlyphs = supportsDominoGlyphs();
  
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
      const trickNumber = $gameState.tricks.length + 1;
      return `complete-trick-${trickNumber}`;
    } else if (action.id === 'score-hand') {
      return 'score-hand';
    } else if (action.id === 'redeal') {
      return 'redeal';
    }
    
    // Default fallback
    return action.id;
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
  
  function formatActionLabel(action: StateTransition): string {
    // For play actions, extract domino values and show glyph
    if (action.id.startsWith('play-') && useGlyphs) {
      const match = action.label.match(/Play (\d)-(\d)/);
      if (match) {
        const high = parseInt(match[1]);
        const low = parseInt(match[2]);
        return valuesToGlyph(high, low);
      }
    }
    return action.label;
  }
  
  function shouldShowGlyph(action: StateTransition): boolean {
    return action.id.startsWith('play-') && useGlyphs;
  }
</script>

<div class="debug-actions">
  <div class="section-header">
    <h3>Available Actions</h3>
    <div class="action-count">
      <span data-testid="actions-count">{availableActions.length}</span>
    </div>
  </div>
  
  {#if availableActions.length === 0}
    <div class="no-actions">
      No actions available
    </div>
  {:else}
    <div class="actions-grid">
      {#each availableActions as action, index}
        <button 
          class="action-compact"
          class:bid-action={action.id === 'pass' || action.id.startsWith('bid-')}
          class:trump-action={action.id.startsWith('trump-')}
          class:play-action={action.id.startsWith('play-')}
          class:trick-action={action.id === 'complete-trick'}
          class:score-action={action.id === 'score-hand'}
          onclick={() => onAction(action)}
          title={`${action.id}: ${action.label}`}
          data-testid={getActionTestId(action)}
          data-generic-testid={getGenericTestId(action)}
          data-action-id={action.id}
        >
          {#if shouldShowGlyph(action)}
            <div class="action-glyph">{formatActionLabel(action)}</div>
          {:else}
            <div class="action-id">{action.id}</div>
            <div class="action-label">{action.label}</div>
          {/if}
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  .debug-actions {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 8px;
    margin: 4px 0;
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }

  .section-header h3 {
    margin: 0;
    font-size: 12px;
    color: #495057;
    font-weight: 600;
  }

  .action-count {
    background: #6c757d;
    color: white;
    padding: 1px 4px;
    border-radius: 2px;
    font-size: 10px;
    font-weight: 500;
  }

  .no-actions {
    text-align: center;
    color: #6c757d;
    font-size: 11px;
    padding: 8px;
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .actions-grid {
    display: flex;
    flex-direction: column;
    gap: 3px;
    flex: 1;
    overflow-y: auto;
  }

  .action-compact {
    background: white;
    border: 1px solid #e9ecef;
    border-radius: 3px;
    padding: 4px 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    min-height: 28px;
    transition: background 0.1s ease;
  }

  .action-compact:hover {
    background: #f8f9fa;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
  }

  .action-compact.bid-action {
    border-left: 3px solid #007bff;
  }

  .action-compact.trump-action {
    border-left: 3px solid #fd7e14;
  }

  .action-compact.play-action {
    border-left: 3px solid #28a745;
    min-height: 48px;
    align-items: center;
    justify-content: center;
    padding: 8px;
  }

  .action-compact.trick-action {
    border-left: 3px solid #6f42c1;
  }

  .action-compact.score-action {
    border-left: 3px solid #dc3545;
  }

  .action-id {
    font-family: monospace;
    font-size: 9px;
    font-weight: 600;
    color: #6c757d;
    min-width: 80px;
  }
  
  .action-label {
    font-size: 10px;
    line-height: 1.2;
    color: #212529;
    text-align: right;
    flex: 1;
  }
  
  .action-glyph {
    font-family: 'Noto Sans Symbols', 'Noto Sans Symbols 2', 'Segoe UI Symbol', 'Segoe UI Emoji', 'Noto Color Emoji', 'Apple Color Emoji', 'Symbola', sans-serif;
    font-size: clamp(32px, 4vw, 48px);
    line-height: 1;
    text-align: center;
    width: 100%;
    font-variant-emoji: text;
  }

  @media (prefers-reduced-motion: reduce) {
    .action-compact {
      transition: none;
    }
  }
</style>