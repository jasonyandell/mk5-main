<script lang="ts">
  import type { Bid, StateTransition } from '../../game/types';
  import { BID_TYPES } from '../../game/constants';
  
  interface Props {
    availableActions: StateTransition[];
    currentBid: Bid | null;
    bids: Bid[];
    onAction: (transition: StateTransition) => void;
  }
  
  let { availableActions, currentBid, bids, onAction }: Props = $props();
  
  function getBidLabel(bid: Bid): string {
    switch (bid.type) {
      case BID_TYPES.PASS:
        return 'Pass';
      case BID_TYPES.POINTS:
        return `${bid.value} points`;
      case BID_TYPES.MARKS:
        return `${bid.value} mark${bid.value !== 1 ? 's' : ''}`;
      case BID_TYPES.NELLO:
        return `Nello ${bid.value}`;
      case BID_TYPES.SPLASH:
        return `Splash ${bid.value}`;
      case BID_TYPES.PLUNGE:
        return `Plunge ${bid.value}`;
      default:
        return 'Unknown';
    }
  }
  
  function getPlayerName(playerId: number): string {
    return `Player ${playerId + 1}`;
  }
  
  const biddingActions = $derived(availableActions.filter(action => 
    action.id.startsWith('bid-') || action.id === 'pass'
  ));
  
  const trumpActions = $derived(availableActions.filter(action => 
    action.id.startsWith('trump-')
  ));
</script>

<div class="bidding-panel">
  <h2>Bidding</h2>
  
  {#if currentBid}
    <div class="current-bid">
      <h3>Current High Bid</h3>
      <div class="bid-info">
        <span class="bid-value">{getBidLabel(currentBid)}</span>
        <span class="bidder">by {getPlayerName(currentBid.player)}</span>
      </div>
    </div>
  {/if}
  
  <div class="bid-history">
    <h3>Bid History</h3>
    <div class="bids-list">
      {#each bids as bid}
        <div class="bid-entry">
          <span class="player">{getPlayerName(bid.player)}:</span>
          <span class="bid">{getBidLabel(bid)}</span>
        </div>
      {/each}
      {#if bids.length === 0}
        <div class="no-bids">No bids yet</div>
      {/if}
    </div>
  </div>
  
  {#if trumpActions.length > 0}
    <div class="trump-selection">
      <h3>Select Trump Suit</h3>
      <div class="trump-options">
        {#each trumpActions as action}
          <button 
            class="trump-btn"
            onclick={() => onAction(action)}
          >
            {action.label}
          </button>
        {/each}
      </div>
    </div>
  {:else if biddingActions.length > 0}
    <div class="bid-actions">
      <h3>Your Bid Options</h3>
      <div class="bid-buttons">
        {#each biddingActions as action}
          <button 
            class="bid-btn"
            class:pass-btn={action.id === 'pass'}
            onclick={() => onAction(action)}
          >
            {action.label}
          </button>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .bidding-panel {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    contain: layout style;
  }
  
  .bidding-panel h2 {
    margin: 0 0 16px 0;
    color: #333;
  }
  
  .bidding-panel h3 {
    margin: 16px 0 8px 0;
    font-size: 14px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .current-bid {
    background: #e3f2fd;
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 16px;
  }
  
  .bid-info {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  
  .bid-value {
    font-weight: bold;
    color: #1976d2;
  }
  
  .bidder {
    color: #666;
    font-size: 14px;
  }
  
  .bids-list {
    max-height: 120px;
    overflow-y: auto;
  }
  
  .bid-entry {
    display: flex;
    gap: 8px;
    padding: 4px 0;
    font-size: 14px;
  }
  
  .bid-entry .player {
    font-weight: 500;
    min-width: 80px;
  }
  
  .no-bids {
    color: #999;
    font-style: italic;
    padding: 8px 0;
  }
  
  .bid-buttons, .trump-options {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  
  .bid-btn, .trump-btn {
    padding: 8px 16px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: white;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s ease;
  }
  
  .bid-btn:hover, .trump-btn:hover {
    background: #f5f5f5;
    border-color: #999;
  }
  
  .bid-btn.pass-btn {
    background: #ffebee;
    border-color: #f44336;
    color: #d32f2f;
  }
  
  .bid-btn.pass-btn:hover {
    background: #ffcdd2;
  }
  
  .trump-btn {
    background: #e8f5e8;
    border-color: #4caf50;
    color: #2e7d32;
  }
  
  .trump-btn:hover {
    background: #c8e6c9;
  }
  
  @media (prefers-reduced-motion: reduce) {
    .bid-btn, .trump-btn {
      transition: none;
    }
  }
</style>