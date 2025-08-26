<script lang="ts">
  import type { GamePhase, TrumpSelection } from '../../game/types';
  
  interface Props {
    phase: GamePhase;
    currentPlayer: number;
    trump: TrumpSelection;
    trickNumber?: number;
    totalTricks?: number;
    bidWinner?: number;
    currentBid?: { type: string; value?: number };
    trickLeader?: number | null;
    ledSuit?: number;
    ledSuitDisplay?: string | null;
  }
  
  let { 
    phase, 
    currentPlayer, 
    trump, 
    bidWinner = -1,
    currentBid,
    ledSuitDisplay = null
  }: Props = $props();
  
  function getTrumpDisplay(): string {
    if (trump.type === 'suit' && trump.suit !== undefined) {
      const suits = ['0s', '1s', '2s', '3s', '4s', '5s', '6s'];
      return suits[trump.suit] || '';
    } else if (trump.type === 'doubles') {
      return 'Doubles';
    }
    return '';
  }
  
  function getPhaseDisplay(): string {
    switch(phase) {
      case 'bidding': return 'Bidding';
      case 'trump_selection': return 'Trump';
      case 'playing': return 'Playing';
      case 'scoring': return 'Scoring';
      case 'game_end': return 'Game Over';
      default: return phase;
    }
  }
</script>

<div class="game-info-bar">
  <!-- Unified Layout (horizontal for all screen sizes) -->
  <div class="flex items-center justify-between px-3 py-2 bg-base-200 rounded-lg shadow-sm">
    {#if phase === 'playing'}
      <!-- Playing phase: Trump, Bid info, and Suit -->
      {#if trump.type !== 'none'}
        <div class="flex items-center gap-1">
          <span class="text-xs text-base-content/70">Trump:</span>
          <span class="badge badge-sm badge-secondary" data-testid="trump-display">{getTrumpDisplay()}</span>
        </div>
      {:else}
        <div></div>
      {/if}
      
      <!-- Bid info in the middle -->
      {#if bidWinner !== undefined && bidWinner >= 0 && currentBid}
        <div class="flex items-center gap-1">
          <span class="text-xs text-base-content/70">Bid:</span>
          <span class="badge badge-sm badge-primary">P{bidWinner}: {currentBid.value}</span>
        </div>
      {/if}
      
      {#if ledSuitDisplay}
        <div class="flex items-center gap-1">
          <span class="text-xs text-base-content/70">Suit:</span>
          <span class="badge badge-sm badge-info">{ledSuitDisplay}</span>
        </div>
      {:else}
        <div></div>
      {/if}
    {:else}
      <!-- Other phases: phase badge + current player -->
      <div class="flex items-center gap-2">
        <span class="badge badge-primary badge-sm">{getPhaseDisplay()}</span>
        <span class="text-sm font-medium">P{currentPlayer}</span>
      </div>
      
      {#if trump.type !== 'none' && phase !== 'bidding'}
        <div class="flex items-center gap-1">
          <span class="text-sm font-medium">Trump:</span>
          <span class="text-sm font-bold" data-testid="trump-display">{getTrumpDisplay()}</span>
        </div>
      {/if}
      
      {#if phase === 'bidding' && currentBid}
        <div class="text-sm">
          <span class="font-medium">Bid: </span>
          <span>{currentBid.value} {currentBid.type}</span>
        </div>
      {/if}
    {/if}
  </div>
</div>

<style lang="postcss">
  /* svelte-ignore css_unused_selector */
  .game-info-bar {
    @apply w-full;
  }
  
  /* Ensure minimum touch target size */
  /* svelte-ignore css_unused_selector */
  .game-info-bar :global(.badge) {
    @apply min-h-[24px] px-3;
  }
</style>