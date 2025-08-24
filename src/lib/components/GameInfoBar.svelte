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
  }
  
  let { 
    phase, 
    currentPlayer, 
    trump, 
    trickNumber = 0, 
    totalTricks = 7,
    bidWinner,
    currentBid
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
  
  function getTrumpIcon(): string {
    if (trump.type === 'suit' && trump.suit !== undefined) {
      const icons = ['⚪', '⚫', '✌️', '🎲', '🍀', '🖐️', '⭐'];
      return icons[trump.suit] || '';
    } else if (trump.type === 'doubles') {
      return '⚡';
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
  <!-- Mobile Layout (horizontal) -->
  <div class="lg:hidden flex items-center justify-between px-3 py-2 bg-base-200 rounded-lg shadow-sm">
    <!-- Phase/Player Info -->
    <div class="flex items-center gap-2">
      <span class="badge badge-primary badge-sm">{getPhaseDisplay()}</span>
      <span class="text-sm font-medium">P{currentPlayer + 1}</span>
    </div>
    
    <!-- Trump Info (if set) -->
    {#if trump.type !== 'none' && phase !== 'bidding'}
      <div class="flex items-center gap-1">
        <span class="text-lg" role="img" aria-label="Trump">{getTrumpIcon()}</span>
        <span class="text-sm font-bold">{getTrumpDisplay()}</span>
      </div>
    {/if}
    
    <!-- Trick Counter (if playing) -->
    {#if phase === 'playing'}
      <div class="flex items-center gap-1">
        <span class="text-sm">Trick</span>
        <span class="badge badge-outline badge-sm">{trickNumber}/{totalTricks}</span>
      </div>
    {/if}
    
    <!-- Bid Info (if bidding) -->
    {#if phase === 'bidding' && currentBid}
      <div class="text-sm">
        <span class="font-medium">Bid: </span>
        <span>{currentBid.value} {currentBid.type}</span>
      </div>
    {/if}
  </div>
  
  <!-- Desktop Layout (vertical) -->
  <div class="hidden lg:flex flex-col gap-2 p-4 bg-base-200 rounded-lg">
    <div class="flex items-center justify-between">
      <span class="text-sm font-medium">Phase</span>
      <span class="badge badge-primary">{getPhaseDisplay()}</span>
    </div>
    
    <div class="flex items-center justify-between">
      <span class="text-sm font-medium">Player</span>
      <span class="font-bold">P{currentPlayer + 1}</span>
    </div>
    
    {#if trump.type !== 'none' && phase !== 'bidding'}
      <div class="flex items-center justify-between">
        <span class="text-sm font-medium">Trump</span>
        <div class="flex items-center gap-1">
          <span class="text-lg">{getTrumpIcon()}</span>
          <span class="font-bold">{getTrumpDisplay()}</span>
        </div>
      </div>
    {/if}
    
    {#if phase === 'playing'}
      <div class="flex items-center justify-between">
        <span class="text-sm font-medium">Progress</span>
        <span class="badge badge-outline">{trickNumber}/{totalTricks}</span>
      </div>
    {/if}
  </div>
</div>

<style>
  .game-info-bar {
    @apply w-full;
  }
  
  /* Ensure minimum touch target size on mobile */
  @media (max-width: 1023px) {
    .game-info-bar :global(.badge) {
      @apply min-h-[24px] px-3;
    }
  }
</style>