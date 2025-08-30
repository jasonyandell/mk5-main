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
      const suits = ["blanks", "aces(1)", "deuces(2)", "tres(3)", "4's", "5's", "6's"];
      return suits[trump.suit] || '';
    } else if (trump.type === 'doubles') {
      return 'doubles';
    }
    return '';
  }
</script>

<div class="game-info-bar">
  {#if phase === 'playing'}
    <!-- Clean single-line display during play -->
    <div class="flex items-center justify-center bg-base-200 border-y border-base-300 divide-x divide-base-300">
      <!-- Current player -->
      <div class="px-3 py-2 text-sm">
        <span class="font-semibold text-primary">P{currentPlayer} turn</span>
      </div>
      
      <!-- Trump -->
      {#if trump.type !== 'not-selected'}
        <div class="px-3 py-2 text-sm">
          <span class="font-semibold text-secondary">{getTrumpDisplay()} trump</span>
        </div>
      {/if}
      
      <!-- Bid info -->
      {#if bidWinner !== undefined && bidWinner >= 0 && currentBid}
        <div class="px-3 py-2 text-sm">
          <span class="font-semibold text-accent">P{bidWinner} bid {currentBid.value}</span>
        </div>
      {/if}
      
      <!-- Led suit (only shows after first domino played) -->
      {#if ledSuitDisplay}
        <div class="px-3 py-2 text-sm bg-info/10">
          <span class="font-bold text-info">{ledSuitDisplay.toLowerCase()} led</span>
        </div>
      {/if}
    </div>
  {:else}
    <!-- Non-playing phases: simplified display -->
    <div class="flex items-center justify-between px-3 py-2 bg-base-100/90 backdrop-blur-sm shadow-sm">
      {#if phase === 'bidding' && currentBid}
        <span class="text-sm">Current bid: <span class="font-semibold">{currentBid.value}</span></span>
      {:else if phase === 'trump_selection'}
        <span class="text-sm">P{currentPlayer} selecting trump...</span>
      {:else if phase === 'scoring'}
        <span class="text-sm">Scoring hand...</span>
      {:else}
        <span class="text-sm text-base-content/60">Ready to play</span>
      {/if}
      
      {#if trump.type !== 'not-selected' && phase !== 'bidding'}
        <span class="text-sm font-semibold text-secondary" data-testid="trump-display">Trump: {getTrumpDisplay()}</span>
      {/if}
    </div>
  {/if}
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