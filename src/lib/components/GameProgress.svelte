<script lang="ts">
  import { gameState, biddingInfo } from '../../stores/gameStore';
  import Domino from './Domino.svelte';
  import type { Trick } from '../../game/types';

  $: tricks = $gameState.tricks || [];
  $: totalPoints = tricks.reduce((sum, trick) => sum + trick.points, 0);
  $: bidTarget = $biddingInfo.currentBid.type === 'marks' 
    ? 42 * ($biddingInfo.currentBid.value || 1)
    : $biddingInfo.currentBid.value || 30;
  $: progress = Math.min((totalPoints / bidTarget) * 100, 100);

  // Get tooltip for a trick
  function getTrickTooltip(trick: Trick, index: number): string {
    const trickNum = index + 1;
    const winner = trick.winner !== undefined ? `P${trick.winner}` : 'Unknown';
    const counters = trick.plays.filter(p => p.domino.points && p.domino.points > 0)
      .map(p => `${p.domino.high}-${p.domino.low} (${p.domino.points}pts)`)
      .join(', ');
    
    let tooltip = `Trick ${trickNum}: Won by ${winner}, ${trick.points} points`;
    if (counters) {
      tooltip += `\nCounters: ${counters}`;
    }
    return tooltip;
  }

  // Track new tricks for animation
  let animatedTrickCount = 0;
  $: if (tricks.length > animatedTrickCount) {
    animatedTrickCount = tricks.length;
  }
</script>

<div class="p-4 h-full flex flex-col">
  <h2 class="mb-4 text-lg font-semibold text-[#002868]">Game Progress</h2>

  <div class="mb-5 p-3 bg-white rounded-lg border border-gray-200">
    <div class="text-sm font-semibold text-gray-700 mb-3">
      Points: {totalPoints}/42
    </div>
    {#if $gameState.phase === 'playing' && $biddingInfo.winningBidder !== -1}
      <div class="mt-2">
        <div class="h-2 bg-gray-200 rounded overflow-hidden mb-1">
          <div class="h-full bg-green-500 transition-all duration-300 ease-out" style="width: {progress}%"></div>
        </div>
        <div class="text-xs text-gray-500">
          Need: {Math.max(0, bidTarget - totalPoints)} more
        </div>
      </div>
    {/if}
  </div>

  <div class="flex-1 overflow-y-auto">
    {#if tricks.length === 0}
      <div class="text-center py-10 px-5 text-gray-400 italic">
        No tricks played yet
      </div>
    {:else}
      {#each tricks as trick, index}
        <div class="bg-white border border-gray-200 rounded-md p-1.5 mb-1 transition-all duration-200 ease-out flex items-center gap-2 min-h-0 {trick.winner !== undefined ? 'opacity-75' : ''} {index === tricks.length - 1 ? 'trick-complete-animation' : ''}" title={getTrickTooltip(trick, index)}>
          <div class="flex flex-row gap-0.5 flex-nowrap">
            {#each trick.plays as play}
              <div class="flex items-center">
                <Domino 
                  domino={play.domino} 
                  small={true}
                  tiny={true}
                  showPoints={false}
                  winner={play.player === trick.winner}
                />
              </div>
            {/each}
          </div>
          <div class="flex flex-col items-center ml-auto text-[11px] text-gray-500">
            <span class="font-semibold text-gray-700">{index + 1}</span>
            <span class="text-gray-400">{trick.points}</span>
          </div>
        </div>
      {/each}
    {/if}
  </div>
</div>