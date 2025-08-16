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

<div class="game-progress">
  <h2>Game Progress</h2>

  <div class="progress-summary">
    <div class="point-count">
      Points: {totalPoints}/42
    </div>
    {#if $gameState.phase === 'playing' && $biddingInfo.winningBidder !== -1}
      <div class="bid-progress">
        <div class="progress-bar">
          <div class="progress-fill" style="width: {progress}%"></div>
        </div>
        <div class="progress-text">
          Need: {Math.max(0, bidTarget - totalPoints)} more
        </div>
      </div>
    {/if}
  </div>

  <div class="tricks-list">
    {#if tricks.length === 0}
      <div class="empty-state">
        No tricks played yet
      </div>
    {:else}
      {#each tricks as trick, index}
        <div class="trick-card compact {index === tricks.length - 1 ? 'trick-complete-animation' : ''}" class:completed={trick.winner !== undefined} title={getTrickTooltip(trick, index)}>
          <div class="trick-dominoes-horizontal">
            {#each trick.plays as play}
              <div class="played-domino">
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
          <div class="trick-info">
            <span class="trick-number">{index + 1}</span>
            <span class="trick-points">{trick.points}</span>
          </div>
        </div>
      {/each}
    {/if}
  </div>
</div>

<style>
  .game-progress {
    padding: 16px;
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  h2 {
    margin: 0 0 16px 0;
    font-size: 18px;
    font-weight: 600;
    color: #002868;
  }

  .progress-summary {
    margin-bottom: 20px;
    padding: 12px;
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
  }

  .point-count {
    font-size: 14px;
    font-weight: 600;
    color: #374151;
    margin-bottom: 12px;
  }

  .bid-progress {
    margin-top: 8px;
  }

  .progress-bar {
    height: 8px;
    background-color: #e5e7eb;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 4px;
  }

  .progress-fill {
    height: 100%;
    background-color: #22c55e;
    transition: width 0.3s ease;
  }

  .progress-text {
    font-size: 12px;
    color: #6b7280;
  }

  .tricks-list {
    flex: 1;
    overflow-y: auto;
  }

  .empty-state {
    text-align: center;
    padding: 40px 20px;
    color: #9ca3af;
    font-style: italic;
  }

  .trick-card {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 6px;
    margin-bottom: 4px;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .trick-card.compact {
    min-height: auto;
  }

  .trick-card.completed {
    opacity: 0.75;
  }

  .trick-info {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-left: auto;
    font-size: 11px;
    color: #6b7280;
  }

  .trick-number {
    font-weight: 600;
    color: #374151;
  }

  .trick-points {
    color: #9ca3af;
  }

  .trick-dominoes-horizontal {
    display: flex !important;
    flex-direction: row !important;
    gap: 2px;
    flex-wrap: nowrap;
  }

  .played-domino {
    display: flex;
    align-items: center;
  }
</style>