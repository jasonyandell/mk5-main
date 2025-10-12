<script lang="ts">
  import { gameActions, viewProjection } from '../../stores/gameStore';
  import type { StateTransition } from '../../game/types';
  import Domino from './Domino.svelte';
  import Icon from '../icons/Icon.svelte';

  interface Props {
    onswitchToPlay?: () => void;
  }

  let { onswitchToPlay }: Props = $props();

  let shakeActionId = $state<string | null>(null);
  let previousPhase = $state($viewProjection.phase);

  // React to phase changes for panel switching
  $effect(() => {
    if ($viewProjection.phase === 'playing' && previousPhase === 'trump_selection') {
      onswitchToPlay?.();
    }
    previousPhase = $viewProjection.phase;
  });

  async function executeAction(action: StateTransition) {
    try {
      await gameActions.executeAction(action);
    } catch (error) {
      // Trigger shake animation on error
      shakeActionId = action.id;
    }
  }

  // Get tooltip for bid actions
  function getBidTooltip(action: StateTransition): string {
    if (action.id === 'pass') {
      return 'Pass on bidding - let others bid';
    }
    
    const bidValue = action.id.match(/\d+/)?.[0];
    if (!bidValue) return '';
    
    const points = parseInt(bidValue);
    if (action.id.includes('mark')) {
      const marks = parseInt(bidValue);
      return `Bid ${marks} mark${marks > 1 ? 's' : ''} (${marks * 42} points) - must win all ${marks * 42} points`;
    } else {
      return `Bid ${points} points - must win at least ${points} out of 42 points`;
    }
  }

</script>

<div class="action-panel h-full flex flex-col bg-base-200 overflow-hidden" data-testid="action-panel">
  <div class="flex-1 overflow-y-auto overflow-x-hidden p-4 touch-pan-y bg-base-200">
    {#if $viewProjection.phase === 'bidding'}
      <!-- Always show compact bid status during bidding -->
      <div class="card bg-base-100 shadow-lg mb-4">
        <div class="card-body p-3">
        <div class="flex gap-2 justify-center flex-wrap">
          {#each $viewProjection.bidding.playerStatuses as status}
            <div class="badge {status.isHighBidder ? 'badge-success badge-lg' : 'badge-outline'} {status.isDealer && !status.isHighBidder ? 'badge-secondary' : ''} gap-1">
              <span class="font-semibold">P{status.player}</span>
              <span>
                {#if status.bid}
                  {#if status.bid.type === 'pass'}
                    Pass
                  {:else}
                    {status.bid.value}
                  {/if}
                {:else}
                  --
                {/if}
              </span>
              {#if status.isDealer}
                <span class="absolute -top-2 -right-2 badge badge-secondary badge-xs" title="Dealer">D</span>
              {/if}
            </div>
          {/each}
        </div>
        {#if $viewProjection.bidding.currentBid.player !== -1}
          <div class="text-center mt-2 text-xs font-semibold">
            High: {$viewProjection.bidding.currentBid.value}
          </div>
        {/if}
        </div>
      </div>
    {/if}

    {#if $viewProjection.phase === 'trump_selection'}
      <!-- Show bid winner info during trump selection -->
      <div class="card bg-base-100 shadow-lg mb-4">
        <div class="card-body p-3">
          <div class="text-center">
            <div class="text-sm opacity-70 mb-1">Winning Bid</div>
            <div class="flex items-center justify-center gap-2">
              <span class="badge badge-success badge-lg">
                P{$viewProjection.bidding.winningBidder} - {$viewProjection.bidding.currentBid.value}
              </span>
            </div>
            <div class="text-xs opacity-60 mt-2">Select trump suit for this hand</div>
          </div>
        </div>
      </div>
    {/if}

    {#if ($viewProjection.phase === 'bidding' || $viewProjection.phase === 'trump_selection') && $viewProjection.hand.length > 0}
      <div class="card bg-base-100 shadow-xl mb-4 animate-fadeInDown">
        <div class="card-body p-4">
          <h3 class="card-title text-sm uppercase tracking-wider justify-center mb-4">Your Hand</h3>
          <div class="grid grid-cols-[repeat(auto-fit,minmax(45px,1fr))] gap-2 max-w-full justify-items-center">
          {#each $viewProjection.hand as handDomino, i (handDomino.domino.high + '-' + handDomino.domino.low)}
            <div class="animate-handFadeIn" style="--delay: {i * 30}ms; animation-delay: var(--delay)">
              <Domino
                domino={handDomino.domino}
                small={true}
                showPoints={true}
                clickable={true}
              />
            </div>
          {/each}
          </div>
        </div>
      </div>
    {/if}
    
    {#if $viewProjection.ui.isWaiting && $viewProjection.ui.isAIThinking}
      <div class="w-full p-5 text-center text-base-content/60 flex items-center justify-center gap-2 animate-pulse">
        <Icon name="cpuChip" size="md" />
        <span class="text-sm">P{$viewProjection.ui.waitingOnPlayer} is thinking...</span>
      </div>
    {/if}
    
    {#if $viewProjection.phase === 'bidding' && $viewProjection.actions.bidding.length > 0}
      <div class="mb-6 animate-fadeInUp">
        <h3 class="mb-4 text-sm font-semibold uppercase tracking-wider text-center opacity-70">Bidding</h3>
        <div class="grid grid-cols-3 gap-3 max-w-[400px] mx-auto">
          {#each $viewProjection.actions.bidding as action}
            {#if action.id === 'pass'}
              <button 
                class="btn btn-error col-span-full {shakeActionId === action.id ? 'invalid-action-shake' : ''}"
                onclick={() => executeAction(action)}
                onanimationend={() => { if (shakeActionId === action.id) shakeActionId = null; }}
                data-testid="pass"
                data-testid-alt="pass-button"
                title={getBidTooltip(action)}
              >
                Pass
              </button>
            {:else if action.id === 'redeal'}
              <button 
                class="btn btn-warning col-span-full {shakeActionId === action.id ? 'invalid-action-shake' : ''}"
                onclick={() => executeAction(action)}
                onanimationend={() => { if (shakeActionId === action.id) shakeActionId = null; }}
                data-testid={action.id}
                title="Redeal the dominoes - all players passed"
              >
                Redeal
              </button>
            {/if}
          {/each}
          
          <div class="col-span-full h-px bg-gradient-to-r from-transparent via-slate-200 to-transparent my-4"></div>
          
          {#each $viewProjection.actions.bidding as action}
            {#if action.id !== 'pass' && action.id !== 'redeal'}
              <button 
                class="btn btn-primary {shakeActionId === action.id ? 'invalid-action-shake' : ''}"
                onclick={() => executeAction(action)}
                onanimationend={() => { if (shakeActionId === action.id) shakeActionId = null; }}
                data-testid={action.id}
                title={getBidTooltip(action)}
              >
                {action.label}
              </button>
            {/if}
          {/each}
        </div>
      </div>
    {/if}

    {#if $viewProjection.phase === 'trump_selection' && $viewProjection.actions.trump.length > 0}
      <div class="mb-6 animate-fadeInUp">
        <h3 class="mb-4 text-sm font-semibold uppercase tracking-wider text-center opacity-70">Select Trump</h3>
        <div class="flex flex-col gap-3 max-w-[320px] mx-auto">
          {#each $viewProjection.actions.trump as action}
            <button 
              class="btn btn-secondary btn-lg w-full min-h-[48px]"
              onclick={() => executeAction(action)}
              data-testid={action.id}
            >
              {action.label}
            </button>
          {/each}
        </div>
      </div>
    {/if}

  </div>
</div>

<style>
  /* svelte-ignore css_unused_selector */
  .tap-highlight-transparent {
    -webkit-tap-highlight-color: transparent;
  }
  
  /* svelte-ignore css_unused_selector */
  .touch-manipulation {
    touch-action: manipulation;
  }
</style>
