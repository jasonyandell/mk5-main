<script lang="ts">
  import { gameActions, viewProjection, controllerManager } from '../../stores/gameStore';
  import type { StateTransition } from '../../game/types';
  import Domino from './Domino.svelte';
  import { slide } from 'svelte/transition';
  
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
      // Find which human controller should handle this
      const playerId = 'player' in action.action ? action.action.player : 0;
      const humanController = controllerManager.getHumanController(playerId);
      
      if (humanController) {
        humanController.handleUserAction(action);
      } else {
        // Fallback to direct execution (used in testMode)
        console.log('[ActionPanel] Direct execution (no controller):', action.label, 'for player', playerId);
        gameActions.executeAction(action);
      }
      
      // Panel switching is handled by the reactive effect above
    } catch (error) {
      // Trigger shake animation on error
      shakeActionId = action.id;
    }
  }

  // Get team names based on perspective
  const teamNames = ['US', 'THEM'];

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

  
  // Track skip attempts for automatic retry
  let skipAttempts = $state(0);
  
  // Reactive skip logic - automatically retry skip in bidding/trump phases
  $effect(() => {
    if (($viewProjection.phase === 'bidding' || $viewProjection.phase === 'trump_selection') && 
        $viewProjection.ui.isAIThinking && skipAttempts > 0 && skipAttempts < 3) {
      // Automatically try skip on state change
      gameActions.skipAIDelays();
      skipAttempts++;
    }
  });
  
  // Reset skip attempts when not AI thinking
  $effect(() => {
    if (!$viewProjection.ui.isAIThinking) {
      skipAttempts = 0;
    }
  });

  // State for expandable team status
  let teamStatusExpanded = $state(false);
</script>

<div class="action-panel h-full flex flex-col bg-base-200 overflow-hidden" data-testid="action-panel">
  {#if ($viewProjection.phase === 'bidding' || $viewProjection.phase === 'trump_selection') && $viewProjection.hand.length > 0}
    <div class="card bg-base-100 shadow-xl m-4 mb-0 flex-shrink-0 animate-fadeInDown">
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
    
    {#if $viewProjection.ui.isWaiting && $viewProjection.ui.isAIThinking}
      <button 
        class="w-full p-5 text-center text-gray-600 flex items-center justify-center gap-2 animate-pulse bg-transparent border-none font-inherit cursor-pointer transition-transform hover:scale-105 active:scale-[0.98] ai-thinking-indicator"
        onclick={() => {
          // Skip current AI delay
          gameActions.skipAIDelays();
          // Enable automatic retry for bidding/trump phases
          if ($viewProjection.phase === 'bidding' || $viewProjection.phase === 'trump_selection') {
            skipAttempts = 1; // Start retry counter
          }
        }}
        type="button"
        aria-label="Click to skip AI thinking"
        title="Click to skip AI thinking"
      >
        <span class="text-xl">🤖</span>
        <span class="text-sm">P{$viewProjection.ui.waitingOnPlayer} is thinking...</span>
        <span class="text-xs opacity-70 ml-1">(tap to skip)</span>
      </button>
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

  <div class="card bg-base-100 shadow-lg mx-4 mb-4 flex-shrink-0">
    <button class="btn btn-ghost w-full justify-between normal-case" onclick={() => teamStatusExpanded = !teamStatusExpanded}>
      <div class="flex items-center justify-between gap-4">
        <div class="flex items-center gap-2 font-semibold">
          <span class="text-xs uppercase tracking-wider text-primary">US</span>
          <span class="text-base">{$viewProjection.scoring.teamMarks[0]}/{$viewProjection.scoring.teamScores[0]}</span>
        </div>
        <div class="flex items-center text-slate-500 transition-transform {teamStatusExpanded ? 'rotate-180' : ''}">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d={teamStatusExpanded ? "M8 10L4 6h8z" : "M8 6l4 4H4z"} />
          </svg>
        </div>
        <div class="flex items-center gap-2 font-semibold">
          <span class="text-xs uppercase tracking-wider text-error">THEM</span>
          <span class="text-base">{$viewProjection.scoring.teamMarks[1]}/{$viewProjection.scoring.teamScores[1]}</span>
        </div>
      </div>
    </button>
    
    {#if teamStatusExpanded}
      <div class="card-body pt-0" transition:slide={{ duration: 200 }}>
        <div class="grid grid-cols-2 gap-4">
          <div class="text-center">
            <h4 class="mb-2 text-sm font-semibold opacity-70">{teamNames[0]}</h4>
            <div class="text-sm my-1">{$viewProjection.scoring.teamMarks[0]} marks</div>
            <div class="text-sm my-1">{$viewProjection.scoring.teamScores[0]} points</div>
          </div>
          <div class="text-center">
            <h4 class="mb-2 text-sm font-semibold opacity-70">{teamNames[1]}</h4>
            <div class="text-sm my-1">{$viewProjection.scoring.teamMarks[1]} marks</div>
            <div class="text-sm my-1">{$viewProjection.scoring.teamScores[1]} points</div>
          </div>
        </div>
        
        {#if $viewProjection.bidding.winningBidder !== -1}
          <div class="divider"></div>
          <div>
            <div class="flex justify-between my-2 text-sm">
              <span class="opacity-70">Current Bid:</span>
              <span class="font-semibold">{$viewProjection.bidding.currentBid?.value || 0} by P{$viewProjection.bidding.winningBidder}</span>
            </div>
            {#if $viewProjection.phase === 'playing'}
              <div class="flex justify-between my-2 text-sm">
                <span class="opacity-70">Points Needed:</span>
                <span class="font-semibold">{Math.max(0, ($viewProjection.bidding.currentBid?.value || 0) - ($viewProjection.scoring.teamScores?.[Math.floor($viewProjection.bidding.winningBidder / 2)] || 0))}</span>
              </div>
            {/if}
          </div>
        {/if}
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