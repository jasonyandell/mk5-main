<script lang="ts">
  import { gamePhase, availableActions, gameActions, teamInfo, biddingInfo, currentPlayer, playerView, gameState, uiState, controllerManager } from '../../stores/gameStore';
  import type { StateTransition } from '../../game/types';
  import Domino from './Domino.svelte';
  import { slide } from 'svelte/transition';
  
  interface Props {
    onswitchToPlay?: () => void;
  }
  
  let { onswitchToPlay }: Props = $props();
  
  // Check if we're in test mode
  const urlParams = typeof window !== 'undefined' ? 
    new URLSearchParams(window.location.search) : null;
  const testMode = urlParams?.get('testMode') === 'true';
  

  // Group actions by type with strong typing
  interface GroupedActions {
    bidding: StateTransition[];
    trump: StateTransition[];
    play: StateTransition[];
    other: StateTransition[];
  }

  const groupedActions = $derived((() => {
    const groups: GroupedActions = {
      bidding: [],
      trump: [],
      play: [],
      other: []
    };

    $availableActions.forEach(action => {
      if (action.id.startsWith('bid-') || action.id === 'pass' || action.id === 'redeal') {
        groups.bidding.push(action);
      } else if (action.id.startsWith('trump-')) {
        groups.trump.push(action);
      } else if (action.id.startsWith('play-')) {
        // Skip play actions - they're handled by domino clicks
      } else {
        groups.other.push(action);
      }
    });

    return groups;
  })());


  let shakeActionId = $state<string | null>(null);
  let previousPhase = $state($gamePhase);
  
  // React to phase changes for panel switching
  $effect(() => {
    if ($gamePhase === 'playing' && previousPhase === 'trump_selection') {
      onswitchToPlay?.();
    }
    previousPhase = $gamePhase;
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

  // Current player's hand (always player 0 for privacy, unless in test mode)
  const playerHand = $derived(testMode ? ($currentPlayer?.hand || []) : ($playerView?.self?.hand || []));

  // Use centralized UI state for waiting logic
  const isWaiting = $derived($uiState.isWaiting);
  const waitingPlayer = $derived($uiState.waitingOnPlayer);
  const isAIThinking = $derived(waitingPlayer >= 0 && controllerManager.isAIControlled(waitingPlayer));
  
  // Track skip attempts for automatic retry
  let skipAttempts = $state(0);
  
  // Reactive skip logic - automatically retry skip in bidding/trump phases
  $effect(() => {
    if (($gamePhase === 'bidding' || $gamePhase === 'trump_selection') && 
        isAIThinking && skipAttempts > 0 && skipAttempts < 3) {
      // Automatically try skip on state change
      gameActions.skipAIDelays();
      skipAttempts++;
    }
  });
  
  // Reset skip attempts when not AI thinking
  $effect(() => {
    if (!isAIThinking) {
      skipAttempts = 0;
    }
  });

  // State for expandable team status
  let teamStatusExpanded = $state(false);
</script>

<div class="action-panel h-full flex flex-col bg-base-200 overflow-hidden" data-testid="action-panel">
  {#if ($gamePhase === 'bidding' || $gamePhase === 'trump_selection') && playerHand.length > 0}
    <div class="card bg-base-100 shadow-xl m-4 mb-0 flex-shrink-0 animate-fadeInDown">
      <div class="card-body p-4">
        <h3 class="card-title text-sm uppercase tracking-wider justify-center mb-4">Your Hand</h3>
        <div class="grid grid-cols-[repeat(auto-fit,minmax(45px,1fr))] gap-2 max-w-full justify-items-center">
        {#each playerHand as domino, i (domino.high + '-' + domino.low)}
          <div class="animate-handFadeIn" style="--delay: {i * 30}ms; animation-delay: var(--delay)">
            <Domino
              {domino}
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
    {#if $gamePhase === 'bidding'}
      <!-- Always show compact bid status during bidding -->
      <div class="card bg-base-100 shadow-lg mb-4">
        <div class="card-body p-3">
        <div class="flex gap-2 justify-center flex-wrap">
          {#each [0, 1, 2, 3] as playerId}
            {@const bid = $biddingInfo.bids.find(b => b.player === playerId)}
            {@const isHighBidder = $biddingInfo.currentBid.player === playerId}
            {@const isDealer = $gameState.dealer === playerId}
            <div class="badge {isHighBidder ? 'badge-success badge-lg' : 'badge-outline'} {isDealer && !isHighBidder ? 'badge-secondary' : ''} gap-1">
              <span class="font-semibold">P{playerId}</span>
              <span>
                {#if bid}
                  {#if bid.type === 'pass'}
                    Pass
                  {:else}
                    {bid.value}
                  {/if}
                {:else}
                  --
                {/if}
              </span>
              {#if isDealer}
                <span class="absolute -top-2 -right-2 badge badge-secondary badge-xs" title="Dealer">D</span>
              {/if}
            </div>
          {/each}
        </div>
        {#if $biddingInfo.currentBid.player !== -1}
          <div class="text-center mt-2 text-xs font-semibold">
            High: {$biddingInfo.currentBid.value}
          </div>
        {/if}
        </div>
      </div>
    {/if}
    
    {#if isWaiting && isAIThinking}
      <button 
        class="w-full p-5 text-center text-gray-600 flex items-center justify-center gap-2 animate-pulse bg-transparent border-none font-inherit cursor-pointer transition-transform hover:scale-105 active:scale-[0.98] ai-thinking-indicator"
        onclick={() => {
          // Skip current AI delay
          gameActions.skipAIDelays();
          // Enable automatic retry for bidding/trump phases
          if ($gamePhase === 'bidding' || $gamePhase === 'trump_selection') {
            skipAttempts = 1; // Start retry counter
          }
        }}
        type="button"
        aria-label="Click to skip AI thinking"
        title="Click to skip AI thinking"
      >
        <span class="text-xl">🤖</span>
        <span class="text-sm">P{waitingPlayer} is thinking...</span>
        <span class="text-xs opacity-70 ml-1">(tap to skip)</span>
      </button>
    {/if}
    
    {#if $gamePhase === 'bidding'}
      <div class="mb-6 animate-fadeInUp">
        <h3 class="mb-4 text-sm font-semibold uppercase tracking-wider text-center opacity-70">Bidding</h3>
        <div class="grid grid-cols-3 gap-3 max-w-[400px] mx-auto">
          {#each groupedActions.bidding as action}
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
          
          {#each groupedActions.bidding as action}
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

    {#if $gamePhase === 'trump_selection'}
      <div class="mb-6 animate-fadeInUp">
        <h3 class="mb-4 text-sm font-semibold uppercase tracking-wider text-center opacity-70">Select Trump</h3>
        <div class="flex flex-col gap-3 max-w-[320px] mx-auto">
          {#each groupedActions.trump as action}
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

    {#if groupedActions.other.length > 0}
      <div class="mb-6 animate-fadeInUp">
        <h3 class="mb-4 text-sm font-semibold uppercase tracking-wider text-center opacity-70">Quick Actions</h3>
        <div class="flex flex-col gap-3 max-w-[320px] mx-auto">
          {#each groupedActions.other as action}
            <button 
              class="btn btn-outline btn-lg w-full min-h-[48px]"
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
          <span class="text-base">{$teamInfo.marks[0]}/{$teamInfo.scores[0]}</span>
        </div>
        <div class="flex items-center text-slate-500 transition-transform {teamStatusExpanded ? 'rotate-180' : ''}">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d={teamStatusExpanded ? "M8 10L4 6h8z" : "M8 6l4 4H4z"} />
          </svg>
        </div>
        <div class="flex items-center gap-2 font-semibold">
          <span class="text-xs uppercase tracking-wider text-error">THEM</span>
          <span class="text-base">{$teamInfo.marks[1]}/{$teamInfo.scores[1]}</span>
        </div>
      </div>
    </button>
    
    {#if teamStatusExpanded}
      <div class="card-body pt-0" transition:slide={{ duration: 200 }}>
        <div class="grid grid-cols-2 gap-4">
          <div class="text-center">
            <h4 class="mb-2 text-sm font-semibold opacity-70">{teamNames[0]}</h4>
            <div class="text-sm my-1">{$teamInfo.marks[0]} marks</div>
            <div class="text-sm my-1">{$teamInfo.scores[0]} points</div>
          </div>
          <div class="text-center">
            <h4 class="mb-2 text-sm font-semibold opacity-70">{teamNames[1]}</h4>
            <div class="text-sm my-1">{$teamInfo.marks[1]} marks</div>
            <div class="text-sm my-1">{$teamInfo.scores[1]} points</div>
          </div>
        </div>
        
        {#if $biddingInfo.winningBidder !== -1}
          <div class="divider"></div>
          <div>
            <div class="flex justify-between my-2 text-sm">
              <span class="opacity-70">Current Bid:</span>
              <span class="font-semibold">{$biddingInfo.currentBid?.value || 0} by P{$biddingInfo.winningBidder}</span>
            </div>
            {#if $gamePhase === 'playing'}
              <div class="flex justify-between my-2 text-sm">
                <span class="opacity-70">Points Needed:</span>
                <span class="font-semibold">{Math.max(0, ($biddingInfo.currentBid?.value || 0) - ($teamInfo.scores?.[Math.floor($biddingInfo.winningBidder / 2)] || 0))}</span>
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