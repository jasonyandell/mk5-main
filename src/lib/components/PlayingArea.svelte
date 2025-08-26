<script lang="ts">
  import { gameActions, viewProjection, controllerManager, availableActions } from '../../stores/gameStore';
  import Domino from './Domino.svelte';
  import GameInfoBar from './GameInfoBar.svelte';
  import TrickHistoryDrawer from './TrickHistoryDrawer.svelte';
  import type { Domino as DominoType } from '../../game/types';
  import { slide } from 'svelte/transition';
  import { createEventDispatcher } from 'svelte';
  
  const dispatch = createEventDispatcher();

  // Handle domino click
  function handleDominoClick(event: CustomEvent<DominoType>) {
    const domino = event.detail;
    const playAction = $availableActions.find(
      action => action.id === `play-${domino.high}-${domino.low}` ||
                action.id === `play-${domino.low}-${domino.high}`
    );
    
    if (playAction) {
      gameActions.executeAction(playAction);
    }
  }

  // Track newly played dominoes for animation
  let playedDominoIds = $state(new Set<string>());
  $effect(() => {
    // Add new plays to the set
    const currentTrickPlays = $viewProjection.trick.current.plays.filter(p => p !== null);
    currentTrickPlays.forEach(play => {
      if (play) {
        const id = `${play.player}-${play.domino.high}-${play.domino.low}`;
        playedDominoIds.add(id);
      }
    });
  });
  
  // Create placeholder array for 4 players
  const playerPositions = [0, 1, 2, 3];

  // State for expandable trick counter
  let showTrickHistory = false;
  let drawerState = $state<'collapsed' | 'expanded'>('collapsed');
  
  // Debounce flag
  let actionPending = false;
  
  // Track phase for transitions
  let previousPhase = $viewProjection.phase;
  
  // React to phase changes for panel switching
  $effect(() => {
    if ($viewProjection.phase === 'bidding' && previousPhase === 'scoring') {
      dispatch('switchToActions');
    }
    previousPhase = $viewProjection.phase;
  });
  
  // Handle action execution
  function handleProceedAction() {
    const proceedAction = $viewProjection.actions.proceed;
    if (!proceedAction || actionPending) return;
    
    // Set debounce flag
    actionPending = true;
    
    // Find which human controller should handle this
    const playerId = 'player' in proceedAction.action ? proceedAction.action.player : 0;
    const humanController = controllerManager.getHumanController(playerId);
    if (humanController) {
      humanController.handleUserAction(proceedAction);
    } else {
      // Fallback to direct execution
      gameActions.executeAction(proceedAction);
    }
    
    // Panel switching is handled by reactive statement above
    
    // Clear debounce flag synchronously
    actionPending = false;
  }
  
  // Handle table click to skip AI delays
  function handleTableClick() {
    // If there's a proceed action, handle it
    if ($viewProjection.actions.proceed) {
      handleProceedAction();
    } else {
      // Otherwise, skip AI delays
      gameActions.skipAIDelays();
    }
  }

</script>

<div class="flex flex-col h-full relative transition-all duration-300" data-testid="playing-area" style="margin-left: {drawerState === 'expanded' ? 'calc(min(70vw, 280px) - 80px)' : '0'}">
  <!-- Mobile-optimized Game Info Bar -->
  <div class="mb-3">
    <GameInfoBar 
      phase={$viewProjection.phase}
      currentPlayer={$viewProjection.currentPlayer}
      trump={$viewProjection.trump.selection}
      trickNumber={$viewProjection.trick.number}
      totalTricks={7}
      bidWinner={$viewProjection.bidding.winningBidder}
      currentBid={$viewProjection.bidding.currentBid}
    />
  </div>
  
  {#if showTrickHistory && ($viewProjection.phase === 'playing' || $viewProjection.phase === 'scoring')}
    <div class="bg-base-100 rounded-xl mx-4 p-3 shadow-md border border-base-300" transition:slide={{ duration: 200 }}>
      <!-- Player headers -->
      <div class="flex items-center gap-3 px-3 pb-3 mb-3 border-b-2 border-base-300">
        <span class="min-w-[16px]"></span>
        <div class="flex gap-1 flex-1 items-center">
          <div class="w-10 text-center text-[11px] font-bold text-base-content/70 uppercase">P0</div>
          <div class="w-10 text-center text-[11px] font-bold text-base-content/70 uppercase">P1</div>
          <div class="w-10 text-center text-[11px] font-bold text-base-content/70 uppercase">P2</div>
          <div class="w-10 text-center text-[11px] font-bold text-base-content/70 uppercase">P3</div>
        </div>
        <span class="min-w-[70px]"></span>
      </div>
      
      {#each $viewProjection.trick.completed as trick, index}
        {@const sortedPlays = [0, 1, 2, 3].map(playerNum => 
          trick.plays.find(play => play.player === playerNum)
        )}
        <div class="flex items-center gap-3 px-3 py-2 rounded-md bg-base-200 mb-2">
          <span class="text-xs font-bold text-base-content/60 w-4">{index + 1}:</span>
          <div class="flex gap-1 flex-1 items-center">
            {#each sortedPlays as play}
              {#if play}
                <div class="inline-flex transition-all {play.player === trick.winner ? 'scale-110 drop-shadow-[0_0_4px_rgba(16,185,129,0.5)]' : ''}">
                  <Domino 
                    domino={play.domino} 
                    small={true}
                    showPoints={false}
                    clickable={false}
                  />
                </div>
              {:else}
                <div class="w-10 h-[60px] inline-flex"></div>
              {/if}
            {/each}
          </div>
          <span class="text-xs font-semibold text-success whitespace-nowrap ml-auto px-2 py-1 bg-success/10 rounded-full">P{trick.winner}✓ {trick.points || 0}pts</span>
        </div>
      {/each}
      {#if $viewProjection.trick.current.plays.filter(p => p !== null).length > 0 && $viewProjection.trick.current.plays.filter(p => p !== null).length < 4 && $viewProjection.phase === 'playing'}
        {@const sortedCurrentPlays = $viewProjection.trick.current.plays}
        <div class="flex items-center gap-3 px-3 py-2 rounded-md bg-warning/20 border border-warning mb-2">
          <span class="text-xs font-bold text-base-content/60 w-4">{$viewProjection.trick.number}:</span>
          <div class="flex gap-1 flex-1 items-center">
            {#each sortedCurrentPlays as play}
              {#if play}
                <div class="inline-flex transition-all">
                  <Domino 
                    domino={play.domino} 
                    small={true}
                    showPoints={false}
                    clickable={false}
                  />
                </div>
              {:else}
                <div class="w-10 h-[60px] inline-flex"></div>
              {/if}
            {/each}
          </div>
          <span class="text-xs font-semibold text-base-content/50 italic whitespace-nowrap ml-auto px-2 py-1 bg-base-300/50 rounded-full">(in progress)</span>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Bidding table only when waiting during bidding phase -->
  {#if $viewProjection.ui.showBiddingTable}
    <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-base-100 rounded-xl p-5 shadow-xl min-w-[280px] z-10">
      <h3 class="text-center mb-4 text-lg text-base-content font-semibold">Bidding Round</h3>
      <div class="flex flex-col gap-2 mb-4">
        {#each $viewProjection.bidding.playerStatuses as status}
          {@const playerId = status.player}
          {@const bid = status.bid}
          {@const isCurrentTurn = status.isCurrentTurn}
          {@const isAI = status.isThinking}
          {@const isYou = playerId === 0}
          
          <div class="flex justify-between px-3 py-2 rounded-md bg-base-200 transition-all {isCurrentTurn ? 'bg-info/20 ring-1 ring-info' : ''} {isYou ? 'font-semibold bg-primary/10' : ''}">
            <span class="flex items-center gap-1">
              <span class="text-base">{isAI ? '🤖' : '👤'}</span>
              P{playerId}{isYou ? ' (You)' : ''}:
            </span>
            <span class="font-medium">
              {#if bid}
                {#if bid.type === 'pass'}
                  <span class="text-base-content/60 italic">Pass</span>
                {:else}
                  <span class="text-info font-bold">{bid.value} {bid.type}</span>
                {/if}
              {:else if isCurrentTurn}
                {#if isAI}
                  <span class="text-warning animate-pulse">Thinking...</span>
                {:else}
                  <span class="text-info">Your turn...</span>
                {/if}
              {:else}
                <span class="text-base-content/60 italic">Waiting...</span>
              {/if}
            </span>
          </div>
        {/each}
      </div>
      
      {#if $viewProjection.bidding.currentBid.player !== -1}
        <div class="pt-3 border-t border-base-300 text-sm text-base-content/70 flex justify-between">
          <div>Current Bid: {$viewProjection.bidding.currentBid.value} (P{$viewProjection.bidding.currentBid.player})</div>
          <div>Dealer: P{$viewProjection.bidding.dealer}</div>
        </div>
      {:else}
        <div class="pt-3 border-t border-base-300 text-sm text-base-content/70 flex justify-between">
          <div>Opening bid</div>
          <div>Dealer: P{$viewProjection.bidding.dealer}</div>
        </div>
      {/if}
    </div>
  {/if}


  <!-- Responsive Trick Table -->
  <div class="relative flex-1" data-testid="trick-area">
    <button
      class="flex items-center justify-center p-4 relative transition-all bg-transparent border-none w-full h-full cursor-pointer tap-highlight-transparent touch-manipulation select-none min-h-[200px]"
      onclick={handleTableClick}
      disabled={false}
      type="button"
      data-testid={$viewProjection.actions.proceed ? $viewProjection.actions.proceed.id : "trick-table"}
      data-trick-button="true"
      aria-label={$viewProjection.tooltips.proceedAction || "Click to skip AI delays"}
    >
      <div class="relative bg-gradient-to-b from-primary via-primary/80 to-primary/60 rounded-full shadow-[inset_0_0_40px_rgba(0,0,0,0.3),0_10px_30px_rgba(0,0,0,0.2)] flex items-center justify-center transition-all duration-300 z-[2] {$viewProjection.actions.proceed ? 'motion-safe:animate-pulse-table' : ''} w-[240px] h-[240px] lg:w-[280px] lg:h-[280px]">
      <div class="absolute inset-5 border-2 border-base-100/10 rounded-full"></div>
      
      {#if $viewProjection.scoring.handResults}
        <!-- Scoring display -->
        <div class="flex flex-col items-center justify-center gap-6 text-base-100 text-center p-5">
          <div class="flex flex-col gap-2">
            <div class="text-sm opacity-90 font-medium">
              P{$viewProjection.scoring.handResults.biddingPlayer} ({$viewProjection.scoring.handResults.biddingTeam === 0 ? 'US' : 'THEM'}) bid {$viewProjection.scoring.handResults.bidAmount}
            </div>
            <div class="text-xl font-bold px-4 py-1.5 rounded-full tracking-wider {$viewProjection.scoring.handResults.bidMade ? 'bg-success/30 text-success-content border-2 border-success' : 'bg-error/30 text-error-content border-2 border-error'}">
              {#if $viewProjection.scoring.handResults.biddingTeam === 0}
                {$viewProjection.scoring.handResults.bidMade ? '✅ WE MADE IT!' : '❌ WE GOT SET!'}
              {:else}
                {$viewProjection.scoring.handResults.bidMade ? '❌ THEY MADE IT!' : '✅ WE SET THEM!'}
              {/if}
            </div>
          </div>
          
          <div class="flex items-center gap-5">
            <div class="flex flex-col items-center gap-1 px-6 py-4 bg-base-100/20 rounded-2xl transition-all {$viewProjection.scoring.handResults.winningTeam === 0 ? 'bg-base-100/30 scale-110 shadow-[0_0_20px_rgba(255,255,255,0.3)]' : ''}">
              <div class="text-xs font-semibold text-base-100 opacity-90 tracking-widest">US</div>
              <div class="text-3xl font-bold text-base-100 leading-none">{$viewProjection.scoring.handResults.team0Points}</div>
              <div class="text-[11px] text-base-100 opacity-80">points</div>
            </div>
            
            <div class="text-sm font-semibold text-base-100 opacity-80">vs</div>
            
            <div class="flex flex-col items-center gap-1 px-6 py-4 bg-base-100/20 rounded-2xl transition-all {$viewProjection.scoring.handResults.winningTeam === 1 ? 'bg-base-100/30 scale-110 shadow-[0_0_20px_rgba(255,255,255,0.3)]' : ''}">
              <div class="text-xs font-semibold text-base-100 opacity-90 tracking-widest">THEM</div>
              <div class="text-3xl font-bold text-base-100 leading-none">{$viewProjection.scoring.handResults.team1Points}</div>
              <div class="text-[11px] text-base-100 opacity-80">points</div>
            </div>
          </div>
        </div>
      {:else}
        <!-- Normal trick display -->
        {#each playerPositions as position}
            {@const play = $viewProjection.trick.current.plays[position]}
            {@const isWinner = $viewProjection.trick.current.winner === position}
            <div class="absolute flex items-center justify-center pointer-events-none" data-position={position} style="{position === 0 ? 'bottom: 20px; left: 50%; transform: translateX(-50%);' : position === 1 ? 'left: 20px; top: 50%; transform: translateY(-50%);' : position === 2 ? 'top: 20px; left: 50%; transform: translateX(-50%);' : 'right: 20px; top: 50%; transform: translateY(-50%);'}">
              {#if play}
                <div class="relative {playedDominoIds.has(`${play.player}-${play.domino.high}-${play.domino.low}`) ? 'motion-safe:animate-drop-in' : ''} {isWinner ? 'motion-safe:animate-winner-glow' : ''}">
                  <Domino 
                    domino={play.domino} 
                    small={true}
                    showPoints={true}
                    clickable={false}
                  />
                  <div class="absolute -bottom-5 left-1/2 -translate-x-1/2 text-[11px] font-bold text-base-100 bg-base-content/70 px-2 py-0.5 rounded-full">P{position}</div>
                {#if isWinner}
                  <div class="absolute -top-6 left-1/2 -translate-x-1/2 bg-gradient-to-br from-yellow-400 to-yellow-500 text-slate-800 px-2.5 py-1 rounded-xl text-[11px] font-bold flex items-center gap-1 shadow-[0_2px_8px_rgba(255,215,0,0.5)] motion-safe:animate-bounce-in whitespace-nowrap z-[15]">
                    <span class="text-sm motion-safe:animate-sparkle">👑</span>
                    <span class="uppercase tracking-wider">Winner!</span>
                  </div>
                {/if}
              </div>
            {:else}
              <div class="relative w-[50px] h-[80px] flex items-center justify-center pointer-events-none">
                <div class="absolute inset-0 border-[3px] border-dashed border-base-100/30 rounded-xl motion-safe:animate-spin-slow"></div>
                <span class="text-xs opacity-70 mr-0.5">
                  {controllerManager.isAIControlled(position) ? '🤖' : '👤'}
                </span>
                <span class="text-sm font-bold text-base-100/60">P{position}</span>
              </div>
            {/if}
          </div>
        {/each}
      {/if}
    </div>
    
    {#if $viewProjection.ui.isAIThinking}
      <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white/95 px-4 py-2 rounded-full shadow-[0_2px_8px_rgba(0,0,0,0.1)] flex items-center gap-2 text-sm text-base-content/70 motion-safe:animate-pulse pointer-events-none z-10">
        <span class="text-lg">🤖</span>
        <span>P{$viewProjection.ui.waitingOnPlayer} is thinking...</span>
      </div>
    {/if}
    
    {#if $viewProjection.actions.proceed}
      <div 
        class="absolute top-[calc(50%+140px+25px)] left-1/2 -translate-x-1/2 flex items-center gap-2 px-5 py-3 bg-secondary text-secondary-content rounded-full text-sm font-semibold shadow-[0_4px_12px_rgba(139,92,246,0.3)] z-10 motion-safe:animate-tap-bounce"
        role="presentation"
      >
        <span class="text-lg motion-safe:animate-tap-point">👆</span>
        <span class="whitespace-nowrap">{$viewProjection.tooltips.proceedAction}</span>
      </div>
    {/if}
    </button>
  </div>

  <!-- Player Hand with Touch-Optimized Spacing -->
  <div class="relative bg-gradient-to-b from-transparent to-base-100/50 pt-4">
    
    {#if $viewProjection.hand.length === 0}
      <div class="flex flex-col items-center gap-3 py-8 text-base-content/50">
        <span class="text-5xl opacity-50">🀚</span>
        <span class="text-sm font-medium">No dominoes</span>
      </div>
    {:else}
      <div class="px-4 py-4">
        <div class="flex flex-wrap gap-3 justify-center">
          {#each $viewProjection.hand as handDomino, i (handDomino.domino.high + '-' + handDomino.domino.low)}
            <!-- Larger tap target wrapper for mobile -->
            <div 
              class="motion-safe:animate-hand-slide touch-manipulation" 
              style="animation-delay: {i * 50}ms" 
              data-testid="domino-{handDomino.domino.high}-{handDomino.domino.low}" 
              data-playable={handDomino.isPlayable}
            >
              <Domino
                domino={handDomino.domino}
                playable={handDomino.isPlayable}
                clickable={handDomino.isPlayable && $viewProjection.phase === 'playing'}
                showPoints={true}
                tooltip={handDomino.tooltip}
                on:click={handleDominoClick}
              />
            </div>
          {/each}
        </div>
      </div>
    {/if}
  </div>
  
  <!-- Trick History Drawer (only during playing/scoring phases) -->
  {#if $viewProjection.phase === 'playing' || $viewProjection.phase === 'scoring'}
    <TrickHistoryDrawer 
      completedTricks={$viewProjection.trick.completed}
      currentTrick={$viewProjection.trick.current.plays.filter(p => p !== null)}
      trickNumber={$viewProjection.trick.number}
      onStateChange={(state) => {
        drawerState = state;
      }}
    />
  {/if}
</div>

<style>
  /* Custom animations that can't be expressed in Tailwind */
  @media (prefers-reduced-motion: reduce) {
    * {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
    }
  }

  @keyframes animate-drop-in {
    from {
      transform: translateY(-100px) scale(0.8);
      opacity: 0;
    }
    to {
      transform: translateY(0) scale(1);
      opacity: 1;
    }
  }
  
  /* svelte-ignore css_unused_selector */
  .animate-drop-in {
    animation: animate-drop-in 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  }
  
  @keyframes animate-winner-glow {
    0%, 100% {
      filter: drop-shadow(0 0 8px rgba(255, 215, 0, 0.6));
      transform: scale(1);
    }
    50% {
      filter: drop-shadow(0 0 20px rgba(255, 215, 0, 0.9));
      transform: scale(1.05);
    }
  }
  
  /* svelte-ignore css_unused_selector */
  .animate-winner-glow {
    animation: animate-winner-glow 2s ease-in-out infinite;
  }
  
  @keyframes animate-bounce-in {
    0% {
      opacity: 0;
      transform: translateX(-50%) translateY(-10px) scale(0.3);
    }
    50% {
      transform: translateX(-50%) translateY(0) scale(1.1);
    }
    100% {
      opacity: 1;
      transform: translateX(-50%) translateY(0) scale(1);
    }
  }
  
  /* svelte-ignore css_unused_selector */
  .animate-bounce-in {
    animation: animate-bounce-in 0.2s cubic-bezier(0.68, -0.55, 0.265, 1.55);
  }
  
  @keyframes animate-sparkle {
    0%, 100% {
      transform: rotate(0deg) scale(1);
    }
    25% {
      transform: rotate(-10deg) scale(1.1);
    }
    75% {
      transform: rotate(10deg) scale(1.1);
    }
  }
  
  /* svelte-ignore css_unused_selector */
  .animate-sparkle {
    animation: animate-sparkle 2s ease-in-out infinite;
  }
  
  @keyframes animate-spin-slow {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
  
  /* svelte-ignore css_unused_selector */
  .animate-spin-slow {
    animation: animate-spin-slow 20s linear infinite;
  }
  
  @keyframes animate-tap-bounce {
    0%, 100% {
      transform: translateX(-50%) translateY(0);
    }
    50% {
      transform: translateX(-50%) translateY(-5px);
    }
  }
  
  /* svelte-ignore css_unused_selector */
  .animate-tap-bounce {
    animation: animate-tap-bounce 1.5s ease-in-out infinite;
  }
  
  @keyframes animate-tap-point {
    0%, 100% {
      transform: translateY(0);
    }
    50% {
      transform: translateY(-3px);
    }
  }
  
  /* svelte-ignore css_unused_selector */
  .animate-tap-point {
    animation: animate-tap-point 1.5s ease-in-out infinite;
  }
  
  @keyframes animate-hand-slide {
    from {
      opacity: 0;
      transform: translateY(30px) rotate(-10deg);
    }
    to {
      opacity: 1;
      transform: translateY(0) rotate(0);
    }
  }
  
  /* svelte-ignore css_unused_selector */
  .animate-hand-slide {
    animation: animate-hand-slide 0.5s cubic-bezier(0.4, 0, 0.2, 1) both;
  }
  
  @keyframes animate-pulse-table {
    0%, 100% {
      transform: scale(1);
      box-shadow: 
        inset 0 0 40px rgba(0, 0, 0, 0.3),
        0 10px 30px rgba(0, 0, 0, 0.2),
        0 0 0 0 rgba(139, 92, 246, 0);
    }
    50% {
      transform: scale(1.05);
      box-shadow: 
        inset 0 0 50px rgba(139, 92, 246, 0.2),
        0 10px 30px rgba(0, 0, 0, 0.2),
        0 0 40px 20px rgba(139, 92, 246, 0.5);
    }
  }
  
  /* svelte-ignore css_unused_selector */
  .animate-pulse-table {
    animation: animate-pulse-table 2s ease-in-out infinite;
  }
  
  /* svelte-ignore css_unused_selector */
  .tap-highlight-transparent {
    -webkit-tap-highlight-color: transparent;
  }
  
  /* Mobile adjustments for tap indicator position */
  @media (max-width: 640px) {
    /* svelte-ignore css_unused_selector */
    .tap-indicator-mobile {
      top: calc(50% + 120px + 20px) !important;
    }
  }
</style>