<script lang="ts">
  import { gameState, availableActions, gameActions, currentPlayer, gamePhase, teamInfo, biddingInfo, playerView, uiState, controllerManager } from '../../stores/gameStore';
  import Domino from './Domino.svelte';
  import type { Domino as DominoType } from '../../game/types';
  import { slide } from 'svelte/transition';
  import { createEventDispatcher } from 'svelte';
  import { calculateTrickWinner } from '../../game/core/scoring';
  
  const dispatch = createEventDispatcher();
  
  // Check if we're in test mode
  const urlParams = typeof window !== 'undefined' ? 
    new URLSearchParams(window.location.search) : null;
  const testMode = urlParams?.get('testMode') === 'true';

  // Extract playable dominoes from available actions
  $: playableDominoes = (() => {
    const dominoes = new Set<string>();
    $availableActions
      .filter(action => action.id.startsWith('play-'))
      .forEach(action => {
        const dominoId = action.id.replace('play-', '');
        dominoes.add(dominoId);
        // Add reversed version (5-3 and 3-5)
        const parts = dominoId.split('-');
        if (parts.length === 2) {
          dominoes.add(`${parts[1]}-${parts[0]}`);
        }
      });
    return dominoes;
  })();

  // Check if a domino is playable
  function isDominoPlayable(domino: DominoType): boolean {
    return playableDominoes.has(`${domino.high}-${domino.low}`) || 
           playableDominoes.has(`${domino.low}-${domino.high}`);
  }

  // Get tooltip for domino based on play state
  function getDominoTooltip(domino: DominoType): string {
    const dominoStr = `${domino.high}-${domino.low}`;
    
    // Not playing phase
    if ($gamePhase !== 'playing') {
      return dominoStr;
    }

    // Check if it's this player's turn
    if ($gameState.currentPlayer !== 0) { // Assuming player 0 is the human player
      return `${dominoStr} - Waiting for P${$gameState.currentPlayer}'s turn`;
    }

    const isPlayable = isDominoPlayable(domino);
    
    // First play of trick
    if ($gameState.currentTrick.length === 0) {
      return isPlayable ? `${dominoStr} - Click to lead this domino` : dominoStr;
    }

    // Must follow suit
    const leadSuit = $gameState.currentSuit;
    if (leadSuit === -1) {
      return isPlayable ? `${dominoStr} - Click to play` : dominoStr;
    }

    // Get suit names
    const suitNames = ['blanks', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixes', 'doubles'];
    const ledSuitName = leadSuit === 7 ? 'doubles' : suitNames[leadSuit];

    if (isPlayable) {
      // Check if this domino follows suit
      if (leadSuit === 7 && domino.high === domino.low) {
        return `${dominoStr} - Double, follows ${ledSuitName}`;
      } else if (leadSuit !== 7 && (domino.high === leadSuit || domino.low === leadSuit)) {
        return `${dominoStr} - Has ${ledSuitName}, follows suit`;
      } else {
        // Must be trump or can't follow suit
        if ($gameState.trump.type === 'doubles' && domino.high === domino.low) {
          return `${dominoStr} - Trump (double)`;
        } else if ($gameState.trump.type === 'suit' && 
                   (domino.high === $gameState.trump.suit || domino.low === $gameState.trump.suit)) {
          return `${dominoStr} - Trump`;
        } else {
          return `${dominoStr} - Can't follow ${ledSuitName}`;
        }
      }
    } else {
      // Not playable - must explain why
      if (leadSuit === 7) {
        return `${dominoStr} - Not a double, can't follow ${ledSuitName}`;
      } else {
        // Check if player has the led suit
        const playerHasLedSuit = playerHand.some(d => 
          d.high === leadSuit || d.low === leadSuit
        );
        
        if (playerHasLedSuit) {
          return `${dominoStr} - Must follow ${ledSuitName}`;
        } else {
          // This shouldn't happen if the domino is marked unplayable
          return `${dominoStr} - Invalid play`;
        }
      }
    }
  }

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

  // Get trump display text
  $: trumpDisplay = (() => {
    if ($gameState.trump.type === 'none') return 'Not Selected';
    if ($gameState.trump.type === 'no-trump') return 'No Trump';
    if ($gameState.trump.type === 'doubles') return 'Doubles';
    if ($gameState.trump.type === 'suit' && $gameState.trump.suit !== undefined) {
      const suitNames = ['Blanks', 'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes'];
      return suitNames[$gameState.trump.suit];
    }
    return 'Unknown';
  })();

  // Get led suit display
  $: ledSuitDisplay = (() => {
    if ($gameState.currentSuit === -1) return null;
    const suitNames = ['Blanks', 'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes'];
    return suitNames[$gameState.currentSuit];
  })();

  // Current player's hand (always player 0 for privacy, unless in test mode)
  $: playerHand = testMode ? ($currentPlayer?.hand || []) : ($playerView?.self?.hand || []);

  // Check if AI is thinking (but not during consensus actions)
  $: isThinking = $gameState.phase === 'playing' && 
                   $gameState.currentPlayer !== 0 &&
                   controllerManager.isAIControlled($gameState.currentPlayer) &&
                   $gameState.currentTrick.length < 4;  // Not thinking if trick is complete

  // Current trick plays
  $: currentTrickPlays = $gameState.currentTrick || [];

  // Track newly played dominoes for animation
  let playedDominoIds = new Set<string>();
  $: {
    // Add new plays to the set
    currentTrickPlays.forEach(play => {
      const id = `${play.player}-${play.domino.high}-${play.domino.low}`;
      playedDominoIds.add(id);
    });
  }
  
  // Calculate the winning player for the current trick
  $: trickWinner = (() => {
    if (currentTrickPlays.length === 4) {
      return calculateTrickWinner(
        currentTrickPlays,
        $gameState.trump,
        $gameState.currentSuit
      );
    }
    return -1;
  })();

  // Create placeholder array for 4 players
  const playerPositions = [0, 1, 2, 3];

  // Highlighting is only available in ActionPanel during bidding/trump selection
  
  // State for expandable trick counter
  let showTrickHistory = false;
  
  // Get completed tricks from game state
  $: completedTricks = $gameState.tricks || [];
  
  // Calculate current trick number (completed + 1, unless hand is over)
  $: currentTrickNumber = (() => {
    if ($gamePhase === 'scoring' || $gamePhase === 'bidding') {
      return completedTricks.length; // Show total tricks completed
    }
    return Math.min(completedTricks.length + 1, 7); // Current trick being played
  })();
  

  
  // Extract simple proceed actions that should show in play area
  $: proceedAction = (() => {
    // These actions should always show when available, regardless of turn
    const alwaysShowActions = ['complete-trick', 'score-hand'];
    
    // First check for actual complete/score actions
    const alwaysShowAction = $availableActions.find(action => 
      alwaysShowActions.includes(action.id)
    );
    
    if (alwaysShowAction) {
      return alwaysShowAction;
    }
    
    // Check for consensus actions that the human player (player 0) can take
    const humanConsensusAction = $availableActions.find(action => 
      (action.id === 'agree-complete-trick-0' || action.id === 'agree-score-hand-0')
    );
    
    if (humanConsensusAction) {
      return humanConsensusAction;
    }
    
    // For other actions, only show if it's the human player's turn
    if ($gameState.currentPlayer !== 0) {
      return null;
    }
    
    // Look for other simple proceed actions
    const simpleAction = $availableActions.find(action => 
      action.id === 'start-hand' ||
      action.id === 'continue' ||
      action.id === 'next-trick'
    );
    
    return simpleAction || null;
  })();
  
  // Calculate hand results for scoring phase
  $: handResults = (() => {
    if ($gamePhase !== 'scoring') return null;

    // Get total points for each team from the authoritative teamInfo store
    const team0Points = $teamInfo.scores[0];
    const team1Points = $teamInfo.scores[1];

    // Get bid information
    const bidAmount = $biddingInfo.currentBid.value || 0;
    const biddingTeam = Math.floor($biddingInfo.winningBidder / 2);

    // Determine if bid was made
    const bidMade = biddingTeam === 0 ? team0Points >= bidAmount : team1Points >= bidAmount;

    return {
      team0Points,
      team1Points,
      bidAmount,
      biddingTeam,
      bidMade,
      winningTeam: bidMade ? biddingTeam : (biddingTeam === 0 ? 1 : 0)
    };
  })();
  
  // Debounce flag
  let actionPending = false;
  
  // Track phase for transitions
  let previousPhase = $gamePhase;
  
  // React to phase changes for panel switching
  $: {
    if ($gamePhase === 'bidding' && previousPhase === 'scoring') {
      dispatch('switchToActions');
    }
    previousPhase = $gamePhase;
  }
  
  // Handle action execution
  function handleProceedAction() {
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
    if (proceedAction) {
      handleProceedAction();
    } else {
      // Otherwise, skip AI delays
      gameActions.skipAIDelays();
    }
  }

</script>

<div class="flex flex-col h-full relative">
  <div class="flex gap-2 px-3 py-2 justify-center items-center flex-wrap min-h-[40px]">
    {#if $gameState.trump.type !== 'none'}
      <div class="badge badge-error gap-1 {$gamePhase === 'trump_selection' ? 'animate-pulse' : ''}">
        <span class="text-[10px] uppercase tracking-wider opacity-70">Trump</span>
        <span class="font-bold">{trumpDisplay}</span>
      </div>
    {/if}
    
    {#if ledSuitDisplay}
      <div class="badge badge-info gap-1">
        <span class="text-[10px] uppercase tracking-wider opacity-70">Led</span>
        <span class="font-bold">{ledSuitDisplay}</span>
      </div>
    {/if}
    
    {#if $gamePhase === 'playing' || $gamePhase === 'scoring'}
      <button 
        class="badge badge-secondary gap-1 {completedTricks.length > 0 ? 'cursor-pointer hover:scale-105 transition-transform' : 'cursor-default'}"
        on:click={() => {
          if (completedTricks.length > 0) {
            showTrickHistory = !showTrickHistory;
          }
        }}
        disabled={completedTricks.length === 0}
        type="button"
      >
        <span class="text-[10px] uppercase tracking-wider opacity-70">Trick</span>
        <span class="font-bold">{currentTrickNumber}/7</span>
        {#if completedTricks.length > 0}
          <span class="text-[9px] transition-transform">{showTrickHistory ? '▲' : '▼'}</span>
        {/if}
      </button>
    {/if}
  </div>
  
  {#if showTrickHistory && ($gamePhase === 'playing' || $gamePhase === 'scoring')}
    <div class="bg-base-100 rounded-xl mx-3 p-2 shadow-md border border-base-300" transition:slide={{ duration: 200 }}>
      <!-- Player headers -->
      <div class="flex items-center gap-2 px-2 pb-2 mb-2 border-b-2 border-base-300">
        <span class="min-w-[16px]"></span>
        <div class="flex gap-1 flex-1 items-center">
          <div class="w-10 text-center text-[11px] font-bold text-base-content/70 uppercase">P0</div>
          <div class="w-10 text-center text-[11px] font-bold text-base-content/70 uppercase">P1</div>
          <div class="w-10 text-center text-[11px] font-bold text-base-content/70 uppercase">P2</div>
          <div class="w-10 text-center text-[11px] font-bold text-base-content/70 uppercase">P3</div>
        </div>
        <span class="min-w-[70px]"></span>
      </div>
      
      {#each completedTricks as trick, index}
        {@const sortedPlays = [0, 1, 2, 3].map(playerNum => 
          trick.plays.find(play => play.player === playerNum)
        )}
        <div class="flex items-center gap-2 px-2 py-1.5 rounded-md bg-base-200 mb-1">
          <span class="text-xs font-bold text-base-content/60 min-w-[16px]">{index + 1}:</span>
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
          <span class="text-[11px] font-semibold text-success whitespace-nowrap ml-auto px-1.5 py-0.5 bg-success/10 rounded-full">P{trick.winner}✓ {trick.points || 0}pts</span>
        </div>
      {/each}
      {#if currentTrickPlays.length > 0 && currentTrickPlays.length < 4 && $gamePhase === 'playing'}
        {@const sortedCurrentPlays = [0, 1, 2, 3].map(playerNum => 
          currentTrickPlays.find(play => play.player === playerNum)
        )}
        <div class="flex items-center gap-2 px-2 py-1.5 rounded-md bg-warning/20 border border-warning mb-1">
          <span class="text-xs font-bold text-base-content/60 min-w-[16px]">{currentTrickNumber}:</span>
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
          <span class="text-[11px] font-semibold text-base-content/50 italic whitespace-nowrap ml-auto px-1.5 py-0.5 bg-base-300/50 rounded-full">(in progress)</span>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Bidding table only when waiting during bidding phase -->
  {#if $uiState.showBiddingTable}
    <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-base-100 rounded-xl p-5 shadow-xl min-w-[280px] z-10">
      <h3 class="text-center mb-4 text-lg text-base-content font-semibold">Bidding Round</h3>
      <div class="flex flex-col gap-2 mb-4">
        {#each [0, 1, 2, 3] as playerId}
          {@const bid = $biddingInfo.bids.find(b => b.player === playerId)}
          {@const isCurrentTurn = $gameState.currentPlayer === playerId}
          {@const isAI = controllerManager.isAIControlled(playerId)}
          {@const isYou = playerId === 0 && !testMode}
          
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
      
      {#if $biddingInfo.currentBid.player !== -1}
        <div class="pt-3 border-t border-base-300 text-sm text-base-content/70 flex justify-between">
          <div>Current Bid: {$biddingInfo.currentBid.value} (P{$biddingInfo.currentBid.player})</div>
          <div>Dealer: P{$gameState.dealer}</div>
        </div>
      {:else}
        <div class="pt-3 border-t border-base-300 text-sm text-base-content/70 flex justify-between">
          <div>Opening bid</div>
          <div>Dealer: P{$gameState.dealer}</div>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Trick table -->
  <button
    class="flex-1 flex items-center justify-center p-5 relative transition-all bg-transparent border-none w-full cursor-pointer tap-highlight-transparent touch-manipulation select-none"
    on:click={handleTableClick}
    disabled={false}
    type="button"
    aria-label={proceedAction ? proceedAction.label : "Click to skip AI delays"}
  >
    <div class="relative w-[280px] h-[280px] bg-gradient-to-b from-primary via-primary/80 to-primary/60 rounded-full shadow-[inset_0_0_40px_rgba(0,0,0,0.3),0_10px_30px_rgba(0,0,0,0.2)] flex items-center justify-center transition-all z-[2] {proceedAction ? 'animate-pulse-table' : ''}">
      <div class="absolute inset-5 border-2 border-base-100/10 rounded-full"></div>
      
      {#if handResults}
        <!-- Scoring display -->
        <div class="flex flex-col items-center justify-center gap-6 text-base-100 text-center p-5">
          <div class="flex flex-col gap-2">
            <div class="text-sm opacity-90 font-medium">
              P{$biddingInfo.winningBidder} bid {handResults.bidAmount}
            </div>
            <div class="text-xl font-bold px-4 py-1.5 rounded-full tracking-wider {handResults.bidMade ? 'bg-success/30 text-success-content border-2 border-success' : 'bg-error/30 text-error-content border-2 border-error'}">
              {handResults.bidMade ? 'BID MADE' : 'BID SET'}
            </div>
          </div>
          
          <div class="flex items-center gap-5">
            <div class="flex flex-col items-center gap-1 px-6 py-4 bg-base-100/20 rounded-2xl transition-all {handResults.winningTeam === 0 ? 'bg-base-100/30 scale-110 shadow-[0_0_20px_rgba(255,255,255,0.3)]' : ''}">
              <div class="text-xs font-semibold text-base-100 opacity-90 tracking-widest">US</div>
              <div class="text-3xl font-bold text-base-100 leading-none">{handResults.team0Points}</div>
              <div class="text-[11px] text-base-100 opacity-80">points</div>
            </div>
            
            <div class="text-sm font-semibold text-base-100 opacity-80">vs</div>
            
            <div class="flex flex-col items-center gap-1 px-6 py-4 bg-base-100/20 rounded-2xl transition-all {handResults.winningTeam === 1 ? 'bg-base-100/30 scale-110 shadow-[0_0_20px_rgba(255,255,255,0.3)]' : ''}">
              <div class="text-xs font-semibold text-base-100 opacity-90 tracking-widest">THEM</div>
              <div class="text-3xl font-bold text-base-100 leading-none">{handResults.team1Points}</div>
              <div class="text-[11px] text-base-100 opacity-80">points</div>
            </div>
          </div>
        </div>
      {:else}
        <!-- Normal trick display -->
        {#each playerPositions as position}
          {@const play = currentTrickPlays.find(p => p.player === position)}
          {@const isWinner = trickWinner === position}
          <div class="absolute flex items-center justify-center pointer-events-none" data-position={position} style="{position === 0 ? 'bottom: 20px; left: 50%; transform: translateX(-50%);' : position === 1 ? 'left: 20px; top: 50%; transform: translateY(-50%);' : position === 2 ? 'top: 20px; left: 50%; transform: translateX(-50%);' : 'right: 20px; top: 50%; transform: translateY(-50%);'}">
            {#if play}
              <div class="relative {playedDominoIds.has(`${play.player}-${play.domino.high}-${play.domino.low}`) ? 'animate-drop-in' : ''} {isWinner ? 'animate-winner-glow' : ''}">
                <Domino 
                  domino={play.domino} 
                  small={true}
                  showPoints={true}
                  clickable={false}
                />
                <div class="absolute -bottom-5 left-1/2 -translate-x-1/2 text-[11px] font-bold text-base-100 bg-base-content/70 px-2 py-0.5 rounded-full">P{position}</div>
                {#if isWinner}
                  <div class="absolute -top-6 left-1/2 -translate-x-1/2 bg-gradient-to-br from-yellow-400 to-yellow-500 text-slate-800 px-2.5 py-1 rounded-xl text-[11px] font-bold flex items-center gap-1 shadow-[0_2px_8px_rgba(255,215,0,0.5)] animate-bounce-in whitespace-nowrap z-[15]">
                    <span class="text-sm animate-sparkle">👑</span>
                    <span class="uppercase tracking-wider">Winner!</span>
                  </div>
                {/if}
              </div>
            {:else}
              <div class="relative w-[50px] h-[80px] flex items-center justify-center pointer-events-none">
                <div class="absolute inset-0 border-[3px] border-dashed border-base-100/30 rounded-xl animate-spin-slow"></div>
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
    
    {#if isThinking}
      <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white/95 px-4 py-2 rounded-full shadow-[0_2px_8px_rgba(0,0,0,0.1)] flex items-center gap-2 text-sm text-base-content/70 animate-pulse pointer-events-none z-10">
        <span class="text-lg">🤖</span>
        <span>P{$gameState.currentPlayer} is thinking...</span>
      </div>
    {/if}
    
    {#if proceedAction}
      <div 
        class="absolute top-[calc(50%+140px+25px)] left-1/2 -translate-x-1/2 flex items-center gap-2 px-5 py-3 bg-secondary text-secondary-content rounded-full text-sm font-semibold shadow-[0_4px_12px_rgba(139,92,246,0.3)] z-10 animate-tap-bounce"
        role="presentation"
      >
        <span class="text-lg animate-tap-point">👆</span>
        <span class="whitespace-nowrap">{proceedAction.label}</span>
      </div>
    {/if}
  </button>

  <div class="relative bg-gradient-to-b from-transparent to-base-100/50 pt-5">
    
    {#if playerHand.length === 0}
      <div class="flex flex-col items-center gap-2 py-10 text-base-content/50">
        <span class="text-5xl opacity-50">🀚</span>
        <span class="text-sm font-medium">No dominoes</span>
      </div>
    {:else}
      <div class="px-4 py-5">
        <div class="flex flex-wrap gap-3 px-2 justify-center">
          {#each playerHand as domino, i (domino.high + '-' + domino.low)}
            <div class="animate-hand-slide" style="animation-delay: {i * 50}ms" data-testid="domino-{domino.high}-{domino.low}" data-playable={isDominoPlayable(domino)}>
              <Domino
                {domino}
                playable={isDominoPlayable(domino)}
                clickable={isDominoPlayable(domino) && $gamePhase === 'playing'}
                showPoints={true}
                tooltip={getDominoTooltip(domino)}
                on:click={handleDominoClick}
              />
            </div>
          {/each}
        </div>
      </div>
    {/if}
  </div>
  
</div>

<style>
  /* Custom animations that can't be expressed in Tailwind */
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
  
  .animate-sparkle {
    animation: animate-sparkle 2s ease-in-out infinite;
  }
  
  @keyframes animate-spin-slow {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
  
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
  
  .animate-pulse-table {
    animation: animate-pulse-table 2s ease-in-out infinite;
  }
  
  .tap-highlight-transparent {
    -webkit-tap-highlight-color: transparent;
  }
  
  /* Mobile adjustments for tap indicator position */
  @media (max-width: 640px) {
    .tap-indicator-mobile {
      top: calc(50% + 120px + 20px) !important;
    }
  }
</style>