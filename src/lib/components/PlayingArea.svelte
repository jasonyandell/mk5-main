<script lang="ts">
  import { gameState, availableActions, gameActions, currentPlayer, gamePhase, teamInfo, biddingInfo } from '../../stores/gameStore';
  import Domino from './Domino.svelte';
  import type { Domino as DominoType } from '../../game/types';
  import { slide } from 'svelte/transition';
  import { createEventDispatcher } from 'svelte';
  
  const dispatch = createEventDispatcher();

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
      return `${dominoStr} - Waiting for Player ${$gameState.currentPlayer}'s turn`;
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

  // Current player's hand
  $: playerHand = $currentPlayer?.hand || [];

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
    
    const alwaysShowAction = $availableActions.find(action => 
      alwaysShowActions.includes(action.id)
    );
    
    if (alwaysShowAction) {
      return alwaysShowAction;
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
    
    // Get total points for each team from won tricks
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
  
  // Handle action execution
  function handleProceedAction() {
    if (!proceedAction || actionPending) return;
    
    // Set debounce flag
    actionPending = true;
    
    // Execute the action
    gameActions.executeAction(proceedAction);
    
    // If we just scored a hand, transition to Actions panel for bidding
    if (proceedAction.id === 'score-hand') {
      // Small delay to let the state update
      setTimeout(() => {
        dispatch('switchToActions');
      }, 100);
    }
    
    // Clear debounce after a short delay
    setTimeout(() => {
      actionPending = false;
    }, 300);
  }

</script>

<div class="playing-area">
  <div class="game-info-strip">
    {#if $gameState.trump.type !== 'none'}
      <div class="info-badge trump" class:pulse={$gamePhase === 'trump_selection'}>
        <span class="info-label">Trump</span>
        <span class="info-value">{trumpDisplay}</span>
      </div>
    {/if}
    
    {#if ledSuitDisplay}
      <div class="info-badge led">
        <span class="info-label">Led</span>
        <span class="info-value">{ledSuitDisplay}</span>
      </div>
    {/if}
    
    {#if $gamePhase === 'playing'}
      <button 
        class="trick-counter"
        class:expandable={completedTricks.length > 0}
        on:click={() => {
          if (completedTricks.length > 0) {
            showTrickHistory = !showTrickHistory;
          }
        }}
        disabled={completedTricks.length === 0}
        type="button"
      >
        <span class="trick-label">Trick</span>
        <span class="trick-number">{currentTrickNumber}/7</span>
        {#if completedTricks.length > 0}
          <span class="expand-arrow">{showTrickHistory ? '▲' : '▼'}</span>
        {/if}
      </button>
    {/if}
  </div>
  
  {#if showTrickHistory && $gamePhase === 'playing'}
    <div class="trick-history" transition:slide={{ duration: 200 }}>
      {#each completedTricks as trick, index}
        <div class="history-row">
          <span class="trick-num">{index + 1}:</span>
          <div class="trick-dominoes-row">
            {#each trick.plays as play}
              <div class="history-domino-wrapper" class:winner={play.player === trick.winner}>
                <Domino 
                  domino={play.domino} 
                  small={true}
                  showPoints={false}
                  clickable={false}
                />
              </div>
            {/each}
          </div>
          <span class="trick-result">P{trick.winner}✓ {trick.points || 0}pts</span>
        </div>
      {/each}
      {#if currentTrickPlays.length > 0 && currentTrickPlays.length < 4}
        <div class="history-row current">
          <span class="trick-num">{currentTrickNumber}:</span>
          <div class="trick-dominoes-row">
            {#each currentTrickPlays as play}
              <div class="history-domino-wrapper">
                <Domino 
                  domino={play.domino} 
                  small={true}
                  showPoints={false}
                  clickable={false}
                />
              </div>
            {/each}
            <span class="in-progress">(in progress)</span>
          </div>
        </div>
      {/if}
    </div>
  {/if}

  <div
    class="trick-table" 
    class:tappable={proceedAction}
    role="region"
    aria-label="Trick table"
  >
    <div class="table-surface" class:glowing={proceedAction}>
      <div class="table-pattern"></div>
      
      {#if handResults}
        <!-- Scoring display -->
        <div class="scoring-display">
          <div class="bid-summary">
            <div class="bid-info-line">
              Player {$biddingInfo.winningBidder + 1} bid {handResults.bidAmount}
            </div>
            <div class="bid-result" class:made={handResults.bidMade} class:set={!handResults.bidMade}>
              {handResults.bidMade ? 'BID MADE' : 'BID SET'}
            </div>
          </div>
          
          <div class="score-breakdown">
            <div class="team-score" class:winner={handResults.winningTeam === 0}>
              <div class="team-label">US</div>
              <div class="score-value">{handResults.team0Points}</div>
              <div class="score-label">points</div>
            </div>
            
            <div class="vs-divider">vs</div>
            
            <div class="team-score" class:winner={handResults.winningTeam === 1}>
              <div class="team-label">THEM</div>
              <div class="score-value">{handResults.team1Points}</div>
              <div class="score-label">points</div>
            </div>
          </div>
        </div>
      {:else}
        <!-- Normal trick display -->
        {#each playerPositions as position}
          {@const play = currentTrickPlays.find(p => p.player === position)}
          <div class="trick-spot" data-position={position}>
            {#if play}
              <div class="played-domino" class:fresh={playedDominoIds.has(`${play.player}-${play.domino.high}-${play.domino.low}`)}>
                <Domino 
                  domino={play.domino} 
                  small={true}
                  showPoints={true}
                  clickable={false}
                />
                <div class="player-indicator">P{position}</div>
              </div>
            {:else}
              <div class="waiting-spot">
                <div class="spot-ring"></div>
                <span class="spot-player">P{position}</span>
              </div>
            {/if}
          </div>
        {/each}
      {/if}
    </div>
    
    {#if proceedAction}
      <button 
        class="tap-indicator" 
        on:click={handleProceedAction}
        type="button"
        aria-label={proceedAction.label}
      >
        <span class="tap-icon">👆</span>
        <span class="tap-text">{proceedAction.label}</span>
      </button>
    {/if}
  </div>

  <div class="hand-container">
    
    {#if playerHand.length === 0}
      <div class="empty-hand">
        <span class="empty-icon">🀚</span>
        <span class="empty-text">No dominoes</span>
      </div>
    {:else}
      <div class="hand-scroll">
        <div class="hand-dominoes">
          {#each playerHand as domino, i (domino.high + '-' + domino.low)}
            <div class="domino-wrapper" style="--delay: {i * 50}ms">
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
  .playing-area {
    display: flex;
    flex-direction: column;
    height: 100%;
    position: relative;
  }

  .game-info-strip {
    display: flex;
    gap: 8px;
    padding: 8px 12px;
    justify-content: center;
    align-items: center;
    flex-wrap: wrap;
    min-height: 40px;
  }

  .info-badge {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 16px;
    font-size: 12px;
    font-weight: 600;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1.5px solid;
  }

  .info-badge.trump {
    border-color: #dc2626;
    background: rgba(220, 38, 38, 0.08);
  }

  .info-badge.led {
    border-color: #3b82f6;
    background: rgba(59, 130, 246, 0.08);
  }

  .info-badge.pulse {
    animation: infoPulse 1.5s ease-in-out infinite;
  }

  @keyframes infoPulse {
    0%, 100% { 
      transform: scale(1); 
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06); 
    }
    50% { 
      transform: scale(1.08); 
      box-shadow: 0 4px 12px rgba(220, 38, 38, 0.25); 
    }
  }

  .info-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    opacity: 0.7;
  }

  .info-value {
    font-size: 13px;
    font-weight: 700;
  }

  .info-badge.trump .info-label,
  .info-badge.trump .info-value {
    color: #dc2626;
  }

  .info-badge.led .info-label,
  .info-badge.led .info-value {
    color: #3b82f6;
  }

  .trick-counter {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    background: rgba(139, 92, 246, 0.08);
    border: 1.5px solid #8b5cf6;
    border-radius: 16px;
    font-size: 12px;
    font-weight: 600;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
    transition: all 0.2s ease;
    cursor: default;
  }
  
  .trick-counter.expandable {
    cursor: pointer;
  }
  
  .trick-counter.expandable:hover {
    background: rgba(139, 92, 246, 0.12);
    transform: translateY(-1px);
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.08);
  }
  
  .trick-counter:disabled {
    cursor: default;
  }

  .trick-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    opacity: 0.7;
    color: #8b5cf6;
  }

  .trick-number {
    font-size: 13px;
    font-weight: 700;
    color: #8b5cf6;
  }
  
  .expand-arrow {
    font-size: 9px;
    margin-left: 2px;
    color: #8b5cf6;
    transition: transform 0.2s ease;
  }

  .trick-table {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    position: relative;
    transition: all 0.3s ease;
    background: none;
    border: none;
    width: 100%;
    font: inherit;
  }
  
  .trick-table:disabled {
    cursor: default;
  }
  
  .trick-table.tappable {
    cursor: pointer;
    -webkit-tap-highlight-color: transparent;
    touch-action: manipulation;
    user-select: none;
  }

  .table-surface {
    position: relative;
    width: 280px;
    height: 280px;
    background: radial-gradient(ellipse at center, #10b981 0%, #059669 70%, #047857 100%);
    border-radius: 50%;
    box-shadow: 
      inset 0 0 40px rgba(0, 0, 0, 0.3),
      0 10px 30px rgba(0, 0, 0, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s ease;
    z-index: 2;
  }
  
  .table-surface.glowing {
    animation: tablePulse 2s ease-in-out infinite;
  }
  
  @keyframes tablePulse {
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
  
  .trick-table.tappable:hover .table-surface {
    animation-play-state: paused;
    transform: scale(1.03);
    box-shadow: 
      inset 0 0 40px rgba(0, 0, 0, 0.3),
      0 10px 30px rgba(0, 0, 0, 0.2),
      0 0 40px 15px rgba(139, 92, 246, 0.4);
  }
  
  .trick-table.tappable:active .table-surface {
    transform: scale(0.98);
  }

  .table-pattern {
    position: absolute;
    inset: 20px;
    border: 2px solid rgba(255, 255, 255, 0.1);
    border-radius: 50%;
  }

  .trick-spot {
    position: absolute;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .trick-spot[data-position="0"] {
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
  }

  .trick-spot[data-position="1"] {
    right: 20px;
    top: 50%;
    transform: translateY(-50%);
  }

  .trick-spot[data-position="2"] {
    top: 20px;
    left: 50%;
    transform: translateX(-50%);
  }

  .trick-spot[data-position="3"] {
    left: 20px;
    top: 50%;
    transform: translateY(-50%);
  }

  .played-domino {
    position: relative;
  }

  .played-domino.fresh {
    animation: dropIn 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  }

  @keyframes dropIn {
    from {
      transform: translateY(-100px) scale(0.8);
      opacity: 0;
    }
    to {
      transform: translateY(0) scale(1);
      opacity: 1;
    }
  }

  .player-indicator {
    position: absolute;
    bottom: -20px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 11px;
    font-weight: 700;
    color: white;
    background: rgba(0, 0, 0, 0.7);
    padding: 2px 8px;
    border-radius: 10px;
  }

  .waiting-spot {
    position: relative;
    width: 50px;
    height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: none; /* Don't block table clicks */
  }

  .spot-ring {
    position: absolute;
    inset: 0;
    border: 3px dashed rgba(255, 255, 255, 0.3);
    border-radius: 12px;
    animation: rotate 20s linear infinite;
  }

  @keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .spot-player {
    font-size: 14px;
    font-weight: 700;
    color: rgba(255, 255, 255, 0.6);
  }

  .hand-container {
    position: relative;
    background: linear-gradient(to bottom, transparent, rgba(255, 255, 255, 0.5));
    padding-top: 20px;
  }


  .empty-hand {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 40px;
    color: #94a3b8;
  }

  .empty-icon {
    font-size: 48px;
    opacity: 0.5;
  }

  .empty-text {
    font-size: 14px;
    font-weight: 500;
  }

  /* Mobile optimizations */
  @media (max-width: 640px) {
    .game-info-strip {
      padding: 6px 8px;
      gap: 6px;
      min-height: 36px;
    }

    .info-badge {
      padding: 3px 8px;
      font-size: 11px;
    }

    .info-label {
      font-size: 9px;
    }

    .info-value {
      font-size: 12px;
    }

    .trick-counter {
      padding: 3px 8px;
    }

    .trick-label {
      font-size: 9px;
    }

    .trick-number {
      font-size: 12px;
    }

    .table-surface {
      width: 240px;
      height: 240px;
    }
  }

  .hand-scroll {
    overflow-x: auto;
    overflow-y: hidden;
    -webkit-overflow-scrolling: touch;
    padding: 20px 16px;
    mask-image: linear-gradient(to right, 
      transparent 0%, 
      black 16px, 
      black calc(100% - 16px), 
      transparent 100%);
    -webkit-mask-image: linear-gradient(to right, 
      transparent 0%, 
      black 16px, 
      black calc(100% - 16px), 
      transparent 100%);
  }

  .hand-dominoes {
    display: flex;
    gap: 12px;
    padding: 0 8px;
    min-width: min-content;
  }

  .domino-wrapper {
    animation: handSlide 0.5s cubic-bezier(0.4, 0, 0.2, 1) both;
    animation-delay: var(--delay);
  }

  @keyframes handSlide {
    from {
      opacity: 0;
      transform: translateY(30px) rotate(-10deg);
    }
    to {
      opacity: 1;
      transform: translateY(0) rotate(0);
    }
  }
  
  
  /* Ensure button appears above trick table */
  .trick-table {
    position: relative;
  }
  
  /* Tap indicator */
  .tap-indicator {
    position: absolute;
    top: calc(50% + 140px + 25px); /* Half table height (140px) + half button height (~25px) */
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 20px;
    background: rgba(139, 92, 246, 0.95);
    color: white;
    border: none;
    border-radius: 24px;
    font-size: 14px;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
    cursor: pointer;
    z-index: 10;
    animation: tapBounce 1.5s ease-in-out infinite;
  }
  
  .tap-indicator:hover {
    background: rgba(139, 92, 246, 1);
    transform: translateX(-50%) scale(1.05);
  }
  
  .tap-indicator:active {
    transform: translateX(-50%) scale(0.95);
  }
  
  @keyframes tapBounce {
    0%, 100% {
      transform: translateX(-50%) translateY(0);
    }
    50% {
      transform: translateX(-50%) translateY(-5px);
    }
  }
  
  .tap-icon {
    font-size: 18px;
    animation: tapPoint 1.5s ease-in-out infinite;
  }
  
  @keyframes tapPoint {
    0%, 100% {
      transform: translateY(0);
    }
    50% {
      transform: translateY(-3px);
    }
  }
  
  .tap-text {
    white-space: nowrap;
  }
  
  /* Mobile adjustments */
  @media (max-width: 640px) {
    .table-surface {
      width: 240px;
      height: 240px;
    }
    
    .tap-indicator {
      top: calc(50% + 120px + 20px); /* Half of smaller table (120px) + half button */
      padding: 6px 16px;
      font-size: 13px;
    }
    
    .tap-icon {
      font-size: 16px;
    }
  }
  
  /* Scoring display styles */
  .scoring-display {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 24px;
    color: white;
    text-align: center;
    padding: 20px;
  }
  
  .bid-summary {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .bid-info-line {
    font-size: 14px;
    opacity: 0.9;
    font-weight: 500;
  }
  
  .bid-result {
    font-size: 20px;
    font-weight: 700;
    padding: 6px 16px;
    border-radius: 20px;
    letter-spacing: 0.05em;
  }
  
  .bid-result.made {
    background: rgba(34, 197, 94, 0.2);
    color: #86efac;
    border: 2px solid #86efac;
  }
  
  .bid-result.set {
    background: rgba(239, 68, 68, 0.2);
    color: #fca5a5;
    border: 2px solid #fca5a5;
  }
  
  .score-breakdown {
    display: flex;
    align-items: center;
    gap: 20px;
  }
  
  .team-score {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 16px 24px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    transition: all 0.3s ease;
  }
  
  .team-score.winner {
    background: rgba(255, 255, 255, 0.2);
    transform: scale(1.1);
    box-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
  }
  
  .team-score .team-label {
    font-size: 12px;
    font-weight: 600;
    opacity: 0.8;
    letter-spacing: 0.1em;
  }
  
  .team-score .score-value {
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
  }
  
  .team-score .score-label {
    font-size: 11px;
    opacity: 0.7;
  }
  
  .vs-divider {
    font-size: 14px;
    font-weight: 600;
    opacity: 0.6;
  }
  
  /* Mobile adjustments for scoring */
  @media (max-width: 640px) {
    .scoring-display {
      gap: 16px;
      padding: 16px;
    }
    
    .bid-result {
      font-size: 16px;
    }
    
    .score-breakdown {
      gap: 12px;
    }
    
    .team-score {
      padding: 12px 16px;
    }
    
    .team-score .score-value {
      font-size: 24px;
    }
  }
  
  /* Trick History Styles */
  .trick-history {
    background: white;
    border-radius: 12px;
    margin: 8px 12px;
    padding: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    border: 1px solid #e2e8f0;
  }
  
  .history-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 8px;
    border-radius: 6px;
    margin-bottom: 4px;
    background: #f8fafc;
  }
  
  .history-row:last-child {
    margin-bottom: 0;
  }
  
  .history-row.current {
    background: #fef3c7;
    border: 1px solid #fbbf24;
  }
  
  .trick-num {
    font-size: 12px;
    font-weight: 700;
    color: #64748b;
    min-width: 16px;
  }
  
  .trick-dominoes-row {
    display: flex;
    gap: 4px;
    flex: 1;
    align-items: center;
  }
  
  .history-domino-wrapper {
    display: inline-flex;
    transition: all 0.2s ease;
  }
  
  .history-domino-wrapper.winner {
    transform: scale(1.1);
    filter: drop-shadow(0 0 4px rgba(16, 185, 129, 0.5));
  }
  
  .trick-result {
    font-size: 11px;
    font-weight: 600;
    color: #10b981;
    white-space: nowrap;
    margin-left: auto;
    padding: 2px 6px;
    background: rgba(16, 185, 129, 0.1);
    border-radius: 10px;
  }
  
  .in-progress {
    font-size: 11px;
    color: #94a3b8;
    font-style: italic;
    margin-left: 8px;
  }
  
  /* Mobile optimizations for history */
  @media (max-width: 640px) {
    .trick-history {
      margin: 6px 8px;
      padding: 6px;
    }
    
    .history-row {
      padding: 4px 6px;
      gap: 6px;
    }
    
    .trick-num {
      font-size: 11px;
    }
    
    .trick-result {
      font-size: 10px;
      padding: 2px 4px;
    }
  }
</style>