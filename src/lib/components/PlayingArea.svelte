<script lang="ts">
  import { gameState, availableActions, gameActions, currentPlayer, gamePhase, teamInfo, biddingInfo, playerView, uiState } from '../../stores/gameStore';
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
    
    {#if $gamePhase === 'playing' || $gamePhase === 'scoring'}
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
  
  {#if showTrickHistory && ($gamePhase === 'playing' || $gamePhase === 'scoring')}
    <div class="trick-history" transition:slide={{ duration: 200 }}>
      <!-- Player headers -->
      <div class="history-header">
        <span class="trick-num"></span>
        <div class="trick-dominoes-row">
          <div class="player-header">P0</div>
          <div class="player-header">P1</div>
          <div class="player-header">P2</div>
          <div class="player-header">P3</div>
        </div>
        <span class="trick-result-header"></span>
      </div>
      
      {#each completedTricks as trick, index}
        {@const sortedPlays = [0, 1, 2, 3].map(playerNum => 
          trick.plays.find(play => play.player === playerNum)
        )}
        <div class="history-row">
          <span class="trick-num">{index + 1}:</span>
          <div class="trick-dominoes-row">
            {#each sortedPlays as play}
              {#if play}
                <div class="history-domino-wrapper" class:winner={play.player === trick.winner}>
                  <Domino 
                    domino={play.domino} 
                    small={true}
                    showPoints={false}
                    clickable={false}
                  />
                </div>
              {:else}
                <div class="history-domino-placeholder"></div>
              {/if}
            {/each}
          </div>
          <span class="trick-result">P{trick.winner}✓ {trick.points || 0}pts</span>
        </div>
      {/each}
      {#if currentTrickPlays.length > 0 && currentTrickPlays.length < 4 && $gamePhase === 'playing'}
        {@const sortedCurrentPlays = [0, 1, 2, 3].map(playerNum => 
          currentTrickPlays.find(play => play.player === playerNum)
        )}
        <div class="history-row current">
          <span class="trick-num">{currentTrickNumber}:</span>
          <div class="trick-dominoes-row">
            {#each sortedCurrentPlays as play}
              {#if play}
                <div class="history-domino-wrapper">
                  <Domino 
                    domino={play.domino} 
                    small={true}
                    showPoints={false}
                    clickable={false}
                  />
                </div>
              {:else}
                <div class="history-domino-placeholder"></div>
              {/if}
            {/each}
          </div>
          <span class="trick-result in-progress">(in progress)</span>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Bidding table only when waiting during bidding phase -->
  {#if $uiState.showBiddingTable}
    <div class="bidding-table">
      <h3 class="bidding-title">Bidding Round</h3>
      <div class="bidding-status">
        {#each [0, 1, 2, 3] as playerId}
          {@const bid = $biddingInfo.bids.find(b => b.player === playerId)}
          {@const isCurrentTurn = $gameState.currentPlayer === playerId}
          {@const isAI = controllerManager.isAIControlled(playerId)}
          {@const isYou = playerId === 0 && !testMode}
          
          <div class="bid-row" class:current-turn={isCurrentTurn} class:you={isYou}>
            <span class="player-label">
              <span class="player-icon">{isAI ? '🤖' : '👤'}</span>
              P{playerId}{isYou ? ' (You)' : ''}:
            </span>
            <span class="bid-value">
              {#if bid}
                {#if bid.type === 'pass'}
                  <span class="bid-pass">Pass</span>
                {:else}
                  <span class="bid-points">{bid.value} {bid.type}</span>
                {/if}
              {:else if isCurrentTurn}
                {#if isAI}
                  <span class="thinking">Thinking...</span>
                {:else}
                  <span class="waiting">Your turn...</span>
                {/if}
              {:else}
                <span class="pending">Waiting...</span>
              {/if}
            </span>
          </div>
        {/each}
      </div>
      
      {#if $biddingInfo.currentBid.player !== -1}
        <div class="current-bid-info">
          <div>Current Bid: {$biddingInfo.currentBid.value} (P{$biddingInfo.currentBid.player})</div>
          <div>Dealer: P{$gameState.dealer}</div>
        </div>
      {:else}
        <div class="current-bid-info">
          <div>Opening bid</div>
          <div>Dealer: P{$gameState.dealer}</div>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Trick table -->
  <button
    class="trick-table" 
    class:tappable={true}
    on:click={handleTableClick}
    disabled={false}
    type="button"
    aria-label={proceedAction ? proceedAction.label : "Click to skip AI delays"}
  >
    <div class="table-surface" class:glowing={proceedAction}>
      <div class="table-pattern"></div>
      
      {#if handResults}
        <!-- Scoring display -->
        <div class="scoring-display">
          <div class="bid-summary">
            <div class="bid-info-line">
              P{$biddingInfo.winningBidder} bid {handResults.bidAmount}
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
          {@const isWinner = trickWinner === position}
          <div class="trick-spot" data-position={position} style="pointer-events: none;">
            {#if play}
              <div class="played-domino" 
                   class:fresh={playedDominoIds.has(`${play.player}-${play.domino.high}-${play.domino.low}`)}
                   class:winner={isWinner}>
                <Domino 
                  domino={play.domino} 
                  small={true}
                  showPoints={true}
                  clickable={false}
                />
                <div class="player-indicator">P{position}</div>
                {#if isWinner}
                  <div class="winner-badge">
                    <span class="winner-icon">👑</span>
                    <span class="winner-text">Winner!</span>
                  </div>
                {/if}
              </div>
            {:else}
              <div class="waiting-spot">
                <div class="spot-ring"></div>
                <span class="spot-icon">
                  {controllerManager.isAIControlled(position) ? '🤖' : '👤'}
                </span>
                <span class="spot-player">P{position}</span>
              </div>
            {/if}
          </div>
        {/each}
      {/if}
    </div>
    
    {#if isThinking}
      <div class="thinking-indicator ai-thinking-pulse">
        <span class="robot-icon">🤖</span>
        <span class="thinking-text">P{$gameState.currentPlayer} is thinking...</span>
      </div>
    {/if}
    
    {#if proceedAction}
      <div 
        class="tap-indicator"
        role="presentation"
      >
        <span class="tap-icon">👆</span>
        <span class="tap-text">{proceedAction.label}</span>
      </div>
    {/if}
  </button>

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
            <div class="domino-wrapper" style="--delay: {i * 50}ms" data-testid="domino-{domino.high}-{domino.low}" data-playable={isDominoPlayable(domino)}>
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
    left: 20px;
    top: 50%;
    transform: translateY(-50%);
  }

  .trick-spot[data-position="2"] {
    top: 20px;
    left: 50%;
    transform: translateX(-50%);
  }

  .trick-spot[data-position="3"] {
    right: 20px;
    top: 50%;
    transform: translateY(-50%);
  }

  .played-domino {
    position: relative;
  }

  .played-domino.fresh {
    animation: dropIn 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  }
  
  .played-domino.winner {
    animation: winnerGlow 2s ease-in-out infinite;
  }
  
  @keyframes winnerGlow {
    0%, 100% {
      filter: drop-shadow(0 0 8px rgba(255, 215, 0, 0.6));
      transform: scale(1);
    }
    50% {
      filter: drop-shadow(0 0 20px rgba(255, 215, 0, 0.9));
      transform: scale(1.05);
    }
  }
  
  .winner-badge {
    position: absolute;
    top: -25px;
    left: 50%;
    transform: translateX(-50%);
    background: linear-gradient(135deg, #ffd700, #ffed4e);
    color: #1e293b;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 4px;
    box-shadow: 0 2px 8px rgba(255, 215, 0, 0.5);
    animation: bounceIn 0.2s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    white-space: nowrap;
    z-index: 15;
  }
  
  .winner-icon {
    font-size: 14px;
    animation: sparkle 2s ease-in-out infinite;
  }
  
  .winner-text {
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  @keyframes bounceIn {
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
  
  @keyframes sparkle {
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

  .thinking-indicator {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(255, 255, 255, 0.95);
    padding: 8px 16px;
    border-radius: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    color: #666;
    animation: pulse 1.5s ease-in-out infinite;
    pointer-events: none;
    z-index: 10;
  }

  @keyframes pulse {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
  }

  .robot-icon {
    font-size: 18px;
  }

  .spot-icon {
    font-size: 12px;
    opacity: 0.7;
    margin-right: 2px;
  }

  /* Bidding table styles */
  .bidding-table {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    min-width: 280px;
    z-index: 5;
  }

  .bidding-title {
    text-align: center;
    margin: 0 0 15px 0;
    font-size: 18px;
    color: #333;
    font-weight: 600;
  }

  .bidding-status {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 15px;
  }

  .bid-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 12px;
    border-radius: 6px;
    background: #f8f8f8;
    transition: all 0.2s;
  }

  .bid-row.current-turn {
    background: #e8f4ff;
    border: 1px solid #4a90e2;
  }

  .bid-row.you {
    font-weight: 600;
    background: #f0f8ff;
  }

  .player-label {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .player-icon {
    font-size: 16px;
  }

  .bid-value {
    font-weight: 500;
  }

  .bid-pass {
    color: #999;
    font-style: italic;
  }

  .bid-points {
    color: #2c5aa0;
    font-weight: bold;
  }

  .bid-value .thinking {
    color: #f39c12;
    animation: pulse 1.5s ease-in-out infinite;
  }

  .bid-value .waiting {
    color: #4a90e2;
  }

  .bid-value .pending {
    color: #999;
    font-style: italic;
  }

  .current-bid-info {
    padding-top: 12px;
    border-top: 1px solid #e0e0e0;
    font-size: 14px;
    color: #666;
    display: flex;
    justify-content: space-between;
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
    padding: 20px 16px;
  }

  .hand-dominoes {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    padding: 0 8px;
    justify-content: center;
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
    padding: 12px 20px;
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
    animation-play-state: paused;
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
  
  .trick-result.in-progress {
    background: rgba(148, 163, 184, 0.1);
    color: #94a3b8;
    font-style: italic;
  }
  
  .history-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 8px;
    margin-bottom: 6px;
    border-bottom: 2px solid #e2e8f0;
  }
  
  .player-header {
    width: 40px;
    text-align: center;
    font-size: 11px;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
  }
  
  .history-domino-placeholder {
    width: 40px;
    height: 60px;
    display: inline-flex;
  }
  
  .trick-result-header {
    min-width: 70px;
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