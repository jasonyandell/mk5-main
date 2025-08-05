<script lang="ts">
  import { gameState, availableActions, gameActions, currentPlayer, gamePhase } from '../../stores/gameStore';
  import Domino from './Domino.svelte';
  import type { Domino as DominoType } from '../../game/types';

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
  $: playerHand = $currentPlayer.hand || [];

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

  // State for hovering during bidding/trump selection
  let hoveredSuit: number | 'doubles' | null = null;
  let hoveredDomino: DominoType | null = null;

  // Check if we should enable suit highlighting
  $: enableSuitHighlighting = $gamePhase === 'bidding' || $gamePhase === 'trump_selection';
  

  // Handle domino hover - now with specific half detection
  function handleDominoHover(domino: DominoType, event: Event | MouseEvent | null, isEntering: boolean) {
    if (!enableSuitHighlighting) {
      hoveredSuit = null;
      return;
    }

    if (!isEntering || !event) {
      hoveredSuit = null;
      hoveredDomino = null;
      return;
    }

    // Store which domino we're hovering over
    hoveredDomino = domino;

    // For doubles, always highlight both doubles and the suit
    if (domino.high === domino.low) {
      hoveredSuit = domino.high; // Will highlight both the suit AND doubles
      return;
    }

    // For non-doubles, determine which half based on mouse position
    const mouseEvent = event as MouseEvent;
    const target = mouseEvent.currentTarget as HTMLElement;
    const rect = target.getBoundingClientRect();
    const relativeY = mouseEvent.clientY - rect.top;
    const halfwayPoint = rect.height / 2;

    // The domino display has high pip on top, low pip on bottom
    // So if mouse is in top half, we want the high value
    const isTopHalf = relativeY < halfwayPoint;
    hoveredSuit = isTopHalf ? domino.high : domino.low;
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
      <div class="trick-counter">
        <span class="trick-label">Trick</span>
        <span class="trick-number">{$gameState.trickNumber || 1}/7</span>
      </div>
    {/if}
  </div>

  <div class="trick-table">
    <div class="table-surface">
      <div class="table-pattern"></div>
      
      {#each playerPositions as position}
        {@const play = currentTrickPlays.find(p => p.player === position)}
        <div class="trick-spot" data-position={position}>
          {#if play}
            <div class="played-domino" class:fresh={playedDominoIds.has(`${play.player}-${play.domino.high}-${play.domino.low}`)}>
              <Domino 
                domino={play.domino} 
                small={true}
                showPoints={true}
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
    </div>
  </div>

  <div class="hand-container">
    {#if hoveredSuit !== null && enableSuitHighlighting}
      <div class="suit-highlight-indicator">
        {(() => {
          const suitName = ['Blanks', 'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes'][hoveredSuit];
          return suitName;
        })()}
      </div>
    {/if}
    
    {#if playerHand.length === 0}
      <div class="empty-hand">
        <span class="empty-icon">🀚</span>
        <span class="empty-text">No dominoes</span>
      </div>
    {:else}
      <div class="hand-scroll">
        <div class="hand-dominoes">
          {#each playerHand as domino, i (domino.high + '-' + domino.low)}
            {@const highlight = (() => {
              if (hoveredSuit === null) return null;
              
              const isHoveringDouble = hoveredDomino && hoveredDomino.high === hoveredDomino.low;
              
              // Always prioritize suit highlighting
              if (domino.high === hoveredSuit || domino.low === hoveredSuit) {
                return 'primary';
              }
              
              // If hovering a double, other doubles get secondary highlight
              if (isHoveringDouble && domino.high === domino.low) {
                return 'secondary';
              }
              return null;
            })()}
            <div class="domino-wrapper" style="--delay: {i * 50}ms">
              <Domino
                {domino}
                playable={isDominoPlayable(domino)}
                clickable={true}
                showPoints={true}
                {highlight}
                tooltip={getDominoTooltip(domino)}
                on:click={handleDominoClick}
                on:mouseenter={(e) => handleDominoHover(domino, e, true)}
                on:mousemove={(e) => handleDominoHover(domino, e, true)}
                on:mouseleave={(e) => handleDominoHover(domino, e, false)}
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

  .trick-table {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    position: relative;
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

  .suit-highlight-indicator {
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    padding: 4px 16px;
    background: linear-gradient(135deg, #fbbf24, #f59e0b);
    color: white;
    font-size: 12px;
    font-weight: 700;
    border-radius: 0 0 16px 16px;
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
    animation: slideDown 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }

  @keyframes slideDown {
    from { transform: translateX(-50%) translateY(-100%); }
    to { transform: translateX(-50%) translateY(0); }
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
</style>