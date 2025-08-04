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
  function handleDominoHover(domino: DominoType, event: MouseEvent | null, isEntering: boolean) {
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
    const target = event.currentTarget as HTMLElement;
    const rect = target.getBoundingClientRect();
    const relativeY = event.clientY - rect.top;
    const halfwayPoint = rect.height / 2;

    // The domino display has high pip on top, low pip on bottom
    // So if mouse is in top half, we want the high value
    const isTopHalf = relativeY < halfwayPoint;
    hoveredSuit = isTopHalf ? domino.high : domino.low;
  }

  // Check if a domino should be highlighted
  function shouldHighlight(domino: DominoType): 'primary' | 'secondary' | null {
    if (hoveredSuit === null) return null;

    const isHoveringDouble = hoveredDomino && hoveredDomino.high === hoveredDomino.low;
    

    if (isHoveringDouble) {
      // When hovering over a double, highlight:
      // - All doubles as primary
      // - All dominoes with that suit value as secondary
      if (domino.high === domino.low) {
        return 'primary'; // This is a double
      } else if (domino.high === hoveredSuit || domino.low === hoveredSuit) {
        return 'secondary'; // This has the suit value
      }
    } else {
      // When hovering over a non-double, just highlight dominoes with that suit
      if (domino.high === hoveredSuit || domino.low === hoveredSuit) {
        return 'primary';
      }
    }

    return null;
  }
</script>

<div class="playing-area">
  <div class="trump-display">
    <div class="trump-box">
      <span class="trump-label">TRUMP:</span>
      <span class="trump-value">{trumpDisplay}</span>
    </div>
    {#if ledSuitDisplay}
      <div class="led-suit-box">
        <span class="led-label">Led:</span>
        <span class="led-value">{ledSuitDisplay}</span>
      </div>
    {/if}
  </div>

  <div class="current-trick-area">
    <h3>Current Trick</h3>
    <div class="trick-grid">
      {#each playerPositions as position}
        {@const play = currentTrickPlays.find(p => p.player === position)}
        <div class="trick-position" data-player={position}>
          <div class="player-label">P{position}</div>
          {#if play}
            <div class="domino-play-animation">
              <Domino 
                domino={play.domino} 
                small={true}
                showPoints={true}
              />
            </div>
          {:else}
            <div class="empty-slot">???</div>
          {/if}
        </div>
      {/each}
    </div>
  </div>

  <div class="player-hand-area">
    <h3>
      Your Hand (P{$currentPlayer.id})
      {#if hoveredSuit !== null && enableSuitHighlighting}
        <span class="suit-indicator">
          - Highlighting: {(() => {
            const suitName = ['Blanks', 'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes'][hoveredSuit];
            const isHoveringOverDouble = hoveredDomino && hoveredDomino.high === hoveredDomino.low;
            return isHoveringOverDouble ? `Doubles & ${suitName}` : suitName;
          })()}
        </span>
      {/if}
    </h3>
    {#if playerHand.length === 0}
      <div class="empty-hand">No dominoes in hand</div>
    {:else}
      <div class="hand-dominoes">
        {#each playerHand as domino (domino.high + '-' + domino.low)}
          {#key hoveredSuit}
            <Domino
              {domino}
              playable={isDominoPlayable(domino)}
              clickable={true}
              showPoints={true}
              highlight={shouldHighlight(domino)}
              tooltip={getDominoTooltip(domino)}
              on:click={handleDominoClick}
              on:mouseenter={(e) => handleDominoHover(domino, e, true)}
              on:mousemove={(e) => handleDominoHover(domino, e, true)}
              on:mouseleave={(e) => handleDominoHover(domino, e, false)}
            />
          {/key}
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .playing-area {
    display: flex;
    flex-direction: column;
    gap: 24px;
    height: 100%;
  }

  .trump-display {
    display: flex;
    gap: 16px;
    justify-content: center;
  }

  .trump-box,
  .led-suit-box {
    padding: 12px 24px;
    background-color: #f3f4f6;
    border: 2px solid #e5e7eb;
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .trump-label,
  .led-label {
    font-weight: 600;
    color: #6b7280;
  }

  .trump-value {
    font-weight: 700;
    color: #dc2626;
    font-size: 18px;
  }

  .led-value {
    font-weight: 700;
    color: #3b82f6;
    font-size: 18px;
  }

  .current-trick-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  h3 {
    margin: 0 0 16px 0;
    font-size: 16px;
    font-weight: 600;
    color: #374151;
  }

  .trick-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    grid-template-rows: repeat(2, 1fr);
    gap: 20px;
    padding: 20px;
    background-color: #f9fafb;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
  }

  .trick-position {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }

  .trick-position[data-player="0"] {
    grid-column: 1;
    grid-row: 1;
  }

  .trick-position[data-player="1"] {
    grid-column: 2;
    grid-row: 1;
  }

  .trick-position[data-player="2"] {
    grid-column: 2;
    grid-row: 2;
  }

  .trick-position[data-player="3"] {
    grid-column: 1;
    grid-row: 2;
  }

  .player-label {
    font-size: 14px;
    font-weight: 600;
    color: #6b7280;
  }

  .empty-slot {
    width: 50px;
    height: 80px;
    border: 2px dashed #d1d5db;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #9ca3af;
    font-weight: 600;
  }

  .player-hand-area {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-bottom: 20px;
  }

  .empty-hand {
    padding: 40px;
    color: #9ca3af;
    font-style: italic;
  }

  .hand-dominoes {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    justify-content: center;
  }

  .suit-indicator {
    font-size: 14px;
    font-weight: normal;
    color: #f59e0b;
    margin-left: 8px;
  }
</style>