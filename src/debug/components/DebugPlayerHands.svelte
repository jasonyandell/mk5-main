<script lang="ts">
  import type { GameState, Play } from '../../game/types';
  import { getPlayerLeftOfDealer, getPlayersInOrder } from '../../game/core/players';
  
  interface Props {
    gameState: GameState;
  }
  
  let { gameState }: Props = $props();
  
  function renderDomino(high: number, low: number): string {
    return `[${high}|${low}]`;
  }
  
  function getCurrentTrickPlays(): Record<number, string> {
    const plays: Record<number, string> = {};
    gameState.currentTrick.forEach(play => {
      plays[play.player] = renderDomino(play.domino.high, play.domino.low);
    });
    return plays;
  }
  
  // Get players arranged for 2x2 grid display: 0-1 top row, 3-2 bottom row
  function getPlayersForTableDisplay(): Array<typeof gameState.players[0] | null> {
    const positions = [null, null, null, null]; // [top-left, top-right, bottom-left, bottom-right]
    
    // Map to grid positions for layout: 0 1 / 3 2
    positions[0] = gameState.players[0]; // Player 0 at top-left
    positions[1] = gameState.players[1]; // Player 1 at top-right  
    positions[2] = gameState.players[3]; // Player 3 at bottom-left (position 2)
    positions[3] = gameState.players[2]; // Player 2 at bottom-right (position 3)
    
    return positions;
  }
  
  const currentTrickPlays = $derived(getCurrentTrickPlays());
  const playersForDisplay = $derived(getPlayersForTableDisplay());
</script>

<div class="debug-player-hands" data-testid="player-hands">
  <div class="hands-header">
    <h3>Player Hands</h3>
  </div>
  
  <div class="players-table">
    {#each playersForDisplay as player, position}
      {#if player}
        <div 
          class="player-section position-{position}"
          class:current-player={player.id === gameState.currentPlayer}
          class:dealer={player.id === gameState.dealer}
          class:winning-bidder={player.id === gameState.winningBidder}
        >
          <div class="player-header">
            <div class="player-name">
              P{player.id}
              {#if player.id === gameState.currentPlayer}
                <span class="badge current">TURN</span>
              {/if}
              {#if player.id === gameState.dealer}
                <span class="badge dealer">DEALER</span>
              {/if}
              {#if player.id === gameState.winningBidder}
                <span class="badge winner">BIDDER</span>
              {/if}
            </div>
            <div class="team-info">
              Team {player.teamId + 1} • {player.hand.length} cards
            </div>
          </div>
          
          <div class="hand-dominoes">
            {#each player.hand as domino}
              <div class="domino-mini">
                {renderDomino(domino.high, domino.low)}
              </div>
            {/each}
            {#if player.hand.length === 0}
              <div class="no-cards">No cards</div>
            {/if}
          </div>
          
          {#if currentTrickPlays[player.id]}
            <div class="current-play">
              Played: {currentTrickPlays[player.id]}
            </div>
          {/if}
        </div>
      {/if}
    {/each}
  </div>
  
  {#if gameState.currentTrick.length > 0}
    <div class="current-trick-info">
      <h4>Current Trick ({gameState.currentTrick.length}/4)</h4>
      <div class="trick-plays">
        {#each gameState.currentTrick as play}
          <div class="trick-play">
            <span class="play-player">P{play.player}:</span>
            <span class="play-domino">{renderDomino(play.domino.high, play.domino.low)}</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .debug-player-hands {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 8px;
    font-size: 10px;
    display: flex;
    flex-direction: column;
  }
  
  .hands-header h3 {
    margin: 0 0 8px 0;
    font-size: 12px;
    font-weight: 600;
    color: #212529;
  }
  
  .players-table {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 8px;
    margin-bottom: 8px;
  }
  
  .position-0 { /* Player 0 - Top Left */
    grid-column: 1;
    grid-row: 1;
  }
  
  .position-1 { /* Player 1 - Top Right */
    grid-column: 2;
    grid-row: 1;
  }
  
  .position-2 { /* Player 3 - Bottom Left (position 2 in array) */
    grid-column: 1;
    grid-row: 2;
  }
  
  .position-3 { /* Player 2 - Bottom Right (position 3 in array) */
    grid-column: 2;
    grid-row: 2;
  }
  
  .player-section {
    border: 1px solid #e9ecef;
    border-radius: 3px;
    padding: 8px;
    background: #f8f9fa;
    transition: all 0.1s ease;
  }
  
  .player-section.current-player {
    border-color: #007bff;
    background: #e3f2fd;
  }
  
  .player-section.dealer {
    border-left: 3px solid #28a745;
  }
  
  .player-section.winning-bidder {
    border-right: 3px solid #fd7e14;
  }
  
  .player-header {
    margin-bottom: 8px;
  }
  
  .player-name {
    font-weight: 600;
    color: #212529;
    display: flex;
    align-items: center;
    gap: 4px;
    flex-wrap: wrap;
  }
  
  .badge {
    font-size: 8px;
    padding: 1px 4px;
    border-radius: 2px;
    font-weight: 600;
    text-transform: uppercase;
  }
  
  .badge.current {
    background: #007bff;
    color: white;
  }
  
  .badge.dealer {
    background: #28a745;
    color: white;
  }
  
  .badge.winner {
    background: #fd7e14;
    color: white;
  }
  
  .team-info {
    font-size: 9px;
    color: #6c757d;
    margin-top: 2px;
  }
  
  .hand-dominoes {
    display: flex;
    flex-wrap: wrap;
    gap: 2px;
    margin-bottom: 4px;
  }
  
  .domino-mini {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 2px;
    padding: 1px 3px;
    font-family: monospace;
    font-size: 9px;
    color: #495057;
    min-width: 18px;
    text-align: center;
  }
  
  .no-cards {
    color: #6c757d;
    font-style: italic;
    font-size: 9px;
  }
  
  .current-play {
    font-size: 9px;
    color: #28a745;
    font-weight: 500;
    margin-top: 4px;
  }
  
  .current-trick-info {
    margin-top: 12px;
    padding-top: 8px;
    border-top: 1px solid #e9ecef;
  }
  
  .current-trick-info h4 {
    margin: 0 0 6px 0;
    font-size: 12px;
    font-weight: 600;
    color: #495057;
  }
  
  .trick-plays {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  
  .trick-play {
    display: flex;
    gap: 4px;
    align-items: center;
  }
  
  .play-player {
    color: #6c757d;
    font-size: 10px;
  }
  
  .play-domino {
    font-family: monospace;
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 2px;
    padding: 1px 4px;
    font-size: 9px;
    color: #495057;
  }
</style>