<script lang="ts">
  import type { GameState, Play } from '../../game/types';
  
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
  
  // Get players in clockwise order starting from dealer+1 (first to bid/play)
  function getPlayersInClockwiseOrder(): typeof gameState.players {
    const startPlayer = (gameState.dealer + 1) % 4;
    const orderedPlayers = [];
    
    for (let i = 0; i < 4; i++) {
      const playerIndex = (startPlayer + i) % 4;
      orderedPlayers.push(gameState.players[playerIndex]);
    }
    
    return orderedPlayers;
  }
  
  const currentTrickPlays = $derived(getCurrentTrickPlays());
  const playersInOrder = $derived(getPlayersInClockwiseOrder());
</script>

<div class="debug-player-hands" data-testid="player-hands">
  <div class="hands-header">
    <h3>Player Hands</h3>
  </div>
  
  <div class="players-grid">
    {#each playersInOrder as player}
      <div 
        class="player-section"
        class:current-player={player.id === gameState.currentPlayer}
        class:dealer={player.id === gameState.dealer}
        class:winning-bidder={player.id === gameState.winningBidder}
      >
        <div class="player-header">
          <div class="player-name">
            P{player.id + 1}
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
    {/each}
  </div>
  
  {#if gameState.currentTrick.length > 0}
    <div class="current-trick-info">
      <h4>Current Trick ({gameState.currentTrick.length}/4)</h4>
      <div class="trick-plays">
        {#each gameState.currentTrick as play}
          <div class="trick-play">
            <span class="play-player">P{play.player + 1}:</span>
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
  
  .players-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
    margin-bottom: 8px;
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