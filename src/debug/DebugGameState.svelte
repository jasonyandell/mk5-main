<script lang="ts">
  import type { GameState } from '../game/types';
  
  export let gameState: GameState;
  
  function getPhaseColor(phase: string): string {
    const colors: Record<string, string> = {
      'bidding': '#2196F3',
      'trump_selection': '#FF9800',
      'playing': '#4CAF50',
      'scoring': '#9C27B0',
      'setup': '#607D8B',
      'game_end': '#F44336'
    };
    return colors[phase] || '#666';
  }
  
  function getTrumpDisplay(trump: GameState['trump']): string {
    if (trump.type === 'none') return 'None';
    if (trump.type === 'suit' && trump.suit !== undefined) {
      const suitNames = ['Blanks', 'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes'];
      return suitNames[trump.suit];
    }
    if (trump.type === 'doubles') return 'Doubles';
    if (trump.type === 'no-trump') return 'No-Trump';
    return 'Unknown';
  }
  
  function getBidDisplay(bid: GameState['currentBid']): string {
    if (bid.player === -1) return 'None';
    if (bid.type === 'pass') return `P${bid.player}: Pass`;
    if (bid.type === 'points') return `P${bid.player}: ${bid.value}pts`;
    if (bid.type === 'marks') return `P${bid.player}: ${bid.value}m`;
    if (bid.type === 'nello') return `P${bid.player}: Nel-O`;
    if (bid.type === 'splash') return `P${bid.player}: Splash`;
    if (bid.type === 'plunge') return `P${bid.player}: Plunge`;
    return `P${bid.player}: ${bid.type}`;
  }
</script>

<div class="game-state-panel">
  <h3>Game State</h3>
  
  <div class="state-grid">
    <div class="state-item">
      <label>phase</label>
      <span class="value phase" style="color: {getPhaseColor(gameState.phase)}" data-testid="phase">
        {gameState.phase}
      </span>
    </div>
    
    <div class="state-item">
      <label>currentPlayer</label>
      <span class="value" data-testid="current-player">{gameState.currentPlayer}</span>
    </div>
    
    <div class="state-item">
      <label>dealer</label>
      <span class="value" data-testid="dealer">{gameState.dealer}</span>
    </div>
    
    <div class="state-item">
      <label>winningBidder</label>
      <span class="value" data-testid="winning-bidder">
        {gameState.winningBidder === -1 ? 'None' : gameState.winningBidder}
      </span>
    </div>
    
    <div class="state-item">
      <label>trump</label>
      <span class="value trump" data-testid="trump">{getTrumpDisplay(gameState.trump)}</span>
    </div>
    
    <div class="state-item">
      <label>currentSuit</label>
      <span class="value" data-testid="current-suit">
        {gameState.currentSuit === -1 ? 'None' : gameState.currentSuit}
      </span>
    </div>
    
    <div class="state-item">
      <label>currentBid</label>
      <span class="value bid" data-testid="current-bid">{getBidDisplay(gameState.currentBid)}</span>
    </div>
    
    <div class="state-item">
      <label>teamScores</label>
      <span class="value scores" data-testid="team-scores">
        <span data-testid="team-0-score">{gameState.teamScores[0]}</span> - 
        <span data-testid="team-1-score">{gameState.teamScores[1]}</span>
      </span>
    </div>
    
    <div class="state-item">
      <label>teamMarks</label>
      <span class="value marks" data-testid="team-marks">
        {gameState.teamMarks[0]} - {gameState.teamMarks[1]}
      </span>
    </div>
    
    <div class="state-item">
      <label>gameTarget</label>
      <span class="value" data-testid="game-target">{gameState.gameTarget}</span>
    </div>
    
    <div class="state-item">
      <label>tournamentMode</label>
      <span class="value" data-testid="tournament-mode">{gameState.tournamentMode}</span>
    </div>
    
    <div class="state-item">
      <label>shuffleSeed</label>
      <span class="value mono" data-testid="shuffle-seed">{gameState.shuffleSeed}</span>
    </div>
  </div>
  
  {#if gameState.bids.length > 0}
    <div class="bidding-history">
      <h4>Bidding History</h4>
      <div class="bid-list">
        {#each gameState.bids as bid, i}
          <span class="bid-item" data-testid="bid-{i}">
            {getBidDisplay(bid)}
          </span>
        {/each}
      </div>
    </div>
  {/if}
  
  <div class="progress-info">
    <span>Trick {gameState.tricks.length + (gameState.currentTrick.length > 0 ? 1 : 0)}/7</span>
    <span>•</span>
    <span>Actions: {gameState.tricks.length * 7 + gameState.currentTrick.length}</span>
  </div>
</div>

<style>
  .game-state-panel {
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 20px;
  }
  
  .game-state-panel h3 {
    margin: 0 0 15px 0;
    color: #4CAF50;
    font-size: 18px;
  }
  
  .state-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
    margin-bottom: 20px;
  }
  
  .state-item {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  
  .state-item label {
    font-size: 12px;
    color: #888;
    font-family: monospace;
  }
  
  .state-item .value {
    font-size: 16px;
    font-weight: 500;
    color: #fff;
  }
  
  .value.phase {
    text-transform: uppercase;
    font-weight: bold;
  }
  
  .value.trump {
    color: #FF9800;
  }
  
  .value.bid {
    color: #2196F3;
  }
  
  .value.scores, .value.marks {
    font-family: monospace;
    font-size: 18px;
  }
  
  .value.mono {
    font-family: monospace;
    font-size: 14px;
  }
  
  .bidding-history {
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid #444;
  }
  
  .bidding-history h4 {
    margin: 0 0 10px 0;
    color: #2196F3;
    font-size: 14px;
  }
  
  .bid-list {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
  }
  
  .bid-item {
    padding: 4px 8px;
    background: #333;
    border-radius: 4px;
    font-size: 14px;
    color: #aaa;
  }
  
  .progress-info {
    margin-top: 15px;
    padding-top: 15px;
    border-top: 1px solid #444;
    display: flex;
    gap: 10px;
    font-size: 14px;
    color: #888;
  }
</style>