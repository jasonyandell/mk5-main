<script lang="ts">
  import type { Player } from '../game/types';
  
  export let players: Player[];
  export let currentPlayer: number;
  
  function getDominoDisplay(domino: { high: number; low: number }): string {
    return `${domino.high}-${domino.low}`;
  }
  
  function getDominoPoints(domino: { high: number; low: number }): number {
    const total = domino.high + domino.low;
    if (total === 10) return 10;
    if (total === 5) return 5;
    return 0;
  }
  
  function isDouble(domino: { high: number; low: number }): boolean {
    return domino.high === domino.low;
  }
  
  function getSuitCounts(player: Player): Record<string, number> {
    if (!player.suitAnalysis) return {};
    return {
      'Blanks': player.suitAnalysis.count[0],
      'Ones': player.suitAnalysis.count[1],
      'Twos': player.suitAnalysis.count[2],
      'Threes': player.suitAnalysis.count[3],
      'Fours': player.suitAnalysis.count[4],
      'Fives': player.suitAnalysis.count[5],
      'Sixes': player.suitAnalysis.count[6],
      'Doubles': player.suitAnalysis.count.doubles,
      'Trump': player.suitAnalysis.count.trump
    };
  }
  
  function getTeamName(teamId: number): string {
    return `Team ${teamId}`;
  }
</script>

<div class="player-hands-panel">
  <h3>Player Hands</h3>
  
  <div class="players-grid">
    {#each players as player}
      <div 
        class="player-card"
        class:current={player.id === currentPlayer}
        data-testid="player-{player.id}-hand"
      >
        <div class="player-header">
          <div class="player-info">
            <span class="player-name">{player.name}</span>
            <span class="player-id">P{player.id}</span>
          </div>
          <div class="player-team">{getTeamName(player.teamId)}</div>
        </div>
        
        <div class="hand-dominoes">
          {#each player.hand as domino, i}
            <div 
              class="domino-tile"
              class:counting={getDominoPoints(domino) > 0}
              class:double={isDouble(domino)}
              data-testid="player-{player.id}-domino-{i}"
              title={`${getDominoDisplay(domino)} (${getDominoPoints(domino)} pts)`}
            >
              {getDominoDisplay(domino)}
            </div>
          {/each}
        </div>
        
        <div class="hand-stats">
          <div class="stat">
            <label>Count</label>
            <span>{player.hand.reduce((sum, d) => sum + getDominoPoints(d), 0)} pts</span>
          </div>
          <div class="stat">
            <label>Doubles</label>
            <span>{player.hand.filter(isDouble).length}</span>
          </div>
        </div>
        
        {#if player.suitAnalysis}
          <div class="suit-analysis">
            <h5>Suit Analysis</h5>
            <div class="suit-counts">
              {#each Object.entries(getSuitCounts(player)) as [suit, count]}
                {#if count > 0}
                  <span class="suit-count" data-testid="player-{player.id}-{suit.toLowerCase()}-count">
                    {suit}: {count}
                  </span>
                {/if}
              {/each}
            </div>
          </div>
        {/if}
      </div>
    {/each}
  </div>
</div>

<style>
  .player-hands-panel {
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 20px;
    height: 100%;
  }
  
  .player-hands-panel h3 {
    margin: 0 0 15px 0;
    color: #2196F3;
    font-size: 18px;
  }
  
  .players-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
  }
  
  .player-card {
    background: #333;
    border: 1px solid #555;
    border-radius: 6px;
    padding: 15px;
    transition: all 0.3s;
  }
  
  .player-card.current {
    border-color: #FFD700;
    box-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
  }
  
  .player-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #444;
  }
  
  .player-info {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  
  .player-name {
    font-weight: bold;
    color: #fff;
    font-size: 16px;
  }
  
  .player-id {
    font-family: monospace;
    color: #888;
    font-size: 14px;
  }
  
  .player-team {
    color: #4CAF50;
    font-size: 14px;
  }
  
  .hand-dominoes {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(50px, 1fr));
    gap: 8px;
    margin-bottom: 15px;
  }
  
  .domino-tile {
    background: #444;
    border: 1px solid #666;
    border-radius: 4px;
    padding: 8px;
    text-align: center;
    font-family: monospace;
    font-size: 14px;
    color: #fff;
    cursor: default;
    transition: all 0.2s;
  }
  
  .domino-tile:hover {
    background: #555;
    transform: translateY(-2px);
  }
  
  .domino-tile.counting {
    border-color: #FFD700;
    color: #FFD700;
  }
  
  .domino-tile.double {
    background: #4a4a4a;
    font-weight: bold;
  }
  
  .hand-stats {
    display: flex;
    gap: 20px;
    margin-bottom: 15px;
    padding: 10px;
    background: #3a3a3a;
    border-radius: 4px;
  }
  
  .stat {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  
  .stat label {
    font-size: 12px;
    color: #888;
  }
  
  .stat span {
    font-size: 16px;
    color: #fff;
    font-weight: 500;
  }
  
  .suit-analysis {
    background: #3a3a3a;
    padding: 10px;
    border-radius: 4px;
  }
  
  .suit-analysis h5 {
    margin: 0 0 8px 0;
    color: #888;
    font-size: 12px;
    text-transform: uppercase;
  }
  
  .suit-counts {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  
  .suit-count {
    padding: 3px 8px;
    background: #444;
    border-radius: 3px;
    font-size: 12px;
    color: #aaa;
  }
</style>