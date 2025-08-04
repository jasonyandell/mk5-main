<script lang="ts">
  import type { Trick } from '../game/types';
  
  export let tricks: Trick[];
  
  function getDominoDisplay(domino: { high: number; low: number }): string {
    return `${domino.high}-${domino.low}`;
  }
  
  function isCountingDomino(domino: { high: number; low: number }): boolean {
    const total = domino.high + domino.low;
    return total === 10 || total === 5;
  }
  
  function getSuitName(suit: number): string {
    const suitNames = ['Blanks', 'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes'];
    return suitNames[suit] || 'Unknown';
  }
</script>

<div class="tricks-panel">
  <h3>Previous Tricks ({tricks.length}/7)</h3>
  
  {#if tricks.length === 0}
    <div class="no-tricks">No tricks played yet</div>
  {:else}
    <div class="tricks-grid">
      {#each tricks as trick, trickIndex}
        <div class="trick-card" data-testid="trick-{trickIndex}">
          <div class="trick-header">
            <span class="trick-number">Trick {trickIndex + 1}</span>
            <span class="trick-points">{trick.points} pts</span>
          </div>
          
          <div class="trick-plays">
            {#each trick.plays as play, playIndex}
              <div 
                class="play-item"
                class:winner={play.player === trick.winner}
                data-testid="trick-{trickIndex}-play-{playIndex}"
              >
                <span class="player-id">P{play.player}</span>
                <span 
                  class="domino"
                  class:counting={isCountingDomino(play.domino)}
                >
                  {getDominoDisplay(play.domino)}
                </span>
                {#if playIndex === 0 && trick.ledSuit !== undefined}
                  <span class="led-suit" title="Led suit">{getSuitName(trick.ledSuit)}</span>
                {/if}
              </div>
            {/each}
          </div>
          
          <div class="trick-footer">
            <span class="winner-tag">Winner: P{trick.winner}</span>
          </div>
        </div>
      {/each}
    </div>
  {/if}
  
  <div class="tricks-summary">
    <div class="summary-item">
      <label>Total Points Captured</label>
      <span>{tricks.reduce((sum, t) => sum + t.points, 0)}</span>
    </div>
    <div class="summary-item">
      <label>Tricks Won by Team</label>
      <span>
        T0: {tricks.filter(t => t.winner === 0 || t.winner === 2).length} | 
        T1: {tricks.filter(t => t.winner === 1 || t.winner === 3).length}
      </span>
    </div>
  </div>
</div>

<style>
  .tricks-panel {
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 20px;
  }
  
  .tricks-panel h3 {
    margin: 0 0 15px 0;
    color: #9C27B0;
    font-size: 18px;
  }
  
  .no-tricks {
    color: #666;
    font-style: italic;
    text-align: center;
    padding: 20px;
  }
  
  .tricks-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 15px;
    max-height: 400px;
    overflow-y: auto;
    padding-right: 10px;
  }
  
  .trick-card {
    background: #333;
    border: 1px solid #555;
    border-radius: 6px;
    padding: 10px;
    font-size: 14px;
  }
  
  .trick-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid #444;
  }
  
  .trick-number {
    color: #aaa;
    font-weight: bold;
  }
  
  .trick-points {
    color: #4CAF50;
    font-weight: bold;
  }
  
  .trick-plays {
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin-bottom: 8px;
  }
  
  .play-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 5px;
    border-radius: 3px;
    transition: background 0.2s;
  }
  
  .play-item.winner {
    background: #1B5E20;
  }
  
  .player-id {
    font-family: monospace;
    color: #888;
    width: 25px;
  }
  
  .domino {
    font-family: monospace;
    color: #fff;
  }
  
  .domino.counting {
    color: #FFD700;
    font-weight: bold;
  }
  
  .led-suit {
    font-size: 11px;
    color: #2196F3;
    margin-left: auto;
  }
  
  .trick-footer {
    padding-top: 8px;
    border-top: 1px solid #444;
  }
  
  .winner-tag {
    font-size: 12px;
    color: #4CAF50;
  }
  
  .tricks-summary {
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid #444;
    display: flex;
    gap: 30px;
  }
  
  .summary-item {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  
  .summary-item label {
    font-size: 12px;
    color: #888;
  }
  
  .summary-item span {
    font-size: 16px;
    color: #fff;
    font-weight: 500;
  }
  
  ::-webkit-scrollbar {
    width: 8px;
  }
  
  ::-webkit-scrollbar-track {
    background: #1a1a1a;
    border-radius: 4px;
  }
  
  ::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 4px;
  }
  
  ::-webkit-scrollbar-thumb:hover {
    background: #666;
  }
</style>