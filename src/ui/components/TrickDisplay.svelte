<script lang="ts">
  import type { Play, Trick } from '../../game/types';
  import DominoCard from './DominoCard.svelte';
  
  interface Props {
    currentTrick: Play[];
    completedTricks: Trick[];
    trump: number | null;
  }
  
  let { currentTrick, completedTricks, trump }: Props = $props();
  
  function getPlayerName(playerId: number): string {
    return `Player ${playerId + 1}`;
  }
  
  function getTrumpName(trump: number | null): string {
    if (trump === null) return 'None';
    const suits = ['Blanks', 'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes'];
    return suits[trump] || 'Unknown';
  }
</script>

<div class="trick-display">
  <div class="trump-info">
    <h3>Trump: {getTrumpName(trump)}</h3>
  </div>
  
  <div class="current-trick">
    <h3>Current Trick</h3>
    <div class="trick-plays">
      {#each currentTrick as play (play.player)}
        <div class="play">
          <div class="player-label">{getPlayerName(play.player)}</div>
          <DominoCard domino={play.domino} />
        </div>
      {/each}
      
      {#if currentTrick.length === 0}
        <div class="no-plays">No plays yet</div>
      {/if}
    </div>
  </div>
  
  <div class="completed-tricks">
    <h3>Completed Tricks ({completedTricks.length}/7)</h3>
    <div class="tricks-summary">
      {#each completedTricks as trick, index}
        <div class="trick-summary">
          <div class="trick-header">
            <span class="trick-number">#{index + 1}</span>
            <span class="trick-winner">
              Won by {getPlayerName(trick.winner || 0)}
            </span>
            <span class="trick-points">{trick.points} pts</span>
          </div>
          <div class="trick-dominoes">
            {#each trick.plays as play}
              <div class="mini-domino" title="{getPlayerName(play.player)}: {play.domino.high}-{play.domino.low}">
                {play.domino.high}-{play.domino.low}
              </div>
            {/each}
          </div>
        </div>
      {/each}
      
      {#if completedTricks.length === 0}
        <div class="no-tricks">No tricks completed yet</div>
      {/if}
    </div>
  </div>
</div>

<style>
  .trick-display {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    contain: layout style;
  }
  
  .trick-display h3 {
    margin: 0 0 12px 0;
    color: #333;
    font-size: 16px;
  }
  
  .trump-info {
    background: #fff3e0;
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 20px;
    border-left: 4px solid #ff9800;
  }
  
  .trump-info h3 {
    margin: 0;
    color: #e65100;
  }
  
  .current-trick {
    margin-bottom: 24px;
  }
  
  .trick-plays {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    min-height: 140px;
    align-items: flex-start;
  }
  
  .play {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }
  
  .player-label {
    font-size: 12px;
    font-weight: 500;
    color: #666;
    text-align: center;
  }
  
  .no-plays, .no-tricks {
    color: #999;
    font-style: italic;
    padding: 20px;
    text-align: center;
  }
  
  .tricks-summary {
    max-height: 200px;
    overflow-y: auto;
  }
  
  .trick-summary {
    border: 1px solid #eee;
    border-radius: 4px;
    padding: 12px;
    margin-bottom: 8px;
    background: #fafafa;
  }
  
  .trick-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    font-size: 14px;
  }
  
  .trick-number {
    font-weight: bold;
    color: #666;
  }
  
  .trick-winner {
    flex: 1;
    text-align: center;
    font-weight: 500;
  }
  
  .trick-points {
    font-weight: bold;
    color: #4caf50;
  }
  
  .trick-dominoes {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }
  
  .mini-domino {
    background: white;
    border: 1px solid #ddd;
    border-radius: 2px;
    padding: 2px 4px;
    font-size: 10px;
    font-family: monospace;
    color: #666;
  }
  
  @media (max-width: 768px) {
    .trick-plays {
      gap: 8px;
    }
    
    .trick-header {
      flex-direction: column;
      gap: 4px;
      text-align: center;
    }
  }
</style>