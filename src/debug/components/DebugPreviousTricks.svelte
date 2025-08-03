<script lang="ts">
  import type { GameState, Trick, Domino } from '../../game/types';
  import { getCurrentSuit } from '../../game/core/rules';
  
  interface Props {
    gameState: GameState;
  }
  
  let { gameState }: Props = $props();
  
  function getDominoDisplay(domino: Domino): string {
    return `${domino.high}-${domino.low}`;
  }
  
  function getTrickSuit(trick: Trick): string {
    if (trick.plays.length === 0 || gameState.trump === null) return '';
    
    // For completed tricks, compute the suit from the first play
    const leadDomino = trick.plays[0].domino;
    const trump = gameState.trump;
    
    
    // Determine what suit was led
    let ledSuit: number;
    if (trump === 7) { // doubles are trump
      ledSuit = leadDomino.high === leadDomino.low ? 7 : Math.max(leadDomino.high, leadDomino.low);
    } else if (leadDomino.high === trump || leadDomino.low === trump) {
      ledSuit = trump; // trump was led
    } else {
      ledSuit = Math.max(leadDomino.high, leadDomino.low); // higher end for non-trump
    }
    
    // Format for display with brackets like [6s]
    const suitFormats: Record<number, string> = {
      0: '[0s]',
      1: '[1s]', 
      2: '[2s]',
      3: '[3s]',
      4: '[4s]',
      5: '[5s]',
      6: '[6s]',
      7: '[doubles]'
    };
    
    return suitFormats[ledSuit] || '[?]';
  }
  
  function getTrumpDisplay(): string {
    if (gameState.trump === null) return '';
    
    // Map trump values to display strings
    switch (gameState.trump) {
      case 0: return '[0s]';
      case 1: return '[1s]';
      case 2: return '[2s]';
      case 3: return '[3s]';
      case 4: return '[4s]';
      case 5: return '[5s]';
      case 6: return '[6s]';
      case 7: return '[doubles]';
      default: return '[?]';
    }
  }
  
  function getCurrentSuitBracketFormat(state: GameState): string {
    if (state.currentSuit === null) return '[?]';
    
    const suitFormats: Record<number, string> = {
      0: '[0s]',
      1: '[1s]', 
      2: '[2s]',
      3: '[3s]',
      4: '[4s]',
      5: '[5s]',
      6: '[6s]',
      7: '[doubles]',
      8: '[no-trump]'
    };
    
    return suitFormats[state.currentSuit] || '[?]';
  }
</script>

<div class="previous-tricks-container">
  <div class="section-header">
    <h3>Previous Tricks ({gameState.tricks.length}/7)</h3>
    {#if gameState.currentTrick.length > 0}
      <div class="current-indicator">Current: {gameState.currentTrick.length}/4</div>
    {/if}
  </div>
  
  {#if gameState.tricks.length === 0 && gameState.currentTrick.length === 0}
    <div class="no-tricks">No tricks yet</div>
  {:else}
    <div class="tricks-grid">
      <!-- Previous completed tricks -->
      {#each gameState.tricks as trick, index}
        <div class="trick-compact">
          <div class="trick-label">#{index + 1}</div>
          <div class="trick-dominoes">
            {#each trick.plays as play}
              <div class="domino-compact" 
                   class:winner={play.player === trick.winner}
                   class:counter={play.domino.points && play.domino.points > 0}
                                      title="P{play.player}: {play.domino.high}-{play.domino.low}{play.domino.points && play.domino.points > 0 ? ` (${play.domino.points}pts)` : ''}">
                <span class="player-num">P{play.player}</span>
                <span class="domino-value" >{getDominoDisplay(play.domino)}</span>
              </div>
            {/each}
          </div>
          <div class="trick-info">
            <div class="trick-result">
              <span class="trump-display">{getTrickSuit(trick)}</span>
              <span class="winner-info">P{trick.winner || 0}</span>
              <span class="points-info">{trick.points + 1}pt</span>
            </div>
          </div>
        </div>
      {/each}
      
      <!-- Current trick in progress -->
      {#if gameState.currentTrick.length > 0}
        <div class="trick-compact current-trick">
          <div class="trick-label">#{gameState.tricks.length + 1}</div>
          <div class="trick-dominoes">
            {#each gameState.currentTrick as play}
              <div class="domino-compact current"
                                      title="P{play.player}: {play.domino.high}-{play.domino.low}{play.domino.points && play.domino.points > 0 ? ` (${play.domino.points}pts)` : ''}">
                <span class="player-num">P{play.player}</span>
                <span class="domino-value" >{getDominoDisplay(play.domino)}</span>
              </div>
            {/each}
            <!-- Empty slots for remaining plays -->
            {#each Array(4 - gameState.currentTrick.length) as _, i}
              <div class="domino-compact empty">
                <span class="empty-slot">—</span>
              </div>
            {/each}
          </div>
          <div class="trick-info">
            <div class="trick-result">
              <span class="trump-display">{gameState.currentSuit !== null ? getCurrentSuitBracketFormat(gameState) : '[...]'}</span>
            </div>
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .previous-tricks-container {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 8px;
    margin: 4px 0;
  }
  
  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }
  
  .section-header h3 {
    margin: 0;
    font-size: 12px;
    color: #495057;
    font-weight: 600;
  }
  
  .current-indicator {
    background: #ffc107;
    color: #856404;
    padding: 1px 4px;
    border-radius: 2px;
    font-size: 10px;
    font-weight: 500;
  }
  
  .no-tricks {
    text-align: center;
    color: #6c757d;
    font-size: 11px;
    padding: 8px;
  }
  
  .tricks-grid {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  
  .trick-compact {
    background: white;
    border: 1px solid #e9ecef;
    border-radius: 3px;
    padding: 4px;
    display: flex;
    align-items: center;
    gap: 4px;
    min-height: auto;
  }
  
  .trick-compact.current-trick {
    background: #fff3cd;
    border-color: #ffc107;
  }
  
  .trick-label {
    font-size: 10px;
    font-weight: 600;
    color: #6c757d;
    min-width: 20px;
    text-align: center;
  }
  
  .trick-dominoes {
    display: flex;
    gap: 2px;
    flex: 1;
  }
  
  .domino-compact {
    background: #e9ecef;
    border: 1px solid #ced4da;
    border-radius: 2px;
    padding: 2px 3px;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 32px;
    font-size: 9px;
    line-height: 1;
  }
  
  .domino-compact.winner {
    background: #d4edda;
    border-color: #28a745;
    box-shadow: 0 0 2px rgba(40, 167, 69, 0.3);
  }
  
  .domino-compact.counter {
    background: #fff3cd;
    border-color: #ffc107;
  }
  
  .domino-compact.current {
    background: #ffeaa7;
    border-color: #fdcb6e;
  }
  
  .domino-compact.empty {
    background: #f8f9fa;
    border: 1px dashed #ced4da;
    opacity: 0.5;
  }
  
  .player-num {
    font-size: 8px;
    color: #6c757d;
    font-weight: 500;
  }
  
  .domino-value {
    font-family: monospace;
    font-size: 9px;
    font-weight: 600;
    color: #212529;
  }
  
  
  .empty-slot {
    font-size: 10px;
    color: #adb5bd;
    align-self: center;
  }
  
  .trick-info {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1px;
    min-width: 30px;
  }
  
  
  .winner-info {
    font-size: 9px;
    font-weight: 600;
    color: #28a745;
  }
  
  .points-info {
    font-size: 8px;
    background: #28a745;
    color: white;
    padding: 1px 3px;
    border-radius: 2px;
    line-height: 1;
  }
  
  
  
  .trick-result {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  
  .trump-display {
    background: #212529;
    color: #fff;
    border: 1px solid #495057;
    border-radius: 3px;
    padding: 2px 4px;
    font-size: 8px;
    font-weight: 600;
    font-family: monospace;
    text-align: center;
    line-height: 1;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    min-width: 24px;
  }
</style>