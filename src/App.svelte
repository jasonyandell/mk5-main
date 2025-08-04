<script lang="ts">
  import { onMount } from 'svelte';
  import DebugUI from './DebugUI.svelte';
  import { gameState, availableActions, gameActions } from './stores/gameStore';
  
  onMount(() => {
    console.log('Texas 42 Debug Mode Loaded');
  });
  
  $: currentPlayer = $gameState.players[$gameState.currentPlayer];
  
  // Extract playable dominoes from available actions
  $: playableDominoes = (() => {
    const dominoes = new Set();
    $availableActions
      .filter(action => action.id.startsWith('play-'))
      .forEach(action => {
        // Extract domino ID from action id like "play-5-3" or "play-6-6"
        const dominoId = action.id.replace('play-', '');
        dominoes.add(dominoId);
        // Also add reversed version in case the domino is stored differently
        const parts = dominoId.split('-');
        if (parts.length === 2) {
          dominoes.add(`${parts[1]}-${parts[0]}`);
        }
      });
    return dominoes;
  })();
  
  // Debug logging
  $: {
    console.log('Game phase:', $gameState.phase);
    console.log('Current player:', $gameState.currentPlayer);
    console.log('Available actions:', $availableActions.map(a => ({ id: a.id, label: a.label })));
    console.log('Playable dominoes set:', Array.from(playableDominoes));
    console.log('Current hand:', currentPlayer.hand.map(d => getDominoDisplay(d)));
  }
  
  function getDominoDisplay(domino: { high: number; low: number }): string {
    return `${domino.high}-${domino.low}`;
  }
  
  function getDominoPoints(domino: { high: number; low: number }): number {
    const total = domino.high + domino.low;
    if (total === 10) return 10;
    if (total === 5) return 5;
    return 0;
  }
  
  function getTrumpDisplay(trump: typeof $gameState.trump): string {
    if (trump.type === 'none') return 'None';
    if (trump.type === 'suit' && trump.suit !== undefined) {
      const suitNames = ['Blanks', 'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes'];
      return suitNames[trump.suit];
    }
    if (trump.type === 'doubles') return 'Doubles';
    if (trump.type === 'no-trump') return 'No-Trump';
    return 'Unknown';
  }
  
  function getSuitName(suit: number): string {
    const suitNames = ['Blanks', 'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes'];
    return suitNames[suit] || '-';
  }
  
  function isPlayable(domino: { high: number; low: number }): boolean {
    const dominoStr = getDominoDisplay(domino);
    return playableDominoes.has(dominoStr);
  }
  
  function handleDominoClick(domino: { high: number; low: number }) {
    const dominoStr = getDominoDisplay(domino);
    if (!isPlayable(domino)) return;
    
    console.log('Clicking domino:', dominoStr);
    
    // Find the matching action - try both orientations
    const action = $availableActions.find(a => 
      a.id === `play-${domino.high}-${domino.low}` ||
      a.id === `play-${domino.low}-${domino.high}`
    );
    
    if (action) {
      console.log('Executing action:', action.id);
      gameActions.executeAction(action);
    } else {
      console.log('No matching action found for domino:', dominoStr);
    }
  }
  
  function executeAction(action: any) {
    gameActions.executeAction(action);
  }
</script>

<main>
  <div class="debug-layout">
    <!-- Header Bar -->
    <header class="header-bar">
      <div class="title-section">
        <h1>Texas 42</h1>
        <span class="phase-badge {$gameState.phase}">{$gameState.phase.replace('_', ' ')}</span>
      </div>
      <div class="score-section">
        <span>Scores: {$gameState.teamScores[0]}-{$gameState.teamScores[1]}</span>
        <span>Marks: {$gameState.teamMarks[0]}-{$gameState.teamMarks[1]}</span>
        <span>Turn: P{$gameState.currentPlayer}</span>
      </div>
    </header>

    <!-- Main Game Area -->
    <div class="game-area">
      <!-- Left: Trick History -->
      <div class="history-panel">
        <h3>Tricks ({$gameState.tricks.length}/7)</h3>
        <div class="trick-list">
          {#each $gameState.tricks as trick, i}
            <div class="past-trick">
              <div class="trick-header">
                <span class="trick-num">#{i + 1}</span>
                <span class="trick-winner">P{trick.winner}</span>
                <span class="trick-points">{trick.points}pt</span>
              </div>
              <div class="trick-dominoes">
                {#each trick.plays as play}
                  <span 
                    class="trick-domino"
                    class:winner={play.player === trick.winner}
                    title="P{play.player}"
                  >
                    {getDominoDisplay(play.domino)}
                  </span>
                {/each}
              </div>
            </div>
          {/each}
        </div>
      </div>

      <!-- Center: Game State & Hand -->
      <div class="center-area">
        <!-- Trump & Suit Display -->
        <div class="game-state-display">
          <div class="trump-display">
            <span class="label">TRUMP</span>
            <span class="value">{getTrumpDisplay($gameState.trump)}</span>
          </div>
          {#if $gameState.currentSuit !== -1}
            <div class="suit-display">
              <span class="label">LED SUIT</span>
              <span class="value">{getSuitName($gameState.currentSuit)}</span>
            </div>
          {/if}
        </div>

        <!-- Current Trick -->
        <div class="current-trick-area">
          <h3>Current Trick</h3>
          {#if $gameState.currentTrick.length > 0}
            <div class="current-trick">
              {#each $gameState.currentTrick as play}
                <div class="trick-play">
                  <span class="player-label">P{play.player}</span>
                  <span class="domino-label">{getDominoDisplay(play.domino)}</span>
                </div>
              {/each}
            </div>
          {:else}
            <div class="no-trick">Waiting for first play...</div>
          {/if}
        </div>

        <!-- Player Hand -->
        <div class="hand-area">
          <h3>{currentPlayer.name}'s Hand</h3>
          <div class="hand-dominoes">
            {#each currentPlayer.hand as domino}
              <button
                class="hand-domino"
                class:playable={isPlayable(domino)}
                class:counting={getDominoPoints(domino) > 0}
                disabled={!isPlayable(domino)}
                on:click={() => handleDominoClick(domino)}
                title="{getDominoDisplay(domino)} ({getDominoPoints(domino)} pts)"
              >
                <span class="pip">{domino.high}</span>
                <div class="divider"></div>
                <span class="pip">{domino.low}</span>
              </button>
            {/each}
          </div>
        </div>
      </div>

      <!-- Right: Actions -->
      <div class="actions-panel">
        <h3>All Actions ({$availableActions.length})</h3>
        <div class="action-categories">
          {#each $availableActions.filter(a => !a.id.startsWith('play-')) as action}
            <button 
              class="action-btn"
              on:click={() => executeAction(action)}
              data-testid={action.id}
            >
              {action.label}
            </button>
          {/each}
        </div>
      </div>
    </div>

    <!-- Footer -->
    <footer class="footer-bar">
      <span>Debug Mode • Press <kbd>Ctrl+Shift+D</kbd> for full Debug UI</span>
    </footer>
  </div>
  
  <DebugUI />
</main>

<style>
  :global(body) {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f5f5;
    color: #333;
    overflow: hidden;
  }
  
  :global(*) {
    box-sizing: border-box;
  }
  
  main {
    height: 100vh;
    overflow: hidden;
  }
  
  .debug-layout {
    height: 100vh;
    display: flex;
    flex-direction: column;
    background: #e9ecef;
  }
  
  /* Header */
  .header-bar {
    background: #2c3e50;
    color: white;
    padding: 8px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }
  
  .title-section {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  
  .title-section h1 {
    margin: 0;
    font-size: 20px;
  }
  
  .phase-badge {
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
  }
  
  .phase-badge.setup { background: #607D8B; }
  .phase-badge.bidding { background: #2196F3; }
  .phase-badge.trump_selection { background: #FF9800; }
  .phase-badge.playing { background: #4CAF50; }
  .phase-badge.scoring { background: #9C27B0; }
  .phase-badge.game_end { background: #F44336; }
  
  .score-section {
    display: flex;
    gap: 20px;
    font-size: 14px;
  }
  
  /* Main Game Area */
  .game-area {
    flex: 1;
    display: grid;
    grid-template-columns: 250px 1fr 250px;
    gap: 1px;
    background: #dee2e6;
    overflow: hidden;
  }
  
  .history-panel, .center-area, .actions-panel {
    background: white;
    padding: 12px;
    overflow-y: auto;
  }
  
  h3 {
    margin: 0 0 10px 0;
    font-size: 14px;
    color: #495057;
    text-transform: uppercase;
    font-weight: 600;
  }
  
  /* Left Panel - Trick History */
  .trick-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .past-trick {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 6px;
    font-size: 11px;
  }
  
  .trick-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
    font-weight: 600;
  }
  
  .trick-num {
    color: #495057;
  }
  
  .trick-winner {
    color: #28a745;
  }
  
  .trick-points {
    color: #6c757d;
  }
  
  .trick-dominoes {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 2px;
  }
  
  .trick-domino {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 2px;
    padding: 2px;
    text-align: center;
    font-family: monospace;
    font-size: 10px;
  }
  
  .trick-domino.winner {
    background: #d4edda;
    border-color: #28a745;
    font-weight: bold;
  }
  
  /* Center Area */
  .center-area {
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding: 20px;
  }
  
  /* Game State Display */
  .game-state-display {
    display: flex;
    justify-content: center;
    gap: 40px;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 8px;
    border: 2px solid #dee2e6;
  }
  
  .trump-display, .suit-display {
    text-align: center;
  }
  
  .trump-display .label, .suit-display .label {
    display: block;
    font-size: 12px;
    color: #6c757d;
    margin-bottom: 4px;
  }
  
  .trump-display .value {
    display: block;
    font-size: 24px;
    font-weight: bold;
    color: #dc3545;
  }
  
  .suit-display .value {
    display: block;
    font-size: 24px;
    font-weight: bold;
    color: #007bff;
  }
  
  /* Current Trick */
  .current-trick-area {
    text-align: center;
  }
  
  .current-trick {
    display: flex;
    gap: 15px;
    justify-content: center;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 8px;
    min-height: 80px;
    align-items: center;
  }
  
  .trick-play {
    background: white;
    border: 2px solid #dee2e6;
    border-radius: 6px;
    padding: 10px 15px;
    text-align: center;
  }
  
  .player-label {
    display: block;
    font-size: 11px;
    color: #6c757d;
    margin-bottom: 4px;
  }
  
  .domino-label {
    display: block;
    font-size: 18px;
    font-weight: 600;
  }
  
  .no-trick {
    color: #6c757d;
    padding: 40px;
    font-style: italic;
  }
  
  /* Player Hand */
  .hand-area {
    text-align: center;
  }
  
  .hand-dominoes {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  
  .hand-domino {
    background: #e9ecef;
    border: 3px solid #adb5bd;
    border-radius: 8px;
    width: 50px;
    height: 100px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    cursor: not-allowed;
    transition: all 0.2s;
    padding: 0;
    position: relative;
  }
  
  .hand-domino:disabled {
    opacity: 0.5;
    background: #f8f9fa;
    border-color: #dee2e6;
  }
  
  .hand-domino:disabled .pip {
    color: #6c757d;
  }
  
  .hand-domino.playable {
    cursor: pointer;
    border-color: #28a745;
    background: #28a745;
    opacity: 1;
    box-shadow: 0 2px 4px rgba(40, 167, 69, 0.2);
  }
  
  .hand-domino.playable .pip {
    color: white;
  }
  
  .hand-domino.playable .divider {
    background: white;
  }
  
  .hand-domino.playable:hover {
    transform: translateY(-5px) scale(1.05);
    box-shadow: 0 6px 12px rgba(40, 167, 69, 0.4);
    background: #1e7e34;
    border-color: #1e7e34;
  }
  
  .hand-domino.counting {
    position: relative;
  }
  
  .hand-domino.counting::before {
    content: '';
    position: absolute;
    top: -3px;
    left: -3px;
    right: -3px;
    bottom: -3px;
    background: #ffc107;
    border-radius: 8px;
    z-index: -1;
  }
  
  .hand-domino.counting.playable {
    background: #28a745;
    border-color: #ffc107;
  }
  
  .hand-domino .pip {
    font-size: 20px;
    font-weight: bold;
    color: #333;
  }
  
  .hand-domino .divider {
    width: 30px;
    height: 2px;
    background: #333;
    margin: 8px 0;
  }
  
  /* Right Panel - Actions */
  .action-categories {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  
  .action-group h4 {
    margin: 0 0 6px 0;
    font-size: 12px;
    color: #6c757d;
    text-transform: uppercase;
  }
  
  .action-btn {
    display: block;
    width: 100%;
    background: #007bff;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
    cursor: pointer;
    transition: background 0.15s;
    text-align: left;
    margin-bottom: 4px;
  }
  
  .action-btn:hover {
    background: #0056b3;
  }
  
  /* Footer */
  .footer-bar {
    background: #f8f9fa;
    border-top: 1px solid #dee2e6;
    padding: 6px 16px;
    text-align: center;
    font-size: 12px;
    color: #6c757d;
    flex-shrink: 0;
  }
  
  .footer-bar kbd {
    display: inline-block;
    padding: 2px 4px;
    font-size: 11px;
    font-family: monospace;
    background: white;
    border: 1px solid #ced4da;
    border-radius: 3px;
  }
</style>