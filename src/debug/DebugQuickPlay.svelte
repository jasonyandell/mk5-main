<script lang="ts">
  import { gameState, availableActions } from '../stores/gameStore';
  import { gameActions } from '../stores/gameStore';
  
  let isAutoPlaying = false;
  let autoPlaySpeed = 500; // ms between actions
  let autoPlayInterval: number | null = null;
  let strategy: 'random' | 'first' | 'aggressive' | 'conservative' = 'random';
  
  function startAutoPlay() {
    isAutoPlaying = true;
    autoPlayInterval = setInterval(() => {
      const actions = $availableActions;
      if (actions.length === 0 || $gameState.phase === 'game_end') {
        stopAutoPlay();
        return;
      }
      
      let action;
      switch (strategy) {
        case 'first':
          action = actions[0];
          break;
        case 'aggressive':
          // Prefer higher bids and offensive plays
          action = actions.find(a => a.id.includes('bid-') && a.id.includes('-4')) || 
                   actions.find(a => a.id.includes('play-') && a.label.includes('10')) ||
                   actions[0];
          break;
        case 'conservative':
          // Prefer pass and defensive plays
          action = actions.find(a => a.id.includes('pass')) || 
                   actions[actions.length - 1];
          break;
        case 'random':
        default:
          action = actions[Math.floor(Math.random() * actions.length)];
      }
      
      if (action) {
        gameActions.executeAction(action);
      }
    }, autoPlaySpeed);
  }
  
  function stopAutoPlay() {
    isAutoPlaying = false;
    if (autoPlayInterval) {
      clearInterval(autoPlayInterval);
      autoPlayInterval = null;
    }
  }
  
  function skipToPhase(targetPhase: string) {
    stopAutoPlay();
    
    // Fast forward through actions until we reach target phase
    let safety = 0;
    while ($gameState.phase !== targetPhase && safety < 100) {
      const actions = $availableActions;
      if (actions.length === 0) break;
      
      // Take reasonable default actions
      const action = actions.find(a => 
        a.id.includes('pass') || 
        a.id.includes('-30') || 
        a.id.includes('set-trump-') ||
        a.id.includes('complete-trick') ||
        a.id.includes('score-hand')
      ) || actions[0];
      
      if (action) {
        gameActions.executeAction(action);
      }
      safety++;
    }
  }
  
  function completeHand() {
    stopAutoPlay();
    
    // Play through to end of hand
    let safety = 0;
    while ($gameState.phase !== 'scoring' && $gameState.phase !== 'game_end' && safety < 200) {
      const actions = $availableActions;
      if (actions.length === 0) break;
      
      const action = actions[0];
      if (action) {
        gameActions.executeAction(action);
      }
      safety++;
    }
  }
</script>

<div class="quickplay-panel">
  <h3>⚡ Quick Play & Simulation</h3>
  
  <div class="autoplay-section">
    <h4>Auto-Play</h4>
    <div class="autoplay-controls">
      <button 
        class="play-btn"
        class:playing={isAutoPlaying}
        on:click={isAutoPlaying ? stopAutoPlay : startAutoPlay}
        data-testid="autoplay-toggle"
      >
        {isAutoPlaying ? '⏸️ Pause' : '▶️ Play'}
      </button>
      
      <div class="speed-control">
        <label for="play-speed">Speed (ms):</label>
        <input 
          id="play-speed"
          type="number"
          bind:value={autoPlaySpeed}
          min="100"
          max="5000"
          step="100"
          disabled={isAutoPlaying}
        />
      </div>
      
      <div class="strategy-control">
        <label for="play-strategy">Strategy:</label>
        <select 
          id="play-strategy"
          bind:value={strategy}
          disabled={isAutoPlaying}
        >
          <option value="random">Random</option>
          <option value="first">First Available</option>
          <option value="aggressive">Aggressive</option>
          <option value="conservative">Conservative</option>
        </select>
      </div>
    </div>
  </div>
  
  <div class="skip-section">
    <h4>Skip To Phase</h4>
    <div class="skip-buttons">
      <button 
        on:click={() => skipToPhase('bidding')}
        disabled={$gameState.phase === 'bidding' || isAutoPlaying}
        data-testid="skip-to-bidding"
      >
        Bidding
      </button>
      <button 
        on:click={() => skipToPhase('trump_selection')}
        disabled={$gameState.phase === 'trump_selection' || isAutoPlaying}
        data-testid="skip-to-trump"
      >
        Trump Selection
      </button>
      <button 
        on:click={() => skipToPhase('playing')}
        disabled={$gameState.phase === 'playing' || isAutoPlaying}
        data-testid="skip-to-playing"
      >
        Playing
      </button>
      <button 
        on:click={() => skipToPhase('scoring')}
        disabled={$gameState.phase === 'scoring' || isAutoPlaying}
        data-testid="skip-to-scoring"
      >
        Scoring
      </button>
    </div>
  </div>
  
  <div class="batch-section">
    <h4>Batch Actions</h4>
    <button 
      class="batch-btn"
      on:click={completeHand}
      disabled={isAutoPlaying || $gameState.phase === 'game_end'}
      data-testid="complete-hand"
    >
      Complete Current Hand
    </button>
  </div>
  
  <div class="status-section">
    <h4>Simulation Status</h4>
    <div class="status-info">
      <div class="status-item">
        <label>Current Phase:</label>
        <span>{$gameState.phase}</span>
      </div>
      <div class="status-item">
        <label>Actions Available:</label>
        <span>{$availableActions.length}</span>
      </div>
      <div class="status-item">
        <label>Auto-Play:</label>
        <span class:active={isAutoPlaying}>{isAutoPlaying ? 'Active' : 'Inactive'}</span>
      </div>
    </div>
  </div>
</div>

<style>
  .quickplay-panel {
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 20px;
  }
  
  .quickplay-panel h3 {
    margin: 0 0 20px 0;
    color: #FFC107;
    font-size: 18px;
  }
  
  .quickplay-panel h4 {
    margin: 0 0 10px 0;
    color: #888;
    font-size: 14px;
  }
  
  .autoplay-section, .skip-section, .batch-section, .status-section {
    margin-bottom: 25px;
    padding-bottom: 20px;
    border-bottom: 1px solid #444;
  }
  
  .status-section {
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
  }
  
  .autoplay-controls {
    display: grid;
    grid-template-columns: auto 1fr 1fr;
    gap: 15px;
    align-items: center;
  }
  
  .play-btn {
    padding: 10px 20px;
    background: #4CAF50;
    border: none;
    border-radius: 6px;
    color: white;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 5px;
  }
  
  .play-btn:hover {
    background: #45a049;
  }
  
  .play-btn.playing {
    background: #FF9800;
  }
  
  .play-btn.playing:hover {
    background: #F57C00;
  }
  
  .speed-control, .strategy-control {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  
  label {
    font-size: 12px;
    color: #aaa;
  }
  
  input, select {
    background: #333;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 8px;
    color: #fff;
    font-size: 14px;
  }
  
  input:focus, select:focus {
    outline: none;
    border-color: #FFC107;
  }
  
  input:disabled, select:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .skip-buttons {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
  }
  
  .skip-buttons button, .batch-btn {
    padding: 10px;
    background: #333;
    border: 1px solid #555;
    border-radius: 6px;
    color: #fff;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 14px;
  }
  
  .skip-buttons button:hover:not(:disabled), .batch-btn:hover:not(:disabled) {
    background: #444;
    border-color: #666;
  }
  
  .skip-buttons button:disabled, .batch-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .batch-btn {
    width: 100%;
    background: #2196F3;
    border-color: #2196F3;
  }
  
  .batch-btn:hover:not(:disabled) {
    background: #1976D2;
  }
  
  .status-info {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .status-item {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
  }
  
  .status-item label {
    color: #888;
  }
  
  .status-item span {
    color: #fff;
    font-weight: 500;
  }
  
  .status-item span.active {
    color: #4CAF50;
  }
</style>