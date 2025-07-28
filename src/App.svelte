<script lang="ts">
  import { gameState, availableActions, currentPlayer, gamePhase, biddingInfo, teamInfo, trickInfo, gameActions } from './ui/stores/gameStore';
  import PlayerHand from './ui/components/PlayerHand.svelte';
  import BiddingPanel from './ui/components/BiddingPanel.svelte';
  import TrickDisplay from './ui/components/TrickDisplay.svelte';
  import ScoreBoard from './ui/components/ScoreBoard.svelte';
  import DebugApp from './debug/DebugApp.svelte';
  import type { StateTransition } from './game/types';
  import { getValidPlays } from './game';
  
  function handleAction(transition: StateTransition) {
    gameActions.executeAction(transition);
  }
  
  function handlePlayDomino(dominoId: string) {
    const playAction = $availableActions.find(action => action.id === `play-${dominoId}`);
    if (playAction) {
      gameActions.executeAction(playAction);
    }
  }
  
  function resetGame() {
    gameActions.resetGame();
  }
  
  // Get valid plays for current player
  const validPlayIds = $derived($gamePhase === 'playing' && $gameState.trump !== null
    ? getValidPlays($currentPlayer.hand, $gameState.currentTrick, $gameState.trump).map(d => d.id)
    : []);
</script>

<main class="game-container">
  <header class="game-header">
    <h1>Texas 42 - mk5</h1>
    <div class="game-controls">
      <button class="reset-btn" onclick={resetGame}>
        New Game
      </button>
    </div>
  </header>
  
  <div class="game-layout">
    <div class="left-panel">
      <ScoreBoard 
        teamScores={$teamInfo.scores}
        teamMarks={$teamInfo.marks}
        gameTarget={$teamInfo.target}
      />
      
      <div data-testid="game-phase" style="display: none;">{$gamePhase}</div>
      
      {#if $gamePhase === 'bidding' || $gamePhase === 'trump_selection' || ($gamePhase === 'playing' && $gameState.trump === null)}
        <BiddingPanel 
          availableActions={$availableActions}
          currentBid={$biddingInfo.currentBid}
          bids={$biddingInfo.bids}
          onAction={handleAction}
        />
      {/if}
      
      {#if $gamePhase === 'playing' || $gameState.tricks.length > 0}
        <TrickDisplay 
          currentTrick={$trickInfo.currentTrick}
          completedTricks={$trickInfo.completedTricks}
          trump={$trickInfo.trump}
        />
      {/if}
    </div>
    
    <div class="right-panel">
      <div class="players-section">
        <h2>Players</h2>
        
        <div class="all-players">
          {#each $gameState.players as player (player.id)}
            <div class="player-section" class:current-player={player.id === $gameState.currentPlayer}>
              <PlayerHand 
                {player}
                validPlays={player.id === $gameState.currentPlayer ? validPlayIds : []}
                onPlayDomino={player.id === $gameState.currentPlayer ? handlePlayDomino : undefined}
              />
            </div>
          {/each}
        </div>
      </div>
    </div>
  </div>
  
  {#if $gamePhase === 'game_end'}
    <div class="game-end-overlay">
      <div class="game-end-modal">
        <h2>Game Complete!</h2>
        <p>
          {#if $teamInfo.marks[0] >= $teamInfo.target}
            Team 1 wins with {$teamInfo.marks[0]} marks!
          {:else}
            Team 2 wins with {$teamInfo.marks[1]} marks!
          {/if}
        </p>
        <button class="play-again-btn" onclick={resetGame}>
          Play Again
        </button>
      </div>
    </div>
  {/if}
  
  <DebugApp />
</main>

<style>
  .game-container {
    min-height: 100vh;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  
  .game-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    color: white;
  }
  
  .game-header h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 300;
  }
  
  .game-controls {
    display: flex;
    gap: 12px;
  }
  
  .reset-btn {
    background: rgba(255, 255, 255, 0.2);
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .reset-btn:hover {
    background: rgba(255, 255, 255, 0.3);
  }
  
  .game-layout {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 20px;
    min-height: calc(100vh - 100px);
  }
  
  .left-panel {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  
  .right-panel {
    display: flex;
    flex-direction: column;
  }
  
  .players-section h2 {
    color: white;
    margin: 0 0 16px 0;
    font-size: 20px;
    font-weight: 300;
  }
  
  .all-players {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  
  .player-section {
    transition: all 0.2s ease;
  }
  
  .player-section.current-player {
    transform: scale(1.02);
    box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.3);
    border-radius: 8px;
  }
  
  .game-end-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }
  
  .game-end-modal {
    background: white;
    padding: 40px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    max-width: 400px;
    width: 90%;
  }
  
  .game-end-modal h2 {
    margin: 0 0 16px 0;
    color: #333;
  }
  
  .game-end-modal p {
    margin: 0 0 24px 0;
    color: #666;
    font-size: 16px;
  }
  
  .play-again-btn {
    background: #4caf50;
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 6px;
    font-size: 16px;
    cursor: pointer;
    transition: background 0.2s ease;
  }
  
  .play-again-btn:hover {
    background: #45a049;
  }
  
  @media (max-width: 1024px) {
    .game-layout {
      grid-template-columns: 1fr;
      gap: 16px;
    }
  }
  
  @media (max-width: 768px) {
    .game-container {
      padding: 12px;
    }
    
    .game-header {
      flex-direction: column;
      gap: 12px;
      text-align: center;
    }
    
    .game-header h1 {
      font-size: 24px;
    }
  }
  
  @media (prefers-reduced-motion: reduce) {
    .player-section,
    .reset-btn,
    .play-again-btn {
      transition: none;
    }
    
    .player-section.current-player {
      transform: none;
    }
  }
</style>