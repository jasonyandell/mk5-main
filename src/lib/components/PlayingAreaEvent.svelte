<script lang="ts">
  import { eventGame } from '../../stores/eventGame';
  import Domino from './Domino.svelte';
  import type { Play } from '../../game/types';

  const gamePhase = eventGame.gamePhase;
  const hands = eventGame.hands;
  const trick = eventGame.trick;
  const teams = eventGame.teams;
  const currentPlayer = eventGame.currentPlayer;
  const playerTypes = eventGame.playerTypes;
  const consensus = eventGame.consensus;
  const validMoves = eventGame.validMoves;

  function handleDominoClick(playerId: number, domino: any) {
    if ($currentPlayer === playerId && $gamePhase === 'playing') {
      const moves = $validMoves.get(playerId) || [];
      const canPlay = moves.some(d => d.high === domino.high && d.low === domino.low);
      if (canPlay) {
        eventGame.playDomino(playerId, domino);
      }
    }
  }

  function handleAgreeToCompleteTrick() {
    const humanPlayer = 0;
    eventGame.agreeToAction(humanPlayer, 'complete-trick');
  }

  function handleAgreeToScoreHand() {
    const humanPlayer = 0;
    eventGame.agreeToAction(humanPlayer, 'score-hand');
  }

  function getPlayerPosition(playerId: number): string {
    const positions = ['bottom', 'left', 'top', 'right'];
    return positions[playerId];
  }

  $: showCompleteTrickButton = $consensus.has('complete-trick') && 
    !($consensus.get('complete-trick')?.has(0));
  
  $: showScoreHandButton = $consensus.has('score-hand') && 
    !($consensus.get('score-hand')?.has(0));
</script>

<div class="relative w-full h-full flex items-center justify-center">
  <div class="absolute inset-0 bg-gradient-to-br from-green-100 to-green-200 dark:from-green-900 dark:to-green-800 rounded-lg">
  </div>

  <div class="relative w-full h-full p-4">
    {#each Object.entries($hands) as [playerId, hand]}
      <div class="absolute {getPlayerPosition(Number(playerId))} flex flex-wrap gap-1">
        {#each hand as domino}
          <button
            onclick={() => handleDominoClick(Number(playerId), domino)}
            disabled={$currentPlayer !== Number(playerId) || $gamePhase !== 'playing'}
            class="transition-transform hover:scale-110"
          >
            <Domino {domino} size="small" />
          </button>
        {/each}
      </div>
    {/each}

    <div class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
      <div class="flex flex-col items-center gap-4">
        <div class="text-xl font-bold">Current Trick</div>
        <div class="flex gap-2">
          {#each $trick as play}
            <div class="flex flex-col items-center">
              <Domino domino={play.domino} size="medium" />
              <span class="text-sm">P{play.player + 1}</span>
            </div>
          {/each}
        </div>

        {#if showCompleteTrickButton}
          <button
            class="btn btn-primary"
            onclick={handleAgreeToCompleteTrick}
          >
            Complete Trick
          </button>
        {/if}

        {#if showScoreHandButton}
          <button
            class="btn btn-primary"
            onclick={handleAgreeToScoreHand}
          >
            Score Hand
          </button>
        {/if}
      </div>
    </div>

    <div class="absolute bottom-4 right-4">
      <div class="stats shadow">
        <div class="stat">
          <div class="stat-title">Team 1</div>
          <div class="stat-value">{$teams.trickPoints[0]}</div>
          <div class="stat-desc">Marks: {$teams.marks[0]}</div>
        </div>
        <div class="stat">
          <div class="stat-title">Team 2</div>
          <div class="stat-value">{$teams.trickPoints[1]}</div>
          <div class="stat-desc">Marks: {$teams.marks[1]}</div>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .bottom {
    bottom: 1rem;
    left: 50%;
    transform: translateX(-50%);
  }
  
  .left {
    left: 1rem;
    top: 50%;
    transform: translateY(-50%);
  }
  
  .top {
    top: 1rem;
    left: 50%;
    transform: translateX(-50%);
  }
  
  .right {
    right: 1rem;
    top: 50%;
    transform: translateY(-50%);
  }
</style>