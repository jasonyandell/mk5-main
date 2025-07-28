<script lang="ts">
  import type { Player } from '../../game/types';
  import DominoCard from './DominoCard.svelte';
  
  interface Props {
    player: Player;
    validPlays?: string[];
    onPlayDomino?: (dominoId: string) => void;
  }
  
  let { player, validPlays = [], onPlayDomino }: Props = $props();
</script>

<div class="player-hand">
  <div class="player-info">
    <h3>{player.name}</h3>
    <div class="team-badge team-{player.teamId}">
      Team {player.teamId + 1}
    </div>
  </div>
  
  <div class="hand-cards">
    {#each player.hand as domino (domino.id)}
      <DominoCard 
        {domino}
        playable={validPlays.includes(domino.id)}
        onClick={() => onPlayDomino?.(domino.id)}
      />
    {/each}
  </div>
</div>

<style>
  .player-hand {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 16px;
    background: #f5f5f5;
    border-radius: 8px;
    contain: layout style;
  }
  
  .player-info {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  
  .player-info h3 {
    margin: 0;
    font-size: 16px;
    color: #333;
  }
  
  .team-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: bold;
    color: white;
  }
  
  .team-badge.team-0 {
    background: #2196F3;
  }
  
  .team-badge.team-1 {
    background: #FF5722;
  }
  
  .hand-cards {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    min-height: 100px;
  }
  
  @media (max-width: 768px) {
    .hand-cards {
      gap: 4px;
    }
  }
</style>