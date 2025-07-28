<script lang="ts">
  interface Props {
    teamScores: [number, number];
    teamMarks: [number, number];
    gameTarget: number;
  }
  
  let { teamScores, teamMarks, gameTarget }: Props = $props();
  
  const team0Progress = $derived((teamMarks[0] / gameTarget) * 100);
  const team1Progress = $derived((teamMarks[1] / gameTarget) * 100);
</script>

<div class="scoreboard">
  <h2>Score</h2>
  
  <div class="teams">
    <div class="team team-0">
      <div class="team-header">
        <h3>Team 1</h3>
        <div class="team-color"></div>
      </div>
      
      <div class="scores">
        <div class="score-item">
          <span class="label">Hand Points:</span>
          <span class="value">{teamScores[0]}</span>
        </div>
        
        <div class="score-item marks">
          <span class="label">Game Marks:</span>
          <span class="value">{teamMarks[0]} / {gameTarget}</span>
        </div>
      </div>
      
      <div class="progress-bar">
        <div 
          class="progress-fill team-0"
          style="width: {team0Progress}%"
        ></div>
      </div>
      
      {#if teamMarks[0] >= gameTarget}
        <div class="winner-badge">Winner!</div>
      {/if}
    </div>
    
    <div class="team team-1">
      <div class="team-header">
        <h3>Team 2</h3>
        <div class="team-color"></div>
      </div>
      
      <div class="scores">
        <div class="score-item">
          <span class="label">Hand Points:</span>
          <span class="value">{teamScores[1]}</span>
        </div>
        
        <div class="score-item marks">
          <span class="label">Game Marks:</span>
          <span class="value">{teamMarks[1]} / {gameTarget}</span>
        </div>
      </div>
      
      <div class="progress-bar">
        <div 
          class="progress-fill team-1"
          style="width: {team1Progress}%"
        ></div>
      </div>
      
      {#if teamMarks[1] >= gameTarget}
        <div class="winner-badge">Winner!</div>
      {/if}
    </div>
  </div>
  
  <div class="game-info">
    <div class="info-item">
      <span class="label">Game Target:</span>
      <span class="value">{gameTarget} marks</span>
    </div>
    <div class="info-item">
      <span class="label">Total Points:</span>
      <span class="value">{teamScores[0] + teamScores[1]} / 42</span>
    </div>
  </div>
</div>

<style>
  .scoreboard {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    contain: layout style;
  }
  
  .scoreboard h2 {
    margin: 0 0 20px 0;
    color: #333;
    text-align: center;
  }
  
  .teams {
    display: flex;
    flex-direction: column;
    gap: 20px;
    margin-bottom: 20px;
  }
  
  .team {
    border: 2px solid;
    border-radius: 8px;
    padding: 16px;
    position: relative;
  }
  
  .team.team-0 {
    border-color: #2196f3;
    background: #e3f2fd;
  }
  
  .team.team-1 {
    border-color: #ff5722;
    background: #fbe9e7;
  }
  
  .team-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
  }
  
  .team-header h3 {
    margin: 0;
    font-size: 18px;
  }
  
  .team-color {
    width: 16px;
    height: 16px;
    border-radius: 50%;
  }
  
  .team-0 .team-color {
    background: #2196f3;
  }
  
  .team-1 .team-color {
    background: #ff5722;
  }
  
  .scores {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 12px;
  }
  
  .score-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 14px;
  }
  
  .score-item.marks {
    font-weight: bold;
    font-size: 16px;
  }
  
  .score-item .label {
    color: #666;
  }
  
  .score-item .value {
    color: #333;
    font-weight: 500;
  }
  
  .progress-bar {
    height: 8px;
    background: #e0e0e0;
    border-radius: 4px;
    overflow: hidden;
  }
  
  .progress-fill {
    height: 100%;
    transition: width 0.3s ease;
  }
  
  .progress-fill.team-0 {
    background: #2196f3;
  }
  
  .progress-fill.team-1 {
    background: #ff5722;
  }
  
  .winner-badge {
    position: absolute;
    top: -10px;
    right: 16px;
    background: #4caf50;
    color: white;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
    text-transform: uppercase;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }
  
  .game-info {
    border-top: 1px solid #eee;
    padding-top: 16px;
    display: flex;
    justify-content: space-between;
    font-size: 14px;
  }
  
  .info-item .label {
    color: #666;
  }
  
  .info-item .value {
    color: #333;
    font-weight: 500;
  }
  
  @media (min-width: 768px) {
    .teams {
      flex-direction: row;
    }
    
    .team {
      flex: 1;
    }
  }
  
  @media (prefers-reduced-motion: reduce) {
    .progress-fill {
      transition: none;
    }
  }
</style>