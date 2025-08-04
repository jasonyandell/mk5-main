<script lang="ts">
  import type { GameState, Bid } from '../../game/types';
  import { isEmptyBid } from '../../game/types';
  import { getCurrentSuit } from '../../game/core/rules';
  
  interface Props {
    gameState: GameState;
  }
  
  let { gameState }: Props = $props();
  
  function getBidLabel(bid: Bid): string {
    switch (bid.type) {
      case 'pass': return 'Pass';
      case 'points': return `${bid.value}pts`;
      case 'marks': return `${bid.value}m`;
      case 'nello': return `N${bid.value}`;
      case 'splash': return `S${bid.value}`;
      case 'plunge': return `P${bid.value}`;
      default: return 'Unknown';
    }
  }
  
  function getTrumpName(trump: any): string {
    if (!trump || trump.type === 'none') return 'None';
    
    if (trump.type === 'suit' && trump.suit !== undefined) {
      const suits = ['0s', '1s', '2s', '3s', '4s', '5s', '6s'];
      return suits[trump.suit] || 'Unknown';
    }
    
    if (trump.type === 'doubles') {
      return 'Doubles';
    }
    
    if (trump.type === 'no-trump') {
      return 'Follow-me';
    }
    
    return 'Unknown';
  }
  
  const currentSuit = $derived(getCurrentSuit(gameState));

  function getHandResultDescription(): { winner: number | null, label: string } {
    if (isEmptyBid(gameState.currentBid) || gameState.winningBidder === -1) {
      return { winner: null, label: 'No bid' };
    }

    const biddingTeam = gameState.players[gameState.winningBidder].teamId;
    const opponentTeam = biddingTeam === 0 ? 1 : 0;
    const biddingTeamScore = gameState.teamScores[biddingTeam];
    
    const bid = gameState.currentBid;
    let handWinnerTeam: number;
    let reason: string;

    switch (bid.type) {
      case 'points': {
        const requiredScore = bid.value!;
        if (biddingTeamScore >= requiredScore) {
          handWinnerTeam = biddingTeam;
          reason = `Made ${requiredScore} (got ${biddingTeamScore})`;
        } else {
          handWinnerTeam = opponentTeam;
          reason = `Set bidders (${biddingTeamScore}/${requiredScore})`;
        }
        break;
      }
      case 'marks': {
        const requiredScore = 42;
        const markValue = bid.value!;
        if (biddingTeamScore >= requiredScore) {
          handWinnerTeam = biddingTeam;
          reason = `Made ${markValue} mark${markValue !== 1 ? 's' : ''} (got ${biddingTeamScore})`;
        } else {
          handWinnerTeam = opponentTeam;
          reason = `Set ${markValue} mark${markValue !== 1 ? 's' : ''} bid (${biddingTeamScore}/42)`;
        }
        break;
      }
      case 'nello': {
        const biddingTeamTricks = gameState.tricks.filter(
          trick => trick.winner !== undefined && 
          gameState.players[trick.winner].teamId === biddingTeam
        ).length;
        
        if (biddingTeamTricks === 0) {
          handWinnerTeam = biddingTeam;
          reason = `Made Nel-O (0 tricks)`;
        } else {
          handWinnerTeam = opponentTeam;
          reason = `Set Nel-O bid (${biddingTeamTricks} tricks)`;
        }
        break;
      }
      default:
        return { winner: null, label: 'Unknown bid type' };
    }

    // Find a representative player from the winning team
    const winnerPlayer = gameState.players.find(p => p.teamId === handWinnerTeam)?.id ?? null;
    return { winner: winnerPlayer, label: reason };
  }
</script>

<div class="debug-game-state" data-testid="debug-panel">
  <div class="state-header">
    <div class="phase-info">
      <span class="label">Phase:</span>
      <span class="value phase-{gameState.phase}" data-testid="phase">Phase: {gameState.phase.toUpperCase()}</span>
    </div>
    <div class="player-info">
      <span class="label">Current:</span>
      <span class="value" data-testid="current-player">Current Player: P{gameState.currentPlayer}</span>
    </div>
    <div class="dealer-info">
      <span class="label">Dealer:</span>
      <span class="value" data-testid="dealer">Dealer: P{gameState.dealer}</span>
    </div>
    <div class="trump-info">
      <span class="label">Trump:</span>
      <span class="value" data-testid="trump">Trump: {getTrumpName(gameState.trump)}</span>
    </div>
  </div>
  
  <div class="core-state-section">
    <h3>Core State</h3>
    <div class="core-state-content">
      Phase: {gameState.phase.toUpperCase()}<br/>
      Dealer: P{gameState.dealer}<br/>
      Current Player: P{gameState.currentPlayer}<br/>
      Trump: {getTrumpName(gameState.trump)}<br/>
      Current Suit: {currentSuit}<br/>
      Bid Winner: {gameState.winningBidder !== -1 ? `P${gameState.winningBidder}` : 'None'}<br/>
      Current Bid: {gameState.currentBid ? getBidLabel(gameState.currentBid) : 'None'}
    </div>
  </div>
  
  <div class="state-details">
    <div class="section">
      <h4>Bidding</h4>
      <div class="bid-history">
        {#each gameState.bids as bid}
          <div class="bid-entry" class:current={bid === gameState.currentBid}>
            <span class="player">P{bid.player}:</span>
            <span class="bid">{getBidLabel(bid)}</span>
          </div>
        {/each}
        {#if gameState.bids.length === 0}
          <div class="empty">No bids yet</div>
        {/if}
      </div>
      {#if gameState.winningBidder !== -1}
        <div class="winning-bidder">
          Bidder: P{gameState.winningBidder}
          {#if gameState.currentBid}
            ({getBidLabel(gameState.currentBid)})
          {/if}
        </div>
        {#if gameState.tricks.length === 7}
          {@const handResult = getHandResultDescription()}
          <div class="winning-bidder">
            Hand Winner: {#if handResult.winner !== null}Team {gameState.players.find(p => p.id === handResult.winner)?.teamId}{:else}None{/if}
            <br/><small>{handResult.label}</small>
          </div>
        {/if}
      {/if}
    </div>
    
    <div class="section" data-testid="score-board">
      <h4>Scores</h4>
      <div class="scores">
        <div class="team">
          <span class="team-label">Team 1:</span>
          <span class="points" data-testid="team-0-score">{gameState.teamScores[0]} points</span>
          <span class="marks" data-testid="team-0-marks">{gameState.teamMarks[0]} marks</span>
        </div>
        <div class="team">
          <span class="team-label">Team 2:</span>
          <span class="points" data-testid="team-1-score">{gameState.teamScores[1]} points</span>
          <span class="marks" data-testid="team-1-marks">{gameState.teamMarks[1]} marks</span>
        </div>
      </div>
    </div>
    
  </div>
</div>

<style>
  .debug-game-state {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 12px;
    font-size: 12px;
  }
  
  .state-header {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #dee2e6;
  }
  
  .state-header > div {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  
  .label {
    font-weight: 600;
    color: #6c757d;
    font-size: 10px;
    text-transform: uppercase;
  }
  
  .value {
    font-weight: 500;
    color: #212529;
  }
  
  .phase-bidding { color: #007bff; }
  .phase-trump_selection { color: #fd7e14; }
  .phase-playing { color: #28a745; }
  .phase-scoring { color: #6f42c1; }
  .phase-game_end { color: #dc3545; }
  
  .core-state-section {
    margin-bottom: 12px;
    padding: 8px;
    background: #e9ecef;
    border-radius: 3px;
    border: 1px solid #dee2e6;
  }
  
  .core-state-section h3 {
    margin: 0 0 6px 0;
    font-size: 12px;
    font-weight: 600;
    color: #495057;
  }
  
  .core-state-content {
    font-size: 11px;
    line-height: 1.4;
    color: #212529;
    font-family: monospace;
  }
  
  .state-details {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }
  
  .section h4 {
    margin: 0 0 6px 0;
    font-size: 11px;
    font-weight: 600;
    color: #495057;
    text-transform: uppercase;
  }
  
  .bid-history {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  
  .bid-entry {
    display: flex;
    gap: 4px;
    font-size: 11px;
  }
  
  .bid-entry.current {
    font-weight: 600;
    color: #007bff;
  }
  
  .player {
    color: #6c757d;
    min-width: 24px;
  }
  
  .winning-bidder {
    margin-top: 4px;
    font-weight: 600;
    color: #28a745;
    font-size: 11px;
  }
  
  .scores {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  
  .team {
    display: flex;
    gap: 6px;
    align-items: center;
    font-size: 11px;
  }
  
  .team-label {
    color: #6c757d;
    min-width: 48px;
  }
  
  .points {
    color: #28a745;
    font-weight: 500;
  }
  
  .marks {
    color: #007bff;
    font-weight: 500;
  }
  
  .empty {
    color: #6c757d;
    font-style: italic;
    font-size: 11px;
  }
</style>