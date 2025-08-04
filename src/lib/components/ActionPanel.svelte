<script lang="ts">
  import { gamePhase, availableActions, gameActions, teamInfo, biddingInfo } from '../../stores/gameStore';
  import type { StateTransition } from '../../game/types';

  // Group actions by type
  $: groupedActions = (() => {
    const groups: { [key: string]: StateTransition[] } = {
      bidding: [],
      trump: [],
      play: [],
      other: []
    };

    $availableActions.forEach(action => {
      if (action.id.startsWith('bid-') || action.id === 'pass') {
        groups.bidding.push(action);
      } else if (action.id.startsWith('trump-')) {
        groups.trump.push(action);
      } else if (action.id.startsWith('play-')) {
        // Skip play actions - they're handled by domino clicks
      } else {
        groups.other.push(action);
      }
    });

    return groups;
  })();


  let shakeActionId: string | null = null;

  async function executeAction(action: StateTransition) {
    try {
      gameActions.executeAction(action);
    } catch (error) {
      // Trigger shake animation on error
      shakeActionId = action.id;
      setTimeout(() => {
        shakeActionId = null;
      }, 300);
    }
  }

  // Get team names based on perspective
  const teamNames = ['US', 'THEM'];

  // Get tooltip for bid actions
  function getBidTooltip(action: StateTransition): string {
    if (action.id === 'pass') {
      return 'Pass on bidding - let others bid';
    }
    
    const bidValue = action.id.match(/\d+/)?.[0];
    if (!bidValue) return '';
    
    const points = parseInt(bidValue);
    if (action.id.includes('mark')) {
      const marks = parseInt(bidValue);
      return `Bid ${marks} mark${marks > 1 ? 's' : ''} (${marks * 42} points) - must win all ${marks * 42} points`;
    } else {
      return `Bid ${points} points - must win at least ${points} out of 42 points`;
    }
  }
</script>

<div class="action-panel">
  <h2>Actions</h2>

  <div class="actions-container">
    {#if $gamePhase === 'bidding'}
      <div class="action-group">
        <h3>Bidding</h3>
        <div class="bid-actions">
          {#each groupedActions.bidding as action}
            {#if action.id === 'pass'}
              <button 
                class="action-button pass {shakeActionId === action.id ? 'invalid-action-shake' : ''}"
                on:click={() => executeAction(action)}
                data-testid={action.id}
                title={getBidTooltip(action)}
              >
                Pass
              </button>
            {/if}
          {/each}
          
          <div class="bid-separator"></div>
          
          {#each groupedActions.bidding as action}
            {#if action.id !== 'pass'}
              <button 
                class="action-button bid {shakeActionId === action.id ? 'invalid-action-shake' : ''}"
                on:click={() => executeAction(action)}
                data-testid={action.id}
                title={getBidTooltip(action)}
              >
                {action.label}
              </button>
            {/if}
          {/each}
        </div>
      </div>
    {/if}

    {#if $gamePhase === 'trump_selection'}
      <div class="action-group">
        <h3>Select Trump</h3>
        <div class="trump-actions">
          {#each groupedActions.trump as action}
            <button 
              class="action-button primary"
              on:click={() => executeAction(action)}
              data-testid={action.id}
            >
              {action.label}
            </button>
          {/each}
        </div>
      </div>
    {/if}

    {#if groupedActions.other.length > 0}
      <div class="action-group">
        <h3>Quick Actions</h3>
        <div class="other-actions">
          {#each groupedActions.other as action}
            <button 
              class="action-button"
              on:click={() => executeAction(action)}
              data-testid={action.id}
            >
              {action.label}
            </button>
          {/each}
        </div>
      </div>
    {/if}
  </div>

  <div class="team-status">
    <h3>Team Status</h3>
    <div class="status-content">
      <div class="score-row">
        <span class="team-name">{teamNames[0]}:</span>
        <span class="score">{$teamInfo.marks[0]} marks ({$teamInfo.scores[0]} pts)</span>
      </div>
      <div class="score-row">
        <span class="team-name">{teamNames[1]}:</span>
        <span class="score">{$teamInfo.marks[1]} marks ({$teamInfo.scores[1]} pts)</span>
      </div>
      
      {#if $biddingInfo.winningBidder !== -1}
        <div class="bid-info">
          <div class="bid-row">
            <span>Bid:</span>
            <span>{$biddingInfo.currentBid.value || 0} (P{$biddingInfo.winningBidder})</span>
          </div>
          {#if $gamePhase === 'playing'}
            <div class="bid-row">
              <span>Need:</span>
              <span>{Math.max(0, ($biddingInfo.currentBid.value || 0) - $teamInfo.scores[Math.floor($biddingInfo.winningBidder / 2)])} more</span>
            </div>
          {/if}
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .action-panel {
    padding: 16px;
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  h2 {
    margin: 0 0 16px 0;
    font-size: 18px;
    font-weight: 600;
    color: #002868;
  }

  h3 {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .actions-container {
    flex: 1;
    overflow-y: auto;
  }

  .action-group {
    margin-bottom: 24px;
  }

  .bid-actions,
  .trump-actions,
  .other-actions {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .action-button {
    padding: 10px 16px;
    background-color: #f3f4f6;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    text-align: center;
  }

  .action-button:hover {
    background-color: #e5e7eb;
    transform: translateY(-1px);
  }

  .action-button.bid {
    background-color: #002868;
    color: white;
    border-color: #002868;
  }

  .action-button.bid:hover {
    background-color: #001a4d;
  }

  .action-button.pass {
    background-color: #dc2626;
    color: white;
    border-color: #dc2626;
  }

  .action-button.pass:hover {
    background-color: #b91c1c;
  }

  .bid-separator {
    height: 1px;
    background-color: #e5e7eb;
    margin: 12px 0;
  }

  .team-status {
    margin-top: auto;
    padding-top: 16px;
    border-top: 1px solid #e5e7eb;
  }

  .status-content {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px;
  }

  .score-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 14px;
  }

  .team-name {
    font-weight: 600;
    color: #374151;
  }

  .score {
    color: #6b7280;
  }

  .bid-info {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #e5e7eb;
  }

  .bid-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    margin-bottom: 4px;
    color: #6b7280;
  }
</style>