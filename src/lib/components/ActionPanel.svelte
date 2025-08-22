<script lang="ts">
  import { gamePhase, availableActions, gameActions, teamInfo, biddingInfo, currentPlayer, playerView, controllerManager, gameState, uiState } from '../../stores/gameStore';
  import type { StateTransition } from '../../game/types';
  import Domino from './Domino.svelte';
  import { slide } from 'svelte/transition';
  
  interface Props {
    onswitchToPlay?: () => void;
  }
  
  let { onswitchToPlay }: Props = $props();
  
  // Check if we're in test mode
  const urlParams = typeof window !== 'undefined' ? 
    new URLSearchParams(window.location.search) : null;
  const testMode = urlParams?.get('testMode') === 'true';
  

  // Group actions by type with strong typing
  interface GroupedActions {
    bidding: StateTransition[];
    trump: StateTransition[];
    play: StateTransition[];
    other: StateTransition[];
  }

  const groupedActions = $derived((() => {
    const groups: GroupedActions = {
      bidding: [],
      trump: [],
      play: [],
      other: []
    };

    $availableActions.forEach(action => {
      if (action.id.startsWith('bid-') || action.id === 'pass' || action.id === 'redeal') {
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
  })());


  let shakeActionId = $state<string | null>(null);
  let previousPhase = $state($gamePhase);
  
  // React to phase changes for panel switching
  $effect(() => {
    if ($gamePhase === 'playing' && previousPhase === 'trump_selection') {
      onswitchToPlay?.();
    }
    previousPhase = $gamePhase;
  });

  async function executeAction(action: StateTransition) {
    try {
      // Find which human controller should handle this
      const playerId = 'player' in action.action ? action.action.player : 0;
      const humanController = controllerManager.getHumanController(playerId);
      
      if (humanController) {
        humanController.handleUserAction(action);
      } else {
        // Fallback to direct execution (used in testMode)
        console.log('[ActionPanel] Direct execution (no controller):', action.label, 'for player', playerId);
        gameActions.executeAction(action);
      }
      
      // Panel switching is handled by the reactive effect above
    } catch (error) {
      // Trigger shake animation on error
      shakeActionId = action.id;
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

  // Current player's hand (always player 0 for privacy, unless in test mode)
  const playerHand = $derived(testMode ? ($currentPlayer?.hand || []) : ($playerView?.self?.hand || []));

  // Use centralized UI state for waiting logic
  const isWaiting = $derived($uiState.isWaiting);
  const waitingPlayer = $derived($uiState.waitingOnPlayer);
  const isAIThinking = $derived(waitingPlayer >= 0 && controllerManager.isAIControlled(waitingPlayer));
  
  // Track skip attempts for automatic retry
  let skipAttempts = $state(0);
  
  // Reactive skip logic - automatically retry skip in bidding/trump phases
  $effect(() => {
    if (($gamePhase === 'bidding' || $gamePhase === 'trump_selection') && 
        isAIThinking && skipAttempts > 0 && skipAttempts < 3) {
      // Automatically try skip on state change
      controllerManager.skipAIDelays();
      skipAttempts++;
    }
  });
  
  // Reset skip attempts when not AI thinking
  $effect(() => {
    if (!isAIThinking) {
      skipAttempts = 0;
    }
  });

  // State for expandable team status
  let teamStatusExpanded = $state(false);
</script>

<div class="action-panel">
  {#if ($gamePhase === 'bidding' || $gamePhase === 'trump_selection') && playerHand.length > 0}
    <div class="hand-section">
      <h3>Your Hand</h3>
      <div class="hand-display">
        {#each playerHand as domino, i (domino.high + '-' + domino.low)}
          <div class="domino-in-hand" style="--delay: {i * 30}ms">
            <Domino
              {domino}
              small={true}
              showPoints={true}
              clickable={true}
            />
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <div class="actions-container">
    {#if $gamePhase === 'bidding'}
      <!-- Always show compact bid status during bidding -->
      <div class="compact-bid-status">
        <div class="bid-chips">
          {#each [0, 1, 2, 3] as playerId}
            {@const bid = $biddingInfo.bids.find(b => b.player === playerId)}
            {@const isHighBidder = $biddingInfo.currentBid.player === playerId}
            {@const isDealer = $gameState.dealer === playerId}
            <div class="bid-chip" class:high-bidder={isHighBidder} class:dealer={isDealer}>
              <span class="chip-player">P{playerId}</span>
              <span class="chip-bid">
                {#if bid}
                  {#if bid.type === 'pass'}
                    Pass
                  {:else}
                    {bid.value}
                  {/if}
                {:else}
                  --
                {/if}
              </span>
              {#if isDealer}
                <span class="dealer-badge" title="Dealer">D</span>
              {/if}
            </div>
          {/each}
        </div>
        {#if $biddingInfo.currentBid.player !== -1}
          <div class="high-bid-indicator">
            High: {$biddingInfo.currentBid.value}
          </div>
        {/if}
      </div>
    {/if}
    
    {#if isWaiting && isAIThinking}
      <button 
        class="waiting-indicator clickable ai-thinking-indicator"
        onclick={() => {
          // Skip current AI delay
          controllerManager.skipAIDelays();
          // Enable automatic retry for bidding/trump phases
          if ($gamePhase === 'bidding' || $gamePhase === 'trump_selection') {
            skipAttempts = 1; // Start retry counter
          }
        }}
        type="button"
        aria-label="Click to skip AI thinking"
        title="Click to skip AI thinking"
      >
        <span class="robot-icon">🤖</span>
        <span class="waiting-text">P{waitingPlayer} is thinking...</span>
        <span class="skip-hint">(tap to skip)</span>
      </button>
    {/if}
    
    {#if $gamePhase === 'bidding'}
      <div class="action-group">
        <h3>Bidding</h3>
        <div class="bid-actions">
          {#each groupedActions.bidding as action}
            {#if action.id === 'pass'}
              <button 
                class="action-button pass {shakeActionId === action.id ? 'invalid-action-shake' : ''}"
                onclick={() => executeAction(action)}
                onanimationend={() => { if (shakeActionId === action.id) shakeActionId = null; }}
                data-testid={action.id}
                title={getBidTooltip(action)}
              >
                Pass
              </button>
            {:else if action.id === 'redeal'}
              <button 
                class="action-button redeal {shakeActionId === action.id ? 'invalid-action-shake' : ''}"
                onclick={() => executeAction(action)}
                onanimationend={() => { if (shakeActionId === action.id) shakeActionId = null; }}
                data-testid={action.id}
                title="Redeal the dominoes - all players passed"
              >
                Redeal
              </button>
            {/if}
          {/each}
          
          <div class="bid-separator"></div>
          
          {#each groupedActions.bidding as action}
            {#if action.id !== 'pass' && action.id !== 'redeal'}
              <button 
                class="action-button bid {shakeActionId === action.id ? 'invalid-action-shake' : ''}"
                onclick={() => executeAction(action)}
                onanimationend={() => { if (shakeActionId === action.id) shakeActionId = null; }}
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
              class="action-button primary trump-button"
              onclick={() => executeAction(action)}
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
              onclick={() => executeAction(action)}
              data-testid={action.id}
            >
              {action.label}
            </button>
          {/each}
        </div>
      </div>
    {/if}
  </div>

  <div class="team-status" class:expanded={teamStatusExpanded}>
    <button class="team-status-toggle" onclick={() => teamStatusExpanded = !teamStatusExpanded}>
      <div class="compact-scores">
        <div class="team-score us">
          <span class="team-label">US</span>
          <span class="score-value">{$teamInfo.marks[0]}/{$teamInfo.scores[0]}</span>
        </div>
        <div class="expand-indicator">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d={teamStatusExpanded ? "M8 10L4 6h8z" : "M8 6l4 4H4z"} />
          </svg>
        </div>
        <div class="team-score them">
          <span class="team-label">THEM</span>
          <span class="score-value">{$teamInfo.marks[1]}/{$teamInfo.scores[1]}</span>
        </div>
      </div>
    </button>
    
    {#if teamStatusExpanded}
      <div class="expanded-content" transition:slide={{ duration: 200 }}>
        <div class="detailed-scores">
          <div class="team-detail">
            <h4>{teamNames[0]}</h4>
            <div>{$teamInfo.marks[0]} marks</div>
            <div>{$teamInfo.scores[0]} points</div>
          </div>
          <div class="team-detail">
            <h4>{teamNames[1]}</h4>
            <div>{$teamInfo.marks[1]} marks</div>
            <div>{$teamInfo.scores[1]} points</div>
          </div>
        </div>
        
        {#if $biddingInfo.winningBidder !== -1}
          <div class="bid-details">
            <div class="bid-info-row">
              <span>Current Bid:</span>
              <span>{$biddingInfo.currentBid?.value || 0} by P{$biddingInfo.winningBidder}</span>
            </div>
            {#if $gamePhase === 'playing'}
              <div class="bid-info-row">
                <span>Points Needed:</span>
                <span>{Math.max(0, ($biddingInfo.currentBid?.value || 0) - ($teamInfo.scores?.[Math.floor($biddingInfo.winningBidder / 2)] || 0))}</span>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  .action-panel {
    height: 100%;
    display: flex;
    flex-direction: column;
    background: linear-gradient(to bottom, transparent, rgba(255, 255, 255, 0.5));
    overflow: hidden;
  }

  h3 {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    text-align: center;
  }

  .actions-container {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    -webkit-overflow-scrolling: touch;
    padding: 16px;
    padding-top: 8px;
  }

  .action-group {
    margin-bottom: 28px;
    animation: fadeInUp 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  }

  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .bid-actions {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    max-width: 400px;
    margin: 0 auto;
  }

  .trump-actions,
  .other-actions {
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-width: 320px;
    margin: 0 auto;
  }

  .action-button {
    padding: 16px 12px;
    background: white;
    border: 2px solid #e2e8f0;
    border-radius: 16px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    text-align: center;
    min-height: 56px;
    -webkit-tap-highlight-color: transparent;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    position: relative;
    overflow: hidden;
  }

  .action-button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(0, 0, 0, 0.05);
    transform: translate(-50%, -50%);
    transition: width 0.4s, height 0.4s;
  }

  .action-button:active::before {
    width: 200px;
    height: 200px;
  }

  .action-button:active {
    transform: scale(0.96);
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
  }

  .action-button.bid {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: white;
    border: none;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
  }

  .action-button.bid:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
  }

  .action-button.pass {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    border: none;
    grid-column: 1 / -1;
    max-width: none;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
  }

  .action-button.pass:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(239, 68, 68, 0.4);
  }

  .action-button.primary {
    background: linear-gradient(135deg, #8b5cf6, #7c3aed);
    color: white;
    border: none;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
  }

  .action-button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
  }

  .trump-button {
    position: relative;
    overflow: visible;
  }



  @keyframes pointDown {
    0%, 100% { transform: translateX(-50%) translateY(0); }
    50% { transform: translateX(-50%) translateY(5px); }
  }
  
  /* Mobile touch improvements */
  @media (hover: none) and (pointer: coarse) {
    .action-button {
      min-height: 48px;
      font-size: 16px;
      touch-action: manipulation;
    }
    
    .domino-in-hand {
      touch-action: manipulation;
    }
    
    .hand-section {
      padding: 12px;
    }
    
    .actions-container {
      padding-bottom: 100px; /* Extra space for mobile navigation */
    }
    
    .trump-actions {
      gap: 12px;
    }
    
    .bid-actions {
      gap: 12px;
    }
  }

  .invalid-action-shake {
    animation: shake 0.3s ease-in-out;
  }

  @keyframes shake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-8px); }
    75% { transform: translateX(8px); }
  }

  .bid-separator {
    height: 1px;
    background: linear-gradient(90deg, transparent, #e2e8f0 20%, #e2e8f0 80%, transparent);
    margin: 16px 0;
    grid-column: 1 / -1;
  }

  .team-status {
    margin: 0 16px 16px 16px;
    flex-shrink: 0;
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    overflow: hidden;
    transition: all 0.3s ease;
  }
  
  .team-status.expanded {
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
  }


  /* New Compact Team Status Styles */
  .team-status-toggle {
    width: 100%;
    padding: 12px 16px;
    background: none;
    border: none;
    cursor: pointer;
    display: block;
    touch-action: manipulation;
  }

  .compact-scores {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
  }

  .team-score {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
  }

  .team-score.us .team-label {
    color: #3b82f6;
  }

  .team-score.them .team-label {
    color: #ef4444;
  }

  .team-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .score-value {
    font-size: 16px;
    color: #1e293b;
  }

  .expand-indicator {
    display: flex;
    align-items: center;
    color: #64748b;
    transition: transform 0.2s ease;
  }

  .team-status.expanded .expand-indicator {
    transform: rotate(180deg);
  }

  .expanded-content {
    padding: 0 16px 16px 16px;
    border-top: 1px solid #e5e7eb;
  }

  .detailed-scores {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 16px;
  }

  .team-detail {
    text-align: center;
  }

  .team-detail h4 {
    margin: 0 0 8px 0;
    font-size: 14px;
    font-weight: 600;
    color: #64748b;
  }

  .team-detail div {
    font-size: 14px;
    color: #475569;
    margin: 4px 0;
  }

  .bid-details {
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid #e5e7eb;
  }

  .bid-info-row {
    display: flex;
    justify-content: space-between;
    margin: 8px 0;
    font-size: 14px;
  }

  .bid-info-row span:first-child {
    color: #64748b;
  }

  .bid-info-row span:last-child {
    font-weight: 600;
    color: #1e293b;
  }

  /* Hand display styles */
  .hand-section {
    padding: 16px;
    margin: 16px 16px 0 16px;
    background: white;
    border-radius: 20px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    position: relative;
    flex-shrink: 0;
  }

  .hand-section h3 {
    margin-bottom: 16px;
  }


  @keyframes fadeInDown {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .hand-display {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(45px, 1fr));
    gap: 8px;
    max-width: 100%;
    justify-items: center;
  }

  .domino-in-hand {
    animation: handFadeIn 0.4s cubic-bezier(0.4, 0, 0.2, 1) both;
    animation-delay: var(--delay);
  }

  @keyframes handFadeIn {
    from {
      opacity: 0;
      transform: translateY(10px) rotate(-5deg) scale(0.9);
    }
    to {
      opacity: 1;
      transform: translateY(0) rotate(0) scale(1);
    }
  }

  .waiting-indicator {
    padding: 20px;
    text-align: center;
    color: #666;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    animation: pulse 1.5s ease-in-out infinite;
    width: 100%;
    background: none;
    border: none;
    font-family: inherit;
  }
  
  .waiting-indicator.clickable {
    cursor: pointer;
    transition: transform 0.2s ease;
  }
  
  .waiting-indicator.clickable:hover {
    transform: scale(1.05);
  }
  
  .waiting-indicator.clickable:active {
    transform: scale(0.98);
  }

  @keyframes pulse {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
  }

  .robot-icon {
    font-size: 20px;
  }

  .waiting-text {
    font-size: 14px;
  }
  
  .skip-hint {
    font-size: 12px;
    opacity: 0.7;
    margin-left: 4px;
  }

  /* Bidding table styles */
  /* Compact bid status styles */
  .compact-bid-status {
    margin: 16px;
    padding: 12px;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 12px;
    border: 1px solid #e2e8f0;
  }

  .bid-chips {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
  }

  .bid-chip {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: white;
    border-radius: 20px;
    border: 2px solid #e2e8f0;
    font-size: 14px;
    position: relative;
    transition: all 0.2s ease;
  }

  .bid-chip.high-bidder {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    border-color: #059669;
    font-weight: 600;
    transform: scale(1.05);
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
  }

  .bid-chip.dealer {
    border-color: #8b5cf6;
  }

  .chip-player {
    font-weight: 600;
    color: #64748b;
  }

  .bid-chip.high-bidder .chip-player {
    color: white;
  }

  .chip-bid {
    color: #1e293b;
  }

  .bid-chip.high-bidder .chip-bid {
    color: white;
  }

  .dealer-badge {
    position: absolute;
    top: -8px;
    right: -8px;
    background: #8b5cf6;
    color: white;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: bold;
    border: 2px solid white;
  }

  .high-bid-indicator {
    text-align: center;
    margin-top: 8px;
    font-size: 13px;
    color: #64748b;
    font-weight: 600;
  }

</style>