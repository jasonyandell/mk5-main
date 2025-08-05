<script lang="ts">
  import { gamePhase, availableActions, gameActions, teamInfo, biddingInfo, currentPlayer } from '../../stores/gameStore';
  import type { StateTransition, Domino as DominoType } from '../../game/types';
  import Domino from './Domino.svelte';
  import { slide } from 'svelte/transition';
  import { createEventDispatcher } from 'svelte';
  
  const dispatch = createEventDispatcher();

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
      
      // If we just selected trump, switch back to play panel
      if (action.id.startsWith('trump-')) {
        // Small delay to let the state update
        setTimeout(() => {
          dispatch('switchToPlay');
        }, 100);
      }
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

  // Current player's hand
  $: playerHand = $currentPlayer.hand || [];

  // State for hovering during bidding/trump selection
  let hoveredSuit: number | 'doubles' | null = null;
  let hoveredDomino: DominoType | null = null;
  let hoveredTrumpButton: string | null = null;
  
  // State for expandable team status
  let teamStatusExpanded = false;

  // Check if we should enable suit highlighting
  $: enableSuitHighlighting = $gamePhase === 'bidding' || $gamePhase === 'trump_selection';

  // Handle trump button hover
  function handleTrumpHover(action: StateTransition, isEntering: boolean) {
    if (!isEntering) {
      hoveredTrumpButton = null;
      hoveredSuit = null;
      return;
    }

    hoveredTrumpButton = action.id;
    
    // Extract suit from action id (e.g., "trump-blanks" -> 0, "trump-fives" -> 5)
    if (action.id === 'trump-doubles') {
      hoveredSuit = 'doubles';
    } else if (action.id.startsWith('trump-')) {
      const suitMap: Record<string, number> = {
        'trump-blanks': 0,
        'trump-ones': 1,
        'trump-twos': 2,
        'trump-threes': 3,
        'trump-fours': 4,
        'trump-fives': 5,
        'trump-sixes': 6
      };
      hoveredSuit = suitMap[action.id] ?? null;
    } else {
      hoveredSuit = null;
    }
  }

  // Handle domino hover
  function handleDominoHover(domino: DominoType, event: any, isEntering: boolean) {
    if (!enableSuitHighlighting) {
      hoveredSuit = null;
      return;
    }

    if (!isEntering) {
      hoveredSuit = null;
      hoveredDomino = null;
      return;
    }

    hoveredDomino = domino;

    // For doubles, always use the suit value
    if (domino.high === domino.low) {
      hoveredSuit = domino.high;
      return;
    }

    // For non-doubles, try to determine which half was hovered
    if (event && event.currentTarget) {
      const target = event.currentTarget as HTMLElement;
      const rect = target.getBoundingClientRect();
      const relativeY = event.clientY - rect.top;
      const halfwayPoint = rect.height / 2;
      const isTopHalf = relativeY < halfwayPoint;
      hoveredSuit = isTopHalf ? domino.high : domino.low;
    } else {
      // Fallback: just use the higher value
      hoveredSuit = Math.max(domino.high, domino.low);
    }
  }


  // Get highlight color based on suit
  function getHighlightColor(domino: DominoType): string | null {
    if (hoveredSuit === null) return null;
    
    const isHoveringDouble = hoveredDomino && hoveredDomino.high === hoveredDomino.low;
    
    if (isHoveringDouble && domino.high === domino.low) {
      return 'doubles';
    } else if (typeof hoveredSuit === 'number' && (domino.high === hoveredSuit || domino.low === hoveredSuit)) {
      return `suit-${hoveredSuit}`;
    }
    
    return null;
  }
</script>

<div class="action-panel">
  {#if ($gamePhase === 'bidding' || $gamePhase === 'trump_selection') && playerHand.length > 0}
    <div class="hand-section">
      <h3>Your Hand</h3>
      {#if hoveredSuit !== null && enableSuitHighlighting}
        <div class="suit-highlight-badge" class:suit-badge-0={hoveredSuit === 0}
             class:suit-badge-1={hoveredSuit === 1}
             class:suit-badge-2={hoveredSuit === 2}
             class:suit-badge-3={hoveredSuit === 3}
             class:suit-badge-4={hoveredSuit === 4}
             class:suit-badge-5={hoveredSuit === 5}
             class:suit-badge-6={hoveredSuit === 6}
             class:suit-badge-doubles={hoveredDomino && hoveredDomino.high === hoveredDomino.low}>
          {(() => {
            if (hoveredSuit === 'doubles') {
              return 'Doubles';
            }
            const suitName = ['Blanks', 'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes'][hoveredSuit];
            return suitName;
          })()}
        </div>
      {/if}
      <div class="hand-display">
        {#each playerHand as domino, i (domino.high + '-' + domino.low)}
          {@const highlight = (() => {
            if (hoveredSuit === null) return null;
            
            // If hovering over trump button
            if (hoveredTrumpButton) {
              if (hoveredSuit === 'doubles') {
                return domino.high === domino.low ? 'primary' : null;
              } else if (typeof hoveredSuit === 'number') {
                return (domino.high === hoveredSuit || domino.low === hoveredSuit) ? 'primary' : null;
              }
            }
            
            // If hovering over domino
            const isHoveringDouble = hoveredDomino && hoveredDomino.high === hoveredDomino.low;
            
            // Always prioritize suit highlighting
            if (domino.high === hoveredSuit || domino.low === hoveredSuit) {
              return 'primary';
            }
            
            // If hovering a double, other doubles get secondary highlight
            if (isHoveringDouble && domino.high === domino.low) {
              return 'secondary';
            }
            return null;
          })()}
          <div class="domino-in-hand" style="--delay: {i * 30}ms">
            <Domino
              {domino}
              small={true}
              showPoints={true}
              {highlight}
              on:mouseenter={(e) => handleDominoHover(domino, e, true)}
              on:mousemove={(e) => handleDominoHover(domino, e, true)}
              on:mouseleave={(e) => handleDominoHover(domino, e, false)}
            />
          </div>
        {/each}
      </div>
    </div>
  {/if}

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
              class="action-button primary trump-button"
              class:trump-hovering={hoveredTrumpButton === action.id}
              on:click={() => executeAction(action)}
              on:mouseenter={() => handleTrumpHover(action, true)}
              on:mouseleave={() => handleTrumpHover(action, false)}
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

  <div class="team-status" class:expanded={teamStatusExpanded}>
    <button class="team-status-toggle" on:click={() => teamStatusExpanded = !teamStatusExpanded}>
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
              <span>{$biddingInfo.currentBid.value || 0} by Player {$biddingInfo.winningBidder + 1}</span>
            </div>
            {#if $gamePhase === 'playing'}
              <div class="bid-info-row">
                <span>Points Needed:</span>
                <span>{Math.max(0, ($biddingInfo.currentBid.value || 0) - $teamInfo.scores[Math.floor($biddingInfo.winningBidder / 2)])}</span>
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

  h2 {
    margin: 0;
    font-size: 20px;
    font-weight: 700;
    color: #1e293b;
    text-align: center;
    margin-bottom: 20px;
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

  .trump-button.trump-hovering {
    transform: translateY(-3px) scale(1.05);
    box-shadow: 
      0 8px 24px rgba(139, 92, 246, 0.5),
      0 0 40px rgba(139, 92, 246, 0.2);
    z-index: 10;
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

  .status-content {
    background: white;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    border: 1px solid rgba(226, 232, 240, 0.5);
  }

  .score-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    padding: 8px 0;
  }

  .team-name {
    font-weight: 700;
    font-size: 15px;
    color: #1e293b;
  }

  .score {
    font-size: 15px;
    color: #64748b;
    font-weight: 600;
  }

  .bid-info {
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid #f1f5f9;
  }

  .bid-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 14px;
    margin-bottom: 8px;
    color: #64748b;
  }

  .bid-row span:last-child {
    font-weight: 600;
    color: #1e293b;
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

  .suit-highlight-badge {
    position: absolute;
    top: 8px;
    right: 16px;
    padding: 4px 12px;
    background: linear-gradient(135deg, #fbbf24, #f59e0b);
    color: white;
    font-size: 12px;
    font-weight: 700;
    border-radius: 16px;
    box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3);
    animation: fadeInDown 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }

  /* Suit-specific badge colors */
  .suit-badge-0 { /* Blanks */
    background: linear-gradient(135deg, #9ca3af, #6b7280);
    box-shadow: 0 2px 8px rgba(107, 114, 128, 0.3);
  }

  .suit-badge-1 { /* Ones */
    background: linear-gradient(135deg, #f87171, #ef4444);
    box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
  }

  .suit-badge-2 { /* Twos */
    background: linear-gradient(135deg, #fb923c, #f97316);
    box-shadow: 0 2px 8px rgba(249, 115, 22, 0.3);
  }

  .suit-badge-3 { /* Threes */
    background: linear-gradient(135deg, #fbbf24, #f59e0b);
    box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3);
  }

  .suit-badge-4 { /* Fours */
    background: linear-gradient(135deg, #a3e635, #84cc16);
    box-shadow: 0 2px 8px rgba(132, 204, 22, 0.3);
  }

  .suit-badge-5 { /* Fives */
    background: linear-gradient(135deg, #60a5fa, #3b82f6);
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
  }

  .suit-badge-6 { /* Sixes */
    background: linear-gradient(135deg, #c084fc, #a855f7);
    box-shadow: 0 2px 8px rgba(168, 85, 247, 0.3);
  }

  .suit-badge-doubles { /* Doubles */
    background: linear-gradient(135deg, #f472b6, #ec4899);
    box-shadow: 0 2px 8px rgba(236, 72, 153, 0.3);
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
</style>