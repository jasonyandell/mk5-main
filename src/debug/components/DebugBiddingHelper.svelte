<script lang="ts">
  import type { GameState, Player } from '../../game/types';
  import { getStrongestSuits } from '../../game/core/suit-analysis';
  
  interface Props {
    gameState: GameState;
  }
  
  let { gameState }: Props = $props();
  
  function getBiddingAnalysis(player: Player) {
    if (!player.suitAnalysis) {
      return { suggestion: 'Pass', reason: 'No suit analysis available', strongSuits: [] };
    }
    
    const analysis = player.suitAnalysis;
    const strongestSuits = getStrongestSuits(analysis);
    const doublesCount = analysis.count.doubles;
    const handStrength = calculateHandStrength(player);
    
    // Get top 3 strongest suits for display
    const topSuits = strongestSuits.slice(0, 3).filter(suit => analysis.count[suit as 0 | 1 | 2 | 3 | 4 | 5 | 6] > 0);
    
    let suggestion = 'Pass';
    let reason = 'Weak hand';
    
    if (doublesCount >= 4) {
      suggestion = 'Plunge (4+ marks)';
      reason = `${doublesCount} doubles - guaranteed laydown potential`;
    } else if (handStrength >= 35) {
      suggestion = '2 marks (84 points)';
      reason = `Very strong hand (${handStrength}/42 estimated)`;
    } else if (handStrength >= 32) {
      suggestion = '1 mark (42 points)';
      reason = `Strong hand (${handStrength}/42 estimated)`;
    } else if (handStrength >= 30) {
      suggestion = '30-35 points';
      reason = `Decent hand (${handStrength}/42 estimated)`;
    } else if (doublesCount >= 3 && handStrength >= 25) {
      suggestion = 'Splash (2-3 marks)';
      reason = `${doublesCount} doubles with reasonable support`;
    }
    
    return {
      suggestion,
      reason,
      strongSuits: topSuits,
      handStrength,
      doublesCount
    };
  }
  
  function calculateHandStrength(player: Player): number {
    if (!player.suitAnalysis) return 0;
    
    const analysis = player.suitAnalysis;
    let strength = 0;
    
    // Add points for strong suits (lots of dominoes in one suit)
    for (let suit = 0; suit <= 6; suit++) {
      const count = analysis.count[suit as 0 | 1 | 2 | 3 | 4 | 5 | 6];
      if (count >= 4) strength += count * 2; // Strong suit bonus
      else if (count >= 2) strength += count; // Moderate suit
    }
    
    // Add points for doubles (trump potential)
    strength += analysis.count.doubles * 3;
    
    // Add base value (each domino has some value)
    strength += player.hand.length * 2;
    
    return Math.min(strength, 42); // Cap at maximum possible
  }
  
  function getSuitName(suit: number): string {
    const names = ['0s', '1s', '2s', '3s', '4s', '5s', '6s'];
    return names[suit] || 'Unknown';
  }
  
  const currentPlayer = $derived(gameState.players[gameState.currentPlayer]);
  const biddingAnalysis = $derived(getBiddingAnalysis(currentPlayer));
</script>

{#if gameState.phase === 'bidding'}
  <div class="bidding-helper">
    <h3>Bidding Helper - P{gameState.currentPlayer}</h3>
    
    <div class="analysis-section">
      <div class="suggestion">
        <strong>Suggestion:</strong> <span class="bid-suggestion">{biddingAnalysis.suggestion}</span>
      </div>
      <div class="reason">
        <strong>Reason:</strong> {biddingAnalysis.reason}
      </div>
    </div>
    
    <div class="hand-analysis">
      <div class="strength-info">
        <span class="label">Hand Strength:</span>
        <span class="strength-bar">
          <span class="strength-fill" style="width: {((biddingAnalysis.handStrength || 0) / 42) * 100}%"></span>
          <span class="strength-text">{biddingAnalysis.handStrength || 0}/42</span>
        </span>
      </div>
      
      <div class="suits-info">
        <span class="label">Strongest Suits:</span>
        <div class="strong-suits">
          {#each biddingAnalysis.strongSuits as suit}
            <span class="suit-badge">
              {getSuitName(suit)} ({currentPlayer.suitAnalysis?.count[suit as 0 | 1 | 2 | 3 | 4 | 5 | 6]})
            </span>
          {/each}
          {#if (biddingAnalysis.doublesCount || 0) > 0}
            <span class="suit-badge doubles">
              Doubles ({biddingAnalysis.doublesCount || 0})
            </span>
          {/if}
        </div>
      </div>
    </div>
    
    <div class="strategy-notes">
      <h4>Strategy Notes:</h4>
      <ul>
        {#if (biddingAnalysis.doublesCount || 0) >= 4}
          <li>🔥 Perfect plunge hand - you can likely win all 7 tricks</li>
        {:else if (biddingAnalysis.doublesCount || 0) >= 3}
          <li>💪 Strong trump potential - consider splash or aggressive bidding</li>
        {/if}
        
        {#if biddingAnalysis.strongSuits.length > 0}
          <li>🎯 Focus trump on {getSuitName(biddingAnalysis.strongSuits[0])} for maximum control</li>
        {/if}
        
        {#if (biddingAnalysis.handStrength || 0) < 25}
          <li>⚠️ Weak hand - only bid if forced or with special contracts</li>
        {:else if (biddingAnalysis.handStrength || 0) > 35}
          <li>🚀 Very strong hand - be aggressive in bidding</li>
        {/if}
      </ul>
    </div>
  </div>
{/if}

<style>
  .bidding-helper {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 4px;
    padding: 12px;
    font-size: 11px;
    margin-bottom: 8px;
  }
  
  .bidding-helper h3 {
    margin: 0 0 8px 0;
    font-size: 12px;
    color: #856404;
    font-weight: 600;
  }
  
  .analysis-section {
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #ffeaa7;
  }
  
  .suggestion {
    margin-bottom: 4px;
  }
  
  .bid-suggestion {
    color: #28a745;
    font-weight: 600;
    font-size: 12px;
  }
  
  .reason {
    color: #6c757d;
    font-size: 10px;
  }
  
  .hand-analysis {
    margin-bottom: 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  
  .strength-info {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  
  .label {
    font-weight: 600;
    color: #856404;
    min-width: 80px;
    font-size: 10px;
  }
  
  .strength-bar {
    position: relative;
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    height: 16px;
    width: 120px;
    overflow: hidden;
  }
  
  .strength-fill {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
    transition: width 0.3s ease;
  }
  
  .strength-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 9px;
    font-weight: 600;
    color: #495057;
    text-shadow: 0 0 2px white;
  }
  
  .suits-info {
    display: flex;
    align-items: flex-start;
    gap: 8px;
  }
  
  .strong-suits {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
  }
  
  .suit-badge {
    background: #e9ecef;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 9px;
    font-weight: 500;
    color: #495057;
  }
  
  .suit-badge.doubles {
    background: #ffd700;
    color: #000;
    font-weight: 600;
  }
  
  .strategy-notes {
    background: #f8f9fa;
    padding: 8px;
    border-radius: 3px;
    border: 1px solid #dee2e6;
  }
  
  .strategy-notes h4 {
    margin: 0 0 6px 0;
    font-size: 10px;
    color: #495057;
    font-weight: 600;
  }
  
  .strategy-notes ul {
    margin: 0;
    padding-left: 16px;
    list-style-type: none;
  }
  
  .strategy-notes li {
    font-size: 9px;
    color: #6c757d;
    margin-bottom: 3px;
    position: relative;
  }
  
  .strategy-notes li::before {
    content: '';
    position: absolute;
    left: -12px;
    top: 50%;
    transform: translateY(-50%);
    width: 4px;
    height: 4px;
    background: #6c757d;
    border-radius: 50%;
  }
</style>