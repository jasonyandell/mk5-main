#!/usr/bin/env npx tsx

/**
 * Bid Calibration Validation Script
 * 
 * Runs simulated games to validate whether AI bidding is properly calibrated.
 * Tracks success rates for different bid levels and hand strengths.
 */

import { createInitialState, getNextStates, executeAction } from '../src/game';
import { BeginnerAIStrategy } from '../src/game/controllers/strategies';
import { calculateHandStrengthWithTrump } from '../src/game/ai/hand-strength';
import type { GameState, TrumpSelection } from '../src/game/types';

// Track bid outcomes
interface BidOutcome {
  bidValue: number;
  bidType: 'points' | 'marks';
  marksValue?: number; // How many marks if it's a marks bid
  handStrength: number;
  actualScore: number;
  madeBid: boolean;
  margin: number;
  trump: TrumpSelection;
}

const bidOutcomes: BidOutcome[] = [];
let totalHands = 0;
const targetHands = 1000; // Full validation run
let laydownsFound = 0;
const maxLaydowns = 25; // Stop after finding 25 laydowns

// Create AI strategy instance
const aiStrategy = new BeginnerAIStrategy();


/**
 * Run a single hand and track the bid outcome
 */
async function playHand(initialState: GameState) {
  let gameState = initialState;
  
  // Track the winning bid info
  interface BidInfo {
    playerId: number;
    teamId: number;
    bidValue: number;
    bidType?: 'points' | 'marks';
    marksValue?: number;
    handStrength: number;
  }
  
  let winningBidInfo: BidInfo | null = null;
  let laydownBidInfo: BidInfo | null = null; // Track laydown separately - MUST be cleared each hand!
  let foundLaydownThisHand = false; // Track if we found a laydown in this specific hand
  
  // Play through ONE hand (stop after scoring, not game_end)
  let handComplete = false;
  let handsPlayed = 0;
  
  while (!handComplete && gameState.phase !== 'game_end') {
    const transitions = getNextStates(gameState);
    
    if (transitions.length === 0) {
      // No valid moves, shouldn't happen but let's handle it
      break;
    }
    
    // During bidding, capture bid info
    if (gameState.phase === 'bidding') {
      const currentPlayer = gameState.players[gameState.currentPlayer];
      // Use null like the AI does to force proper trump analysis
      const handStrength = calculateHandStrengthWithTrump(currentPlayer.hand, null);
      
      // Track laydowns with logging
      if (handStrength === 999) {
        if (!foundLaydownThisHand) { // Only count once per hand
          foundLaydownThisHand = true;
          laydownsFound++;
          console.log(`🎯 LAYDOWN #${laydownsFound} at hand ${totalHands + 1}: Player ${gameState.currentPlayer} with ${currentPlayer.hand.map(d => `${d.high}-${d.low}`).join(', ')}`);
        }
      }
      
      
      // Choose action
      const chosenTransition = aiStrategy.chooseAction(gameState, transitions);
      
      // Remove debug logging for cleaner output
      
      if (chosenTransition && chosenTransition.action.type === 'bid') {
        const bidAction = chosenTransition.action;
        
        // Store laydown bid info separately
        if (handStrength === 999) {
          if (bidAction.bid === 'points' && bidAction.value) {
            laydownBidInfo = {
              playerId: gameState.currentPlayer,
              teamId: currentPlayer.teamId,
              bidValue: bidAction.value,
              bidType: 'points',
              handStrength: handStrength
            };
          } else if (bidAction.bid === 'marks' && bidAction.value) {
            laydownBidInfo = {
              playerId: gameState.currentPlayer,
              teamId: currentPlayer.teamId,
              bidValue: 42, // Marks bids always require exactly 42 points
              bidType: 'marks',
              marksValue: bidAction.value,
              handStrength: handStrength
            };
            // Tracked silently
          }
        }
      }
      
      // Execute the transition
      if (chosenTransition) {
        gameState = executeAction(gameState, chosenTransition.action);
      } else {
        // AI couldn't choose, pick first valid
        gameState = executeAction(gameState, transitions[0].action);
      }
    } else if (gameState.phase === 'trump_selection' && !winningBidInfo) {
      // Bidding just ended - capture the actual winner (only once!)
      if (gameState.winningBidder !== -1) {
        const winningPlayer = gameState.players[gameState.winningBidder];
        const winningBid = gameState.currentBid;
        
        // If the laydown player won, use their stored hand strength
        if (laydownBidInfo && laydownBidInfo.playerId === gameState.winningBidder) {
          winningBidInfo = laydownBidInfo;
        } else {
          // Calculate the winning player's hand strength using null for proper trump analysis
          const winnerStrength = calculateHandStrengthWithTrump(
            winningPlayer.hand, 
            null
          );
          
          
          winningBidInfo = {
            playerId: gameState.winningBidder,
            teamId: winningPlayer.teamId,
            bidValue: winningBid.type === 'marks' ? 42 : (winningBid.value || 0),
            bidType: winningBid.type === 'marks' ? 'marks' : 'points',
            marksValue: winningBid.type === 'marks' ? winningBid.value : undefined,
            handStrength: winnerStrength
          };
          
        }
      }
      
      // Continue with trump selection
      const chosenTransition = aiStrategy.chooseAction(gameState, transitions);
      if (chosenTransition) {
        gameState = executeAction(gameState, chosenTransition.action);
      } else {
        gameState = executeAction(gameState, transitions[0].action);
      }
    } else {
      // Not bidding - just play normally
      const chosenTransition = aiStrategy.chooseAction(gameState, transitions);
      
      if (chosenTransition) {
        gameState = executeAction(gameState, chosenTransition.action);
      } else {
        // AI couldn't choose, pick first valid
        gameState = executeAction(gameState, transitions[0].action);
      }
      
      // Check if we've completed scoring (one hand is done)
      if (gameState.phase === 'scoring' && chosenTransition?.action.type === 'agree-score-hand') {
        // If all players agreed to score, the next phase will be bidding for a new hand
        const nextState = executeAction(gameState, chosenTransition.action);
        if (nextState.phase === 'bidding' && handsPlayed === 0) {
          // We just finished scoring our first hand
          handsPlayed++;
          handComplete = true;
          gameState = nextState;
        }
      }
    }
    
  }
  
  // After the hand is complete (reached scoring phase), track the outcome ONCE
  if ((gameState.phase === 'scoring' || gameState.phase === 'game_end') && winningBidInfo && !handComplete) {
    handComplete = true; // Mark as complete to avoid double-counting
    // Use the game's own score tracking
    const biddingTeam = winningBidInfo.teamId;
    const teamScore = gameState.teamScores[biddingTeam];
    
    // Record the outcome
    const madeBid = teamScore >= winningBidInfo.bidValue;
    const margin = teamScore - winningBidInfo.bidValue;
    
    // Track laydown results silently
    
    
    
    bidOutcomes.push({
      bidValue: winningBidInfo.bidValue,
      bidType: winningBidInfo.bidType || 'points',
      marksValue: winningBidInfo.marksValue,
      handStrength: winningBidInfo.handStrength,
      actualScore: teamScore,
      madeBid: madeBid,
      margin: margin,
      trump: gameState.trump
    });
    
    totalHands++;
    
    // Progress indicator
    if (totalHands % 100 === 0) {
      console.log(`Completed ${totalHands}/${targetHands} hands...`);
    }
  }
}

/**
 * Run the simulation
 */
async function runSimulation() {
  console.log(`Starting bid validation simulation for ${targetHands} hands...`);
  console.log(`Looking for laydown hands...\n`);
  
  // Run full simulation (stop early if we find enough laydowns)
  while (totalHands < targetHands && laydownsFound < maxLaydowns) {
    // Create a new game with all AI players
    const gameState = createInitialState({
      playerTypes: ['ai', 'ai', 'ai', 'ai'],
      shuffleSeed: Date.now() + Math.random() * 1000000
    });
    
    // Play one hand
    await playHand(gameState);
  }
  
  // Analyze results
  analyzeResults();
}

/**
 * Analyze and report the results
 */
function analyzeResults() {
  console.log('\n=== BID CALIBRATION ANALYSIS ===\n');
  
  if (bidOutcomes.length === 0) {
    console.log('No bid outcomes recorded. Check if games are completing properly.');
    return;
  }
  
  // Group by bid level
  const bidLevels: { [key: number]: {
    total: number;
    made: number;
    totalMargin: number;
    strengths: number[];
    margins: number[];
  }} = {};
  
  for (const outcome of bidOutcomes) {
    // For display purposes, group marks bids by their marks value
    const level = outcome.bidType === 'marks' 
      ? `${outcome.marksValue}m`
      : outcome.bidValue.toString();
    if (!bidLevels[level]) {
      bidLevels[level] = {
        total: 0,
        made: 0,
        totalMargin: 0,
        strengths: [],
        madeStrengths: [],  // Track strengths of successful bids
        failedStrengths: [], // Track strengths of failed bids
        margins: []
      };
    }
    
    bidLevels[level].total++;
    if (outcome.madeBid) {
      bidLevels[level].made++;
      bidLevels[level].madeStrengths.push(outcome.handStrength);
    } else {
      bidLevels[level].failedStrengths.push(outcome.handStrength);
    }
    bidLevels[level].totalMargin += outcome.margin;
    bidLevels[level].strengths.push(outcome.handStrength);
    bidLevels[level].margins.push(outcome.margin);
  }
  
  // Report by bid level
  console.log('Success Rate by Bid Level:');
  console.log('Bid | Count | Success % | Avg Margin | Avg Strength | Made Str | Failed Str');
  console.log('----+-------+-----------+------------+--------------+----------+------------');
  
  const sortedLevels = Object.keys(bidLevels).sort((a, b) => {
    // Handle marks bids (e.g., "1m", "2m")
    const aIsMarks = a.endsWith('m');
    const bIsMarks = b.endsWith('m');
    
    if (aIsMarks && bIsMarks) {
      return parseInt(a) - parseInt(b);
    }
    if (aIsMarks) return 1; // Marks after points
    if (bIsMarks) return -1;
    
    return parseInt(a) - parseInt(b);
  });
  for (const level of sortedLevels) {
    const data = bidLevels[level]; // Use level directly, not parseInt
    const successRate = (data.made / data.total * 100).toFixed(1);
    const avgMargin = (data.totalMargin / data.total).toFixed(1);
    const avgStrength = (data.strengths.reduce((a, b) => a + b, 0) / data.strengths.length).toFixed(1);
    
    // Calculate average strengths for made vs failed bids
    const avgMadeStr = data.madeStrengths.length > 0 
      ? (data.madeStrengths.reduce((a, b) => a + b, 0) / data.madeStrengths.length).toFixed(1)
      : 'N/A';
    const avgFailedStr = data.failedStrengths.length > 0
      ? (data.failedStrengths.reduce((a, b) => a + b, 0) / data.failedStrengths.length).toFixed(1)
      : 'N/A';
    
    console.log(`${level.padEnd(3)} | ${data.total.toString().padEnd(5)} | ${successRate.padStart(8)}% | ${avgMargin.padStart(10)} | ${avgStrength.padStart(12)} | ${avgMadeStr.padStart(8)} | ${avgFailedStr.padStart(10)}`);
  }
  
  // Group by strength ranges
  console.log('\n\nSuccess Rate by Hand Strength:');
  console.log('Strength Range | Count | Avg Bid | Success %');
  console.log('---------------+-------+---------+----------');
  
  const strengthRanges = [
    { min: 0, max: 25, label: '0-25' },
    { min: 25, max: 35, label: '25-35' },
    { min: 35, max: 45, label: '35-45' },
    { min: 45, max: 60, label: '45-60' },
    { min: 60, max: 80, label: '60-80' },
    { min: 80, max: 100, label: '80-100' },
    { min: 100, max: 999, label: '100+' }
  ];
  
  for (const range of strengthRanges) {
    const inRange = bidOutcomes.filter(o => 
      o.handStrength >= range.min && o.handStrength < range.max
    );
    
    if (inRange.length > 0) {
      const avgBid = (inRange.reduce((a, b) => a + b.bidValue, 0) / inRange.length).toFixed(1);
      const successRate = (inRange.filter(o => o.madeBid).length / inRange.length * 100).toFixed(1);
      
      console.log(`${range.label.padEnd(14)} | ${inRange.length.toString().padEnd(5)} | ${avgBid.padStart(7)} | ${successRate.padStart(8)}%`);
    }
  }
  
  // Overall statistics
  const totalOutcomes = bidOutcomes.length;
  const totalMade = bidOutcomes.filter(o => o.madeBid).length;
  const overallSuccess = (totalMade / totalOutcomes * 100).toFixed(1);
  const avgMargin = (bidOutcomes.reduce((a, b) => a + b.margin, 0) / totalOutcomes).toFixed(1);
  
  console.log('\n\nOverall Statistics:');
  console.log(`Total Bids: ${totalOutcomes}`);
  console.log(`Overall Success Rate: ${overallSuccess}%`);
  console.log(`Average Margin: ${avgMargin}`);
  console.log(`\nLaydowns found: ${laydownsFound}`);
  
  if (laydownsFound >= maxLaydowns) {
    console.log(`(Stopped early after finding ${maxLaydowns} laydowns)`);
  }
  
  // Correlation analysis
  const madeBids = bidOutcomes.filter(o => o.madeBid);
  const failedBids = bidOutcomes.filter(o => !o.madeBid);
  
  if (madeBids.length > 0 && failedBids.length > 0) {
    const avgStrengthMade = madeBids.reduce((a, b) => a + b.handStrength, 0) / madeBids.length;
    const avgStrengthFailed = failedBids.reduce((a, b) => a + b.handStrength, 0) / failedBids.length;
    
    console.log(`\nAverage Strength - Made Bids: ${avgStrengthMade.toFixed(1)}`);
    console.log(`Average Strength - Failed Bids: ${avgStrengthFailed.toFixed(1)}`);
  }
}

// Run the simulation
runSimulation();