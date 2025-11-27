import type { FilteredGameState, StateTransition, Domino, Play, Trick, TrumpSelection, Bid, GamePhase, LedSuit } from './types';
import { DOUBLES_AS_TRUMP } from './types';
import { calculateTrickWinner } from './core/scoring';
import { GAME_PHASES } from './index';
import { dominoHasSuit } from './core/dominoes';
import { getSuitName, getTrumpDisplay } from './game-terms';

// Represents a player's bid status for UI display
export interface BidStatus {
  player: number;
  bid: Bid | null;
  isHighBidder: boolean;
  isDealer: boolean;
  isCurrentTurn: boolean;
  isThinking: boolean;
}

// Represents a domino in the player's hand with UI metadata
export interface HandDomino {
  domino: Domino;
  isPlayable: boolean;
  tooltip: string;
}

// Represents a play action that can be executed
export interface PlayAction {
  id: string;
  label: string;
  action: StateTransition;
}

// Represents the current trick display state
export interface TrickDisplay {
  plays: (Play | null)[];  // Indexed by player position (0-3)
  winner: number;  // -1 if not complete
  isComplete: boolean;
  currentPlayer: number;
}

// Represents hand scoring results
export interface HandResults {
  team0Points: number;
  team1Points: number;
  bidAmount: number;
  biddingTeam: number;
  bidMade: boolean;
  winningTeam: number;
  biddingPlayer: number;
  // Perspective-aware messages for each player
  resultMessage: string;  // Full message with emoji (for backwards compatibility)
  resultText: string;  // Text without emoji
  isSuccess: boolean;  // true for success, false for failure
  teamLabel: string;  // e.g., "US" or "THEM" for the bidding team
}

// Complete view projection for UI rendering
export interface ViewProjection {
  // Game phase and flow
  phase: GamePhase;
  currentPlayer: number;
  isPlayer0Turn: boolean;
  isPerspectiveTurn: boolean;
  perspectiveIndex: number;
  canAct: boolean;
  
  // Player's hand
  hand: HandDomino[];
  
  // Available actions grouped by type
  actions: {
    bidding: StateTransition[];
    trump: StateTransition[];
    proceed: StateTransition | null;  // Single proceed action if available
    consensus: StateTransition | null;  // Consensus action for player 0
  };
  
  // Bidding state
  bidding: {
    currentBid: { player: number; value?: number; type: string };
    winningBidder: number;
    dealer: number;
    playerStatuses: BidStatus[];  // One per player
    showBiddingTable: boolean;
  };
  
  // Trump state
  trump: {
    selection: TrumpSelection;
    display: string;  // Human-readable trump
  };
  
  // Trick state
  trick: {
    current: TrickDisplay;
    completed: Trick[];
    number: number;  // Current trick number (1-7)
    ledSuit: number;  // -1 if no suit led
    ledSuitDisplay: string | null;
    trickLeader: number | null;  // Player who led the current trick
  };
  
  // Scoring state
  scoring: {
    teamScores: [number, number];
    teamMarks: [number, number];
    currentHandPoints: [number, number];  // Current hand's trick points
    handResults: HandResults | null;  // Only during scoring phase
  };
  
  // UI state flags
  ui: {
    showActionPanel: boolean;
    showBiddingTable: boolean;
    isWaiting: boolean;
    waitingOnPlayer: number;  // -1 if not waiting
    isAIThinking: boolean;
    activeView: 'game' | 'actions';  // Which panel to show
  };
  
  // Tooltips and labels
  tooltips: {
    proceedAction: string | null;
    skipAction: string;
  };
}

// Create view projection from game state and available actions
export function createViewProjection(
  gameState: FilteredGameState,
  availableActions: StateTransition[],
  options: {
    isTestMode?: boolean;
    viewingPlayerIndex?: number;
    canAct?: boolean;
    isAIControlled?: (player: number) => boolean;
  } = {}
): ViewProjection {
  const {
    isTestMode = false,
    viewingPlayerIndex,
    canAct = true,
    isAIControlled = () => false
  } = options;

  const phase = gameState.phase;
  const currentPlayer = gameState.currentPlayer;
  const perspectiveIndex = viewingPlayerIndex ?? 0;
  const isPerspectiveTurn = currentPlayer === perspectiveIndex;
  
  // Group actions by type
  const biddingActions = availableActions.filter(a =>
    a.id.startsWith('bid-') || a.id === 'pass' || a.id === 'redeal'
  );
  
  const trumpActions = availableActions.filter(a =>
    a.id.startsWith('trump-')
  );
  
  // Extract playable dominoes
  const playableDominoIds = new Set<string>();
  availableActions
    .filter(a => a.id.startsWith('play-'))
    .forEach(a => {
      const dominoId = a.id.replace('play-', '');
      playableDominoIds.add(dominoId);
      const parts = dominoId.split('-');
      if (parts.length === 2) {
        playableDominoIds.add(`${parts[1]}-${parts[0]}`);
      }
    });
  
  // Get player hand (player 0 unless in test mode)
  const playerHand = isTestMode
    ? (gameState.players[currentPlayer]?.hand || [])
    : (gameState.players[perspectiveIndex]?.hand || []);
  
  // Build hand with metadata
  const hand: HandDomino[] = playerHand.map(domino => ({
    domino,
    isPlayable: playableDominoIds.has(`${domino.high}-${domino.low}`) ||
                playableDominoIds.has(`${domino.low}-${domino.high}`),
    tooltip: getDominoTooltip(domino, gameState, playableDominoIds)
  }));
  
  // Find proceed and consensus actions
  const alwaysShowActions = ['complete-trick', 'score-hand'];
  const proceedAction = availableActions.find(a =>
    alwaysShowActions.includes(a.id) ||
    (isPerspectiveTurn && (a.id === 'start-hand' || a.id === 'continue' || a.id === 'next-trick'))
  ) || null;

  const consensusAction = availableActions.find(a =>
    a.id === `agree-trick-p${perspectiveIndex}` || a.id === `agree-score-p${perspectiveIndex}`
  ) || null;
  
  // Build bidding statuses
  const biddingStatuses: BidStatus[] = [0, 1, 2, 3].map(playerId => ({
    player: playerId,
    bid: gameState.bids.find(b => b.player === playerId) || null,
    isHighBidder: gameState.currentBid.player === playerId,
    isDealer: gameState.dealer === playerId,
    isCurrentTurn: currentPlayer === playerId,
    isThinking: currentPlayer === playerId && isAIControlled(playerId)
  }));
  
  // Trump display
  const trumpDisplay = getTrumpDisplay(gameState.trump);
  
  // Current trick display
  const currentTrickDisplay: TrickDisplay = {
    plays: [0, 1, 2, 3].map(p =>
      gameState.currentTrick.find(play => play.player === p) || null
    ),
    winner: gameState.currentTrick.length === 4
      ? calculateTrickWinner(gameState.currentTrick, gameState.trump, gameState.currentSuit)
      : -1,
    isComplete: gameState.currentTrick.length === 4,
    currentPlayer
  };
  
  // Trick number
  const trickNumber = phase === 'scoring' || phase === 'bidding'
    ? gameState.tricks.length
    : Math.min(gameState.tricks.length + 1, 7);
  
  // Led suit display
  const ledSuitDisplay = gameState.currentSuit >= 0 && gameState.currentSuit <= 7
    ? getSuitName(gameState.currentSuit as LedSuit)
    : null;
  
  // Hand results (only during scoring)
  // Use current player's perspective in test mode, otherwise always player 0
  const playerPerspective = isTestMode ? currentPlayer : perspectiveIndex;
  const handResults = phase === 'scoring' ? calculateHandResults(gameState, playerPerspective) : null;
  
  // Determine waiting state
  const actionableActions = canAct ? availableActions : [];
  const playerActions = actionableActions.filter(a => {
    if (a.id.startsWith('bid-') || a.id === 'pass' || a.id === 'redeal') return true;
    if (a.id.startsWith('trump-')) return true;
    if (a.id.startsWith('play-')) return true;
    if (a.id === 'complete-trick' || a.id === 'score-hand' || a.id.startsWith('agree-')) return false;
    return true;
  });
  
  const hasActions = playerActions.length > 0;
  
  // Check if we're in a consensus phase (only agree actions available)
  const isConsensusPhase = actionableActions.length > 0 && 
    actionableActions.every(a => a.id.startsWith('agree-'));
  
  const isWaitingDuringBidding = phase === 'bidding' && canAct && (!isPerspectiveTurn || !hasActions);
  const isWaitingDuringTrump = phase === 'trump_selection' && canAct && (!isPerspectiveTurn || !hasActions);
  const isWaitingDuringPlay = phase === 'playing' && canAct && (!isPerspectiveTurn || !hasActions);
  const isWaiting = isWaitingDuringBidding || isWaitingDuringTrump || isWaitingDuringPlay;
  const waitingOnPlayer = (canAct && (!isPerspectiveTurn || !hasActions)) ? currentPlayer : -1;
  // Don't show AI thinking during consensus phases
  const isAIThinking = waitingOnPlayer >= 0 && isAIControlled(waitingOnPlayer) && !isConsensusPhase;
  
  // Determine active view
  let activeView: 'game' | 'actions' = 'game';
  if (phase === GAME_PHASES.BIDDING || phase === GAME_PHASES.TRUMP_SELECTION) {
    activeView = 'actions';
  }
  
  return {
    phase,
    currentPlayer,
    isPlayer0Turn: isPerspectiveTurn,
    isPerspectiveTurn,
    perspectiveIndex,
    canAct,
    hand,
    actions: {
      bidding: canAct ? biddingActions : [],
      trump: canAct ? trumpActions : [],
      proceed: canAct ? (proceedAction || consensusAction) : null,
      consensus: canAct ? consensusAction : null
    },
    bidding: {
      currentBid: gameState.currentBid,
      winningBidder: gameState.winningBidder,
      dealer: gameState.dealer,
      playerStatuses: biddingStatuses,
      showBiddingTable: isWaitingDuringBidding && !isTestMode
    },
    trump: {
      selection: gameState.trump,
      display: trumpDisplay
    },
    trick: {
      current: currentTrickDisplay,
      completed: gameState.tricks,
      number: trickNumber,
      ledSuit: gameState.currentSuit,
      ledSuitDisplay,
      trickLeader: gameState.currentTrick[0]?.player ?? null
    },
    scoring: {
      teamScores: gameState.teamScores,
      teamMarks: gameState.teamMarks,
      currentHandPoints: calculateTeamPoints(gameState.tricks),
      handResults
    },
    ui: {
      showActionPanel: canAct && (phase === 'bidding' || phase === 'trump_selection'),
      showBiddingTable: isWaitingDuringBidding && !isTestMode,
      isWaiting,
      waitingOnPlayer,
      isAIThinking,
      activeView
    },
    tooltips: {
      proceedAction: canAct ? (proceedAction?.label || consensusAction?.label || null) : null,
      skipAction: isAIThinking ? `P${waitingOnPlayer} is thinking...` : ''
    }
  };
}

// Helper to get domino tooltip
function getDominoTooltip(
  domino: Domino,
  gameState: FilteredGameState,
  playableDominoIds: Set<string>
): string {
  const dominoStr = `${domino.high}-${domino.low}`;
  
  if (gameState.phase !== 'playing') {
    return dominoStr;
  }
  
  if (gameState.currentPlayer !== 0) {
    return `${dominoStr} - Waiting for P${gameState.currentPlayer}'s turn`;
  }
  
  const isPlayable = playableDominoIds.has(`${domino.high}-${domino.low}`) ||
                    playableDominoIds.has(`${domino.low}-${domino.high}`);
  
  if (gameState.currentTrick.length === 0) {
    return isPlayable ? `${dominoStr} - Click to lead this domino` : dominoStr;
  }
  
  const leadSuit = gameState.currentSuit;
  if (leadSuit === -1) {
    return isPlayable ? `${dominoStr} - Click to play` : dominoStr;
  }
  
  const ledSuitName = getSuitName(leadSuit as LedSuit, { lowercase: true });
  
  if (isPlayable) {
    if (leadSuit === DOUBLES_AS_TRUMP && domino.high === domino.low) {
      return `${dominoStr} - Double, follows ${ledSuitName}`;
    } else if (leadSuit !== DOUBLES_AS_TRUMP && dominoHasSuit(domino, leadSuit)) {
      return `${dominoStr} - Has ${ledSuitName}, follows suit`;
    } else {
      if (gameState.trump.type === 'doubles' && domino.high === domino.low) {
        return `${dominoStr} - Trump (double)`;
      } else if (gameState.trump.type === 'suit' &&
                gameState.trump.suit !== undefined &&
                dominoHasSuit(domino, gameState.trump.suit)) {
        return `${dominoStr} - Trump`;
      } else {
        return `${dominoStr} - Can't follow ${ledSuitName}`;
      }
    }
  } else {
    if (leadSuit === DOUBLES_AS_TRUMP) {
      return `${dominoStr} - Not a double, can't follow ${ledSuitName}`;
    } else {
      const playerHand = gameState.players[0]?.hand || [];
      const playerHasLedSuit = playerHand.some(d =>
        d.high === leadSuit || d.low === leadSuit
      );
      
      if (playerHasLedSuit) {
        return `${dominoStr} - Must follow ${getSuitName(leadSuit as LedSuit, { lowercase: true })}`;
      } else {
        return `${dominoStr} - Invalid play`;
      }
    }
  }
}

// Note: getTrumpDisplay is now imported from game-terms.ts

// Helper to calculate current hand points for each team from tricks
function calculateTeamPoints(tricks: Trick[]): [number, number] {
  const team0Points = tricks
    .filter(t => t.winner !== undefined && t.winner % 2 === 0)
    .reduce((sum, t) => sum + (t.points || 0), 0);
  const team1Points = tricks
    .filter(t => t.winner !== undefined && t.winner % 2 === 1)
    .reduce((sum, t) => sum + (t.points || 0), 0);
  return [team0Points, team1Points];
}

// Helper to calculate hand results with perspective-aware messages
function calculateHandResults(gameState: FilteredGameState, playerPerspective: number = 0): HandResults | null {
  const tricks = gameState.tricks;
  const [team0Points, team1Points] = calculateTeamPoints(tricks);
  
  // Default to 42 points required (for marks, plunge, splash, etc.)
  // Only use bid value for explicit points bids
  const requiredPoints = (gameState.currentBid.type === 'points' && gameState.currentBid.value != null)
    ? gameState.currentBid.value
    : 42;
  
  const biddingPlayer = gameState.winningBidder;
  const biddingTeam = gameState.players[biddingPlayer]?.teamId;
  
  if (biddingTeam === undefined) {
    return null; // No valid bidder
  }
  
  const bidMade = biddingTeam === 0 
    ? team0Points >= requiredPoints 
    : team1Points >= requiredPoints;
  
  // Determine player's team from player object
  const playerTeam = gameState.players[playerPerspective]?.teamId;
  
  if (playerTeam === undefined) {
    return null; // Invalid player perspective
  }
  
  // Generate perspective-aware messages
  let resultMessage: string;
  let resultText: string;
  let isSuccess: boolean;
  let teamLabel: string;
  
  // Determine if the bidding team is "us" or "them" from player's perspective
  const isBiddingTeamUs = biddingTeam === playerTeam;
  teamLabel = isBiddingTeamUs ? 'US' : 'THEM';
  
  // Generate result message based on perspective
  if (isBiddingTeamUs) {
    // We bid
    if (bidMade) {
      resultMessage = '✅ WE MADE THE BID!';
      resultText = 'WE MADE THE BID!';
      isSuccess = true;
    } else {
      resultMessage = '❌ WE GOT SET!';
      resultText = 'WE GOT SET!';
      isSuccess = false;
    }
  } else {
    // They bid
    if (bidMade) {
      resultMessage = '❌ THEY MADE THE BID!';
      resultText = 'THEY MADE THE BID!';
      isSuccess = false;
    } else {
      resultMessage = '✅ WE SET THEM!';
      resultText = 'WE SET THEM!';
      isSuccess = true;
    }
  }
  
  return {
    team0Points,
    team1Points,
    bidAmount: requiredPoints,
    biddingTeam,
    bidMade,
    winningTeam: bidMade ? biddingTeam : (biddingTeam === 0 ? 1 : 0),
    biddingPlayer,
    resultMessage,
    resultText,
    isSuccess,
    teamLabel
  };
}
