import type { ActionTransformerFactory } from './types';
import type { GameAction, GameState } from '../types';

/**
 * Show hints action transformer: Annotate actions with educational hints.
 *
 * Adds hint metadata to actions to help players learn optimal strategy.
 * Hints are only visible to players with 'see-hints' capability.
 *
 * Hint categories:
 * - Bidding strategy (when to bid, how much)
 * - Trump selection (based on hand strength)
 * - Play strategy (setting up tricks, following suit)
 *
 * Implementation:
 * - Analyzes game state to generate contextual hints
 * - Adds hints as action metadata
 * - Capability filtering handles visibility
 * - No changes to game logic, just action annotation
 */
export const hintsActionTransformer: ActionTransformerFactory = () => (base) => (state) => {
  const baseActions = base(state);

  // Annotate actions based on game phase
  return baseActions.map(action => {
    const hint = generateHint(action, state);

    if (!hint) {
      return action;
    }

    return {
      ...action,
      meta: {
        ...('meta' in action ? action.meta : {}),
        hint,
        requiredCapabilities: [{ type: 'see-hints' as const }]
      }
    };
  });
};

/**
 * Generate contextual hint for an action.
 */
function generateHint(action: GameAction, state: GameState): string | null {
  switch (action.type) {
    case 'bid':
      return generateBidHint(action, state);

    case 'select-trump':
      return generateTrumpHint(action, state);

    case 'play':
      return generatePlayHint(action, state);

    case 'pass':
      return generatePassHint(action, state);

    default:
      return null;
  }
}

/**
 * Generate hint for bidding action.
 */
function generateBidHint(action: GameAction & { type: 'bid' }, _state: GameState): string {
  // Nello is not a bid type - it's a trump selection

  if (action.bid === 'splash') {
    return 'Splash: Partner takes all 7 tricks. Very risky - requires dominant hand.';
  }

  if (action.bid === 'plunge') {
    return 'Plunge: Bidder takes all 7 tricks alone. Extremely risky.';
  }

  const value = action.value || 0;

  if (value >= 42) {
    return `Bidding ${value}: Confident you can make all tricks. Make sure your hand is strong.`;
  }

  if (value >= 36) {
    return `Bidding ${value}: Aiming for most tricks. Consider your trump strength.`;
  }

  if (value >= 30) {
    return `Bidding ${value}: Minimum bid. Safe if you have decent trump or high dominoes.`;
  }

  return `Bidding ${value}: Low bid - consider passing if hand is weak.`;
}

/**
 * Generate hint for trump selection.
 */
function generateTrumpHint(action: GameAction & { type: 'select-trump' }, _state: GameState): string {
  const trump = action.trump;

  if (trump.type === 'no-trump') {
    return 'No Trump: High-value dominoes will likely win. Good if you have 6s and 5s.';
  }

  if (trump.type === 'doubles') {
    return 'Doubles as Trump: All doubles become trump. Good if you have many doubles.';
  }

  if (trump.type === 'suit' && trump.suit !== undefined) {
    const suitNames = ['blanks', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixes'];
    const suitName = suitNames[trump.suit] || 'unknown';
    const capitalizedName = suitName[0]?.toUpperCase() + suitName.slice(1);
    return `${capitalizedName} as Trump: All dominoes with ${suitName} become trump.`;
  }

  return 'Select trump based on your hand strength.';
}

/**
 * Generate hint for play action.
 */
function generatePlayHint(_action: GameAction & { type: 'play' }, state: GameState): string {
  if (state.phase !== 'playing') {
    return 'Play a domino to the trick.';
  }

  const currentTrick = state.currentTrick || [];
  const isLeading = currentTrick.length === 0;

  if (isLeading) {
    return 'Leading: Play your strongest domino to win the trick, or a weak one to give opponents points.';
  }

  return 'Following: Must follow suit if possible. Play high to win, low to let partner win.';
}

/**
 * Generate hint for pass action.
 */
function generatePassHint(_action: GameAction & { type: 'pass' }, state: GameState): string {
  if (state.phase !== 'bidding') {
    return 'Pass your turn.';
  }

  const currentBid = state.currentBid?.value || 0;

  if (currentBid === 0) {
    return 'Passing: No bid yet. Pass if your hand is weak.';
  }

  return `Passing: Current bid is ${currentBid}. Pass if you can't beat it or your hand is too weak.`;
}
