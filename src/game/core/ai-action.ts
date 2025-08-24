import type { Command } from '../events/types';
import type { GameState, Bid, Domino, TrumpSelection } from '../types';
import { SimpleAIStrategy, SmartAIStrategy } from '../controllers/strategies';
import { getValidPlays, isValidBid } from './rules';

const smartStrategy = new SmartAIStrategy();
const simpleStrategy = new SimpleAIStrategy();

export function getAIAction(state: any, playerId: number): Command | null {
  const phase = state.phase;
  
  switch (phase) {
    case 'bidding':
      return getAIBidAction(state, playerId);
    case 'trump_selection':
      return getAITrumpAction(state, playerId);
    case 'playing':
      return getAIPlayAction(state, playerId);
    default:
      return null;
  }
}

function getAIBidAction(state: any, playerId: number): Command | null {
  const validBids = getValidBids(state, playerId);
  
  if (validBids.length === 0) {
    return { type: 'PASS', player: playerId };
  }
  
  const transitions = validBids.map(bid => ({
    action: { type: 'bid' as const, player: playerId, bid: bid.type, value: bid.value },
    newState: state
  }));
  
  const choice = smartStrategy.chooseAction(state, transitions);
  
  if (choice && choice.action.type === 'bid') {
    const bid: Bid = {
      type: choice.action.bid,
      value: choice.action.value,
      player: playerId
    };
    return { type: 'PLACE_BID', player: playerId, bid };
  }
  
  return { type: 'PASS', player: playerId };
}

function getAITrumpAction(state: any, playerId: number): Command | null {
  const hand = state.players[playerId]?.hand || [];
  const bid = state.currentBid;
  
  const trumpOptions = getTrumpOptions(bid);
  const transitions = trumpOptions.map(trump => ({
    action: { type: 'select-trump' as const, player: playerId, trump },
    newState: state
  }));
  
  const choice = smartStrategy.chooseAction(state, transitions);
  
  if (choice && choice.action.type === 'select-trump') {
    return { type: 'SELECT_TRUMP', player: playerId, trump: choice.action.trump };
  }
  
  return { 
    type: 'SELECT_TRUMP', 
    player: playerId, 
    trump: { type: 'suit', suit: 6 } as TrumpSelection 
  };
}

function getAIPlayAction(state: any, playerId: number): Command | null {
  const validPlays = getValidPlays(state, playerId);
  
  if (validPlays.length === 0) return null;
  
  const transitions = validPlays.map(domino => ({
    action: { type: 'play' as const, player: playerId, dominoId: `${domino.high}-${domino.low}` },
    newState: state
  }));
  
  const choice = smartStrategy.chooseAction(state, transitions);
  
  if (choice && choice.action.type === 'play') {
    const [high, low] = choice.action.dominoId.split('-').map(Number);
    const domino = validPlays.find(d => d.high === high && d.low === low);
    if (domino) {
      return { type: 'PLAY_DOMINO', player: playerId, domino };
    }
  }
  
  return { type: 'PLAY_DOMINO', player: playerId, domino: validPlays[0] };
}

function getValidBids(state: any, playerId: number): Bid[] {
  const bids: Bid[] = [];
  
  const pointBids = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42];
  for (const value of pointBids) {
    const bid: Bid = { type: 'points', value, player: playerId };
    if (isValidBid(state, bid)) {
      bids.push(bid);
    }
  }
  
  const markBids = [{ type: 'marks' as const, value: 1 }, { type: 'marks' as const, value: 2 }];
  for (const markBid of markBids) {
    const bid: Bid = { ...markBid, player: playerId };
    if (isValidBid(state, bid)) {
      bids.push(bid);
    }
  }
  
  return bids;
}

function getTrumpOptions(bid: Bid): TrumpSelection[] {
  if (bid.type === 'nello') {
    return [{ type: 'doubles' }];
  }
  
  return [
    { type: 'suit', suit: 0 },
    { type: 'suit', suit: 1 },
    { type: 'suit', suit: 2 },
    { type: 'suit', suit: 3 },
    { type: 'suit', suit: 4 },
    { type: 'suit', suit: 5 },
    { type: 'suit', suit: 6 },
    { type: 'doubles' },
    { type: 'no-trump' }
  ];
}