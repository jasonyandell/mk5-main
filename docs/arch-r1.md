# Event-Sourced Architecture - Revision 1

## Overview
Complete transition from mutable state to event-sourced architecture. Every state change is an event. State is always computed from events. No storage except URL.

## Core Principles
1. **Single source of truth**: Append-only event log
2. **Derived state**: All game state computed from events
3. **Deterministic replay**: Identical event sequence produces identical state
4. **Effect isolation**: Time-based operations separated from pure logic
5. **URL-only persistence**: No localStorage, no cookies, only URL

## Preserved Game Logic
These pure functions contain validated game logic and remain unchanged:
- `src/game/core/rules.ts` - Move validation, bidding rules, trick winners
- `src/game/core/scoring.ts` - Point calculation, score determination
- `src/game/core/dominoes.ts` - Dealing, shuffling, suit/value functions
- `src/game/core/players.ts` - Turn order, player utilities
- `src/game/core/suit-analysis.ts` - Suit counting and ranking for AI

## Event System

### Complete Event Taxonomy
```typescript
type GameEvent = 
  // Game lifecycle
  | { type: 'GAME_STARTED'; seed: number; dealer: number; playerTypes: ('human' | 'ai')[] }
  | { type: 'HANDS_DEALT'; hands: Record<number, Domino[]> }
  | { type: 'GAME_ENDED'; winningTeam: 0 | 1 }
  
  // Bidding events  
  | { type: 'BID_PLACED'; player: number; bid: Bid }
  | { type: 'PLAYER_PASSED'; player: number }
  | { type: 'BIDDING_COMPLETED'; winner: number; bid: Bid }
  | { type: 'ALL_PLAYERS_PASSED'; dealer: number }
  | { type: 'REDEAL_INITIATED'; newSeed: number }
  
  // Playing events
  | { type: 'TRUMP_SELECTED'; player: number; trump: TrumpSelection }
  | { type: 'DOMINO_PLAYED'; player: number; domino: Domino }
  | { type: 'TRICK_COMPLETED'; winner: number; points: number }
  
  // Scoring events
  | { type: 'HAND_SCORED'; teamScores: [number, number] }
  | { type: 'MARKS_AWARDED'; team: 0 | 1; marks: number }
  | { type: 'GAME_TARGET_REACHED'; team: 0 | 1; finalMarks: [number, number] }
  
  // Consensus events
  | { type: 'CONSENSUS_REQUESTED'; action: 'complete-trick' | 'score-hand' }
  | { type: 'PLAYER_AGREED'; player: number; action: string }
  | { type: 'CONSENSUS_REACHED'; action: string }
  
  // AI/scheduling events
  | { type: 'AI_SCHEDULED'; player: number; executeAt: number }
  | { type: 'AI_THINKING'; player: number }
  | { type: 'AI_DECIDED'; player: number; action: Command }
  
  // Quickplay events
  | { type: 'QUICKPLAY_ENABLED' }
  | { type: 'QUICKPLAY_DISABLED' }
  | { type: 'QUICKPLAY_SPEED_SET'; speed: 'instant' | 'fast' | 'normal' }
  
  // Effect events
  | { type: 'EFFECT_SCHEDULED'; effect: Effect; executeAt: number }
  | { type: 'ANIMATION_STARTED'; animation: string; duration: number }
  | { type: 'ANIMATION_COMPLETED'; animation: string };
```

### Event Store
```typescript
export class EventStore {
  private events: Event[] = [];
  private subscribers = new Map<string, Set<Subscriber>>();
  
  append(event: GameEvent): EventId {
    const wrapped = {
      id: generateId(),
      timestamp: Date.now(),
      payload: event
    };
    
    try {
      this.events.push(wrapped);
      this.notify(wrapped);
      this.updateURL();
      return wrapped.id;
    } catch (error) {
      console.error('Failed to append event:', event, error);
      throw error;
    }
  }
  
  // URL is the ONLY persistence
  private updateURL(): void {
    const compressed = this.compressEvents(this.events);
    const url = new URL(window.location.href);
    url.searchParams.set('g', compressed);
    window.history.replaceState(null, '', url);
  }
  
  // Load from URL on init
  static fromURL(): EventStore {
    const url = new URL(window.location.href);
    const compressed = url.searchParams.get('g');
    if (!compressed) return new EventStore();
    
    try {
      const events = decompressEvents(compressed);
      const store = new EventStore();
      store.events = events;
      return store;
    } catch (error) {
      console.error('Failed to load events from URL:', error);
      return new EventStore();
    }
  }
  
  // Time travel
  replaceTo(index: number): void {
    this.events = this.events.slice(0, index);
    this.notifyReset();
    this.updateURL();
  }
}
```

## Projection System

### Base Projection
```typescript
export abstract class Projection<T> {
  protected cache: T;
  protected dirty = true;
  
  abstract handles(): GameEvent['type'][];
  abstract apply(event: GameEvent): void;
  abstract compute(): T;
  
  handleEvent(event: Event): void {
    if (this.handles().includes(event.payload.type)) {
      this.apply(event.payload);
      this.dirty = true;
    }
  }
  
  get value(): T {
    if (this.dirty) {
      this.cache = this.compute();
      this.dirty = false;
    }
    return this.cache;
  }
}
```

### Critical Projections

```typescript
// Tracks game phase
export class GamePhaseProjection extends Projection<GamePhase> {
  private phase: GamePhase = 'setup';
  
  handles() {
    return ['GAME_STARTED', 'BIDDING_COMPLETED', 'TRUMP_SELECTED', 'HAND_SCORED'];
  }
  
  apply(event: GameEvent): void {
    switch (event.type) {
      case 'GAME_STARTED': this.phase = 'bidding'; break;
      case 'BIDDING_COMPLETED': this.phase = 'trump_selection'; break;
      case 'TRUMP_SELECTED': this.phase = 'playing'; break;
      case 'HAND_SCORED': this.phase = 'scoring'; break;
    }
  }
  
  compute(): GamePhase {
    return this.phase;
  }
}

// Tracks player types (human vs AI)
export class PlayerTypesProjection extends Projection<('human' | 'ai')[]> {
  private playerTypes: ('human' | 'ai')[] = ['human', 'ai', 'ai', 'ai'];
  
  handles() {
    return ['GAME_STARTED'];
  }
  
  apply(event: GameEvent): void {
    if (event.type === 'GAME_STARTED') {
      this.playerTypes = event.playerTypes;
    }
  }
  
  compute(): ('human' | 'ai')[] {
    return this.playerTypes;
  }
}

// Tracks consensus state
export class ConsensusProjection extends Projection<Map<string, Set<number>>> {
  private agreements = new Map<string, Set<number>>();
  
  handles() {
    return ['CONSENSUS_REQUESTED', 'PLAYER_AGREED', 'CONSENSUS_REACHED'];
  }
  
  apply(event: GameEvent): void {
    switch (event.type) {
      case 'CONSENSUS_REQUESTED':
        this.agreements.set(event.action, new Set());
        break;
      case 'PLAYER_AGREED':
        this.agreements.get(event.action)?.add(event.player);
        break;
      case 'CONSENSUS_REACHED':
        this.agreements.delete(event.action);
        break;
    }
  }
  
  compute(): Map<string, Set<number>> {
    return this.agreements;
  }
}

// Valid moves - uses existing pure functions
export class ValidMovesProjection extends Projection<Domino[]> {
  constructor(
    private playerId: number,
    private deps: ProjectionDependencies
  ) {
    super();
  }
  
  handles() {
    return ['DOMINO_PLAYED', 'TRICK_COMPLETED', 'TRUMP_SELECTED'];
  }
  
  apply(): void {
    // Just mark dirty, compute on demand
  }
  
  compute(): Domino[] {
    // Call existing pure function
    return getValidPlays(
      this.deps.hand.value,
      this.deps.trick.value,
      this.deps.trump.value,
      this.deps.leadSuit.value
    );
  }
}

// Team tracking (0,2 vs 1,3)
export class TeamProjection extends Projection<{ scores: [number, number]; marks: [number, number] }> {
  private scores: [number, number] = [0, 0];
  private marks: [number, number] = [0, 0];
  
  handles() {
    return ['HAND_SCORED', 'MARKS_AWARDED'];
  }
  
  apply(event: GameEvent): void {
    switch (event.type) {
      case 'HAND_SCORED':
        this.scores = event.teamScores;
        break;
      case 'MARKS_AWARDED':
        this.marks[event.team] += event.marks;
        break;
    }
  }
  
  compute() {
    return { scores: this.scores, marks: this.marks };
  }
  
  getPlayerTeam(player: number): 0 | 1 {
    return player % 2 as 0 | 1; // Players 0,2 are team 0; 1,3 are team 1
  }
}
```

## Command System

### Command Types
```typescript
export type Command = 
  | { type: 'START_GAME'; seed?: number; playerTypes?: ('human' | 'ai')[] }
  | { type: 'PLACE_BID'; player: number; bid: Bid }
  | { type: 'PASS'; player: number }
  | { type: 'SELECT_TRUMP'; player: number; trump: TrumpSelection }
  | { type: 'PLAY_DOMINO'; player: number; domino: Domino }
  | { type: 'AGREE_TO_ACTION'; player: number; action: string }
  | { type: 'REQUEST_REDEAL' }
  | { type: 'SCORE_HAND' }
  | { type: 'ENABLE_QUICKPLAY'; speed?: 'instant' | 'fast' | 'normal' }
  | { type: 'DISABLE_QUICKPLAY' };
```

### Command Processor
```typescript
export class CommandProcessor {
  constructor(
    private eventStore: EventStore,
    private projections: ProjectionManager
  ) {}
  
  process(command: Command): Result<EventId[]> {
    try {
      // Validate command
      const validation = this.validate(command);
      if (!validation.valid) {
        console.error('Invalid command:', command, validation.error);
        return { error: validation.error };
      }
      
      // Convert to events
      const events = this.commandToEvents(command);
      
      // Append to store
      const ids = events.map(e => this.eventStore.append(e));
      
      return { value: ids };
    } catch (error) {
      console.error('Command processing failed:', command, error);
      return { error: error.message };
    }
  }
  
  private validate(command: Command): ValidationResult {
    const state = this.projections.gameState.value;
    
    switch (command.type) {
      case 'PLACE_BID':
        // Use existing isValidBid pure function
        const valid = isValidBid(state, command.bid);
        return { valid, error: valid ? null : 'Invalid bid' };
        
      case 'PLAY_DOMINO':
        // Use existing isValidPlay pure function
        const validPlay = isValidPlay(state, command.domino, command.player);
        return { valid: validPlay, error: validPlay ? null : 'Invalid play' };
        
      case 'PASS':
        // Can only pass during bidding
        return { 
          valid: state.phase === 'bidding', 
          error: state.phase === 'bidding' ? null : 'Can only pass during bidding' 
        };
        
      default:
        return { valid: true };
    }
  }
  
  private commandToEvents(command: Command): GameEvent[] {
    switch (command.type) {
      case 'START_GAME':
        const seed = command.seed ?? Date.now();
        const playerTypes = command.playerTypes ?? ['human', 'ai', 'ai', 'ai'];
        const handsArray = dealDominoesWithSeed(seed);
        const hands = {
          0: handsArray[0],
          1: handsArray[1],
          2: handsArray[2],
          3: handsArray[3]
        };
        return [
          { type: 'GAME_STARTED', seed, dealer: 3, playerTypes },
          { type: 'HANDS_DEALT', hands }
        ];
        
      case 'PLAY_DOMINO':
        const events: GameEvent[] = [
          { type: 'DOMINO_PLAYED', player: command.player, domino: command.domino }
        ];
        
        // Check if trick complete
        const trick = this.projections.trick.value;
        if (trick.length === 3) {
          const completeTrick = [...trick, { player: command.player, domino: command.domino }];
          const leadSuit = getDominoSuit(trick[0].domino, this.projections.trump.value);
          const winner = determineTrickWinner(completeTrick, this.projections.trump.value, leadSuit);
          const points = calculateTrickPoints(completeTrick);
          events.push({ type: 'TRICK_COMPLETED', winner, points });
        }
        
        return events;
        
      case 'PASS':
        const events: GameEvent[] = [{ type: 'PLAYER_PASSED', player: command.player }];
        
        // Check if all players passed (redeal)
        const bids = this.projections.bids.value;
        const passes = bids.filter(b => b.type === 'pass');
        if (passes.length === 3) { // This is the 4th pass
          events.push({ type: 'ALL_PLAYERS_PASSED', dealer: this.projections.dealer.value });
          events.push({ type: 'REDEAL_INITIATED', newSeed: Date.now() });
        }
        
        return events;
        
      case 'SCORE_HAND':
        const scores = this.projections.teams.value.scores;
        const events: GameEvent[] = [{ type: 'HAND_SCORED', teamScores: scores }];
        
        // Calculate marks
        const bidder = this.projections.winningBidder.value;
        const bidderTeam = bidder % 2 as 0 | 1;
        const bid = this.projections.winningBid.value;
        const bidTarget = bid.value || 0;
        
        if (scores[bidderTeam] >= bidTarget) {
          // Made the bid
          const marks = bid.type === 'marks' ? bid.value : 1;
          events.push({ type: 'MARKS_AWARDED', team: bidderTeam, marks });
        } else {
          // Set - opponents get marks
          const opponentTeam = (bidderTeam + 1) % 2 as 0 | 1;
          const marks = bid.type === 'marks' ? bid.value : 1;
          events.push({ type: 'MARKS_AWARDED', team: opponentTeam, marks });
        }
        
        // Check for game end
        const newMarks = [...this.projections.teams.value.marks];
        if (events.some(e => e.type === 'MARKS_AWARDED')) {
          const markEvent = events.find(e => e.type === 'MARKS_AWARDED');
          if (markEvent?.type === 'MARKS_AWARDED') {
            newMarks[markEvent.team] += markEvent.marks;
            if (newMarks[markEvent.team] >= 7) {
              events.push({ type: 'GAME_TARGET_REACHED', team: markEvent.team, finalMarks: newMarks as [number, number] });
              events.push({ type: 'GAME_ENDED', winningTeam: markEvent.team });
            }
          }
        }
        
        return events;
        
      default:
        return [];
    }
  }
}
```

## Effect System

### Effect Types
```typescript
export type Effect = 
  | { type: 'SCHEDULE_AI'; player: number; afterMs: number }
  | { type: 'DELAY'; ms: number; then: Effect }
  | { type: 'EMIT_EVENT'; event: GameEvent }
  | { type: 'ANIMATE'; animation: string; duration: number };
```

### Effect Handler
```typescript
export class EffectHandler {
  private scheduled = new Map<string, number>();
  
  constructor(
    private eventStore: EventStore,
    private projections: ProjectionManager
  ) {}
  
  handle(effect: Effect): void {
    switch (effect.type) {
      case 'SCHEDULE_AI':
        // Only schedule if player is AI
        const playerTypes = this.projections.playerTypes.value;
        if (playerTypes[effect.player] !== 'ai') return;
        
        const id = setTimeout(() => {
          this.eventStore.append({ type: 'AI_THINKING', player: effect.player });
          
          // Get AI decision
          const state = this.projections.gameState.value;
          const strategy = getAIStrategy(effect.player);
          const action = strategy.decide(state);
          
          this.eventStore.append({ type: 'AI_DECIDED', player: effect.player, action });
          
          // Process the AI's command
          commandProcessor.process(action);
        }, effect.afterMs);
        
        this.scheduled.set(`ai-${effect.player}`, id);
        break;
        
      case 'EMIT_EVENT':
        this.eventStore.append(effect.event);
        break;
    }
  }
  
  cancel(key: string): void {
    const id = this.scheduled.get(key);
    if (id) {
      clearTimeout(id);
      this.scheduled.delete(key);
    }
  }
}
```

### Effect Triggers
```typescript
export class EffectTriggers {
  constructor(
    private eventStore: EventStore,
    private handler: EffectHandler,
    private projections: ProjectionManager
  ) {
    eventStore.subscribe(this.onEvent.bind(this));
  }
  
  private onEvent(event: Event): void {
    const effects = this.computeEffects(event);
    effects.forEach(effect => this.handler.handle(effect));
  }
  
  private computeEffects(event: Event): Effect[] {
    const quickplay = this.projections.quickplay.value;
    const delay = quickplay.enabled ? 
      (quickplay.speed === 'instant' ? 0 : quickplay.speed === 'fast' ? 100 : 600) : 
      600;
    
    switch (event.payload.type) {
      case 'GAME_STARTED':
      case 'BIDDING_COMPLETED':
      case 'TRICK_COMPLETED':
      case 'REDEAL_INITIATED':
        // Schedule next player if AI
        const nextPlayer = this.projections.currentPlayer.value;
        const playerTypes = this.projections.playerTypes.value;
        
        if (playerTypes[nextPlayer] === 'ai') {
          return [{ type: 'SCHEDULE_AI', player: nextPlayer, afterMs: delay }];
        }
        return [];
        
      default:
        return [];
    }
  }
}
```

## Svelte Integration

```typescript
// src/stores/game.ts
import { readable, derived } from 'svelte/store';

// Initialize from URL
const eventStore = EventStore.fromURL();
const projections = new ProjectionManager(eventStore);
const commandProcessor = new CommandProcessor(eventStore, projections);
const effectHandler = new EffectHandler(eventStore, projections);
const effectTriggers = new EffectTriggers(eventStore, effectHandler, projections);

// Expose as stores
export const events = readable(eventStore.getEvents(), set => 
  eventStore.subscribe(() => set(eventStore.getEvents()))
);

export const gamePhase = readable(projections.phase.value, set =>
  projections.phase.subscribe(set)
);

export const validMoves = readable(projections.validMoves.value, set =>
  projections.validMoves.subscribe(set)
);

export const currentPlayer = readable(projections.currentPlayer.value, set =>
  projections.currentPlayer.subscribe(set)
);

export const teams = readable(projections.teams.value, set =>
  projections.teams.subscribe(set)
);

// Commands are the ONLY way to change state
export function sendCommand(command: Command) {
  return commandProcessor.process(command);
}

// Time travel
export function timeTravel(index: number) {
  eventStore.replaceTo(index);
}

// Start game on load if no events
if (eventStore.getEvents().length === 0) {
  sendCommand({ type: 'START_GAME' });
}
```

## Testing Strategy

### Event Testing
```typescript
test('game flow', () => {
  const store = new EventStore();
  store.append({ type: 'GAME_STARTED', seed: 12345, dealer: 3, playerTypes: ['human', 'ai', 'ai', 'ai'] });
  store.append({ type: 'BID_PLACED', player: 0, bid: { type: 'points', value: 30 } });
  
  const projections = new ProjectionManager(store);
  expect(projections.phase.value).toBe('bidding');
  expect(projections.bids.value).toHaveLength(1);
});
```

### Effect Testing
```typescript
test('AI scheduling', async () => {
  const store = new EventStore();
  const handler = new EffectHandler(store);
  
  handler.handle({ type: 'SCHEDULE_AI', player: 1, afterMs: 0 });
  
  await delay(10);
  
  const events = store.getEvents();
  expect(events).toContainEqual(
    expect.objectContaining({ payload: { type: 'AI_THINKING', player: 1 } })
  );
});
```

## File Structure

```
src/
  game/
    core/           # Pure game logic (UNCHANGED)
    events/         
      types.ts      
      store.ts      
    projections/    
      base.ts       
      game-phase.ts 
      valid-moves.ts
      teams.ts
      consensus.ts
      player-types.ts
    effects/        
      types.ts      
      handler.ts    
      triggers.ts   
    commands/       
      types.ts      
      processor.ts  
  stores/
    game.ts         # Svelte integration
```

## Implementation Notes

### Function Signatures
- `determineTrickWinner(trick, trump, leadSuit)` - requires lead suit as 3rd parameter
- `dealDominoesWithSeed(seed)` - returns `[hand0, hand1, hand2, hand3]` tuple, convert to Record
- `getDominoSuit(domino, trump)` - needed to determine lead suit from first play

### Required Imports
See each module's imports from `src/game/core/*` for specific function usage.

## Multiplayer Preparation

When ready for multiplayer:
1. Events flow through server instead of direct append
2. Server validates commands before broadcasting events
3. Clients reconcile optimistic updates with server events
4. Add event versioning for backward compatibility

## Error Handling

All errors are logged with context to console. More sophisticated error handling infrastructure will be added in future iterations.

```typescript
try {
  // operation
} catch (error) {
  console.error('Context description:', { command, state, error });
  // Continue operation or fail gracefully
}
```

## Conclusion

This architecture provides:
- Complete event sourcing with no mutable state
- URL-only persistence (no localStorage, cookies, or other storage)
- All game logic preserved in pure functions
- Clear path to multiplayer
- Comprehensive event coverage for all game mechanics
- Simple error handling through console logging