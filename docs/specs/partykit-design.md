# Adapting Texas 42 to PartyKit

This shows exactly how to wrap your existing game code for multiplayer with PartyKit.

## Your Existing Code Structure

```typescript
// What you have now:
import { GameState, StateTransition } from './game/types';
import { createInitialState, getNextStates } from './game';

// Your current flow:
let gameState = createInitialState();
const actions = getNextStates(gameState);
const action = actions.find(a => a.id === 'bid-30');
gameState = action.newState;
```

## PartyKit Wrapper

```typescript
// party/server.ts
import type * as Party from "partykit/server";
import { GameState, StateTransition } from '../game/types';
import { createInitialState, getNextStates } from '../game';

export default class Texas42Room implements Party.Server {
  constructor(private readonly room: Party.Room) {}

  // The game state - exactly your existing type
  private gameState: GameState | null = null;
  
  // Track who's connected
  private players = new Map<string, Party.Connection>();
  private spectators = new Set<Party.Connection>();

  async onStart() {
    // Restore game if it exists
    const saved = await this.room.storage.get<GameState>("gameState");
    if (saved) {
      this.gameState = saved;
    }
  }

  async onConnect(conn: Party.Connection, ctx: Party.ConnectionContext) {
    const url = new URL(ctx.request.url);
    const playerId = url.searchParams.get("player");
    
    if (playerId && ["0", "1", "2", "3"].includes(playerId)) {
      // Player connection
      this.players.set(playerId, conn);
      conn.setState({ playerId });
      
      // Start game when all 4 connect
      if (this.players.size === 4 && !this.gameState) {
        this.gameState = createInitialState();
        await this.room.storage.put("gameState", this.gameState);
      }
    } else {
      // Spectator
      this.spectators.add(conn);
      conn.setState({ spectator: true });
    }
    
    // Send current state
    this.sendStateToConnection(conn);
  }

  async onMessage(message: string, sender: Party.Connection) {
    const playerId = sender.state?.playerId;
    if (!playerId || !this.gameState) return;
    
    const { actionId } = JSON.parse(message);
    
    // Use your existing game logic!
    const availableActions = getNextStates(this.gameState);
    const action = availableActions.find(a => a.id === actionId);
    
    // Validate it's this player's turn
    if (action && this.gameState.currentPlayer === parseInt(playerId)) {
      // Apply the action
      this.gameState = action.newState;
      await this.room.storage.put("gameState", this.gameState);
      
      // Update everyone
      this.broadcastState();
    } else {
      sender.send(JSON.stringify({ 
        error: "Invalid action or not your turn" 
      }));
    }
  }

  onClose(conn: Party.Connection) {
    // Remove from tracking
    const playerId = conn.state?.playerId;
    if (playerId) {
      this.players.delete(playerId);
    } else {
      this.spectators.delete(conn);
    }
  }

  private sendStateToConnection(conn: Party.Connection) {
    if (!this.gameState) {
      conn.send(JSON.stringify({ waiting: true }));
      return;
    }

    const playerId = conn.state?.playerId;
    
    if (playerId) {
      // Player view - hide other hands
      const publicState = this.getPublicState(parseInt(playerId));
      const actions = this.gameState.currentPlayer === parseInt(playerId) 
        ? getNextStates(this.gameState).map(a => ({ id: a.id, label: a.label }))
        : [];
        
      conn.send(JSON.stringify({ 
        state: publicState,
        actions,
        yourTurn: this.gameState.currentPlayer === parseInt(playerId)
      }));
    } else {
      // Spectator - hide all hands
      const spectatorState = this.getSpectatorState();
      conn.send(JSON.stringify({ 
        state: spectatorState,
        spectator: true 
      }));
    }
  }

  private broadcastState() {
    // Send personalized state to each connection
    this.players.forEach(conn => this.sendStateToConnection(conn));
    this.spectators.forEach(conn => this.sendStateToConnection(conn));
  }

  private getPublicState(forPlayer: number): GameState {
    // Hide other players' hands
    return {
      ...this.gameState!,
      players: this.gameState!.players.map(p => ({
        ...p,
        hand: p.id === forPlayer ? p.hand : [],
        suitAnalysis: p.id === forPlayer ? p.suitAnalysis : undefined
      }))
    };
  }

  private getSpectatorState(): GameState {
    // Spectators see everything (honor system for family)
    return this.gameState!;
  }
}
```

## Client Code

```typescript
// client/game-client.ts
import PartySocket from "partysocket";

export class GameClient {
  private socket: PartySocket;
  public state: GameState | null = null;
  public actions: Array<{id: string, label: string}> = [];
  public yourTurn = false;

  constructor(
    private roomId: string,
    private playerId: string | null,
    private onUpdate: () => void
  ) {
    const query = playerId ? `?player=${playerId}` : '';
    
    this.socket = new PartySocket({
      host: PARTYKIT_HOST,
      room: roomId,
      query
    });

    this.socket.addEventListener("message", (e) => {
      const data = JSON.parse(e.data);
      
      if (data.error) {
        console.error(data.error);
        return;
      }
      
      if (data.waiting) {
        console.log("Waiting for players...");
        return;
      }
      
      this.state = data.state;
      this.actions = data.actions || [];
      this.yourTurn = data.yourTurn || false;
      this.onUpdate();
    });
  }

  executeAction(actionId: string) {
    this.socket.send(JSON.stringify({ actionId }));
  }

  disconnect() {
    this.socket.close();
  }
}
```

## React Integration

```tsx
// components/Game.tsx
function Game({ roomId, playerId }: { roomId: string, playerId: string | null }) {
  const [client, setClient] = useState<GameClient | null>(null);
  const [, forceUpdate] = useReducer(x => x + 1, 0);

  useEffect(() => {
    const gameClient = new GameClient(roomId, playerId, forceUpdate);
    setClient(gameClient);
    
    return () => gameClient.disconnect();
  }, [roomId, playerId]);

  if (!client?.state) {
    return <div>Connecting...</div>;
  }

  return (
    <div>
      {/* Your existing UI components work as-is! */}
      <GameBoard gameState={client.state} />
      
      {client.yourTurn && (
        <div>
          <h3>Your Turn!</h3>
          {client.actions.map(action => (
            <button 
              key={action.id}
              onClick={() => client.executeAction(action.id)}
            >
              {action.label}
            </button>
          ))}
        </div>
      )}
      
      {playerId === null && (
        <div>Spectating as guest</div>
      )}
    </div>
  );
}
```

## Deployment

```json
// partykit.json
{
  "name": "texas-42",
  "main": "./party/server.ts"
}
```

```bash
# Deploy
npx partykit deploy

# Your game is now live!
# Players connect to: https://texas-42.username.partykit.dev/party/room-name
```

## What This Gives You

1. **Your game logic unchanged** - Still using `getNextStates()` and your types
2. **Automatic reconnection** - PartyKit handles it
3. **Spectators** - Just connect without a player ID  
4. **Persistence** - Game survives server restarts
5. **Consistency** - One server, one source of truth

## Common Scenarios Handled

**Player refreshes page**
- Reconnects with same player ID
- Gets current state
- Continues playing

**New spectator joins mid-game**
- Sees current state (no hands)
- Gets live updates

**Player loses connection**
- Others can continue
- They rejoin when connection returns

**Server restarts**
- Game state restored from storage
- Players reconnect automatically

## Security Note

Currently spectators see all hands (honor system for family play). To prevent cheating:

```typescript
// Option 1: Track IP addresses
onConnect(conn: Party.Connection, ctx: Party.ConnectionContext) {
  const ip = ctx.request.headers.get("CF-Connecting-IP");
  if (this.playerIPs.has(ip)) {
    return conn.close(1008, "Already connected from this IP");
  }
}

// Option 2: One connection per browser
// Set a cookie when joining as player, check it for spectators

// Option 3: Room passwords
// Separate passwords for players vs spectators
```

But for family game night, the current honor system works great!

## Next Steps

1. Add room creation/joining UI
2. Add player names
3. Add chat (just broadcast messages)
4. Add game history/replay

The key insight: **Your existing game code needs zero changes**. PartyKit is just the multiplayer transport layer.