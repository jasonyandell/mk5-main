import type { GameState } from '../types';
import type { MultiplayerGameState, PlayerSession, Result } from './types';
import { ok, err } from './types';

export interface CreateMultiplayerGameOptions {
  gameId: string;
  coreState: GameState;
  players: PlayerSession[];
}

export function createMultiplayerGame({
  gameId,
  coreState,
  players
}: CreateMultiplayerGameOptions): MultiplayerGameState {
  return {
    gameId,
    coreState,
    players: players.map(player => ({ ...player }))
  };
}

export function addPlayer(
  mpState: MultiplayerGameState,
  session: PlayerSession
): Result<MultiplayerGameState> {
  if (mpState.players.some(player => player.playerId === session.playerId)) {
    return err(`Player with ID ${session.playerId} already exists`);
  }

  if (mpState.players.some(player => player.playerIndex === session.playerIndex)) {
    return err(`Seat ${session.playerIndex} is already occupied`);
  }

  const players = [...mpState.players, { ...session }].sort(
    (a, b) => a.playerIndex - b.playerIndex
  );

  return ok({
    ...mpState,
    players
  });
}

export function removePlayer(
  mpState: MultiplayerGameState,
  playerId: string
): MultiplayerGameState {
  const players = mpState.players.map(session =>
    session.playerId === playerId
      ? { ...session, isConnected: false }
      : session
  );

  return {
    ...mpState,
    players
  };
}

export function updatePlayerSession(
  mpState: MultiplayerGameState,
  playerId: string,
  update: Partial<PlayerSession>
): Result<MultiplayerGameState> {
  const index = mpState.players.findIndex(player => player.playerId === playerId);

  if (index === -1) {
    return err(`Player with ID ${playerId} not found`);
  }

  const players = mpState.players.map(player =>
    player.playerId === playerId
      ? { ...player, ...update }
      : player
  );

  return ok({
    ...mpState,
    players
  });
}
