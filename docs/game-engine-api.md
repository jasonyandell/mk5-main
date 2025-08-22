# Game Engine API (from src/game/index.ts)

## Types (./types)
- GameState, Player, Domino, Bid, Trick, Play, StateTransition, GameAction, GameHistory, GamePhase, GameConstants, PlayerView, PublicPlayer

## Constants (./constants)
- GAME_CONSTANTS, BID_TYPES, TRUMP_SELECTIONS, DOMINO_VALUES, POINT_VALUES

## Core State (./core/state)
- createInitialState(), createSetupState(), cloneGameState(), validateGameState(), isGameComplete(), getWinningTeam(), advanceToNextPhase()

## Players (./core/players)
- getNextDealer(), getPlayerLeftOfDealer()

## Engine (./core/gameEngine)
- GameEngine, getValidActions(), actionToId(), actionToLabel(), getNextStates()

## Actions (./core/actions)
- executeAction()

## Player View (./core/playerView)
- getPlayerView()

## Controllers (./controllers)
- ControllerManager

## Rules (./core/rules)
- isValidBid(), isValidOpeningBid(), isValidPlay(), getValidPlays(), canFollowSuit(), getBidComparisonValue(), getTrickWinner(), getTrickPoints(), determineTrickWinner(), isValidTrump(), getTrumpValue()

## Dominoes (./core/dominoes)
- createDominoes(), shuffleDominoesWithSeed(), dealDominoesWithSeed(), getDominoSuit(), getDominoValue(), getDominoPoints(), isDouble(), countDoubles()

## Scoring (./core/scoring)
- calculateTrickWinner(), calculateTrickPoints(), calculateRoundScore(), calculateGameSummary(), calculateGameScore(), getWinningTeam as getWinningTeamFromMarks

## URL Compression (./core/url-compression)
- compressGameState(), expandMinimalState(), compressActionId(), decompressActionId(), encodeURLData(), decodeURLData(); Types: MinimalGameState, CompressedAction, URLData

