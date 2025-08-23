## R2 Architecture Spec — Clean Greenfield Event-Sourced 42 (with Full Consensus)

### Goals
- Single, clean, complete design (no backward compatibility or migrations)
- Event sourcing end-to-end: every change is an event; derived via projections
- Deterministic replay and AI; URL-only persistence
- Strict reuse of existing pure core logic (rules, scoring, dominoes, suit-analysis)

### Non-goals
- No legacy support, no v1 formats, no dual stores

---

## Reused Pure Game Logic (unchanged)
- src/game/core/rules.ts: isValidBid, getValidPlays(state, playerId), isValidPlay, determineTrickWinner
- src/game/core/scoring.ts: calculateTrickPoints, calculateRoundScore, checkHandOutcome, isGameComplete
- src/game/core/dominoes.ts: dealDominoesWithSeed(seed), getDominoSuit
- src/game/core/players.ts: turn/dealer utilities
- src/game/core/suit-analysis.ts: analysis for AI and valid move computation

---

## Event Model (single clean schema)

### Event Envelope
- id: stable content hash (deterministic; excludes wall time)
- idx: monotonic sequence number (0-based)
- timestamp: wall time for UX (not used in hashing)
- correlationId: string (all events from a command share it)
- causationId?: previous event id (optional)
- payload: GameEvent (below)

Event id generation: id = base64url(sha256(JSON.stringify(payload) + ':' + idx))

### GameEvent union (complete)
- Lifecycle:
  - GAME_STARTED { seed: number; dealer: number; playerTypes: ('human'|'ai')[] }
  - HANDS_DEALT { hands: Record<number, Domino[]> }
  - REDEAL_INITIATED { newSeed: number }
  - GAME_ENDED { winningTeam: 0|1 }
- Bidding:
  - BID_PLACED { player: number; bid: Bid }
  - PLAYER_PASSED { player: number }
  - BIDDING_COMPLETED { winner: number; bid: Bid }
  - ALL_PLAYERS_PASSED { dealer: number }
- Trump/Playing:
  - TRUMP_SELECTED { player: number; trump: TrumpSelection }
  - DOMINO_PLAYED { player: number; domino: Domino }
  - TRICK_COMPLETED { winner: number; points: number }
- Scoring:
  - HAND_READY_FOR_SCORING { reason: 'all-tricks' | 'determined-early' }
  - HAND_SCORED { teamScores: [number, number] }
  - MARKS_AWARDED { team: 0|1; marks: number }
  - GAME_TARGET_REACHED { team: 0|1; finalMarks: [number, number] }
- Consensus (ConsensusAction = 'complete-trick' | 'score-hand'):
  - CONSENSUS_REQUESTED { action: ConsensusAction }
  - PLAYER_AGREED { player: number; action: ConsensusAction }
  - CONSENSUS_REACHED { action: ConsensusAction }
- AI/Quickplay/Effects:
  - AI_SCHEDULED { player: number; executeAt: number }
  - AI_THINKING { player: number }
  - AI_DECIDED { player: number; action: Command }
  - QUICKPLAY_ENABLED | QUICKPLAY_DISABLED | QUICKPLAY_SPEED_SET { speed: 'instant'|'fast'|'normal' }
  - EFFECT_SCHEDULED { effect: Effect; executeAt: number }
  - ANIMATION_STARTED { animation: string; duration: number } | ANIMATION_COMPLETED { animation: string; duration?: number }

Internal pseudo-event: RESET (not stored; emitted by EventStore on time travel/clear to instruct projections to rebuild)

---

## Commands (only way to change state)

Command union:
- START_GAME { seed?: number; playerTypes?: ('human'|'ai')[] }
- PLACE_BID { player: number; bid: Bid }
- PASS { player: number }
- SELECT_TRUMP { player: number; trump: TrumpSelection }
- PLAY_DOMINO { player: number; domino: Domino }
- AGREE_TO_ACTION { player: number; action: 'complete-trick' | 'score-hand' }
- ENABLE_QUICKPLAY { speed?: 'instant'|'fast'|'normal' }
- DISABLE_QUICKPLAY
- REQUEST_REDEAL

Validation (reuse core logic/state):
- PLACE_BID: isValidBid(state, bidWithPlayer)
- PASS: phase === 'bidding' and currentPlayer === player
- SELECT_TRUMP: phase === 'trump_selection' and currentPlayer === player
- PLAY_DOMINO: currentPlayer === player; domino present in hand; domino in projections.getValidMovesForPlayer(player)
- AGREE_TO_ACTION:
  - 'complete-trick': trick.length === 4; TRICK_COMPLETED not yet emitted since TRUMP_SELECTED; player hasn’t already agreed
  - 'score-hand': phase === 'scoring'; HAND_SCORED not yet emitted since HAND_READY_FOR_SCORING; player hasn’t already agreed

Command → events mapping:
- START_GAME: GAME_STARTED; HANDS_DEALT
- PLACE_BID: BID_PLACED; (when appropriate) BIDDING_COMPLETED
- PASS:
  - PLAYER_PASSED
  - If 4 total passes: ALL_PLAYERS_PASSED; REDEAL_INITIATED { newSeed }; HANDS_DEALT
  - Else if bidding completion condition met: BIDDING_COMPLETED
- SELECT_TRUMP: TRUMP_SELECTED
- PLAY_DOMINO:
  - DOMINO_PLAYED
  - If 4th play: CONSENSUS_REQUESTED('complete-trick')
- AGREE_TO_ACTION:
  - Emit PLAYER_AGREED; if size reaches 4: CONSENSUS_REACHED(action), then:
    - action === 'complete-trick':
      - leadSuit = getDominoSuit(firstPlay.domino, trump)
      - winner = determineTrickWinner(trick, trump, leadSuit)
      - points = calculateTrickPoints(trick)
      - Emit TRICK_COMPLETED { winner, points }
      - If trickCount == 7 OR checkHandOutcome says determined early:
        - Emit HAND_READY_FOR_SCORING { reason: 'all-tricks' | 'determined-early' }
        - Emit CONSENSUS_REQUESTED('score-hand')
    - action === 'score-hand':
      - Emit HAND_SCORED { teamScores: projections.teams.value.trickPoints }
      - Compute marks from highestBid/highestBidder; Emit MARKS_AWARDED
      - If marks reach target: GAME_TARGET_REACHED; GAME_ENDED
      - Else: REDEAL_INITIATED { newSeed }; HANDS_DEALT
- ENABLE_QUICKPLAY / DISABLE_QUICKPLAY: emit corresponding quickplay events

Correlation/causation: All events from one command share correlationId; causationId can point to prior event id

---

## Phases and State Machine

GamePhaseProjection:
- 'bidding': on GAME_STARTED, ALL_PLAYERS_PASSED, REDEAL_INITIATED
- 'trump_selection': on BIDDING_COMPLETED
- 'playing': on TRUMP_SELECTED
- 'scoring': on HAND_READY_FOR_SCORING
- 'game_over': on GAME_ENDED

Trick count:
- Managed via CompletedTricksProjection or TrickCountProjection
  - increment on TRICK_COMPLETED; reset on TRUMP_SELECTED/HAND_SCORED/GAME_STARTED

---

## Projections (final list)
- Base Projection<T>: cache + subscribe + RESET handling
- GamePhaseProjection
- PlayerTypesProjection
- HandsProjection
- TrickProjection: current trick plays
- TrumpProjection
- CurrentPlayerProjection
- TeamsProjection:
  - trickPoints += points on TRICK_COMPLETED (winner determines team); reset on TRUMP_SELECTED/HAND_SCORED/GAME_STARTED
  - marks updated on MARKS_AWARDED
- BidsProjection: bidder state, highest bid, passes
- CompletedTricksProjection (new): append on TRICK_COMPLETED; reset on TRUMP_SELECTED/HAND_SCORED/GAME_STARTED
- TrickCountProjection (optional if not deriving): maintains count similarly
- ConsensusProjection (new): Map<'complete-trick'|'score-hand', Set<number>>
  - CONSENSUS_REQUESTED initializes empty set, PLAYER_AGREED adds, CONSENSUS_REACHED clears; clear sets on TRICK_COMPLETED (complete-trick), HAND_SCORED (score-hand), GAME_STARTED, REDEAL_INITIATED
- ValidMovesProjection:
  - Uses rules.getValidPlays(state, playerId). Build minimal state from projections (phase, players with hand+suitAnalysis, currentTrick, currentSuit, currentPlayer, trump) or delegate to ProjectionManager.getValidMovesForPlayer

ProjectionManager:
- Registers all projections; forwards every event; provides getValidMovesForPlayer(playerId)

---

## Effects and AI

Delays (quickplay): instant 0ms; fast 100ms; normal 600ms

Triggers:
- GAME_STARTED/REDEAL_INITIATED: schedule first bidder if AI
- BID_PLACED/PLAYER_PASSED: schedule next bidder if AI
- BIDDING_COMPLETED: schedule trump selector if AI
- TRUMP_SELECTED: schedule first player (left of dealer) if AI
- DOMINO_PLAYED with trick < 4: schedule next if AI
- CONSENSUS_REQUESTED('complete-trick'): schedule all AI to AGREE_TO_ACTION('complete-trick')
- TRICK_COMPLETED: if not in scoring, schedule winner
- HAND_READY_FOR_SCORING: schedule all AI to AGREE_TO_ACTION('score-hand')
- HAND_SCORED and not GAME_ENDED: after REDEAL_INITIATED + HANDS_DEALT, schedule first bidder

EffectHandler:
- Cancellable timeouts keyed by situation; cancelAll on RESET
- getAIAction:
  - If consensus pending and player hasn’t agreed: return AGREE_TO_ACTION
  - Else bidding/trump/play using strategies; only legal actions

---

## URL Persistence (single clean format)
- Single param: g
- Content: base64url(LZ-compressed JSON) of the event log as array of { t: type; p: payload-minus-defaults }
- On load:
  - Decode g; rebuild wrapped events with idx ascending; timestamps set to now (UX only); id recomputed
  - Notify subscribers for initial projection builds
- On append/time travel/clear: re-encode entire event list and replaceState()
- Chunking (first-class): if g exceeds threshold (e.g., ~1900 chars), split into g1,g2,...; on load, concatenate sequential gN

---

## Determinism
- Seed chain:
  - START_GAME: seed = provided or Date.now()
  - REDEAL_INITIATED.newSeed = deterministic hash32(prevSeed, handIndex, 'redeal')
  - HANDS_DEALT uses dealDominoesWithSeed(newSeed)
- Event ids independent of timestamp; replay reproduces same ids/order
- All validation/AI derive from projections which derive from events

---

## Svelte Integration
- Single store: src/stores/eventGame.ts
  - EventStore.fromURL() → ProjectionManager → CommandProcessor → EffectHandler/Triggers
  - Export stores: events, gamePhase, currentPlayer, playerTypes, hands, trick, teams, bids, trump, consensus, validMoves
  - sendCommand(command), timeTravel(index), clear()
  - If no events on load: send START_GAME
- App/UI use eventGame exclusively
  - Show consensus “Agree” buttons after 4 plays (pre-TRICK_COMPLETED) and in scoring (pre-HAND_SCORED)

---

## Multiplayer (optional; off by default)
- Authority: server appends; clients send commands; server validates with same pure logic
- Transport: WebSocket; server ensures total order; clients dedup by id
- Optimistic UI: disabled by default
- When enabled, URL ‘g’ replaced by session id; otherwise URL-only persistence is used

---

## Testing Strategy
- Rules consistency: getValidPlays vs isValidPlay across randomized states
- Consensus invariants: no TRICK_COMPLETED before 4-agree after 4 plays; no HAND_SCORED before 4-agree during scoring
- Determinism: replay yields same projections and ids; seed chain reproducible
- URL serialization: round-trip g encode/decode; chunking reconstruction
- Effects: AI auto-agree under quickplay; RESET cancels schedules
- Commands: PASS×4 ⇒ ALL_PLAYERS_PASSED ⇒ REDEAL_INITIATED ⇒ HANDS_DEALT; full game to GAME_ENDED at target marks

---

## File Structure (final)
- src/game/core/… (unchanged)
- src/game/events/
  - types.ts (event/command/result types)
  - store.ts (EventStore with g encoding, RESET replay)
- src/game/projections/
  - base.ts, manager.ts, game-phase.ts, player-types.ts, hands.ts, trick.ts, trump.ts, current-player.ts, bids.ts, teams.ts
  - completed-tricks.ts (new), trick-count.ts (optional), consensus.ts (new), valid-moves.ts
- src/game/commands/
  - types.ts, processor.ts
- src/game/effects/
  - types.ts, handler.ts, triggers.ts
- src/stores/
  - eventGame.ts (Svelte integration)

---

## Conformance
- Always reuse existing core logic for validation and outcomes
- ValidMovesProjection and PLAY_DOMINO validation must use rules.getValidPlays(state, playerId)
- Consensus is first-class for trick completion and scoring
- Phase transitions are driven only by the specified events

