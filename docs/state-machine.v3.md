# Game State Machine (v3)

Concise, technically-accurate spec derived directly from the codebase.

- Source of truth for valid actions: `src/game/core/gameEngine.ts#getValidActions`
- State application: `src/game/core/actions.ts#executeAction` and specific handlers

## Phases
- setup
- bidding (normal initial phase)
- trump_selection
- playing
- scoring
- game_end

## Actions overview
Each entry specifies: phase, availability (code-equivalent condition), handler, and resulting phase transitions.

- pass (bidding)
  - availableWhen: `state.phase==='bidding' && !state.bids.some(b=>b.player===state.currentPlayer) && isValidBid(state,{type:'pass',player:state.currentPlayer}, hand)`
  - handler: executePass (src/game/core/actions.ts)
  - phaseTransitions:
    - bidding when `state.bids.length < 3`
    - trump_selection when `state.bids.length===3 && any non-pass among bids+this`

- bid(points N: 30..41) (bidding)
  - availableWhen: as per isValidBid(points)
  - handler: executeBid
  - phaseTransitions: bidding if first 3; trump_selection on 4th bid

- bid(marks M: 1..4) (bidding)
  - same as points, handler executeBid

- redeal (bidding)
  - availableWhen: all 4 passed
  - handler: executeRedeal
  - phaseTransitions: bidding (new deal)

- select-trump (trump_selection)
  - availableWhen: winningBidder chooses one of suit|doubles|no-trump and isValidTrump
  - handler: executeTrumpSelection
  - phaseTransitions: playing

- play (playing)
  - availableWhen: trick not full, currentPlayer owns domino and play is valid
  - handler: executePlay
  - phaseTransitions: playing

- agree-complete-trick (playing)
  - availableWhen: trick full and player not yet agreed
  - handler: executeAgreement(completeTrick)
  - phaseTransitions: playing

- complete-trick (playing)
  - availableWhen: trick full and all agreed
  - handler: executeCompleteTrick
  - phaseTransitions:
    - scoring if 7th trick or hand determined by outcome
    - otherwise playing

- agree-score-hand (scoring)
  - availableWhen: player not yet agreed
  - handler: executeAgreement(scoreHand)
  - phaseTransitions: scoring

- score-hand (scoring)
  - availableWhen: all agreed
  - handler: executeScoreHand
  - phaseTransitions:
    - game_end if marks reach target
    - otherwise bidding (next hand)

## IDs
- Produced by `actionToId` in `src/game/core/gameEngine.ts`
- Examples: `pass`, `bid-30`, `bid-2-marks`, `redeal`, `trump-fours`, `trump-doubles`, `trump-no-trump`, `play-<dominoId>`, `agree-complete-trick-<p>`, `complete-trick`, `agree-score-hand-<p>`, `score-hand`

## Determinism and consensus
- Consensus actions (agree-*) are required before neutral actions (complete-trick, score-hand) become available
- AI scheduler influences timing but not legality; state machine remains pure

## Derivability
This spec is intentionally regular to support downstream derivations:
- Enumerating all action IDs for a given phase
- Computing valid next phases given current phase + action
- Generating a diagram or test scenarios programmatically

See machine-readable companion file: `docs/state-machine.v3.json`

