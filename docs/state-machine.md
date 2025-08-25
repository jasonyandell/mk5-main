# Game State Machine

This document summarizes the phases, actions, and transitions as implemented in the codebase.

## Phases
- setup
- bidding (initial for normal play)
- trump_selection
- playing
- scoring
- game_end

## Bidding
- Actions (for current player, if they have not already bid):
  - pass → id: `pass`
  - bid points N ∈ [30..41] (when `isValidBid`) → id: `bid-N`
  - bid marks M ∈ [1..4] (when `isValidBid`) → id: `bid-M-marks`
- Completion after 4 total bids:
  - If all passed → `redeal` → stays in bidding with new dealer/currentPlayer/shuffleSeed and fresh hands
  - Else → determine winning bidder → phase `trump_selection`, currentPlayer = winningBidder, currentBid = winning bid

## Trump Selection
- Only winningBidder acts:
  - select trump suit (blanks..sixes) → `trump-<suit>`
  - doubles → `trump-doubles`
  - no-trump → `trump-no-trump`
- Effects: phase → `playing`, trump set, suitAnalysis recomputed for all players, currentPlayer = winningBidder

## Playing
- If currentTrick length < 4: valid plays for currentPlayer from `getValidPlays`
  - play domino D → `play-D` → removes domino, appends to currentTrick, sets currentSuit on first play, currentPlayer → next
- If currentTrick length == 4:
  - consensus: each player can `agree-complete-trick-p` until all 4 agree
  - then `complete-trick`:
    - determine winner, points; add trick; teamScores += points + 1; clear currentTrick/currentSuit; currentPlayer = winner
    - If 7th trick or hand outcome determined → phase `scoring`; else remain in `playing`

## Scoring
- consensus: each player can `agree-score-hand-p` until all 4 agree
- then `score-hand`:
  - Calculate round marks; if a team reaches target → phase `game_end`, set winner, clear hands, clear consensus
  - Else start next hand → phase `bidding`, rotate dealer, left-of-dealer starts, new hands, reset hand-level state, keep teamMarks

## Game End
- No further actions; `getValidActions` returns []

## IDs and labels
- Encoded via `actionToId` and `actionToLabel` in `src/game/core/gameEngine.ts`
- Examples:
  - `pass`, `bid-30`, `bid-2-marks`, `redeal`
  - `trump-fours`, `trump-doubles`, `trump-no-trump`
  - `play-<dominoId>`
  - `agree-complete-trick-<p>`, `complete-trick`
  - `agree-score-hand-<p>`, `score-hand`

## Consensus semantics
- Both complete-trick and score-hand require unanimity (4 agreements) before the neutral action is available
- Consensus sets are cleared after the neutral action executes

## Files of reference
- Types: `src/game/types.ts`
- Engine: `src/game/core/gameEngine.ts`
- Action execution: `src/game/core/actions.ts`
- Initial state: `src/game/core/state.ts`
- Store and URL/history: `src/stores/gameStore.ts`

