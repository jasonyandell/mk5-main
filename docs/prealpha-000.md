ServerState
- game id
- game state
- players
- consensus: new type of action, consensus.  server only proceeds once consensus collected.  once collected, individual consensus actions are irrelevant
- AI runs on the server

ClientState
- gameStore

Server has everything.  It can run in the browser, but the client doesn't know about it

- sends messages to server

## Display
- Display dominoes 4 over 3, horizontally centered, not wrapped - remove from bottom row first as played

## Multiplayer
- PartyKit
- Implement offline mode first and first-class.
- player sees what partykit sees, AI are real partykit players as far as events are concerned

## View States
- Create test scenarios panel in settings: bidding, waiting on others, all bid, win the bid trumps, lose the bid, playing human turn, playing AI turn, early termination we/they made/set, full hand complete, game over, plunge, we won, we lost - clicking jumps directly via URL
- Identify and fix/embrace/delete the panel that pops up and disappears between playing and bidding phases
- At game end: display "we won/we lost", AI should not appear thinking, click panel starts new game, show tricks

## State representation
- Settings panel should have separate URL parameter orthogonal to state URL
- On new game, dealer should be randomly assigned

## AI
- Make AI smarter - lead with double of trump suit, then other high dominoes first

- Fix time travel on mobile (currently only undo button works, not full time travel)
- Game over should offer new game option
- Add "New Game" button available anytime with confirmation dialog
- Add super easy share URL feature - not browser-dependent, easy to share
- **GOAL** Add game "hands" - NYT style puzzles with set hand, forced bid, try to win it, keep streak of wins in a row
- Add contract 42 - like contract bridge, everyone gets same hands, no cheating
- Later: Add always-on multiplayer with drop-in/drop-out, consensus-based new games, tiebreaker cancels