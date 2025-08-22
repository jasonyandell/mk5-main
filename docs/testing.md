# Testing (Implemented)

## Deterministic Playwright E2E
- Determinism: Tests encode a seed and actions into d= URL and pass testMode=true to disable AI and URL mutations; animations are disabled in-page for stable timing.
- Playwright features: Extensively uses waitForFunction() for state changes, waitForSelector() for DOM readiness, auto-waiting locators, and networkidle for page loads.
- Helper: src/tests/e2e/helpers/game-helper.ts → PlaywrightGameHelper.goto(seed, { disableUrlUpdates }), loadStateWithActions(seed, actions)
- Sample: src/tests/e2e/basic-gameplay.spec.ts → helper.goto(12345); asserts UI/phase/actions deterministically

## Unit/Integration Focus
- Rules/Scoring: src/game/core/rules.*, src/game/core/scoring.*
- Engine: getValidActions(), getNextStates(), executeAction(), getPlayerView()
- URL compression: encodeURLData()/decodeURLData(), compress/decompress action IDs

## Existing Coverage
- E2E
  - basic-gameplay.spec.ts: Loads UI, verifies bidding UI/valid options, transitions to playing, plays a domino, checks scores/mobile responsiveness.
  - back-button-comprehensive.spec.ts: Verifies browser back/forward across bidding, trump selection, playing, scoring, including time travel in debug panel and AI variants.
  - url-state-management.spec.ts: Ensures URL updates after actions, restores from URL, handles back/popstate, refresh, compression/minimal state, and error/corruption cases.
  - complete-trick-in-play-area.spec.ts: Confirms “Complete trick” appears after four plays and tapping the trick table advances to next trick.
  - history-navigation.spec.ts: Ensures debug History tab shows actions and time-travel updates URL/state while panel remains stable.
- Integration: basic-game-flow.test.ts, complete-game-flow.test.ts, state-transitions.test.ts (core flows and recomposition from transitions).
- Unit: deterministic-shuffle.test.ts, url-compression.test.ts, gameEngine.test.ts, state.test.ts (PRNG/deal, compression, engine invariants, state helpers).
- Rules: advanced-bidding.test.ts, trick-winner-validation.test.ts, trump-validation.test.ts, scoring-validation.test.ts, suit-analysis-integration.test.ts (bids, trick winner, trump, scoring, suit analysis).
- Gameplay: full-game.test.ts, edge-cases.test.ts, special-scenarios.test.ts, tournament-scenarios.test.ts (end-to-end scenarios and edge cases).

## How to Run
- Playwright: npx playwright test
- Unit (Vitest): npx vitest run

