** Mobile UI ** - currently in good working order.  Changes should be thought through VERY carefully.

** Game logic ** - root is src/game/index.ts
pure functions and states. strictly.  fix any issues if you discover them

** Unit Tests ** - for pure game logic.

** E2E Tests ** - for UI. REQUIREMENT: fast. never longer than 5s timeout. should always hit the Debug UI via a unified, authoritative helper in src/tests/e2e/helpers/playwrightHelper.ts. CRITICAL: locators go in playwrightHelper.ts, not in the tests.  The tests should be pure and readable.

** Playwright ** - all use of playwright be strictly in non-interactive mode.  We should never wait on reports or users to hit ctrl+c.  This is true in Claude as well as in any scripts

** No legacy ** - CRITICAL. this is a greenfield project.  everything should be unified, even if it takes significant extra work

** No skipped tests ** - this is a greenfield project.  all tests should pass and be valuable, even if it takes significant extra work

** Temporary files ** - All temporary files, test artifacts, and scratch work should be placed in the scratch/ directory, which is gitignored

To check behavior, consult docs/rules.md

** Decode URL State ** - Run `node scripts/decode-url.js <base64-param>` to decode game state from URL parameter `d=`
** Encode URL State ** - Run `node scripts/encode-url.js <seed> [actions...]` to create URL (e.g. `node scripts/encode-url.js 12345 30 p p p trump-blanks 32 63`)