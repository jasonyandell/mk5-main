** Debug UI ** - should display all state and transitions at any given time, hiding nothing.  There should only be display logic and basic interaction here, as it is a Debug UI. the state variables and values represent the ubiquitous language for this debug UI.  it should have Undo and Report Bug sections

** E2E Tests ** - should be fast. never longer than 5s timeout. should always hit the Debug UI via a unified, authoritative helper in src/tests/e2e/helpers/playwrightHelper.ts. 

** Game logic ** - pure functions and states. strictly.  fix any issues if you discover them

** Playwright ** - all use of playwright be strictly in non-interactive mode.  We should never wait on reports or users to hit ctrl+c.  This is true in Claude as well as in any scripts

** No legacy ** - CRITICAL. this is a greenfield project.  everything should be unified, even if it takes significant extra work
** No skipped tests ** - this is a greenfield project.  all tests should pass and be valuable, even if it takes significant extra work

** Temporary files ** - All temporary files, test artifacts, and scratch work should be placed in the scratch/ directory, which is gitignored

To check behavior, consult @docs/rules.md

** Decode URL State ** - Run `node scripts/decode-url.js <base64-param>` to decode game state from URL parameter `d=`