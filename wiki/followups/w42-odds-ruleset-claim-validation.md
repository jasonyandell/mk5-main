Reviewed against code on 2026-07-07 — no issues found.

Verified: all seven artifact paths under `w42/odds_ruleset_claim_validation/` exist; commit `c603a0d9` exists; headline counts (1,184,040 hands; modal 213,255 / 18.011%; 10/27 and 14/27; 4,422,600 follow-suit cases; 357 = 7+1+196+147+6 Chapter 13 fixtures) all match `validation_summary.json` and the CSVs.

- A cheap next probe: re-run `validate_odds_ruleset.py` in CI or a wiki-lint hook so the CSVs can never drift from the script.
- The `ch13-small-end-illegal` predicate encodes "led suit fixed to high pip" as an assumption of the predicate itself (see details column); a follow-up could check this against the actual engine's lead-suit resolution in `src/`.
