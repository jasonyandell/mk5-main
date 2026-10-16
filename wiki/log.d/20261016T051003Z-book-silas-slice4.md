## [2026-10-16 | book | Silas reads the arena's ledger; the differential check finds no disagreement (#130 slice 4)]

- `silas/src/ledger.rs`: `Ledger::from_json` (outcomes, never utilities; any other key refused), `Paired::from_ledger` replaying and settling every row under Silas's rules, `silas compare --ledger FILE`. `arena/ledger.py`: the fixed-state runner and ledger writer, `SilasSeat`, the chance arm, `overbidding_field(k)`. Choices in `silas/docs/cross-game.md`, Slice 4 as built. [[silas]]
- Round trip: two arena ledgers of a Silas-only field (marks with the straight auction; points from 200-180 with the forced shaker) are Silas's own `play_out` hand for hand, 192 hands, and read to the arena's paired mean exactly (`silas/tests/ledger.rs`, `arena/test_ledger.py`).
