## [2026-11-02 | codex | Bourne-0: a field net written from scratch]

Jason requested a fresh field net after the weight-space programming article.
`zeb/scripts/written_field.py` writes a 343→111→108→28 integer ReLU network from
zero, with 1,990 nonzero parameters and no training or checkpoint input. The
frozen first version matches a symbolic policy on 27,017 legal scores, native
Rust on 599 positions, and passes a 28-position planner smoke test. In 1,008
paired deals per opponent, it wins 68.90% [67.31%, 70.49%] against random legal
play and 41.27% [39.88%, 42.66%] against the trained standalone field. Fixed bid
30, no auction; this is not full-player strength. Model and replay ledger are
archived on Hugging Face; Plunge and the shipped net are unchanged.
[[bourne-written-field-2026-11-02]] has the protocol, named circuits and receipts.
