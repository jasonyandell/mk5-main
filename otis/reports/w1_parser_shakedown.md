# Otis W1 parser shakedown — eq-corpus leg (P1)

Integration + shakedown of the otis fate parser against the TypeScript cross-engine
referee, on the bid-30-fixed eq corpus. This is the **eq-corpus leg** of prediction
P1. The **fresh-arena leg** (on-policy self-play under the incumbent
`margin:wp(r8)` bidder + `lens:ev` play) lands in **W2** — it is not covered here.

## Headline

| Metric | Result |
|---|---|
| Games parsed | **1000** |
| Chunks used | **10** (`corpus_train_chunk_{0-99 … 900-999}.pt`) |
| Parser-side P1 identity pass | **1000 / 1000 (100%)** |
| Referee-checked games | **900** (all non-`doubles-suit`) |
| Referee per-team-points match | **900 / 900 (100%, 0 mismatches)** |
| Export wall | 3.86 s (10 chunks, torch mmap) |
| Referee wall | 0.53 s (900 games) |

Deliverable data: `scratch/otis-night/w1_games.jsonl` (interchange),
`scratch/otis-night/w1_fates.csv` (fate rows), `scratch/otis-night/w1_referee_out.jsonl`
(referee output), `scratch/otis-night/w1_fate_dist.json` (distribution tables),
`scratch/otis-night/w1_shakedown.json` (machine-readable summary).

## What P1 asserts

For every parsed game: `team0_count + team0_tricks + team1_count + team1_tricks == 42`,
each team's `points == count + tricks`, **and** those per-team points equal the
TypeScript engine's own scoring of the same 28-play action sequence. The parser
asserts the internal identity in `parse_game_fates` (`otis/fates.py:315`); the TS
referee (`otis/referee/check_scores.ts`) is the independent second engine.

Both legs held: 1000/1000 internal, 900/900 cross-engine.

## Corpus

10 chunks, 100 games each, exactly **10 games per declaration** (balanced):
`blanks, ones, twos, threes, fours, fives, sixes, doubles-trump, doubles-suit, notrump`
= 100 games each across the 1000. All games are full 28-play playouts, bid-30 fixed.

## Bugs found and fixed this session

1. **Referee early-termination (FIXED — referee translation layer).**
   The base TS engine short-circuits to the `scoring` phase as soon as the bid
   outcome is mathematically decided (`checkHandOutcome`,
   `src/game/core/handOutcome.ts`; consumed at `src/game/core/actions.ts:331`),
   after which it refuses further plays and `teamScores` holds only a **partial**
   count. The forge eq-corpus, by contrast, is a **full 28-play playout** that
   distributes all 42 points. The referee originally played all 28 dominoes and
   hit `play N but phase is scoring` on the first game whose contract decided
   early (e.g. `corpus_train_chunk_0-99:0`, a blanks sweep decided after 6 tricks).

   Fix: added an `otis-full-playout` layer that overrides `checkHandOutcome` to
   always return `{isDetermined: false}`, forcing all seven tricks. The 7-trick
   terminal transition (`actions.ts:327`) is untouched, so the hand still ends
   after 28 plays. This changes only **when** scoring fires, never **how** points
   are counted — the engine's own `calculateTrickWinner`/`calculateTrickPoints`
   still do all arithmetic. The parser and corpus adapter were **not** changed;
   the mismatch was purely a referee-translation gap.

## Known referee gap (not a parser bug)

**`doubles-suit` (forge decl 8) — 100 games, excluded from the referee leg.**
Forge's `doubles-suit` = doubles form their own suit with **no** trump power. The
base TS engine has no representation for this contract short of the `nello` layer,
which changes scoring — so the referee throws rather than silently misrepresent it.
These 100 games are therefore not cross-engine refereeable. This is a
representational gap in the referee, **not** a parser defect:

- Their parser-side P1 identity holds (they are part of the 1000/1000).
- The parser scores them with `forge.oracle.tables` / `forge.oracle.declarations`
  — the exact CPU suit algebra backing the GPU engine that **generated** the
  corpus — so the parser agrees with the generator by construction; we simply lack
  an independent second engine for this one contract.

Closing this gap would require a new base-engine `doubles-suit` contract; deferred
(out of scope for the parser shakedown). 900 refereeable games clears the ≥500 bar.

## Fate distribution — per count tile (N = 1000 games each)

`capture_side × played_mode` frequencies on this bid-30 corpus. `played_mode`
describes how the tile's **holder** played it. `holder_team` = the tile's points
went to the holder's own team; `opp_team` = to the opponents.

### 5-5 (10 pts) — holder-capture 74.1%, H(8-class)=2.478 b, H(capture)=0.825 b
| side | led | followed | trumped_in | sloughed |
|---|---|---|---|---|
| holder_team | 336 | 250 | 42 | 113 |
| opp_team | 110 | 56 | 0 | 93 |

### 6-4 (10 pts) — holder-capture 61.9%, H(8-class)=2.586 b, H(capture)=0.959 b
| side | led | followed | trumped_in | sloughed |
|---|---|---|---|---|
| holder_team | 111 | 301 | 65 | 142 |
| opp_team | 58 | 243 | 6 | 74 |

### 5-0 (5 pts) — holder-capture 56.0%, H(8-class)=2.708 b, H(capture)=0.990 b
| side | led | followed | trumped_in | sloughed |
|---|---|---|---|---|
| holder_team | 91 | 210 | 72 | 187 |
| opp_team | 86 | 231 | 5 | 118 |

### 4-1 (5 pts) — holder-capture 54.4%, H(8-class)=2.724 b, H(capture)=0.994 b
| side | led | followed | trumped_in | sloughed |
|---|---|---|---|---|
| holder_team | 80 | 201 | 82 | 181 |
| opp_team | 70 | 229 | 9 | 148 |

### 3-2 (5 pts) — holder-capture 54.3%, H(8-class)=2.719 b, H(capture)=0.995 b
| side | led | followed | trumped_in | sloughed |
|---|---|---|---|---|
| holder_team | 75 | 184 | 97 | 187 |
| opp_team | 56 | 208 | 8 | 185 |

### Per-tile marginal entropy (P2 baseline base rates)

The 8-class marginal entropy is the constant-prediction ceiling a P2 baseline must
beat with info-state conditioning:

| tile | holder-capture rate | H(capture) bits | H(8-class) bits |
|---|---|---|---|
| 5-5 | 0.741 | 0.825 | 2.478 |
| 6-4 | 0.619 | 0.959 | 2.586 |
| 5-0 | 0.560 | 0.990 | 2.708 |
| 4-1 | 0.544 | 0.994 | 2.724 |
| 3-2 | 0.543 | 0.995 | 2.719 |

Reading: the high double 5-5 is the most predictable tile — most often **led** and
**held** by its owner's team (74%), lowest entropy. Capture fate flattens toward a
coin-flip (≈54%) as tile rank falls, and 8-class entropy rises monotonically from
5-5 (2.48 b) to 4-1 (2.72 b). `opp_team × trumped_in` is near-zero everywhere (a
holder who trumps in almost always wins the trick for their own team; the rare
nonzero cases are over-trumps). These are the marginal base rates for P2's baseline
comparison later.

## Reproduce

```bash
# export (10 chunks, one at a time)
.venv/bin/python -u -m otis.export_games \
  --chunk gus/data/corpus_train_chunk_0-99.pt ... --chunk gus/data/corpus_train_chunk_900-999.pt \
  --out-jsonl scratch/otis-night/w1_games.jsonl \
  --out-fates scratch/otis-night/w1_fates.csv

# referee (exclude decl-8 first), then compare per-team points parser-vs-engine
npx tsx otis/referee/check_scores.ts scratch/otis-night/w1_referee_in.jsonl
```

Tests: `.venv/bin/python -m pytest otis/ -q` → 14 passed.
