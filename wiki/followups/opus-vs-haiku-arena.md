# opus-vs-haiku-arena — audit 2026-07-07

## Corrections

- Page said the arena lives at `burl/eval/arena.py`; the actual path is `burl/selfplay/arena.py` (evidence: `burl/selfplay/arena.py` exists; `burl/eval/` has no arena.py; commit 39aafaf touches arena in selfplay).

## Verified

- Results table (Haiku 0/42 $0.84 7m22s; Opus 7/35 $5.58 8m50s), opening leads 3(2|0) vs 2(1|1), 7 eq calls on Opus's opening, tool-count table (trump_declared 24 vs 1, is_legal 92 vs 27, eq ~1.5/turn, conditional_outcome 0), all match `burl/experiments/arena_notes.md` and the 39aafaf commit message.

## Follow-ups

- The "145+ decisions across every model" claim is a cross-session tally not directly checkable in one artifact; a cheap probe would be a grep-and-count over the arena JSONL artifacts to pin the exact denominator.
- Game artifacts `seed_900010.jsonl`/`seed_900010_opus.jsonl` were not found in the repo tree; if they live outside git, note their location on the page.

## Review (second pass, 2026-07-07)

- Verified — corrections stand.
- Path fix re-derived: `burl/selfplay/arena.py` exists at HEAD and at commit 39aafaf (`git ls-tree 39aafaf -r` shows only `burl/selfplay/arena.py`; no `burl/eval/arena.py` ever); the edit touched only that one line, nothing nearby damaged.
- Verified numbers independently against `burl/experiments/arena_notes.md` (Outcome + tool-use tables) and the 39aafaf commit message: results table, opening leads, 7 eq calls, trump_declared 24/1, is_legal 92/27, eq ~1.5/turn, conditional_outcome 0, "4th session observation / 56 decisions" all match.
- Both follow-up suggestions kept: `git ls-files | grep 900010` confirms the game artifacts are not in git, and the 145+ cross-model tally is indeed not pinned by any single artifact in the repo.
