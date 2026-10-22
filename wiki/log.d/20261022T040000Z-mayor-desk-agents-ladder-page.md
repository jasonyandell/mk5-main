## [2026-10-22 | mayor | Desks become agent files; The 42 Ladder page]

- Four desks now (Mayor, Media, Book, Lem), each an agent file in `.claude/agents/` with its
  model and effort, run as short sessions that start from the latest handoff in
  `~/code/mk5-main/scratch/desks/<desk>/` and end by writing the next one. Archive and Sweep fold
  into the Mayor as subagents; the Mayor's watch is a Haiku session (`mayor-loop`). Playbook:
  [[standing-desks]].
- The tournament ladder's loop moved from a job's temp folder to `scripts/ladder-loop` (180 s
  chunk budget, so the last faceoff ends inside cpu-slot's 300 s cap), and its standings export to
  `scripts/ladder_export.py`. The page: [The 42 Ladder](https://claude.ai/artifact/TJYRL2ac7afwMookxGPdhp),
  18,247 faceoffs and 37,514 games as of 2026-10-22. Zeb 0.4.1 (contested) leads at 1812
  [1797, 1826]; the Demon with the c102 straight bidder is next at 1787 [1773, 1802], intervals
  overlapping. See [[tournament]].
