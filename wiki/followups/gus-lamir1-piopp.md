# gus-lamir1-piopp — audit 2026-07-07

## Corrections

- Page said Bug 6 (zeroing played-domino rows from world_assign before the leaf Q_head call) "correctly identifies and fixes the data invariant violation" and matches the training convention; §20 says the fix made regret WORSE across all modes (lamir1-piopp 2.268 → 2.350) because the training convention preserves the ORIGINAL world_assignment as played_mask advances — the zeroing broke an invariant (evidence: `git show b42669a:gus/PRACTICALITIES.md` §20; `wiki/topics/lamir1-ceiling.md` "What Bug 6 revealed").
- Page said the root cause was "Q_head OOD at depleted leaf states"; §20's documented root cause is distilled scalar V/Q noise flipping argmax at decision boundaries (Kubíček & Lisý warning), plus V_head being architecturally world-blind (std=0.000 across 200 world samples) (evidence: `git show b42669a:gus/PRACTICALITIES.md` §20).
- Page's four "§20 pivot options" (partial-depletion augmentation / end-to-end LAMIR / abandon look-ahead / detect-and-route) did not match the source; actual four options in MORNING4_STATUS @ b42669a are: ship q-bootstrap as router second opinion, train look-ahead-compatible V-head, implement LAMIR faithfully (multi-valued states + CFR+), Bridge-AI/BMCS PPO self-play (evidence: `git show b42669a:gus/MORNING4_STATUS.md`).
- Ladder table: added missing "lamir1-piopp + Fix 6 | 2.350 | 62.50%" row, filled q-bootstrap bot-match (72.50%), and fixed the lamir1 note ("argmax opp" → rotated π_me rollout, V_head leaf) (evidence: `git show b42669a:gus/MORNING4_STATUS.md` table; `gus/eval/lamir1.py` module docstring).

## Follow-ups

- The verified numbers themselves (0.551 / 0.679 / 1.645 / 2.006 / 2.094 / 2.268 / 68.6%) all check out against b42669a docs and commit messages; per-mode JSONs (`scratch/lamir1_*.json`) are gitignored so raw eval artifacts are unverifiable in-repo.
- Pivot option 2 (V trained on expected Q under sampled opp play) was never tried per the wiki; a cheap probe now that E[Q]-distill is on the champion v2 roadmap.
