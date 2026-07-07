## Corrections

- Decomposition table said the v2-3k baseline regret was 1.346; commit 31f0ec3 shows v2_voids_3000g = 1.391 (1.346 is v3_consistency_3000g). The −41% and −60% figures were computed from 1.391 and are correct as stated. (evidence: git show 31f0ec3)

## Follow-ups

- The page cites only commit messages, but no raw eval JSON/CSV for the regret set exists in-repo; the numbers are nonetheless re-checkable without git archaeology via `gus/EVENING_STATUS.md` at HEAD (full v2/v3 comparison table). Linking that file from the page would suffice.
- The qMAE plateau section names two fix candidates — worth a pointer noting that [[experiments/gus-q-head-augmentation]] tested §20 path (a) (partial-depletion augmentation), which is neither of the §18 candidates (multi-world variance reg, joint co-training); both remain untested.

## Review (second pass, 2026-07-07)

- Verified — corrections stand. The 1.346 → 1.391 fix is confirmed by the table in `gus/EVENING_STATUS.md` as of 31f0ec3 ("mean regret | 1.391 | 1.346 | 0.818 | **0.551**"); 1.346 was v3-cons-3k, and the −41%/−60% figures derive from 1.391 as the auditor stated. No collateral damage to the page.
- Independently spot-checked the untouched claims: w_consistency=0.3 default + 10-epoch warmup (`gus/train/train_v3_consistency.py:146-147`), peak RSS ~3.4 GB (`wiki/sources/f138069.md`), 3.4M params / bot-match 76.07% vs 73.21% (`gus/EVENING_STATUS.md` table), qMAE-plateau section matches 41fdb3c's commit message. All hold.
- Refined follow-up bullet 1: no raw eval JSON/CSV exists in-repo, but `gus/EVENING_STATUS.md` at HEAD already makes the numbers re-checkable without git archaeology.
- Refined follow-up bullet 2: [[experiments/gus-q-head-augmentation]] tested §20 path (a), not either §18 fix candidate — the suggested pointer should say "neither tested yet".
