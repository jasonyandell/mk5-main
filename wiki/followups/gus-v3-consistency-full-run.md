## Corrections

- Decomposition table said the v2-3k baseline regret was 1.346; commit 31f0ec3 shows v2_voids_3000g = 1.391 (1.346 is v3_consistency_3000g). The −41% and −60% figures were computed from 1.391 and are correct as stated. (evidence: git show 31f0ec3)

## Follow-ups

- The page cites only commit messages; if the underlying eval JSON/CSV for the 560-decision regret set exists in-repo, linking it directly would make the numbers re-checkable without git archaeology.
- The qMAE plateau section names two fix candidates — worth a pointer to whether [[experiments/gus-q-head-augmentation]] later tested either.
