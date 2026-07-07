# Follow-ups: gus-belief-calibration-diagnostic

Reviewed against commit 137a8e7 and gus/PRACTICALITIES.md receipt 15 on 2026-07-07.

## Corrections

- Page said KL 0.078 → 0.062 was "47% closer to perfect"; the receipt defines the 47% as closing the gap between the uniform-belief baseline KL (0.081) and hypothetical perfect belief (0) — the raw KL drop is only ~20% (evidence: gus/PRACTICALITIES.md receipt 15 @ 137a8e7). Added the uniform baseline row and the correct gloss.
- Page listed blunder detector AUC as "flat"/PR-AUC "regressed" without numbers; receipt has ROC-AUC 0.792 → 0.793 (flat) and PR-AUC 0.175 → 0.133 (regressed) — filled in the actual values (evidence: gus/PRACTICALITIES.md receipt 15).

## Notes

- Training script `scratch/train_belief_distribution.py` (named in the receipt) no longer exists — scratch/ is gitignored, so the artifact is unrecoverable from the repo. The page's "kept in scratch/" is quoting the commit but the file is gone.
- Base adapter `v2_voids_3000g_big` is not named in receipt 15 itself but is the standing promoted adapter referenced across the repo (skills, wiki index); plausible, not directly verified.

## Follow-ups

- If the co-training takeaway is ever acted on, this page should link to that experiment; currently the "co-train {belief, world_encoder, Q_head}" implication has no successor page.
