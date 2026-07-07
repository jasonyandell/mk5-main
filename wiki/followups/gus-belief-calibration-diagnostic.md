# Follow-ups: gus-belief-calibration-diagnostic

Reviewed against commit 137a8e7 and gus/PRACTICALITIES.md receipt 15 on 2026-07-07.

## Corrections

- Page said KL 0.078 → 0.062 was "47% closer to perfect"; the receipt glosses the 47% as closing the gap "between uniform-prior and a hypothetical perfect belief" — the raw KL drop is only ~20% (evidence: gus/PRACTICALITIES.md receipt 15 @ 137a8e7). Added the uniform baseline row and the receipt's gloss. Note (second pass): the receipt does NOT say the perfect-belief floor is 0, and the numbers rule that out — (0.081−0.062)/0.081 ≈ 23%, so 47% implies a floor of ~0.041 (plausibly the finite-M sampling floor of the empirical marginal).
- Page listed blunder detector AUC as "flat"/PR-AUC "regressed" without numbers; receipt has ROC-AUC 0.792 → 0.793 (flat) and PR-AUC 0.175 → 0.133 (regressed) — filled in the actual values (evidence: gus/PRACTICALITIES.md receipt 15).

## Notes

- Training script `scratch/train_belief_distribution.py` (named in the receipt) no longer exists — scratch/ is gitignored, so the artifact is unrecoverable from the repo. The page's "kept in scratch/" is quoting the commit but the file is gone.
- Base adapter `v2_voids_3000g_big` is not named in receipt 15 itself but is the standing promoted adapter referenced across the repo (skills, wiki index); plausible, not directly verified.

## Follow-ups

- ~~If the co-training takeaway is ever acted on, this page should link to that experiment; currently the "co-train {belief, world_encoder, Q_head}" implication has no successor page.~~ **Wrong — already done.** The co-training was run in §21 (cf8ff79) and has a wiki page: `wiki/experiments/gus-belief-co-train.md` (which already links back here). The diagnostic page now links to it and states the §21 outcome (co-training falsified the propagation hypothesis; q-bootstrap-belief was the win).

## Review (second pass, 2026-07-07)

- Downstream-table correction stands: ROC-AUC 0.792 → 0.793 (flat) and PR-AUC 0.175 → 0.133 (regressed) match the receipt 15 table exactly (evidence: gus/PRACTICALITIES.md §15, lines ~317–323).
- Amended the 47% gloss on the page: the auditor's replacement asserted the perfect-belief floor is "(0)", which the receipt never states and which the receipt's own numbers contradict ((0.081−0.062)/0.081 ≈ 23%, not 47%; a 47% closure implies a floor ≈ 0.041). Page now quotes the receipt's gloss verbatim and flags the implied nonzero floor (evidence: gus/PRACTICALITIES.md §15 @ 137a8e7).
- Dropped the "no successor page" follow-up as factually wrong: `wiki/experiments/gus-belief-co-train.md` (§21, cf8ff79) is the co-training successor. Added the link + outcome to the diagnostic page's Takeaway and Links (evidence: wiki/experiments/gus-belief-co-train.md; gus/PRACTICALITIES.md §21).
- Notes verified: `scratch/train_belief_distribution.py` absent from worktree and never committed (`git log --all -- scratch/train_belief_distribution.py` is empty); base adapter `v2_voids_3000g_big` is not named in receipt 15 (it appears as "best single" in the §14 router table, line ~226) — "plausible, not directly verified" is the right caveat.
