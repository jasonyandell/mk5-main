# Follow-ups: gus-belief-co-train

Reviewed against code and gus/PRACTICALITIES.md §20-21 on 2026-07-07.

## Corrections

- Page's mode table said "q-bootstrap (corpus worlds) 0.679"; for the §21 A/B the original adapter's corpus-worlds regret is 0.685 (evidence: gus/PRACTICALITIES.md lines 849-852; commit cf8ff79 message "0.685→0.718"). 0.679 appears twice in the sources — as the earlier §20/MORNING4_STATUS ladder q-bootstrap result (a different run; gus/MORNING4_STATUS.md line 20) and as the co-trained adapter's belief-sampled figure — so the page's number was likely the stale §20 figure, wrong either way for this table. Table replaced with the full 2x2 (0.655/0.685 original, 0.679/0.718 co-trained) plus §20 full-rollout range 1.645-2.350.

## Follow-ups

- The "smoother belief sampling beats oracle adaptive sampling" mechanism is explicitly marked unverified in §21 — a cheap probe is comparing world-distribution entropy of belief-sampled vs corpus worlds at the decisions where regret differs.
- K sweep for q-bootstrap-belief is already proposed in PRACTICALITIES §21 ("Next experiment (not run): vary K"), which also notes K=200 already matches/beats corpus at M~3000 — suggesting distribution, not diversity, is what's better. Still unrun; worth doing.

## Review (second pass, 2026-07-07)

- Page corrections stand: the 2x2 table (0.655/0.685 original adapter, 0.679/0.718 co-trained), the 4.4% relative claim, and the §20 full-rollout range 1.645-2.350 all match gus/PRACTICALITIES.md (§20 lines 723-729, §21 table lines 849-852) and commit cf8ff79's message ("0.685→0.718", belief-sampled 0.655). Cited artifacts exist: gus/eval/lamir1.py (q-bootstrap-belief mode, lines 407-466, 986), gus/model/sample_worlds.py, gus/train/train_belief_q_joint.py.
- Amended the Corrections bullet's diagnosis: 0.679 is not only the co-trained belief-sampled figure — it's also the earlier §20 ladder q-bootstrap result (gus/MORNING4_STATUS.md line 20, a separate run), the likelier source of the page's stale number. The §20-vs-§21 discrepancy (0.679 vs 0.685 for original-adapter corpus worlds) is a real inconsistency in the source docs, presumably different runs/configs; the §21 A/B is authoritative for this experiment.
- Amended the second Follow-ups bullet: the K sweep is already proposed in PRACTICALITIES §21 as "Next experiment (not run)" with the K=200 vs M~3000 observation; kept it (still unrun) but credited the existing proposal.
