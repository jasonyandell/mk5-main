# Follow-ups: gus-belief-co-train

Reviewed against code and gus/PRACTICALITIES.md §20-21 on 2026-07-07.

## Corrections

- Page's mode table said "q-bootstrap (corpus worlds) 0.679"; PRACTICALITIES §21 table says original adapter corpus-worlds regret is 0.685 — 0.679 is the co-trained adapter's belief-sampled figure (evidence: gus/PRACTICALITIES.md lines 849-852). Table replaced with the full 2x2 (0.655/0.685 original, 0.679/0.718 co-trained) plus §20 full-rollout range 1.645-2.350.

## Follow-ups

- The "smoother belief sampling beats oracle adaptive sampling" mechanism is explicitly marked unverified in §21 — a cheap probe is comparing world-distribution entropy of belief-sampled vs corpus worlds at the decisions where regret differs.
- Could test q-bootstrap-belief with more sampled worlds (n sweep) since sampling is now decoupled from the corpus.
