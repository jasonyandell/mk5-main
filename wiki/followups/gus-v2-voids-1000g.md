Reviewed against code on 2026-07-07 — no issues found.

All metrics (66.1→65.2 π_me, 37.2→38.6 belief, V/Q MAE), the VoidsEncoder mechanism ([24]-dim = 3 opponents × 8 suits, added to pooled state_emb), and file paths (gus/model/voids.py, gus/train/train_v2_voids.py) match commit 3c02d10; the 57.9% baseline figure matches wiki/experiments/gus-4head-baseline.md, and sources/2e4f586 exists.

- Could try ablating voids on a larger corpus/model — the "already inferred attentionally" conclusion is only tested at d=192/1000g.
- A cheap probe: linear-probe the v1 encoder's pooled embedding for void indicators to directly confirm attentional void inference rather than infer it from the flat delta.
