# gus-q-head-augmentation — audit 2026-07-07

## Corrections

- Page said the training script is `gus/train/train_q_aug.py`; the actual file is `gus/train/train_q_head_augmented.py` (evidence: gus/train/, commit a9fa0c6 stat).
- Page compared the 2.216 result only to the 0.551 direct baseline and called it "worse than the pre-bug-fix rollouts"; the recorded direct comparison is lamir1-qleaf + Bug6 without aug at 2.156, and the postmortem explicitly calls the 0.06 delta noise, not signal (evidence: gus/MORNING4_STATUS.md, path (a) section added in 5f390fb).
- Page's postmortem attributed the failure to a random-vs-structured depletion distribution mismatch; the recorded postmortem attributes it to scalar distillation noise being large relative to the action-value gap at decision boundaries — augmentation only fixed a q_mae measurement artifact (8.277 → 8.169) (evidence: gus/MORNING4_STATUS.md @ 5f390fb).
- Frontmatter status was `active`; the experiment is finished and path (a) is closed — changed to `complete`.
- Added recorded training details (15 epochs, lr=5e-5, 160,007 trainable params, adapter `gus/adapters/q_head_aug.pt`) to the Setup section.

## Follow-ups

- A cheap probe would be measuring the per-decision action-value gap distribution to quantify how much noise headroom any leaf evaluator has. The machinery already exists: `gus/analysis/add_real_drama_columns.py` computes `marginal_eq_gap` per decision (currently used only as a drama filter, not as a noise-headroom analysis). Note: the scalar-noise-vs-action-value-gap wall is named in [[topics/q-head-augmentation]] and the gus entity page — the mode-comparison page names a different wall (V_head distribution shift).

## Review (second pass, 2026-07-07)

- All five claimed corrections verified against primary evidence and stand: script name (commit a9fa0c6 stat + `gus/train/train_q_head_augmented.py` on disk), result table 2.156/2.216 with 0.06-is-noise framing, postmortem rewrite, training details (15 epochs, lr=5e-5, 160,007 params, `gus/adapters/q_head_aug.pt`), and status `active` → `complete` — all match `gus/MORNING4_STATUS.md` path (a) section added in 5f390fb.
- Amended the page's Conclusion section, which the auditor missed: it still asserted "the gap between training distribution and rollout distribution cannot be bridged" — the old fabricated diagnosis, contradicting the corrected postmortem (the source says augmentation DID fix the OOD measurement artifact; the real problem is scalar distillation noise vs the action-value gap). Rewrote to match `gus/MORNING4_STATUS.md` @ 5f390fb and added the citation.
- Corrected the Follow-ups bullet's citation: the scalar-noise wall is named in `wiki/topics/q-head-augmentation.md` (and `wiki/entities/gus.md:604`), not in `wiki/experiments/gus-lamir1-mode-comparison.md`, whose named wall is V_head distribution shift. Kept the probe suggestion (still undone — `marginal_eq_gap` exists only as a drama filter).
- Added [[topics/q-head-augmentation]] to the page's Links line — the topic page covers this exact experiment and was not linked.
- Note: the original page's "worse than the pre-bug-fix rollouts" claim was numerically defensible (pre-Bug6 lamir1-qleaf was 2.006 < 2.216, per pre-5f390fb MORNING4_STATUS.md) but cited evidence that doesn't contain it; the auditor's replacement uses the comparison the source actually records.
