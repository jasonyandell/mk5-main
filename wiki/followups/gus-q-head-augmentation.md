# gus-q-head-augmentation — audit 2026-07-07

## Corrections

- Page said the training script is `gus/train/train_q_aug.py`; the actual file is `gus/train/train_q_head_augmented.py` (evidence: gus/train/, commit a9fa0c6 stat).
- Page compared the 2.216 result only to the 0.551 direct baseline and called it "worse than the pre-bug-fix rollouts"; the recorded direct comparison is lamir1-qleaf + Bug6 without aug at 2.156, and the postmortem explicitly calls the 0.06 delta noise, not signal (evidence: gus/MORNING4_STATUS.md, path (a) section added in 5f390fb).
- Page's postmortem attributed the failure to a random-vs-structured depletion distribution mismatch; the recorded postmortem attributes it to scalar distillation noise being large relative to the action-value gap at decision boundaries — augmentation only fixed a q_mae measurement artifact (8.277 → 8.169) (evidence: gus/MORNING4_STATUS.md @ 5f390fb).
- Frontmatter status was `active`; the experiment is finished and path (a) is closed — changed to `complete`.
- Added recorded training details (15 epochs, lr=5e-5, 160,007 trainable params, adapter `gus/adapters/q_head_aug.pt`) to the Setup section.

## Follow-ups

- The postmortem's diagnosis (scalar distillation noise vs action-value gap) is the same wall named in the mode-comparison page; a cheap probe would be measuring the per-decision action-value gap distribution to quantify how much noise headroom any leaf evaluator has.
