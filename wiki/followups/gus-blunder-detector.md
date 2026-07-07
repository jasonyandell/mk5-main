Reviewed against code on 2026-07-07 — no issues found.

All headline numbers (v1 AUC 0.926, v2 AUC 0.839, PR-AUCs, regret 1.23→0.46 and 1.13→0.49, ensemble 1.43-1.55 range, oracle-per-decision ceiling 0.36) match commits f90682c / 5373223 / 109f9e1 and gus/PRACTICALITIES.md receipts 11-13. Scripts gus/eval/blunder_detector.py and gus/eval/blunder_detector_student.py exist.

- Cheap next probe: rerun the v2 detector with K=50 sampled worlds at inference (the receipt-13 fix) — no retraining needed, just a flag change in blunder_detector_student.py.
- Note the v1/v2 baseline regrets differ (1.23 vs 1.13) because they use different eval slices; a one-line clarification could pre-empt confusion.
- Status is "active" — if the router pilot ([[experiments/gus-router-pilot]]) closed this line, this page may deserve status: closed.
