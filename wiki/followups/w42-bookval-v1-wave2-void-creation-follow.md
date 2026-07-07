Reviewed against code on 2026-07-07 — no issues found.

- Headline metrics (EV delta +0.77 [+0.12, +1.42], p_set +0.0087 straddling zero, 52.6% void-better, CVaR +0.17), all six slice rows, and the lead-position comparison column all match `w42/book_validation_v1/wave2/probes/t42-z31l_void_creation_follow/summary.json` and the lead-position probe's summary.json exactly.
- All five artifact paths, both scripts, and the checkpoint `forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt` exist; manifest confirms the follow-position filter (setter following non-trump lead, cannot follow, holds a non-led non-trump singleton).
- Cheap next probe: the suggested Wave 2.E.2 high-bid (35+) re-run — the corpus filter is already parameterized in `build_void_creation_follow_corpus.py`, so only the source chunks/bid filter change.
- The count_exposed slice (N=30, p_set +0.063) is the only sub-slice with a notable signal; a targeted mine to grow that slice to N~150 would settle whether count exposure is the moderator.
