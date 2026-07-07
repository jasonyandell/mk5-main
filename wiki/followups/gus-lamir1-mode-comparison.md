Reviewed against code on 2026-07-07 — no issues found.

- Commit hashes (7d2af99, e4e6862, 566bc4d, 8544fbe, fb03970, b42669a) all resolve and their messages match the page's claims verbatim (regret 2.777 / 59.46%, per-trick-pos 7.41 and 0.43, V_head distribution-shift finding).
- Headline numbers cross-check against wiki/experiments/gus-lamir1-pilot.md and wiki/topics/lamir1-ceiling.md; the page's own 2026-07-06 correction note (canonical post-fix v-bootstrap = 1.645) is accurate.
- gus/eval/lamir1.py implements all described modes (`_MODES` includes direct, v-bootstrap, q-bootstrap, lamir1, lamir1-qleaf) and `_world_game_hands` exists as described for Bug 5.
- Minor follow-up: the code now also has `q-bootstrap-belief` and `lamir1-piopp` modes (post-cluster); the "results in later ingest" placeholders for q-bootstrap/qleaf could be closed out by pointing to the lamir1-ceiling ladder, which already carries their final numbers.
