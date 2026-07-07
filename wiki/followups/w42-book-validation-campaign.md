Reviewed against code on 2026-07-07 — no issues found.

Verified: headline metrics (+5.42 pts/hand [+4.03, +6.81] ev vs p_make; 41.2% [36.8%, 45.6%] argmax divergence; 0/500 p_make/mark_ev disagreement; void-pick rates 29.4/38.6/42.6%), wave5 artifact at `w42/book_validation_v1/wave5/champion_teaching_battery/` (manifest created 2026-06-13), and `select_actions` in `forge/eq/generate/actions.py` really is p_make-argmax with E[Q] tiebreak as the page claims.

- The status table still carries six stale "blocked on 2.A/2.B" rows (2.C–2.H) below their closed counterparts; the page's own audit note explains the table was never maintained, but deleting the superseded rows would remove a reader trap.
- Wave 2 probe artifacts actually live under `w42/book_validation_v1/wave2/probes/<bead>_<slug>/`, one level deeper than the page's `wave<N>/<bead>_<slug>/` template — worth a parenthetical if anyone navigates from the page.
- The `t42-10yj` one-line ev-argmax switch follow-up predates the beads→GitHub-issues migration; a cheap check is whether it was ever carried over or superseded by the champion `lens:ev` result.
