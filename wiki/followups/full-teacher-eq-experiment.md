Reviewed against code on 2026-07-07 — no issues found.

- Commit `6081420` message matches the page verbatim (1059 cycles, ~74% vs random, single-line `--eval-every` addition), and `3e6ad2b` matches the described [N,7,85] PDF → p_make policy-target fix (in `forge/zeb/eq_player.py`).
- `forge/zeb/learner/go-full-teacher.sh` confirms `--eval-aux-policy-weight` default 1.0 and 25% eval-aux mix; the 95% regime was presumably set via env override at runtime — not verifiable from the script alone (page already treats it as chat-sourced).
- Minor precision nit (not corrected): the bugfixed file is `forge/zeb/eq_player.py`, not under `learner/` — the page names only the basename, which is fine.
- Cheap next probe if this era is ever revisited: the page's own framing suggests testing distribution-consuming targets (e.g., train the policy head on the full E[Q] histogram via a distributional loss) rather than another p_make-collapse mix sweep.
