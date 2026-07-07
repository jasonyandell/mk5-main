Reviewed against code on 2026-07-07 — no issues found.

- Headline metrics verified against `w42/book_validation_v1/wave2/snapshots/reentry_preservation/summary.json` (ev_delta −1.619, CI [−2.93, −0.31], 44.5% preserve-better); artifact tree, seeds 0–17, 50-sample CPU config, and the 6 tests in `forge/eq/test_state_injection.py` all check out.
- A v2 follow-up already exists (`wiki/experiments/w42-bookval-v1-wave2-reentry-v2.md`, artifacts in `snapshots/reentry_preservation_v2/`) — the page could link it from the Interpretation section since v2 is exactly the proposed follow-up.
- Cheap next probe: rerun the same 200 pairs with oracle-greedy off-suit selection only (keep random-play corpus) to isolate how much of the −1.62 comes from the first-available-slot bias.
