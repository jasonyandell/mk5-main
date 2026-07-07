# w42-bookval-v1-wave1-mark-utility-transform — audit 2026-07-07

## Corrections

- Page said 10 genuine flips (3.6% of 280; 15.4% of flips) vs 55 zero-gain (84.6%); artifact says 12 genuine (4.3%; 18.5%) vs 53 zero-gain (81.5%) — the 81.5% figure elsewhere on the page was already correct, so the page was internally inconsistent (evidence: w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/action_mark_ev_scalars.csv, filtering top1_flips rows: 65 flips, 12 with mark_gain_if_flip > 0).
- Page said genuine flips have mean EV-cost 1.46 pts and mean mark-gain 0.044; artifact says 1.32 pts and 0.030 over the 12 genuine flips. The 1.46 figure matches only the 10 detector-endorsed genuine flips (detector_endorsed_flips.csv), whose mean gain is 0.032 — 0.044 matches neither subset. Page now states both subsets explicitly.
- Note: these errors originate in the run's own README.md (w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/README.md), which contains the same 10/55/84.6%/0.044 numbers alongside the correct 81.5% — the wiki page inherited them faithfully.

Verified clean: headline 23.2% flip rate (65/280), 54 endorsed flips, no-trump 46.4%, blanks 7.1%, bidder 32.9%, setters 14.3–18.6%, Ch10 enrichments (1.21x/1.13x/0.95x/0x), top-5 flip table, transform mechanism in run_mark_utility_transform.py, all artifact paths, input PT SHA, and repo commit 12064bf.

## Follow-ups

- The run README.md carries the same wrong 10/55/0.044 numbers; worth a one-line errata there in a session that touches that directory.
- The 12-vs-10 gap is exactly the two genuine flips with no active detector — a cheap probe: inspect those two decisions to see whether the detector vocabulary is missing a family, or they are noise (gains 0.002–0.092 range).
- mark_gain values quantize at 0.002 (2/1000 worlds); a >= threshold (e.g. 0.01) rather than > 0 would give a more robust genuine/artifact split.
