# w42-bookval-v1-wave1-mark-utility-transform — audit 2026-07-07

## Corrections

- Page said 10 genuine flips (3.6% of 280; 15.4% of flips) vs 55 zero-gain (84.6%); artifact says 12 genuine (4.3%; 18.5%) vs 53 zero-gain (81.5%) — the 81.5% figure elsewhere on the page was already correct, so the page was internally inconsistent (evidence: w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/action_mark_ev_scalars.csv, filtering top1_flips rows: 65 flips, 12 with mark_gain_if_flip > 0).
- Page said genuine flips have mean EV-cost 1.46 pts and mean mark-gain 0.044; artifact says 1.32 pts and 0.030 over the 12 genuine flips. The 1.46 figure matches only the 10 detector-endorsed genuine flips (detector_endorsed_flips.csv), whose mean gain is 0.032 — 0.044 matches neither subset. Page now states both subsets explicitly.
- Note: these errors originate in the run's own README.md (w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/README.md), which contains the same 10/55/84.6%/0.044 numbers alongside the correct 81.5% — the wiki page inherited them faithfully.

Verified clean: headline 23.2% flip rate (65/280), 54 endorsed flips, no-trump 46.4%, blanks 7.1%, bidder 32.9%, setters 14.3–18.6%, Ch10 enrichments (1.21x/1.13x/0.95x/0x), top-5 flip table, transform mechanism in run_mark_utility_transform.py, all artifact paths, input PT SHA, and repo commit 12064bf.

## Follow-ups

- The run README.md carries the same wrong 10/55/0.044 numbers; worth a one-line errata there in a session that touches that directory.
- The 12-vs-10 gap is exactly the two genuine flips with no named book-position detector (game 6 decision 13, gain 0.030; game 8 decision 10, gain 0.002 — their `matched_position_detectors` slot is `nan`, only shape tags fire) — a cheap probe: inspect those two decisions to see whether the detector vocabulary is missing a family, or they are noise (genuine gains overall span 0.002–0.092).
- mark_gain values quantize at 0.002 (2/1000 worlds); a >= threshold (e.g. 0.01) rather than > 0 would give a more robust genuine/artifact split.

## Review (second pass, 2026-07-07)

- Auditor's numeric corrections all re-derived and confirmed from `w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/action_mark_ev_scalars.csv` (deduped to 65 flipped decisions: 12 genuine / 53 zero-gain = 18.5%/81.5%; 12/280 = 4.3%; mean cost 1.3188, mean gain 0.0295) and `detector_endorsed_flips.csv` (10 endorsed genuine flips: mean cost 1.4593, mean gain 0.0322). Original page's 10/55/15.4%/84.6%/0.044 were wrong as claimed; README.md in that directory still carries them (lines 50–56).
- Amended the page's percentage-point glosses, which overstate win-probability shifts by 2x: mark utility is the ±1 (team0−team1) mark differential (`score_hand_marks` in `w42/phase4_scoring_objective_tests/run_phase4_scoring_objective_tests.py` awards exactly 1 mark to one team at bid 30; `mark_ev` in `action_mark_ev_scalars.csv` spans exactly [−1, 1]), so a mean-utility gain Δ = 2·Δp(win). Fixed: "~3.0 pp" → "~1.5 pp" (the auditor's replacement inherited the original's factor-2 error), top-5 flip #3 "7%" → 3.5 pp, #5 "2.8%" → 1.4 pp. The README's "~4.4 percentage points" gloss has the same factor-2 error on top of the wrong 0.044.
- Tightened follow-up bullet 2: the two non-endorsed genuine flips are game 6 decision 13 (gain 0.030) and game 8 decision 10 (gain 0.002); the 0.002–0.092 range is the span of all genuine gains, not those two.
- Follow-up suggestions otherwise stand: README errata still needed (verified wrong numbers remain), and the 0.002 quantization claim is correct (all genuine gains are even multiples of 0.002).
- Everything the auditor marked "verified clean" re-checked against `summary.json`, `flip_rate_by_slice.csv`, `per_ch10_claim_correlation.csv`: 23.2% (65/280), 54 endorsed, no-trump 46.4%, blanks 7.1%, bidder 32.9%, setters 14.3–18.6%, enrichments 1.21x/1.13x/0.95x/0x, top-5 table, PT SHA, repo commit — all confirmed.
