Reviewed against code on 2026-07-07 — no issues found.

- All headline numbers (88.9%→70% bot-match, eq_outcome_distribution 15→2, 5–11 vocab mentions/trace, ~11 KB / 2.7K tokens) match the b8116b5 commit record exactly; primer.md is verified at 1549 words and `game_summary` exists in burl/tools/engine.py.
- The N=10 eval underlying the 18.9pp regression is small; a cheap follow-up would be rerunning the same primer condition on a larger held-out set to bound the noise on that delta.
- The page's claim that `game_summary` is "not yet wired into native registry" is worth rechecking if the page is revisited — it may have been wired in later iterations.
