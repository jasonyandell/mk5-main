Reviewed against code on 2026-07-07 — no issues found.

- All headline numbers (K1 54% = 27/50, mean_eq_delta -5.07, tool histogram 72/8/8, 23/23 rationalization convergence, $0.69/$0.91, 1479→5517 chars) match the fd6032b commit message and wiki/sources/fd6032b.md; the rationalization mechanism (ground-truth play appended to the system prompt) matches the script at fd6032b.
- Note for maintainers: `burl/eval/run_move4_star_rollout.py` at HEAD is now the iter-2 EQ-gate rewrite (reveal-and-rationalize replaced due to the 100% yes-bias this page flagged); the page's @fd6032b pin is still correct, but a forward-pointer to the eq-gate replacement could be added.
- The corpus file `burl/data/star_iter0_corpus.jsonl` is gitignored as stated — unverifiable in-repo, regenerable per the page.
