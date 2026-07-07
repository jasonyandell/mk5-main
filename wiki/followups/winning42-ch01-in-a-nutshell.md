Reviewed against code on 2026-07-07 — no issues found.

- Source slice verified: `scratch/winning42/winning42.with_figures.md` lines 479-734 do cover "In a Nutshell" through the scoring examples (line 479 opens the chapter; ~728 ends the bid-35/take-33 → 44 example). Note the file lives in gitignored `scratch/` in the main checkout, not in worktrees.
- Count-domino set/values and all three scoring examples (33 vs 9, 0 vs 44, 0 vs 47) are arithmetically consistent with the rules.
- Claim ledger matches [[w42-phase4-laydown-rule-accounting]]: 23 rule assertions, bead `t42-br7n.4`.
- Follow-up: beads are retired (bd → GitHub issues); the `t42-ni1l.1` / `t42-br7n.4` references are now historical identifiers only — a cheap pass could annotate them as archived-bead IDs.
- The three "Underpowered / model-untested" ledger rows (walker regret bucket, Burl trace audit) remain the obvious next probes.
