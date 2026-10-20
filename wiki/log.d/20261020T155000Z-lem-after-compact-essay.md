## [2026-10-20 | claude | lem: the after-compact hook reloads the essay's words, and the card leads with the work]

- Diagnosis (Jason, 2026-10-20: after one compact, models drop the article's words, revert to
  priors and test off-path): the essay postdates every model's training, so what a compaction
  summary drops is refilled from the superposition literature. The hook reloaded only our card's
  guardrails, about 1,650 words with no sentence of the essay and 19 uses of the prior's terms.
- `.claude/hooks/after-compact.sh` now reloads six method sections from the local verbatim copy
  (Cars and Animals, Recursive Generalization, How I Think About Neurons, What I Mean by Clean,
  Motifs Over Monosemanticity, What a Solution Might Look Like), checks it against SOURCE.txt's
  sha256, and says plainly when the copy is missing. The copy stays local and uncommitted.
- [[clean-neurons-42]] gains "The work": the essay's loop (label, circuit, motif, formation,
  steering, look harder) and the unit paper as the artifact. The hook reloads that section; the
  guardrails stay on the card.
- CLAUDE.md: Lem sessions hand off and restart rather than compact.
