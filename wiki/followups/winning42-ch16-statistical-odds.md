Reviewed against code on 2026-07-07 — no issues found.

- All exact-enumeration figures (C(28,7)=1,184,040; void distribution 42.314/46.982/10.349/0.355; double-count distribution 9.821/32.081/36.091/17.692/3.931/0.372/0.0124/0.000084; modal (2 doubles, 1 void) = 18.011%) were independently recomputed by full enumeration and match to the reported precision.
- Referenced paths verified: `scratch/winning42/winning42.with_figures.md` (main checkout; scratch is gitignored so absent in worktrees) and `gus/eval/strategy_probe.py` exist.
- Unverifiable: bead references `t42-ni1l.16` and `t42-br7n.7` (beads retired 2026-06; the 58.098713% partner two-plus-double prior is attributed to the bead and was not recomputed here — a cheap next probe would be to recompute it, it's a one-line hypergeometric enumeration).
- Follow-up: the 27-configuration four-trump counts (10/27, 14/27) were checked only for internal consistency with the stated partner/non-threat interpretation, not re-derived; a tiny script enumerating the 3^3 assignments would pin them permanently.
