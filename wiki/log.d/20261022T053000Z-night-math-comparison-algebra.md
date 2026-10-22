## [2026-10-22 | claude | night-math: the minimal comparison algebra in Lean, the decision quotient in Silas]

- Lean (`scenario/lean/Stab/Hexagon.lean`, `Residual.lean`, `Mixing.lean`): 22 theorems on the
  three standard axioms, no sorry. They cover:
  - the note's two numbers per decision and the hexagon;
  - the 13 sign patterns as the sequential classes;
  - the coarsest output-keeping right congruence, attained by the quotient machine;
  - refinement exact after n − 1 blocks;
  - convex mixing forcing equality at the 1/5 boundary.
- Two brief claims proved false. The root Q triple has 4 weak orders, not 13, and one block is
  not enough (a three-tile witness needs two). [[gus-decision-quotient-2026-10-21]]'s "every
  ladder stabilizes after one block" is corrected: shape needs two.
- Silas `quotient.rs` and `silas quotient --course FILE` (fed by `lem/quotient.py --export`)
  reproduce `gus-decision-quotient-v1/report.json` in every count: 8,257 raw, 477 shape and 144
  fork states, 0 differences.
- Gate call: these Lean files are additive proofs in `scenario/lean`, as the brief placed them,
  with no Walt code touched, so they merged ungated (precedent cb95168e). Silas changes avoid
  core pins, physics and frame. [[comparison-algebra-2026-10-21]].
