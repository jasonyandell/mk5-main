## [2026-10-22 | claude | night-math: the whole ending in Silas; good play meets only the knots]

- Silas rebuilds the note's 560-deal no-trump ending on its own rules (`silas/src/family.rs`,
  `silas/tests/controller.rs`) and reproduces every number:
  - 14 thresholds and 31 classes;
  - the 767-state controller, root ladder 1, 18, 31, 31;
  - 302 states with 2-1 committed.
- It derives the gap (5p − 1)/(1 + 18p) that Lean had taken as given.
- New: the root optimum has five affine pieces, so five fixed plans suffice and are needed
  (`scenario/lean/Stab/Plans.lean`, two lemmas).
- Along good play only the four knots of the optimum move the best play. The path controller
  has 132 states; the other ten thresholds are counterfactual.
- The comparison dimension (`silas/src/comparison.rs`) is 2 at the note's local history and
  168 at the root.
- Page: [[comparison-algebra-2026-10-21]].
