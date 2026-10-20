## [2026-10-20 | lem | the first transformer audit (Gus): its untrained units read position for free; training builds declaration and contract units that keep their family]

[[lem-gus-formation-2026-10-20]]: an all-neuron audit of Gus `pretrained-v0`, read-only on Codex's
checkpoints (#203), over 15 checkpoints and three untrained floors, with list v2:
- Untrained Gus units are 0.91–0.94 clean, nearly all by the mover's seat in the trick, because a
  transformer's decision token reads position. Trained Gus (0.517) reads below that floor. This is
  a tools rung, and a position-conditioned audit is filed.
- Training dissolves the free units (0.29 by step 5,000) and builds units named by the
  declaration, the contract and the count. They keep their family, 0.825 against 0.291.
- 3 of 5 predictions held.

The clean-neuron skill's first new friction log since the fold came from this run.
