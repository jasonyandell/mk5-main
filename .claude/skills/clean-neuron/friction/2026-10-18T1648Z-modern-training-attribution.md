<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

- Task: apply Cammarata's learning improvements to an already pretrained GPT.
- Friction: the skill routes to the essay, but its learning rung does not
  distinguish a missing architecture feature from one the model already has.
  Reading only the excerpt also conflates visualization regularization with
  training regularization, and supplies no numerical training recipe.
- Resolution: read the complete `https://already-clean.goodfire.com/essay.md`,
  inventory the existing residuals/LayerNorm/optimizer, retain the architecture,
  and label the tested cosine continuation as standard nanoGPT practice rather
  than an article prescription. Preserve both trajectories and native units.
- Suggested integration: require the short source-to-existing-model inventory
  when proposing learning changes. Preserve the separation between learning
  quality, native-mechanism evidence and the observation tool itself.
