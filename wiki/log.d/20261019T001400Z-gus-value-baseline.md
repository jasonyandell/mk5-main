## [2026-10-18 | codex-gus | Gus value diagnostic and PPO baseline]

<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

[[gus-value-baseline-2026-10-18]] completes the registered frozen-feature
diagnostic: held-out terminal-success prediction improves over fitted priors
and matched random features, without an actor update. Public evidence includes
all chunks, fits, independent audit and actual-signal recovery. The next estimator
in [[gus-success-probability-planning]] is PPO with a separate linear value
baseline, complete returns and unchanged clean inputs/fixed other seats.
Actor and critic recover exactly in CPU interruption tests; the real MPS smoke
preserves trajectories but fails strict actor-weight identity at a small numeric
delta. The training skill records the distinction. No playing promotion.
