## [2026-10-18 | gus | Splat forecasts with the same clean inputs]

Implemented joint tile/weighted-outcome pretraining in `lem/observer/splat/`,
completed the fixed 500k-deal, 200k-update trajectory, and preserved 84 formation
checkpoints. Paired diagnostics favor G0 over the first splat policy; the bounded
16-game smoke offers no promotion evidence. Exact inputs, labels, trajectory,
diagnostics and replayed games are public CC0. [[gus-splat-pretraining-2026-10-18]]
is the completed result; [[gus-tiny-gpt]] and [[lem-nanogpt-foundations]] route it.
The `gus-training` skill now documents this workflow and its `clean-neuron`
companion. No further optimizer updates follow the evaluations.
