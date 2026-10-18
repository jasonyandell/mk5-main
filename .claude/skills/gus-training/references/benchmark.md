<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

# Frozen games and representation checks

Start at `wiki/experiments/gus-frozen-benchmark-2026-10-18.md` for the immutable
HF revision and current measured results. Download the complete study prefix
from `jasonyandell/texas-42-zeb-research-2026-10` at that revision, preserving
relative paths. Verify publication checksums. The local original lives in
`scratch/lem/gus-frozen-benchmark-v1`; generated data never belongs in Git.

The panel fixes 18 duplicate pairs, 36 reference games and 11,116 decisions.
Sixteen pairs are reused earlier regression games; two G0/Zeb anchor pairs
were selected before outcomes. Use the same rows for every new checkpoint.
Do not create a new seed sample each time or count repeated scores as fresh
evidence. Include `panel/reserved-hands.npy` in every subsequent training
exclusion: all 234 potential scheduled deals, not just the 217 previously
played deals. Do not train on these benchmark rows or their exact certificates.

## First: a few-second static check

Run from the selected checkout, with a fresh result directory:

```sh
scripts/cpu-slot --threads 1 --timeout 300 -- scratch/venv/bin/python -u \
  -m lem.observer.benchmark.evaluate \
  --benchmark scratch/lem/gus-frozen-benchmark-v1/panel \
  --checkpoint PATH_TO_CHECKPOINT --head tile \
  --reference scratch/lem/gus-frozen-benchmark-v1/evaluations/g0 \
  --out scratch/lem/MODEL_NAME-frozen-eval
```

Use `--head splat` only for a joint model deployed by its make/set forecast.
The original tile-only observer rejects it. Score the actual deployment head
and record it; changing heads changes the policy. Use the saved parent capture
as a reference for a continuation, as well as G0 for a common anchor.

Only the original clean tokens enter the network: own original hand and public
contract/history. Legal masks act on outputs. Outcomes, reference moves, exact
values, public bookkeeping labels and full-hand audit files never enter it.

`report.json` holds raw/legal NLL, reference-action agreement, impossible mass,
entropy, restricted exact regret, and observed-outcome proper scores. Prefer
the unsettled discretionary slice. Splat forecast scores use the **recorded
action's** outcome even when the new model would choose another tile. Pair-level
bootstrap intervals describe this fixed regression panel, not repeated-model
selection or fresh population confirmation.

`representations.npz` preserves stable row IDs, all 28 logits/probabilities,
legal probabilities and choice, residuals, MLP pre/post activations, causal
attention and, for joint models, all 28×43 forecast probabilities. It is the
native-unit handoff for `clean-neuron`. Coordinate drift is descriptive, not
proof of unit identity or a causal explanation. Existing outputs are reused
only when every recorded identity and representation checksum matches.

There are 471 exact finite-model endgames, of which 287 have tied best choices.
Uniform legal choice is already 80.47% optimal-answer accurate, with 5.336
percentage points of expected regret. The three-style continuation population
differs from actual Gus/Zeb play. Retain this baseline and tied-best set;
missing certificates mean missing counterfactual evidence.

## Then: actual play on a fixed anchor

```sh
scripts/cpu-slot --threads 1 --timeout 60 -- scratch/venv/bin/python -u \
  -m lem.observer.benchmark.h2h \
  --benchmark scratch/lem/gus-frozen-benchmark-v1/panel \
  --checkpoint PATH_TO_CHECKPOINT --head tile --pair anchor-0 \
  --out scratch/lem/MODEL_NAME-anchor-0
```

`anchor-1` is the second frozen G0/Zeb pair. Both swap absolute teams, use
the common bid-30 bidder and freeze all thirteen potential initial deals and
contracts. A changed move produces fresh subsequent play; never replay a
reference opponent's later moves blindly into a changed state. Reports retain
full games, paired known outcomes and mark-margin changes. A single pair is
an integration/regression observation, not a reliable strength estimate.

The wrapper verifies reference assets, audit/schedule hashes, 173 runtime
source pins and exact Python/NumPy/Torch/native-package versions. It refuses
silent protocol drift. Source pins govern native builds; new-host numerical
parity still requires reproducing the known G0 reference trajectory. A runtime
change requires a new declared protocol and parity evidence. No timeout becomes
a loss or a reason to reroll seeds; retain incomplete artifacts explicitly.

## Publish once per completed comparison

Archive reports, arrays, cost logs, source/weight hashes and any actual games
together in one hash-checked HF commit. The first release contains four model
captures and a G0 anchor repeat matching every reference action/outcome.
Use the wiki's immutable revision to locate them. Keep future run entries
distinct; do not overwrite the frozen baseline or promote a player from a
static regression score alone.
