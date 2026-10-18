# Gus — a tiny GPT for 42

**[Gus](https://huggingface.co/jasonyandell/gus)** began with the 500k-deal model
tagged `pretrained-v0`; population candidates have `gus-gNNN` tags. The model cards
supply weights, exact token schema and loading instructions. Population reward training and the depth ladder are
implemented in `population.py`, `reward.py`, and `campaign.py`; the
[registered campaign](../../wiki/experiments/gus-population-depth-2026-10-18.md)
records the exact scope, stopping rule, and current results.

The optional `python -u -m lem.observer.plot --root <campaign>` command writes
`progress.png`, `progress.pdf` and their exact source values in `progress.json`.
It needs Matplotlib; a separate plotting environment keeps training dependencies
fixed. Rerender after reserved confirmation before final publication.

The first lesson is **next-tile prediction**. The model sees one player's original
seven tiles and the public play history, then guesses the next tile. It learns
recorded behavior, not an optimal move or a value function.

`model.py` adapts [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291),
preserving its MIT license in `LICENSE.nanoGPT`. The surrounding data and experiment
code is CC0. There are no downloaded weights.

## Follow a prediction

1. `data.encode` turns **only** the viewer's original hand, the declaration,
   bidder, fixed bid 30 and public seat/tile pairs into tokens. Seats are relative
   to the viewer. The 46 context tokens include 28 tile IDs.
2. `Observer` turns token IDs and positions into width-64 vectors. Two residual
   blocks each apply two attention heads and a 256-unit GELU MLP. Layer
   normalization precedes each operation.
3. At each actor token, the model produces 28 tile scores. Softmax makes them
   probabilities. The output tile weights are the same weights used to embed tiles.
4. Cross-entropy is `-log(probability of the recorded tile)`. Only the 28 play
   targets contribute loss; there is no loss on sorted hand tokens or metadata.
5. Backpropagation and AdamW adjust the parameters. Every prediction is causally
   masked, so a future tile cannot help predict an earlier tile. Each batch row
   is a separate episode: there is no attention across deals.

The causal mask cannot hide private information already placed in the prefix.
That is why `encode` accepts only one initial hand. Using a later remaining hand
to supervise earlier moves would leak the future.

## Seat the frozen model

`ObserverPlay` implements the arena's batched play interface. It uses the exact
training prefix for the acting seat and chooses the highest-scoring legal tile.
The legal mask constrains outputs; it is not another input. Bid values other than
30 are refused because the current tokenizer cannot represent them.

```sh
scripts/cpu-slot --threads 2 -- scratch/venv/bin/python -u -m lem.observer.h2h \
  --candidate scratch/lem/observer-corpus-squeeze/best.pt \
  --opponent scratch/lem/observer-data-scale/best.pt \
  --exclude-deals scratch/lem/observer-corpus-squeeze/deals.json \
  --games 2 --seed 20261018 --device mps --out scratch/lem/h2h-smoke
```

Opponent may also be `random`, another checkpoint (including `initial.pt`), or
`walt:<arena-spec>`. Both partnerships use `Bid30Bidder`; games run to seven
marks with paired team swaps. The report resamples complete seed pairs and
checks the replay and optional excluded physical deals. Complete game records
are saved alongside checkpoint hashes and the runtime configuration. Output
directories must be fresh. This measures frozen play; no learning occurs.
The default is now one duplicate pair (two games): check that the integration
runs and replays legally. A smoke's win count or bootstrap interval is not a
strength verdict. Larger measurements need an explicitly selected game count.

## Train, then look

Use this checkout's `scratch/venv` from `scripts/check`. PyTorch and NumPy are its
existing dependencies. Source directories contain `pour.npz` and `manifest.json`;
these are the Pour A `merged/pip` and `merged/dtnt` directories, or raw chunk halves
from [the public pour](https://huggingface.co/datasets/jasonyandell/texas-42-lem-pour-a).
The loader reads only `hands`, `decl` and `plays`, not the large worlds/outcomes arrays.

```sh
scripts/cpu-slot --threads 2 -- scratch/venv/bin/python -u -m lem.observer.train \
  --source /path/to/pour-a/merged/pip --source /path/to/pour-a/merged/dtnt \
  --out scratch/lem/observer-first --deals 5000 --steps 2000 --device mps

scripts/cpu-slot --threads 2 -- scratch/venv/bin/python -u -m lem.observer.uncertainty \
  scratch/lem/observer-first

scratch/venv/bin/python -u -m lem.observer.inspect scratch/lem/observer-first \
  --out scratch/lem/observer-first/worked-hand.html

scripts/check py-lem wiki reuse conflict-markers
```

The lesson is a standalone HTML file. It selects the first test deal in seed
order; there is no search for a flattering example. It lets the reader change
viewer, play, block, attention head and MLP unit; compare predictions with the
saved untrained twin; and reveal the recorded tile. The traces are native units,
not an interpretation or a causal attribution.

For a machine with CUDA, explicitly choose `--device cuda`. `cpu` is useful for
small correctness tests. Model sizes can be selected with `--width`, `--layers`
and `--heads`; width must divide into heads.

## Evidence and limits

### Live W&B and a longer continuation

Install the optional SDK with
`uv pip install --python scratch/venv/bin/python -r lem/observer/requirements-wandb.txt`.
W&B uses its existing login (`docs/SECRETS.md`); credentials do not go in arguments
or source. Add `--wandb-project lem-observer --wandb-entity YOUR_ENTITY
--wandb-name YOUR_RUN` to enable online tracking. The live URL is printed and
saved in the output directory's `wandb-run.json`.

`--init-from EXISTING_RUN --out NEW_RUN` continues a saved model, optimizer and
sampler into a separate experiment, preserving the original lesson and run.
`--resume` continues the same output directory and the same W&B run ID. Choose a
larger total `--steps`; use `--checkpoint-every 5000` for a long formation study.
It must be a multiple of `--eval-every`. Latest and best are still saved at every
evaluation; the periodic formation snapshots are also uploaded to W&B.
Steps 250, 500 and 1,000 also receive full validation and formation snapshots,
so a coarse later interval does not erase early learning. Use repeated
`--wandb-tag` arguments to describe the experiment; overtraining is not assumed.

Charts use `epoch_equivalent = training_step * batch / training_views`. Sampling
is with replacement, so these are equivalent passes, not completed shuffled
epochs. Train-probe metrics use the first 256 training deals, all four views;
validation uses the full validation split. Test is evaluated only after training,
for both the best and final checkpoints. No early stopping is applied.

The deliberately long run is registered in
[the overbake experiment](../../wiki/experiments/lem-observer-overbake-2026-10-17.md).
Ordinary heavy work retains the default cpu-slot time cap; that run's explicit
30-minute exception and fixed update budget are documented there.

The capacity-pressure run is registered in
[the corpus squeeze experiment](../../wiki/experiments/lem-observer-corpus-squeeze-2026-10-17.md):
500,000 distinct deals, unchanged architecture and objective, about eight
equivalent passes. Preparation preallocates numeric arrays so the full pours do
not first become billions of Python boolean references. Tokens and labels are
unchanged; early formation, saved initialization and native units remain available.

### What a result measures

For a nested data experiment, compare both models on the same fresh test deals:

```sh
scripts/cpu-slot --threads 2 -- scratch/venv/bin/python -u -m lem.observer.compare \
  scratch/lem/observer-overbake scratch/lem/observer-data-scale \
  --step 20000 --out scratch/lem/observer-data-scale/comparison.json
```

This verifies identical initial parameters and unchanged original recordings /
splits, then separates test deals outside the original corpus from the reused
test cohort. Both fixed-update and validation-best comparisons use paired
whole-deal intervals. It reports native predictions without legality filtering;
visible-uniform guessing and the saved untrained twin remain baselines.

The first N deals by seed are split by a hash of the physical deal, before
constructing four views. Train, validation and test deals are disjoint. Each
checkpoint records the config, tokenizer version, data hash, optimizer, sampler
and RNG state. `initial.pt`, genuine `best.pt`, `latest.pt` and periodic snapshots
are distinct. Evaluation does not consume the sampler. Resume with the same
arguments plus `--resume` and a larger total `--steps`; extending after inspecting
test results makes that test exploratory, not a fresh confirmation.

The raw test metrics include next-tile accuracy, log loss and probability on
visibly impossible tiles, separated into own/other plays, discretionary/forced
moves and coverage-forced/book declarations. The visible-support baseline uses
own legality or public exclusions for another seat; it never reads that seat's
hidden hand. Hidden-hand legality labels discretionary moves for diagnostics
only. The model's probabilities are never repaired with a legality mask.

Views and moves from one deal are correlated. The uncertainty command bootstraps
whole deals, keeping all their views and plays together. Its intervals condition
on this one trained model, not uncertainty across training seeds. The fixed
bid-30 pour has no auction, persistent identities or cross-hand memory. All
predictions use the recorded past; this is not a generated game or a strength
evaluation. No native neuron has yet been named or causally tested.

Registration and results: [the first observer experiment](../../wiki/experiments/lem-observer-2026-10-17.md).
