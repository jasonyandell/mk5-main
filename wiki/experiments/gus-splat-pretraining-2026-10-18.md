---
title: "Gus splat pretraining: the same view, richer consequences"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: active
---

## Question and preregistration

Can the small [[gus-tiny-gpt]] learn useful per-action consequence distributions
from the existing lawful pours, with exactly the inputs already approved for
[[lem-nanogpt-foundations]]? Jason authorized implementation and one pretraining
run on 2026-10-18. The earlier splat interface was selected but had not been
implemented; neither the corpus squeeze nor the population campaign trained it.
This is the output extension, not another reward/depth campaign.

Registered before the full training run:

- All 500,000 distinct deals from [[lem-observer-corpus-squeeze-2026-10-17]],
  with the identical physical-deal split: 399,752 train / 50,210 validation /
  50,038 test. All four viewer episodes remain. Previously inspected test deals
  are a development diagnostic, not a new confirmation set.
- Original seven own tiles, bid 30, public bidder/declaration, relative actor
  and public tile history; unchanged 46-token vocabulary, 68-token context.
  No action query, evaluate token, legal-set input or teacher feature.
- Same freshly initialized seed-42 backbone: width 64, two blocks, two heads,
  256 GELU units per block. Add a linear 28 × 43 outcome head. Candidate identity
  is an output axis. Existing tile head remains tied to the input embeddings.
- Labels are weighted marginal histograms of lawful Walt outcomes, mapped from
  margin `m` to bidder-team points `(m+42)/2`. Verify make-tail mass against
  recorded `q` for every action. Hidden world arrays are never opened by this
  compiler. Retain source manifests and compiled-array checksums.
- Splat supervision only at the viewer's own planned decisions and the legal
  candidates with teacher labels. Missing labels contribute no loss. Average
  soft histogram cross entropy over legal candidates, then labelled positions.
  Add mean next-tile cross entropy over all 28 observed plays with weight 1:1.
  The labels' irreducible entropy is reported separately from model KL error.
- AdamW 0.001 constant, betas (0.9, 0.95), matrix weight decay 0.1, gradient clip
  1; batch 64, 200,000 updates, 8.005 equivalent passes sampled with replacement.
  No early stopping or depth growth. Selection is minimum full-validation
  **joint loss**. Test is evaluated only after the trajectory completes.
- Preserve untrained initialization, steps 250/500/1,000, every 2,500, full
  optimizer/sampler/RNG recovery, and separate best/final checkpoints. Full
  validation and fixed 1,024-view train probe at each formation point.
- Local MPS, two CPU threads, shared-machine `cpu-slot`. Cache each source in
  its own 300-second bounded job. Full training gets a 3,600-second outer cap:
  one fixed trajectory with repeated full validation; rebuilding accelerator
  corpus state every five minutes would add overhead. Recovery resumes the
  same update budget rather than extending it based on results.

## Meaning of the target and playing policy

These are bid-30 **make-utility** pours. Their post-settlement continuation is
random legal play. The full histogram describes that teacher-plus-finish rule,
not optimal point maximization or a policy for arbitrary bids. Marginalization
also discards cross-action scenario pairing; the original raw splats retain it.
The behavioral fields and uniform hidden-hand sampling are specified in source
manifests; neither an optimal opponent nor a behavior-conditioned posterior is
implied. The [[strategy-fusion-fence]] remains intact.

At deployment, bidder partners maximize predicted mass at 30–42; defenders
maximize its complement. Output legality is enforced using the acting seat's
hand. Exact ties choose the lowest tile ID. The unchanged next-tile head is also
available as a separate policy for diagnosing which output learned useful play.

## Evaluation and stopping

Report raw next-tile NLL/top-1/impossible mass, own/other slices, and discretionary
own decisions before public settlement. For the new head report histogram CE,
label entropy, KL, make-probability MAE, tie-aware teacher-best accuracy and
teacher regret against uniform legal choice. These measure imitation of named
teachers, not strength or perfect calibration to actual play.

Reserve fresh H2H seeds with namespace `gus-splat-v1-20261018-h2h`: four duplicate
pairs against frozen G0 and four against the campaign's pinned Zeb, two games
per faceoff, outer cap 60 seconds per faceoff. Both sides use the same Bid30Bidder
and first-to-seven arena. Replay every completed game and reject physical-deal
overlap with training or previous experiment panels. Publish all scheduled
faceoffs, including incomplete outcomes. This 16-game pilot is a smoke and
cannot by itself promote a stronger-player claim. No retry based on losses,
no further optimizer updates after seeing these results.

The [[clean-neurons-42]] interpretation question remains ungraded until native
unit/causal evidence exists. The new specimen and matched untrained twin are
preserved for that work. The implementation is `lem/observer/splat/` and the
reusable workflow is the `gus-training` skill, with companion `clean-neuron`.

## Initial evidence

[Exact compiled targets, input audit and untrained twin](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/454bb660295c57f8061d45d82cad014d3265ba90/lem/gus-splat-pretraining-2026-10-18)
are public CC0. Seven compressed numeric caches contain 8,693,386 planned
positions and 31,688,423 legal candidate histograms. Maximum make-tail discrepancy
from source `q`, before float32 cache transport, is 2.22e-16. Every token in all
two million views and every physical-deal split matches the original G0 corpus;
all initial backbone tensors also match exactly. The caches use source order
rather than the original globally seed-sorted training order, so individual
minibatches are not matched between the two trajectories.

Source implementation and registration: `c2903541`. Training telemetry:
[W&B run 28dc40c6](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/28dc40c6).
The initial archive is 1,283,235,412 bytes; small receipts/checkpoints passed fresh
anonymous downloads. All archive sizes and LFS hashes are checked; cache 0 also
passed a fresh anonymous download and all decoded-array checksum checks. The
other six compressed caches have metadata/hash verification, not fresh downloads.
