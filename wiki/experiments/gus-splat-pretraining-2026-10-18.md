---
title: "Gus splat pretraining: the same view, richer consequences"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---

## Outcome

The joint model learned the richer forecasts but did **not** improve the
measured decisions over G0. On the shared diagnostic deals, its splat policy
has higher teacher-relative regret than G0; its tile head also loses accuracy.
The fresh game smoke finished 3–5 against G0 and 3–5 against the campaign Zeb.
Four duplicate pairs per opponent cannot establish a general win rate. No
promotion or further training followed these results.

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

## Completed formation and diagnostics

The 185,652-parameter model completed 200,000 updates / 8.005 equivalent passes
in 1,576.45 seconds (26m16s, training plus evaluation). Minimum full-validation
joint loss selected step 195,000. The 84 formation checkpoints include the
untrained twin, early steps and every periodic step; all parameters are finite,
optimizer/sampler state is preserved, and the selection criterion was audited.
Best full checkpoint SHA256:
`f261e0f257047565cb481f88c5480e39e219d951173710c3dbf858b4b9cd563c`.

The paired diagnostic covers 50,038 previously inspected test deals, 200,152
views and 679,985 discretionary own choices before public settlement. Intervals
below use 2,000 physical-deal bootstrap draws; they measure this corpus, not
playing strength against fresh opponents.

| Measure | G0 tile policy | Joint tile policy | Splat policy | Uniform legal |
|---|---:|---:|---:|---:|
| All-play raw tile accuracy | 52.90% | 50.73% | — | — |
| All-play tile NLL | 1.2620 | 1.3275 | — | — |
| Consequential own legal teacher-move accuracy | 58.32% | 56.36% | — | — |
| Consequential teacher make/set regret, percentage points | 2.413 | 2.700 | 3.040 | 5.256 |

Splat minus G0 regret is **+0.627 pp [0.604, 0.649]**, and splat minus its own
tile policy is **+0.340 pp [0.318, 0.362]**. Lower regret is better: the selected
move gives up that much of the best labelled move's teacher make/set probability.
G0's matched advantage is diagnostic evidence against an improvement from this
first joint recipe. The joint tile-accuracy change is −2.166 pp
[−2.212, −2.122].

The splat head's consequential make-probability MAE is **10.050 pp
[10.010, 10.089]**, histogram KL **0.3681 nats [0.3669, 0.3693]**, and tie-aware
teacher-best accuracy **55.47% [55.34, 55.60]**. All-labelled-position histogram
CE is 2.3649 nats, comprising teacher entropy 1.9647 plus KL 0.4002. Probability
reliability bins compare forecasts with teacher probabilities, not actual game
outcomes. Literal inputs, 43-bin predictions and each policy's choice are in
`example.json`; no extra candidate feature enters the prefix.

## Fresh short games

All eight scheduled pairs completed, 16 games / 179 hands, with legal-play and
score replay checks and zero overlap against 584,958 excluded physical deals.
Both partnerships use the same bid-30 bidder; games finish at seven marks.
Zeb is the campaign's frozen 160-world posterior/make configuration, not the
separate tournament's native Zeb 0.4.1 reference. Its precise spec and field hash
are in the published manifests.

| Opponent | Splat wins | Marks per game | Duplicate pairs |
|---|---:|---:|---:|
| G0 | 3 / 8 | −0.875 | 4 |
| Campaign Zeb | 3 / 8 | −2.000 | 4 |

Chance is 50% game wins; neutral marks/game is zero. The exploratory pair
bootstrap win interval is [12.5%, 50.0%] for each opponent, but four pairs cannot
support reliable tail estimates: the conservative bounded-pair 95% interval is
[0%, 100%]. These are smoke results, not a strength ranking. Every faceoff took
under 5.81 seconds, within its 60-second cap; all eight invocations took 28.55
seconds including queue/startup. The full paired forecast diagnostic took 19.2
wall seconds. These times are descriptive, not player thinking budgets.

## Evidence and interpretation handoff

The [complete 1.488 GB formation and evaluation archive](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/ae0c9139dd74a5c281353319bd6db5571223c613/lem/gus-splat-pretraining-2026-10-18)
contains exact compiled targets, all 84 formation checkpoints, full best/final
optimizer states, paired per-deal diagnostics, every scheduled H2H attempt and
replay, cost receipts, literal example, input and trajectory audits, and
[phone-readable training curves](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/resolve/ae0c9139dd74a5c281353319bd6db5571223c613/lem/gus-splat-pretraining-2026-10-18/progress.png).
The manifest, input audit, initialization, best checkpoint, diagnostic report
and H2H summary passed fresh anonymous download/hash verification.

This is a useful specimen for `clean-neuron`: the same initial backbone and
clean inputs now faced a richer supervised task. No native-unit or causal
interpretation has been performed here. Loss balancing, competition for the
small shared representation, and whether histogram fitting preserves fine
between-action differences are hypotheses for later controlled experiments,
not established explanations or authorization for another run.

## Published model and verification

[Gus splat v1](https://huggingface.co/jasonyandell/gus/tree/81a8a2da9dbcaeee754072a90e931c0c72c9d739)
is tagged `splat-v1-20261018`, on separate branch `splat-v1`. The model repository's
main branch remains the earlier campaign release. Its 11 export files passed
fresh anonymous download and checksum verification; safe-format tensors and
both CPU float32 output heads exactly match the source best checkpoint. The
model card pins merged implementation commit `6cbf7db7` and distinguishes the
inference export from the full resumable research archive.

[Publication and verification receipts](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/4d3d70ed1de11806127249d08d757453fb773104/lem/gus-splat-pretraining-2026-10-18)
include final archive verification: every one of its 159 evidence files had
size and LFS SHA256 checked, or a fresh anonymous download for non-LFS files;
the manifest was separately freshly downloaded. This is distinct from fresh
downloading every bulk cache. The source merge passed `scripts/check --fast`:
69 Lem tests, wiki, conflict-marker and REUSE checks; zero skipped tests.
The completed W&B run also links the model/archive and records the paired
regret comparison and both smoke win counts, verified by API readback.
