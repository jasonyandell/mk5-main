---
title: "Gus route/rescue: actor-query path and scalar restoration"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and preregistered predictions

Does attenuating layer-0 attention to the current lead tile at the final actor
query change L1/U65 more than attenuating the bundle of earlier queries? Does
restoring the actor-site U65 post-GELU scalar to baseline move output
probabilities closer to baseline more than restoring U193?

The frozen predictions were negative for both contrasts: actor-only minus
nonactor-only paired raw-pre65 change at cosine-final, both heads, dose 0.5;
and U65-minus-U193 change in raw probability TV distance after versus before
restoration, on positive rows at cosine-final, both heads, all queries, dose
0.5. The second prediction was specified separately for the two existing
panels. The full Cammarata essay, all 943 lines including appendices and
references, was reread before protocol design; no article synopsis is needed
for this result.

## Fixed rows and interventions

The diagnostic reused exactly the 48 lawful own-actor prefixes selected for
[[gus-first-look-2026-10-19]]: 12 metadata-matched positive/negative physical
pairs in each already-inspected panel. The rows, original 512 saved-record
checksums, model hashes, and input fields were pinned; no activations selected
rows. The two specimens were the original parent and the cosine-final
continuation. The 607,306 reserved physical roots stayed excluded. No games,
training, player search, opponent change, or move evaluation occurred.

Only layer-0 attention changed. The source key was the current lead TILE or its
preceding ACTOR marker; head0, head1 and both-head bundles were tested at doses
0.25 and 0.5, under final-actor-only, earlier-query-only and all-query masks.
Zero-dose and empty-source shams were retained. “Earlier query” is a positional
bundle containing earlier actor tokens; comparing the final query with this
bundle does not estimate per-edge importance. The source weights were modified
before softmax renormalization, so nominal dose is not the measured attention
mass removed.

For all-query lead-tile conditions, the actor-site L1/U65 or control L1/U193
post-GELU scalar was separately replaced with that same row/model's exact
unmodified baseline scalar immediately before the final MLP projection. Other
coordinates and token sites remained at their intervened values. This is a
specific local restoration, not a mediation percentage or a unit ablation.

## Prediction 1: final actor query versus earlier-query bundle

The primary quantity is the paired positive-minus-negative change in signed
raw pre-GELU U65, then actor-only minus nonactor-only. Values below are raw
preactivation units, **not standard deviations**. Intervals are descriptive
95% percentile intervals from 2,000 paired physical-row bootstrap resamples.

| Existing panel | Matched pairs | Actor-only minus nonactor-only | Descriptive 95% interval |
|---|---:|---:|---:|
| Development | 12 | −1.874618 | [−1.939087, −1.810742] |
| Confirmation | 12 | −1.457860 | [−1.786442, −1.046980] |

The prediction received support in both reused panels. At dose 0.5, the
both-head actor-only changes were −1.874785 and −1.457389; the earlier-only
changes were −0.000167 and +0.000472. The route effect was nearly all in
zero-based head1, the second head of layer 0: its actor-only values were
−1.876189 and −1.459685, while head0 values were +0.000715 and +0.006495.
The all-query values closely match actor-only. This does not give head1 an
exclusive semantic label or establish per-edge importance. The corresponding
original-parent contrasts were −1.643392 [−1.751028, −1.534047] and −1.248418
[−1.546950, −0.894824], so this routing sensitivity predates the cooldown.

## Prediction 2: restoring one scalar

The endpoint is the positive-row mean change in raw 28-way probability TV
distance to baseline after minus before restoration, then U65 minus U193. The
physical-pair bootstrap is retained. This is in probability units. The negative
prediction did **not** hold across both reused panels:

| Existing panel | Positive rows | U65-minus-U193 distance-change contrast | Descriptive 95% interval |
|---|---:|---:|---:|
| Development | 12 | +0.012365 | [−0.035934, +0.085561] |
| Confirmation | 12 | −0.031431 | [−0.055560, −0.011284] |

Development has the opposite mean sign and an interval spanning zero;
confirmation alone does not establish replication. On positive development
rows, raw-TV distance was 0.056994 under the cut, 0.069180 after U65
restoration and 0.056816 after U193 restoration. On confirmation it was
0.047918, 0.016185 and 0.047616, respectively. Legal-TV distances were
0.038493, 0.057954 and 0.038367 in development; 0.014023, 0.012158 and
0.013721 in confirmation. Centered-logit L2 distance increased after U65
restoration in both panels: 4.6272 to 7.0115 in development, and 4.2465 to
5.3273 in confirmation. Output proximity depends on the metric; raw and legal
probability distributions and centered logits are all retained.

The negative-row raw-TV contrasts were −0.002692 [−0.007890, +0.000527] in
development and −0.000255 [−0.000844, +0.000234] in confirmation. Each panel
contains forced one-legal-action rows (4 per side in development; 5 per side
in confirmation), where legal-TV is necessarily zero although raw 28-way
probabilities may change. The positive-side panels likewise have 8/4 and 7/5
discretionary/forced rows, respectively.

The scalar patches were not magnitude- or direction-matched. Mean absolute
post-GELU replacement was 1.7451/1.5209 for U65 versus 0.01805/0.01729 for
U193 in development/confirmation. Their projection-column norms were 0.541538
and 0.433596 with different directions; implied residual-write displacement
norms were 0.9450/0.8236 for U65 versus 0.00783/0.00750 for U193. Therefore
the comparison is only about these exact scalar restorations under the cut,
not intrinsic unit importance or mediation. Failure to restore output proximity
does not show that U65 is mixed or lacks a coherent role.

## Audit and public evidence

The final instrument completed 48 rows across two models with 69 saved cases
per row (6,624 total). All numeric saved native activations, logits and
source-query profiles matched an independent functional forward bit-identically
(maximum error 0); exact zero/empty/no-cut shams, restoration-site checks,
cleanup and unchanged weights passed. Reference actions remain distinct from
model probabilities. Full row-level records, checksums, route/restoration
series, support counts and audit receipts are in the immutable evidence archive:
[verified evidence at revision `a6d9cc4f06b196b5a2f0ca495e52b27502313a33`](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/a6d9cc4f06b196b5a2f0ca495e52b27502313a33/lem/gus-route-rescue-2026-10-20/evidence), 8 files, 117 members, 36,985,839 bytes, freshly anonymous-verified. The frozen protocol and registration were published and verified before trained-response measurement at [registration revision `51c8e51d97929cf56ce56293b63d998ad4a52753`](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/51c8e51d97929cf56ce56293b63d998ad4a52753/lem/gus-route-rescue-2026-10-20/registration).

The initial unpublished preparation's recovery guard could have overwritten
the first failure receipt. That preparation was preserved under
`scratch/lem/gus-route-rescue-v1/preparation-v1/`; the guard was repaired,
synthetic checks and preparation were repeated, and the new registration was
reviewed before publication. The superseded hash was never publicly registered
and no trained response used it. No measured failure occurred under the final
instrument. This preserved setup failure is separate from the completed
measurement result.

## Interpretation boundary

The evidence supports a local layer-0 final-query route into the U65
preactivation on these reused contexts, concentrated in the second head under
the registered intervention. It does not identify what the head writes, an
abstract blank-suit feature, U65's exclusive meaning, or a unit-specific route
to the final output. Both predictions remain reported, including prediction
2's panel disagreement. There is no new semantic confirmation, global
cleanliness claim, training gain, or playing/winning result. The interpretation
ladder reaches native tools and a local route description only; systematic
formation analysis remains an unselected next candidate.

The [same First Look Site report](https://gus-first-look.psychometrix.chatgpt.site/#rescue)
now includes interactive route and restoration panels. All 24 selector combinations
and the 390-pixel layout passed local checks; owner-private native deployment
succeeded from source `c6f863a62e92f4cc26fde3540706f3f395fdbf0b`. See also
[[gus-book-silas-tools]], [[gus-native-cooldown-2026-10-19]], and
[[gus-first-look-2026-10-19]].
