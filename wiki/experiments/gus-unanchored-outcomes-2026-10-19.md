---
title: "Gus: outcome learning without teacher anchoring"
kind: experiment
first_seen: 2026-10-19
last_updated: 2026-10-19
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

Does removing the teacher-anchoring loss package improve verified full-hand
success over anchored GRPO and imitation alone? Jason's steering in
[[gus-success-probability-planning]] prioritizes winning and expects behavior to
drift after pretraining. [[gus-ppo-value-2026-10-19]] established neither stronger
play nor useful online critic forecasts. This comparison keeps the existing
GRPO estimator, separating the teacher-constraint question from critic repair.

## Registered comparison

All arms initialize the same 185,652-parameter [[gus-tiny-gpt]] splat actor.

| Arm | Verified-outcome gradient | Imitation weight | Frozen-parent KL weight |
|---|---|---:|---:|
| Unanchored | GRPO | 0 | 0 |
| Anchored | GRPO | 1 | .02 |
| Rehearsal | None | 1 | None |

The changed factor is a **package**, not the individual effect of either loss.
All arms select by greedy then sampled development success, including
initialization and keeping earliest ties. Original tile NLL and splat CE are
recorded without an eligibility veto. This common selection rule differs from
prior studies; confirmation cannot revise the selections.

Original own-seven/public-history tokens, bid 30/pip, fixed G0 partner and pinned
Zeb opponents stay unchanged. Each arm receives the same 256 fresh physical
roots, eight complete 28-play draws per root, 32 rounds and 64 actor updates.
Learning rate3e-5, eight roots/batch, two inner updates, batch112, gradient clip1,
AdamW matrix decay.1, full seven-turn credit and evaluation every8 rounds remain.
Group-standardized terminal returns supply credit. No learned critic is fitted.

The clipped probability ratio uses the policy that generated the current batch;
it remains at .2 even when the frozen-parent penalty is zero. Clipping controls
the surrogate objective, not a guaranteed hard bound on policy change.
Sources: [PPO](https://arxiv.org/abs/1707.06347),
[GRPO](https://arxiv.org/abs/2402.03300). Rehearsal/fixed-parent penalties are the
experimental controls, not universal requirements of policy-gradient learning.

Zero-weight replay still consumes identical diagnostic draws under no-grad.
Unused splat-head parameters receive neither gradients nor AdamW decay. Their
outputs may change with the shared backbone; this does not train calibrated
current-Gus histograms. Hidden simulator state and outcome sidecars never enter
actor inputs. No player search, new opponent, or Demon label is introduced.

## Selection, confirmation and budget

The common 128-root development panel selects each arm. The registered method
criterion is unanchored greedy success at least 2/128 above both controls.
After all selections freeze, 512 untouched roots evaluate the parent and three
selected labels, greedy and one coupled sampled rollout. Identical exact
inference functions may share records under the pinned runtime and stream.

Promotion requires unanchored greedy gains of at least 2pp over parent and
1pp over both controls, each paired 95% lower bound above zero. Imitation losses
are not promotion gates. Intervals resample physical roots:20,000 draws,
seed91942. Sampled results are secondary; one seed/recipe does not rank methods
generally or establish unrestricted auction strength.

Each arm has at most 16 cooperative 45-second segments with 90-second outer caps,
max 2 live arms. Confirmation has at most 12 such segments/state, max 3 live.
Identity, nonfinite, replay and budget failures halt with recovery preserved.
No automatic budget increase. Initial/per-round/final checkpoints preserve
AdamW and RNG. CPU signal/crash recovery is tested; MPS bitwise identity is not
promised. Frozen CPU evaluation retains strict lazy-record byte checks.
Final native captures preserve the existing 11,116-row panel for `clean-neuron`;
unit explanations remain separate work. Registered fixed G0/Zeb/fenced-Demon
smokes are one duplicate pair per distinct selected function and opponent,
at most 60 seconds each, and do not establish playing strength.

## Public registration and capability refresh

[Preregistration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/0618d5979298b8a044a5fa98bca39969166ab159/lem/gus-unanchored-outcomes-2026-10-19/registration)
precedes training: 12 files, 11,785,970 bytes, all freshly anonymously downloaded
and SHA256-verified. Registration SHA256:
`aaefde365f4e27667db96689fb2cd30084ab14d0bcf1ae3fb095765cf3c8efbb`.
Source is `b9a44691`; local recovery root is `scratch/lem/gus-unanchored-v1`.
All 896 reserved roots exclude 603,260 prior physical deals and become future
training exclusions. `run.py` owns bounded child segments; inspect its process
and completion receipts before resuming. The completed results are below.

The refresh in [[gus-book-silas-tools]] finds no new Lem/Book replies; requests
remain pending. Silas PR #166 remains open, and existing main outcome verification
suffices. Native transformer captures exist; the causal audit bridge remains
separate work. These boundaries do not change the comparison or its controls.

## Completed training and evaluation

Development method criterion: **False**.
Fresh promotion criterion: **False**. No new incumbent
is promoted. All selections froze before confirmation; original imitation
losses were diagnostics, never an eligibility or promotion veto.

| Player | Selected round | Development wins /128 | Fresh wins /512 | Fresh sampled /512 |
|---|---:|---:|---:|---:|
| initial_parent | 0 | 56 | 186 | 174 |
| unanchored | 24 | 58 | 182 | 184 |
| anchored | 16 | 58 | 185 | 182 |
| rehearsal | 8 | 57 | 186 | 174 |

- Unanchored versus initial_parent: **-0.781pp**, paired 95% **[-2.344, +0.781]**; 7 gained wins and 11 lost wins.
- Unanchored versus anchored: **-0.586pp**, paired 95% **[-1.758, +0.586]**; 3 gained wins and 6 lost wins.
- Unanchored versus rehearsal: **-0.781pp**, paired 95% **[-2.148, +0.586]**; 5 gained wins and 9 lost wins.

Intervals resample 512 physical roots. These results do not establish stronger
play from removing teacher anchoring in this short recipe; they do not show that
imitation is necessary or that a student cannot improve beyond its teacher.
The registered primary endpoint is greedy hand success. Sampled hand success
is secondary; prediction loss does not substitute for either outcome measure.

## A secondary sampled-policy signal

The sampled policy we optimize differs from greedy deployment. Unanchored
sampled play gains 10/512 wins over the sampled parent: **+1.953pp**, paired 95%
**[+0.195,+3.711]**. Anchored sampled play gains 8/512: **+1.562pp**
**[+0.195,+3.125]**. This is evidence worth independently confirming for the
sampled policies under this fixed field, not a substitute primary endpoint.
Unanchored versus anchored sampled play is only +0.391pp
**[−1.172,+1.953]**; removal of the anchor package is not established as better.
Unanchored versus sampled rehearsal is +1.953pp **[0.000,+3.906]**.

Only one coupled sampled continuation per fresh root was registered. Do not
reselect checkpoints or switch the promotion endpoint after seeing this result.
The strongest greedy baseline still wins 186 hands versus unanchored sampled 184;
there is no demonstrated improvement over the deployed parent here. The useful
next question is whether the sampled-policy signal repeats under independently
seeded continuations, and how decoding relates to the objective actually trained.
That requires its own registration after Jason's discussion window. These are
new confirmation roots; absolute rates should not be compared to older panels.

## Drift and learning signal

The old retention rule would change none of the three selected checkpoints.
The counterfactual is descriptive only; no alternate candidate was selected
using confirmation. See `training-diagnostics.json` for all development points,
losses, reference drift, entropy and clipping. The unanchored outcome head's
parameters stayed exactly unchanged at every formation boundary; its outputs
can still drift through the shared backbone. Neither this drift nor the old
teacher rehearsal calibrates a current-Gus outcome histogram.

In the unanchored arm, 71.875% of root groups have eight identical win/loss
results, versus 71.484% for anchored and 70.703% for rehearsal. These groups
supply no within-root GRPO reward contrast. This does not prove that their
positions are unavoidably won or lost.

Every root receives eight complete 28-play hands, with seven learner decisions
and verified terminal make/set. Group-normalized return credits the sampled
learner trajectory; it does not isolate the causal value of each move. This
comparison does not implement action-conditioned branching at public prefixes
or full-match reward. Those remain separate proposed collectors/objectives,
with the clean input and fixed-opponent controls preserved.

## One actual input and update

The first teaching example is the opening prefix: bid 30, you bid, fives trump;
your tiles 41, 44, 51, 53, 63, 65, 66; you lead. Three of its eight sampled complete
hands win. In the preserved winning sample Gus leads 51 and receives normalized
advantage +1.291. Across the two whole-batch updates, that move's probability
changes 24.380% to 24.351%; 65 changes 26.259% to 27.109%. A positive advantage does
not guarantee that every winning sample's probability rises after the shared
parameter update: other decisions and hands contribute gradients too. This is
a literal before/after example, not causal proof that either opening is better.
Tokens, legal probabilities and all raw trajectories remain in the archive.

## Audit, specimens and cost

Independent checks replay 6,144 training hands and
4,096 confirmation hands, verify
43,008 reference-policy
probabilities, reconstruct all selections, regenerate 896 reserved roots and
check fixed seats until learner divergence. 16
lazy CPU units replay byte-identically in reverse order. CPU/MPS probability
checks retain the preregistered 5e-5 tolerance. All 96 formation checkpoints and
full optimizer/RNG recovery remain archived. Three final-round native captures
contain 11,116 rows each for `clean-neuron`; these are not selected checkpoints
or causal explanations of units.

Selected research specimens, with exact tensor and both-head CPU logit export
parity plus fresh anonymous hashes:
[unanchored](https://huggingface.co/jasonyandell/gus/tree/f84d07124159fdbe27e272a673b21bc940a526a3), [anchored](https://huggingface.co/jasonyandell/gus/tree/c98fa2136dea0a8aafc309a1ed78c019d56ce2c6), [rehearsal](https://huggingface.co/jasonyandell/gus/tree/50847701d872f35af52eca54eac90c5609416c32).
The main Hub incumbent remains unchanged. The registered integration smoke
completes 9 duplicate faceoffs, 18 games,
198 hands and 5544 plays, all independently replayed.
Maximum faceoff wall time 5.45s under 60-second caps.
[Canonical tournament records](https://huggingface.co/datasets/jasonyandell/texas-42-tournament/tree/b2d54393002a549fb1407380852706aaca3a1ddc) are published and verified:15 immutable
records,12 new files in one batch; every file freshly downloaded anonymously
and SHA256-checked. The original hourly-quota failure remains in the research
archive. No game was rerun. These are integration records, not strength
confirmation.
Whole-arm host instructions, CPU and wall receipts are archived; host
instructions do not count MPS GPU work and wall time is descriptive under load.

## Public evidence and completed delivery

[Verified archive](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/c0fb9da4fd8b493c7fa5e58228d097da98775178/lem/gus-unanchored-outcomes-2026-10-19): 18 files,
557,814,648 bytes, 4,953 bundle members. Every fresh
anonymous file download and every archive member matches its SHA256 manifest.
The archive includes scripts, inputs, raw trajectories, formation/recovery,
native traces, literal worked updates and restore paths. Source remains
`b9a44691`; recovery root remains `scratch/lem/gus-unanchored-v1`.

The completion capability refresh is recorded in [[gus-book-silas-tools]].
Canonical delivery is resolved. The completed comparison retains its original
endpoints and selections. A new sampled-policy confirmation requires its own
registration and fresh roots; its proposal is in [[gus-success-probability-planning]].
