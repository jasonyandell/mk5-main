---
title: "Gus: exact endgames under the fixed G0/Zeb field"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and scope

Can exact short-endgame success feedback under the full-hand stage's fixed
G0/Zeb policies improve [[gus-tiny-gpt]] beyond matched prediction rehearsal,
and does any gain transfer to fresh full-hand outcomes? This follows the
completed update-scale comparison in [[gus-update-scale-2026-10-18]]. The
pretrained joint model, clean inputs and nonlearner policy weights/budgets stay
fixed. The earlier lowest/highest/uniform verifier keeps its separate identity.

The root belief is uniform over all legal-consistent remaining deals. It omits
pre-root behavior likelihood, so exact rational values certify this finite
approximation, not the true posterior or unknown players. Each root action is
held across every world; the learner's later move must be forced. Other players
respond to each continuation through their own information-limited policy.
Gus itself remains a direct neural policy. [[gus-success-probability-planning]]
records the expected-success objective and its textbook justification.

## Implemented construction and recovery

`lem.observer.rlvr.fixed_field` produces exact rational certificates plus every
terminal continuation, independently checked by the rules replay. Its identity
pins G0/Zeb weights, deterministic policy specification, source/native bytes,
Torch/platform and individual CPU float32 inference. Counterfactual hidden
hands stay in the witness, never the neural input.

`lem.observer.rlvr.fixed_data --plan PLAN --out DIR --seconds 45` constructs a
curriculum from predeclared, disjoint physical root pools. Histories use one
sampled pretrained learner, its fixed G0 partner and pinned Zeb opponents under
the existing bid30 auction. Each root supplies at most one unsettled two-legal
choice during the penultimate trick. Split quotas choose the first eligible
deals, without filtering by wins, mistakes or label gaps. Screening cost and
all potentially used source deals remain recorded. Collection batch size is
pinned; compute slicing does not change it.

The existing immutable `Extension` commits a complete collection/verification
unit. Signals stop between units; crashes redo only an uncommitted unit.
Completed records are reused, and `--recheck` recomputes their canonical bytes.
The loader checks clean prefixes, rational supports and every selected row
against its hashed witness record. Completed resumes preserve dataset bytes.
Source, weights, quotas or seating changes require a new plan/output identity.

The existing exact-success trainer now also supports `--mode rehearsal` as a
matched supervised-only control. It uses the same question/rehearsal streams
and optimizer counts, with zero policy-reward and reference-KL gradients.
Optimizer, all RNG, formation, initial/best/final and original-task retention
remain saved. Training resume pins the complete observer Python source tree.

## Experiment state

Source `d012a0c0` is merged; the repository gate passed 143 Python tests,
wiki, conflict and license checks. The [public preregistration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/a19cc769d66fc8e38228d07d57eb4045b4f8cf8f/lem/gus-fixed-field-curriculum-2026-10-18)
preserves the full root plan, exclusions and protocol. All eight files,
16,226,353 bytes, passed fresh anonymous SHA256 checks. Registration SHA256:
`c4b166620125698fb7050285fbffa382c142450937c6bebefd0ec9b58444f043`.

Construction targets 512 train, 128 development and 256 held-out endgames from
separate pools capped at 4,096/1,024/2,048 source deals. Another 512 roots are
reserved for fresh full-hand confirmation. All 7,680 roots were drawn without overlap with
591,968 prior physical deals/schedules before results and remain future training
exclusions even when screening never uses them. Units contain 16 source hands;
45s cooperative segments have 90s outer caps and a maximum of 32 segments.

| Arm | Seeds | Objective | Rounds / AdamW steps per seed |
|---|---|---|---|
| Success |42, 43, 44|Exact expected success + parent KL + original-task rehearsal|256 / 512|
| Rehearsal |42, 43, 44|Original tile/splat supervised losses only|256 / 512|

All start from the same 185,652-parameter splat parent with fresh AdamW, no
growth, learning rate 1e-5, question batch 64, replay batch 32, KL .02 and
rehearsal weight 1. Paired seeds share question and rehearsal streams. Original
loss bounds are +.02 each. Checkpoint selection includes initialization and minimizes exact dev
regret within those bounds, then favors accuracy and the earliest round.
Each training run permits four 45s segments under 90s outer caps; at most two
run concurrently.

The method criterion is mean development regret improvement of at least .5pp
over rehearsal, with improvement in at least two of three seeds. Seed42 is
always the deployment representative; it is not chosen from results. All six
selections freeze before held-out metrics. Exact-model confirmation reports all
seeds and paired physical-deal intervals. The representative material criterion
is .5pp regret reduction and a positive paired 95% lower bound versus both
parent and rehearsal, with retention.

The separate 512-hand confirmation compares parent and the two seed42 selections
with greedy and sampled deployment. Promotion requires greedy gains of at least
2pp over parent and 1pp over rehearsal, with positive paired 95% lower bounds
and retention. Brief fixed G0/Zeb/Demon duplicate pairs are integration checks,
never a promotion basis. The registered comparison is complete. The exact-model criterion passes;
fresh full-hand confirmation does not support promotion.

The series remains tracked by [issue #223](https://github.com/jasonyandell/mk5-main/issues/223).
Interpretation specimens follow `clean-neuron`; unit claims need separate causal
evidence. The `gus-training` skill carries the construction and recovery route.


## Results

The selected dataset contains 512 training, 128 development and 256 held-out
questions from 2,912 screened hands. The 182 immutable units contain 901 eligible
questions (five exceed split quotas), 15,786 root worlds and 31,572 complete
counterfactual hands. Independent replay verified every screened/counterfactual
hand, recounted each rational value and checked complete world/action coverage.
Three actual collection units recomputed in reverse split order matched bytes.
Dataset SHA256: `6d01356e2f58c6d9c44d9d3eb9aadd79d7ce7e0b1420e4aa`.

All six runs finished 256 rounds/512 steps in one segment each. Mean development
regret reduction versus rehearsal was 1.144 percentage points; all three seeds
improved, passing the registered method criterion. These development selections
were frozen before held-out evaluation:

| Seed | Success selected round | Rehearsal selected round | Development regret: success / rehearsal | Held-out regret: success / rehearsal |
|---|---:|---:|---:|---:|
|42|256|192|3.861% / 4.967%|3.534% / 4.072%|
|43|224|96|4.355% / 5.575%|3.664% / 4.055%|
|44|256|192|4.382% / 5.488%|3.534% / 3.936%|

On 256 held-out endgames the parent has 4.059% regret and uniform-legal play
4.445%. Seed42 improves over the parent by **0.525pp [0.117, 1.061]** and over
rehearsal by **0.538pp [0.078, 1.098]**; these are paired 95% physical-deal
bootstrap intervals. The registered finite-model material criterion passes.
Best-set accuracy is 85.16% versus parent 82.81% and uniform 83.79%; ties count
as correct. Accuracy alone hides how consequential the mistakes are. Training
seeds share the same test deals; they are not pooled as independent evidence.
All seeds retain the original prediction tasks. Seed42 tile NLL improves from
1.32996 to 1.30789 and splat CE from 2.34933 to 2.31667.

Fresh full-hand confirmation does **not** establish stronger play:

| Predeclared representative | Greedy wins / 512 | Sampled wins / 512 |
|---|---:|---:|
|Original splat parent|206 (40.23%)|195 (38.09%)|
|Exact success, seed42 selected|204 (39.84%)|194 (37.89%)|
|Rehearsal, seed42 selected|204 (39.84%)|194 (37.89%)|

Greedy success-minus-parent is -0.391pp [-2.539, 1.563]; success-minus-rehearsal
is 0.000pp [-1.367, 1.367]. The same totals do not mean identical plays: success
wins seven paired hands that rehearsal loses and loses seven that rehearsal
wins. No reselection or incumbent replacement follows confirmation. These are
sampled physical outcomes under fixed policies, not exact success probabilities.

All 3,072 confirmation trajectories independently reconstruct literal inputs,
legal masks, actions, terminal masks and rewards. Twelve actual units recomputed
in reverse order match canonical bytes. Frozen other seats agree until the
learner first changes history in all 2,048 paired comparisons. Three inference
states have separate immutable stores; no model-label alias inflates the count.
Six **final** checkpoints also preserve native representations on all 11,116
original frozen rows. Final traces are distinct from selected-model confirmation;
`clean-neuron` interpretation still requires separate causal evidence.

Nine duplicate faceoffs produced 18 games, 198 hands and 5,544 independently
replayed plays against fixed G0, native Zeb and fenced Demon. The longest
faceoff was 5.20s, below the 60s cap. Each model/opponent cell has only one pair;
these are adapter checks, not a strength ranking. Canonical tournament upload
is pending the service's explicit commit-quota reset; the full records and audit
are preserved in the research archive. No games need rerunning.

## A literal learning signal

The first training question with unequal values has legal tiles six-two and
six-six, worth exactly 1/3 and 2/3 success under the declared three-world root
belief. The collected parent's legal probabilities are 96.22% and 3.78%.
The reward-only logit derivatives are +0.01211 and -0.01211 before batch
averaging: gradient descent reduces the six-two logit and increases six-six.
This is the analytic expected-success signal for the recorded distribution,
not a claim about a single AdamW update. Actual updates also include KL,
rehearsal, clipping, optimizer moments and decay. The archive's
`worked-success-gradient.json` preserves the literal input, history, certificate
and derivative. No hidden world or outcome is added to the input.

## Cost and evidence

Construction used 1.295e13 retired CPU instructions across nine segments,
414.29s summed wall and 718.67s CPU. Each training run used 1.049–1.061e11 host
instructions, 14.19–15.54s wall and 9.31–9.65s CPU. MPS work is not included in
host instructions. Three full-hand confirmations used 1.292e13 instructions,
644.71s summed wall and 644.15s CPU across fifteen segments. Wall sums are not
campaign elapsed time; timings describe this execution, not policy budgets.

The [complete research evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/0d326ef3037013006880ed79675f0510e5d2ddf1/lem/gus-fixed-field-curriculum-2026-10-18)
contains inputs, exclusions, all immutable units/witnesses, full formation and
optimizer/RNG recovery, exact/full confirmation, native traces, games, scripts,
costs and audit receipts. All 18 uploaded files (838,526,594 bytes) and all
3,543 bundle members passed fresh anonymous SHA256 verification. Source is
pinned to `d012a0c0`.
[Seed42 success model](https://huggingface.co/jasonyandell/gus/tree/ce9347273391b29ac7f94aefb9ba128538fe1e78)
and [matched rehearsal model](https://huggingface.co/jasonyandell/gus/tree/4da0d81dbbe89fd4cf780e87cc71b684fb483aab)
are research branches; both heads pass exact export parity and all release
files passed fresh anonymous hashes. The main Hub incumbent is unchanged.
[Seed42 W&B run](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/ef4ae851)
is complete; the archive retains all six run URLs.

## Next question

Can the successful endgame curriculum serve as a better initialization for
full-hand outcome post-training? A bounded staged-training comparison can
hold the established GRPO recipe, original rehearsal and G0/Zeb field fixed,
and vary only its starting checkpoint: original splat parent, endgame success,
and matched rehearsal. To isolate initialization, all arms must use the same
original frozen KL reference and common retention bounds; loading the initial
policy must be separate from loading that reference. Reserve fresh confirmation
before training. [[gus-staged-outcomes-2026-10-18]] implements and registers
that comparison, with a supervised-only continuation control. The current
endgame gain does not already prove transfer. External search,
future-splat targets and evolving opponents remain deferred.
