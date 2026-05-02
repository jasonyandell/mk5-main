---
title: w42 Dataset Manifest
kind: experiment
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Summary

[[w42]] uses a reusable dataset manifest before any generated corpus, probe, or
report is treated as durable. The manifest records where the data came from, how
it was generated, which split policy applies, which labels and tags are available,
which leakage paths are excluded, and how later generated data versions supersede
or extend earlier ones.

This is a policy/specification page only. It does not generate a corpus, upload an
artifact, train a model, or move any claim in the [[winning42-strategy-measurement]]
claim ledger.

## Manifest Location

Every durable w42 dataset or promoted scratch corpus should have a sibling
manifest named `manifest.json` or `manifest.yaml`. For scratch work, place it next
to the data under `scratch/w42/<dataset-version>/`. For promoted local corpora,
place it beside the data path chosen by the promotion bead. If a dataset is later
published, the same manifest content should travel with the HF dataset card.

The manifest is the source of truth. Experiment prose may summarize it, but
training and evaluation scripts should read or print the manifest before consuming
the corpus.

## Required Fields

```yaml
schema_version: w42.dataset_manifest.v1
dataset_id: w42-<short-purpose>-<YYYYMMDD>
dataset_version: v0
created_at: <ISO-8601 timestamp>
repo_commit: <git commit used to generate or assemble the data>
owner_bead: <bead id>
status: scratch | promoted | retired | superseded

source_corpora:
  - path: <repo-relative or absolute external path>
    role: train | val | test | eval-only | hypothesis-source | trace-source
    exists_at_manifest_time: true | false
    source_kind: gus-joint-world | forge-eq | forge-analysis | winning42-wiki | burl-trace | other
    provenance: <how this file/path was produced or why it is trusted>
    split_policy: <named split policy below>

generation:
  command: <exact command, or "not run">
  cwd: <working directory>
  environment:
    device: cuda | mps | cpu | not applicable
    wandb: not applicable | <entity/project/run-url>
    huggingface: not applicable | <org/repo/path-or-revision>
  inputs:
    checkpoint: <path or not applicable>
    seed_ranges: [<ranges>]
    declarations: <decl ids or policy>
    sampling: <fixed/adaptive/enumerated/posterior settings>

splits:
  policy: w42-seed-bucket-v1
  train: <seed/declaration/file rule>
  val: <seed/declaration/file rule>
  test: <seed/declaration/file rule>
  eval: <seed/declaration/file rule>
  random_seeds:
    data_generation: <int or not applicable>
    dataset_shuffle: <int or not applicable>
    train_loader: <int or not applicable>
    eval_sampling: <int or not applicable>

leakage_exclusions:
  - <path/range/policy excluded from training>

labels_available:
  - <label name>

tags_available:
  cheap_strategy_tags: [<tag names>]
  chapter_derived_tags: [<tag names>]
  analysis_buckets: [<bucket names>]

versioning:
  parent_dataset_id: <id or null>
  parent_dataset_version: <version or null>
  change_type: initial | appended-seeds | regenerated | relabeled | retagged | filtered | split-fix
  compatibility: compatible | incompatible
  supersedes: [<dataset ids/versions>]
  notes: <short caveat>

claim_ledger_impact: no claim-ledger change
```

## Source Corpus Policy

w42 may reuse known project data formats, but each source path must be named
explicitly in the manifest. Expected source families:

- [[gus]] joint-world corpora, such as `gus/data/corpus_train_100.pt`,
  `gus/data/corpus_train_chunk_*-*.pt`, `gus/data/corpus_v2_train_*_d0-9.pt`,
  `gus/data/corpus_eval_20.pt`, and `gus/data/corpus_v2_eval.pt`.
- [[forge]] E[Q] or oracle data, including `data/eq-games/{train,val,test}/`
  outputs from `forge.cli.generate_eq_continuous` and external
  `forge-analysis` shard locations such as `/mnt/d/shards-standard/`.
- Winning 42 hypothesis sources: `wiki/experiments/winning42-ch*.md`,
  `wiki/experiments/winning42-strategy-measurement.md`, and scratch harvest
  notes under `scratch/winning42/`.
- [[burl]] traces only when the question is reasoning, explanation, tool use, or
  post-commit text. Burl held-out eval traces must not silently become w42
  training data.

If a listed source path is absent in the current checkout, set
`exists_at_manifest_time: false` and keep the path as provenance only. Do not
infer row counts, seeds, or labels from prose when the file is absent.

## Generation Provenance

Generated w42 corpora should record the exact command and the generating commit.
Known command shapes to preserve:

```bash
python -u -m forge.eq.generate \
  --start-seed 0 --n-games 100 \
  --adaptive \
  --min-samples 100 --max-samples 50000 \
  --sem-threshold 0.5 --batch-size 200 \
  --save-joint-worlds \
  -o gus/data/corpus_100.pt
```

```bash
python -m forge.cli.generate_eq_continuous \
  --checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt \
  --start-seed <seed> \
  --limit <games> \
  --adaptive \
  --output-dir data/eq-games
```

If a w42 run assembles existing corpora rather than generating fresh examples,
the manifest should still record the assembly command, glob expansion, filters,
and row counts. If no command ran, write `command: not run` and explain the
source as a policy or provenance-only entry.

## Split Policy

Default split policy: `w42-seed-bucket-v1`.

| Seed rule | Role |
|---|---|
| `0 <= seed <= 899999` and `seed % 1000 < 900` | train |
| `0 <= seed <= 899999` and `900 <= seed % 1000 < 950` | val |
| `0 <= seed <= 899999` and `950 <= seed % 1000 < 1000` | test |
| `900000 <= seed <= 909999` | eval-only; never train, never validate |

This follows the repo's [[eval-seed-holdout]] boundary and forge's deterministic
`seed % 1000` train/val/test routing. w42 reports may sample from test or eval
only after model selection is frozen. When the same base seed is expanded across
multiple declarations, every `(seed, decl_id)` row inherits the seed's split.

Randomization must be deterministic and named. Recommended defaults:

- data generation seed: the deal seed in the source corpus.
- dataset shuffle seed: `42` unless a bead declares another value.
- train loader seed: the training script seed, recorded explicitly.
- E[Q] / world sampling seed: the script seed for sampled-world selection,
  recorded explicitly; common existing eval loaders use `42`.

## Leakage Exclusions

Training manifests must exclude:

- all seeds in `900000-909999`;
- any `gus/data/*eval*.pt` corpus unless the dataset role is `eval-only`;
- `lem/data/narrations_eval.jsonl` and any artifact produced with
  `--allow-eval-seeds`;
- Burl held-out eval traces and any scratch corpus whose manifest says it came
  from held-out eval, including directories marked `FROM_HELD_OUT_EVAL_DO_NOT_TRAIN`;
- private, illegal, or partner-only table-talk information from
  [[winning42-ch11-table-talk]];
- perfect-information labels as model inputs unless the experiment is explicitly
  an oracle-analysis report rather than a deployable public-state model.

Oracle outputs may be labels. They must not be smuggled into public-state feature
columns intended for inference-time w42 models.

## Labels Available

Known reusable labels include:

- `e_q` / `e_q_mean`: marginal E[Q] per legal action;
- `e_q_var`, `e_q_pdf`, `p_make`: uncertainty and outcome-distribution labels;
- `action_taken`: the oracle/E[Q] policy action used in the generated game;
- `legal_mask`: action legality;
- `world_hands` and `q_per_world`: joint-world tensor labels for
  world-conditioned analysis or training;
- `oracle_softmax_per_seat`, `legal_mask_per_seat`, and `voids_per_seat` when
  schema-v2 Gus corpora are used;
- regret labels derived as `oracle_best_eq - chosen_action_eq`;
- belief labels from true deal ownership, with public-state masking;
- claim-ledger labels from Winning 42 pages only as hypothesis/status metadata,
  not as empirical support.

## Tags Available

Cheap public-state tags already have evidence from
[[gus-strategy-tags-probe]] and may be reused as features or report columns:

- global public-state features: declaration, phase, trick position, hand shape,
  trump/called-suit/double/count summaries, legal-action summaries, live count,
  void summaries, visible pip coverage, trick pressure, current-winner relation,
  and unseen count estimates;
- action-local features: legal/called/trump/double/count identity, rank and pip
  identity, follows/beats-current-trick flags, point-dump and trump-in flags,
  higher-live tiles, suit pressure, pip coverage, count risk, same-pip-double
  protection, and count donation to partner or opponent.

Chapter-derived tags should be grouped by concept bucket before they are used:

- bidding risk: `candidate_trump_shape`, `off_count_liability`,
  `off_protected_by_double`, `duplicate_count_exposure`,
  `natural_bid_bucket`, `eighty_four_candidate`;
- play sequencing: `pulls_trump`, `saves_reentry`, `plays_off`,
  `throws_count`, `donates_to_partner`, `pounces_count`, `creates_void`,
  `breaks_pair`, `preserves_last_trick_weapon`, `promoted_tile`;
- belief and discipline: `void_evidence_strength`,
  `bid_implied_suit_strength`, `failed_bid_signal`,
  `outstanding_trump_danger`, `legal_info_boundary`,
  `partner_legibility`, `style_prior_bucket`;
- analysis buckets: `ruleset_variant`, `score_objective_mode`,
  `partner_fit_bucket`, `population_style_bucket`,
  `odds_threshold_bucket`, and `claim_ledger_status`.

Tags are feature candidates and report slices. A tag's existence is not evidence
that the corresponding Winning 42 claim is supported.

## Versioning

Future generated data should be versioned by dataset id and manifest version, not
only by directory timestamp. A new version is required when:

- seed ranges or declaration coverage change;
- generation code, oracle checkpoint, sampling policy, or schema version changes;
- filters, labels, or tag definitions change;
- a leakage bug is fixed;
- a scratch dataset is promoted or published.

Compatible appends keep the same `dataset_id` and increment `dataset_version`.
Regenerations, split fixes, schema changes, or retagging that changes semantics
must record `compatibility: incompatible` and list what they supersede.

## Lab Coordination

The lab-infrastructure bead for W&B/HF conventions is still open, so this page
does not reserve a W&B entity/project, HF org/repo, or artifact naming scheme for
w42. Manifests should still include W&B/HF fields and set them to
`not applicable` until that convention exists or a run actually uploads artifacts.

## Commands / Checks Run

No data generation, training, upload, or artifact creation ran for this page.

Exact checks run:

```bash
git status --short --branch
bd show t42-csw6.3 --json
bd show t42-csw6.1 --json
bd show t42-csw6.2 --json
sed -n '1,240p' wiki/AGENTS.md
sed -n '1,260p' wiki/entities/w42.md
sed -n '1,260p' wiki/experiments/winning42-strategy-measurement.md
sed -n '1,280p' wiki/experiments/gus-strategy-tags-probe.md
sed -n '1,300p' wiki/entities/forge-analysis.md
sed -n '1,300p' wiki/entities/forge.md
sed -n '1,700p' wiki/entities/gus.md
sed -n '1,120p' wiki/decisions/eval-seed-holdout.md
sed -n '88,110p' forge/ml/tokenize.py
sed -n '390,410p' forge/ORIENTATION.md
sed -n '856,872p' forge/ORIENTATION.md
sed -n '1,140p' gus/BUILD_PLAN.md
sed -n '1,160p' gus/GEN_FLEET.md
find gus/data -maxdepth 1 -type f
rg -n "corpus_train|corpus_v2|corpus_eval|train_chunk|eval_20|seed_900000|JointWorld" gus -g '*.py' -g '*.md'
rg -n "w42|W&B|wandb|HuggingFace|HF|artifact|claim-ledger|claim ledger|not applicable|manifest|split policy|seed" wiki/entities/w42.md wiki/experiments wiki/decisions wiki/topics -g '*.md'
bd close t42-csw6.3 --reason "w42 reusable dataset manifest and split policy documented; W&B links: not applicable; HF links: not applicable; claim-ledger impact: no claim-ledger change"
bd show t42-csw6.3 --json
bd export --no-memories -o /tmp/w42-issues-export.jsonl
```

`find gus/data -maxdepth 1 -type f` reported that `gus/data` is absent in this
checkout, so this page treats Gus corpus paths as referenced provenance rather
than locally verified artifacts.

The tracked `.beads/issues.jsonl` line for `t42-csw6.3` was then updated from
the fresh `bd export` record only for this bead, so unrelated closed beads from
the shared database were not pulled into this worktree commit.

W&B links: not applicable.

HF links: not applicable.

Claim-ledger impact: no claim-ledger change.

## Links

[[w42]] | [[winning42-strategy-measurement]] | [[gus-strategy-tags-probe]] |
[[forge-analysis]] | [[forge]] | [[gus]] | [[eval-seed-holdout]]
