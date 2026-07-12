# Partnership failure atlas

This directory is the first architecture-neutral measurement spine for the
partnership wall. It joins only evidence with a demonstrated shared identity and
keeps relevant but incompatible evidence visible in an inventory.

The executable join is exact across:

- `w42/joined_claim_row_model_table/joined_claim_action_rows.csv`
- `w42/sequence_seat_counterfactuals/labeled_sequence_action_rows.csv`
- `w42/seat_position_claim_tests/labeled_action_rows.csv`
- `w42/phase4_sequence_handshape_tests/labeled_handshape_action_rows.csv`
- `w42/book_validation_v1/wave1/t42-m2i7_cross_ai_agreement/per_action_source_picks.csv`

Each source has one unique row per `(decision key, candidate domino)`, and all
five identity sets match. The full atlas therefore remains a legal-action table;
decision-level value-spread and actual-action diagnostics are repeated on the
actions in their decision.

The cross-AI table is an exactly joinable prior attempt, not clean cross-AI
evidence. Its `gus_top_action` was generated directly from
`is_best_threshold`; the atlas renames it
`threshold_utility_top_action_proxy`. Its 997 nonblank
`dist_lens_top_action` rows came from a different corpus joined by positional
`game_idx + decision_idx + candidate_domino`; the values remain present only as
`dist_lens_top_action_confounded` with a confounded status.

Gus drama tables, arena matches, and Champion/Jud evidence are not force-joined.
They use different state namespaces or coarser granularities. The inventory now
includes both complete 280,560-row Gus parquet surfaces. Apparent partial
integer-key collisions were tested, but state, E[Q], and legal-action counts
diverge. The Champion inventory is intentionally bounded to summaries, metrics,
manifests, registered predictions/build reports, Jud demo manifests, and run26
self-play negatives; weights, images, NPZ files, and source code are omitted.
Paths, hashes, row counts, schemas, scientific status, confounds, and non-join
reasons live in `evidence_inventory.csv`.

## What survives

- uncertainty: per-action Q mean, regret, role/bid-dependent threshold mass, and
  `P(Q <= -18)` lower-tail mass, plus decision-level value span and
  best-versus-second gap. World vectors, sample count, sampler/version,
  weighting, and calibration are unavailable;
- role/order: bidder/partner/setter role, team, actor, trick seat, phase, current
  control, and sequence/seat detector labels;
- partner coordination: public/action-local partner-support and donation proxies;
- action observation: which legal action the source trajectory actually took;
- plan readiness: the `bidder_lead_plan` detector firing, kept explicitly as a
  proxy rather than a persistent plan;
- distributional utility: fixed threshold/tail ranking proxies only. No utility
  transform, policy consumer, or context-sensitive selection survives;
- handshape/action set: current trick count, threshold/tail ranks and gap,
  legal/called/off/count/double/beater counts, and phase-4 book labels;
- bidding: declaration and partial risk proxies at fixed `bid_value=30`.
  Cardinality is one, so this corpus cannot estimate bidding or bid-level effects.

## What does not survive

Every missing construct has both a blank value column and a status column. The
atlas does not invent:

- full `q_per_world` vectors or a joined Gus drama score;
- sampled-world count, sampler/version, world weighting, or distribution
  calibration;
- partner identity or a fixed-versus-shuffled partner condition;
- actor-policy likelihoods or action-conditioned posterior updates;
- the source trajectory's policy identity and any joinable current-champion
  action, so actual-action regret cannot be attributed to the champion;
- plan IDs, steps, completion, disruption, or cross-decision persistence;
- auction choices, varying bid levels, or complete auction history;
- pre-hand match score.

The `eval_only_labels` column remains offline evidence. It is not asserted to be
publicly observable or safe as a live policy feature.

## Reproduce and validate

The build does not need a GPU. Standard-library Python handles the action joins;
optional `pyarrow` reads only parquet footer metadata for Gus row counts and
schema. If it is absent, the inventory marks those metadata fields unavailable.

```bash
python w42/partnership_failure_atlas/run_atlas.py
python w42/partnership_failure_atlas/validate_outputs.py
python -m unittest w42.partnership_failure_atlas.test_failure_atlas
```

Generated artifacts:

- `atlas_full.csv.gz` — all compatible legal-action rows;
- `atlas_sample.csv` — deterministic hash sample for inspection;
- `evidence_inventory.csv` — compatible and incompatible evidence surfaces,
  including live hashes, counts, schemas, scientific status, and confounds;
- `summary.json` — coverage and structural limits;
- `manifest.json` — source/artifact hashes, schema, statuses, and field order.

Use `--max-rows N --output-dir PATH` for a smoke build. A limited build may end
mid-decision and is for runner validation only; the checked-in full build is the
scientific artifact.

Validation re-hashes every manifest input and inventoried live source, and
rechecks row counts/schemas where metadata support is available. A stale source
or generated output therefore fails validation.
