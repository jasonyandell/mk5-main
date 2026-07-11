## Canonical decision measurement

`arena.decision_records` converts a completed Arena `MatchResult` into one
JSONL row per play decision. It is post-match and read-only: policy execution,
batching, world sampling, and action selection are unchanged.

The CLI surface is:

```bash
python -m arena.cli \
  --team-a margin:wp,model=champion/margin_net_r8.pt+lens:ev \
  --team-b margin:wp,model=champion/margin_net_r8.pt+lens:ev \
  --n-games 64 \
  --emit-decisions arena/results/decisions.jsonl.gz
```

The command also writes a sibling `.manifest.json`, containing schema versions, record count,
record checksum, match configuration, policy fingerprints, artifact hashes,
and the leakage boundary.

`.jsonl.gz` is recommended for atlas-scale runs. Compression is deterministic:
the gzip timestamp is zero and no source filename is embedded, so identical
records produce identical compressed bytes and checksums. Plain `.jsonl`
remains supported.

The canonical C0 spelling is explicit about its promoted bidder head:
`margin:wp,model=champion/margin_net_r8.pt+lens:ev`. The implicit
`champion/margin_net.pt` default has different weights and must not be labeled
C0. Artifact paths and contents are hashed only for components selected by the
actual team spec; absent selected paths are recorded as `missing`, and
unselected defaults are not added.

### Identity layers

| ID | Inputs | Intended comparison |
|---|---|---|
| `rules_state_id` | actor, winning contract, public play, hand points | Same rules/play state across auction-evidence and score controls |
| `actor_information_state_id` | rules state plus actor's remaining hand | Canonical runtime information set |
| `decision_context_id` | actor information plus full auction and match score | Exact policy decision context |
| `world_state_id` | decision context plus all remaining hands | Offline truth and forced-fixture replay only |

The first three IDs never hash an opponent's private hand. `world_state_id`
and `offline_truth` are explicitly eval-only. Raw deal seed, game index, and
hand index live only under `offline_truth`; the online trajectory exposes only
an opaque content ID and decision index. The manifest marks `match.base_seed`
as offline provenance because it also reconstructs deals.

`trajectory_id` deliberately hashes deal provenance and the complete future
play so two stochastic reruns cannot collide; `record_id` derives from it.
Both are **join/grouping-only** and must never be policy inputs. The manifest
lists them under `join_only_ids`. Only `rules_state_id`,
`actor_information_state_id`, and `decision_context_id` are runtime-safe state
identity claims.

### Mechanism boundary

Every row keeps eight separate sections: `uncertainty`, `role_order`,
`partner_coordination`, `action_derived_inference`, `plan_persistence`,
`distributional_utility`, `bidding`, and `match_score`.

The current Arena can observe roles, legal actions, static seat partnerships,
the auction, score, chosen action, and realized outcome. It cannot yet emit
actor action likelihoods, posterior changes, a persistent plan, a sender or
receiver convention, fixed-versus-shuffled cohort assignment, or Lens's full
Q/PDF tensors. Those fields remain null with explicit statuses. They must not
be filled from detector proxies.

### Falsifier

The exporter is invalid if a recorded prefix cannot replay through the game
engine, the selected domino is not legal, hidden-hand swaps change an online
ID, auction/score controls fail to separate at `decision_context_id`, raw deal
reconstruction fields escape the offline boundary, IDs duplicate, or the
manifest checksum/count disagrees with the JSONL. The writer fails closed
before output on count or ID violations. `arena/test_decision_records.py`
checks those conditions on CPU without loading a policy network.
