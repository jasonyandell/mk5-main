---
title: Partnership decision record v1
kind: experiment
first_seen: 2026-07-11
last_updated: 2026-07-11
status: complete
---

[[partnership-failure-atlas-v0]] found a useful action spine but zero
attributable current-champion failures. The next Stage-0 question was therefore
instrumental: can a completed [[arena]] match produce one replayable decision
record whose joins preserve public information, actor information, auction and
score context, hidden-world truth, policy identity, and the eight mechanisms in
[[partnership-value]] without collapsing them into proxies?

## Build

`arena/decision_records.py` replays each completed `HandRecord` through the
rules engine after the match. It emits one canonical JSONL row per actual play
and refuses the artifact if replay actor, legality, terminal score, row count,
or identity uniqueness disagrees with the retained trajectory. The CLI adds
`--emit-decisions`; `.jsonl.gz` uses deterministic compression and receives a
checksum-bearing sibling manifest.

Four versioned identities separate comparisons that the old archive could not:

| identity | contents | boundary |
|---|---|---|
| `rules_state_id` | actor, winning contract, public play, hand points | public rules state; excludes losing bids, match score, and all hands |
| `actor_information_state_id` | rules state plus actor's remaining hand | canonical runtime information set |
| `decision_context_id` | actor information plus full auction and pre-hand match score | exact public policy context |
| `world_state_id` | decision context plus every remaining hand | offline truth only |

`record_id` and `trajectory_id` are join/grouping keys, not policy features.
Raw seed, game index, hand index, initial/remaining hidden hands, and realized
outcome are eval-only. This distinction is load-bearing because Arena's deal
seed can be reconstructed from base seed plus game/hand/redeal coordinates.
The manifest marks base seed as offline provenance rather than treating a
hashable identifier as automatically safe.

Policy fingerprints name bidder, player, utility, exact selected artifact
paths and hashes, sampler algorithm/world count/device, corpus availability,
and Git-visible code state. Dirty checkouts include a streamed tracked-patch
hash and untracked-content manifest hash, so two different dirty policies do
not share an identity. The canonical C0 spelling is explicit:
`margin:wp,model=champion/margin_net_r8.pt+lens:ev`; the implicit
`champion/margin_net.pt` has different weights and is not C0.

## Result

Every row has separate `uncertainty`, `role_order`,
`partner_coordination`, `action_derived_inference`, `plan_persistence`,
`distributional_utility`, `bidding`, and `match_score` sections. Current Arena
observations now include:

- replayed actor, own hand, legal candidates, chosen action, hand score, and
  role/trick/auction order;
- complete public auction and pre-hand match score;
- exact seat partner/opponents and their policy IDs;
- bidder/player/artifact/sampler/utility/code provenance; and
- eval-only hidden world and realized hand/mark outcome.

The build deliberately leaves belief posterior, actor action likelihood,
posterior update, full Q/PDF, fixed-versus-shuffled cohort, sender/receiver
convention, and persistent plan state null with explicit statuses. It does not
upgrade W42 detectors into runtime mechanisms or call a static seat pairing
partnership value.

The full Arena suite passes `61` CPU tests. It runs a two-game Arena, replays
every 28-decision hand, isolates hidden-world versus auction versus score
identity changes, exercises the deterministic deal-leak boundary, verifies C0
artifact/sampler fingerprints without loading the
models, and round-trips plain and deterministic-gzip artifacts. This is an
instrument result, not a C0 reproduction or a policy win.

The retained integration smoke at
`arena/evidence/partnership_decision_record_v1/` then loads the real canonical
C0 from clean commit `bc4eb386`: explicit r8 bidder head, current large oracle,
`uniform-completion-dp-v1`, and `n=10`. Two symmetric games contain 23 hands
and 644 replayed decisions; the compressed record is 334 KiB and its manifest
count and SHA-256 agree. The 2-game score is not a policy estimate.

## Falsifier

The instrument is falsified if any retained prefix cannot replay, the chosen
tile is illegal, a hidden-hand swap changes an online-safe ID, losing-auction
or score controls fail to change only the context identity, deterministic deal
coordinates escape the offline boundary, different dirty code states collide,
or row IDs/count/checksum fail. The writer fails closed before producing an
artifact on structural mismatch.

## Decision

The shared future-state identity and policy-provenance seam is built. The
Stage-0 record still does not supply the causal variables it marks absent. A
small C0 reproduction can now produce an honest decision corpus; the next
microgame instrument must add one randomized arm at a time: auction evidence,
sender/decoder convention, persistent plan state, or contextual PDF consumer.
No successor architecture becomes eligible from this build alone.

## Reproduction

```bash
pytest -q arena/test_decision_records.py
python -m arena.cli \
  --team-a margin:wp,model=champion/margin_net_r8.pt+lens:ev \
  --team-b margin:wp,model=champion/margin_net_r8.pt+lens:ev \
  --n-games 2 --device cpu \
  --emit-decisions /tmp/c0-decisions.jsonl.gz
```

## Artifacts

- `arena/evidence/partnership_decision_record_v1/README.md`
- `arena/evidence/partnership_decision_record_v1/decisions.jsonl.gz`
- `arena/evidence/partnership_decision_record_v1/decisions.jsonl.gz.manifest.json`
- `arena/evidence/partnership_decision_record_v1/summary.json`

## Links

[[partnership-wall-research]] [[partnership-value]]
[[partnership-research-gates]] [[partnership-failure-atlas-v0]] [[arena]]
[[jud]] [[world-sampler-mrv-audit]]
