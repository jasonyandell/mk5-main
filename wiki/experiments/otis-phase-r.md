---
title: Otis Phase R — the clean deck (corpus regen on Vast)
kind: experiment
first_seen: 2026-07-15
last_updated: 2026-07-15
status: active
---

## Questions

Executing [[otis]] Phase R per the bridge ticket
([issue #55](https://github.com/jasonyandell/mk5-main/issues/55)), which
executes [issue #52](https://github.com/jasonyandell/mk5-main/issues/52)
(27–67% malformed stored worlds in the April eq corpus) gated by
[issue #51](https://github.com/jasonyandell/mk5-main/issues/51) (decl 8
purge — decided 2026-07-15: 9 declarations). Overnight 2026-07-15→16,
worktree branch `worktree-otis-night2`. In question form:

1. Can the joint-world corpus be regenerated on the repaired sampler
   ([[stage-0-closure]]) such that **zero** malformed worlds survive an
   independent read-side referee — with validity asserted at write time so
   this class of rot cannot fossilize silently again? (R1, R3)
2. Where does the inherited belief Bayes ceiling
   ([[belief-bayes-ceiling]], 39.184% measured on contaminated tensors) land
   when re-derived on the clean deck? (R2)
3. Can per-world posterior weights — the never-done next step from
   [[w42-phase2-hidden-domino-threat-attribution]] — be recorded at write
   time for every decision? (R4)
4. Does the regenerated corpus carry exactly 9 declarations, with decl 8
   (doubles-suit, unrepresentable in [[engine]] base rules) purged from
   enumeration? (R5)

## Predictions — registered before the regen runs

Bands written 2026-07-15 before any Vast instance was provisioned. Grading
fills the last column. The R2 prior was formed by a declared proxy
instrument (below), not by the graded run.

| # | claim | band (pass) | falsifier (genuine negative) | graded |
|---|---|---|---|---|
| R1 | Writer-side validity assertions hold | 100.000% of stored worlds in every regenerated chunk pass the independent read-side 28-domino-deal referee (exact cover of the unseen set + per-seat cardinality) | any invalid stored world = stop-the-line; the write-time assertion missed a rot class | |
| R2 | Bayes ceiling rises on the clean deck | `gus/eval/belief_ceiling.py` verbatim on regenerated `corpus_eval_20.pt` lands in **[39.3%, 41.5%]** top-1 (inherited: 39.184% contaminated; proxy prior: 39.949%) | < 39.0% (contamination was inflating the ceiling — reopens the sampler-distribution question) or > 43% (the proxy misunderstands the repaired posterior) | |
| R3 | Contamination is zero by the old referee | re-measuring the #52 probe on regenerated counterparts (`corpus_v2_train_20-29` d0, `corpus_train_chunk_0-99` d0) yields 0.000% invalid (was 66.9% / 57.5%) | any nonzero rate | |
| R4 | Posterior weights recorded | every regenerated decision carries per-world posterior weights, sum = 1 ± 1e-5; ESS distribution reported descriptively (no band — first measurement) | weights missing or unnormalized on any decision | |
| R5 | Decl mix is clean | 0 decl-8 games in any regenerated file; exactly 9 declarations represented where the recipe enumerates declarations | any decl-8 game | |

Ops are descriptive, not graded: 4090 games/s, $ spent, chunk manifest +
sha coverage, HF push latency.

### The R2 prior instrument (declared)

The 39.184% inherited number reproduced **exactly** on
`corpus_eval_20.pt` via an independent vectorized re-implementation of the
ceiling (5,880 slots — referee agreement with `gus/eval/belief_ceiling.py`).
Filtering each decision's stored worlds to strict validity (the
[[otis]] `valid_world_indices` semantics) before the argmax yields
**39.949%** — the proxy for the clean posterior. Largest gains sit exactly
where contamination is worst: d0 +3.6pp (54.2% of stored worlds valid),
d1 +3.0pp (53.3% valid). Caveats the band must absorb: the regenerated
corpus draws fresh worlds (sampling noise), the repaired sampler is uniform
over consistent completions (the filtered legacy sampler only approximates
it), adaptive E[Q]-driven action selection may shift trajectories, and the
decl-8 purge changes the game mix (~10% of games).

## Method

Regen runs on Vast.ai (cheapest reliable 4090) per the [[zeb-fleet-ops]]
pattern: keychain `vastai-api-key`, **rsync the worktree — never
clone-forge** (the [[stage-0-closure]] CUDA arm's rule), one GPU job per
machine, heartbeat logs ≥ 1/60s, `yes | vastai destroy` on completion.
Generator: `forge.eq.generate` with (new this session) write-time validity
assertions on every stored world, per-world posterior weights, and
chunk manifests with shas — chunked and resumable (skip-if-exists). Chunks
push as they land to a **new** HF dataset repo
(`jasonyandell/texas-42-joint-world-corpus-v2`, sibling of the
contaminated original; token per `docs/SECRETS.md`); the old repo stays up,
READMEs cross-link both directions with the #52 note. Interim
defense-in-depth regardless of regen:
`gus/model/dataset_seq_world.py` gains read-time validity filtering (the
`valid_world_indices` lift), so any training between now and full corpus
adoption auto-filters. Grading referee: a standalone read-side validity
scan (not the writer's own assertion) over every pushed chunk.

## Receipts

(fills as chunks land)

## Links

[[otis]] · [[otis-v0]] · [[count-fate-ledger]] · [[world-sampler-mrv-audit]] ·
[[belief-bayes-ceiling]] · [[stage-0-closure]] · [[zeb-fleet-ops]] ·
[[joint-world-tensor]] · [[run-artifacts-policy]] ·
[[w42-phase2-hidden-domino-threat-attribution]]
