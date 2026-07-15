# Otis W6 — tied-strategy rollout (P7, the count-fate-ledger open question)

**SMOKE ONLY — probe pending.** All numbers below are a CPU shakedown of the
tool on 2 decisions × M=20 worlds. The registered P7 probe (probe set, M=50, on
MPS) has not been run; CPU timings are **upper bounds**. An ungraded P7 is "not
run," never "failed" ([[otis-v0]]).

Instrument: `otis/tiedroll.py`. Machine output: `scratch/otis-night/w6_smoke.json`.

## What it builds

The [[count-fate-ledger]] says pricing guards (junk kept to deny opponents'
rows) and walkers (junk that catches sloughed count late) "needs tied-strategy
rollouts — one action across worlds until the player's own observations differ —
which no current tool performs." W6 is that tool.

**Core design (binding).** A rollout where *every seat plays a deterministic
function of its OWN information state* (own hand + public history only) is
**tied-by-construction**: across sampled worlds the actor's actions are identical
exactly until its observations diverge — no divergence bookkeeping, only
batching. This is the information-honest evaluator that prices retention, which a
clairvoyant per-world playout zeroes via strategy fusion.

- **Tied leg.** Every seat plays the gus student's `pi_me` argmax over legal
  actions, queried with a ZERO world assignment. The `pi_me` head reads only
  `state_emb = cls_h + voids_encoder(voids)` — never the world — so the argmax is
  a pure function of the actor's info-state (the pattern from
  `gus/eval/lamir1.py:direct_decision`). Worlds are batched into one model call
  per ply; identical info-states yield identical actions by construction.
- **Clairvoyant leg (fusion baseline).** The SAME worlds, every seat playing the
  Stage-1 oracle's full-deal Q argmax (`forge/eq/oracle.py` via
  `forge/eq/generate/model.py:query_model` + `tokenize_batched`). Full-deal
  knowledge ⇒ strategy fusion ⇒ retention priced at ~0.

**State stepping is the exact engine.** Each world is reconstructed as a full 4×7
INITIAL deal — P's real hand + each opponent's `{prefix plays} ∪ {corpus
world_hands remaining}` — then `forge.eq.game_tensor.GameStateTensor` replays the
recorded prefix and rolls the tail. Completed 28-play trajectories are refereed
by `otis.fates.parse_game_fates` (the P1 identity asserts inside the parser),
yielding per-team points and the count-tile fates.

**Fates output.** Per world: final per-team points + full trajectory → tied-roll
FATES. Aggregate: belief-weighted (gus belief head, reusing
`otis/analysis/worldbank.py`) and uniform means; per-tile 8-class fate
distribution `(captured_by ∈ {my_team, their_team}) × (mode ∈ {led, followed,
trumped_in, sloughed})`.

**Retention pricing (the fusion gap).** For a keep/release junk pair, two paired
commitments run on common random worlds:
`c1 = (keep=X, release=Y)`, `c2 = (keep=Y, release=X)`, applied to the actor only
whenever it is void and about to slough a non-trump tile. `tied_delta =
mean_my_points(c1) − mean_my_points(c2)`; `clairvoyant_delta` likewise;
`fusion_gap = tied_delta − clairvoyant_delta`.

## Corpus-quality note

The eq eval corpus's opening-lead (`d_idx=0`) `world_hands` contain a fraction of
**inconsistent samples** — a domino the actor holds leaks into an opponent, or
two opponents share a tile (≈21% of worlds at the probed decisions;
`worlds_with_internal_dup` 842/4000 in one case). `valid_world_indices` filters
these to genuine 28-domino permutations before any rollout, so every priced world
is a real full deal. (The corpus is eval-fixture-only per the context pack; this
is a sampler artifact, not a parser bug.)

## Smoke results (2 decisions, M=20, CPU, seed 0)

| game | decl | actor | tied base (my pts, belief) | clair base | cells | P7 gap cells | best fusion gap |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 19.01 | 22.59 | 3 | **1** | +1.85 |
| 1 | 1 | 0 | 30.13 | 33.51 | 3 | 0 | +0.19 |

**The fusion gap is real and matches P7's shape.** Game 0, cell keep=24 /
release=17: `tied_delta = −2.69`, `clairvoyant_delta = 0.00`, `fusion_gap =
−2.69` — the tied evaluator prices which junk to keep at 2.7 points while the
clairvoyant playout is **exactly blind** to it. In fact every clairvoyant delta
at game 0 is 0.00: strategy fusion zeroes the whole retention economy, while the
tied deltas span −2.69…+1.85. This is the [[strategy-fusion]] claim the ledger is
built on, made concrete on real worlds.

(P7's registered gap cell wants `|tied| ≥ 2` while `|clairvoyant| ≤ 0.5`; game 0
keep24/rel17 satisfies it. The full probe scans the probe set for replication.)

## Cost (CPU upper bounds → M=50 extrapolation)

- Two unconstrained baseline legs (tied + clairvoyant), M=20: **0.46 s**.
- One retention cell = 4 rollouts (2 legs × 2 commitments), M=20: **0.94 s/cell**.
- Linear extrapolation to M=50: **≈ 2.35 s/cell**.

P7's cost band is **≤ 10 s/decision at M=50 on MPS**. Even on CPU (an upper
bound; MPS moves the model forwards off the CPU) the tool sits at ~2.4 s/cell —
comfortably inside the band. **Feasibility: PASS (pending the MPS probe.)**

## `--device` and the probe

`otis/tiedroll.py --device mps` is supported: the lightweight rollout stepping and
tokenization stay on CPU (they are cheap and the exact engine already lives
there), and only the heavy `pi_me` / oracle forwards run on the requested device.
The probe (probe set, M=50, `--device mps`) is the declared next step and owns the
GPU when it runs.

## Files

- Tool + CLI: `otis/tiedroll.py`
- Tests (tied-by-construction, legality/identity, determinism): `otis/tests/test_tiedroll.py`
- Smoke output: `scratch/otis-night/w6_smoke.json`
- Reused: `otis/fates.py` (referee), `otis/analysis/worldbank.py` (belief weights),
  `gus/eval/lamir1.py` (`_build_tokens_voids`), `forge/eq/*` (engine + oracle).
