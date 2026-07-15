# Otis W6 — tied-strategy rollout (P7, the count-fate-ledger open question)

**PROBE RUN (2026-07-15).** The registered P7 probe is complete — 32 genuine
retention decisions, M=50 valid worlds, two world-sample seeds, both legs, on
MPS. See the **Probe** section at the bottom for the graded result. The
**Smoke** section below (2 decisions × M=20, CPU) is the earlier shakedown and
its CPU timings are upper bounds superseded by the MPS probe.

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

---

## Probe (M=50 valid worlds, 2 seeds, MPS) — the graded P7 run

Machine output: `scratch/otis-night/w6_probe.json`. Runner:
`scratch/otis-night/run_w6_probe.py` (finder: `scratch/otis-night/find_retention.py`).
Reuses `otis/tiedroll.py` end-to-end. (One MPS bug fixed for the probe:
`belief_weights_for_decision` now places the student's inputs on the model's
device — the smoke ran CPU-only and never exercised it.)

### Probe set — genuine retention choices, tricks 1–4

A **genuine retention choice** is a ply where the actor is *void in the led
suit* (must discard) **and** has ≥ 2 legal non-trump **junk** (non-count)
sloughs — the exact condition under which the keep/release lever bites. Found by
replaying each game's true deal to the decision ply and checking void + junk
count (not merely "holds junk in hand", the smoke's weaker filter). **879**
such decisions exist across two v2 chunks (train 0–9, 10–19); the probe takes
the **top 8 by drama per trick (tricks 1–4) = 32 decisions**. *Drama* = std over
the decision's valid worlds of the oracle Q at a\* — the world-disagreement
signal W5's bimodality test measured, applied to retention plies. Every one of
the 32 had ≥ 50 valid worlds, so **0 skips** and M=50 throughout (the
issue-#52 `valid_world_indices` filter was applied to every leg). Coverage:
8 decisions per trick 1–4; declarations {1,2,3,4,5,6,7,8,9} represented.

### Cost — the band (≤ 10 s/decision at M=50 on MPS)

The band's unit is a single M=50 evaluation (baselines + all retention cells for
one seed). Per-seed:

| stat | s/decision (M=50, one seed) |
|---|---|
| median | **1.0** |
| p90 | **1.52** |
| max | 1.63 |

Running **both** seeds plus all cells: total median **2.06 s/decision**, p90
**3.11 s**. **Feasibility PASS with ~6–10× headroom** — comfortably under the
10 s band even doubled for replication. (Probe wall: 63.5 s for all 32
decisions × 2 seeds × both legs × up to 6 cells each.)

### The P7 cell test

Registered gate: **≥ 1 replicable cell where tied guard-retention ≥ +2 points
while clairvoyant ≤ +0.5**. Replicable = same tied sign AND |tied Δ| within 50 %
across the two world-sample seeds. Each ordered junk pair (keep, release) is one
cell; the tied price is antisymmetric so both orientations are recorded (144
ordered cells total).

**PASS: 3 replicable qualifying ordered cells, across 2 distinct decisions / 3
distinct retention levers.**

| decision | decl | keep / release | tied Δ (seed0, seed1) | tied mean | clair mean | fusion gap |
|---|---|---|---|---|---|---|
| g58 d5 (t1) | 8 | **0-0 / 3-0** | 2.79, 2.70 | **+2.75** | −0.24 | **+2.99** |
| g58 d5 (t1) | 8 | 0-0 / 3-1 | 1.66, 2.63 | +2.14 | −0.24 | +2.38 |
| g74 d5 (t1) | 4 | 6-2 / 1-1 | 3.01, 2.05 | +2.53 | −1.28 | +3.82 |

**Best cell — g58 d5, keep 0-0 over release 3-0:** the tied evaluator prices
retaining the double-blank over the 3-0 at **+2.75 points** (replicated across
seeds: 2.79 / 2.70), while the clairvoyant playout prices it at **−0.24** — a
**fusion gap of +2.99 points** the full-deal oracle is structurally near-blind
to. This is the [[strategy-fusion]] claim the ledger is built on, replicated on
real M=50 worlds under a repaired sampler.

### Fusion gap, aggregate (144 ordered cells)

Strategy fusion does **not** zero the clairvoyant retention delta universally
(only 5.6 % of cells are exactly 0 at M=50 — the smoke's "always 0.00" was a
small-M / single-decl artifact). It **systematically shrinks** it: |clair Δ|
median **0.50** / p90 **1.45**, versus |tied Δ| median **0.68** / p90 **2.88**,
max 8.46. 19.4 % of cells clear |tied| ≥ 2; 51.4 % have |clair| ≤ 0.5. The
qualifying cells are precisely where the tied leg prices real retention that the
clairvoyant leg cannot see.

### Fusion-gap meter — per-tile fate distributions, tied vs clairvoyant

The instrument the otis entity page names: per count tile, the 8-class fate
distribution `(captured_by ∈ {my_team, their_team}) × (mode ∈
{led, followed, trumped_in, sloughed})`, averaged over the 32 probed decisions,
under the tied (info-honest) vs clairvoyant (fusion) baseline legs.

my_team **capture probability** (Σ of the four my_team modes), tied vs
clairvoyant — the fusion gap made concrete at the tile level:

| tile | tied captures | clairvoyant captures | Δ (fusion) |
|---|---|---|---|
| 5-5 | 0.478 | 0.512 | −0.034 |
| 6-4 | 0.507 | 0.551 | −0.044 |
| 5-0 | 0.589 | 0.584 | +0.005 |
| 4-1 | 0.611 | 0.655 | −0.044 |
| 3-2 | 0.562 | 0.551 | +0.011 |

The info-honest tied player captures a few points less of the big count tiles
(5-5, 6-4, 4-1) than the clairvoyant upper bound — the cost of not knowing the
deal — while the small/scattered tiles (5-0, 3-2) are near parity. The full
8-class tied/clairvoyant tables per tile are in `w6_probe.json`
(`fusion_meter`); the largest mode-level shifts are in the *followed* and
*their_team/sloughed* columns.

### Honest notes / scope

- **Probe corpus is bid-30 eq data.** These are fixed-bid-30 solve fixtures, not
  on-policy auction states; the retention *prices* are honest for these worlds
  but the decision *distribution* is not the incumbent's play distribution.
- **Retention lever = slough-only.** The commitment fires only when the actor is
  void and about to discard a non-trump tile (guard = keep, walker = release).
  **Leading the walker** (playing junk to bait/trigger a late count catch) is a
  different lever and is **out of scope** for W6.
- **2 of 3 qualifying cells are decl 8 (doubles-suit).** decl 8 is not
  representable in the TS engine's base rules (issue #51), so those trajectories
  are not cross-engine-refereed — but the Python parser's P1 identity
  (`our + their = 42`) still asserts internally on every rolled world, so the
  prices are self-consistent. The third qualifying cell (g74, decl 4) is fully
  refereeable.
- **Replication is real, not trivial.** The two seeds draw different 50-world
  subsets (all 32 decisions had > 50 valid worlds), so the ~40 % of cells that
  replicate did so across genuinely different world samples; the 3 qualifying
  cells all replicated with |tied Δ| within 50 % and identical sign.

**P7 verdict: feasibility PASS (≈ 1 s/decision at M=50 on MPS, far under the
10 s band), and the fusion gap exists and replicates (≥ 1 qualifying cell — 3
found).** Both halves of the registered P7 band are met.

### Independent gate check (2026-07-15)

A second agent re-verified the probe against the raw JSON and re-ran the best
cell from scratch on MPS:

- All 144 ordered cells' means, fusion gaps, replication flags, and P7 flags
  recompute exactly from the stored per-seed deltas; the 3-qualifying-cell /
  2-decision / 3-lever tally reproduces. Tile-id decode confirmed
  (0=0-0, 6=3-0, 7=3-1, 23=6-2, 2=1-1).
- Cost accounting is honest: `secs_per_seed` covers belief weights + both
  baseline legs + all cell rollouts for one seed (the band's unit); the only
  excluded work is the valid-world scan (decision totals are 1.02–1.09× the
  two-seed sum). Even the full two-seed total (median 2.06 s) sits under the
  band.
- Best cell re-run (fresh process, MPS, both seeds): tied Δ **+2.793 / +2.700**
  bit-identical to the JSON; clairvoyant Δ −0.423 / −0.062 (mean −0.242,
  matching); fusion gaps 3.216 / 2.761 (mean 2.989, matching). 0.4–0.6 s per
  seed.
- Fusion-meter distributions sum to 1.0 per tile; the capture-probability table
  above reproduces; the tied leg sloughs more big-tile count (insurance-shaped)
  and the clairvoyant leg shows no protection-priced qualifying cell (both
  qualifying decisions' clairvoyant means ≤ 0, well under the +0.5 gate).

**P7 will grade (W7) as: PASS both halves** — cost band met with ~5–10×
headroom, gap cell exists and replicates (3 cells, 2 decisions, 3 levers) — with
the recorded caveats (bid-30 fixtures, slough-only lever, 2/3 cells decl 8).
