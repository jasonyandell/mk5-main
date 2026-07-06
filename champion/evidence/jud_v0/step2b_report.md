# jud v0 — Step 2b: GPU corpus generation + margin_net train/eval

**GitHub #32.** This agent owned the GPU stretch: generate the realized-outcome
corpus via arena runs on MPS (Phase A), then train + evaluate `margin_net` on CPU
(Phase B). Sole GPU user — one arena process at a time throughout.

**Bottom line:** 20/20 chunks generated (22,001 hands), `margin_net` trained and
evaluated. The model is **well-calibrated to realized 4-seat outcomes** (ECE
P(pts≥30) = 0.046; predicted vs test-empirical exceedance within ≤0.029 at every
threshold) and **reproduces the optimism gap** — its exceedance sits ~0.09–0.13
below the reliable double-dummy oracle at bids 30/36 (registered prediction 2 ✓).

## Config

- venv: `forge/venv/bin/python`; checkpoint:
  `forge/models/domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt`
- per chunk: `--n-games 100 --n-samples 10 --device mps`
- Chunks 01–12: `net:wp+lens:ev` self-play (on-policy auctions), seeds 3000000 +50000/chunk
- Chunks 13–20: `random+lens:ev` vs `random+lens:ev` (coverage), seeds 4000000 +50000/chunk
- Actual wall: **~85–89 s / chunk** (~28 min total) — far under the 8-min estimate;
  MPS is fast and net-vs-net bidding is <1 ms/hand.

### Pre-flight: random bidder (chunk 13+) sanity — static + empirical

Static read of `arena/bidders.py::RandomBidder`: `declare()` = `rng.choice(PIP_TRUMPS)`
(`PIP_TRUMPS = range(7)`, hand_metrics.py:26) → uniform over all 7 pip trumps;
`bid()` bids independent of hand strength (p_bid=0.3). In random-vs-random whoever
bids wins → varied auction winners. **Confirmed empirically on chunk 13** before
committing to 14–20: declarations near-uniform (130–166 per pip trump), made-rate
collapsed to 0.12 (weak declarers ✓), `bidder_team_pts` near-uniform across all 9
buckets, bidder seats balanced (233–290). **Not degenerate → no substitution;**
kept `random` vs `random` for 13–20 as specified.

## Phase A — corpus generation (all 20 chunks)

| chunk | regime | seed | wall(s) | n_hands | made | mean_bid |
|-------|--------|------|---------|---------|------|----------|
| 01 | self-play | 3000000 | 89.0 | 1125 | 0.689 | 30.4 |
| 02 | self-play | 3050000 | 87.3 | 1101 | 0.691 | 30.4 |
| 03 | self-play | 3100000 | 89.0 | 1106 | 0.675 | 30.4 |
| 04 | self-play | 3150000 | 86.8 | 1084 | 0.719 | 30.3 |
| 05 | self-play | 3200000 | 87.2 | 1124 | 0.659 | 30.4 |
| 06 | self-play | 3250000 | 87.0 | 1116 | 0.691 | 30.4 |
| 07 | self-play | 3300000 | 85.3 | 1076 | 0.683 | 30.4 |
| 08 | self-play | 3350000 | 86.9 | 1085 | 0.699 | 30.4 |
| 09 | self-play | 3400000 | 85.3 | 1094 | 0.698 | 30.4 |
| 10 | self-play | 3450000 | 85.3 | 1086 | 0.690 | 30.4 |
| 11 | self-play | 3500000 | 85.4 | 1088 | 0.698 | 30.4 |
| 12 | self-play | 3550000 | 87.3 | 1155 | 0.681 | 30.4 |
| 13 | random | 4000000 | 84.8 | 1062 | 0.118 | 37.7 |
| 14 | random | 4050000 | 84.1 | 1098 | 0.129 | 37.5 |
| 15 | random | 4100000 | 87.3 | 1098 | 0.105 | 37.9 |
| 16 | random | 4150000 | 85.3 | 1114 | 0.130 | 37.9 |
| 17 | random | 4200000 | 86.0 | 1131 | 0.117 | 37.9 |
| 18 | random | 4250000 | 87.3 | 1084 | 0.120 | 37.8 |
| 19 | random | 4300000 | 85.5 | 1084 | 0.133 | 37.9 |
| 20 | random | 4350000 | 83.6 | 1090 | 0.140 | 37.9 |

Full per-chunk `bid_hist`/`decl_hist` in `corpus/gen.log`; complete stats
(buckets, seats, regime totals) in `corpus/manifest.json`.

### Corpus totals (from manifest.json)

| regime | chunks | hands | made-rate | mean bidder_team_pts |
|--------|--------|-------|-----------|----------------------|
| **all** | 20 | **22001** | 0.464 | 28.6 |
| self-play (01–12) | 12 | 13240 | 0.689 | 33.4 |
| random (13–20) | 8 | 8761 | 0.124 | 21.3 |

The two regimes give the intended **bimodal realized-outcome coverage**: self-play
concentrates high (mean 33.4 pts, 40–42 the modal bucket — sensible net bids that
usually make), random fills the low/middle uniformly (mean 21.3 pts, made 0.12 —
weak declarers spanning the whole 0–42 range). `bidder_team_pts` 5-pt buckets over
the full corpus (from manifest `totals.all`):

```
0-4:938  5-9:1010  10-14:1268  15-19:1915  20-24:2186  25-29:2646  30-34:3245  35-39:2870  40-42:5923   (sum 22001)
```

## Phase B — train + evaluate margin_net (CPU)

`margin_net.py` landed as commit `537b3e2` (checked `git log`) with
`scratch/jud-v0/step2a_report.md` giving the exact CLIs. Ran both on CPU.

**Corpus glob deviation (flagged):** step2a's CLI uses `corpus/*.json`, but the
team lead asked for `corpus/manifest.json` in the same dir — and the dataset loader
(`margin_net.py:288`) does `payload["snapshots"]`, which **KeyErrors on any dict
without that key** (manifest.json, and `bookkeep.py`/`manifest_build.py` are `.py`).
So I ran train/eval with **`scratch/jud-v0/corpus/snaps_*.json`** — functionally
the exact 20-file corpus, just excluding the manifest. No code was modified.

### Train

```
forge/venv/bin/python -u -m champion.margin_net train --corpus 'scratch/jud-v0/corpus/snaps_*.json'
```
- Dataset after dedup (paired-half replays collapse via `(seed,hand_idx,bids,bidder,decl_id)`):
  **10455 train / 616 val / 568 test** (22001 raw → 11639 unique; 90/5/5 by deal hash).
- Early-stopped at epoch 19 (best val CE 3.0046 @ epoch 11). Adam, CE loss.
- Saved `champion/margin_net.pt` (**uncommitted**, per instruction — gate review first).

### Evaluate

```
forge/venv/bin/python -u -m champion.margin_net eval --corpus 'scratch/jud-v0/corpus/snaps_*.json' --model champion/margin_net.pt
```
Writes `scratch/jud-v0/margin_net_eval.json` + `margin_net_reliability.png`.

**Headline test metrics (N=568):**

| metric | value |
|--------|-------|
| test CE (43-class, uniform=3.76) | **2.943** |
| MAE(mean-pts) | **7.34** |
| ECE P(pts≥30) | **0.046** |

### Prediction-2 comparison — exceedance at 30 / 36 / 42

margin_net predicted E[P(pts≥t)] and test-set empirical P(pts≥t) vs the
`optimism_gap.json` realized make-rate curve (N=604) and reliable oracle
double-dummy p_make (N=50):

| threshold | net predicted | net empirical | realized (N=604) | oracle DD (N=50) |
|-----------|---------------|---------------|------------------|------------------|
| **30** | 0.548 | 0.553 | 0.520 | **0.640** (reliable) |
| **36** | 0.331 | 0.359 | 0.298 | **0.466** (reliable) |
| **42** | 0.149 | 0.146 | 0.112 | 0.000 (**UNRELIABLE** — atlas top-bid artifact) |

Reading of the three points:
1. **Calibration (the gate):** net predicted ≈ net empirical at all three (Δ =
   0.005 / 0.028 / 0.003); across all 13 thresholds max |Δ| = 0.029 (@32). The
   value head is honest about *its own* realized-outcome distribution.
2. **Optimism gap reproduced (prediction 2):** where the oracle is reliable (30,
   36) the net's exceedance sits **0.09–0.13 below** the double-dummy oracle —
   the same over-optimism direction `optimism_gap.json` measured (oracle − realized
   = 0.12 @30, 0.168 @36). A model trained on realized 4-seat play does *not*
   inherit the double-dummy's optimism.
3. **@42 the oracle is unusable** (0.0, flagged `reliable:false` in the source —
   atlas top-bid threshold artifact), so the only valid comparison at 42 is
   net-vs-realized: 0.149/0.146 vs 0.112, i.e. the net slightly over the parquet
   realized curve but exactly on its own test empirical.

**Caveat:** the net's exceedance is unconditional over the test hands' *actual*
declarations (mixed self-play + random), whereas `optimism_gap`'s curves are
conditioned on bid *level* over a different, unpaired hand pool (bidding-results
parquet). So the realized/oracle overlays are reference curves, not paired
comparisons — the net-predicted-vs-net-empirical pair is the paired, in-corpus
ground truth. Both curves are monotone-declining with the same 0.55→0.15 shape.

### Gate verdict (registered prediction 2) — PASS

Team-lead framing: the exceedance/reliability curve must track the
`optimism_gap.json` **realized** shape (0.52@30 → ~0.19@41), **not the oracle's**.

| anchor | net predicted | net empirical | realized | oracle DD |
|--------|---------------|---------------|----------|-----------|
| t=30 | 0.548 | 0.553 | **0.520** | 0.640 |
| t=41 | 0.226 | 0.232 | **0.192** | n/a (unreliable) |

- net_pred MAE vs **realized** curve over 30–41: **0.020**
- net_pred MAE vs **reliable oracle** points (30,32,35,36,39): **0.118**
- ⟹ the net is **6× closer to realized than to oracle**, and sits *inside* the
  optimism gap (below oracle, on/just above realized) with the correct
  monotone-declining 0.55→0.19 shape. **Gate met.**

### Figure

`scratch/jud-v0/margin_net_reliability.png` (two panels): (1) reliability of
P(pts≥30), points on the diagonal, ECE 0.046; (2) exceedance 30→42 — green
(net predicted) tracks black (test empirical) tightly, both below the blue oracle
triangles (optimism gap) and just above the orange realized curve.

## Artifacts (all under scratch/jud-v0/, none committed)

- `corpus/chunk_01..20/` — summary.json + per_hand.csv + per_game.csv per chunk
- `corpus/snaps_01..20.json` — realized-outcome snapshot payloads (the corpus)
- `corpus/gen.log` — one line/chunk (teams, seed, wall, n_hands, bid/decl hists)
- `corpus/manifest.json` — per-chunk + regime totals
- `corpus/{bookkeep,manifest_build}.py` — bookkeeping helpers (scratch)
- `champion/margin_net.pt` — trained model (**uncommitted**, awaiting gate review)
- `margin_net_eval.json`, `margin_net_reliability.png` — eval outputs
- `margin_net_train.log`, `margin_net_eval.log` — run logs

## Surprises

1. **Wall time ~9× under estimate** (85 s vs 8 min/chunk). MPS + <1 ms/hand net
   bidding. The whole GPU phase was ~28 min, not 2.5 h.
2. **Login shell is zsh (1-indexed arrays).** My first batch loop used
   `${arr[$i]}` with a 0-based index → the i=0 iteration ran with empty args
   (harmless: the CLI arg-errored and bookkeep crashed before writing) and chunk
   05 was skipped. Caught it immediately, reran 05, and switched to a
   `for pair in "nn:seed"` idiom (index-free, zsh/bash-safe) for all later batches.
   No corrupt data reached the corpus.
3. **The random regime is a near-perfect coverage complement**, not just "some
   weak declarers" — `bidder_team_pts` came out essentially *uniform* across all
   9 buckets (self-play alone is heavily 40–42-modal). The union gives full
   support 0–42, which is exactly what a calibrated value head needs.
4. **Random bids escalate to a 42 spike** (mean bid 37.7, bid_hist peaks at 42).
   Uniform choice among remaining legal raises walks the auction up. Irrelevant to
   the corpus — featurization is level-blind (verified by step2a's test 2) — but
   it's why the random made-rate is so low (many forced 42-point contracts).
