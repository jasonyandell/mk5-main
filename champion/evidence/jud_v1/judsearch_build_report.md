# judsearch build report (jud v1 · JS1)

Branch: `worktree-agent-aa7dd6afab74190b1`, commit `5f539e0` on top of `f550205`.
Files: `arena/jud_search.py` (new), `arena/test_jud_search.py` (new), `arena/cli.py` (registry).

## What it is

`JudSearch` — same `choose(states, bid_values, marks, marks_to_win)` protocol as
`JudPlay`/`LensPlay`. Per root decision:

1. **Belief lift (reused, not reinvented)**: `zeb_states_to_game_state_tensor` →
   `sample_worlds_batched` → `WorldSamplerMRV` — byte-for-byte the LensPlay path,
   including `_max_pool_size` and the global-torch-RNG consumption. Worlds are
   sampled ONCE per state and shared across candidate moves (the belief doesn't
   depend on the move; common worlds cancel sampling noise out of the comparison).
2. **Determinization**: a sampled world gives each hidden seat's *remaining* tiles;
   the original 7-tile hand is reconstructed as sampled-remaining ∪ tiles that seat
   already played (public history). `dataclasses.replace(state, hands=...)` yields a
   fully consistent engine state, so the engine's own `legal_actions`/`apply_action`
   (legality, trick resolution, scoring) drive the rollout unchanged.
3. **In-trick rollout**: apply the candidate move, then each remaining seat of the
   trick replies greedily via the jud head — featurizing ITS OWN info-state inside
   the sampled world (its sampled hand + public history), offense argmax E[pts],
   defense argmin: exactly JudPlay's rule. Forced follows are applied without a net
   query.
4. **Leaf**: at trick resolution (possibly TERMINAL), `featurize_state(leaf,
   seat=root_mover)` → `mean_points` — the root mover's post-resolution info-state,
   which only sees the world through the simulated opponents' plays (no hidden-hand
   leak; `featurize_state` reads only `hands[pov]` + public history).
5. Average E[pts] over the N worlds per move; argmax (root defender: argmin).

No oracle anywhere. Net stays on CPU by default (470k MLP — dispatch beats MPS;
same reasoning as JudPlay); `device=` is plumbed through sampler + forwards so a
device flag works.

## How the batching works

One flat rollout list over (state × legal-move × world), advanced in lockstep:

- Root: `L_i × N` rollouts per state (skipped entirely when `len(legal) == 1`).
- ≤3 reply steps: each step collects every active rollout's legal children into ONE
  feature batch → one net forward → per-rollout signed argmax. Rollouts whose trick
  already resolved (mover was last) never enter the loop.
- One final leaf forward over all rollouts; per-(state, move) mean over the
  world-contiguous slice; per-state signed argmax.

So a tick costs ≤ 3 reply forwards + 1 leaf forward regardless of batch width —
the forwards batch across states, moves, and worlds simultaneously. The Python
cost is `featurize_state` (per-child), which dominates; the structure keeps every
net call maximal.

`choose = _sample_worlds ∘ _choose_given_worlds`. The second stage is a pure
function of (states, worlds) — that's where batched == sequential holds and is
tested. A strict end-to-end batched==sequential is impossible while preserving the
sampler's global-RNG behavior (batch composition changes the stream — the same
reason `--fast-batching` is documented "statistically equivalent, not
byte-identical").

## Registry

`judsearch[:n<worlds>][,model=<path>]` in `arena/cli.py::parse_play`, default
`n10`, default model `champion/jud_net.pt`. Mirrors judplay's spec style; does not
require the oracle (`needs_model` untouched).

## Test status

`arena/test_jud_search.py` — 10 tests, all passing (~1s):

1. **Sign through the rollout**: PreferDomino stub — offense plays the favorite,
   defender starves it, mixed batch routes signs per game (favorite is in the
   mover's hand, so no sampled world can ever contain it → world-independent).
2. **Masking**: illegal favorite in a forced-follow state never chosen; untrained
   JudNet through a real engine match (apply_action validates every root choice
   AND every simulated reply).
3. **Mover-last degeneracy**: trick resolves on the mover's own play → leaf is
   world-independent → N=1 and N=10 choices identical (JudSearch still differs
   from JudPlay by design: post-resolution vs post-move pricing).
4. **Batched == sequential** on `_choose_given_worlds` with fixed (true-hands)
   worlds.
5. **Forced-follow short circuit** (no rollout at all).
6. **Registry**: parse (`n2`, default `n10`) + full CPU match through
   `parse_bidder("jud:...")` + `parse_play("judsearch:n2,...")`.

Full suites: `pytest arena champion` → **143 passed, 3 skipped** (baseline 133+3
at `f550205` in the worktree — belief adapters gitignored — plus the 10 new).

## Smoke timing (CPU, M-series, 4-game batch, marks-to-win 3)

- `jud+judsearch:n10` vs `jud+judplay`: 4 games / 16 hands in **2.8s**.
- `jud+judplay` vs `jud+judplay`, same seeds: **0.2s**.

So the judsearch side costs ≈ **0.16s/hand** at n=10 in a 4-wide batch — roughly
**10–15× judplay** per decision (judplay: 1 forward of L rows; judsearch:
≤4 forwards over ~L×N×(replies) rows plus L×N×~10 Python featurizations, which
dominate). Wider batches amortize the fixed overhead further; a 512-game A/B at
n=10 should land in the tens-of-minutes range on CPU, comfortably comparable to
lens on MPS. Versus `lens:ev n=10`: judsearch has no oracle transformer and no
GPU dependency; wall-clock per decision is the featurization loop, not the net.

## Registered A/B commands (orchestrator runs these — NOT run here)

Heads: `champion/jud_net_r4.pt` (best loop head) — also worth one pass with
`champion/jud_net_r0.pt`. Reserved definitive seeds 7000000 / 9000000 per the
grading protocol; JS1's bar is **play-only ≥ +1.0 marks/game over judplay, same
head** (prediction in `champion/evidence/jud_v1/jud_v1_predictions.md`).

(a) **Play channel** — same bid (net:wp), judsearch vs lens:ev n=10:

```bash
python -u -m arena.cli \
  --team-a "net:wp+judsearch:n10,model=champion/jud_net_r4.pt" \
  --team-b "net:wp+lens:ev" \
  --n-games 512 --n-samples 10 --device mps --base-seed 7000000 \
  --out-dir arena/results/js1_playonly_judsearch_vs_lens
```

and the prediction's literal play-only bar (judsearch vs judplay, same head, bid fixed):

```bash
python -u -m arena.cli \
  --team-a "net:wp+judsearch:n10,model=champion/jud_net_r4.pt" \
  --team-b "net:wp+judplay:model=champion/jud_net_r4.pt" \
  --n-games 512 --device cpu --base-seed 7000000 \
  --out-dir arena/results/js1_playonly_judsearch_vs_judplay
```

(b) **Full stack** — jud bidder + judsearch vs the reigning baseline:

```bash
python -u -m arena.cli \
  --team-a "jud:model=champion/jud_net_r4.pt+judsearch:n10,model=champion/jud_net_r4.pt" \
  --team-b "net:wp+lens:ev" \
  --n-games 512 --n-samples 10 --device mps --base-seed 7000000 \
  --out-dir arena/results/js1_fullstack_judsearch_vs_lens
```

Notes for the orchestrator: `--device` steers the oracle/lens side; judsearch is
CPU-pinned in the registry (intentional). Repeat at `--base-seed 9000000` for the
second reserved measurement. The r4/r0 heads live in the main checkout only
(untracked at `f550205`), so run from the main checkout or pass absolute paths.

## Unresolved / follow-ups

- `featurize_state` is the wall-clock bottleneck (pure-Python play-history loop,
  ~10 calls per rollout). If A/Bs feel slow, a vectorized batch featurizer is the
  first lever; the lockstep structure already isolates it.
- Terminal leaves are evaluated by the head like every other leaf (spec-literal).
  When all 28 tiles are public the exact declaring-team points are computable —
  swapping exact values in at terminal leaves is a one-line experiment if leaf
  noise matters.
- Worlds are shared across candidate moves (common-random-numbers). If a strict
  per-move resample is ever wanted for a diagnostic, it's a loop-bound change.
- This report lives in the WORKTREE's gitignored `scratch/jud-v1/` (the agent
  sandbox blocks writes to the main checkout); copy to
  `scratch/jud-v1/judsearch_build_report.md` in the main checkout if wanted there.
