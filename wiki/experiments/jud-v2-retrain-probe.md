---
title: jud v2 retrain probe — clean sampler + exploration
kind: experiment
first_seen: 2026-07-17
last_updated: 2026-07-17
status: complete
---

Born from the table42 field report ([issue #66](https://github.com/jasonyandell/mk5-main/issues/66),
[issue #69](https://github.com/jasonyandell/mk5-main/issues/69)): the game-night
jud seat was terrible, and the standing conjecture (Jason, 2026-07-17) was that
the [[world-sampler-mrv-audit]] Metal defect had poisoned jud's training corpus —
"it makes bids its play can't cash because its play is compromised."

## Question

Is jud v1's play badness data contamination (the pre-repair `WorldSamplerMRV`
on MPS generated the entire 2026-07-06 corpus) — or mechanism? Secondary: does
per-move exploration in self-play, which directly attacks the measured
label-coverage fault, move the full-stack grade?

## Prior findings the probe builds on

A 10-agent adversarial audit (find + verify per dimension) of featurization,
emission, training, bid path, and sampler found **no live line-bug in the jud
path** — and dissolved one plausible candidate: the zeroed-non-winner-bids
constructor (`forge/zeb/game.py`) is used only by forge oracle tooling; the
arena engine, the training snapshots, and the table42 host all carry the real
auction. What it did confirm and measure (receipts on #69):

1. **Greedy zero-exploration self-play**: 4459/5929 corpus deals carry exactly
   ONE distinct opening lead (rest ≤2) of 7 legal — 6 of 7 serve-time children
   are off-distribution.
2. **Single MC label per hand** — every info-state row carries the one realized
   outcome; no counterfactual coverage.
3. **Ply-1 discriminability**: at an opening lead the 7 children differ in
   **9 of 350 input dims**; a 1-ply value head reading that yields the observed
   ~2-pt flat spread (the 6-3 lead of game 0716-222859).
4. The served 0.68 bid claim was the **round-0 checkpoint** at the table
   (`jud_net.pt`); `jud_net_r5.pt` prices the same bid 0.333 and passes.

## Method

- Regenerate the full 7-chunk base corpus with the **repaired** sampler on MPS,
  same seeds/composition as 2026-07-06 (`champion/evidence/jud_v2/gen_corpus_repaired.sh`);
  chunk m1's same-seed regen doubles as a field-drift probe.
- Re-run the v1 loop recipe (`champion/evidence/jud_v2/run_jud_loop_v2.py`)
  with one change: self-play corpus generation uses **ε=0.15 per-move uniform
  exploration** (`EpsilonJudPlay`); grading A/Bs stay greedy. Heads saved as
  `champion/jud_net_v2_r{r}.pt`.
- A new regression test pins the sampler in the float32-unsafe regime
  (`forge/eq/test_sampling_mrv_gpu.py::test_float32_unsafe_root_count_regime_valid_and_uniform`,
  21-tile pool, 399,072,960-count root, cpu+mps).

## Result

**Same-seed field drift: null.** Chunk m1 pre- vs post-repair: made-rate
63.8/63.9% → 63.5/63.3%, bid and declaration histograms near-identical,
5549 → 5552 snapshots. The Metal defect flipped argmax only in tail states;
corpus aggregates are stable.

**Loop trajectory: unchanged.** Full-stack jud+judplay vs net:wp+lens:ev
(512-game r0, 256-game rounds):

| round | v2 (clean + ε-explore) | v1 (contaminated, greedy) |
|---|---|---|
| r0 | −4.76 [−4.91, −4.60] | −4.37 [−4.55, −4.19] |
| r1 | −4.30 [−4.54, −4.06] | −4.38 [−4.62, −4.13] |
| r2 | −3.95 [−4.24, −3.66] | −4.16 [−4.40, −3.89] |

(v2 rounds 3–4 not run — the loop was stopped externally mid-r3; state is
resumable, and r0–r2 already decide the question.)

**Per-hand note (attribution-limited):** v2 r0 prices the #66 hand at 0.454
and passes (old r0: 0.679, bid 31), and ε-exploration widens the opening-lead
value spread ~2 → ~7 pts — but single-hand deltas confound training variance
with data effects; the graded rows above are the finding. Aggregate round-0
over-bidding persists in v2 (offense share 59%, made 28%) and dissolves through
the loop exactly as in v1.

## Epistemic classification

- **Contamination hypothesis: measured null at corpus scale.** Clean-sampler
  corpus + exploration reproduces v1's grade trajectory within CI. jud's play
  badness is **mechanistic**: greedy 1-ply value consumption over a
  representation whose per-move signal is 9 dims in 350, trained on
  single-label MC targets. The [[world-sampler-mrv-audit]] defect was real and
  is repaired; it is not why jud plays badly.
- The label-coverage fault (finding 1) is real but fixing it (ε-exploration)
  does not move marks at v1 capacity — consistent with
  [[jud-target-granularity]]'s capacity×target framing.
- Serving-side conclusion, independent of retraining: the table seat must be
  the graded champion (`margin:wp`(r8) + `lens:ev`), not a loop artifact.
  judplay in any round remains a ~−4 to −5.5 configuration
  ([[w42-jud-v1]], `champion/evidence/jud_v1/ab_playonly_summary.json`).

## Links

- [[jud]] — the player; [[w42-jud-v1]] — the v1 build this reruns
- [[world-sampler-mrv-audit]] — the defect that motivated the probe
- [[jud-target-granularity]] — the surviving v2-residual framing
- [issue #66](https://github.com/jasonyandell/mk5-main/issues/66) ·
  [issue #69](https://github.com/jasonyandell/mk5-main/issues/69) — receipts
