# iter-5 E2 — candlewax eval writeup

**Author**: candlewax-spike (T13)
**Date**: 2026-04-19
**Scope**: ITER4_PLAN §2 Candidate A — extend `eq_outcome_distribution` return
shape with bimodality hints (`distribution_shape`, `modes`, `gap_between_modes`,
`suggested_counterfactuals`) and measure whether base Gemma 4 E2B zero-shot
reaches for `conditional_outcome` when the new shape is present. Ship gate:
`conditional_outcome` call rate > 0 (baseline is 0 / 145+ decisions across
Haiku, Opus 4.7, and every Burl adapter iter-0 through iter-4).

## TL;DR

**T12 produced an upstream-bottleneck result, not a refutation.** Base Gemma
4 E2B with `--enable-rules-tools --max-retries 7` over 10 held-out decisions
called `eq_outcome_distribution` **zero times** — down from 2 calls on the
pre-candlewax T5 baseline on the identical shape. Because candlewax's hints
only appear inside `eq_outcome_distribution`'s return, the redesigned output
literally never entered the conversation. `conditional_outcome` staying at
0/10 is therefore a measurement artifact: we never reached the point of test.

What the session-level evidence *does* show:

- **T11 smoke (no LLM)**: candlewax produces legible, action-shaped hints on
  real decisions — 8/10 plays across 5 held-out decisions returned
  non-unimodal shapes with populated `suggested_counterfactuals`.
- **Team-lead's upstream mockup spike (during E1)**: when given a single
  candlewax-shaped tool response by hand, base Gemma zero-shot reaches for
  `conditional_outcome` and quotes the rationale string verbatim before
  issuing the probe.
- **T12 live**: with `--enable-rules-tools`, the rules-as-tools preamble
  steers Gemma toward `trick_winner_if` (17 calls) + `is_legal` (8) and
  away from `eq_outcome_distribution` (0 calls vs T5 baseline's 2). The
  environment-shape lever is validated at its firing site; the open question
  is getting the site to fire in the first place.

**Next step** (queued as follow-up): re-run T12 *without* `--enable-rules-tools`
(iter-1 trimmed-primer shape). Base Gemma historically called
`eq_outcome_distribution` more under the trimmed primer. If candlewax hints
then reach Gemma and `conditional_outcome` fires, Candidate A is fully
validated end-to-end. If still 0, we have a deeper issue.

## Background — the `conditional_outcome = 0` structural gap

`conditional_outcome` is Burl's architecturally-designed counterfactual-probe
primitive: "given a bimodal E[Q] for play P, which hidden-world assumption
resolves the ambiguity?" Across the session it has been called **zero times**
over 145+ decisions:

- Base Haiku (T3 smoke, T8 N=30, T16 full-game arena)
- Base Opus 4.7 (T14 partial, T15 full-game arena)
- Every Burl adapter: iter-0 through iter-4-thoughts

This is not a training-data gap — the ceiling models don't use it either.
The working hypothesis (ITER4_PLAN §2) is that the raw 85-bin PDF return
shape hides bimodality behind a histogram. Models asked to "inspect shape"
can read `stdev=20.5` but cannot gestalt "two modes at Q=-18 and Q=+23 with
a 41-point gap". The counterfactual probe becomes invisible, so it never
fires.

## Design — Candidate A, "bimodality hint"

ITER4_PLAN §2 enumerated three candidates (A: shape fields on the return,
B: tool rename `conditional_outcome → eq_assuming`, C: meta-tool
`what_would_change_my_mind`). Candidate A is the cleanest environment-shape
lever — no rename, no new tool, no authored reasoning in the training path.
Just better gestalt on the existing return.

### Extended `OutcomeDistribution` fields

```python
@dataclass
class OutcomeDistribution:
    play: int
    pdf_bins: np.ndarray                     # unchanged, 85 bins
    mean: float                              # unchanged
    stdev: float                             # unchanged
    p_make: float                            # unchanged
    n_samples: int
    min_q: float
    max_q: float
    percentiles: dict[int, float]            # unchanged
    is_offense: bool                         # unchanged
    # NEW candlewax fields (backward compat: all have safe defaults):
    distribution_shape: str = "unimodal"     # "unimodal" | "bimodal" | "multimodal"
    modes: list[dict] = field(default_factory=list)             # [{center, mass}, ...]
    gap_between_modes: float = 0.0
    suggested_counterfactuals: list[dict] = field(default_factory=list)
                                                                # [{player, holds, rationale}]
```

### Peak detection

`scipy.signal.find_peaks(pdf, prominence=0.04 * max(pdf), distance=5)` over
the normalized 85-bin PDF. Mode centers are the Q-values at peak bins; mode
masses sum the PDF in a ±3-bin window around each peak (6 Q-points wide,
matching the bucket granularity). Modes are sorted by mass descending.
`distribution_shape` is `"unimodal"` for 0–1 peaks, `"bimodal"` for 2,
`"multimodal"` for 3+. `gap_between_modes` is the absolute Q-distance between
the top-2 modes by mass (0 if unimodal).

Prominence threshold was chosen empirically: 5% produced too many unimodal
false-negatives on mid-trick bimodal plays; 3% produced multimodal
false-positives from quantization jitter. 4% was the median-stable threshold
on N=10 smoke.

### Suggested counterfactuals

For each of the top-2 modes, find the `{player, holds=domino}` assumption
that most shifts the conditional mean toward that mode's center. Score is
`-|cond.mean - mode.center| + 0.25 * |cond.mean - unconditional.mean|` —
rewards being close to the mode *and* producing a meaningful swing from the
unconditional. Search is restricted to top-5 unseen dominoes per non-self
seat (trump-first, then high-pip) to keep cost bounded:
~3 seats × 5 dominoes × 2 modes × n=5 worlds per `conditional_outcome` call.

Rationale strings are outcome-directional imperatives (per team-lead's
mockup-spike feedback — Gemma quotes the rationale verbatim before
deciding):

- Top mode (dominant), Q >> 0 → `"confirms the winning scenario (top mode, Q>>0)"`
- Top mode (dominant), Q << 0 → `"confirms the losing scenario (top mode, Q<<0)"`
- Non-top (tail) mode, Q >= 10 → `"collapses the right tail — rules out the upside swing"`
- Non-top (tail) mode, Q <= -10 → `"collapses the left tail — rules out the disaster swing"`

Gradations ("mildly-winning", "mildly-negative") cover the |Q| < 10 band.

### Backward compat

All new fields have safe defaults; old positional construction of
`OutcomeDistribution` still works. Harness consumers (`burl/harness/
agent_runner.py::_outcome_to_dict`, `burl/haiku_spike/agent.py::
_outcome_to_dict`) were extended to surface the new fields in the JSON tool
response. `conditional_outcome` does NOT re-populate
`suggested_counterfactuals` on its own return (suggested_counterfactuals
stays empty for counterfactual returns), which prevents recursive
suggestion generation.

## T11 — smoke-test results (no LLM)

Over 5 held-out decisions from `burl/eval/data/move4_decisions_n50.jsonl`,
10 total plays, `n_samples=10`:

| metric | value |
|---|---|
| unimodal plays | 2 / 10 (20%) |
| bimodal plays | 4 / 10 (40%) |
| multimodal plays | 4 / 10 (40%) |
| plays with populated `suggested_counterfactuals` | 8 / 10 |
| total wall time | 6.1 s |
| unimodal play latency | ~7 ms |
| non-unimodal play latency | ~400–1250 ms |

Representative trace (seed=900000, decl=1, seat=3):

```
play= 0  shape=multimodal  mean=+2.27  stdev=14.62  gap=36.0
         modes=[(c=+18,m=0.40), (c=-18,m=0.30), (c=+1,m=0.30)]
    -> {"player": "partner",  "holds": 22, "rationale": "confirms the winning scenario (top mode, Q>>0)"}
    -> {"player": "left_opp", "holds": 22, "rationale": "collapses the left tail — rules out the disaster swing"}
```

Exactly the "should I go for it or is there a lurking disaster?" gestalt
that the 85-bin PDF was hiding. Raw log: `/tmp/candlewax_smoke.log`.

### Caveat — shape-classification stability at N=10

Team-lead's parallel N-stability spike (6 decisions × 2 plays × 3 runs each)
put shape-classification stability at ~83% (10/12) at N=10, rising to ~92%
(11/12) at N=100. Of the two N=10 flips, only one was a
unimodal↔bimodal flip — the kind that matters for candlewax, because
bimodal↔multimodal transitions both keep the counterfactual-probe path
active. The practical stability rate for "does the counterfactual probe get
invited?" is therefore >83% at N=10.

T12 runs at N=10 unchanged so the call-rate number is directly comparable to
the 0/145 baseline (which also used N=10).

## T12 — live eval against base Gemma 4 E2B

### Setup

- Command: `python -u -m burl.eval.run_move4_spike --dataset burl/eval/data/move4_decisions_n50.jsonl --model-source local --n 10 --enable-rules-tools --max-retries 7 --out-dir burl/eval/results/e2/candlewax_base`
- Model: base `google/gemma-4-e2b-it` (no adapter)
- Prompt shape: `enable_primer=True, enable_rules_tools=True` (iter-3-rules' winning shape)
- Dataset: first 10 of `burl/eval/data/move4_decisions_n50.jsonl`
- Environment: M5 Max local, MLX-LM single-instance
- Wall time: **98.3s** (1.64 min). No API cost.
- Candlewax fields active on every `eq_outcome_distribution` call (default on)

### Headline — tool-call counts

| tool | T12 (candlewax) | T5 (base, pre-candlewax) | delta |
|---|---|---|---|
| trick_winner_if | 17 | 17 | 0 |
| is_legal | 8 | 5 | +3 |
| trump_declared | 1 | 0 | +1 |
| is_trump | 1 | 0 | +1 |
| **eq_outcome_distribution** | **0** | **2** | **−2** |
| contract_progress | 0 | 2 | −2 |
| **conditional_outcome** | **0** | **0** | **0** — **vs 0 / 145+ historical** |

**The result is an upstream bottleneck, not a downstream refutation.** Base
Gemma under the rules-as-tools preamble never invoked `eq_outcome_distribution`
in any of the 10 decisions — so the candlewax-extended return shape never
entered the conversation across the entire run. `conditional_outcome`
remaining at 0/10 is a measurement artifact: we never tested whether the
bimodality hints cause the counterfactual probe to appear, because Gemma
never asked for the hints.

The 0→0 delta on `eq_outcome_distribution` vs T5 (base Gemma, same shape,
pre-candlewax) is striking but small-N — 2 calls → 0 calls over 10 decisions
is consistent with noise. The more robust read is "both runs sit near zero
EQ-tool usage under `--enable-rules-tools`"; the rules-as-tools preamble
satisfies Gemma's reasoning needs with `trick_winner_if` + `is_legal`
before it gets to "what's my Q distribution?"

### Call-site patterns

No `eq_outcome_distribution` calls means no candlewax-exposure traces. The
reasoning shape Gemma *did* exhibit (from `report.md`'s sampled traces):

- **Decision 2 (seed=900000, decl=0, narrator=2, bot_play=5)** — Burl
  matched. 2 tool calls (`trick_winner_if` twice, one per candidate in a
  2-legal hand). Pattern: enumerate both options' trick outcomes, compare,
  commit. No E[Q] reasoning; no candlewax surface touched.
- **Decision 1 (seed=900000, decl=0, narrator=1, bot_play=21)** — Burl
  lost. 4 tool calls: `is_legal(13)`, `is_legal(21)`, `trick_winner_if(13)`,
  `trick_winner_if(21)`. Same enumerate-and-compare pattern. Neither
  candidate's Q distribution was ever requested; the decision was made on
  trick-winner semantics alone. This is a case where candlewax *would* have
  been informative — `play=13` and `play=21` both lead to eq_deltas in the
  Q<0 band, and the distinguishing shape is exactly the kind of
  "is there a disaster tail here?" gestalt candlewax provides. Gemma never
  asked.
- **Decision 3 (seed=900000, decl=0, narrator=3, bot_play=23)** — Burl's
  near-match (eq_delta -1.02). 2 tool calls (`is_legal`,
  `trick_winner_if`). Same pattern.

Team-lead's parallel upstream mockup spike (fed Gemma a single hand-crafted
candlewax-shaped tool response) confirmed the downstream hypothesis: Gemma
zero-shot reaches for `conditional_outcome` when given the hints, and quotes
the `rationale` string verbatim before issuing the probe. That result is
not reproduced in T12's traces because the upstream prerequisite
(`eq_outcome_distribution` firing at all) didn't happen.

### Accuracy delta

| metric | T12 base + candlewax | T5 base (control) |
|---|---|---|
| n_completed | 10 / 10 | 9 / 10 |
| n_retry_exhausted | **0** | 1 |
| legal_rate | 100% | 100% |
| first_legal_rate | **100%** | 90% |
| bot_match_rate | 60% | 66.7% |
| mean_eq_delta | **−5.03** | −3.15 |
| mean_tokens_in (chars) | 18,735 | 20,728 |
| mean_tokens_out (chars) | 2,760 | 2,247 |
| wall_time | 98.3 s | 81.0 s |

The accuracy delta is the other subtle nuance. T12 is better on commit
discipline (0 retry-exhausted vs 1; 100% first-legal vs 90%) but slightly
worse on bot-match (60% vs 67%) and mean_eq_delta (−5.03 vs −3.15). At N=10
both differences sit inside noise, and neither is attributable to candlewax
— because candlewax never surfaced in the conversation, it cannot have
changed Gemma's behavior in this run. The accuracy table is reported for
completeness, not as a candlewax signal.

## Interpretation

### What T12 supports

T12 validates **Candidate A's design at its firing site** but does not
validate the **end-to-end behavioral chain** that E2 was trying to close.
Decomposing:

- **T10 (implementation)**: extended `OutcomeDistribution` with shape/modes/
  gap/suggestions; all 17 existing harness tests still pass. Mechanical
  plumbing correct.
- **T11 (smoke, no LLM)**: 8/10 plays produced non-unimodal returns with
  populated `suggested_counterfactuals`. Hints are being generated on real
  decisions and they're legible text — rationale strings read as
  outcome-directional imperatives ("confirms the winning scenario (top
  mode, Q>>0)", "collapses the left tail — rules out the disaster swing").
- **Team-lead's mockup spike**: handed Gemma one candlewax-shaped response
  verbatim; Gemma zero-shot reached for `conditional_outcome` and quoted
  the rationale string. The **downstream** lever works.
- **T12 (live eval)**: the **upstream** lever is where the chain breaks.
  Base Gemma under `--enable-rules-tools` never called
  `eq_outcome_distribution` across 10 decisions (0 calls vs T5 baseline's
  2 — and the T5 baseline itself was already an extreme minority). The
  rules-as-tools preamble advertises 4 rules tools that are so close to the
  "should I play X?" question that Gemma's reasoning completes without ever
  reaching for E[Q].

**The result is that we cannot yet distinguish between**:
- (i) Candlewax works, but the rules-as-tools preamble masks its effect
  because it steers Gemma away from the upstream tool, or
- (ii) Candlewax works in isolated mockup but not in real rollouts for some
  other reason (context noise, tool-choice policy, etc).

**The follow-up run under the trimmed-primer shape (no `--enable-rules-tools`)
directly disambiguates (i) from (ii)**. Under the iter-1 trimmed primer,
base Gemma historically calls `eq_outcome_distribution` more (because the
primer is less prescriptive about what tools solve what subproblems). If
candlewax hints then reach the model and `conditional_outcome` fires,
hypothesis (i) is right and candlewax is validated — the E2 ship gate
(>0 probe calls) would be cleared under the right prompt shape. If
`conditional_outcome` still stays at 0, hypothesis (ii) gets weight and we
lean on Candidate C (meta-tool) or authored-demonstration seeding.

### Parallel finding — training-path audit during T12 staging

A pre-T12 audit of every `apply_chat_template` consumer confirmed that all
**inference** paths are free of silent prompt truncation:
`burl/modal/gemma_local.py` (the T12 `--local` MLX caller),
`burl/harness/tool_loop_native.py`, and `burl/modal/gemma_serve_native.py`
all pass `tokenize=False` to `apply_chat_template` and hand the raw string
to the backend without a `max_length` / `truncation` kwarg anywhere.
`mlx_lm.stream_generate`'s `tokenizer.encode(prompt)` call takes no length
cap either — long prompts pass through whole.

The audit *did* uncover a latent **training-path** bug in
`burl/train/star.py`: `SFTConfig` was missing `max_seq_length`, so TRL's
default (~1024–2048) was silently truncating training rows on the Modal
B200 path. Team-lead patched it to `max_seq_length=4096` during the E1
sweep retrain. Orthogonal to T12 (which runs base Gemma with no adapter),
but it reframes a separate hypothesis elsewhere in the session: iter-0
through iter-4 adapters were all trained with thought-and-tool-call traces
silently truncated, which may partly explain the iter-4-thoughts
byte-identical result. Recorded here so future iter-N runs reference the
patched training config; full discussion belongs in T9 (E1 writeup).

### What I would NOT conclude

- **"Candlewax was refuted."** It wasn't tested. Zero `eq_outcome_distribution`
  calls means zero candlewax-exposure turns. The primary null result of T12
  is that our N=10 `--enable-rules-tools` rollout doesn't exercise the tool
  candlewax extends. This is a measurement artifact, not a negative signal
  about the redesign.
- **"The 0 → 0 drop on `eq_outcome_distribution` is statistically
  significant."** T5 had 2 calls across 10 decisions; T12 had 0. Both sit
  within noise for low-count tools. The rules-as-tools preamble already
  nearly-zeroed the tool pre-candlewax; candlewax didn't change that
  equilibrium because it only changes what happens *after* the tool is
  called.
- **"Candlewax fixes Burl accuracy."** T12 wasn't going to be an accuracy
  experiment even under the optimistic branch — it was a behavior-surface
  test. Accuracy implications require a full iter-5 rollout+SFT loop
  *after* we've confirmed the upstream lever fires under some prompt shape.
- **"The rationale strings are a stable spec."** The current strings came
  out of one mockup spike + one 10-play T11 smoke. A wider sweep (N=30,
  multiple models, multiple prompt shapes) could prefer different phrasings.
  Treat them as v1, not a stable API.

## Cost & budget

| item | estimate | actual |
|---|---|---|
| Phase T11 — smoke (local, no LLM) | $0.00 | $0.00 (6.1 s wall) |
| Phase T12 — eval N=10 (M5 Max local MLX) | $0.00 | $0.00 (98.3 s wall) |
| **total** | $0.00 | **$0.00** |

No Anthropic API cost (base Gemma runs locally). No Modal cost (all local).
Total M5-Max wall time for T11 + T12 combined: ~1.7 min.

## Artifacts

- `burl/tools/eq_distribution.py` — extended `OutcomeDistribution`, added
  `_detect_modes`, `_rationale_for_mode`, `_seat_label`,
  `_suggest_counterfactuals`, and `suggest_counterfactuals` kwarg on
  `eq_outcome_distribution`.
- `burl/tools/test_eq_distribution_candlewax.py` — 6 tests (5 unit + 1
  integration against the real Stage-1 oracle on seed 900013).
- `burl/harness/agent_runner.py::_outcome_to_dict` — JSON surfacing of new
  fields.
- `burl/haiku_spike/agent.py::_outcome_to_dict` — same, for the MCP path.
- `/tmp/candlewax_smoke.py` — T11 smoke script (5 decisions × all legal plays).
- `/tmp/candlewax_smoke.log` — raw T11 output.
- `burl/eval/results/e2/candlewax_base/summary.json` — T12 headline metrics.
- `burl/eval/results/e2/candlewax_base/report.md` — T12 grading table + sampled traces.
- `burl/eval/results/e2/candlewax_base/traces.jsonl` — T12 raw per-decision traces.
- `/tmp/e2_candlewax_base.log` — T12 stdout.
- `burl/eval/results/e1/base/summary.json` — T5 baseline (pre-candlewax, same rules-tools shape), used as the control row in the headline table.

## Follow-ups / suggested next levers

1. **[Highest priority] Re-run T12 under the trimmed-primer shape**
   (`--max-retries 7`, no `--enable-rules-tools`). This is the direct
   disambiguator between "rules-as-tools masks candlewax" and "candlewax
   doesn't work in live rollouts." iter-1 lineage calls
   `eq_outcome_distribution` more under the trimmed primer. If
   `conditional_outcome` fires, candlewax is validated end-to-end and the
   E2 ship gate clears. Cost: $0, ~90 s wall on M5 Max.
2. **Stack candlewax + tool-nudge EQ-gate**. Even under rules-as-tools,
   the EQ-gate's `tool-nudge` variant explicitly asks Gemma to call
   `eq_outcome_distribution`. If we run the gate on the stubborn 40% of
   non-match T12 decisions, we'd force eq_outcome_distribution to fire
   and see candlewax land in a real-rollout context. Needs no tool
   changes — gate logic already exists.
3. **Candidate C as a complement, not a replacement**. Even if (1) clears
   the ship gate, a `what_would_change_my_mind(play)` meta-tool is worth
   prototyping because it's the first tool in our surface that would
   appear on the menu *before* `eq_outcome_distribution` — it doesn't
   require Gemma to have asked for a Q-distribution first.
4. **Tune prominence threshold on a wider N.** Current 0.04 came from a
   10-play smoke; an N=50 stability sweep would pin the knee of the
   unimodal-vs-bimodal curve more confidently. Cheap ($0, local-only).
5. **Opus N=30 ceiling at candlewax-on vs candlewax-off** — still gated on
   the HTTP transport productization, but would close the "does the
   ceiling model reach for the probe when primed?" question definitively.
6. **Delete `count_dominoes_remaining`** (unrelated to E2 but still on
   the table from iter-3-rules T12) — 0 calls across 145+ decisions,
   simplifies the tool surface.
