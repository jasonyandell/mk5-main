# EQ-Gate — STaR trace-collection rejection-sampling mechanism

Design for Burl iter-2 Phase B (rationalization).  Lives in
`burl/harness/eq_gate.py` (pure functions) and hooks into
`burl/eval/run_move4_star_rollout.py` at the Phase B boundary.

## Motivation

Iter-0 Phase B rationalized every legal-loss by re-running Gemma with the
ground-truth play revealed in the system prompt and saying **"show how you
would reason your way to this play."**  Phase 2 reported **23/23
rationalizations converged**.  At the time we called that a surprise; in
hindsight it is the yes-bias fingerprint.  When you tell Gemma the answer,
Gemma rationalizes any chain that reaches it.  The corpus ends up teaching
"once you somehow know the right play, narrate a path to it."  That is not
the skill Burl needs.

Iter-1 trimmed the primer and re-harvested but kept the same rationalization
step.  The adapter shifted tool usage but did not recover spike v2's
`eq_outcome_distribution` reflex (iter-1: 0 eq calls; spike v2: 15).  Again
consistent with yes-bias — Gemma never needs the belief tool when the play
is already revealed.

The EQ-gate flips the rationalization polarity: **never reveal the bot play;
only reveal that the commit was legal but sub-optimal, and optionally nudge
toward tool use.**  A trace is kept only if Gemma, under that information
budget, **self-corrects** to the bot play.  Stubborn commits, forced flips
to other non-bot plays, and exhausted retries are classified and logged but
not mixed into the K1 corpus.  That reshapes the training signal from
"rationalize the answer" to "explore under pressure and land on the
right-by-construction play."

## Gate firing conditions

The gate fires on a rollout commit iff *all* of the following hold:

1. `committed_play` is not `None` (the rollout reached a terminal action).
2. `final_play_legal` is `True` (the engine accepted the commit).
3. `decision.per_play_eq[committed_play]` exists (we can score it).
4. `burl_eq + eq_epsilon < bot_eq`.  Default `eq_epsilon = 0.0` — any
   strict K1 loss is gate-eligible.  A positive epsilon (e.g. `0.25`) lets
   us skip near-ties during early iterations to reduce gate noise; wiring
   is kept for experiment-time overrides.

The gate does **not** fire on:

| Situation | Reason | What happens instead |
| --- | --- | --- |
| K1 win (`burl_eq >= bot_eq`) | Already a positive exemplar | Rollout trace goes straight into corpus as `source=rollout_win` |
| Illegal commit (final play rejected after retries exhausted) | Not a judgment failure — protocol failure | Existing engine-retry loop already handled it; rollout category stays `illegal`/`exhausted` |
| Missing `per_play_eq[committed_play]` | Can't score the commit | Fail closed: do not gate, flag the decision for manual review |

## Feedback prompts — three candidate variants

All variants share an invariant: **they never reveal `bot_play`, never
reveal any per-play E[Q] number, and never narrow the answer to fewer than
two legal plays.**  This is the yes-bias guardrail.  A teammate reviewing a
new variant should grep for the bot's domino id in the string and be able
to show, structurally, that no hint of "it's X" leaks.

### Variant A — minimal (baseline)

```
Your commit was legal, but another of your legal plays has a higher
expected-Q outcome. Take one more look: use your tools to compare legal
plays, then commit again.
```

*Signal taught:* "sub-optimal, try again."  No structural hint about what
to reconsider.  This is the purest rejection-sampling prompt and the
cleanest baseline for A/B against the others.

### Variant B — tool-nudge

```
Your commit was legal, but another of your legal plays looks stronger
under the E[Q] distribution. Before re-committing, call
eq_outcome_distribution on each remaining legal option and compare mean,
stdev, and p_make. You may also call conditional_outcome if a particular
hidden hand would swing it. Then commit.
```

*Signal taught:* "sub-optimal; the belief tool is how you check."  Designed
to address iter-0/iter-1's `eq_outcome_distribution` suppression directly.
Risk: if Gemma over-uses eq and still commits wrong, we've taught
redundant tool calls.  Worth measuring via tool histogram shift in the
gated corpus.

### Variant C — social

```
Think again — your partner needs this one. A different legal play in your
hand has a higher expected contribution to count. Re-examine the position
(count dominoes loose, trump structure, who is void in what) and commit
to the play that best supports your team.
```

*Signal taught:* 42-aware re-framing.  Leans into the Layer-1 framing
substrate (partner/team/count) that Gemma already speaks fluently.  Risk:
potentially over-anthropomorphizes and could induce theatrical rather than
analytical reasoning.  Scientifically interesting because it tests whether
Gemma's best re-exploration path is *social* or *structural*.

**Recommended default for iter-2's first run:** `tool-nudge` (Variant B).
It directly targets the belief-tool suppression we observed and preserves
the tool-mediated reasoning bet Burl is built on.  Variants A and C go in
parked-for-ablation status.

## Data schema — rationalization classes

The gate classifies every gated decision into one of five buckets.  The
classifier is a pure function on a sequence of `AttemptSummary` records
(one per commit attempt), so it can be unit-tested without a harness.

| Class | Condition | Training-corpus policy (default) |
| --- | --- | --- |
| `converged_first_try` | Single attempt, `final_play == bot_play` | Add as `rollout_win` — never actually gated, listed here so one classifier covers both corpora |
| `self_corrected` | `len(attempts) >= 2`, `attempts[-1].committed_play == bot_play` | **Keep** as `source=eq_gate_self_correct` — the signal the gate is designed to harvest |
| `forced_flip` | `len(attempts) >= 2`, `attempts[-1].committed_play != attempts[0].committed_play`, `!= bot_play`, commit was legal | Drop by default (open question 3) — Gemma moved but didn't land; adds noise to the K1 corpus |
| `stubborn` | `len(attempts) >= 2`, `attempts[-1].committed_play == attempts[0].committed_play`, `!= bot_play` | Drop.  Optional: log counts for diagnostic purposes |
| `exhausted` | Final attempt has no legal commit (retry-exhausted or `None`) | Drop.  Matches existing exhausted-rollout policy |

`AttemptSummary` is deliberately narrow:

```python
@dataclass(frozen=True)
class AttemptSummary:
    committed_play: int | None    # None when retry-exhausted
    legal: bool                    # engine legality of committed_play
    eq: float | None               # per_play_eq[committed_play], None if missing
    retry_exhausted: bool
```

The classifier never peeks at tool-call histograms or token counts — those
ride in the full `BurlTrace`s and can be joined by seed for downstream
analysis, but they do not influence the branch.

## Yes-bias mitigation — how we know it actually helps

1. **Feedback prompts never contain the answer.**  See invariant above.
   Enforced by a test that asserts `str(bot_play)` does not appear in any
   variant's output for an arbitrary `bot_play` id, and that the
   substring "domino_id=" does not appear in any variant.

2. **Gate fires on E[Q] dominance, not domino-id equality.**  Burl could
   in principle land on a different domino with equal or higher E[Q]
   (`check_commit` would not fire at all).  The gate never tries to steer
   Gemma to a specific domino — only *away* from objectively worse plays.

3. **The classifier only keeps `self_corrected`.**  A forced flip from
   play X to play Y (with Y still wrong under E[Q]) is not a yes-bias
   success; it's Gemma capitulating to pressure.  We reject it.

4. **Sanity check in the post-hoc stats.**  If `self_corrected` ever
   approaches 100% of gated decisions (like iter-0's 23/23 rationalization
   rate), that's a red flag — either the gate is leaking or the
   `eq_epsilon` is too loose.  Target regime: 25–50% `self_corrected`
   among gated losses.  This is a sanity band, not a KPI.

## Integration plan — hooks into `run_move4_star_rollout.py`

The current file structure:

- Phase A: rollout N=50 → `records[]` with `category ∈ {win, legal_loss, illegal, exhausted}`.
- Phase B: `_rationalize_one(decision, ...)` for every `legal_loss`.
- Phase C: `compose_sft_record(...)` across rollout wins and converged rationalizations.

Replacement diff is surgical — Phase B swaps from "reveal the answer and
rationalize" to "call `check_commit`, if fire then nudge and re-run up to
`max_gate_retries` times":

```python
# burl/eval/run_move4_star_rollout.py, Phase B loop
from burl.harness.eq_gate import (
    AttemptSummary, check_commit, classify_rationalization,
    gate_feedback_prompt,
)

for rec in records:
    if rec["category"] != "legal_loss":
        continue
    decision = _decision_for(rec)
    gate = check_commit(
        committed_play=rec["final_play"],
        legal=rec["final_play_legal"],
        per_play_eq=decision.per_play_eq,
        bot_play=int(decision.bot_play),
        bot_eq=float(decision.bot_eq),
    )
    if not gate.fire:
        continue  # K1 pass, illegal, or unresolvable — skip gate

    attempts = [AttemptSummary(
        committed_play=rec["final_play"],
        legal=rec["final_play_legal"],
        eq=rec["burl_eq"],
        retry_exhausted=rec["retry_exhausted"],
    )]

    for gi in range(1, max_gate_retries + 1):
        nudge = gate_feedback_prompt(gate_variant, attempt_idx=gi)
        nudged_trace, exhausted = _run_with_feedback(
            decision, model_fn, nudge, max_turns, max_retries,
        )
        nudged_rec = _build_record(decision, nudged_trace, exhausted, elapsed=0.0)
        attempts.append(AttemptSummary(
            committed_play=nudged_rec["final_play"],
            legal=nudged_rec["final_play_legal"],
            eq=nudged_rec["burl_eq"],
            retry_exhausted=nudged_rec["retry_exhausted"],
        ))
        if attempts[-1].committed_play == int(decision.bot_play):
            break
        if attempts[-1].retry_exhausted:
            break

    verdict = classify_rationalization(attempts, int(decision.bot_play))
    gate_records.append({"decision": decision, "attempts": attempts,
                         "verdict": verdict, "trace": nudged_trace})
```

`_run_with_feedback` is a thin wrapper over `NativeHarness.run` that either
(a) re-renders `render_native_messages` and then appends a `role="user"`
message containing the nudge before the harness drives turn 0, or (b)
passes an `extra_user_messages: list[str]` kwarg threaded through the
harness.  Option (a) keeps the harness unchanged; option (b) is cleaner.
Decision is an iter-2 implementation detail, not a design requirement.

Phase C gains one line: corpus composition includes records from
`gate_records` where `verdict == "self_corrected"`, tagged
`source=eq_gate_self_correct`.

### Where in the current file the hook lands

`burl/eval/run_move4_star_rollout.py`:

- Current line ~360-410: `# Phase B: rationalize the legal losses` —
  replace this block.
- Current line ~430-445: `# Phase C` — add one loop over `gate_records`
  analogous to the `rationalization_records` loop, composing SFT records
  for `self_corrected` verdicts.
- Stats block at line ~460: add `n_gate_converged`, `n_gate_stubborn`,
  `n_gate_forced_flip`, `n_gate_exhausted` alongside the rationalization
  counts so the before/after is legible.

No change is needed to `burl/harness/tool_loop_native.py` or
`burl/harness/agent_runner_native.py` for a minimal-viable hookup.  Adding
the `extra_user_messages` kwarg to `NativeHarness.run` is a 3-line change
if we want the cleanest integration.

## Open questions (resolve before iter-2 rollout launch)

1. **Retry budget (`max_gate_retries`).**  Proposal: default 1 (one nudge,
   one retry).  The higher we set this, the more compute per gated
   decision and the more we dilute the "self-corrected on first nudge"
   signal.  But 1 may be too tight — Gemma on iter-1 sometimes needs a
   second exploration pass.  Suggest running the first 10 decisions at
   `max_gate_retries=2` and looking at how many self-corrections happen at
   `gi=1` vs `gi=2`.  If the marginal second retry converts <10% more
   decisions, lock to 1.

2. **Default feedback variant.**  Proposal: **`tool-nudge`** for the
   headline iter-2 corpus.  Reasons: (a) directly targets
   `eq_outcome_distribution` suppression, which is the specific
   regression iter-0/iter-1 could not fix; (b) aligns with Burl's
   distribution-shape-reasoning thesis; (c) easiest to measure — we can
   look at tool histograms pre-/post-gate to see if it worked.  A/B with
   `minimal` on 10 decisions before committing to the full N=50 run.

3. **Stubborn-trace policy.**  Three options:

   - **Drop (proposed default).**  Cleanest corpus; throws away signal.
   - **Keep as labeled negatives.**  Tag `source=eq_gate_stubborn`; train
     with a label the loss can down-weight.  Requires corpus schema
     change and SFT recipe change — not justified for iter-2.
   - **Keep but only the *first-turn* tool-calls, discarding the commit.**
     Teaches tool usage without teaching the wrong decision.  Elegant in
     theory; in practice truncating mid-trace breaks the chat-format
     round-trip the SFT recipe expects.  Not worth the tooling cost for
     iter-2.

   **Go with Drop.**  Revisit if the gated corpus is too small to train
   on (`self_corrected < ~10` at N=50, which would mean gate fired on ≤20
   decisions and fewer than half self-corrected — unlikely but possible).

4. **What if the gate itself destabilizes commit discipline?**  Iter-1's
   failure mode was commit-chain exhaustion.  The gate adds a round-trip
   that can plausibly make this worse.  Mitigation: re-use the spike v2
   `max_retries=7` budget and reuse the commit-instruction nudge already
   in `_COMMIT_INSTRUCTION`.  If gate-driven decisions exhaust at >20%,
   pause and diagnose before shipping the corpus.

5. **Should `eq_epsilon` be non-zero?**  Near-tie decisions (|eq_delta| <
   0.5) are dominated by sampling noise in the E[Q] framework itself.
   Gating them wastes compute on decisions where there is nothing to
   teach.  Proposal: `eq_epsilon=0.25` for iter-2.  Cheap to adjust; the
   function already accepts it as a kwarg.
