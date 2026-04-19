# T14 — Opus 4.7 reference-trace spike: PARTIAL, halted at d06

**Status**: Run halted at 6 decisions (3/6 final_play=-1) due to a second
SDK/MCP failure mode the lock fix (`2830be0`) does not cover. Scientific
question is **partially answered** from the 3 clean decisions. Full N=30
at this model requires the HTTP MCP transport path.

## TL;DR

- **Lock fix is necessary but insufficient.** Validated on d01 where 8
  parallel `tool_use` blocks (2× `eq_outcome_distribution` + 6×
  `void_audit`) landed cleanly. Pre-fix, that batch would have
  guaranteed-wedged the channel.
- **Second failure mode exists.** Independently of parallelism, the first
  tool call of a session stream-closes roughly 50% of the time. When it
  does, the channel is dead for the rest of the decision and the model
  retries commit_play 3–10 times against a closed pipe, landing
  `final_play=-1`. Likely the long pre-tool assistant-text stream
  (~500 chars of Opus reasoning) stalls the CLI↔MCP stdio pipe past some
  timeout.
- **Partial scientific signal is strong.** Across the 3 clean decisions,
  Opus reached for `eq_outcome_distribution` on every non-trivial choice
  (d00, d01 — both had eq_gap > 17). It skipped the tool on the
  trivially-wide decision (d04, eq_gap 27.88). **Zero** calls to
  `conditional_outcome` in 3 clean + 3 broken sessions — Opus, like
  Haiku, does not reach for the counterfactual tool.

## TL;DR for the Burl training question

If the only ceiling-model signal we needed was "does the frontier model
engage with distribution shape when given the tools?", **yes** — Opus
does. `eq_outcome_distribution` was called on 2/3 clean decisions (the
two with eq_gap in the 17-18 range), and Opus explicitly compared means
in commentary ("much higher expected value +12.3 vs −6.5").

The open, unaddressed signal is `conditional_outcome` (counterfactual
probes). Neither Haiku (T8, 0/30) nor Opus (T14 partial, 0/6) reached
for it even once. Five decisions across two model classes now consistent.
**Read**: the tool surface, as currently framed, does not cue either
model to reach for counterfactuals. This is tool-surface feedback, not
a training-corpus problem.

## Run config

- Model: `claude-opus-4-7`
- Thinking: `max_thinking_tokens=16000`
- Subset: first 10 of `move3_decisions.jsonl` + first 20 of
  `move4_decisions_n50.jsonl` (same as T8 Haiku).
- Lock fix: `2830be0` live (verified: `Query._handle_sdk_mcp_request ==
  _locked_handle_sdk_mcp`).
- Halted at d05 (d06 started but didn't complete). Cumulative cost
  through d05: **$4.57**. Wall: ~10 min (would have been ~45 min at that
  rate, at ~50% usable return).

## The 6 decisions

| d | source | legal | bot | eq_gap | opus | match | tools | errs | notes |
|---|---|---|---|---|---|---|---|---|---|
| 00 | move3 | 14, 21 | 21 | 17.28 | 21 | ✓ | 5 | 0 | is_legal×2 + eq_dist×2 + commit |
| 01 | move3 | 15, 23 | 23 | 17.66 | 23 | ✓ | 9 | 0 | **8 parallel tools** → lock fix catch; eq_dist×2 + void_audit×6 + commit |
| 02 | move3 | 15, 23 | 15 | 1.86 | -1 | ✗ | 6 | 6 | 2 parallel eq_dist @ n=40 both fail |
| 03 | move3 | 22, 24 | 22 | 1.40 | -1 | ✗ | 11 | 11 | lone is_legal fails first; 10 commit retries all fail |
| 04 | move3 | 4, 20 | 4 | 27.88 | 4 | ✓ | 1 | 0 | direct commit, no probe (wide gap) |
| 05 | move3 | 9, 27 | 9 | 24.81 | -1 | ✗ | 4 | 4 | lone is_legal fails first; 3 commit retries all fail |

**Clean rate**: 3/6. **Bot-match rate on clean**: 3/3 (d00, d01, d04).

## Scientific observations from the 3 clean decisions

### Opus engages with `eq_outcome_distribution` on narrow-to-medium gaps

- **d00** (eq_gap 17.28): Opus called `eq_outcome_distribution` for BOTH
  legal plays. Made a decision based on expected value.
- **d01** (eq_gap 17.66): Opus called `eq_outcome_distribution` for both
  legal plays **plus** 6× `void_audit` to resolve opponent hand
  distributions, then committed. Post-tool commentary explicitly cited
  E[Q]: "much higher expected value (+12.3 vs −6.5)". This is the
  cleanest "Opus reaches for distribution shape" data point we have.
- **d04** (eq_gap 27.88): Opus called `commit_play` directly, no probe.
  Reasonable — the pre-tool text already identified the dominating play
  ("2-1 is the highest remaining trump"). A wide-gap decision doesn't
  need E[Q] to resolve.

**Read**: Opus's default reach is conservative on wide-gap decisions, but
it does engage the distribution tool when the gap is narrow enough to
matter. This is the behavior we want Burl (Gemma) to learn.

### `conditional_outcome` never called (N=6, including the broken ones)

Across all 6 decisions (clean + broken), zero calls to
`conditional_outcome`. Consistent with Haiku's 0/30 in T8. **Cumulative
across two model classes: 0 calls in 36 decisions.**

Given that both the Haiku and Opus defaults never reach for this tool,
the framing is doing something wrong. Candidates: tool name is opaque;
tool signature (`assumption: {"player": abs_seat, "holds": dom_id}`) is
intimidating; system prompt doesn't cue "use this to probe counterfactual
worlds." Recommended: redesign the tool surface before iter-4 rollouts
trust it to be learned from corpus alone.

### Pre-action thinking vs post-tool commentary

- Thinking content came back **empty string** on every turn (all 6 decisions
  — thinking_blocks count 1–3 each, each with `chars=0`). Separate known
  SDK bug; commit message for `2830be0` notes it remains unfixed.
- Pre-tool assistant text averaged ~500 chars, content-rich: Opus lays
  out the game situation ("Looking at this position: Partner is winning,
  my hand has 6-0, all trumps except ones are out...") before issuing
  any tool calls. This is substantively different from Haiku, which
  tended to probe first and comment second.
- Post-tool commentary on clean decisions is brief — 1–2 sentences that
  explicitly cite returned E[Q] values. ("much higher expected value
  +12.3 vs −6.5").

**Read**: Opus's shape is "think + commit" with targeted tool use in the
middle, versus Haiku's "probe widely, synthesize at end". Both are
reasonable; they'd produce very different training corpora for iter-4.

## Failure mode analysis

### Mode 1 — parallel tool_use race (lock fix covers)

Pre-fix: Opus emits multiple `tool_use` blocks in one assistant message,
the SDK spawns a concurrent handler task per block, and the MCP
low-level server's shared tool cache races. Visible as `"Stream closed"`
on every slot in the parallel batch, then channel-dead for the rest of
the session.

Post-fix (`2830be0`): `_Query._handle_sdk_mcp_request` wrapped with a
process-global `asyncio.Lock`. Only one handler runs at a time.

**Confirmation**: d01 had 8 parallel tool_use blocks (2× eq_dist + 6×
void_audit) and all 8 returned successfully. This is unambiguous
validation of the lock fix.

### Mode 2 — lone-first-call stream-close (NOT covered by lock)

Observed on d03, d05: the very first tool call in the session is a
**lone** `is_legal(dom_id)` — no parallelism at all — and it returns
`{"type":"text","text":"Stream closed"}` with `is_error=true`. Opus
sees the error, retries commit_play 3–10 times, every retry hits the
same dead channel.

**Characterization**:
- d03 pre-text: 532 chars
- d05 pre-text: 535 chars
- d02 first call: 2-parallel eq_dist @ n_samples=40 (long-running handler)
- d00 pre-text: shorter (Opus committed direct on d04 which had 571)

No single feature cleanly predicts failure, but a working hypothesis is
**timing**: the combination of long assistant text streaming + the first
handshake of the MCP tool channel sometimes exceeds a stdio buffer or
idle timeout in the CLI↔MCP pipe. Once closed, no recovery within the
session.

The lock fix does not help here because there's no concurrency to
serialize — the channel is simply dead at first tool dispatch.

## Recommendation: HTTP MCP transport

The other worktree agent's HTTP MCP transport path sidesteps BOTH
failure modes:
1. HTTP is request/response, no persistent stdio pipe to go stale.
2. Parallel tool calls become parallel HTTP requests, no shared handler
   state to race.

**Estimated effort**: ~60 min to productize `scratch/mcp_http_*` into
`burl/haiku_spike/agent_http.py` + corresponding runner. Lands an
alternate `run_decision_haiku_http` with the same interface so
`run_opus.py` can swap it in one line.

**Alternative if HTTP is too invasive**: pivot T14 to `claude-sonnet-4-6`
as a near-ceiling substitute — Sonnet's tool-use parallelism pattern and
pre-tool text length distribution may differ and might not hit Mode 2 as
frequently. Lower effort (just change `MODEL`) but no guarantee.

## Artifacts

- `burl/haiku_spike/agent.py` — `model=` + `thinking_tokens=` kwargs +
  import-time MCP dispatch lock (from `2830be0`). Keep regardless.
- `burl/haiku_spike/run_opus.py` — Opus runner at thinking=16k. Keep;
  ready to rerun against `agent_http.py` if that path ships.
- `scratch/burl_p5_iter2_prep/opus_traces/full_d{00..05}.jsonl` — 6
  decision traces (3 clean, 3 broken). The 3 clean are the scientific
  material; the 3 broken are the failure-mode evidence.
- `scratch/burl_p5_iter2_prep/opus_run.log` — run log.
- `scratch/burl_p5_iter2_prep/opus_ping.py`, `opus_diag.py` — diagnostic
  scripts, retain for future debugging sessions.
- `scratch/burl_p5_iter2_prep/opus_ping_d2_postlock_t32000.jsonl` —
  post-lock probe on d2 at 32k thinking, parallel eq_dist succeeded.
  Useful comparison when debugging Mode 2.
- `scratch/burl_p5_iter2_prep/opus_ping_d0_think8k_longsession.jsonl` —
  pre-lock probe showing the Mode 1 failure (parallel batch all fail,
  session poisoned).

## Status for closeout

Paused per team-lead's direction. Writeup captures the scientific
signal we DO have + the precise failure mode blocking the full N=30.
User will decide whether HTTP MCP transport rewiring is worth the
effort given iter-4-thoughts is providing parallel signal on the Burl
side.
