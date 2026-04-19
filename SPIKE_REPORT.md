# R3 Spike — Gemma 4 Native Tool-Use

**Date**: 2026-04-19
**Owner**: native-toolcall-spike (worktree)
**Budget**: ~$0.10 / $1.00 (well under cap)
**Headline**: **Mixed — one true positive, one fixable regression.** Native
format unlocks exactly the tool-use breadth we wanted, but the current
`<commit>INT</commit>` tag doesn't play nice with the native protocol and
burns 40% of decisions in retry-exhaustion. Recommend Move 4 build on this
spike, but swap the commit mechanism before re-grading.

## TL;DR

| metric | Move 3 (XML) | Move 4 spike (native) | delta |
|---|---|---|---|
| n_completed | 10 / 10 | **6 / 10** | **−4** (retry-exhaustion) |
| first_legal_rate | 100% | 60% | −40pp |
| legal_rate (among completed) | 100% | 100% | 0 |
| bot_match_rate (among completed) | 60% | **66.7%** | **+6.7pp** |
| p_eq_geq_bot (among completed) | 70% | 66.7% | −3.3pp |
| tool-use breadth | `{is_legal: 10, play: 8*}` | `{is_legal: 22, eq_outcome_distribution: 10, is_trump: 2}` | **+2 real tools** |
| mean_eq_delta | −4.65 | −4.05 (smaller sample) | — |
| wall time | 412s | 161s | much faster |
| Modal spend | $0.09 | **$0.04** | cheaper |

`*` In Move 3, `play: 8` was Gemma hallucinating a `play(domino_id)` tool
that doesn't exist. The native path does not hallucinate tool names — the
schema is rendered authoritatively by the chat template.

## What worked — the fruit

### 1. Native `<|tool_call>` shape emits cleanly (10/10 decisions)

vLLM's detokenizer preserves Gemma's special tokens when we pass
`SamplingParams(skip_special_tokens=False)`. Every raw completion contained
one or more canonical envelopes of the form

```
<|tool_call>call:NAME{arg:value,...}<tool_call|>
```

The warmup probe on a trivial `add(17, 25)` emitted 17 tokens total — zero
thinking channel — and returned the exact native form. Scaled up to the full
Burl schema (7 tools, ~1.3 KB of rendered `<|tool>…<tool|>` declarations),
the shape stayed stable.

### 2. Tool-use breadth achieved — the whole point of the spike

Move 3 tool histogram: `{is_legal: 10, play: 8}`. The `play` entry was
Gemma inventing a function name that isn't in the registry (10/10 decisions
asked `is_legal`; never called a distribution primitive).

Move 4 spike tool histogram: `{is_legal: 22, eq_outcome_distribution: 10,
is_trump: 2}`. **10/10 decisions called `eq_outcome_distribution`** — the
exact distribution primitive Burl was designed to exploit. On the very first
Burl-shaped smoke probe (before the full run), Gemma picked
`eq_outcome_distribution` as its first tool call over `is_legal` or
`trump_declared`.

The mechanism is that apply_chat_template renders our JSON schemas into
Gemma's trained shape:

```
<|tool>declaration:eq_outcome_distribution{description:<|"|>…<|"|>,
  parameters:{properties:{play:{type:<|"|>INTEGER<|"|>},
  n_samples:{type:<|"|>INTEGER<|"|>}},required:[<|"|>play<|"|>],…}}<tool|>
```

Gemma is trained on this shape, so it actually reads the menu.

### 3. Legal rate holds at 100% on completed decisions

Zero illegal commits on the 6 decisions that produced a final play. The XML
retry loop works unchanged against the native path — the only Gemma-facing
change is the prompt and parser.

### 4. Faster + cheaper

Native path averaged 16 s per decision (excluding cold start) vs. Move 3's
41 s/decision. Fewer tokens wasted on a thinking preamble — Gemma goes
straight to tool calls. N=10 run cost $0.04 vs. $0.09 for the XML path.

## What broke — the regression

### 40% retry-exhaustion (4 / 10): Gemma never emits `<commit>INT</commit>`

All four exhaustions share the same failure shape: Gemma calls a few tools,
gets the observations back, then calls more tools — and keeps chaining. It
never converges to `<commit>17</commit>`.

Raw completion from decision 7 turn 3 (representative):

```
I will commit to playing 15. <|tool_call>call:is_legal{domino_id:15}<tool_call|>
```

The prose says "I will commit to playing 15" — but the structured output
is a `<|tool_call>` for `is_legal`. Gemma's post-training reflex is: output
goes in a tool call envelope. Our Burl-custom `<commit>` tag has no trained
prior, so the model substitutes the shape it DOES have a prior for — a tool
call.

This is isomorphic to Gemma's "if in doubt, wrap it in a tool call" behavior.
The XML path avoided it by putting `<tool>` and `<commit>` in the same XML
family — both felt foreign, neither was privileged over the other. The
native path privileges tool calls and the commit tag gets crowded out.

### Secondary: hallucinated distribution numbers (decision 1, representative)

Gemma called `eq_outcome_distribution` for both legal plays (14, 21). The
tool returned `mean: −5.4, p_make: 1.0` for 14 and `mean: +10.3, p_make:
1.0` for 21. In turn 3 reasoning, Gemma wrote:

> Playing 14: mean: 0.123, p_make: 0.450
> Playing 21: mean: 0.088, p_make: 0.380

Those numbers are fabricated — they match neither the tool return nor each
other. Gemma then committed to 14 "because 0.450 > 0.380," losing the
decision.

This is not a protocol bug — it's a base-model reading-comprehension bug
that STaR-style SFT is designed to fix. But it confirms the hypothesis that
getting the format right is a necessary but not sufficient condition.

## Evidence artifacts

- Endpoint: `burl/modal/gemma_serve_native.py` (Modal app `burl-gemma-serve-native`)
- Runner: `burl/harness/agent_runner_native.py`
- Parser / loop: `burl/harness/tool_loop_native.py` (self-test passes; handles
  4 surface shapes for the tool-call envelope)
- Grader: `burl/eval/run_move4_spike.py`
- N=10 results: `burl/eval/results/move4_spike_native_n10/{traces.jsonl,
  summary.json,report.md}`
- Live log: `burl/eval/results/move4_spike_native_n10.log`
- Debug-one (full raw dump for decision 0): captured inline in the session
  transcript — shows Gemma using 3 tools → commit path that worked.

## Budget accounting

| item | cost |
|---|---|
| Warmup (cold start + 1 generate) | ~$0.02 |
| Smoke (cold start + 1 generate) | ~$0.02 |
| Debug-one (cold start + 4-turn decision) | ~$0.02 |
| N=10 full run | $0.04 |
| **Total** | **~$0.10** |

Well inside the $1.00 cap — plenty of headroom for follow-up iterations.

## Recommendation for Move 4

**Merge the spike, then change one thing before re-grading.** The native
format is clearly the right long-term path — we can see Gemma reaching for
distribution primitives zero-shot, which is the whole premise of Burl.
Throwing this out to stay on the XML path forfeits that.

The fix for the 40% retry-exhaustion is to stop asking Gemma to emit a
non-native tag for the final answer. Two choices in order of preference:

1. **Promote commit to a native tool call.** Add a `commit_play(domino_id)`
   tool to the schema. When the parser sees it, treat it as the final
   answer and exit the loop. Gemma already knows how to emit tool calls —
   we just consume one of them as the terminal action. Concretely:
   `<|tool_call>call:commit_play{domino_id:17}<tool_call|>` → trace.final_play = 17.
   This keeps STaR data in the trained distribution.

2. **Post-N-turns forced commit.** After N turns of tool-only calls, append
   a user message `The game clock is ticking — emit <commit>INT</commit>
   now.` This is uglier but doesn't require a schema change and can be
   stacked with option 1 as a safety net.

Option 1 is the right structural fix and almost certainly brings the
completion rate back to 10/10 — every exhaustion in this spike had Gemma
already deciding on a play ("I will commit to 15") then emitting a tool
call instead of the commit tag. Give it a tool that matches its reflex.

Expected cost of the follow-up: $0.05–0.10 (one short re-run at N=10).

If that re-run shows completion rate back above 90% *and* tool-use breadth
preserved, native is the Move 4 baseline and XML can be deleted.

## Four-box scorecard (from the assignment)

| box | status | note |
|---|---|---|
| `<|tool_call>` syntax emits on ≥8/10 | **✅ 10/10** | zero-shot, clean |
| Tool-use reaches beyond `is_legal` | **✅** | `eq_outcome_distribution` 10×, `is_trump` 2× |
| Legal rate stays 100% | **✅ on completed** | ❌ if counted as 60% incl. exhaustions |
| K1 doesn't collapse below 70% | **⚠️ 66.7% on 6 completed**, not directly comparable |

Two boxes tick unambiguously. Two tick only if we fix the commit mechanism.
The spike answers the question it was designed to answer: **R3 is the right
long-term direction, and the one blocker is mechanical (tag choice), not
fundamental (model can't handle native shape).**

---

## Follow-up after commit_play fix (2026-04-19)

**Headline: all four boxes green. 9/10 completed, 100% legal, 88.9% bot match,
breadth preserved.** The original recommendation (option 1 — promote commit to
a native tool) is the right call and the re-run confirms it.

### The change

Added `commit_play(domino_id: int)` to `TOOL_SCHEMAS`. The native parser in
`tool_loop_native.py` sieves `<|tool_call>call:commit_play{domino_id:N}<tool_call|>`
out of the envelope list before dispatching to the registry, sets
`trace.final_play`, runs it through the existing legality check, and exits
the turn on success. XML `<commit>INT</commit>` regex removed — native path
speaks one language only. System prompt and the "you forgot to emit anything"
nudge now reference the tool, not the tag.

### One twist — retry budget was a second, hidden blocker

The first re-run with the commit_play fix came back WORSE: 8/9 exhausted, all
with `turns=4`. Trace inspection showed the pattern

```
[['is_legal'], ['is_legal'], ['eq_outcome_distribution'], ['eq_outcome_distribution']]
committed=[None, None, None, None]
```

Gemma was chaining tools, never committing, and running out of turns. The
cap was `max_retries+1 = 4` in `retry.py:retry_on_illegal` — a budget the
XML path had gotten away with because Gemma sometimes packed multiple
`<tool>` tags into one completion, giving it 2-3 tool effects per turn. In
the native format it emits exactly one `<|tool_call>` envelope per turn, so
4 turns = 4 tools and zero room for a commit.

Fix: rerun with `--max-retries 7` (matches `max_turns=8`). No file change —
the script already exposed the flag. This is worth recording because it's
the sort of thing that would bite Move 4 in production: **native Gemma is
one tool call per turn, so turn budgets need to sum tools + commit + retry,
not just retries.**

### Results — N=10, `--max-retries 7`, cost-cap $0.60

| metric | Move 3 (XML) | Move 4 spike v1 (native) | **Move 4 spike v2 (native + commit_play)** |
|---|---|---|---|
| n_completed | 10/10 | 6/10 | **9/10** |
| legal_rate | 100% | 100% (completed) | **100%** |
| first_legal_rate | 100% | 60% | **90%** |
| bot_match_rate | 60% | 66.7% (completed) | **88.9%** |
| p_eq_geq_bot | 70% | 66.7% (completed) | **88.9%** |
| mean_eq_delta | −4.65 | −4.05 | **−1.92** |
| tool histogram | `{is_legal:10, play:8*}` | `{is_legal:22, eq_outcome_distribution:10, is_trump:2}` | `{is_legal:20, eq_outcome_distribution:15, trump_declared:9}` |
| wall time | 412s | 161s | **246s** |
| Modal spend | $0.09 | $0.04 | **$0.055** |

`*` Move 3's `play` count was hallucinated tool name.

### Why I think these numbers are real, not noise

- Bot-match rate jumped 60% → 88.9% on the same seed/decision set, driven
  by Gemma actually *using* `eq_outcome_distribution` (15×, up from 10×) and
  `trump_declared` (9×, up from 0 meaningful uses on XML). More tool usage
  correlated with better picks.
- The one remaining exhaustion (decision 4: `turns=8 tools=8 committed=None`)
  is Gemma chaining genuinely too many tools, not failing to emit a commit —
  so this is a policy issue, not a format issue. STaR SFT can plausibly
  shorten the exploration.
- Legal rate is the strongest signal that `commit_play` lands cleanly —
  zero illegal commits across 9 completed decisions, no retries needed.

### Four-box scorecard — final

| box | status | note |
|---|---|---|
| `<|tool_call>` syntax emits on ≥8/10 | **✅ 10/10** | unchanged from v1 |
| Tool-use reaches beyond `is_legal` | **✅** | `eq_outcome_distribution:15, trump_declared:9` |
| Legal rate stays 100% | **✅ 100%** | on 9/10 completed, zero illegal commits |
| K1 doesn't collapse below 70% | **✅ 88.9%** | +18.9pp over Move 3 baseline |

### Recommendation for Move 4 — unchanged, now with receipts

Native is the Move 4 baseline. `commit_play` is the right terminal-action
mechanism. The one code-level carry-forward is that **the Move 4 runner
needs `max_retries` sized for "tools + commit + illegal retries," not just
"illegal retries"** — default this to something like 7 in the production
config so the tool-call-chain pattern has room to breathe.

Total session cost: ~$0.16 of the $1.00 cap.

---

## Layer 1: 42-aware prompt framing + primer (2026-04-19)

**Headline: reasoning substrate flipped in 42 vocabulary (0 → 5-11 mentions/trace
across partner/team/count/bid/offense-defense), but bot-match regressed 88.9% →
70% because the enriched prompt displaced E[Q] distribution tool use.** Modest
positive on the substrate (the point of the task), soft regression on the
scoreboard. The right substrate for STaR; not the right artifact for live play.

### What changed

Two enrichments bundled into one experiment:

1. **42-aware framing block** prepended to the system prompt — partnership
   (teams/partner seat/opp seats), bidder + target bid, offense vs defense
   role, running `team_points` score, count dominoes still in play (5-pt and
   10-pt lists with ids + labels), trump-suit membership (all 7 trump ids
   with in-hand / played / unseen breakdown). Notrump and doubles-suit get
   explicit "what's special" prose.
2. **LEM rules primer** prepended verbatim from `lem/rules/primer.md` — 1,549
   words of engine-verified Texas 42 rules (suits, led-suit rule, follow
   rules, trick resolution, count dominoes, contract satisfaction).

Both land in the system message; the user message is unchanged.

Only `burl/harness/agent_runner_native.py` was edited. Helpers
(`_trumps_under_declaration`, `_count_dominoes_remaining`,
`_render_42_framing`) live in that same file. Primer loaded once at import.
Self-test extended to verify the prompt renders with the expected vocabulary
tokens.

### Results — N=10, `--max-retries 7`, cost-cap $0.30

| metric | spike v2 (native + commit_play) | **Layer 1 (framing + primer)** |
|---|---|---|
| n_completed | 9/10 | **10/10** |
| legal_rate | 100% | **100%** |
| first_legal_rate | 90% | **100%** |
| bot_match_rate | 88.9% | **70.0%** (−18.9pp) |
| mean_eq_delta | −1.92 | **−3.00** |
| p_eq_geq_bot | 88.9% | **70.0%** |
| empty_tool_rollout_rate | 0% | **20%** (2/10 decisions with zero tool calls) |
| tool histogram | `{is_legal:20, eq_outcome_distribution:15, trump_declared:9}` | `{is_legal:13, eq_outcome_distribution:2, is_trump:2}` |
| mean_tokens_in (chars) | 13,564 | 35,323 (2.6× from primer) |
| mean_tokens_out (chars) | 1,479 | 5,517 (3.7× — longer thoughts) |
| wall time | 246 s | 683 s (incl. 167 s cold start on decision 1) |
| Modal spend | $0.055 | $0.152 |

### The reasoning substrate genuinely flipped

42-vocabulary mentions per trace — both runs, same 10 decisions:

| term | spike v2 | Layer 1 |
|---|---|---|
| `partner` | 0.0/trace | **5.2/trace** |
| `team` | 0.0/trace | **11.4/trace** |
| `offense`/`defense` | 0.0/trace | **2.2/trace** |
| `count` | 0.0/trace | **8.4/trace** |
| `bid` | 0.0/trace | **3.0/trace** |
| `trump` | 5.6/trace | **50.2/trace** |
| `set them`/`set the` | 0.0/trace | 0.4/trace |

The spike v2 traces never once used "partner", "team", "bid", "offense", or
"defense". Layer 1 uses them every trace. This is the success criterion the
task was designed to test, and it passes unambiguously.

### Representative trace excerpts (reasoning depth, not numbers)

**Decision 1, match — 42 framing shows up verbatim in Gemma's thought.**
(seed=900000 decl=0 narrator=1; burl=21 bot=21 delta=0.0)

> **My Position:** I am at seat 1 (Team 1). My partner is at seat 3 (Team 1).
> Opponents are at seat 0 (Team 0, plays before me) and seat 2 (Team 0,
> plays after me).
> **Role:** I am on DEFENSE. Team 0 (bidder) bid 30 count. I need to SET
> them (keep them below 30 count).

Note: v2 on this exact decision picked 14 and lost 17.3 points; Layer 1 picks
21 and matches the bot. Framing may have carried this one.

**Decision 4, loss — partnership + score awareness is visible but no E[Q]
call.** (seed=900000 decl=2 narrator=0; burl=24 bot=22 delta=−1.4)

> **Role:** I am at seat 0 (Team 0). My partner is at seat 2.
> **Contract:** Team 0 bid 30 count. We need at least 30 count points.
> **Score Status:** Team 0 has captured 29 count points; Team 1 has captured
> 1 count point. We are very close to making the bid (need 1 more count
> point to make 30).

This is *exactly* the situation `eq_outcome_distribution` was built for — a
1-point-from-contract offense pick, only 2 legal plays, trump is twos.
Gemma surfaces the team state correctly, then commits without calling the
distribution tool. Small regression (−1.4), but the shape is the story.

**Decision 6, large miss — 2 is_legal calls and commit, no distribution
check.** (seed=900000 decl=2 narrator=3; burl=27 bot=9 delta=**−24.8**)

eq_gap was 24.81 — the bot pick was far ahead of the second-best legal play.
Burl picked the *worst* of its legal options with two `is_legal` checks and
zero probability reasoning. In v2 this decision category drove
`eq_outcome_distribution` calls; in Layer 1 the primer's rule text appears
to have satisfied Gemma's "do I know enough?" heuristic.

### Interpretation — the trade-off is real and legible

The primer + framing make Gemma confident enough to skip the belief tool.
v2's tool histogram averaged 4.4 calls per decision with `eq_outcome_distribution`
on 15/9 completions (1.67/trace). Layer 1 averages 1.7 calls per decision
with `eq_outcome_distribution` on 2/10 (0.2/trace). The model is reading
the rules, reading the 42 framing, and deciding it has enough signal.

On decisions with a clear play (matches), this is fine — both runs agree.
On decisions with a close-call or counterintuitive answer (misses), losing
the distribution tool costs real bot-match points. The `mean_eq_delta`
moved from −1.92 to −3.00, and the big-miss decision (6) shows the failure
mode clearly.

Interestingly, the primer also **eliminated the one retry-exhaustion** v2
had (decision 4), because Gemma now commits faster instead of chaining
tools. That's why completion rate went 9/10 → 10/10.

### Verdict — does the prompt upgrade earn its keep?

**For STaR rationalization training: yes.** Rationalizations that reference
partner, count dominoes, bid target, and offense/defense now occur natively
— that's the teaching signal the task was set up to improve. v2 traces
simply had nothing 42-shaped to rationalize.

**For raw bot-match on the spike set: no — STaR is still the lever.** The
framing substrate isn't enough alone to beat the v2 numbers, and the primer
crowds out the exact tool that gave v2 its bot-match boost. The 70% number
is worth sitting with: it's still +10pp over Move 3 XML (60%), so Layer 1
isn't *regressing relative to pre-spike baseline* — it just gives back the
tool-use gains the native path unlocked.

**Recommended carry-forward into Move 4:**

1. **Keep the 42-aware framing block** — it's cheap, it worked, and it's the
   right scaffold for STaR to latch onto.
2. **Reconsider the primer's placement.** Candidates: (a) drop it to see if
   framing alone recovers the tool-use, (b) trim the primer to a one-paragraph
   "trump shapes the deck; count dominoes are (5-5, 6-4, 5-0, 4-1, 3-2); bidder
   must make the contract" summary instead of the full 1.5 KW, (c) add a
   one-line nudge after the primer that says "the rules above are reference —
   still call `eq_outcome_distribution` on close calls."
3. **The real experiment**: run STaR on *these* traces (even with the 70% bot
   match — K1 filter keeps the matches, rationalizations now carry 42
   vocabulary, and the corpus will be cheap).

### Budget

| item | cost |
|---|---|
| Failed first kickoff (killed after decision 1) | $0.01 |
| N=10 full run with primer | $0.15 |
| **Total** | **~$0.16** |

Well under the $0.30 cap for this task. Cumulative spike spend to date
(including pre-Layer-1 work): ~$0.32.
