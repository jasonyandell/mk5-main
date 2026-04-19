# T8 — Haiku 4.5 full reference-trace run (N=30)

**Status:** completed. 29/30 decisions landed; 1 (d11) hit the 10-turn cap before
`commit_play`. Total cost **$0.7764** (35% under the $1.20 cap, tracks the
$0.87 smoke-based projection within 11%).

## TL;DR

- 10 of `move3_decisions.jsonl` (overlaps iter-0/iter-1 eval) + 20 of
  `move4_decisions_n50.jsonl` (STaR set) = 30 total; subset unchanged from
  team-lead's proposal.
- **Bot-match: 72.41% (21/29 completed).** On `move3` it's 80% (8/10);
  on `move4` it's 68.42% (13/19 completed). The `move4` rows are harder —
  they're the same distribution the STaR rollouts sample from, so this
  establishes the bot-match ceiling we'd hope Gemma approaches.
- **Mean E[Q] delta vs bot: −3.50** (dragged down by 4 big-gap misses).
  **P(Haiku E[Q] ≥ bot−0.5): 75.86%** — i.e. 22/29 decisions are within
  half a Q-point of optimal, so most "mismatches" are close calls.
- `conditional_outcome` **was never called** across all 30 decisions.
  Even on decisions with E[Q] gaps of 17–25 points, Haiku does not reach
  for the counterfactual tool. This is the single strongest training
  signal for Gemma STaR: the tool is load-bearing in Burl's design but
  even a strong tool-user skips it.

## Results

### Aggregate

| metric                      | value             |
|-----------------------------|-------------------|
| n_attempted                 | 30                |
| n_completed                 | 29                |
| total_cost_usd              | $0.7764           |
| total_wall_seconds          | 469.3s (~7m49s)   |
| bot_match_rate              | 72.41% (21/29)    |
| mean_eq_delta               | −3.50             |
| p_eq_geq_bot_minus_0.5      | 75.86% (22/29)    |
| mean_tool_calls/decision    | 6.9               |
| mean_turns/decision         | 7.9               |
| mean_cost/decision          | $0.0268           |

### By source

| source | n  | match  | mean_eq_delta |
|--------|----|--------|---------------|
| move3  | 10 | 80.0%  | −1.58         |
| move4  | 19 | 68.4%  | −4.51         |

Move 4 is harder. The STaR-rollout distribution has tighter decision
windows (narrator options already partially constrained by earlier plays),
and Haiku's skip-E[Q] failure mode (see below) bites harder there.

### Tool histogram (all 29 completed)

| tool                    | calls | per-decision |
|-------------------------|-------|--------------|
| is_legal                | 48    | 1.66         |
| is_trump                | 31    | 1.07         |
| eq_outcome_distribution | 30    | 1.03         |
| commit_play             | 29    | 1.00         |
| trump_declared          | 23    | 0.79         |
| unseen                  | 19    | 0.66         |
| void_audit              | 19    | 0.66         |
| **conditional_outcome** | **0** | **0.00**     |

`is_legal` at ~1.7/decision confirms Haiku self-checks both candidates
before committing — the "defensive call" pattern noted in the smoke writeup.

## Failure modes (8 misses on completed + 1 d11 incomplete)

### The zero-E[Q] cluster (4/8 misses)

Misses on d15, d18, d19, d25 all share one feature: **Haiku did not call
`eq_outcome_distribution` on either candidate.** Gaps were 24.4, 4.4,
21.6, 25.2 — three of them >20 E[Q] points. These are not close calls
and they are not Monte Carlo noise. They're decisions where Haiku
prose-reasoned its way to a wrong play, skipping the one tool that
would have forced a numerical tie-break.

Examples:

- **d15** (declaration=twos, gap=24.4): Haiku reasoned "I must follow
  suit since trump was led and I hold trump. Playing 23(6-2) is
  mandatory." — a misread of twos-trump following-suit mechanics.
  Picked the wrong candidate without probing E[Q].
- **d19** (gap=21.6): only 4 tool calls (`trump_declared`, 2×`is_legal`,
  `commit_play`) — a premature commit on surface reasoning.
- **d25** (gap=25.2): Haiku reasoned "we're already decisively ahead"
  and sluffed, without checking whether the alternative wins count.

Takeaway: strong prose-reasoning does not substitute for the E[Q] tool
even at Haiku scale. A nudge in the system prompt ("if two plays look
reasonable, probe both with `eq_outcome_distribution`") would likely
recover 2–4 of these.

### Oracle-sampling divergence (1 miss, d01)

d01 is the most interesting failure. Haiku **did** call
`eq_outcome_distribution` with n_samples=100 on both candidates and got:

- play 15: mean=−1.35, p_make=0.74
- play 23: mean= 7.04, p_make=0.63

Haiku chose 15 on higher p_make. Dataset truth (from the bot's rollouts):
per_play_eq[15]=3.55, per_play_eq[23]=**21.22**.

The oracle's 100-sample estimate for play 23 (mean=7.04) is off by 14 Q
points from ground truth (21.22). This isn't Haiku's fault — it's the
tool's sample-size limit. The model made a reasonable choice given noisy
inputs. Implication: for large-gap decisions, the EQ tool's default
n=10 (or even 100) is under-powered, and Haiku has no way to know it
should bump n higher.

### Tight close-call flips (3/8 misses)

d02 (gap=1.86), d14 (gap=3.84), d18 (gap=4.40), d21 (gap=6.21) are
within-noise. The d02 miss was already seen in the smoke; bumping
n_samples=100 there did not disambiguate a 1.86-point gap, which is
below the gap floor in the dataset's own noise envelope. These are
essentially ties.

### Turn-cap timeout (d11)

Haiku hit `max_turns=10` without emitting `commit_play`. Tool calls
before the cap: `trump_declared`, `unseen`, `is_legal`×2, `is_trump`×2,
`void_audit`×4. The four `void_audit` calls — one per opponent seat —
are excessive for a straightforward narrator decision. Suggests
`max_turns=12–15` for comfort, or a system-prompt nudge discouraging
exhaustive void probing.

## Comparison to Gemma iter-1 on move3[0:10]

Gemma iter-1 eval (`burl/eval/results/move4_iter1_eval/summary.json`)
ran the same overlapping 10 decisions:

| metric            | Gemma iter-1        | Haiku (this run)    |
|-------------------|---------------------|---------------------|
| n_completed       | 5/10                | **10/10**           |
| bot_match_rate    | 80% (of 5)          | **80% (of 10)**     |
| mean_eq_delta     | −0.76 (5)           | −1.58 (10)          |
| retry_exhausted   | 5/10                | 0/10                |

**Tool histogram (move3 decisions only):**

| tool                    | Gemma iter-1 | Haiku (move3) |
|-------------------------|--------------|---------------|
| trump_declared          | 11           | 8             |
| is_legal                | 9            | 18            |
| is_trump                | 4            | 15            |
| unseen                  | 0            | 6             |
| void_audit              | 0            | 5             |
| eq_outcome_distribution | 0            | 10            |
| conditional_outcome     | 0            | 0             |
| commit_play             | 0 (parsed)   | 10            |

Three structural differences:

1. **Haiku completes everything; Gemma retry-exhausts half.** Haiku's
   self-check via `is_legal` before `commit_play` (~1.8/decision) is the
   pattern we'd like STaR to reinforce in Gemma. Gemma relies on the
   engine's retry loop after an illegal attempt, which doesn't always
   recover within budget.
2. **Haiku reaches for `eq_outcome_distribution` ~1×/decision; Gemma
   never does.** This is the biggest tool-diversity gap and probably
   the biggest signal STaR should capture.
3. **Neither uses `conditional_outcome`.** Both have the tool surface
   available; neither reaches for it. For Haiku this is a surprise
   (strong tool-using model, never once called it). For Gemma it's a
   consequence of primer exposure — the tool isn't demonstrated. If we
   want the counterfactual probe in the STaR rollout corpus, we may
   need to synthesize or gate-in an example.

The 80% vs 80% match-rate parity on completed decisions is misleading:
Gemma only completed 5, and those 5 are selection-biased toward easier
decisions (the ones it didn't get stuck on). The honest comparison is
Haiku 8/10 vs Gemma 4/10 (80% of 50% completed).

## Recommendations for iter-2

1. **Keep `conditional_outcome` on the tool surface but plan to synthesize
   examples.** Neither Haiku nor Gemma uses it zero-shot. If it's load-
   bearing for the big-gap decisions, the STaR corpus blend needs to
   include at least a few demonstrations.
2. **Consider a "probe both candidates with EQ" rule in the primer.**
   4 of Haiku's 8 misses correlate with zero E[Q] calls. Cheap fix,
   big impact at Gemma scale.
3. **max_turns=12–15, not 10.** Haiku's d11 timeout shows the current
   10-turn cap is tight for decisions where an agent wants to probe
   all seats via `void_audit`. Cheap insurance.
4. **Keep the Haiku traces as a ceiling reference.** Per the original
   T3 framing, these anchor comparisons for Move 5+. If STaR-trained
   Gemma reaches 70%+ bot-match on move4 at matched cost, we have
   an objective yardstick that says the training worked.

## Files & reproduce

- Per-decision traces: `scratch/burl_p5_iter2_prep/haiku_traces/full_d00.jsonl`
  through `full_d29.jsonl` (d11 has no `commit_play` event — turn-cap
  timeout).
- Summary: `scratch/burl_p5_iter2_prep/haiku_traces/full_summary.json`.
- Stdout mirror: `scratch/burl_p5_iter2_prep/haiku_traces/full_stdout.log`.
- Runner: `burl/haiku_spike/run_full.py`.

```bash
python -u -m burl.haiku_spike.run_full
```

Auth via existing Claude Code session; no `ANTHROPIC_API_KEY` needed.
Hard spend cap $1.20, per-decision cap $0.09.
