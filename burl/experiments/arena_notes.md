# Arena pilot observations — seed 900010, Haiku 4.5

**Game:** 28/28 turns played end-to-end. Trump = blanks. Bidder team 0 bid 30,
earned 0, set. Final 0–42 (complete shutout for team 1). Total cost $0.839,
wall 7m22s.

## What Haiku's full-game reasoning looks like

- **Narration is generic.** Nearly every turn opens with a variant of
  "I'll analyze this step by step" or "Let me understand the game state."
  Rarely does the narration name a specific strategic goal (e.g. "partner
  probably has a high trump, so I'll signal").
- **Tool-use shape is dominated by two tools**: `is_legal` (92 calls across
  28 turns, 3.3/turn) and `eq_outcome_distribution` (50 calls, 1.8/turn).
  Haiku repeatedly asks the engine "is X legal?" one domino at a time
  rather than reading off the "legal" field from the prompt. Also calls
  `trump_declared` 24 times (redundantly — it's in the system prompt).
- **E[Q] sampling is the anchor.** When Haiku reaches for a decision
  signal, it almost always runs `eq_outcome_distribution` on 2-5 candidate
  plays. This is sensible; the iter-1 primer is already pushing it toward
  that tool.
- **Partner tracking is weak / absent.** I saw zero calls to
  `conditional_outcome` across 28 turns. No seat reasoned aloud about
  "what partner might hold" — they stayed model-of-their-own-hand.
- **Void audits run late.** `void_audit` was called 17× but clustered in
  tricks 4-7 after enough plays had landed to make it informative. That
  timing is correct.

## Opening-lead regret

Seat 0 (bidder) opened with **3(2|0)** — a small blank — which lost
immediately to seat 3's **15(5|0)** (larger blank). The single costliest
play of the game: bidder led a low trump into a vulnerable defense and
the hand cascaded from there. Haiku's rationale on that move was the
generic "I'll analyze this situation and make my decision" — no explicit
reasoning for leading a trump vs. holding it.

## Turn-count distribution

Per-decision SDK turn counts ranged 3 (trick 7 forced play) → 19 (trick
2 seat 3, multi-candidate analysis). Mean ≈ 10. My first run capped
at `max_turns=10` and died on turn 1; bumped to 20, no further misses.
**Recommend `max_turns=20` as the arena default going forward** — or
higher if we want to give Opus more exploration space on the ceiling
runs.

## Cost shape

| trick | mean cost/turn |
|-------|----------------|
| 1     | $0.028         |
| 2     | $0.032         |
| 3     | $0.031         |
| 4     | $0.033         |
| 5     | $0.030         |
| 6     | $0.032         |
| 7     | $0.025         |

Flat across the game — Haiku does not meaningfully reduce effort as the
game narrows. End-of-game plays (legal=1) still cost ~$0.024, because
the SDK round still runs even if only commit is needed. A cheap win
might be: if `len(legal) == 1`, short-circuit in the arena without
invoking the model. Skipped for now (not in scope, and user wants to
see how the model handles forced plays).

## Tool histogram vs. iter-1 training

From iter-0/iter-1 eval writeups, the SFT adapter never called
`eq_outcome_distribution`. Opus/Haiku at ceiling uses it 1-2× per
decision on average. This is the gap rules-as-tools / verbosity blend
was trying to close on the trained side.

## Scaffold health

- One live issue: initial `max_turns=10` silently hit cap and returned
  no-commit. Arena handled it gracefully (wrote `turn_no_commit` event,
  exited), but an operator needs to know to bump. Consider defaulting
  `--max-turns 20` in the arena CLI.
- No SDK stream errors across 28 Haiku turns — clean run.
- Fresh-session-per-turn pattern worked as designed.

## Next moves (if asked)

- Run a second seed (e.g. 900011) to see whether the shutout was
  specific to this deal or indicative.
- Re-run seed 900010 with `enable_rules_tools=True` to see whether the
  rules-as-tools preamble changes tool histogram / game outcome.
- If Opus becomes available (parallel-tool-bug unblocked), rerun seed
  900010 with Opus for side-by-side ceiling comparison. **DONE — see
  next section.**

---

# Opus head-to-head — same seed 900010

After the Agent-SDK parallel-tool-call lock fix landed (commit 2830be0),
re-ran the arena end-to-end with `claude-opus-4-7`, max_turns=25, same
seed/declaration. Artifacts: `seed_900010_opus.jsonl` + `.md`.

## Outcome

| run    | final (team0 / team1) | bidder_team points | made_bid | cost    | wall  |
|--------|-----------------------|--------------------|----------|---------|-------|
| Haiku  | 0 / 42                | 0                  | set      | $0.84   | 7m22s |
| Opus   | 7 / 35                | 7                  | set      | $5.58   | 8m50s |

Both runs set the bidder (the deal is bad for team 0 — weak count, weak
blanks). Opus salvaged 7 points vs Haiku's 0 on the same deal, a
concrete strategic uplift.

## Opening lead — the single decisive move

- **Haiku** led `3(2|0)` — a low trump, 0 count. Lost immediately to
  seat 3's `15(5|0)` (higher trump). Rationale: *"I'll analyze this
  situation and make my decision."*
- **Opus** led `2(1|1)` — the double-ones, NOT a trump, high "ones"
  suit lead. Opponents were forced to follow suit with their ones and
  couldn't trump in. Seat 0 WON trick 1 with 1 count. Rationale: *"I'm
  leading trick 1 as the bidder (30 on blanks). Let me think about
  this."* followed by seven `eq_outcome_distribution` calls across the
  candidate leads.

Opus's opening instinct — "don't bleed trump; lead a strong
non-trump" — is exactly the heuristic the iter-1 primer hoped to
install in the adapter. Haiku, despite having the same primer, did
not reach for it.

## Narration quality — same data, different voice

Haiku sample (trick 1 seat 0):
> "I'll analyze this situation and make my decision."

Opus sample (trick 1 seat 2):
> "I need to analyze this trick. Partner (seat 0) led 1-1 (ones). Left
> opp (seat 1) played 5-1, following suit with a five-one. The led
> suit is ones."

Opus repeatedly:
- names specific dominoes by pip pair (5-0, 4-1, 3-2, 6-4)
- tracks partner vs. opponent positions correctly
- uses the right vocabulary ("trump", "led suit", "following suit",
  "void", "trumped in")
- self-corrects mid-thought (*"5-5... wait, let me re-read. Trick 5
  so far: ..."*)
- references count dominoes as a class (*"6-4 (10 count!)"*)

Haiku's narration is scaffold; Opus's narration is reasoning.

## Tool-use shape — the diagnostic signal

| tool                     | Haiku calls | Opus calls | ratio (H/O) |
|--------------------------|-------------|------------|-------------|
| `eq_outcome_distribution`| 50          | 43         | 1.2×        |
| `commit_play`            | 28          | 28         | 1.0×        |
| `is_legal`               | 92          | 27         | 3.4×        |
| `trump_declared`         | 24          | 1          | 24×         |
| `is_trump`               | 18          | 1          | 18×         |
| `unseen`                 | 17          | 4          | 4.3×        |
| `void_audit`             | 17          | 1          | 17×         |

Two things jump out:

1. **Both models use `eq_outcome_distribution` heavily** (~1.5/turn).
   This is the anchor tool. The iter-1 primer nudges them toward it and
   they both take the bait — but the SFT adapter doesn't, which is the
   training gap the verbosity blend targets.
2. **Opus barely re-queries facts already in the prompt.**
   `trump_declared: 1` (vs 24 for Haiku), `is_trump: 1` (vs 18). Opus
   reads the system prompt once and trusts it. Haiku repeatedly
   double-checks. Same primer, different trust. That 23× delta on
   `trump_declared` and 17× on `void_audit` is the cheap-vs-ceiling
   distinction.

Neither model called `conditional_outcome` — no partner-modelling across
either full game. That's a real gap the iter-3 / arena variants might
explore.

## Cost and reliability

- Opus: $5.58 / 28 turns = $0.20/turn average. Task estimate was
  $0.25/turn; Opus came in slightly under. No 429s, no session errors.
- Opus cost distribution ranged $0.10 (forced final plays) →
  $0.41 (opening lead with 7 `eq_outcome_distribution` candidate
  evaluations). Cost tracks decision difficulty.
- **Lock-fix held across all 28 turns.** Zero stdio-poison events, zero
  "Stream closed" errors.

## Scaffold observations

- `--tag` flag added to arena.py so Opus output doesn't clobber Haiku
  (`seed_900010_opus.jsonl/.md` vs `seed_900010.jsonl/.md`). No other
  code changes were needed to swap models — the arena was already
  model-agnostic via `run_decision_haiku(model=...)`. (The function
  name is a misnomer now; could be renamed to `run_decision_sdk` but
  not essential.)
- `max_turns=25` was comfortable; Opus mean turns/decision was ≈ 5,
  max observed 12. We did not see Opus fall off a turn-budget cliff the
  way Haiku did at `max_turns=10`.
- Full game takes ~9 minutes of wall time on Opus — similar to Haiku
  (7m). Per-decision latency is higher for Opus but it uses fewer
  per-decision tool rounds.
