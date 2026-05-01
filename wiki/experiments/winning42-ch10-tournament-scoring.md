---
title: Winning 42 Ch10 Tournament Scoring
kind: experiment
first_seen: local-2026-05-01
last_updated: local-2026-05-01
status: active
---

## Summary

This page is the bead-backed work surface for `t42-ni1l.10`: Chapter 10, "Tournament
Scoring." It harvests scoring-objective concepts from Winning 42 into measurable
hypotheses for [[gus]], [[burl]], and [[forge]].

Chapter 10 is small but strategically sharp: it treats tournament marks as a
reward-function switch, not just a bookkeeping convention. Under point scoring, every
captured point matters until a team reaches 250. Under marks, most ordinary bids collapse
to a one-mark win/loss threshold, with 84 worth two marks, 126 worth three, and the match
ending at seven marks. The book argues that this speeds tournaments but erases some
point-taking skill, especially nonbidder partial points and extra rewards for large sets
or high-scoring makes. (Winning 42 Ch10, `scratch/winning42/winning42.with_figures.md`
lines 4243-4331)

The local [[gus]] practicalities notes already reached a compatible utility conclusion:
for mark scoring, the utility cliff is the point. Above the contract threshold, extra
margin has no immediate durable value; below it, all losses give the mark away. Chapter
10 adds the experimental agenda: compare that mark objective against the richer point
objective and make score mode a first-class detector, eval bucket, and explanation
condition.

## Work Surface

The chapter slice is `scratch/winning42/winning42.with_figures.md` lines 4243-4331.
Completion requires source-backed concepts, detector inputs, metrics/tests, likely
data sources, implementation notes, and readiness labels for enumeration, oracle
rollouts, Gus analysis, or Burl traces.

## Concept Table

| concept | source-backed claim | detector/state inputs | metric/test | likely data source | priority | implementation notes and readiness |
|---|---|---|---|---|---:|---|
| `score_mode_objective` | Marks award a hand-level mark, not the exact ordinary bid amount; point scoring preserves every captured point until 250. | Ruleset/scoring mode, bid value, declaration, current hand points by team, match score. | Compare argmax action under point-EV vs mark-EV; report policy-divergence rate and regret under the alternate objective. | Forge generated games plus synthetic replay of identical states under both utilities. | P0 | Enumeration-ready for terminal utilities; oracle-ready once E[Q] can swap point and mark reward functions; Gus-ready as a global tag; Burl-ready as a prompt/tool field. |
| `early_terminal_under_marks` | Under marks, a 30 bid that is already made or a set that is already guaranteed can end before all seven tricks. | Bid value, team points captured so far, maximum possible remaining points for each team, trick index, current winner, live count. | Earliest terminal trick distribution; saved-trick count; missed-terminal detector precision. | Deterministic engine state plus generated game traces. | P0 | Enumeration-ready because make/set impossibility is arithmetic over remaining points; oracle-ready for rollout truncation; Gus-ready as a state feature; Burl-ready as an explanation/tool fact. |
| `nonbidder_partial_points_erased` | Point scoring rewards defenders for taking some points even when the bidder makes the bid; marks erase that value. | Defender captured points, bidder made flag, bid value, scoring mode, hand final score. | Frequency and size of defender partial-point value under made contracts; actions whose point objective values defensive points but mark objective ties them. | Forge traces with point totals; paired mark-vs-point action scoring. | P0 | Enumeration-ready at terminal states; oracle-ready for policy deltas; Gus-ready as a concept bucket for defensive count-taking; Burl-ready for traces that explain whether a count grab matters only in points. |
| `set_severity_compression` | Setting a bidder by a little and setting badly both award one mark, except higher mark bids scale by their mark multiplier. | Bid value/mark multiplier, bidder shortfall, defender captured points, final hand outcome. | Compression ratio: point swing or shortfall severity collapsed into same mark; policy divergence in already-set states. | Terminal hand records and counterfactual rollouts from set-secured states. | P1 | Enumeration-ready terminally; oracle-ready for "how hard to set" action comparisons; Gus-ready as set-margin bucket; Burl-ready as a reason to stop chasing extra count after the mark is secured. |
| `special_bid_mark_multiplier` | 84 is worth two marks either way, 126 three, and so on; ordinary bids all collapse to one mark. | Bid class, mark multiplier, contract made/set flag, match score to seven. | Threshold policy shift for 42-vs-84-vs-126; escalation value by score context; make-probability needed for positive mark EV. | Bidding records plus oracle/rollout make probabilities. | P1 | Enumeration-ready for terminal scoring; oracle-ready if bid value and multiplier are plumbed; Gus-ready as bid-regime tag; Burl-ready for objective-aware explanations of 84 urgency. |
| `low_bid_score_distortion` | Three barely made 30 bids can be 3-0 in marks while only 90-36 in points if defenders took 12 each hand. | Sequence of hands, bid values, bidder points, defender points, cumulative point score, mark score. | Match-score distortion index: mark lead minus point-score-equivalent lead; examples where marks overstate dominance. | Full-game simulations with both scoreboards tracked. | P1 | Enumeration-ready from completed hands; oracle-ready for match-level rollouts; Gus-ready as score-context feature if match score enters state; Burl-ready for post-hand explanation. |
| `point_system_skill_signal` | The book says point scoring is fairer and better for skill-honing in small groups because every point remains live. | Player/team policy identity, score mode, action chosen, point swing, mark outcome, skill proxy or oracle regret. | Does point scoring produce higher separation between strong and weak policies than marks? Compare Elo/Bradley-Terry or paired regret by score mode. | Arena simulations: E[Q], Gus variants, heuristic bots, Burl traces. | P2 | Needs oracle rollouts and population eval; not pure enumeration; Gus-ready as eval bucket; Burl-ready once enough traces exist. |
| `tournament_speed_tradeoff` | Marks speed tournaments by allowing early hand termination and more hands/interactions in large groups. | Score mode, terminal trick index, hand duration proxy, match length in hands/tricks, tournament schedule model. | Expected tricks saved per hand/match; hands per hour; advancement variance under fixed time. | Simulated game logs, wall-clock play traces if available, tournament schedule model. | P2 | Enumeration-ready for terminal-trick math; oracle-ready for full-match simulation; Gus/Burl readiness depends on match/tournament harness rather than single-decision data. |
| `timed_marks_advancement_objective` | Timed competitions may rank teams by cumulative marks, making advancement probability the real objective. | Round clock/time budget, mark score, opponent pool, current hand state, tournament format. | Difference between per-hand mark EV and advancement EV; late-round risk-taking threshold. | Synthetic tournament simulations over policy populations. | P2 | Not enumeration-only; needs tournament simulator; Gus-ready only with explicit tournament context; Burl-ready as a future high-level planning trace. |

## First Detectors And Eval Buckets

1. `scoring_mode`: attach `points_to_250`, `marks_to_7`, or `timed_marks` to every generated state before any policy/regret comparison.
2. `mark_terminal_state`: compute whether the hand is already made, already set, or still live under marks; record trick index and remaining maximum points.
3. `point_mark_policy_divergence`: rescore the same legal actions under point utility and mark utility, then bucket where the argmax differs or where mark utility creates large ties.
4. `partial_points_erased`: for made contracts, measure defender points that mattered under points but not marks.
5. `set_severity_compressed`: for failed contracts, measure bidder shortfall and point swing collapsed into the same mark result.
6. `scoreboard_distortion`: replay completed matches with both scoreboards and flag sequences where mark score and point score tell different stories.

## Analysis Routes

### Enumeration-ready

- Terminal scoring under points vs marks.
- Early make/set arithmetic under marks.
- Mark multipliers for 84, 126, and higher mark bids.
- Scoreboard replay over completed hand records.

### Oracle-ready

- Paired action scoring under point-EV vs mark-EV.
- Rollout truncation once a mark terminal state is reached.
- Policy divergence in "already made", "already set", and "still live for points"
  states.
- Match-level simulations that track both scoreboards.

### Gus-ready

- Add `score_context` tags: scoring mode, contract threshold, mark multiplier,
  made/set/live status, point-vs-mark divergence bucket.
- Evaluate [[gus]] regret separately under point objective and mark objective, rather
  than mixing score regimes.
- Test whether action-local strategy tags help most in states where mark utility creates
  ties but point utility still distinguishes count-taking quality.

### Burl-ready

- Expose scoring mode and terminal-status facts in [[burl]] tools/prompts so traces can
  say why extra count matters or no longer matters.
- Audit Burl rationales for objective mismatch: e.g. chasing extra point margin in a
  secured mark hand, or ignoring defensive partial points in a point-scored match.
- Build contrast traces where the same public hand has different best explanations under
  points and marks.

## Claim Ledger

No empirical run was performed for this chapter harvest. All statuses below are therefore
book-hypothesis statuses, not measurement results.

| claim | status | evidence and next check |
|---|---|---|
| Marks change the objective from exact hand points to a hand-level threshold. | context-limited | Supported by Chapter 10's scoring description and compatible with the existing Gus mark-utility note. Next check: implement explicit point-vs-mark utility swap and measure action divergence. |
| Marks create early terminal states where the rest of the hand need not be played. | underpowered | Chapter gives examples of making a 30 bid on trick 3 or setting on trick 4. Next check: deterministic early-terminal detector over generated traces. |
| Marks erase defender partial-point rewards when the bidder makes the contract. | underpowered | Chapter gives the 90-36 vs 3-0 example. Next check: replay made contracts and quantify defender points lost to mark compression. |
| Marks erase set severity except for bid-class mark multipliers. | underpowered | Chapter says setting by any ordinary margin gives one mark, while 84/126 scale. Next check: terminal compression distribution by bid class and shortfall. |
| Point scoring better hones skill for small groups. | underpowered | Plausible but not measured. Next check: compare policy separation and regret signal under point-score and mark-score arenas. |
| Marks speed tournaments enough to justify the strategic loss. | underpowered | Chapter asserts only a modest time cost for point matches and a speed/social benefit for tournaments. Next check: simulate tricks saved and tournament throughput under early termination. |

## Links

[[winning42-strategy-measurement]] - [[gus-strategy-tags-probe]] - [[gus]] - [[burl]] - [[forge]] - [[texas-42]]
