---
title: Winning 42 Ch15 Celebrities Style
kind: experiment
first_seen: local-2026-05-01
last_updated: afd4802
status: retired
---

**Phantom plan.** This one-session chapter harvest (2026-05-01) registered 6
measurement targets and got zero empirical follow-up in the two months
since — none of its detectors (`dot_count_discipline`,
`watchfulness_entropy_delta`, `overbid_restraint_bucket`,
`partner_fit_residual`, `partner_legibility_support`,
`table_play_believability`) were ever built, and none of its claim ids
appear in the central 64-row ledger ([[w42-phase4-final-claim-audit]]). The
project's actual frontier moved through the phase-4 closure and on to the
[[champion]]/[[jud]] ladder.

## Summary

This page is the bead-backed work surface for `t42-ni1l.15`: Chapter 15, "Texas 42
Celebrities." It harvests player-style and partnership concepts from Winning 42 into
measurable hypotheses for [[gus]], [[burl]], and [[forge]].

Chapter 15 is not mainly a tactics chapter. It is a population-and-style chapter: 42
appears as a family, touring, political, workplace, and reunion game, and the direct
strategy comments are small but useful. Robert Crippen emphasizes watching the other
players; B. J. Thomas emphasizes playing the hand honestly, avoiding overbids, patience,
and willingness to go low; Robert Earl Keen emphasizes dot-count accuracy; and the Preston
Gray note frames strength as partner-dependent. Those claims fit the broader
[[winning42-strategy-measurement]] catalog's `style_and_partnership`, attention, bid-risk,
and model-evaluation surfaces.

The useful frontier stance is: do not hard-code "celebrity advice." Convert it into
style-conditioned buckets, partner-fit residuals, and trace/accounting tests. [[gus]] can
use them as public-state concept buckets and optional style tags. [[burl]] can use them as
trace faithfulness and table-play believability checks. [[forge]] can provide the oracle
baseline for whether a style is disciplined, exploitable, or merely different.

## Source Hooks

The chapter slice is `scratch/winning42/winning42.with_figures.md` lines 8915-9499.

- The chapter opens by framing 42 as a socially widespread Texas game across politicians,
  singers, astronauts, journalists, families, and local communities (lines 8925-8931,
  8966-8977, 8996-9010, 9113-9131, 9413-9446).
- Ann Richards says she learned by watching her father play, making observation itself a
  learning mode (lines 9029-9040).
- Robert Crippen's direct advice is to watch what the other players do (lines 9170-9199).
- B. J. Thomas reports a family style that enjoyed going low and Nel-O, then gives advice:
  play the hand, avoid overbidding, be patient, and do not fear low bids (lines 9222-9244,
  9272-9275).
- Robert Earl Keen's advice is dot-count discipline, and he identifies Preston Gray as
  exceptional "given the right partner" (lines 9307-9315).
- Bill Moyers frames skill as culturally acquired and unevenly distributed, with strong
  family players and a social setting where losing carries little shame (lines 9403-9419,
  9448-9484).

## Concept Table

| Concept | Source hook | Detector/state inputs | Metric/test | Likely data source | Priority | Readiness | Implementation notes |
|---|---|---|---|---|---:|---|---|
| Watchfulness / opponent-model quality | Richards learns by watching; Crippen advises watching players. | Full public play history, failures to follow, bids/passes, donations/non-donations, current trick, legal actions, hidden-tile posterior if available. | Belief Brier/log loss before and after each public reveal; regret delta on decisions where a prior reveal changes the best action; trace mentions grounded in public evidence. | Generated games with true hidden hands, Gus belief outputs, Burl traces. | P0 | Enumeration: partial; Oracle: yes; Gus: yes; Burl: yes. | This is the Ch15 bridge back to Ch6. The first deterministic tag is `public_reveal_count`; the belief tag is `owner_entropy_delta`. |
| Overbid restraint | Thomas says not to overbid and to play the hand as dealt. | Hand shape, candidate trump, auction position, score, partner bid/pass history, count in hand, off-risk, oracle make probability by bid bucket. | Overbid rate where chosen bid exceeds calibrated make threshold; set-rate tail by style; bid regret versus best oracle-backed bid. | Bidding corpus, forge bid evaluator, generated auction states. | P0 | Enumeration: hand-shape priors; Oracle: yes; Gus: optional; Burl: yes if bidding traces exist. | Style label should be relative to hand opportunity, not raw bid height. A 36 can be restraint; a 31 can be reckless. |
| Low-game / Nel-O willingness | Thomas's father liked going low and bidding Nel-O; Thomas says not to fear low bids. | Rule variant, Nel-O legality, hand high/low distribution, doubles, count exposure, score pressure, auction position. | Frequency of valid low-game opportunities; regret of low-game bid versus ordinary bid; false-low attempts under straight-42 rules. | Variant-enabled generated deals; optional-rules simulations. | P1 | Enumeration: yes for eligibility; Oracle: yes if variant solver exists; Gus: later; Burl: later. | Gate behind `rule_variant`. This is a contamination guard as much as a strategy bucket. |
| Dot-count discipline | Keen says counting the right number of dots matters. | Current trick points, captured points, remaining live count, bid target, set threshold, hand count liability, trace arithmetic. | Count-accounting error rate; impossible-score statements; regret when count math would change action ranking; exactness of Burl thought math. | Generated decisions with deterministic point totals; Burl traces. | P0 | Enumeration: yes; Oracle: yes; Gus: tags yes; Burl: yes. | Cheap to implement as deterministic features: `points_needed_to_make`, `points_needed_to_set`, `live_count_remaining`, and `count_at_risk`. |
| Partner-fit residual | Keen says Preston Gray could beat anyone with the right partner. | Fixed player/team IDs or simulated policy IDs, seat partnership, style tags, action predictability to partner, donation windows, support leads, defensive saves. | Team performance minus additive individual ratings; paired win-rate or E[Q] residual for fixed versus random partners; partner-support regret. | Self-play policy pools, Gus variants, Burl trace cohorts. | P1 | Enumeration: no; Oracle: yes with policy simulations; Gus: yes; Burl: yes. | Needs repeated identities or artificial style policies. Start with synthetic style policies before claiming human-like partnership. |
| Partner legibility | Family and reunion settings imply stable repeated partners; "right partner" implies readable coordination. | Partner public state, partner legal actions, donation opportunities, lead-control transfers, whether a move helps or strands partner. | Partner action predictability from public evidence; safe-donation precision; support-window hit rate; trace correctly names partner's role. | Generated team-play logs, Burl traces. | P1 | Enumeration: partial; Oracle: yes; Gus: yes; Burl: yes. | Distinguish legal inference from illegal table talk per [[winning42-ch11-table-talk]]. |
| Style classification | Chapter profiles cautious, low-game, watcher, dot-counter, family/social, and partner-dependent archetypes. | Bid aggression, low-bid use, count-accounting accuracy, belief-update strength, support/donation behavior, set-line sharpness. | Multi-label style classifier stability; style-conditioned regret; exploitability against E[Q] bot and mixed style pool. | Self-play logs, generated policy variants, future human/game logs. | P1 | Enumeration: no; Oracle: yes; Gus: yes; Burl: yes. | Treat style tags as descriptive cohorts, not value judgments. A style is good only in context. |
| Social-pressure robustness | Moyers and Gayler frame 42 as a high-emotion social game where losing is not shameful. | Score pressure, near-win/near-loss state, table stakes mode, prompt distractors, time pressure, repeated loss streak. | Tail-regret under pressure buckets; illegal/unsupported trace claims under distractor prompts; recovery after blunder. | Burl eval prompts, generated tournament/timed states, future arena logs. | P2 | Enumeration: no; Oracle: yes for state buckets; Gus: limited; Burl: yes. | Best used as an evaluation wrapper: same state, clean prompt versus noisy/social prompt. |
| Believable table-play evaluation | The chapter treats 42 as a lived social practice, not just optimal action selection. | Thought trace, tool calls, legality, public-evidence references, partnership language, humility around uncertainty, final commit. | Human-readable trace rubric: public evidence only, no hidden leakage, count math correct, partner/opponent roles coherent, action regret acceptable. | Burl traces, STaR corpora, evaluator annotations. | P1 | Enumeration: no; Oracle: yes for regret; Gus: no; Burl: yes. | This is a rationale-quality bucket. Pair it with E[Q] regret so "sounds Texan" never outranks a bad play. |

## Highest-Value First Detectors

1. `dot_count_discipline`
   - Emit deterministic point-accounting tags: bid target, points captured, points needed,
     live count remaining, count at risk, and whether a trace/action contradicts the math.
   - First report: count-math error rate in Burl traces and regret for decisions where count
     math changes the oracle best action.

2. `watchfulness_entropy_delta`
   - Track how much each public reveal should move hidden-tile ownership entropy.
   - First report: Gus belief Brier/log-loss before and after failures to follow, donations,
     and non-donations.

3. `overbid_restraint_bucket`
   - Label bids by calibrated make probability, tail set risk, and whether the chosen bid is
     aggressive, disciplined, or undercalled relative to the hand.
   - First report: set rate and bid regret by restraint bucket.

4. `partner_fit_residual`
   - Compare team E[Q]/win-rate against the sum of individual policy ratings under fixed and
     shuffled partners.
   - First report: synthetic style-policy pool, not human claims.

5. `partner_legibility_support`
   - Label donation/support/lead-transfer windows and whether the partner's action is
     readable from public evidence.
   - First report: safe-donation precision and missed partner-support rate.

6. `table_play_believability`
   - Score Burl traces on public-evidence grounding, count arithmetic, role language,
     uncertainty, and absence of hidden leakage.
   - First report: paired with oracle regret so the metric cannot reward plausible nonsense.

## Analysis Routes

Enumeration-ready:

- Dot/count arithmetic, live/dead count status, bid thresholds, legal low-game eligibility
  under a variant flag, public reveal counts, and simple hand-shape priors.

Oracle-rollout-ready:

- Overbid restraint, low-game opportunity value if the variant solver is available,
  style-conditioned regret, partner-support windows, pressure buckets, and team residuals
  over synthetic style policies.

Gus-ready:

- Watchfulness as belief calibration by public-reveal bucket, strategy-tag ablations for
  `dot_count_discipline`, `overbid_restraint_bucket`, `partner_legibility_support`, and
  style-conditioned regret buckets.

Burl-ready:

- Trace faithfulness to public evidence, count-math correctness, grounded partner/opponent
  references, overbid rationales if bidding traces are added, and social-pressure prompt
  robustness.

## Claim Ledger

No empirical run was performed for this chapter harvest.

| Claim | Status | Reason |
|---|---|---|
| Watching other players is strategically valuable. | underpowered | Source-backed as advice, but no belief-update or regret analysis was run. |
| Avoiding overbids is valuable. | underpowered | Source-backed as advice and consistent with the bidding workstream, but Ch15 adds no empirical threshold. |
| Patience and low-game willingness can be a legitimate style. | context-limited | Source-backed, but low-game/Nel-O depends on enabled rule variants and should not contaminate straight-42 evals. |
| Dot-count discipline is a measurable skill. | supported-as-detector | The detector is deterministic and implementation-ready; the strategic effect size still needs oracle/Burl analysis. |
| Partner fit can explain performance beyond individual skill. | underpowered | Source-backed as a player observation; requires repeated identities or synthetic style-policy simulations. |
| Social table play should influence Burl evaluation. | context-limited | Useful for trace believability and robustness, but must be paired with oracle regret to avoid rewarding flavor over decision quality. |

## Links

[[winning42-strategy-measurement]] | [[gus-strategy-tags-probe]] | [[gus]] | [[burl]] | [[forge]] | [[winning42-ch06-concentration-style]] | [[winning42-ch11-table-talk]] | [[winning42-ch14-history-tournaments]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- Verifiability caveat: the chapter source `scratch/winning42/winning42.with_figures.md` lives only in gitignored scratch on the main machine (absent from worktrees), so the line-range citations depend on that local file surviving; if any detector is revived, re-anchor the ranges first.
- Cheap next probe if ever revived: `dot_count_discipline` is the only detector that is pure deterministic accounting, buildable in an afternoon against existing [[burl]] traces without the other five.
