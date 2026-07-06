---
title: Winning 42 Ch09 Doubles No Trump
kind: experiment
first_seen: local-2026-05-01
last_updated: local-2026-05-01
status: active
---

## Summary

This page is the bead-backed work surface for `t42-ni1l.9`: Chapter 9, "Doubles as
Trumps and No Trumps." It harvests doubles-as-trump and no-trump concepts from Winning
42 into measurable hypotheses for [[gus]], [[burl]], and [[forge]].

The chapter is mostly about regime switching. A hand that looks fragile under a pip-suit
declaration can become playable when doubles become trump, because doubles leave their
native suits and the non-double tops of sixes/fives/fours change. A different hand can
look like a doubles-trump bid but become safer as no-trump, because no-trump keeps doubles
as suit tops and removes the scary first-trick trump-loss count dump. The book's examples
are therefore good detector material: they ask whether the player can compare two rule
regimes over the same tiles instead of applying a single "more doubles is better" rule.

Source slice: `scratch/winning42/winning42.with_figures.md` lines 3819-4242, read from
the main checkout because this isolated worktree does not contain `scratch/`.

## Chapter Claims

- Doubles-as-trump is a distinct declaration where all seven doubles are trump and no
  longer belong to their native suits; therefore a non-double such as `6-5` can become
  the high tile in both sixes and fives. Source: Ch9 opening and first doubles-as-trump
  example, lines 3819-3890.
- Four or five doubles are only a candidate signal, not the answer. Missing high doubles,
  off-side weakness, and count exposure still define the bid ceiling. Source: Ch9 examples
  around pages 76-79, lines 3840-4020.
- Surface loss accounting can overstate doubles-trump risk when trump draws force key
  suits and turn a high off into a later walker. Source: the planned low-double sacrifice
  and `6-5`/`6-1` sequence, lines 3860-3955.
- A deliberate first trick loss can be correct if it buys trump exhaustion and leaves a
  no-more-loss contract budget. Source: the low-double lead that loses 11 planned points,
  lines 3885-3945.
- Under doubles-as-trump, the model must recompute suit tops after doubles are removed:
  a high non-double can protect two suits at once. Source: `6-5`, `6-4`, and `5-4`
  examples, lines 3855-4020.
- No-trump can dominate doubles-as-trump when doubles serve better as suit-top support
  for weak offs than as trump, and when declaring doubles would create a first-trick
  count-dump risk. Source: Hand 25 no-trump analysis, lines 4021-4175.
- No-trump play resembles an 84 endgame plan: keep the weakest offs for late, save support
  doubles ahead of them, and count suit depletion to choose which support double/off to
  lead first. Source: no-trump play-through, lines 4090-4185.
- No-trump defense borrows 84-defense preservation: save doubles and same-suit pairs
  that can beat the bidder's likely last weak domino. Source: setting no-trump section,
  lines 4190-4242.
- The variant where no-trump makes doubles their own suit is explicitly non-standard and
  excluded from tournament play. Source: rule-variant note, lines 4180-4210.

## Concept Table

| Concept | Detector/state inputs | Metric/test | Likely data source | Priority | Implementation notes | Readiness |
|---|---|---|---|---:|---|---|
| Doubles-as-trump candidate | Hand doubles count; double ranks held/missing; auction seat; candidate bid floor; count in hand/off | Bid EV, set rate, and over/underbid regret for 3/4/5+ doubles buckets | Generated deal corpus plus [[forge]] E[Q] bid/play rollouts | P0 | Do not tag "has four doubles" as sufficient; add high-double coverage and off-risk features. | Enumeration: yes; Oracle: yes; Gus: feature; Burl: prompt/tool bucket |
| Doubles removed from native suits | Declaration; tile suit membership under doubles; non-double suit tops; legal-follow tables | Legality correctness and suit-top recomputation accuracy | Engine table tests and generated games | P0 | This is a rules substrate detector: `double-five` stops being a five when doubles are trump; `6-5` can become top five and top six. | Enumeration: yes; Oracle: no; Gus: feature; Burl: tool fact |
| Non-double dual-suit top protection | Off tile pips; higher live non-doubles in each native suit; doubles removed; owned/played blockers | Protected-off make probability; false-safety rate; regret when protection is ignored | Generated decisions with doubles-trump declarations | P0 | Generalizes existing `off_protection`: a tile can be high on both sides after doubles leave suits. | Enumeration: yes; Oracle: yes; Gus: feature; Burl: explanation bucket |
| Surface-loss overestimate | Naive loss budget under candidate declaration; simulated forced-follow consequences; key suit depletion | Difference between naive max-loss estimate and oracle make/set probability | Enumerated hands plus oracle replay | P1 | Compare static hand scoring to reachable play lines; mark book-style "looks worse but plays safer" cases. | Enumeration: partial; Oracle: yes; Gus: eval tag; Burl: reasoning trace |
| Low-double sacrificial lead | Doubles in hand; missing higher doubles; contract margin; count in first trick; remaining loss budget | Planned-loss success rate; tail set risk after losing first trump trick | Oracle rollout from bidder first lead states | P0 | Detector fires when leading a low double intentionally draws a high double and accepts a bounded trick/count loss. | Enumeration: partial; Oracle: yes; Gus: action-local tag; Burl: trace bucket |
| Planned-loss budget | Bid amount; points already conceded; remaining live count; tricks bidder can still lose | Contract survival after a planned lost trick; catastrophic second-loss rate | Generated game traces and E[Q] rollouts | P0 | Book examples repeatedly say the bidder can lose once, then "cannot lose another trick"; make that an explicit state feature. | Enumeration: yes; Oracle: yes; Gus: feature; Burl: explanation bucket |
| Trump-draw-to-walker plan | Outstanding doubles; suit counts after each trump lead; candidate final off; partner/opponent follow events | Walker creation precision/recall; regret of failing to exhaust trump/suit before off | Bidder-play trajectories under doubles-trump | P0 | Extends `walker_candidates` with declaration-specific suit depletion caused by trump leads. | Enumeration: partial; Oracle: yes; Gus: feature; Burl: trace bucket |
| Draw specific count tile | Count tile identity such as `6-5`; support lead that forces it; off risk neutralized by its appearance | Make probability after key tile drawn; missed neutralization regret | Targeted generated positions and oracle counterfactuals | P1 | Ch9 examples often hinge on drawing one exact count tile before a risky off becomes safe. | Enumeration: yes; Oracle: yes; Gus: belief target; Burl: tool-query target |
| No-trump over doubles-trump choice | Same hand evaluated under doubles-trump and no-trump; first-lead risk; doubles as support vs trump | Delta E[Q]/make probability by declaration; bid ceiling calibration | Candidate-bid enumerations plus oracle rollout | P0 | Build paired declaration examples: same seven tiles, two candidate regimes. This is the chapter's highest-value bucket. | Enumeration: partial; Oracle: yes; Gus: feature/eval; Burl: compare-regime trace |
| No-trump first-trick count-dump risk | Doubles held; missing top double; opponent void probability in doubles if doubles are trump; live count throwable | Expected count conceded on first trump trick; tail count-dump probability | Exact combinatorics plus oracle/state sampling | P1 | Book estimates the risk qualitatively; enumeration can measure the actual count exposure conditional on hand shape. | Enumeration: yes; Oracle: yes; Gus: feature; Burl: explanation bucket |
| No-trump lead-control state | No-trump declaration; current leader; remaining suit tops; no trump reentry available; late-off order | Regret of losing lead before final planned offs; contract make probability | No-trump generated games | P0 | Unlike trump hands, lead loss is hard to repair; tag "cannot regain by trump" as a regime-level control feature. | Enumeration: yes; Oracle: yes; Gus: feature; Burl: trace bucket |
| Save support doubles ahead of offs | Doubles held matching off pips; off tiles remaining; suit depletion count; play order | Regret of spending support double early; safe-off conversion rate | Oracle rollouts and generated no-trump decisions | P0 | Mirrors 84 asset preservation but with doubles as native suit tops, not trump. | Enumeration: partial; Oracle: yes; Gus: action tag; Burl: trace bucket |
| Dynamic suit-count tracker under no-trump | Count of remaining tiles per off suit after each lead; public follow events; candidate walker status | Prediction accuracy for final walker and optimal next-to-last lead | Game traces with hidden truth for validation | P0 | Turn the book's "count fives/deuces played" instruction into public-state counters for each off side. | Enumeration: yes; Oracle: yes; Gus: belief/calibration; Burl: tool-use bucket |
| Defender no-trump set plan | Defender held doubles/pairs; bidder likely weak final suit; preservation opportunity; forced-discard status | Missed set rate; preservation-vs-discard regret; set attribution | No-trump defense positions and Burl traces | P1 | Reuse Ch8 84-defense detectors, but gate them on no-trump declaration and bidder late-off profile. | Enumeration: partial; Oracle: yes; Gus: eval bucket; Burl: trace bucket |
| Rule-variant legality flag | Ruleset metadata; declaration type; whether doubles are native suit tops or separate suit in no-trump | Variant contamination rate; illegal/incorrect follow-suit under tournament rules | Engine tests and synthetic variant games | P1 | Standard tournament no-trump keeps doubles in native suits; separate-suit no-trump must be a variant gate. | Enumeration: yes; Oracle: no; Gus: guard feature; Burl: tool fact |

## Highest-Value First Detectors

1. `declaration_regime_compare`: for each biddable hand with 4+ doubles, compute paired
   no-trump, doubles-trump, and best pip-trump features: high-double coverage, off-risk,
   support doubles, planned first-loss budget, and oracle bid ceiling.
2. `doubles_native_suit_removal`: deterministic rules detector that recomputes suit tops
   under doubles-trump and catches false assumptions such as treating a double as still
   belonging to fives.
3. `planned_low_double_sacrifice`: action-local tag for low-double leads that intentionally
   draw higher doubles while preserving enough contract budget to make the bid.
4. `dual_suit_top_protection`: public-state tag for high non-doubles that protect both
   sides of an off after doubles are removed from suits.
5. `no_trump_support_double_preservation`: no-trump action tag for saving a double that
   protects a late off, plus a regret bucket when it is spent early.
6. `dynamic_no_trump_suit_counter`: per-suit remaining-tile counters that estimate which
   late off has become a walker and which support double should be led next.

## Analysis Routes

Enumeration-ready:

- Rule-state checks: doubles-as-trump suit membership, non-double suit tops, no-trump
  tournament suit membership, legal-follow behavior, remaining suit counts, first-trick
  count-dump combinatorics.
- Hand-shape priors: frequency of 4+ doubles with/without high doubles, no-trump candidate
  shapes, and support-double/off-pair configurations.

Oracle-ready:

- Paired declaration rollouts for the same hand under doubles-trump, no-trump, and pip
  trump choices.
- Counterfactual play lines for planned low-double sacrifice, support-double preservation,
  and trump-draw-to-walker sequences.
- Regret buckets for spending a support double early, failing to draw a key count tile,
  or leading the wrong late off in no-trump.

Gus-ready:

- Add global tags for `declaration_regime`, `doubles_removed_from_suits`,
  `no_trump_lead_control`, and `support_double_count`.
- Add action-local tags for `low_double_sacrifice`, `support_double_spent`,
  `dual_suit_top_play`, and `late_off_walker_candidate`.
- Evaluate belief calibration on key owner questions: missing high double, specific count
  tile such as `6-5`, and whether a defender can hold the last winning tile until late.

Burl-ready:

- Inspect whether traces explicitly compare regimes instead of anchoring on "many doubles."
- Inspect whether the tool loop asks for `trump_declared()`, legal suit membership, unseen
  doubles/count, and void or owner facts before committing.
- Score reasoning faithfulness: the committed play should match the stated plan about
  planned loss, support-double preservation, or no-trump late-off ordering.

## Claim Ledger

[[w42-phase4-doubles-notrump-regime-tests]] now supplies same-hand paired
evidence for the central Chapter 9 declaration-choice claims. It is generated
policy simulation rather than exhaustive oracle proof, so rows remain
context-limited where optimal play or auction pressure matters.

| Claim | Status | Next empirical check |
|---|---|---|
| Doubles leave native suits under doubles-trump; non-doubles become suit tops. | supported-rules | Add engine/table assertions and strategy tags for suit-top recomputation. |
| Four or five doubles can justify doubles-trump, but only with high-double/off-risk context. | context-limited / caveat supported | In `t42-br7n.5`, no-trump beats doubles-trump on `66.1%` of four-plus-double hands; high double control is the near-flat pro-doubles slice. |
| A low-double planned sacrifice can turn a fragile hand into a makeable 30/31. | underpowered | Run oracle counterfactuals on Ch9-like hands with and without the low-double lead. |
| No-trump can be safer than doubles-trump when doubles protect late offs and avoid first-trick count dump. | supported in generated paired-regime test | Missing-top / low-exposure and no-trump support buckets prefer no-trump about `76%` of the time. |
| Dynamic no-trump suit counting can select the safer late off. | underpowered | Evaluate final-two-trick lead choice against perfect-information and E[Q] rollouts. |
| No-trump defenders should preserve doubles/pairs like 84 defenders. | context-limited | Reuse Ch8 preservation detector on no-trump defense positions and measure missed-set regret. |
| No-trump with doubles as a separate suit is a non-standard variant. | supported-rules | Add ruleset gate so tournament analyses exclude variant contamination. |

## Wave 1 Findings (Book Validation v1)

The distribution-lens reranker in
[[w42-bookval-v1-wave1-distribution-lens-reranker]] produced two findings
that change how Ch 9 declaration-choice work should be framed:

- **No-trump shows a 90% EV-lying rate** when ranking actions under
  alternative utility lenses. More than for any other regime. In no-trump,
  multiple actions cluster near the make threshold and scalar EV becomes
  an unstable predictor of optimal play; tail-aware utilities (CVaR,
  robust_q25) separate them more cleanly. This sharpens the no-trump-as-
  late-off-management framing the chapter already uses.
- **Doubles-trump shows 39% CVaR disagreement with EV** - nearly double
  the corpus average. Confirms the chapter's "doubles-trump is high-
  variance" intuition at the action-ranking level, not just at the
  declaration-choice level.

The hidden-threat impact ranker in
[[w42-bookval-v1-wave1-hidden-threat-impact-ranker]] adds a no-trump-
specific finding:

- **`4-4` held by `bidder_partner` in no-trump** is the second-highest-
  impact non-trump load-bearing tile class (impact 47-49 across 8
  decisions). The chapter notes that no-trump bosses determine count
  timing; the data confirms a specific per-suit boss is load-bearing in a
  way the existing `hidden_proxy_early_high_uncertainty` proxy partially
  but incompletely captures. A dedicated `4-4-in-nt` belief label is
  warranted.

## Links

[[winning42-strategy-measurement]] / [[gus-strategy-tags-probe]] / [[gus]] / [[burl]] / [[forge]]
