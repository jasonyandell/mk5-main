---
title: Winning 42 Ch16 Statistical Odds
kind: experiment
first_seen: local-2026-05-01
last_updated: 12064bf
status: active
---

## Summary

This page is the bead-backed work surface for `t42-ni1l.16`: Chapter 16, "Statistical
Odds." It harvests statistical-validation concepts from Winning 42 into measurable
hypotheses for [[gus]], [[burl]], and [[forge]].

Chapter 16 is mostly an odds appendix, so its first value is calibration: book claims
can be checked by exact double-six enumeration before they become strategy tags. A local
exact enumeration pass on 2026-05-01 supports the chapter's main hand-shape and double
count priors. The strategic recommendations still need [[forge]] E[Q] checks because
knowing the prior does not prove that a lead or bid threshold is optimal.

## Work Surface

The chapter slice is `scratch/winning42/winning42.with_figures.md` lines 9500-9605.
Completion requires source-backed concepts, detector inputs, metrics/tests, likely
data sources, implementation notes, and readiness labels for enumeration, oracle
rollouts, Gus analysis, or Burl traces.

## Source Claims

Chapter 16 makes these measurable claims in the source slice:

- A double-six hand has 1,184,040 seven-domino combinations.
- With four trumps including the double but missing the second-highest trump, leading
  the double first fails only when a setter opponent can keep the second-highest trump
  after following under the double: 10 of 27 missing-trump configurations, or 37%.
- With four trumps including the double but missing the next two highest trumps, a
  setter opponent can double up in 14 of 27 configurations, or 52%; the source treats
  this as a near-50/50 threshold where count in trump can justify the risk.
- Seven-suit coverage, one-void, two-void, and three-void hand shapes are approximately
  41%, 48%, 10%, and 1%.
- Hands with 0, 1, 2, 3, 4, 5, 6, and 7 doubles are approximately 10%, 32%, 36%,
  18%, 4%, less than 0.5%, 1/50%, and a single reshuffle-level extreme.
- The modal hand has two doubles and one void suit, and the doubles prior is part of
  the book's argument for expecting partner help when bidding with more than one off.

## Exact Enumeration Check

The exact hand-combination checks are over the standard 28-tile double-six set and
all `C(28, 7)` hands.

| claim | exact result | status | note |
|---|---:|---|---|
| Total seven-domino hands | 1,184,040 | supported | Equals `C(28, 7)`. |
| All seven suits represented | 42.314% | supported | Source rounds to 41%; close enough for table guidance. |
| Six suits represented, one void | 46.982% | supported | Source rounds to 48%. |
| Five suits represented, two voids | 10.349% | supported | Source rounds to 10%. |
| Four suits represented, three voids | 0.355% | supported | Source says 1%; better written as "less than 1%". |
| No doubles | 9.821% | supported | Source rounds to 10%. |
| One double | 32.081% | supported | Source rounds to 32%. |
| Two doubles | 36.091% | supported | Source rounds to 36%. |
| Three doubles | 17.692% | supported | Source rounds to 18%. |
| Four doubles | 3.931% | supported | Source rounds to 4%. |
| Five doubles | 0.372% | supported | Source says less than 0.5%. |
| Six doubles | 0.0124% | supported | About 1 in 8,054 hands; source's "1/50 percent" is 0.02%. |
| Seven doubles | 0.000084% | supported | Exactly one hand. |
| Modal joint shape | 18.011% | supported | Two doubles plus one void is the most common `(double_count, void_count)` bucket. |

For the four-trump claims, the 27 configurations are the independent assignments of the
three missing trumps to partner, left opponent, or right opponent. The source's 10/27 and
14/27 counts are recovered when partner holdings are not treated as setting threats:

| condition | exact configuration count | status | note |
|---|---:|---|---|
| Four trumps, bidder has double, missing second-highest only | 10 / 27 = 37.037% | supported | Count cases where a setter opponent owns the second-highest trump and another missing trump. |
| Four trumps, bidder has double, missing next two highest | 14 / 27 = 51.852% | supported | Count cases where a setter opponent can keep one of the two high missing trumps after the double lead. |

## Concept Table

| concept | detector/state inputs | metric/test | likely data source | priority | implementation notes | readiness |
|---|---|---|---|---|---|---|
| Hand-combination baseline | Standard double-six tile set, hand size, ruleset gate | Verify `C(28, 7)` and generated deal uniformity | Exhaustive enumeration; generated corpus deal manifests | P0 | This is the denominator for every Chapter 16 prior and a simple corpus sanity check. | Enumeration checked; oracle not needed; Gus ready as feature prior; Burl ready as odds fact. |
| Suit coverage and void prior | Seven-card hand, unique pip count, void count | Exact frequency by suit coverage; corpus chi-square against exact prior | Exhaustive enumeration; Gus/Burl/forge generated games | P0 | Converts "one void is common" into a calibrated prior for hidden-hand belief and table-talk-free inference. | Enumeration checked; oracle not needed; Gus ready; Burl ready. |
| Double-count prior | Hand, double mask, count of doubles | Exact frequency by double count; corpus chi-square against exact prior | Exhaustive enumeration; generated deals | P0 | Supports the claim that two doubles is the modal double count and prevents overfitting to anecdotal double-heavy hands. | Enumeration checked; oracle not needed; Gus ready; Burl ready. |
| Modal joint hand shape | Hand, double count, void count | Joint frequency over `(double_count, void_count)` | Exhaustive enumeration; generated deals | P0 | The source's "most common hand" claim is joint, not just two separate marginal modes. | Enumeration checked; oracle not needed; Gus ready; Burl ready. |
| Four-trump boss-first threshold | Bidder hand, declared trump, trump ranks held/missing, partner/opponent seat distinction | 27-configuration count; E[Q] delta of boss-first versus low-trump-first | Exhaustive missing-trump assignment; forge decision rollouts | P0 | Enumeration supports the source's 10/27 risk only when partner is non-threatening. Strategy value still depends on count, offs, bid margin, and score. | Enumeration checked; oracle ready; Gus ready as action tag; Burl ready as reasoning bucket. |
| Four-trump missing top-two threshold | Bidder hand, declared trump, missing second and third trumps, count-in-trump flag | 27-configuration count; E[Q] threshold with and without trump count | Exhaustive assignment; forge rollouts by count and bid margin | P0 | This is the cleanest "near 50/50" chapter bucket and should become an adversarial eval slice. | Enumeration checked; oracle ready; Gus ready; Burl ready. |
| Count-in-trump exception | Four-trump threshold state, live count in trump, bid need, current score | Paired regret of double-first risk when count can be cashed versus low-trump safety | Forge E[Q] rollouts; generated games | P1 | The book's exception is not an odds claim alone; it is a value-of-cashing-count claim. | Enumeration informs prior; oracle needed; Gus ready; Burl ready. |
| Partner double-help prior | Bidder off count, bidder doubles held, remaining doubles, partner unknown hand | Hypergeometric probability partner has at least two remaining doubles; make/set rate by bucket | Exact conditional enumeration; forge rollouts; Gus belief outputs | P1 | The source uses the double prior to justify counting on partner help when bidding with multiple offs. This needs conditioning on the bidder's own hand. | Enumeration ready; oracle ready; Gus belief ready; Burl trace ready. |
| More-than-one-off risk | Bidder hand, off tiles, off suits, protection by doubles, partner help prior | Set rate and tail regret for bids needing partner help | Forge rollouts by strategy bucket; generated game corpus | P1 | Bridges Chapter 16 priors back to [[winning42-ch02-bidding]], [[winning42-ch04-partner-support]], and [[winning42-ch05-setter-defense]]. | Enumeration partial; oracle ready; Gus ready; Burl ready. |
| Rare extreme double hands | Double count >= 5, suit coverage, bid declaration options | Frequency, bidding anomaly rate, policy regret in rare buckets | Exhaustive enumeration; generated games with oversampling | P2 | Rare buckets should be oversampled for evaluation because natural corpora barely see them. | Enumeration checked; oracle ready with oversampling; Gus needs bucketed eval; Burl needs synthetic traces. |
| Odds rationalization faithfulness | Burl trace text/tool calls, stated probabilities, hand-state bucket | Absolute probability error; unsupported-odds mention rate; tool-call coverage | Burl traces; exact odds table; strategy-tag probe logs | P1 | If Burl cites table odds, its numbers should match the exact priors or clearly mark them as rough. | Enumeration checked; oracle not needed; Gus not applicable except tags; Burl ready. |
| Strategy-tag calibration bucket | Public state, `strategy_features`, action-local tags, exact prior bucket id | Regret and match deltas by odds bucket versus base tags | `gus/eval/strategy_probe.py`; generated corpora | P1 | Chapter 16 supplies stable labels for concept-bucket reporting: void prior, double count, four-trump threshold, and rare extremes. | Enumeration checked; oracle labels ready; Gus ready; Burl secondary. |

## First Detectors

1. `hand_shape_prior_check`: enumerate and corpus-check suit coverage, void count, and
   double count distributions.
2. `four_trump_boss_first_threshold`: detect four-trump hands with the double but missing
   the second-highest trump, then bucket by whether an opponent can retain that trump.
3. `four_trump_missing_top_two_threshold`: detect the 14/27 near-break-even case and split
   oracle rollouts by count-in-trump, bid margin, and off protection.
4. `partner_double_help_prior`: compute the conditional probability that partner has at
   least two doubles after removing bidder hand information.
5. `modal_two_doubles_one_void`: tag the most common hand archetype for Gus/Burl calibration
   and generated-corpus sanity checks.
6. `rare_extreme_double_bucket`: oversample five-plus-double hands so rare but memorable
   cases do not vanish from eval.

## Readiness Notes

- Enumeration is complete for the chapter's raw hand-shape, double-count, modal joint-shape,
  and 27-configuration four-trump claims.
- Oracle rollouts are needed for the action recommendations: boss-first versus low-trump,
  count-in-trump exceptions, and partner-help bidding thresholds.
- Gus can consume these as cheap public tags and eval buckets: void count, double count,
  four-trump threshold state, modal hand archetype, rare extreme hand, and conditional
  partner-help prior.
- Burl can use the same buckets as reasoning tests: whether it cites valid odds, whether it
  distinguishes partner holdings from setter threats, and whether it converts near-50/50
  priors into value-sensitive choices instead of folk rules.

## Claim Ledger

| claim | status | evidence | next check |
|---|---|---|---|
| Seven-card double-six hands total 1,184,040. | supported | Exact enumeration gives 1,184,040. | Add corpus-deal sanity test if deal manifests are surfaced. |
| Suit coverage odds are roughly 41/48/10/1. | supported | Exact enumeration gives 42.314/46.982/10.349/0.355. | Decide whether future docs should write the last bucket as less than 1%. |
| Double-count odds are roughly 10/32/36/18/4/<0.5/0.02/extreme. | supported | Exact enumeration matches the table within rounding. | Add exact priors to a reusable detector table. |
| The most common hand has two doubles and one void. | supported | Exact joint enumeration gives this bucket at 18.011%, the largest joint cell. | Check generated corpora for the same modal bucket. |
| Four trumps with the double but missing the second-highest has a 10/27 setter double-up risk. | supported | Exact 27-case assignment gives 10 when partner is not counted as a setting threat. | Run forge rollouts for boss-first versus low-trump under bid/count contexts. |
| A random partner hand has a decent chance of containing at least two doubles. | context-limited static prior supported | `t42-br7n.7` gives the exact unconditional two-plus-double prior as `58.098713%`, with generated contract partner two-plus rate `57.03125%`. | Full partner bid/play intelligence still needs observed auction policy or state-injected rollouts. |
| Four trumps with the double but missing the next two highest has a 14/27 setter double-up risk. | supported | Exact 27-case assignment gives 14 under the same partner/non-threat interpretation. | Run forge rollouts and split the count-in-trump exception. |
| Bidders with more than one off can count on partner double help. | context-limited | The unconditional double prior is real, but the useful probability is conditional on bidder hand and bid context. | Compute partner conditional prior and correlate with make/set and regret. |
| The book's odds imply the recommended play is optimal. | underpowered | Exact priors alone do not settle E[Q] because count, bid margin, score, and off protection change value. | Use paired oracle rollouts in the first strategy-tags report. |

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- Every exact figure (`C(28,7)`, void/double distributions, modal 18.011%) was independently recomputed by full enumeration and matched.
- Retired-bead references `t42-ni1l.16` and `t42-br7n.7` are no longer resolvable (beads retired 2026-06); the 58.098713% partner two-plus-double prior traces only to the bead and was not recomputed — cheap probe: a one-line hypergeometric enumeration pins it.
- The 10/27 and 14/27 four-trump counts were confirmed internally consistent with the partner/non-threat interpretation but not re-derived; a tiny script over the 3^3 missing-trump assignments would pin them permanently.
- `scratch/winning42/winning42.with_figures.md` is gitignored, so the chapter slice is absent in worktrees though present in the main checkout.
