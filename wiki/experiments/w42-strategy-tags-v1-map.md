---
title: w42 Strategy Tags v1 Map
kind: experiment
first_seen: local-2026-05-01
last_updated: local-2026-05-01
status: active
---

## Summary

[[w42]] Strategy Detector v1 is a chapter-derived detector map, not a Gus retrofit and
not an empirical result. It expands the cheap public-state/action-local surface from
[[w42-strategy-tags-v0]] into a design map for bidding, pounce, donation, 84,
no-trump/doubles, scoring, style, and odds buckets.

The map turns the Winning 42 chapter harvest into detector families that can be computed
from public state, action-local facts, ruleset metadata, or report-time replay. It also
marks concepts that need enumeration, [[forge]] oracle labels, [[gus]]/w42 feature
emission, or [[burl]] trace review before they can support a claim.

Machine-readable design artifacts:

- `w42/strategy_tags_v1_map/detector_map.json`
- `w42/strategy_tags_v1_map/README.md`

W&B links: not applicable.

HF links: not applicable.

Claim ledger impact: no claim-ledger change; no empirical claim tested.

## Scope And Status

This page is the deliverable for `t42-csw6.8`. It defines detector readiness and
implementation shape only. It does not run enumeration, train a model, change Gus code,
change [[forge]] oracle semantics, or update [[burl]] behavior.

Inputs are the existing chapter pages, [[winning42-strategy-measurement]],
[[gus-strategy-tags-probe]], [[w42-claim-ledger]], [[w42-report-template]], and
[[w42-strategy-tags-v0]]. The prior v0 surface already validates that w42 can name and
emit cheap global/action tags from public state; v1 names the deeper chapter-derived
families that should be implemented or reported later.

Readiness labels used below:

| label | meaning |
|---|---|
| `online-computable` | Can be emitted during a decision from public state, legal action set, action candidate, ruleset metadata, and public trick/auction history. |
| `report-only` | Needs completed-hand replay, sampled hidden worlds, paired counterfactuals, trace text, population identity, or aggregate corpus context. |
| `enumeration-ready` | Can be validated by deterministic tables, exact hand enumeration, legal-mask fixtures, or scoring fixtures before any model run. |
| `oracle-ready` | Needs forge E[Q], counterfactual rollout, or perfect-information replay to test value/regret. |
| `gus-ready` | Suitable as a public-state/action-local feature or concept bucket for Gus/w42-style policy, belief, or regret reports. |
| `burl-ready` | Suitable for trace/tool-use/rationale audits without exposing direct best-move or private-hand answers. |
| `not-live-safe` | Cannot be a live feature without illegal/private information; it may still be a report label or leakage audit. |

## v1 Detector Families

### Bidding Risk

| detector family | chapter concepts | concrete public-state/action-local features | mode | readiness | non-goal |
|---|---|---|---|---|---|
| `candidate_bid_loss_budget` | Ch02 backward bidding, Ch03 bid justification, Ch12 natural bid buckets | For each candidate declaration: trump count/rank coverage, unique exposed count tiles, side-specific off risk, duplicate exposure set, estimated affordable loss, current high bid, minimum winning bid. | online-computable for static budget; report-only for calibrated make probability | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not prove a bid is good without rollout or score context. |
| `side_specific_off_risk` | Ch02 off exposure, Ch03 off timing, Ch12 four-off exceptions | For each off tile: high-side and low-side exposed count, neutralized-by-trump flag, in-hand count flag, double-ahead protection by side, four/five catastrophic exposure. | online-computable | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not use hidden owner facts as live inputs. |
| `unnecessary_bid_margin` | Ch02 bid only enough, Ch10 scoring objective | Actual bid minus minimum bid needed to win, gated by scoring mode and auction legality. | online-computable | enumeration-ready, oracle-ready, gus-ready, burl-ready | Not valid under variant objectives unless the scoring mode is explicit. |
| `strong_trump_bad_risk_trap` | Ch02 pass strong-looking trumps, Ch12 no-doubles four-off low bid | High trump count with high off/count exposure, missing boss/second trump, or one-trick catastrophic count loss. | online-computable for label; report-only for regret | enumeration-ready, oracle-ready, gus-ready | Not a hard-coded pass rule. |
| `partner_help_dependency` | Ch02 one-off pessimism, Ch16 partner double-help prior | Loss-opportunity count, partner bid/pass signal, conditional partner double prior from public/bid context, needed rescue class. | online-computable for public proxy; report-only for true help | enumeration-ready, oracle-ready, gus-ready, burl-ready | Live feature must not know partner hand. |
| `natural_bid_bucket_anomaly` | Ch02 natural bids, Ch12 30/31/35/36 vs 32/33 | Bid bucket, last-seat raise exception, current auction, bid ceiling bucket, score pressure. | online-computable | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not treat 32/33 as wrong without context. |

### Play Sequencing, Pounce, And Donation

| detector family | chapter concepts | concrete public-state/action-local features | mode | readiness | non-goal |
|---|---|---|---|---|---|
| `bidder_reentry_trump_preserved` | Ch03 save a trump, Ch12 dangerous trump crisis | Bidder trumps remaining after candidate action, unresolved offs, lead control, highest missing trump, bid margin. | online-computable | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not assume always saving trump is optimal. |
| `pull_trump_vs_early_off_window` | Ch03 trump-first default and early-off exception | Trump count, off count, off suit, outstanding trump danger, opponents' public void/follow evidence, candidate lead class. | online-computable for window; report-only for value | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not flatten default and exception into one rule. |
| `safe_partner_count_donation` | Ch03/Ch04 donation windows, Ch06 unsafe count after trump-in | Current trick winner after each seat, partner/bidder relation, guarantee strength, later seats, action count value, overtrump possibility. | online-computable for public guarantee | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not use whether partner secretly has count as live input. |
| `setter_pounce_window` | Ch03 pounce risk, Ch05 setting, Ch12 high-bid off pounce | Bidder non-trump/off lead, defender void/winning double public proxy, live count, points needed to set, defender seat order. | online-computable for public proxy; report-only for perfect-info attribution | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not reveal hidden pounce ownership to a live player. |
| `count_before_certainty` | Ch05 defender donation differs from partner support | Defender cannot follow bidder off, count in legal actions, partner winner uncertain, bid value/margin, rarity of pounce window. | online-computable for action class; oracle-ready for correctness | oracle-ready, gus-ready, burl-ready | Not a general permission to dump count. |
| `lead_away_from_count_damage` | Ch04 support leads, Ch05 count-calling defense | Candidate leads ranked by live count callable, current role, partner/bidder relation, points needed to make/set, low-count-caller trap flag. | online-computable; report-only for regret | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not replace oracle value in close cases. |
| `count_protection_throwaway` | Ch05 protectors, Ch08 pair protectors, Ch12 asset preservation | Candidate discard preserves/exposes same-suit protector for count tile, live pair, or final 84 asset. | online-computable | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not penalize forced follow-suit breaks as voluntary mistakes. |
| `effective_walker_or_promoted_tile` | Ch01 walker, Ch04 effective double, Ch06 promoted low tile | Candidate led tile is highest live in its suit under declaration, higher-live count, played/dead higher tiles, trump exclusion. | online-computable | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not require hidden unseen ownership except for report precision. |

### 84 Buckets

| detector family | chapter concepts | concrete public-state/action-local features | mode | readiness | non-goal |
|---|---|---|---|---|---|
| `84_contract_regime` | Ch07 all-tricks bid, Ch08 defense mode | Bid value/mark multiplier, contract requires every trick, bidder team trick count, score context. | online-computable | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not mix ordinary count-taking incentives with 84 mode. |
| `laydown_84_proof` | Ch07 laydown proof, Ch14 tournament challenge | Remaining legal continuations under exact state; every defender line loses all remaining tricks. | report-only/proof checker | enumeration-ready, oracle-ready, burl-ready | Not a heuristic feature; false positives are severe. |
| `protected_one_off_84_shape` | Ch07 canonical protected-off 84 | Trump count/rank, doubles count, exactly one off, double-ahead protection, final-off threat count. | online-computable for hand shape; report-only for make rate | enumeration-ready, oracle-ready, gus-ready | Does not assert the bid is supported without rollout. |
| `final_walker_counter` | Ch07 final off, Ch09 no-trump late off | Saved final off, higher same-suit live count, off suit exhaustion, outstanding trumps, trick index. | online-computable | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not smuggle hidden final defender hand into live feature. |
| `live_final_weapon_tiles` | Ch08 defender preservation | For each defender tile in report/full-world replay: can beat plausible final off, double/pair/protector class, forced-vs-voluntary break. | report-only for true owner; public proxy online | enumeration-ready, oracle-ready, gus-ready, burl-ready | Live policy cannot know which defender owns the weapon. |
| `live_same_suit_pair` | Ch08 double-ahead-off defense, Ch12 advanced 84 | Same-suit two-tile asset that can answer next-to-last double and final off, protector count, target suit still live. | report-only for owner; online public proxy for risk | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not treat all pairs as live once watched targets are dead. |
| `throwaway_ladder_rank` | Ch08 throwaway priority | Legal discard rank: live double, live pair, protector, non-winning tile, partner-readable low tile; forced break flag. | online-computable for own hand/action; report-only for opponent hidden assets | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not punish a player for being forced to follow. |
| `84_set_attribution` | Ch08 set causes | Completed-hand attribution: preserved double, preserved pair, partner rescue, bidder structure, forced break, voluntary blunder. | report-only | enumeration-ready, oracle-ready, gus-ready, burl-ready | Not a live feature. |

### Doubles-As-Trump And No-Trump

| detector family | chapter concepts | concrete public-state/action-local features | mode | readiness | non-goal |
|---|---|---|---|---|---|
| `declaration_regime_compare` | Ch09 paired doubles/no-trump/pip-trump comparison | Same hand under candidate regimes: doubles count, high-double coverage, off risk, support doubles, first-loss budget, suit-top changes. | report-only for paired declaration comparison | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not choose declaration from one final declaration record alone. |
| `doubles_native_suit_removal` | Ch09 doubles as trumps, Ch13 fallback variant | Declaration regime, legal suit membership, non-double suit-top recomputation, fallback-follow flag. | online-computable | enumeration-ready, gus-ready, burl-ready | Does not contaminate no-trump or straight pip-trump rules. |
| `dual_suit_top_protection` | Ch09 non-double tops under doubles-trump | Non-double tile that is high on both native suits after doubles are removed, off/protection side labels. | online-computable | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not treat doubles as native suit blockers in doubles-trump. |
| `planned_low_double_sacrifice` | Ch09 deliberate first-trick loss, Ch12 low-trump-first exception | Low double lead, missing higher doubles, planned first-loss points, remaining bid margin, future suit exhaustion. | online-computable for action class; report-only for correctness | oracle-ready, gus-ready, burl-ready | Not every low double lead is strategic. |
| `no_trump_support_double_preservation` | Ch09 no-trump play, Ch08 preservation reuse | No-trump regime, support doubles ahead of offs, late-off ordering, support double spent/preserved. | online-computable | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not use doubles-as-trump follow logic. |
| `dynamic_no_trump_suit_counter` | Ch09 no-trump suit depletion | Remaining public count by off suit, played same-suit tiles, support double/off order, late walker candidate. | online-computable | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not infer private suit exhaustion beyond public evidence unless in report mode. |

### Scoring And Ruleset

| detector family | chapter concepts | concrete public-state/action-local features | mode | readiness | non-goal |
|---|---|---|---|---|---|
| `score_mode_objective` | Ch10 points vs marks, Ch13 high-bid scoring | Scoring mode, bid class, mark multiplier, captured points, match score, terminal threshold. | online-computable | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not compare policies without matching objective labels. |
| `mark_terminal_state` | Ch10 early terminal under marks | Points captured so far, maximum remaining points, bid target, made/set/live status by trick. | online-computable | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not apply to point-scored hands. |
| `point_mark_policy_divergence` | Ch10 objective drift | Same decision rescored under point utility and mark utility; action argmax/tie divergence. | report-only | oracle-ready, gus-ready, burl-ready | Not a live tag unless the utility being optimized is explicit. |
| `rule_variant_gate` | Ch13 variants, Ch14 strict tournament regime | Ruleset id, allowed bid types, direct 84 flag, small-end disallowance, fallback-follow flag, no-trading invariant, tournament high-42 mode. | online-computable | enumeration-ready, gus-ready, burl-ready | Does not train straight policy on variant records. |
| `straight_auction_legality` | Ch01 auction order, Ch13 direct 84 and raise grammar | Minimum bid, one chance, direct 84 legality, 84 raise increment mode, forced fourth-seat 30 if enabled. | online-computable | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not conflate legal bid grammar with strategic bid quality. |
| `high_bid_scoring_invariant` | Ch13 42/84/higher scoring | Made/set high bids score the bid amount or mark multiplier, not bid plus captured points. | online-computable/report fixture | enumeration-ready, gus-ready, burl-ready | Not an empirical strategy claim. |

### Style, Belief, And Trace Discipline

| detector family | chapter concepts | concrete public-state/action-local features | mode | readiness | non-goal |
|---|---|---|---|---|---|
| `legal_inference_boundary` | Ch11 anti-leakage, Ch06 public inference | Feature audit: hidden partner/opponent facts must be derivable from bids, legal plays, failures to follow, and public trick history. | report-only audit; online metadata gate | enumeration-ready, gus-ready, burl-ready | Not a strategy feature that reveals private hand facts. |
| `early_trick_belief_discovery` | Ch06/Ch11/Ch14 watchfulness | Public reveal count, void events, failed bids, donations/non-donations, owner entropy delta, belief calibration bucket. | report-only for belief metrics; online for public reveal tags | oracle-ready, gus-ready, burl-ready | Does not treat true owner as a live input. |
| `trace_public_evidence_faithfulness` | Ch11 Burl trace discipline, Ch15 believable table play | Trace claims about ownership, voids, support, count, or trump must cite public/tool evidence; unsupported-private-claim rate. | report-only | burl-ready | Does not reward plausible wording over legal public evidence or action quality. |
| `dot_count_discipline` | Ch15 dot counting, Ch01 count identity | Points captured, points needed to make/set, live count remaining, count-at-risk in candidate action, trace arithmetic checks. | online-computable; report-only for trace math | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not test style by itself. |
| `overbid_restraint_bucket` | Ch15 overbid restraint, Ch02 bidding risk | Bid aggression residual relative to hand risk, auction context, score, and make-probability bucket. | report-only until bid calibration exists | oracle-ready, gus-ready, burl-ready | Raw bid height is not a style label. |
| `partner_fit_residual` | Ch14/Ch15 partner synergy | Fixed vs random partner ids, support-window hit rate, donation precision, team EV residual beyond individual ratings. | report-only | oracle-ready, gus-ready, burl-ready | Not measurable from one independent hand or without repeated identities/synthetic policies. |
| `style_prior_bucket` | Ch06 style, Ch12 reputation, Ch15 celebrities | Player/policy id, historical bid aggression, low-game use, set pressure, count accuracy, partner legibility. | report-only | oracle-ready later, burl-ready later | Not live-safe unless style history is legally available in the scenario. |

### Odds And Statistical Calibration

| detector family | chapter concepts | concrete public-state/action-local features | mode | readiness | non-goal |
|---|---|---|---|---|---|
| `hand_shape_prior_check` | Ch16 exact hand counts | Deal distribution by void count, suit coverage, double count, modal two-doubles/one-void bucket. | report-only corpus sanity; online prior tag if desired | enumeration-ready, gus-ready, burl-ready | Exact priors do not prove action optimality. |
| `four_trump_boss_first_threshold` | Ch16 10/27 missing-trump risk | Four-trump hand with double, missing second-highest, partner/opponent distinction, count/bid margin context. | online-computable for hand state; report-only for 27-case/e_q | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not collapse partner holdings into setter threats. |
| `four_trump_missing_top_two_threshold` | Ch16 14/27 near-50/50 case | Four-trump hand missing second and third, count-in-trump flag, bid margin, off protection. | online-computable for hand state; report-only for value | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not claim the near-50/50 prior settles play value. |
| `partner_double_help_prior` | Ch16 partner help, Ch02 partner-dependence | Conditional probability partner holds useful doubles after removing bidder hand, auction/bid context. | report-only or online probabilistic public prior | enumeration-ready, oracle-ready, gus-ready, burl-ready | Does not know partner's actual hand live. |
| `odds_rationalization_faithfulness` | Ch16 Burl odds explanations | Trace probability statements compared with exact prior table and tool observations. | report-only | burl-ready | Does not require Burl to quote exact odds unless it cites them. |
| `rare_extreme_double_bucket` | Ch16 rare hands | Five-plus-double hands, seven-doubles reshuffle extreme, oversampled eval bucket id. | report-only for corpus/eval | enumeration-ready, oracle-ready, gus-ready, burl-ready | Not expected to appear often in natural corpora. |

## Coverage Notes By Chapter

| chapter | v1 coverage | primary buckets | readiness notes |
|---|---|---|---|
| Ch01 In a Nutshell | Rule substrate for count, suit/trump, legal follow, trick winner, make/set, lead control, walkers. | scoring/ruleset, donation/pounce substrate, odds substrate | Mostly online-computable and enumeration-ready; walker value is oracle-ready. |
| Ch02 Bidding | Bid risk budget, side-specific off risk, double-ahead protection, duplicate exposure, partner-help dependency, bid-only-enough. | bidding, odds | Online-computable static labels; bid correctness needs oracle and score context. |
| Ch03 Bidder Play | Trump pull defaults, early-off exceptions, reentry trump, partner donation, pounce creation, dangerous trump, laydown proof. | donation, pounce, bidding, 84 precursor | Public/action labels are ready; sequence value and claim proof need oracle/enumeration. |
| Ch04 Partner Support | Safe count donation, low-trump traps, lead-away-from-count, effective double, partner trump-lead exceptions. | donation, scoring, style/partner | Donation guarantee is online-computable; hidden partner count is report-only. |
| Ch05 Setter Defense | Pounce windows, deliberate void creation, extra count to set, count-before-certainty, trump set, count protection. | pounce, donation, scoring | Pounce opportunity can be public-proxy online; attribution and regret are report-only/oracle. |
| Ch06 Concentration Style | Void inference, failed-bid signals, trump-memory traps, promoted tiles, unsafe count after trump-in, score/style shifts. | style, donation, pounce | Public evidence tags are online-computable; belief/style metrics are report-only. |
| Ch07 Taking Every Trick / 84 | 84 regime, laydown proof, protected one-off 84, final walker, straight-off risk, score gate, partner double support. | 84, scoring, odds | Regime and shape tags are online-computable; proof/risk claims need enumeration/oracle. |
| Ch08 Setting 84 | 84-defense mode, live final weapons, live doubles/pairs, protectors, throwaway ladder, set attribution. | 84, style/partner | Many live asset truths are report-only for opponent hands; own-hand/action tags are online-safe. |
| Ch09 Doubles No Trump | Doubles-as-trump suit removal, paired regime comparison, low-double sacrifice, no-trump support doubles, dynamic suit counting. | no-trump/doubles, bidding, 84-like preservation | Rules are enumeration-ready; declaration and play value need paired oracle reports. |
| Ch10 Tournament Scoring | Point vs mark objective, early terminal marks, partial-point erasure, set compression, mark multipliers. | scoring | Online-computable scoring tags; action divergence is report-only/oracle. |
| Ch11 Table Talk | Legal inference boundary, anti-leakage, renege detection, early-trick belief discovery, trace public-evidence faithfulness. | style, ruleset | Leakage/private facts are not live features; they are audits and trace filters. |
| Ch12 Advanced Bidding/Playing | Crisis trump, protector stripping, maximum-damage leads, double-not-always-right, pounce urgency, advanced 84 preservation, natural bid anomalies. | bidding, pounce, donation, 84, style | Mostly exception buckets; deterministic windows are ready but correctness is oracle-heavy. |
| Ch13 Optional Variations | Ruleset gates, direct 84 legality, first-lead legality, small-end disallowance, high-bid scoring, variant contamination. | scoring/ruleset, no-trump/doubles, 84 | High-value enumeration guards; variants should not contaminate straight tags. |
| Ch14 History Tournaments | Strict tournament regime, memory/hand-reading, partner synergy, aggressive bidding culture, time pressure, app weakness. | style, scoring, bidding | Tournament rules are online metadata; population claims need repeated identities or synthetic cohorts. |
| Ch15 Celebrities Style | Watchfulness, overbid restraint, low-game willingness, dot-count discipline, partner fit, table-play believability. | style, bidding, scoring | Count discipline is deterministic; most style claims are report-only and context-limited. |
| Ch16 Statistical Odds | Exact hand priors, double-count priors, four-trump 10/27 and 14/27 thresholds, partner double-help prior. | odds, bidding, 84 | Enumeration evidence exists on the chapter page; strategy value still needs oracle reports. |

## Online Surface Versus Report Surface

Online-computable v1 features should be limited to:

- ruleset/scoring metadata and legal action facts;
- public auction, declaration, score, trick history, captured points, current trick winner,
  void evidence, and current player's hand;
- action-local properties of the candidate move: follows, wins, donates count, calls count,
  spends/protects trump, preserves/exposes count, changes lead, creates/uses a public
  window;
- public priors and public-proxy buckets such as double-count prior, partner bid signal,
  and visible reveal history.

Report-only labels may use:

- completed-hand replay, full deal, or hidden owner labels for evaluation;
- paired declaration/action counterfactuals and forge E[Q];
- exact enumeration tables and corpus distribution checks;
- Burl trace text, tool-call evidence, and public-evidence citations;
- repeated player/policy ids for style, partnership, and population reports.

## Non-Goals And Illegal/Private Boundaries

- Do not use true partner/opponent hands as live strategy features. Hidden owner labels are
  allowed only for reports, belief calibration, leakage audits, and oracle analysis.
- Do not encode illegal table talk, trump-identification assists, partner cues, physical
  signals, or Plunge-like preplay information as ordinary straight-42 features.
- Do not promote any detector to a supported claim. Detector existence is not evidence;
  the claim ledger stays unchanged until enumeration, oracle rollout, Gus/w42 probe, or
  Burl trace review runs.
- Do not hard-code book advice as a policy rule. The detector map names states, windows,
  and report slices so oracle/model evidence can agree or disagree.
- Do not mix straight 42, doubles-as-trump, no-trump, marks, 84, Nel-O, Sevens, Plunge,
  fallback-follow, or high-bid scoring variants without explicit ruleset gates.
- Do not modify Gus core model/training paths, Burl behavior, or forge oracle semantics
  from this bead.
- Do not infer stable player style, partner fit, or reputation from a single hand. Those
  are population/report concepts unless a legal repeated-identity context is present.
- Do not let exact odds claims imply action optimality. Odds buckets calibrate priors;
  strategy recommendations still need value/regret tests.

## Implementation Notes For A Later Bead

The first implementation bead should prefer a small public-safe schema over a maximal
feature dump:

1. Emit v1 detector ids, boolean/window flags, and numeric severities into concept buckets
   rather than trying to train on every chapter concept.
2. Keep `online_features` separate from `report_labels`.
3. Gate every row with `ruleset_id`, `scoring_mode`, `contract_regime`, and `chapter_bucket`.
4. Start with deterministic fixtures for ruleset/scoring/donation/pounce windows before
   touching oracle value reports.
5. Reuse the [[w42-report-template]] fields so W&B/HF, claim ledger impact, data inputs,
   commands, and commit SHA stay visible.

Suggested first implementation order:

1. `rule_variant_gate`, `score_mode_objective`, `84_contract_regime`.
2. `side_specific_off_risk`, `candidate_bid_loss_budget`, `unnecessary_bid_margin`.
3. `safe_partner_count_donation`, `setter_pounce_window`, `count_protection_throwaway`.
4. `final_walker_counter`, `doubles_native_suit_removal`,
   `no_trump_support_double_preservation`.
5. Report-only odds and trace audits: `hand_shape_prior_check`,
   `trace_public_evidence_faithfulness`, `odds_rationalization_faithfulness`.

## Checks Run For This Map Bead

No empirical training, enumeration, oracle rollout, Gus/w42 probe, Burl trace review, W&B
run, HF upload, table, figure, or dataset artifact was produced for this bead.

Exact commands/checks run:

```bash
git status --short --branch
bd show t42-csw6.8 --json
rg --files wiki/experiments | rg 'winning42-ch.*\.md$|w42-(claim-ledger|report-template|strategy-tags-v0)\.md$|winning42-strategy-measurement\.md$|gus-strategy-tags-probe\.md$'
sed -n '1,220p' wiki/AGENTS.md
sed -n '1,260p' wiki/entities/w42.md
sed -n '1,260p' wiki/experiments/w42-claim-ledger.md
sed -n '1,240p' wiki/experiments/w42-report-template.md
sed -n '1,260p' wiki/experiments/w42-strategy-tags-v0.md
sed -n '1,280p' wiki/experiments/winning42-strategy-measurement.md
sed -n '1,280p' wiki/experiments/gus-strategy-tags-probe.md
sed -n '1,260p' wiki/experiments/winning42-ch01-in-a-nutshell.md
sed -n '1,300p' wiki/experiments/winning42-ch02-bidding.md
sed -n '1,300p' wiki/experiments/winning42-ch03-bidder-play.md
sed -n '1,300p' wiki/experiments/winning42-ch04-partner-support.md
sed -n '1,300p' wiki/experiments/winning42-ch05-setter-defense.md
sed -n '1,320p' wiki/experiments/winning42-ch06-concentration-style.md
sed -n '1,340p' wiki/experiments/winning42-ch07-taking-every-trick-84.md
sed -n '1,340p' wiki/experiments/winning42-ch08-setting-84.md
sed -n '1,320p' wiki/experiments/winning42-ch09-doubles-no-trump.md
sed -n '1,320p' wiki/experiments/winning42-ch10-tournament-scoring.md
sed -n '1,320p' wiki/experiments/winning42-ch11-table-talk.md
sed -n '1,340p' wiki/experiments/winning42-ch12-advanced-bidding-playing.md
sed -n '1,320p' wiki/experiments/winning42-ch13-optional-variations.md
sed -n '1,320p' wiki/experiments/winning42-ch14-history-tournaments.md
sed -n '1,320p' wiki/experiments/winning42-ch15-celebrities-style.md
sed -n '1,360p' wiki/experiments/winning42-ch16-statistical-odds.md
find scratch -maxdepth 4 -type f | sort | rg 'w42|winning42|claim|strategy'
git rev-parse HEAD
date +%Y-%m-%d
jq . w42/strategy_tags_v1_map/detector_map.json >/tmp/w42-detector-map.json && wc -c /tmp/w42-detector-map.json
git diff --check -- wiki/experiments/w42-strategy-tags-v1-map.md w42/strategy_tags_v1_map/README.md w42/strategy_tags_v1_map/detector_map.json
git status --short --untracked-files=all
git diff --stat
git check-ignore -v w42/strategy_tags_v1_map/README.md w42/strategy_tags_v1_map/detector_map.json || true
rg -n "W&B links|HF links|Claim ledger impact|no claim-ledger change|online-computable|not-live-safe|Non-Goals|bidding|pounce|donation|84|no-trump|scoring|style|odds" wiki/experiments/w42-strategy-tags-v1-map.md
sed -n '1,220p' wiki/experiments/w42-strategy-tags-v1-map.md
sed -n '220,520p' wiki/experiments/w42-strategy-tags-v1-map.md
bd close t42-csw6.8 --reason "w42 strategy detector v1 chapter-derived detector map documented; W&B links: not applicable; HF links: not applicable; claim-ledger impact: no claim-ledger change"
bd show t42-csw6.8 --json
bd export --no-memories -o /tmp/w42-csw6-8-issues-export.jsonl
node -e 'const fs=require("fs"); const id="t42-csw6.8"; const src="/tmp/w42-csw6-8-issues-export.jsonl"; const dst=".beads/issues.jsonl"; const replacement=fs.readFileSync(src,"utf8").split(/\n/).find(line=>line.includes(`"id":"${id}"`)); if(!replacement) throw new Error(`missing ${id} in export`); const lines=fs.readFileSync(dst,"utf8").split(/\n/); let count=0; const out=lines.map(line=>{ if(line.includes(`"id":"${id}"`)){ count++; return replacement; } return line; }).join("\n"); if(count!==1) throw new Error(`expected one ${id} line, found ${count}`); fs.writeFileSync(dst,out); console.log(`replaced ${id}`);'
```

Run/artifact fields:

| field | value |
|---|---|
| empirical run | not applicable |
| configs | not applicable |
| data inputs | wiki pages listed in the exact commands above |
| commit SHA at map drafting | `3a5575fc319b3e347f2537acdd3843777e9eefb0` |
| random seeds | not applicable |
| W&B links | not applicable |
| HF links | not applicable |
| claim ledger impact | no claim-ledger change |

## Links

[[w42]] | [[w42-strategy-tags-v0]] | [[w42-claim-ledger]] |
[[w42-report-template]] | [[winning42-strategy-measurement]] |
[[gus-strategy-tags-probe]] | [[gus]] | [[burl]] | [[forge]]
