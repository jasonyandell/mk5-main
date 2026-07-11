---
title: Winning 42 Ch14 History Tournaments
kind: experiment
first_seen: local-2026-05-01
last_updated: afd4802
status: retired
---

**Phantom plan.** This one-session chapter harvest (2026-05-01) registered 8
measurement targets and got zero empirical follow-up in the two months
since — none of its detectors (`strict_tournament_regime`,
`belief_memory_curve`, `partner_synergy_residual`,
`aggressive_bidding_calibration`, `laydown_challenge_proof`,
`setter_partner_app_gap`) were ever built, and none of its claim ids appear
in the central 64-row ledger ([[w42-phase4-final-claim-audit]]). The
project's actual frontier moved through the phase-4 closure and on to the
[[champion]]/[[jud]] ladder.

## Summary

This page is the bead-backed work surface for `t42-ni1l.14`: Chapter 14, "The Lone
Star Domino Phenomenon / History and Tournaments." It harvests tournament and
population-ecology concepts from Winning 42 into measurable hypotheses for [[gus]],
[[burl]], and [[forge]].

Chapter 14 is not a normal tactics chapter. It is the book's evidence that 42 is a
population game: local cultures, family training loops, repeated partners, tournament
rules, time pressure, and online-app quality all shape which skills matter. The useful
ML move is to turn those claims into evaluation regimes and population metadata rather
than hard-coded play advice.

## Source Anchors

The chapter slice is `scratch/winning42/winning42.with_figures.md` lines 6826-8914.

- The chapter frames 42 as a Texas-local culture that spread through families,
  rainy-day gatherings, domino halls, and church/social groups, but remained weak
  outside Texas except where Texans carried it. (`winning42.with_figures.md:6849-6894`,
  `6902-6914`, `6931-6957`)
- Several anecdotes name memory, hand-reading, anticipation, and tracking every play
  as the marks of strong players. (`7004-7007`, `7191-7195`, `7363-7370`, `7552-7555`)
- Several anecdotes name partner cohesion or repeated-pair synergy as a source of
  strength, including Calvin and John Jackson's long run and later Roberson/Jackson
  tournament success. (`7197-7238`)
- Tournament culture introduces a standardized, high-42 rule regime: no Nel-O,
  no variants, marks scoring, no table talk after trump declaration, no signaling,
  lay-down challenge rules, slow-play bans, no spectators, no forced bidding, and
  time-limited rounds. (`8226-8234`, `8404-8564`)
- The state championship record is a usable historical graph of player, partner,
  town, year, placement, repeat appearance, and team continuity. (`8565-8871`)
- Online and mobile implementations are explicitly criticized as weak setters,
  poor partner helpers, and risky bidders; this is directly aligned with the current
  [[gus]] and [[burl]] question of concept-specific evaluation rather than aggregate
  bot-match alone. (`8315-8349`)

## Concept Table

| Concept | Detector/state inputs | Metric/test | Likely data source | Priority | Implementation notes | Readiness |
|---|---|---|---|---|---|---|
| Texas-local population prior | Player/home region, venue, platform, local rule set, tournament id | Skill/style variance by region; out-of-population regret delta; transfer loss between local-style buckets | Championship list, tournament metadata, online-site surveys, generated style cohorts | P3 | Mostly metadata. Useful when real human logs or public tournament data exists; not a core forge state detector. | enumeration: no; oracle: no; Gus: later; Burl: later |
| Intergenerational apprenticeship | Player age, teacher relation, years played, repeated-family group, practice frequency | Learning-curve slope; error-rate reduction by exposure; youth/novice robustness under strict rules | Human logs, app telemetry, synthetic novice/expert agents | P3 | The chapter supplies anecdotes, not measurements. Treat as corpus annotation if real human data appears. | enumeration: no; oracle: no; Gus: later; Burl: traces later |
| Public memory and hand-reading skill | Trick history, void evidence, played tiles, bids/passes, prior donations/non-donations | Owner-belief Brier/log loss; entropy reduction after each trick; regret delta when belief is updated | Gus belief corpora, forge deal records, Burl tool traces | P1 | This is the strongest bridge from the history chapter to [[gus]]: "remember every play" becomes belief calibration over public evidence. | enumeration: partial; oracle: yes; Gus: ready; Burl: ready |
| Partner synergy residual | Fixed partner id, random partner id, bid aggression, partner donation windows, support leads, missed rescues | Team EV residual beyond individual skill; make/set rate under fixed vs random partner; support-action precision/recall | Self-play with seeded style agents, Burl traces, future human/tournament logs | P1 | Model repeated-pair advantage as interaction effect, not magic. Use chapter anecdotes as hypotheses for partner-legibility features. | enumeration: no; oracle: rollout-ready; Gus: later; Burl: ready |
| Offensive bidding culture | Auction position, bid/no-bid choice, hand strength tags, off-risk, score context | Underbid regret; overbid tail loss; contract-make rate by aggression bucket; tournament advancement by aggression | Generated games with bidding logs, oracle rollouts, Burl decision traces | P1 | Anderson's "if you don't bid, you can't win" and Hencerling's aggressive comeback become calibration tests, not blanket rules. | enumeration: partial; oracle: ready; Gus: if bidding data exists; Burl: ready |
| Adaptive style vs regimented rules | Hand shape, trump double ownership, off count, candidate bid, local heuristic fired | Regret of rigid folk rules; exception-recognition precision; bucket shift when model receives strategy tags | Forge/Gus decision corpora, chapter 2/12 detectors, Burl traces | P1 | St. Clair criticizes never bidding without the trump double and never bidding above 30 with 4/5/6 off. This is a prime contrast-pair generator. | enumeration: partial; oracle: ready; Gus: ready; Burl: ready |
| Tournament time pressure | Round type, game clock, tricks remaining, score/marks, bid margin, slow-play flag | Fast-decision regret; lay-down frequency; timeout/slow-play incidence; regret under reduced thinking/tool budget | Simulated time budgets, Burl token/tool traces, tournament logs | P2 | Map 25-minute qualifying games and 75-minute matches to inference-budget stress tests. | enumeration: no; oracle: rollout-ready; Gus: later; Burl: ready |
| Standardized high-42 regime | Rule flags: no Nel-O, no variants, no forced bid, doubles/no-trump behavior, minimum bid, marks scoring | Legality correctness by rule flag; contamination rate from variant behaviors; score-objective divergence | Rule-engine tests, forge variants, Burl legality traces | P1 | Chapter 14 gives the tournament baseline that should gate chapter 13 variant tests. | enumeration: ready; oracle: ready; Gus: ready; Burl: ready |
| Mark-scoring and tiebreak pressure | Marks, game-to-seven state, total marks tiebreaker, bracket status, bid value | Policy divergence from point scoring; late-match bid escalation/restraint; tiebreaker EV | Forge scoring simulations, chapter 10 buckets, tournament formats | P2 | Extends chapter 10 into bracket advancement objective, especially total marks as first tiebreaker. | enumeration: partial; oracle: rollout-ready; Gus: later; Burl: ready |
| Etiquette and anti-signaling boundary | Bidding utterance, post-declaration talk, tile touch/slam/toss, spectator presence, accidental exposure | Illegal information leakage rate; false-positive signal flags; trace mentions of non-public evidence | Tournament rules, Burl thoughts/tool traces, possible UI logs | P2 | Links to chapter 11: legal public inference is useful; signaling or table-talk leakage should be impossible in training/eval traces. | enumeration: ready for rules; oracle: no; Gus: no; Burl: ready |
| Lay-down proof under tournament challenge | Remaining legal lines, bidder claim point, possible defender counterline, bid margin | False lay-down rate; missed lay-down rate; proof search correctness; time saved vs forfeiture risk | Forge exhaustive endgame search, Burl claim traces | P1 | The tournament rule makes lay-down an explicit proof obligation: if any set line exists, the bidder forfeits. | enumeration: ready; oracle: ready; Gus: later; Burl: ready |
| Setter and partner-app weakness | Setter pounce windows, partner donation windows, bid risk, reckless partner bids | Concept regret vs app/baseline; missed setter pounce rate; unsafe partner-bid rate | Gus/Burl eval buckets, future app benchmark logs | P1 | The chapter directly names weak setters, weak partner helpers, and risky bidders as online-app failures, matching the strategy-tags agenda. | enumeration: partial; oracle: ready; Gus: ready; Burl: ready |
| Historical championship graph | Year, placement, player, partner, town, repeat team, repeat opponent | Repeat-finalist centrality; partner persistence; town clusters; era drift; cross-region success | Extracted table from lines 8565-8871 plus external tournament records | P2 | Build a small structured dataset from the book list before seeking external records. This is descriptive, not move-policy supervision. | enumeration: no; oracle: no; Gus: no; Burl: no |
| Random/fixed partner robustness | Blind draw flag, fixed partner flag, partner familiarity, team result | Performance drop under blind draw; partner-legibility robustness; support-action resilience | Cafe/club/tournament formats, self-play with random partner assignment | P2 | Dugan's local blind draw and Crow's stranger-partner win suggest testing models under partner uncertainty. | enumeration: no; oracle: rollout-ready; Gus: later; Burl: ready |

## First Detectors And Eval Buckets

1. `strict_tournament_regime`: emit rule flags for high 42, no low/Nel-O, no forced bidding,
   marks-to-seven, no post-trump table talk, no signaling, and no spectators. This should
   be a baseline gate for every other chapter bucket before variants are allowed.
2. `belief_memory_curve`: measure hidden-owner calibration after each trick using only public
   play history, bids, and void evidence. This operationalizes the chapter's repeated claim
   that great players remember every play and read hands.
3. `partner_synergy_residual`: compare fixed-pair rollouts against random-partner rollouts
   after controlling for individual policy strength. The output is a team interaction residual,
   not a mystical partner tag.
4. `aggressive_bidding_calibration`: bucket bid/pass decisions by hand strength, off-risk,
   auction pressure, and score context, then report underbid regret, set tail risk, and
   tournament advancement proxy.
5. `laydown_challenge_proof`: for late-hand states, determine whether a proposed lay-down
   has any defender set line. This is immediately checkable by exhaustive search on small
   remaining-state trees.
6. `setter_partner_app_gap`: evaluate models and baselines on chapter 4/5 buckets: missed
   partner donation, unsafe partner bid, missed setter pounce, and reckless overbid.

## Analysis Routes

- Enumeration now: strict tournament legality/ruleset tests, anti-signaling rule checks,
  lay-down proof search in constrained late-hand states, and score/mark accounting.
- Oracle rollouts: aggressive-bidding calibration, fixed-vs-random partner deltas, mark
  tiebreaker policy shifts, and setter/partner failure buckets.
- Gus-ready: belief-memory curves, public-state strategy tags for adaptive-vs-regimented
  decisions, and setter/partner concept labels where decision corpora already contain the
  needed public state.
- Burl-ready: tournament time-budget experiments, anti-leakage trace audits, strict-regime
  legality/tool-use checks, partner-support/setter-pounce rationales, and lay-down proof
  explanations.
- Later/data-dependent: Texas-local population priors, intergenerational apprenticeship,
  real tournament style drift, and the championship graph beyond the book's static table.

## Claim Ledger

No empirical run was performed for this chapter harvest. Claims are therefore not marked
supported or contradicted; they are converted into measurable hypotheses.

| Claim | Source | Status | Next check |
|---|---|---|---|
| Strong players distinguish themselves by memory, hand-reading, and anticipation. | `winning42.with_figures.md:7004-7007`, `7191-7195`, `7363-7370`, `7552-7555` | underpowered | Run Gus owner-belief calibration by trick depth and void-evidence events. |
| Repeated partners can create a measurable advantage beyond individual strength. | `7197-7238`, `7963-7978` | underpowered | Run fixed-pair vs random-pair self-play with matched policies and compare team residuals. |
| Tournament 42 is a standardized high-42/no-variant regime. | `8226-8234`, `8404-8564` | context-limited | Encode the Hallettsville-style rule profile and verify legality/scoring coverage. |
| Aggressive bidding can be tournament-correct after a setback. | `7970-7978`, `7980-7984` | underpowered | Bucket bid/pass decisions by score, bid margin, and bracket objective; compare oracle EV and advancement proxy. |
| Rigid folk bidding rules miss valuable exceptions. | `7679-7687` | underpowered | Build contrast pairs for "requires trump double" and "never above 30 with 4/5/6 off" heuristics. |
| Current online/mobile bots are weak at setter play, partner support, and bid restraint. | `8315-8349` | underpowered | Compare app-like baselines, Gus, Burl, and E[Q] on chapter 4/5 concept buckets. |
| Tournament lay-down is a proof problem, not a confidence statement. | `8532-8538` | context-limited | Implement late-hand proof search: any defender set line invalidates the claim. |
| The championship list can seed a descriptive player-partner-town graph. | `8565-8871` | context-limited | Parse years, placements, players, partners, and towns; report repeat teams and regional clusters. |

## Links

[[winning42-strategy-measurement]] - [[gus-strategy-tags-probe]] - [[gus]] - [[burl]] - [[forge]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- Cheapest live probe remaining is `laydown_challenge_proof`: exhaustive late-hand set-line search overlaps existing forge endgame enumeration and could piggyback on [[w42-phase4-laydown-rule-accounting]].
- Source slice `scratch/winning42/winning42.with_figures.md` is gitignored, so it exists only in the main working tree and is absent from git and worktrees; anchor citations trace only against that local file.
- Bead id `t42-ni1l.14` is unverifiable (beads retired 2026-06, old beads grep-able only in `.beads/issues.jsonl`).
