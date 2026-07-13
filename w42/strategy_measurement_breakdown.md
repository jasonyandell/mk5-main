# Texas 42 Strategy Measurement Breakdown

Source: `scratch/winning42/winning42.with_figures.md`, preview pass by one agent per chapter.

Thesis: treat the book as a hypothesis generator, not an oracle. Each concept should become one or more detector labels, adversarial buckets, regret metrics, belief-calibration tests, or training examples for Gus/Burl.

## Project Shape

1. Build a strategy ontology
   - Chapter tags: bidding, bidder play, partner support, setting, attention/inference, 84, doubles/no-trump, scoring, etiquette, advanced exceptions, variants, tournament ecology, player style, odds.
   - Concept tags: `risk_budget`, `off_risk`, `count_liability`, `trump_control`, `reentry`, `void_inference`, `partner_support`, `setter_pounce`, `84_preservation`, `walker`, `rule_variant`, `score_context`, `player_style`.

2. Implement detector substrate
   - Deterministic state detectors first: legal follow, trick winner, live count, count already dead, outstanding trump, effective double, walker, bid made/set threshold.
   - Belief detectors second: void probability, key-tile owner distribution, partner support likelihood, bidder off vulnerability, trump concentration.
   - Regime detectors third: ordinary bid, partner-helping-bidder, setting bidder, 84 bidder, 84 defender, doubles-as-trump, no-trump, marks scoring, variant rules.

3. Validate book claims empirically
   - Exhaustive enumeration where possible.
   - Oracle rollouts for tactical claims.
   - Public-belief rollouts for Gus-shaped hidden-state claims.
   - Mark claims as supported, contradicted, context-limited, or untested.

4. Audit Gus/Burl by concept
   - Gus: belief calibration, owner localization, hidden-hand entropy reduction, concept-tag accuracy.
   - Burl: paired regret, tail regret, forced-commit errors, concept-specific thought/tool faithfulness.
   - Report concept buckets before aggregate scores.

5. Use concepts for training
   - Generate contrast pairs: normal rule vs exception, locally tempting move vs strategic move, unsupported folk rule vs empirically supported line.
   - Prefer explanations that cite public evidence and detector facts.
   - Avoid live strategy hints until detectors are validated; use tags first for eval and curriculum.

## High-Value MVP Detectors

1. `bid_risk_budget`
   - Inputs: hand, candidate trump, offs, exposed count, partner-help windows, current auction.
   - Metrics: bid EV calibration, set rate by bucket, overbid/underbid regret.

2. `off_risk_and_protection`
   - Inputs: off tile sides, matching doubles, live count, already played/in-hand/trump count.
   - Metrics: protected vs unprotected loss delta, false safety rate, tail set risk.

3. `count_liability_surface`
   - Inputs: legal leads, callable count, likely winner distribution, count already dead.
   - Metrics: expected count donated/captured, catastrophic count-dump rate.

4. `trump_control_and_reentry`
   - Inputs: outstanding trumps, highest missing trump, bidder trumps, offs remaining.
   - Metrics: regret of pulling/spending final trump, crisis-management success.

5. `partner_count_donation`
   - Inputs: partner currently winning, guarantee strength, count in hand, opponent trump risk.
   - Metrics: good donation rate, unsafe donation tail loss.

6. `setter_pounce`
   - Inputs: bidder off lead, defender void/winner status, count available, bid margin.
   - Metrics: missed pounce rate, set conversion, held-count-never-used rate.

7. `void_and_owner_inference`
   - Inputs: failures to follow, bids/pass history, donations/non-donations, trick history.
   - Metrics: owner Brier/log loss, entropy reduction, regret improvement after belief update.

8. `effective_double_or_walker`
   - Inputs: suit exhaustion, remaining higher tiles, trumps remaining, trick depth.
   - Metrics: recognition precision/recall, endgame regret, final-walker prediction.

9. `84_bidder_plan`
   - Inputs: trump count/rank, doubles, protected offs, final-off suit threats.
   - Metrics: lay-down proof, protected-off make probability, straight-off set risk, 42-vs-84 score gating.

10. `84_defender_preservation`
    - Inputs: live doubles, same-suit pairs, protectors, forced breaks, dead assets.
    - Metrics: voluntary discard regret, final weapon recall, pair survival, set attribution.

11. `regime_and_rule_variant`
    - Inputs: ruleset, bid type, scoring mode, variant flags.
    - Metrics: legality correctness, point-vs-mark policy divergence, variant contamination checks.

12. `style_and_partnership`
    - Inputs: repeated player/team identities, bid aggression, low-bid use, defensive skill, partner legibility.
    - Metrics: partner synergy residual, style-conditioned regret, tournament advancement probability.

## Chapter Harvest Map

- Ch1: foundations and rule/term detectors.
- Ch2: bidding as risk budget; off/count/trump candidate evaluation.
- Ch3: bidder sequencing; trump exhaustion, reentry, off timing, count budget.
- Ch4: helping partner make bid; safe donation, lead control, count liability.
- Ch5: setting bidder; pounce, void creation, count pressure, trump-set detection.
- Ch6: concentration/style; public inference, attention traps, score/style-conditioned decisions.
- Ch7: making 84; proof-like plans, protected offs, final walkers, score-aware escalation.
- Ch8: setting 84; asset preservation, live doubles/pairs, defender endgame.
- Ch9: doubles/no-trump; regime switch, planned losses, support doubles, dynamic suit counters.
- Ch10: tournament scoring; point-vs-mark reward shift and early terminal thresholds.
- Ch11: table talk; anti-leakage, legal inference, renege and attention tests.
- Ch12: advanced exceptions; crisis management, double-not-always-right, count forcing.
- Ch13: variants; ruleset gates, legality traps, contamination guards.
- Ch14: tournament ecology; setter skill, aggressive calibration, partner synergy, belief memory.
- Ch15: celebrity/player style; overbid restraint, low-game willingness, dot counting, watchfulness.
- Ch16: statistical odds; enumerate claims, hand-shape priors, probability calibration.

## Analysis Catalog

This is the broader inventory of book-derived things we can analyze. The first pass should be generous: some items become detectors, some become eval buckets, some become training tags, and some only become "unsupported folk rule" entries after the oracle disagrees.

### 1. Rules And State Accounting

- Legal follow-suit detection.
- Led-suit detection under pip trump, doubles trump, doubles-as-suit, no trump, and optional variants.
- Trick-winner resolution.
- Trick point accounting.
- Hand point total accounting.
- Count-domino identity and live/dead status.
- Bid made threshold.
- Bid set threshold.
- Maximum remaining points for each team.
- Earliest mathematically terminal trick.
- Legal first lead, including non-trump first leads.
- Illegal small-end lead interpretation.
- Direct 84 bid legality.
- High-bid scoring rule: award bid value, not bid plus captured points.
- Renege detection at play time.
- Delayed renege reconstruction after later information contaminates the hand.
- Claim/lay-down proof: can the player prove all remaining tricks?

### 2. Bidding Analysis

- Candidate trump suit scoring.
- Backward bidding from expected losses.
- Bid floor vs hand ceiling.
- Bid only enough to win the auction.
- Overbid vs underbid regret.
- Natural bid buckets: 30, 31, 35, 36, 42, 84, 126.
- Odd bid detection: when 32/33/34-style bids are meaningful vs noise.
- Auction position effects.
- Last-seat forced or strategic bid effects.
- Partner prior-bid signal.
- Failed-bid signal.
- Opponent prior-bid signal.
- Score-aware bid escalation.
- Score-aware bid restraint near game point.
- Bid-risk distribution, not just expected bid value.
- Make probability vs expected point value.
- Tail set risk for aggressive bids.
- Partner-help dependency rate.
- Hand shapes that are biddable only with partner help.
- Overcall 84/126/Game escalation thresholds.
- Low-game willingness if variants are enabled.

### 3. Hand-Shape Analysis

- Trump count.
- Trump rank coverage.
- Highest missing trump.
- Missing second-highest trump.
- Four-trump special cases.
- Four trumps with boss but missing next trump.
- Doubles count.
- Count points in hand.
- Count points exposed outside trump.
- Count points in trump.
- Off count.
- Off suit identity.
- Same-suit two-off shape.
- Multiple-off shape.
- Void suit count.
- Suit coverage count.
- Strong double plus weak off packages.
- Protected high tile.
- Weak tile with no reentry.
- Rare extreme double hands.
- Common one-void hand archetype.

### 4. Off-Risk And Protection

- Off-risk by both pips of an off tile.
- Count exposed by each off side.
- Duplicate count accounting when multiple offs expose the same count tile.
- Count already in hand, played, or trumped away.
- Double-ahead protection.
- Double-behind false protection.
- Companion tile protection.
- Protected-off make probability.
- Straight-off set risk.
- Off-first exception.
- Off timing after trump control.
- Off timing before opponents are void.
- Off ordering when two offs share a suit.
- Last-off paired-holder risk.
- Final-off walker proof.
- Opponent pounce risk when bidder leads off.
- Partner rescue probability on bidder off.

### 5. Count-Liability Analysis

- Live count by suit.
- Count that can be called by each candidate lead.
- Count at risk in hand.
- Count already secured.
- Count needed to make bid.
- Count needed to set bidder.
- Expected count donated by a legal play.
- Catastrophic count-dump tail risk.
- Ten-count attack opportunities.
- Five-count exposure vs ten-count exposure.
- Count-forcing doubles.
- Protection stripping before forcing count.
- Count donation to partner.
- Unsafe count donation to partner.
- Count pounce on bidder off.
- Holding count too long and never getting a chance.
- Count played after opponent has already won the trick.
- Count in trump as exception to ordinary lead rules.

### 6. Trump Control And Reentry

- Outstanding trump count.
- Outstanding higher trump count.
- Highest missing trump.
- Trump exhaustion plan.
- Whether bidder can pull all trumps safely.
- Save-one-trump-for-reentry cases.
- Spending last trump too early.
- Low-trump-first vs boss-trump-first.
- Sacrificial low trump to preserve command.
- Dangerous outstanding trump.
- Trump concentration on one opponent.
- Trump-set detector.
- Trump-rich setter strategy.
- Bidder crisis trigger when trump control is lost.
- Emergency trump-in to prevent count loss.
- Avoid leading trump while helping partner make bid.
- Exception where trump lead is least bad.

### 7. Lead Analysis

- Opening lead quality.
- Trump lead vs non-trump lead.
- Double lead value.
- Low lead value.
- Lead double ahead of off.
- Lead suit with maximum follow burden.
- Lead suit with minimum count liability.
- Lead that creates opponent voids.
- Lead that preserves partner's opportunity.
- Lead that forces specific count tile.
- Lead that strips protector.
- Lead that communicates useful public information.
- Lead timing under 84, no-trump, and doubles-as-trump regimes.
- Lead choice under score pressure.

### 8. Follow, Slough, And Discard Analysis

- Forced follow vs free discard.
- Free discard priority.
- Discard to create a void.
- Discard to protect count.
- Discard lower of two in a non-count suit.
- Discard dead double.
- Discard dead suit tile.
- Avoid discarding pair protectors in 84 defense.
- Voluntary vs forced break of a defensive pair.
- Signaling-like discard value without illegal communication.
- Partner-readable discard.
- Opponent-readable discard risk.

### 9. Partnership Analysis

- Partner-bid support regime.
- Partner as bidder vs partner as setter.
- Win trick to gain lead for partner support.
- Safe partner count donation.
- Partner count donation only on guaranteed trick.
- Partner count-sacrifice implying void.
- Partner rescue of vulnerable off.
- Partner double-saving rule for 84.
- Partner dead-double release.
- Partner decoy low tiles in 84.
- Partner legibility.
- Partner-help calibration in bidding.
- Partner synergy residual.
- Partner style compatibility.
- Unknown partner robustness.
- Teaching/coaching mode as separate regime.

### 10. Setter And Defense Analysis

- Setter role detector.
- Pounce window on bidder off.
- Missed pounce rate.
- Extra count requirement to set bidder.
- Count expected by bidder vs count that actually sets.
- Trump-set recognition.
- Trump-rich setter policy.
- Double-not-always-right cases.
- Low five-count caller trap.
- Lead only count-calling dominoes unless one point sets.
- Position-sensitive trump-in decision.
- Defenders' bid-risk inference.
- Setter skill gap vs offensive skill.
- Missed set-line rate.
- Bidder escape rate after defensive mistakes.

### 11. Belief And Inference Analysis

- Void inference from failure to follow.
- Bid-derived hand inference.
- Failed-bid inference.
- Partner donation inference.
- Non-donation inference.
- Seat-specific key-tile owner distribution.
- Hidden count ownership.
- Hidden trump ownership.
- Hidden double ownership.
- Bidder off vulnerability belief.
- Trump concentration belief.
- Early-trick belief update quality.
- Belief entropy by trick.
- Belief update after each reveal.
- Action regret before vs after belief update.
- Gus belief calibration by concept bucket.
- Burl thought faithfulness to public evidence.
- Illegal leakage probe: exact hidden tile claims not supported by public evidence.

### 12. Attention And Memory Analysis

- Trick-by-trick attention.
- Trump-suit retention under visually tempting non-trump pip.
- Low trump disguise traps.
- Late-hand promoted low domino.
- Effective double recognition.
- Full-history vs last-two-tricks memory ablation.
- Suit-exhaustion counters.
- Remaining higher tile counters.
- Dot-count discipline.
- Count math in thought traces.
- Impossible-score reasoning errors.
- Watchfulness/opponent-model quality.
- Repeated mistake classes.
- Learning-curve reduction by trap family.

### 13. 84 Bidder Analysis

- 84 eligibility classifier.
- Lay-down 84 proof.
- Protected-off 84.
- Straight-off 84.
- Two-off same-suit 84.
- Trump exhaustion before doubles.
- Next-to-last forcing double.
- Final walker prediction.
- Score-aware 42 vs 84 choice.
- 84 overcall thresholds.
- Bidder structure: straight off vs double-ahead off.
- Bidder endgame plan.
- Premature protective double.
- Opponent set-world frequency.
- Opponent adaptation to protected-off style.

### 14. 84 Defender Analysis

- 84-defense mode switch.
- Live last-trick weapons.
- Live doubles.
- Live same-suit pairs.
- Pair protectors.
- Pair vulnerability under follow-suit.
- Double preservation.
- Dynamic abandonment of dead assets.
- Throwaway priority ladder.
- Endgame asset bottleneck.
- Forced break vs voluntary blunder.
- Set attribution: preserved asset vs bidder error vs partner card.
- Tracking-load curriculum.
- Partner signaling in 84 defense.
- Suppression of ordinary count/trump incentives during 84 defense.

### 15. Doubles-As-Trump And No-Trump Analysis

- Doubles-as-trump declaration candidate.
- Doubles removed from native suits.
- Non-double suit top under doubles trump.
- Surface-loss overestimate.
- Low-double sacrificial lead.
- Planned-loss budget after sacrificing a trick.
- High off protected by dual-suit top.
- Trump-draw-to-walker plan.
- Draw specific count tile to neutralize risk.
- No-trump over doubles-trump candidate.
- No-trump first-trick count-dump risk.
- No-trump lead-control state.
- Save support doubles ahead of offs.
- Dynamic suit-count tracker under no-trump.
- Defender no-trump set plan.
- Rule-variant legality flag.

### 16. Scoring And Tournament Objective Analysis

- Points-to-250 vs marks-to-7 objective switch.
- Point EV vs mark EV.
- Early hand termination under marks.
- Nonbidder point-taking value erased by marks.
- Set severity erased by marks.
- Low-bid score distortion.
- Tournament speed incentive.
- Endgame pressure under points.
- Special bid mark multiplier.
- Timed match pressure.
- Bracket/round-robin pressure.
- Advancement probability rather than hand EV.
- Time-management strategy.
- Slow-play risk.
- Fatigue and momentum buckets.

### 17. Rule Variant And Etiquette Analysis

- Straight-42 vs variant mode.
- Nel-O eligibility and objective.
- Nel-O protected-high shape.
- Nel-O set path.
- Sevens objective.
- Sevens forced closest play.
- Plunge/Splash eligibility.
- Plunge as legal/illegal communication depending ruleset.
- Forced bidding after three passes.
- 84 raise increment variants.
- Doubles-trump follow-secondary-suit variant.
- Talking-across-the-board violations.
- Trump verbal-identification assistance.
- Unauthorized partner information.
- Post-hand discussion boundary.
- Coaching mode exception.
- Collusion/cue detection.
- Tournament infraction policy.

### 18. Model Evaluation And Training Analysis

- Gus strategy-tag accuracy.
- Gus belief quality by concept bucket.
- Gus policy regret by concept bucket.
- Burl regret by concept bucket.
- Burl forced-commit errors by concept bucket.
- Burl thought/tool faithfulness by concept bucket.
- Bot-match vs regret divergence.
- Near-tie rate by concept.
- Tail regret by concept.
- Concept-specific blunder detector.
- Strategy-tag feature ablation.
- Global tags vs action tags.
- Concept tags as curriculum selectors.
- Strategy tags as explanation scaffolds.
- Contrast pairs: normal rule vs exception.
- Unsupported folk-rule bucket.
- Human-readable rationale quality.
- Whether strategy tags help small models more than large models.

### 19. Population, Style, And Partnership Analysis

- Aggressive bidding style.
- Overbid restraint.
- Low-game willingness.
- Cautious vs wild bidder archetypes.
- Strong setter archetype.
- Dot-counter archetype.
- Memory-heavy archetype.
- Partner-dependent player.
- Opponent adaptation speed.
- Style-conditioned exploitability.
- Partnership synergy residual.
- Fixed partner vs random partner performance.
- Player rating and team rating separation.
- Tournament ecology and venue effects.
- Online vs local-club population shift.
- Repeat finalist / dynasty detection from historical tournament results.
- Teaching-pipeline improvement curves.

### 20. Statistical Analysis

- Verify the book's stated odds by exhaustive enumeration.
- Compare book odds to forge/oracle empirical outcomes.
- Hand-combination baselines.
- Shape-frequency tables.
- Trump-count distributions.
- Double-count distributions.
- Void-count distributions.
- Count-domino exposure distributions.
- Make/set probability by bid bucket.
- Make/set probability by hand-shape bucket.
- Regret distribution by concept bucket.
- Tail-risk distribution by concept bucket.
- Calibration curves for bid make probability.
- Calibration curves for Gus belief probabilities.
- Brier/log-loss for hidden tile ownership.
- Confidence intervals for strategy-claim effects.
- Bootstrap effect sizes for protected vs unprotected offs.
- Permutation tests for paired policy comparisons.
- McNemar tests for paired action-class flips.
- Survival analysis for count remaining live over tricks.
- Information value of bids, voids, and donations.
- Mutual information between public events and hidden hand features.
- Variance decomposition: hand shape vs play sequence vs belief uncertainty.
- Interaction effects: trump count x off risk, count risk x partner support, score x bid aggression.
- Robustness checks across declarations.
- Robustness checks across early/mid/late decisions.
- Multiple-comparison control for many book claims.
- Supported / contradicted / context-limited / underpowered claim ledger.

## First Work Package

Build a `strategy_tags` analyzer that can run over generated games and emit per-decision JSON:

- `role_regime`
- `risk_budget`
- `live_count`
- `count_liability`
- `outstanding_trumps`
- `void_evidence`
- `key_tile_owner_belief`
- `off_protection`
- `walker_candidates`
- `partner_donation_window`
- `setter_pounce_window`
- `rule_variant`

Then make a first report with three tables:

1. Book claims checked against enumeration/oracle.
2. Gus belief quality by concept bucket.
3. Burl regret/tail-risk by concept bucket.

The first valuable answer is not "does the model play well?" It is "which pieces of real 42 strategy does the model understand, fake, ignore, or invert?"
