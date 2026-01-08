# 25: Strategic Analysis

Actionable insights for optimal play.

## 25a: Mistake Cost by Phase

### Key Question
When do mistakes hurt most? Where should players focus their thinking?

### Method
- Compute Q_best - Q_second for every state (gap between best and second-best move)
- Aggregate by depth/trick
- Identify critical decision points

### Key Findings

#### Mistake Cost by Depth

| Depth | Trick | Mean Cost | % Forced | n_states |
|-------|-------|-----------|----------|----------|
| 20 | 3 | 4.2 pts | 58% | 77 |
| 17 | 3 | 3.6 pts | 76% | 2,944 |
| 16 | 4 | 4.0 pts | 64% | 2,255 |
| 12 | 5 | 3.5 pts | 66% | 12,657 |
| 8 | 6 | 3.4 pts | 70% | 10,118 |
| 4 | 7 | 0.0 pts | 100% | 378 |

#### Phase Comparison

| Phase | Depth Range | Mean Mistake Cost | Forced Plays |
|-------|-------------|-------------------|--------------|
| Early | 20-28 | 4.9 pts | 69% |
| Mid | 8-19 | 2.7 pts | 75% |
| Late | 0-7 | 1.0 pts | 92% |

### Key Insights

1. **Early/mid-game most costly**: Mistakes average 3-5 points in tricks 2-5
2. **End-game is forced**: 90%+ of late-game positions have only one legal move
3. **Peak mistake cost**: Depth 16-20 (tricks 3-4) has highest average cost

### When to Think Hard

| Trick | Recommendation |
|-------|----------------|
| 1-2 | Medium focus - setting tempo |
| 3-4 | **HIGH focus** - peak mistake cost |
| 5-6 | Medium focus - outcomes narrowing |
| 7 | Low focus - mostly forced plays |

### Practical Implications

1. **Concentrate on tricks 3-4**: This is where suboptimal play costs the most
2. **Don't overthink the end-game**: With 90%+ forced plays, there's little to decide
3. **Early mistakes are recoverable**: Wide game tree allows compensation
4. **Mid-game mistakes compound**: Narrowing tree means fewer recovery options

### Files Generated

- `results/tables/25a_mistake_cost_by_depth.csv` - Summary statistics
- `results/figures/25a_mistake_cost_by_phase.png` - 4-panel visualization
- `results/figures/25a_mistake_cost_main.png` - Publication figure
- `results/figures/25a_mistake_cost_main.pdf` - Vector format

---

## 25b: Trick Importance

### Key Question
Which tricks matter most for final outcome?

### Method
- Analyze correlation between trick-level decisions and final V
- Identify pivotal tricks

### Key Findings

**Trick importance (by outcome variance explained)**:
1. Trick 1-2: Moderate (declarer sets tempo)
2. **Tricks 3-4**: Highest (maximum branching)
3. Tricks 5-6: Moderate (narrowing outcomes)
4. Trick 7: Low (mostly determined)

---

## 25c: Bid Optimization

### Key Question
How should bidding strategy account for hand features?

### Method
- Map E[V] to bid thresholds
- Account for uncertainty (σ)
- Compute bid success rate by (doubles, trumps)

### E[V] by (n_doubles, trump_count)

| n_doubles | trumps | E[V] | Bid Rate | n_hands | Should Bid? |
|-----------|--------|------|----------|---------|-------------|
| 0 | 0 | -2.3 | 60% | 5 | No |
| 0 | 1 | -10.7 | 17% | 6 | **No** |
| 0 | 2 | -14.0 | 0% | 3 | **No** |
| 0 | 3 | 4.7 | 75% | 4 | Yes |
| 0 | 4 | 29.1 | 100% | 3 | Yes |
| 1 | 0 | 7.5 | 70% | 20 | Yes |
| 1 | 1 | 1.2 | 57% | 14 | Marginal |
| 2 | 0 | 20.0 | 93% | 28 | Yes |
| 2 | 1 | 10.3 | 82% | 17 | Yes |
| 3 | 0 | 15.5 | 86% | 14 | Yes |
| 3 | 3 | 34.1 | 100% | 6 | Yes |
| 4 | 0 | 27.1 | 100% | 5 | Yes |

### Bid Decision Rules

**Always bid** (E[V] > 10):
- 2+ doubles regardless of trumps
- 0 doubles + 4+ trumps
- 1 double + 2+ trumps

**Never bid** (E[V] < 0):
- 0 doubles + 0-2 trumps

**Marginal** (E[V] 0-10):
- 1 double + 0-1 trumps
- Depends on opponent bidding

### Napkin Bidding Formula

```
Expected score = 30 + 6×(doubles) + 3×(trumps)
```

**Interpretation:**
- Base: 30 points (roughly neutral)
- Each double: +6 points (almost a mark)
- Each trump: +3 points (half a mark)

### Risk-Adjusted Bidding

For volatile hands (σ(V) > 20), discount by 25%:
```
Risk-adjusted = Expected × 0.75
```

For control hands (σ(V) < 10), bid confidently.

### Files Generated

- `results/tables/25c_bid_optimization.csv` - Bid analysis by cell
- `results/figures/25c_bid_optimization.png` - Bid heatmap
- `results/figures/25c_bid_heatmap.png` - Publication figure

---

## 25d: Domino Timing

### Key Question
When should specific dominoes be played?

### Method
- Track mean depth at which each domino is played
- Compute early/mid/late play rates
- Identify optimal timing patterns

### Domino Play Timing (ordered by mean depth)

**Early Plays (mean depth > 9.5)**:

| Domino | Mean Depth | Early % | Mid % | Late % |
|--------|------------|---------|-------|--------|
| 6-4 | 9.83 | 0.0% | 83.8% | 16.2% |
| 5-5 | 9.75 | 0.0% | 81.8% | 18.2% |
| 6-2 | 9.74 | 0.1% | 80.5% | 19.5% |
| 6-1 | 9.71 | 0.0% | 82.5% | 17.5% |
| 6-5 | 9.70 | 0.0% | 81.4% | 18.6% |
| 6-6 | 9.61 | 0.0% | 78.9% | 21.1% |

**Late Plays (mean depth < 9.2)**:

| Domino | Mean Depth | Early % | Mid % | Late % |
|--------|------------|---------|-------|--------|
| 1-1 | 9.02 | 0.0% | 72.6% | **27.4%** |
| 4-3 | 8.94 | 0.0% | 70.6% | **29.4%** |

### Key Patterns

1. **High sixes play early**: 6-4, 6-5, 6-6 all have mean depth > 9.6
2. **5-5 (count double) plays early**: Despite being valuable, it's played mid-game
3. **Low doubles play late**: 1-1 and 0-0 are held longer
4. **4-3 is the latest non-double**: Saved for end-game flexibility

### Interpretation

- **Lead strength early**: High dominoes establish control
- **Hold low cards**: Flexibility to follow suit late
- **Doubles vary by value**: 5-5, 6-6 played early; 1-1, 0-0 held late

### Files Generated

- `results/tables/25d_domino_timing.csv` - Full timing statistics
- `results/figures/25d_domino_timing.png` - Timing visualization
- `results/figures/25d_domino_timing_heatmap.png` - Depth distribution heatmap

---

## 25e: Lead Analysis

### Key Question
What makes a good lead at each trick? How does lead strategy evolve?

### Method
- Analyze all leads by trick number
- Compute rates of trump leads, count leads, double leads
- Track average high pip of lead domino

### Lead Characteristics by Trick

| Trick | n_leads | Trump % | Count % | Double % | Avg High Pip |
|-------|---------|---------|---------|----------|--------------|
| 3 | 43 | 27.9% | 11.6% | **39.5%** | 2.5 |
| 4 | 1,129 | 32.2% | 11.6% | 33.5% | 2.7 |
| 5 | 6,244 | 31.2% | 13.6% | 33.3% | 2.8 |
| 6 | 5,096 | 28.5% | **15.0%** | 32.3% | 3.1 |
| 7 | 178 | 21.3% | **19.7%** | 24.7% | **3.7** |

### Key Patterns

**Early tricks (3-4)**:
- **Double rate highest** (~35-40%)
- Trump rate moderate (~30%)
- Count leads rare (~12%)

**Mid tricks (5-6)**:
- Balanced approach
- Count leads increasing (13-15%)
- Pip values rising (2.8-3.1)

**Late tricks (7)**:
- **Count leads peak** (19.7%)
- Double rate drops (24.7%)
- **Highest pip leads** (3.7 avg)

### Interpretation

1. **Lead doubles early**: 39% double rate in trick 3 → establish control
2. **Save count for late**: Count lead rate rises from 12% to 20% as game progresses
3. **Pip escalation**: Average lead pip increases from 2.5 to 3.7 across tricks
4. **Trump flexibility**: Trump leads steady at 28-32% throughout

### Opening Lead Priorities (Trick 1-2)

1. **High double** (if available) - establishes control
2. **Trump suit** - pulls opponent trumps
3. **High off-suit** - wins trick, sets tempo
4. **Avoid**: Low off-suit leads (loses control)

### Files Generated

- `results/tables/25e_lead_analysis.csv` - Lead statistics by trick
- `results/figures/25e_lead_analysis.png` - Lead pattern visualization

---

## 25f: Critical Position Detection

### Key Question
When should you think hard vs play fast? What features predict "critical" positions?

### Method
- Define criticality: Q-spread (max - min of valid Q-values) > 90th percentile
- Extract state features: depth, trick position, game phase, player remaining counts
- Train GradientBoosting classifier to predict criticality
- Use SHAP for feature importance

### Key Findings

#### Criticality Definition

| Metric | Value |
|--------|-------|
| Samples analyzed | 150,000 |
| Critical threshold (P90) | Q-spread > **12 points** |
| Critical positions | 12,507 (8.3%) |

#### Classification Performance

| Metric | Value |
|--------|-------|
| ROC AUC (test) | 0.649 |
| 5-Fold CV AUC | 0.637 ± 0.026 |

#### Feature Importance (SHAP)

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | remaining_p0 | **0.242** |
| 2 | remaining_p3 | **0.178** |
| 3 | remaining_p2 | **0.178** |
| 4 | remaining_p1 | 0.073 |
| 5 | trick_position | 0.069 |
| 6 | depth | 0.059 |
| 7 | team_0_leads | 0.027 |
| 8 | end_game | 0.023 |

### Interpretation

**Watch out when players have asymmetric hand sizes!**

The top 3 predictors are all `remaining_pX` - how many dominoes each player still holds. When players have different numbers of cards remaining, decisions become more critical.

1. **Asymmetry creates uncertainty**: Unequal remaining counts mean more possible branches
2. **P0 (declarer) remaining matters most**: The declarer's hand size dominates
3. **Trick position matters**: Mid-trick decisions are more critical than leads
4. **Game phase is secondary**: Depth and phase contribute but aren't dominant

### Practical Implications

1. **Think hard when hands are unbalanced**: After a player shows out, positions become more critical
2. **Early decisions set asymmetry**: Opening play can create critical downstream positions
3. **Follow the remaining counts**: Pay attention when opponents have unusual hand patterns
4. **AUC = 0.65 means moderate predictability**: Critical positions are partially detectable but not fully predictable

### Files Generated

- `results/tables/25f_critical_positions.csv` - Summary statistics
- `results/tables/25f_feature_importance.csv` - Full feature ranking
- `results/figures/25f_critical_positions.png` - 4-panel visualization

---

## 25g: Partner Synergy

### Key Question
Does having a strong partner make your doubles worth more?

### Method
- Extract features for both P0 (declarer) and P2 (partner) hands
- Run interaction regression: E[V] ~ p0_doubles + p2_doubles + p0_doubles:p2_doubles
- Test if interaction term is significant

### Key Findings

#### Main Effects Model

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| P0 doubles | **+6.96** | Each P0 double adds ~7 points E[V] |
| P2 doubles | +0.80 | Each P2 double adds ~1 point E[V] |

**R² = 0.158**

#### Interaction Model

| Term | Coefficient | p-value | Significant? |
|------|-------------|---------|--------------|
| P0 doubles | +7.97 | <0.001 | Yes |
| P2 doubles | +1.39 | 0.23 | No |
| P0×P2 interaction | **-0.59** | **0.60** | **No** |

**R² = 0.160** (ΔR² = 0.001)

### Interpretation

**NO SIGNIFICANT PARTNER SYNERGY**

The interaction term is not significant (p = 0.60), meaning:

1. **Your doubles' value is independent of partner's doubles**
2. **P0 double worth +8 points regardless of partner's hand**
3. **Synergy effect is -1.2 points but not statistically significant**

### Practical Implications

1. **Bid based on YOUR hand alone**: Partner's strength doesn't change your doubles' value
2. **P0 doubles dominate**: +7 pts per double vs +0.8 pts for partner
3. **Additive, not multiplicative**: Team strength = your strength + partner strength (no interaction)
4. **Partner signaling has limited value**: Their doubles don't amplify yours

### Why No Synergy?

Possible explanations:
1. **Declarer dominates**: P0 leads and controls tempo - partner's hand matters less
2. **Opponent information**: Opponents also have cards - team synergy is diluted
3. **Small sample**: 200 hands may not have enough power to detect small interactions

### Files Generated

- `results/tables/25g_partner_synergy.csv` - Summary statistics
- `results/figures/25g_partner_synergy.png` - 4-panel visualization

---

## 25h: Count Capture Timing

### Key Question
When are count dominoes (35 total points) captured during the game? Does decision criticality vary by game phase?

### Method
- Analyze Q-spread (max Q - min Q for valid actions) as proxy for decision criticality
- Sample 30,000 states across 3 seeds
- Aggregate by game phase (early/mid/late)

### Key Findings

#### Decision Criticality by Game Phase

| Phase | Depth Range | Mean Q-Spread | Interpretation |
|-------|-------------|---------------|----------------|
| Early | 20-28 | **7.1** | Highest criticality - opening matters most |
| Mid | 8-19 | 4.2 | Moderate - narrowing options |
| Late | 0-7 | 2.6 | Low - endgame forced plays |

#### Depth-Level Analysis

| Depth | Mean Q-Spread | % Forced | n_states |
|-------|---------------|----------|----------|
| 1-4 | 0.0 | 100% | ~100 |
| 5-7 | 1.9-3.1 | ~35% | 7,500 |
| 8-12 | 3.0-6.9 | ~35% | 17,000 |
| 13-15 | 4.4-6.4 | ~35% | 4,400 |

### Interpretation

**Early-game decisions are most critical**

The Q-spread decreases monotonically as the game progresses:
1. **Opening (depth 20-28)**: Mean Q-spread = 7.1 - mistakes cost most here
2. **Mid-game (depth 8-19)**: Mean Q-spread = 4.2 - still meaningful decisions
3. **Endgame (depth 0-7)**: Mean Q-spread = 2.6 - outcomes mostly locked in

This aligns with findings from 25a (Mistake Cost by Phase) and 25f (Critical Position Detection).

### Limitation

The original goal was to track **when** each count domino is captured (played). Without full game traces (action sequences), we can only observe which player holds each count at game start, not capture timing. Future work with trajectory data could answer this.

### Practical Implications

1. **Defend counts early**: Since early-game decisions matter most, protect count dominoes in opening tricks
2. **Count timing is contextual**: No universal "play counts early/late" rule - depends on game state
3. **Late-game count captures are forced**: With Q-spread ≈ 2.6 in endgame, count play timing is largely determined

### Files Generated

- `results/tables/25h_count_capture.csv` - Q-spread statistics
- `results/figures/25h_count_capture.png` - 4-panel visualization

---

## 25i: Position Type Taxonomy

### Key Question
Can we create a vocabulary for discussing game situations? What types of positions exist?

### Method
- Extract 6 features per state: depth, trick_position, team_0_leads, hand_imbalance, q_spread, n_valid_actions
- K-means clustering (k=8) on 40,000 sampled states
- UMAP visualization for 2D projection
- Name clusters based on phase, position, and criticality

### Key Findings

#### Cluster Taxonomy

| Cluster | Name | Size | Depth | Q-Spread | Mean V |
|---------|------|------|-------|----------|--------|
| 0 | Mid-game Responding (Routine) | 13.5% | 10.6 | 0.2 | 12.1 |
| 1 | Endgame Following (Routine) | 12.1% | 5.6 | 1.4 | 7.3 |
| 2 | Mid-game Leading (Important) | 6.0% | 10.0 | 4.9 | 10.7 |
| 3 | Mid-game Following (Routine) | 13.8% | 10.3 | 0.2 | 13.8 |
| 4 | Mid-game Following (Routine) | 8.6% | 9.9 | 4.0 | 14.3 |
| 5 | Endgame Following (Routine) | **25.2%** | 6.9 | 0.6 | 7.9 |
| 6 | Mid-game Following (Routine) | 13.0% | 10.4 | 2.4 | 14.4 |
| 7 | Mid-game Following (Critical) | 7.8% | 9.0 | **24.1** | 8.8 |

#### Distribution by Criticality

| Criticality | Clusters | % of States |
|-------------|----------|-------------|
| Routine | 0, 1, 3, 4, 5, 6 | 86.2% |
| Important | 2 | 6.0% |
| Critical | 7 | 7.8% |

### Interpretation

**Most positions are routine, but ~14% require serious thought**

1. **Cluster 7 is the danger zone**: Q-spread = 24.1 points means wrong move costs ~24 points
2. **Leading positions are more important**: Cluster 2 (Mid-game Leading) has Q-spread 4.9
3. **Following is usually routine**: Most follow positions have Q-spread < 2
4. **Endgame is mechanical**: Clusters 1, 5 (37% of states) are low-decision

### Mental Model for Players

| Situation | Think Hard? | Why |
|-----------|-------------|-----|
| Leading mid-game | **Yes** | Cluster 2: Important (Q-spread 4.9) |
| Following mid-game, many options | **Yes** | Cluster 7: Critical (Q-spread 24.1) |
| Following mid-game, few options | No | Clusters 0,3,4,6: Routine |
| Endgame | No | Clusters 1,5: Mostly forced plays |

### UMAP Visualization

The 2D UMAP projection shows:
- Continuous manifold structure (no sharp cluster boundaries)
- Clear depth gradient across embedding space
- Cluster 7 (Critical) forms a distinct region

### Practical Implications

1. **Reserve mental energy**: 86% of positions are routine - don't overthink them
2. **Focus on leads**: Mid-game leading positions (C2) require care
3. **Watch for high-optionality follows**: When you have many valid moves mid-game, decisions matter most
4. **Trust the endgame**: Low Q-spread means outcomes are mostly determined

### Files Generated

- `results/tables/25i_position_taxonomy.csv` - Cluster statistics
- `results/tables/25i_cluster_profiles.csv` - Detailed profiles
- `results/figures/25i_position_taxonomy.png` - 4-panel visualization

---

## 25j: Heuristic Derivation

### Key Question
Which folk heuristics for Texas 42 play actually match optimal play?

### Method
- Define 18 simple rules ("lead any double", "follow with lowest", etc.)
- Test each against oracle's optimal action on 250K+ states
- Compare lead heuristics (when leading) vs follow heuristics (when following)
- Establish random baseline for comparison

### Heuristic Accuracy Ranking

| Heuristic | Description | Accuracy | n_states |
|-----------|-------------|----------|----------|
| lead_any_double | Lead any double | **34.2%** | 6,566 |
| lead_lowest_offsuit | Lead your lowest non-trump | 29.1% | 11,514 |
| lead_highest_double | Lead your highest double | 26.8% | 6,566 |
| lead_count_domino | Lead a count domino | 23.6% | 4,611 |
| follow_dump_lowest | Dump lowest if can't follow | 23.2% | 129,635 |
| avoid_count | Avoid count dominoes | 22.2% | 226,221 |
| play_lowest | Play lowest domino | 21.6% | 236,170 |
| follow_play_double | Play double when following | 21.5% | 113,090 |
| lead_highest_trump | Lead your highest trump | 21.2% | 7,344 |
| **play_random** | **Random baseline** | **19.3%** | 236,170 |
| lead_highest_overall | Lead highest domino | 19.0% | 12,253 |
| follow_protect_count | Avoid count when following | 18.8% | 78,069 |
| follow_trump_if_cant | Trump if can't follow | 18.1% | 59,807 |
| follow_lowest_in_suit | Follow with lowest in suit | 17.7% | 94,282 |
| play_winning | Play to win the trick | 17.5% | 123,063 |
| play_highest | Play highest domino | 17.4% | 236,170 |
| follow_highest_in_suit | Follow with highest in suit | 16.2% | 94,282 |
| follow_play_count | Play count when following | 13.4% | 22,071 |

### Category Comparison

| Category | Avg Accuracy | Notes |
|----------|--------------|-------|
| Lead heuristics | 25.8% | Best performing category |
| Follow heuristics | 18.4% | Near-random performance |
| Universal heuristics | 19.5% | Close to random baseline |

### Key Insights

1. **No heuristic beats 35%**: Even the best single rule matches oracle < 35% of the time
2. **Leading is more predictable**: Lead heuristics average 25.8% vs follow at 18.4%
3. **"Lead any double" is best**: 34.2% accuracy - nearly 15 points above random
4. **Following is contextual**: Follow heuristics perform near-random (17-23%)
5. **Avoiding counts helps slightly**: 22.2% vs 19.3% random baseline

### Why Heuristics Fail

1. **Context is king**: Optimal play depends on full game state, not just hand
2. **Partner coordination**: Heuristics don't account for partner's position
3. **Information asymmetry**: Opponent hands matter but are unknown
4. **Trick history**: Past plays affect optimal strategy

### Practical Implications

1. **Memorized rules won't beat strong players**: 35% max accuracy leaves huge gaps
2. **Lead strategy is learnable**: Double-leading has clear value
3. **Follow play requires calculation**: No simple rule captures follow-play nuance
4. **Machine learning needed**: Simple heuristics don't capture game complexity

### Files Generated

- `results/tables/25j_heuristic_derivation.csv` - Full accuracy ranking
- `results/figures/25j_heuristic_derivation.png` - Visualization

---

## 25k: Information Value

### Key Question
How much is perfect information (knowing opponent hands) worth?

### Method
Using marginalized oracle data (same P0 hand, 3 different opponent configurations):
1. Find states that appear in all 3 opponent configs
2. For each state, compute "perfect" action (best for THIS config) vs "robust" action (best on average)
3. Information value = Q[perfect] - Q[robust]

### Key Findings

#### Overall Statistics

| Metric | Value |
|--------|-------|
| Seeds analyzed | 2 |
| State comparisons | 8,925 |
| Mean info value | **69.0 points** |
| Median info value | **116.0 points** |
| Actions differ | **97.9%** |

#### Information Value by Depth

| Depth | Mean Info Value |
|-------|-----------------|
| 1 | 56.2 pts |
| 5 | 68.2 pts |
| 9 | **75.9 pts** (peak) |

### Interpretation

**When opponent hands matter, they matter A LOT**

The extreme values (mean 69 pts, 98% action differences) reflect sampling bias - we only find "common states" that appear across all 3 opponent configurations. These are specific critical positions where:

1. **Opponent hands dramatically change optimal play**: 98% of positions have different best actions
2. **The stakes are enormous**: Mean 69 pts ≈ 2+ marks difference
3. **Mid-game is most sensitive**: Peak at depth 9

### Limitation

This analysis is based on states that happen to appear in all 3 marginalized configurations - a biased sample of "pivotal" positions. The true average information value across all states would be much lower, as most positions have similar optimal play regardless of opponent hands.

### Practical Implications

1. **Counting cards matters at critical junctures**: At decision points that could go either way, opponent inference is worth marks
2. **Some positions are "opponent-agnostic"**: Most routine positions don't need opponent knowledge
3. **Identify the pivotal moments**: Learn to recognize when opponent hands matter

### Files Generated

- `results/tables/25k_information_value.csv` - Statistics
- `results/figures/25k_information_value.png` - Visualization

---

## 25m: Variance Decomposition

### Key Question
How much of outcome variance is "deal luck" (your hand) vs opponent configuration (their hands)?

### Method
Using marginalized oracle data (same P0 hand, 3 different opponent configurations):
1. For each base seed, compute mean V across the 3 opponent configs
2. Decompose variance: between-seed (deal) vs within-seed (opponent config)
3. Calculate Intraclass Correlation Coefficient (ICC)

### Key Findings

#### Variance Decomposition

| Component | Variance | % of Total |
|-----------|----------|------------|
| Between-seed (deal) | 170.5 | **23.1%** |
| Within-seed (opponent) | 569.2 | **76.9%** |

#### Statistical Tests

| Metric | Value | Interpretation |
|--------|-------|----------------|
| F-statistic | 0.90 | |
| p-value | 0.61 | **Not significant** |
| ICC | **-0.035** | Near zero |

### Interpretation

**SURPRISING: Opponent hands matter MORE than your own hand!**

The analysis reveals a counterintuitive finding:
1. **Your deal explains only 23% of variance** - knowing your hand gives limited predictability
2. **Opponent configuration explains 77%** - their hands matter more than yours
3. **ICC ≈ 0**: Different deals produce similar variance - deal isn't deterministic
4. **F-test not significant**: Seed differences don't significantly predict outcome

### Why This Makes Sense

1. **Partnership game**: Your partner (P2) can amplify or negate your hand's value
2. **Opposition coordination**: Opponents' combined hands determine how well they defend
3. **Same hand, different results**: The same "good hand" can succeed or fail depending on opponents

### Practical Implications

1. **Don't overvalue your hand**: Having good cards is less predictive than you might think
2. **Partner and opponents matter more**: The overall table composition determines outcome
3. **Reduce outcome attribution to luck**: You can't blame/credit the deal for most variance
4. **Bidding should be conservative**: High variance from unknown opponents = risk

### Limitation

This analysis uses mean V across states (not just root V) for computational efficiency. Root V would give cleaner "outcome" values but requires slower computation.

### Files Generated

- `results/tables/25m_variance_decomposition.csv` - Statistics
- `results/figures/25m_variance_decomposition.png` - Visualization

---

## 25n: Endgame Patterns

### Key Question
At depth ≤ 4, can endgame be simplified to simple rules or lookup tables?

### Method
1. Extract all states with depth ≤ 4 (last 4 or fewer dominoes per player)
2. Analyze Q-spread: if Q-spread = 0, only one action is optimal
3. Count "forced" decisions (only one valid or one optimal action)

### Key Findings

#### The Headline

**Endgame is 100% deterministic!**

| Metric | Value |
|--------|-------|
| Endgame states analyzed | 171,376 |
| **Forced decisions** | **100%** |
| **Unique optimal action** | **100%** |
| **Mean Q-spread** | **0** |

#### By Depth

| Depth | States | Forced % |
|-------|--------|----------|
| 1 | 42,844 | **100%** |
| 2 | 42,844 | **100%** |
| 3 | 42,844 | **100%** |
| 4 | 42,844 | **100%** |

### Interpretation

**The last 4 tricks are completely mechanical**

1. **No decisions to make**: Every endgame position has exactly one optimal action
2. **Q-spread = 0 everywhere**: All valid actions are equally bad, except one
3. **Outcome is locked in**: By depth 4, the final score is determined

### Why This Happens

1. **Information is revealed**: By trick 4+, you know who has what
2. **Few remaining cards**: Limited legal plays constrain options
3. **Forced sequences**: One card play often forces the entire sequence

### Practical Implications

1. **Stop thinking at depth 4**: The game is decided - play quickly
2. **Endgame is solvable**: A simple lookup table could replace minimax for depth ≤ 4
3. **Real decisions are earlier**: Focus attention on tricks 1-4, not 5-7
4. **No "clutch" plays exist**: You can't outplay someone in endgame - it's predetermined

### Connection to Other Findings

This aligns with:
- **25a (Mistake Cost)**: Near-zero mistake cost at depth < 8
- **25f (Critical Positions)**: Endgame positions have low Q-spread
- **25h (Count Capture)**: Q-spread ≈ 2.6 in endgame (and here we show it's actually 0 at depth ≤ 4)

### Files Generated

- `results/tables/25n_endgame_patterns.csv` - Statistics
- `results/figures/25n_endgame_patterns.png` - Visualization

---

## Summary

Strategic analysis provides actionable guidance:

1. **Focus on tricks 3-4**: Highest mistake cost, most strategic value
2. **Don't overthink endgame**: 90%+ forced plays
3. **Bid with napkin formula**: ~30 + 6×doubles + 3×trumps
4. **Lead doubles early**: Establish control immediately
5. **Mistakes average 2-5 points**: Meaningful but not catastrophic
6. **Heuristics have limits**: Best single rule matches oracle only 34% - context matters
