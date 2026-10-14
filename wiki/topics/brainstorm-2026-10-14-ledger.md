---
title: "Brainstorm notes of 2026-10-14, checked against the record: a claim-by-claim ledger"
kind: topic
first_seen: 2026-10-14
last_updated: 2026-10-14
status: complete
---

## What this is

On 2026-10-14 Jason pasted brainstorm notes from another chat ("Texas 42 AI — Brainstorm
Notes (no strategy fusion, so say we all)") and said: "suggests exploit and alphazero with
splats. idk." This page checks each claim in the notes against this repo, texas-42 (read
from its working tree and its GitHub PRs) and the cross-game lab in
`~/code/plunge-walt-bridge/lab/`. It also assesses the two suggestions he flagged and
recommends one next experiment. One pilot was run for it, the criticality histogram
below; it reads two existing pours, took 1 s and 10 s, and runs no player.

Verdicts used: **done** (with the citation and result), **partly**, **new**, **fenced**
(blocked or limited by [[strategy-fusion-fence]] or another house rule; the reason is given),
**conflicts** (the record measured something different), **unclear**.

## Headline

- The notes' **"Results so far" are this project's results, but two of the four are restated
  wrongly.** "BR(BR(uniform)) beats PIMC" is the +0.74 marks/game of [[scenario-player]].
  "BR(BR(net)) beats BR²u by +0.111 points/game, interval excludes zero, 250k games" matches
  the 250,000-deal field-net row: **+0.11 [−0.19, +0.41] marks/game**. That is marks, not
  points, its interval includes zero, and it re-graded at +0.04 [−0.27, +0.33] on Zeb 0.2.0.
  The champion is still `k1,n512,m8`. The cross-game summary ("great for euchre, well for
  spades and hearts, meh for bridge") is rosier than the lab's own reports, which tie PIMC
  at best and lose to it in bridge.
- **The demon's role in the notes ("grade, label, or act as critic") is wider than the
  fence.** Grading is a sanctioned measuring stick. Labels and critics from a per-world
  clairvoyant value are how the oracle era fused ([[the-wall]]), and the notes say so of
  their own AlphaZero run. Both are fenced.
- **Most of the notes' program has been run.** Expert iteration, splats and lumps, belief
  from plays, conventions, the level ladder and the coordination gap are all measured, and
  most are nulls or small. Three things are new: an **exploiter** (#77, open since
  2026-07-17), the **80/20 "farmboy" field as a play field**, and DSL search scored by
  duplicate games (AlphaEvolve).

## Ledger: "Results so far"

| claim | verdict | record |
|---|---|---|
| BR(BR(uniform)) beats PIMC in duplicate, "not a blowout, not close" | **done** | `walt:k1,n160,m8` (two rungs over the dice = BR²u) vs E[Q] n=10 (`lens:ev`, PIMC over the exact oracle): **+0.74 [+0.42, +1.05] marks/game**, 384 mirrored games ([[scenario-player]] § Walt beats E[Q]). texas-42's bridge at 50/8: +0.38/game per seed (Jason's 2026-08-17 receipt, cited there). |
| ~160 root positions out of ~399M | **done** | 160 root worlds; the opening fiber is 21!/(7!)³ = 399,072,960 hidden completions ([[walt-spec]]). The champion now uses 512 (+0.30 for 160 → 512). |
| model-free, runs on a phone | **done** (wording) | learning-free, not model-free: it plans with the rules. Zeb in WASM: 44 ms per opening lead at 160 worlds (`@jasonyandell/zeb`). |
| optimizes pmake, resamples worlds on every own move | **done** | `u=make`. Live play resamples each turn; that is what keeps the plan honest ([[scenario-player]] § The forecast lies). |
| BR(BR(net)) beats BR²u by +0.111 points/game, interval excludes zero, net trained on 250k games | **conflicts** | Zeb = `k0,n512,F=margin_d250k_h128x64` vs champion `k1,n512,m8`: **+0.11 [−0.19, +0.41] marks/game** (+0.15 points/hand), and +0.13 [−0.17, +0.44] for the 512×256 net. Re-graded on 0.2.0: +0.04 [−0.27, +0.33]. Verdict on the hub: "parity at a quarter of the cost", "a cheaper equal, not a stronger player". Also Zeb is one BR over the net, `k0`, not two. The net imitates the previous Zeb, so "BR(Π BR(net_prev))" with Π the projection onto the net class is the precise name ([[research-night-2026-10-12]]: "one policy-iteration step of its own net, evaluated by a 512-world rollout"). The nearest interval that excludes zero is texas-42 PR #102's net inside a 160-world one-ring search: +.100 [+.086, +.115], a net paired-deal share (wins minus losses over deals), against phone Walt at 40/8, with nets trained on 2.9–7.6M rows. Not the same claim. |
| cross-game: great for euchre, well for spades and hearts, meh for bridge | **conflicts** | `~/code/plunge-walt-bridge/lab/*/REPORT.md` (exploratory tier, small samples). Euchre: Walt L1 vs PIMC +0.000 ± 0.104 per hand at n=24, +0.093 ± 0.104 at n=96. Spades: vs PIMC +2.4 [−18.8, +23.5], and significantly weaker than flat MC against the rule bot (−38.0 [−57.4, −18.6]). Hearts: −0.63 [−1.27, +0.01] against MC with rule-bot rollouts. Bridge: duplicate Walt − PIMC **−0.133 [−0.261, −0.006]**. Skat, the notes' proposed "middle case", is already done: −0.040 [−0.128, +0.048] vs PIMC. Hold'em: "doesn't work". So it ties PIMC at best outside 42. `silas/docs/cross-game.md` is about match scoring (marks, points to 250), not other card games. |
| earlier AlphaZero on oracle targets never matched PIMC; fused by its targets | **done** | [[zeb]] (January): never beat E[Q] n=10 (Elo 1579 vs the 1600 anchor; ~74–76% vs random). [[full-teacher-eq-experiment]]: E[Q] teacher at 25% and 95% moved nothing. Fusion came in twice, through the determinized-UCT search (per-world playouts) and through the E[Q] targets ([[alphazero-under-imperfect-information]]). |
| the search has to come from outside: the net is the prior | **done** | Imitation stops at parity: 5k → 50k → 250k deals, regret 0.65 → 0.26, arena flat ([[scenario-player]] § The 250,000-deal pour). Research night names the mechanism: a self-confirming fit of a misspecified model class (Berk–Nash). texas-42 PR #102: net alone ≈ +.05, the same net inside search ≈ +.10, and the search-over-net gap stays near .09 as nets improve. |

## Ledger: "Why BR²u works"

| hypothesis | verdict | record |
|---|---|---|
| 1. PIMC and BR(uniform) are cousins; uniform discards plays' information | **partly** | Uniform over the void-consistent fiber is hard to beat in 42: auction −0.05, dice −0.06, field tilts −0.65/−0.96 (night 1); posterior +0.026 marks/hand only against a table of nets, null in self-play (#111); reading plays null in marks ([[reads-plays-2026-10-13]]). "BR² is trained against the PIMC worldview" is untested. |
| 2. Level-k with low useful depth; deeper hurts | **conflicts** (in 42 one more level helped) | mk5: rung 0 +0.18 vs E[Q], two rungs +0.74. texas-42 `experiments/walt42x-intake/H2H.md`: L2 beat phone Walt (L1) at 160/24, 53.7% [51.6, 55.9] of games on 512 pairs. The lab card has L2 vs L1 tied 14/14/72 on 100 deals; euchre L2 vs L1 −0.083 ± 0.142. Under a fixed belief the ladder settles to a unique fixed point by rung 24, Lean-proved; in practice by rung 2–3 ([[research-night-2026-10-12]], [[constellations-2026-10-14]]). k2 has never met k1 in the mk5 arena. |
| 3. Uniform is the hero: full support, no belief collapse | **done** (supported) | A deterministic net excluded the real deal at 586 of 674 positions; a stochastic field (#108) was needed. Reweighting collapses ESS to ~10 worlds mid-hand. |
| 4. Structure in the dynamics, not features; features rare in the sample | **partly** | The book's 18 rules: Zeb's pick is the book's at 56% vs a 43% floor ([[book-study-2026-10-12]]); as a player they lose about 1.1 marks/game after encoding repairs ([[book-investigation-2026-10-14]]). The trump-set rules never fire in 3,000 random games and do fire under bidding populations (#141): rare in the sample, not mis-encoded. |

## Ledger: the splat, cheap experiments, RL designs

| idea | verdict | record |
|---|---|---|
| splat = sampled distribution of pmake per move | **done** | `Walt.analyze`, the pour, [[splat-visualizer]]; calibrated forecast at scale. |
| lumpy, usually more than 2 modes, lumps driven by hidden facts (partner holds the 5-5) | **partly / conflicts on "usually"** | [[lumps-voi-2026-10-13]]: 96.5% of (position, tile) splits are robust beyond the bar, so the lumps mostly agree. The 5-5 is the tile most worth knowing (0.39 pp beyond the null), and the money sits disproportionately in the partner's lump. Modality was never counted. |
| train on lumps; spend samples where lumps disagree | **new** (proposed, unbuilt) | Racing among near-ties is named in [[scenario-player]] § What must be true. [[splat-search-bounds]] gives the sound pruning rules. |
| cheap 1: criticality histogram | **done** (this page, below) | Partly before: 714 decisions on research night (critic: 40% exact ties, 17% > 5 pp). Oracle-era [[gus-drama-atlas]] is the fenced cousin: oracle disagreement per world. |
| cheap 2: rule-mine only the disagreements | **partly** | Against the book: [[metasplat-2026-10-13]], the reason report (#124), [[book-head-to-head-2026-10-13]]. Against PIMC: not done. E[Q] is a sanctioned baseline to contrast with, never a label. |
| cheap 3: DSL as deal generator | **partly** | #141 (bidding populations, constructed fibers); [[silas]] enumerates worlds under a condition exactly. |
| cheap 4: exploiter against the champion | **done** (2026-10-14, [[exploiter-endgames-2026-10-14]]): on three-trick endgames, G = +0.50 [+0.42, +0.58] margin, neither sturdy nor shown exploitable; +0.004 contract probability per endgame. Before that, **new** | #77 probe 2 ("counter-walt"), open since 2026-07-17. Research-night ledger row 12: "train a best-response opposing pair against frozen Zeb; a lower bound on the O3 gap". [[endgames-o2-vs-o1-2026-10-12]]: "O3 not attempted". The only exploiter-shaped results are readers of plays, which are null (#111 in self-play, [[reads-plays-2026-10-13]]). |
| A. expert iteration keeping the farmboy (net mixed 80/20 with uniform) | **partly** | Expert iteration ran three turns (5k → 50k → 250k pours, each poured by the last net) and stopped at parity. The 80/20 mix exists as `field_eps` = 0.2 (arena `E=0.2`, #108) but has been graded only as a belief, never as Zeb's play field. Untested and cheap. |
| B. AlphaEvolve on the DSL, scored by duplicate games vs the champion | **new** | The scoring harness exists: [[book-head-to-head-2026-10-13]] seats [[silas]] policies in paired games. Every hand-written rule set tried so far lost (−1.1 to −1.9 marks/game) except as Zeb's near-tie break (53.2% of games to 250, [[book-investigation-2026-10-14]]). |
| C. fusion-free AlphaZero (policy, distributional value, belief heads) | **partly; value head blocked by a house rule** | See the assessment below. |
| demon labels and critic (Suphx-style oracle guiding) | **fenced** | Labels and critics carry per-world clairvoyant values into the seat's choice. The fence sanctions only bounds, baselines and measuring fusion. Oracle guiding with features faded out still trains on clairvoyant gradients, which is how jud, gus and January's Zeb hit [[the-wall]]. texas-42 PR #102 (open, unmerged), checked in its code on 2026-10-14: its labels come from an information-state search (one tile per public branch, summed over worlds; `ladder/csrc/turbo.cpp`), but with no cell floor, so thinned groups make the continuation clairvoyant on the sample, and it is a best response to the net. It is Zeb's class minus the floor, not per-world double-dummy. Its +.100 is an honest paired play result against texas-42's production Walt. Nothing in mk5-main uses its labels or weights. The deployed field net trains on Zeb's own plan values from pours with the floor at 16, and only the `pmake` target's name was borrowed. |

## Ledger: theory, fairy asks, open questions

| item | verdict | record |
|---|---|---|
| partnership breaks vanilla ReBeL; team optima may need correlation | **done** (the gap's existence) | [[endgames-o2-vs-o1-2026-10-12]]: O2 beats the Walt map by +0.027 [+0.022, +0.033] margin per three-trick endgame, and ~70% of what remains is a convention. TMECor itself is new. |
| ReBeL², SPARTA-like single-seat search against a blueprint | **done** in effect | Zeb is single-seat search against a learned field. |
| pure BR chains cycle; averaging gives fictitious play | **partly / conflicts** | Under a fixed belief, Zeb's case, the ladder provably settles (T2, Lean). With beliefs recomputed from the rung below it can cycle (Lean witness; mini-42 periods 4/6/12; Walt-map cycles on endgames were period 2, off path, value constant). |
| fairy 1: adversarial gap | **new** | The exploiter; see the recommendation. |
| fairy 2: coordination gap | **done** (endgames) | +0.027 margin per endgame (O2 − O1 Walt), +0.115 against plain BR dynamics. |
| fairy 3: leakage profile | **partly** | Posterior ESS by play (#108); lumps VOI peaks at tricks 4–5; "hand-bleed audit" (#77 probe 1) unbuilt. |
| fairy 4: knowledge depth | **partly** | See hypothesis 2. |
| fairy 5: value of correlation | **new** | O2+ − O2 = +0.46 is a shared-hands relaxation, not correlation. |
| fairy 6: prescription compression | **partly** | Constellations (exact late, no reach to trick 1); hidden worlds never compress ([[endgame-equivalence-census]]). |
| fairy 7: convention map | **partly** | The book's conventions are null in the arena ([[book-study-2026-10-12]]); the first convention was found on endgames. |
| fairy 8: team-optimal endgame labels, exact belief oracle | **partly** | O2 labels for 533 endgames; exact beliefs: enumerated fibers, Silas. |
| where do BR²'s points come from? | **partly** | Arenas fix the bidders (`bid30`), so every measured mark is from play; the signature is "marks up, points down". Criticality by trick: below. |
| why does the net help (+0.111)? | **conflicts** (premise) | In mk5 it did not measurably help; it made Zeb about 4× cheaper at parity. |
| self-similar, each trick a smaller 42 | **partly** | Constellations (a late-hand quotient, not a recursion). |
| the true ceiling; "Truck-kun" | **unclear** | Not found in either repo. |

## The two suggestions Jason flagged

### (a) An exploiter against Zeb: is the ledge sturdy?

- **Prior.** The readers that have been tried exploit nothing. Zeb with the net's posterior gains
  +0.026 marks/hand against seats that *are* the net, but +0.003 ± 0.004 against Zeb
  (#111/E1c). Reading plays is a null ([[reads-plays-2026-10-13]]). Zeb is a deterministic
  best response, maximally readable by construction (#77). At endgames a single-seat best
  response to the net gains +1.64 margin per endgame over the net's own play, so the field
  model itself is very exploitable ([[endgames-o2-vs-o1-2026-10-12]]). Whether Zeb's own
  play is exploitable is unmeasured.
- **Cheapest fence-clean version:** three-trick endgames, exact. The exploiter is an
  information-state solve (one action per information set), so it needs no fence. A
  full-game exploiter that models Zeb exactly costs 60–270 s per decision (research-night
  row 9). That does not fit the budget, and it is a design mismatch, not a tuning problem.
- **Cost:** half a day to adapt `scratch/endgames-o2/eg.py` (main checkout), then four runs
  of about 200 s each.

### (b) Fusion-free AlphaZero with a distributional (splat) value head and a belief head

- **Already built, in pieces.** The policy distilled from search is the field net. The search
  is Zeb. The distributional value is the splat, computed by rollout and calibrated. The
  belief head is the posterior through the net (#111), and texas-42's `BEL-po-d` gained
  +.021 [+.008, +.034] by resampling at 160 worlds (PR #102, exploratory).
- **Predicted payoff: small.**
  - Imitation has stopped at parity, so more distillation or a bigger net predicts no gain.
  - Every belief in self-play has been null: posterior, readers, conventions.
  - The late-hand net already plays the settled policy's best move 96% of the time
    ([[constellations-2026-10-14]]).
  - Where the decision matters (gap > 5 pp) the net already agrees 88% of the time.
  - A value head would buy depth and speed, not a better policy at the root.
- **House rule.** "The net never enters a value or a leaf" (Jason, 2026-10-10, [[scenario-player]]
  § The net in the field) blocks a learned value at the leaf until he lifts it. Value targets
  must be Zeb's own outcomes or splats, never oracle values: that is the fence.
- **Cheapest fence-clean version:** a belief head trained on true hands from the 250k pour,
  used as the live draw by sampling, never reweighting. It needs a core change to accept an
  external belief, bit-identical referee ↔ core, so it is a build. The predicted gain is
  under +0.03 marks/hand on the #111 precedent, and it may be null in self-play.

## Pilot: the criticality histogram (run 2026-10-14)

For each planned (non-forced) position of a pour: the mover's best q minus the runner-up's,
in make probability, with a noise bar of 2 paired standard errors of that difference over
the position's own worlds. The values are the planner's own, with the continuation fixed. No
perfect-information value is used. The gap is in-sample, so the maximum over a sample biases
it upward. Reader: `scratch/brainstorm/criticality.py` (numpy over `pour.npz`).

**Pour A** is the champion `k1,n512,m8` under `make`, 5,000 deals, 86,407 planned positions,
1 s. **Pour B** is the net-field `k0,n512,F=…_v2` pour (Zeb's shape), 50,000 deals, 864,604
positions, 10 s.

| gap, best minus runner-up | decisions A / B | share of gap mass A / B | beyond its own bar A / B |
|---|---|---|---|
| exact tie | 39.3% / 39.8% | 0 | 0 |
| 0–1 pp | 17.7% / 17.5% | 2.6% / 2.5% | 1% / 1% |
| 1–5 pp | 25.9% / 24.6% | 17.6% / 16.2% | 17% / 28% |
| 5–20 pp | 13.1% / 13.9% | 34.5% / 35.6% | 89% / 89% |
| > 20 pp | 4.1% / 4.3% | 45.3% / 45.7% | 86% / 87% |

- **The notes' prediction holds, and the two players agree to a point.**
  - 57% of planned decisions are within 1 pp of a tie.
  - The top 5% of decisions carry half of all gap mass, and the top 20% carry 83%.
  - About three decisions per hand clear 5 pp: 17% of ~17 planned plays.
- **The money is spread across the hand, not concentrated at the lead.** By trick, the share
  of gap mass is 14 / 14 / 15 / 16 / 18 / 23% (pour A). Exact ties rise from 7% at trick 1
  to 84% at trick 6, but the late gaps that remain are large. This differs from the oracle
  era's "62% of drama at the lead" ([[gus-drama-atlas]]), which counted per-world oracle
  disagreement, a different quantity.
- **Bidders and setters are alike:** 53% / 47% of the mass, 17% of decisions above 5 pp
  for both.
- **What it bounds.** A policy or search improvement can only pay at the ~18% of decisions
  above 5 pp, and the net already agrees there 88% of the time. Racing among near-ties
  (the 1–5 pp band, 17–28% beyond the bar) is worth at most about a sixth of the gap mass.

## Recommendation: one next experiment

**Run 2026-10-14:** [[exploiter-endgames-2026-10-14]].
- G = +0.50 [+0.42, +0.58] margin per endgame: not sturdy (the bar was an upper bound < 0.10), and not shown exploitable (the bar was a lower bound > 0.50).
- Half of G is Zeb's lowest-id tie rule once the contract is decided.
- In contract probability, knowing Zeb is worth +0.0040 [+0.0028, +0.0054].

**An exploiter on three-trick endgames: a lower bound on Zeb's exploitability.** This is
#77 probe 2, research-night row 12 and the notes' fairy ask 1, in the only setting where it
is exact in minutes.

- **Design.** #113's endgames and code (`scratch/endgames-o2/`; the same deal seeds; the
  250k net). Freeze the opponents at a Zeb proxy: exact one-rung Walt over the net at each
  seat's own information state (Zeb at three tricks left, without the 512-world sample).
  Then compute the team's responses:
  1. **E1**, a single-seat best response by V to the Zeb-proxy table, with P also the Zeb
     proxy. This is realizable: V reads only its own hand and the record.
  2. **E2**, the O2 coordinator against the Zeb-proxy opponents. This is the upper side
     under #113's common-knowledge relaxation.
  3. **Placebo**, V's best response to the *net* opponents (#113's O1), played against the
     Zeb proxy.
  4. **Base**, V and P both Zeb proxies (Zeb self-play).
- **Pre-registered statistic.** The Zeb-specific exploit
  G = V(E1) − V(placebo), in team margin per endgame, with a deal-resampling bootstrap 95%
  interval over ≥ 140 deals (~560 endgames). Report beside it E1 − base, E2 − base, and the
  scale references +0.027 (O2 − O1 Walt) and +1.64 (best response to the net over the net).
- **Verdicts, fixed now.**
  - **Sturdy at the endgame:** G's upper bound < 0.10.
  - **Exploitable:** G's lower bound > 0.50.
  - **Small but real:** G's lower bound > 0 and its upper bound ≥ 0.10.
  - Anything else is inconclusive and says so.
  - The chance level is 0: an exploiter can always copy Zeb, so E1 ≥ base under exact
    evaluation, and the placebo is the control for "better play in general".
- **Runtime.** A pilot of one deal (four endgames) runs in about 1 minute, as #113's
  `probe.py` did. The full run is four chunks of about 200 s, each under `timeout 300`.
  The Zeb-proxy decisions cache by (seat, own hand, record).
- **Why this one.** It is the only flagged idea with no prior measurement. It has a clean
  fence status, and either answer changes what to do next. A sturdy ledge retires the
  exploit worry for Plunge. An exploitable one prices #77's countermeasures (mixing on
  ties, `field_eps`).

Runner-up, cheaper still: grade the farmboy field as a *play* field, `walt:k0,n512,F=…,E=0.2`
vs `E=0`, in #111's paired-deal harness at about 1,500 deals a minute. The prediction is null
or slightly negative: a best response to a sloppier model of a non-sloppy table.

## Links

[[the-wall]] · [[strategy-fusion-fence]] · [[scenario-player]] · [[research-night-2026-10-12]]
· [[endgames-o2-vs-o1-2026-10-12]] · [[lumps-voi-2026-10-13]] · [[metasplat-2026-10-13]]
· [[reads-plays-2026-10-13]] · [[constellations-2026-10-14]] · [[book-investigation-2026-10-14]]
· [[zeb]] · [[full-teacher-eq-experiment]] · [[splat-search-bounds]] · [[exploiter-endgames-2026-10-14]]
