---
title: Champion design review — Fable 5's recovered reasoning
kind: topic
first_seen: 2026-06-09
last_updated: 2026-07-13
status: complete
---

The [[champion]] direction was set in a single design session (2026-06-09,
Fable 5). [[champion]] records the **conclusions** — the decision loop, the
asset map, the marginal-value ranking, the build ladder. This page preserves
the **reasoning**: the verbatim reviews Fable wrote, and a graded ledger of the
predictions in them, checked against what the rungs actually measured. Fable is
no longer available; this is the mined record, not a paraphrase. Recovered from
transcript session `0a708a4e` (the design turns were compacted out of live
context; primary source is reproduced verbatim at the bottom).

It exists because distillation shears caveats: two of Fable's load-bearing
qualifications did **not** survive into [[champion]], and their absence was
quietly steering later sessions toward wrong conclusions (see "Two caveats").

## The five turns

1. **Catch-up** — fresh-eyes read of the project state on return.
2. **BSP build blueprint** — how the [[book-strategy-player]] algebra would
   thread into the batched simulator (L1 true by construction; fallback =
   `argmax_under_utility` on a 1-game slice). The chassis was later repointed
   from play to the auction.
3. **Critical review** ("chasing fireflies?") — why play-phase book strategy is
   low-yield: forge Q is already plan-aware within each sampled world, so PIMC
   misses only *information-set* value — and a both-PIMC harness can't even see
   that.
4. **Bidding inventory** — contract evaluation is done twice; nobody can bid;
   the auction is the missing game. The B1/B2/B3 staging that became rungs
   #22/#21+#20/#24.
5. **Forward design** ("the best 42 player you could") — the belief-state spine,
   the marginal-value ranking, self-consistency, the teaching half.

## Predictions, graded

Fable made concrete calls; the rungs since have measured them. The grade is the
mined value — including where the call overshot.

| Fable's call | Outcome | Grade |
|---|---|---|
| "Arena first — until it exists, 'best player' isn't a measurable sentence" | [[arena]] built (#20); first physics confirmed it | ✅ dead-on |
| "Auction v0 — gus/bidding + risk-budget, end to end, this month" | #21 GusBidder +1.09, #22 net +1.29 marks/game | ✅ hit, on timeline |
| "The corpus generator never ran — died when attention jumped to LEM" | verified; #22 resurrected it Gus-backed | ✅ hit |
| **auction ≫** belief-weighted worlds > score-utility ≫ card-play polish | only CI-excluding-zero arena wins are the auction (#21/#22) | ✅ the ≫ confirmed |
| both sides PIMC → harness is **information-blind**; concealment can't be rewarded | the #25 belief null is exactly this | ✅ prescient (caveat 1) |
| score state "bites hardest at the **auction**" | #27 tested score-conditioning in *play* → lost −1.20; auction side barely tested | ⚠️ tested the wrong phase (caveat 2) |
| belief-weighting: "one change, bidding/**play**/defense improve together — highest-leverage" | #25 measured **null in play** (3 configs) | ❌ enthusiasm overshot; his own ranking put it in the card-polish tail, which is where it landed |

**The meta-pattern.** Every null/negative the rungs produced (#25 belief-weighting, #27 play-risk) is a lever tested in the **play** phase — and Fable's whole thesis is that the value lives at the **auction**. The pattern of nulls is the marginal-value ranking confirming itself from the bottom up (`≫ card-play polish`), not the levers being worthless. The sharpest thing about the forward-design turn: it contains its own correction — the "highest-leverage" enthusiasm for belief-weighting overshot, but the *ranking* in the same turn placed it exactly where the data later put it.

## Two caveats (the load-bearing corrections)

These two qualifications were in Fable's reviews, did not make it into
[[champion]]'s verdicts, and each corrects a live over-claim:

1. **The arena is information-blind by construction.** Both sides of the paired
   match are PIMC (information-blind) players, so the harness *cannot reward
   belief or concealment value via play-marks* — independent of whether that
   value exists. So "#24's better belief is marks-neutral in play" is not
   evidence the belief is useless; the instrument is blind. #24's belief value
   must be shown through belief accuracy (done: +2.59pp) or the bidder/defense,
   never arena play-marks. This reframes the #25 decisive null from "belief→play
   is dead" to "the harness can't see it here." See [[pimc]],
   [[belief-bayes-ceiling]].
2. **Score-conditioning was tested in the wrong phase.** Fable located mark-state
   value at the **auction** ("bites hardest at the auction"), where bid
   thresholds shift with the score. #27 measured score-conditioned *play* risk
   (−1.20, a clean negative) — the play phase, not the auction. The auction-side
   (`MarksToSeven` thresholds varying with the score) is unrun and is the live
   opportunity. Note #31's "bid-magnitude → belief is dead" is a *different*
   channel; don't conflate it with score → auction-policy.

Both consequences are folded back into the rung record (#25, #27 on
[[champion-ladder]]) so the conclusions there carry their caveats.

The review is concluded: its predictions are graded above, its caveats are
folded into the pages that needed them, and the ladder it designed has been
walked ([[champion-ladder]]). The page stands as the preserved primary source.

## Links

- [[champion]] — the conclusions this preserves the reasoning for
- [[arena]] — the measuring stick built from the forward-design turn
- [[gus]] · [[forge]] · [[pimc]] · [[belief-bayes-ceiling]] — the organs and the
  belief-ceiling evidence the reviews lean on
- [[book-strategy-player]] — the algebra the BSP-blueprint turn repointed at the auction
- [[w42-lens-v1-utility-head-to-head]] — the EV-ceiling result the critical review triangulates from

---

## Primary source — verbatim (Fable 5, session 0a708a4e, 2026-06-09)

Reproduced exactly. `[[backlinks]]` are Fable's own.


### Catch-up — fresh-eyes read of the project on return

> **Jason:** "check the wiki. I've been away for a couple of weeks... please take a look around and let me know what you see."

Here's what I see. The short version: **you did finish the book.** All 64 Winning 42 claims got closed out with a final independent audit, the campaign produced a decisive empirical answer about utilities, and then the work pivoted into two genuinely new things — a planning algebra born from the book's blind spot, and a microscope session where Burl designed its own tools. Last commit was May 7; one finished piece of work is still sitting uncommitted in the tree.

## The book campaign — finished

The trail page ([[w42-book-validation]]) records the full arc. Phase 4 ended with a **completion board and an independent final audit confirming every one of the 64 book claims has evidence, explicit bounded blockers, or ledger-review routing — no row ownerless**. Then book-validation v1 ran four waves on top of that baseline and produced real promotions and demotions: *bid-only-enough* was promoted to **supported**, *void creation* turned out position-dependent, and *ch12 high-bid pounce* was **contradicted** once tested at the action-contrast level. The capstone was Lens v1 ([[w42-lens-v1-utility-head-to-head]]): a round-robin of utility functions where **EV won decisively** — beating p_make by +5.42 pts/hand, all confidence intervals excluding zero. That settled the "is the book secretly p_make-optimized?" theory.

## What grew out of it: BookStrategyPlayer

The campaign's closing insight was that many book claims are *multi-step plans*, while every probe measured isolated decisions. The answer is [[book-strategy-player]] — a fully specified algebraic contract (May 3–4): pure Reader/State/Writer decision core, finite strategy library with a `recognize → commit → execute → retire` lifecycle, monoid-valued shared facts, deterministic arbitration, and the load-bearing identity `BSP(empty, Lens("ev")) == Lens("ev")`. Ten laws, property tests specified, build order written down. **Status: designed, not built.** Bead `t42-zrf9` is the in-progress marker. This is the largest staged piece of work waiting for you.

## The Burl thread — the microscope and Burl's own tool requests

In parallel, May 2–7:

- **burl-lab** landed as a platform (31 files): event-sourced sessions, phase state machine, first-class ToolSpecs, HATEOAS tool advertisement, plus an LM Studio SDK lane.
- **burl-microscope** (the last commits, May 6–7) is a lighter Pi-driven workbench — load one harvested decision, edit a recipe, step Gemma turn-by-turn. The standout story: you asked Burl what information it was missing, and **Burl proposed its own tools**. That became `simulate_hand_impact` (hypothesis: "if seat X holds Y, how does my play's outcome shift?") and then, after using it, Burl asked for the higher-level `calculate_expected_utility` — now the main synthesis tool in the `snapshot-utility` recipe. Those two tools are your final two commits.
- An honest correction is recorded there too: the first microscope "win" on the `BURL_BREAKS_CONSENSUS` case was reference leakage (the prompt exposed the oracle play). With a fair prompt, Burl still picks 25 where the oracle prefers 19 — that's now the top open question on the page.

## Loose ends in the working tree

1. **Stranded finished work**: `14d_shap_qval_transformer.ipynb` (May 6) — SHAP via GradientExplainer on the actual 3.3M-param oracle transformer, with all figures/tables generated and a 42-line writeup already added to `forge/analysis/CLAUDE.md`. Finding: the transformer distributes attribution far more broadly than the 14a surrogate implied (doubles only 1.13× elevated, not ~3×; 6-3 is the top token, 5-5 confirms at rank 3). It's complete but **never committed** — it predates your last commits and just got left behind.
2. **`.intent/`** is noise — Augment Intent auto-scanned package.json scripts on June 6 (you must have opened the repo in it briefly). Safe to gitignore.
3. **`bd` isn't on PATH** in my shell right now, so I couldn't pull the ready-work queue — worth checking whether beads needs reinstalling, since the CLAUDE.md workflow depends on it.

## Where I'd pick up

The natural next move is committing the stranded 14d SHAP work (it's done and documented), then deciding whether the next session is **building BookStrategyPlayer Phase 1** (`t42-zrf9` — fake strategies + law tests first, per the build page) or continuing the microscope thread on the fair-prompt case. Say the word and I'll start with either.


### BSP build blueprint — after reading bead t42-zrf9

> **Jason:** "let's you and me look at implementing that [BookStrategyPlayer]. that is what I was excited about."

The SHAP work is landed and pushed — commit `b512324` on `origin/forge`, 8 files, `.intent/` left out. And I've read the bead (RIP beads — the JSONL grep worked fine), the algebra pages, and the Lens v1 code. Here's the lay of the land for the thing you were excited about.

## The bead: t42-zrf9 — "Wave 5 Phase 1: BookStrategyPlayer framework + first strategy + recording"

Priority 1, marked in-progress May 4, ~3–4h estimated. The motivation is the campaign's closing insight: Lens v1 proved **EV is the ceiling for fixed pointwise utilities at one-step lookahead** — but the book was never in that contest, because its value lives in multi-step plans. The strategy-selector path is the only live route past EV (multi-utility heads, soft-cliff utilities, MCTS, and K=2 lookahead are all explicitly dead).

The acceptance criteria are crisp and very you:

1. Framework + all algebraic law tests pass on **fake strategies** before any book tactic is encoded.
2. **L1 verified empirically, not just in unit tests**: 100 paired hands of `BSP([], Lens("ev"))` vs `Lens("ev")` must produce *100% identical actions* — exact, since both sides are deterministic given seeds.
3. Head-to-head: `BSP([SingletonLeadToVoid()], Lens("ev"))` vs `Lens("ev")`, 1000 paired-seed hands, N=10, fp32, MPS, bid=30, bootstrap CI, with an explicit three-way verdict (supports / neutral / contradicted-at-every-abstraction).
4. Diagnostics (firing rate, completion rate, bail count, fallback fraction), DecisionRecord parquet, a wiki experiment page, and the promotion guard — the framework run does *not* move the ledger row; that's a separate reconciliation decision.

The design note in the bead says the negative result is a finding too: if singleton-lead-to-void stays point-negative even with multi-step execution, the Wave 2.C contradiction holds at every abstraction level.

## One reconciliation to make before writing code

The bead (May 3) and the wiki algebra (refactored May 4) drifted slightly, and the bead's own design field settles it: *"the algebraic specification in [[book-strategy-player]] IS the design."* So we follow the amended wiki where they differ:

- **Protocol**: the bead still says `applies_to`; the amended algebra split that into `recognizes(gs, ps)` for fresh opportunities vs. `name in active_plans` for continuing plans. We build the five-method protocol: `recognizes`, `priority`, `commit_recognition`, `next_action`, `plan_done`.
- **Layout**: bead says `w42/book_strategy_player/core.py`; the build page says `w42/book_strategy/` with finer-grained modules (`player.py`, `strategy.py`, `plan_state.py`, `facts.py`, `recording.py`, `fallback.py`, `errors.py`). I'd take the wiki layout — it's the later refinement.
- **Tests**: the build page's T1–T17 property tests are the concrete expansion of the laws (permuted library orderings, fact-merge associativity/commutativity, namespace hermeticity, NaN priorities, recording purity).

## The interesting design point: how BSP plugs into the batched simulator

`parallel_match.py` advances all hands in lockstep: each step it gathers every game where it's team A's turn, runs **one batched oracle query** (world sampling → tokenize → Q-PDF per legal action), and argmaxes under the utility. Our player is per-decision and stateful (per-hand `PlanState`). The bridge that makes everything fall out cleanly:

- Keep the batched oracle query exactly as-is. After it, loop per game index and call `choose_action` with the per-game slice of `(e_q, e_q_pdf, legal_mask, bidder, current_player, bid)` packed into the environment, alongside the `ZebGameState` and that game's `PlanState`.
- The **fallback adapter is then `argmax_under_utility("ev", ...)` on a 1-game slice** — same tensors, same lowest-slot tie-break as Lens v1. That makes acceptance criterion 2 (exact 100% action identity) true *by construction*, and satisfies T16 (fallback consumes no live RNG) because the oracle sampling already happened once, shared, before any decision logic runs.
- Strategies read `gs.play_history` to detect what happened between their turns (disruption, void suit led) — no observation hook needed in Phase 1, which matches the extension-points boundary.

## Proposed build order

Straight from the build contract: **(1)** carriers + errors (`PlanState`, `DeltaPlanState`, fact monoids with naked-scalar rejection), **(2)** `Strategy` protocol + arbitration loop, **(3)** fake strategies + T1–T17 law tests green, *checkpoint here*, **(4)** Lens fallback adapter + the 100-hand empirical L1 run, **(5)** recording, **(6)** `singleton_lead_to_void`, **(7)** the 1000-hand head-to-head + analysis + wiki page.

The contract is explicit: no book tactic until the laws pass on fakes. I'm ready to start scaffolding milestone 1–3 — anything you want to adjust first, or shall I dig in?


### Critical review — is this defensible? (the firefly question)

> **Jason:** "is what we're doing sensible? defensible? I wonder if I was chasing fireflies rather than getting the most out of the book for the goal, which is a better 42 player... I learned it on grandma's knee, literally."

Great question to stop and ask, and your own wiki gives a surprisingly sharp answer. Let me be honest on both halves.

## Is BSP defensible? Yes — as science. As a strength play, your firefly instinct is right.

Three results you already produced triangulate this:

1. **[[gus-strategy-tags-probe]]**: book strategy tags improved the *tiny* 28k model dramatically (2.012 → 1.181 regret) — but `E[Q] N=10` stayed boss at 0.167, untouched. The book helped the player that can't afford the oracle, and did nothing for the player that can.
2. **The wave campaign itself**: tactical book claims kept coming up underpowered, context-limited, or contradicted when tested against oracle contrasts at the action level. The one clean promotion to *supported* was **bid-only-enough — a bidding claim**.
3. **Lens v1**: EV over the oracle PDF beats every alternative summarization decisively.

And there are two structural reasons the play-phase BSP head-to-head is unlikely to pay, beyond these empirics. First, forge's Q is a *minimax value to the end of the hand* in each sampled world — Lens(ev) is already plan-aware within every world it imagines. "Lead the singleton so you can ruff later" is a within-world plan; Q already prices it. The only value PIMC structurally misses is *information-set* value: concealment, partner inference (your Ch 11 "table talk is legal inference"), strategy fusion. Second — and this is the killer — **your measurement harness can't see that value either**, because both sides of the paired match are PIMC players, which are information-blind. A concealment strategy can't be rewarded by an opponent that doesn't exploit revelation. So BSP-vs-Lens(ev) is a contest played entirely on terrain where the oracle is already near-ceiling.

So: chasing "beat the oracle at play time with book plans" — that's the firefly. The bead even predicts it ("the negative result is itself a finding").

## Where the strength is actually hiding

Here's the thing that jumped out reading everything fresh: **nobody in your entire stack can bid.** Lens v1, the branch atlases, parallel_match — all force `bid=30`. A "better 42 player at the single-game level" literally cannot exist yet, because a game is *bid → declare → play → marks to 7*, and you've built a world-class back two-thirds. Meanwhile:

- The book's strongest, *validated* content is the bidding theory — Ch 2 risk budgets, at-risk points, bid-only-enough (supported!).
- The infrastructure already exists from Wave 2: `from_snapshot` state injection and the bid-aware E[Q] driver, with the atlas showing exactly how mark_ev diverges from threshold mass as bids rise.
- An oracle-backed bidder is cheap to build: sweep (bid, declaration) candidates, sample worlds, read the E[Q] PDF per declaration, choose under a Roberson-shaped risk budget. The book supplies the *policy structure*; the oracle supplies the *distribution*.

Second, **Lens v2 (t42-nwuu)** — your own ICM insight from May 3: utility conditioned on mark score. Up 6–3, take p_make; down 1–6, chase the upper tail. That's single-game strength, sitting on finished infrastructure, and it tests a lovely thesis you wrote into the bead: *the book's chapter organization may be an unwritten state→utility lookup table*.

Third, the evidenced home of book value: **the small end of your domino stack**. Detectors as features and Roberson-voiced rationales as curriculum for Gus-tiny, Burl, and LEM STaR. "The book helps players who can't afford the oracle" is a finding you already made — lean into it.

(One sleeper that strengthens *everyone*: belief-weighted world sampling — Gus's calibrated belief head reweighting the oracle's sampled worlds instead of uniform-consistent sampling. Your q-bootstrap-belief 0.655 result already hinted belief-sampled worlds beat corpus worlds. Improve the boss and every player downstream inherits it. That's also Gus finally doing the job he was conceived for.)

## The reframe that saves the fun part

Here's what I'd tell you if we were leaning on the project car: **the BSP algebra isn't wasted — it's pointed at the wrong phase.** At play time, "commit to a plan" is redundant with Q. At *auction* time, it's structurally true: a bid IS a commitment — `recognize` (hand shape) → `commit` (contract + declaration) → `execute` (pull trumps, cash count, double ahead of your off) → `retire` (made/set). The lifecycle, the recording contract, the replay purity — all of it is exactly the chassis a bidder-and-contract-player needs, and it's where the book's at-risk-points framework becomes executable rather than decorative.

**My recommendation**: build the bidder first — oracle-backed declaration sweep + book risk-budget policy, measured by full games to 7 marks instead of forced-bid-30 hands. Fold Lens v2's score conditioning into it (the bid decision is where mark state matters most). Keep the BSP algebra as the skeleton for contract plans. The play-phase singleton experiment can still run someday as a cheap epilogue — but as a measurement, not a hope.

That gives you the thing grandma's table actually had: a player who looks at a hand and says "31, on sixes" — and means something by it. Want to sketch what bidder Phase 1 looks like?


### Bidding inventory review

> **Jason:** "bidding. ok I'm interested. the thought was that bidding is kinda.. done? just slow... bidding work pre-dates the wiki significantly and might not have been promoted into it yet."

Found it all. Your memory is right — and the gap is precisely locatable. Here's the genuine answer.

## What exists: contract evaluation is done. Twice.

**`forge/bidding/` (January 2026, pre-wiki).** A complete Monte Carlo contract evaluator: hand → P(make) matrix over 9 declarations × 13 bid thresholds, Wilson CIs, convergence analysis (N=500 = ±0.04, ~70s/hand), a PDF poster generator, multi-GPU support, and a **continuous corpus generator** with a finished 365-column parquet schema and train/val/test splits. The funny part: as far as I can tell, **the corpus generator never ran** — there's no `data/bidding-results/` on disk. The January plan (simulate slow → corpus → presumably train a fast estimator) died at the corpus step, probably when attention jumped to LEM in April.

**`gus/bidding/` (April 24).** You quietly built generation two: same `evaluate_bids` aggregation, but Gus plays all four seats instead of the old 817k policy model — batched all-trumps × N games, and `find_best_bid` already prints "best bid: **31 on sixes** (expected mark swing +0.41)". This is a working "what would I bid this hand" oracle, seconds per hand.

**Around the edges:** the Wave-2 bid-aware E[Q] atlas (259k action rows across bids 30–84, mark-multiplier machinery validated, bid-only-enough *supported* with tight CIs), the `hand_eval()` static metrics in w42 (which literally implement Roberson's at-risk points: `unique_exposed_points`, `bid_ceiling_proxy`), and the TS-side `BeginnerAIStrategy` (5 sims per bid, fixed 50% threshold — elementary but wired into the real game). One known landmine documented in gen-fleet: corpus generation's action selection hardcodes bid-30 thresholds (`bid_value` was never plumbed through), so generated play is subtly bid=30-shaped regardless of recorded bid.

## The gap: you built contract evaluators. A bidder is not a contract evaluator.

Every piece above answers the same question: *"P(make) if I play contract (decl, B)?"* — the hand in a vacuum. None of it answers the question a player at grandma's table actually faces:

1. **The pass alternative isn't zero.** Passing means partner might bid, or opponents win a contract you get to *set*. The bid decision is marginal value vs. that counterfactual, and nothing computes it.
2. **The auction is information.** Partner opens 31 — his hand distribution just shifted, and so should your overcall math. Opponents bid 34 — Roberson's whole Ch 2 is about whether your hand can punish that. Nothing models bids-as-evidence anywhere in the stack (and this is exactly the belief machinery Gus already has for the play phase, unused for the auction).
3. **Score state.** Bid thresholds at 6–6 going to 7 are not the thresholds at 0–0. This is your Lens v2 / ICM insight, and it bites hardest at the auction.
4. **No full-game measuring stick.** Forcing bid=30 isn't just cheap — it dodges all three problems at once. There is no arena anywhere that plays *auction + play to 7 marks*. "Better 42 player" can't currently be measured, only "better contract executor."

So: "kinda done, just slow" is exactly right about evaluation, and the slow part is even mostly solved (Gus version). What's missing is the part that was never started — **the auction as a game** — and it happens to be where the book's validated content is concentrated.

## What richer bidding looks like, in three independently-valuable stages

**B1 — Finish the January plan with April's engine.** Run the corpus generator with `gus/bidding` as the simulator, then distill into a tiny bid-strength net: hand → full (decl × bid) p_make table in <1ms. Every seat at every table can now evaluate instantly. This is a weekend, and most of the code has existed since January.

**B2 — Auction policy v0 + the real arena.** A Roberson risk-budget policy over the B1 table: bid the minimum positive-mark-EV bid given score state, overcall rules, pass otherwise — Ch 2 as executable structure, maybe 100 lines. Then the thing I'd argue is the actual prize: a **full-game arena** — four players, real auctions, hands until someone reaches 7 marks. That replaces forced-bid-30 as the measuring stick for *every* player in your stack, and retroactively makes all your players comparable at the level the family actually plays.

**B3 — The auction as an information game.** Bids condition beliefs; beliefs condition the sampled worlds feeding the play-phase oracle. Partner's 31 literally changes which worlds Lens(ev) imagines. This is where the PhDs live, and the lovely part is you can stop anywhere — B1 and B2 are complete artifacts without it.

And to connect back to yesterday's question: B2 is where the BSP plan algebra naturally lands if you still want it — a won contract *is* a committed plan — and it gives Burl his best material yet ("I'd go 31 on sixes; my off is the 4-2 and it's covered"). That's at-risk points in your cousins' voice.

Want me to start with B1 — resurrect the corpus generator on top of `gus/bidding` and see what a night of generation buys us?


### Forward design — the best 42 player you could build

> **Jason:** "if you wanted to make the best 42 player you could. just wins every tournament and teaches everyone a better way to play.. how would you?"

I'd build it in two halves that share one spine: a player that's *provably hard to beat*, and a teaching layer that makes its strength *legible*. And the honest headline is: you are unusually close to the first half, because you already own the rarest piece.

## The spine: a belief-state player

The endgame architecture for any trick-taking game with hidden hands is the same shape — at every decision, bid or play:

1. **Maintain a posterior over the 21 hidden tiles**, conditioned on *all* evidence: the auction (partner's 31 means something), every play, every failure to follow suit.
2. **Sample worlds from that posterior** — not uniformly from "consistent" worlds, which is what the stack does today.
3. **Evaluate each world with the exact perfect-info solver**, marginalize.
4. **Choose under the right utility** — not points, not even p_make, but *marks-to-7 win probability* given the score.

This is the architecture behind the strongest bridge, Skat, and poker programs (belief-conditioned search over exact or learned values). 42 is *small* for this class — 7 tricks, ~400M worlds at deal time collapsing fast — which is why I think "wins every tournament" is genuinely reachable rather than romantic.

Now map it onto your shelf: the exact solver is **forge** — done, and it's the crown jewel; almost nobody building a hobby game AI has ground truth. The fast student is **Gus** (0.55 regret, 0.49 with routing). The posterior is **Gus's belief head** — exists, Bayes-ceiling-analyzed, but trained only on play-phase evidence and *never wired into world sampling*. The utility machinery is the **mark_ev transform** from Wave 2 plus your own Lens-v2 ICM insight. The missing organs are exactly the two we found yesterday: **the auction**, and **beliefs feeding the sampler**.

## Where the strength actually comes from

A thing bridge people learned the hard way: once card play is near-double-dummy, almost all remaining edge lives in the auction and in belief quality. Your card play is already near-oracle. So the marginal-value ranking is lopsided:

**Auction ≫ belief-weighted worlds > score-conditioned utility ≫ card-play polish.**

And there's one more piece nobody's stack has yet, which is what separates "strong engine" from "wins tournaments": **self-consistency**. A bid is only information if the policy that makes bids is the same policy the belief model was trained on. That's a fixed point, and you reach it by self-play iteration: play full games with the current bidder+player → retrain the belief model on those games → rederive the policy with belief-weighted oracle search → repeat. Two or three rounds and your partnership has an actual *bidding system* — conventions that emerged because they're optimal, not because someone wrote them down. For a game this size, that loop fits on the M5 Max with Modal bursts.

So my build order, concretely:

1. **The arena first** — full games, real auctions, marks to 7. Until this exists, "best player" isn't a measurable sentence. Everything else gets judged by game win rate here, forever.
2. **Auction v0** — `gus/bidding` + a Roberson risk-budget policy. Complete player, end to end, this month.
3. **Beliefs everywhere** — condition the belief head on auction + play history (the arena generates the training data for free), then swap uniform world sampling for belief-weighted sampling *inside the oracle player*. One change, and bidding, play, and defense all improve together. Highest-leverage ML move on the board.
4. **The self-play fixed point** — iterate policy ↔ belief until conventions stabilize.
5. **Marks-to-7 utility** over everything — your ICM idea, governing both bid thresholds and play risk.
6. **The summit, if you ever want it**: depth-limited subgame re-solving on late tricks (DeepStack-style, Gus V as leaf values) to squeeze out PIMC's last theoretical flaw. 42's endgames are small enough to solve *exactly* at the information-set level. That's the "multiple PhDs" door — and it's optional, because steps 1–5 already produce something no human partnership will beat reliably.

## The teaching half — and why your weird detour was secretly the point

Here's what I find genuinely beautiful: a maximally strong player is mute. Q-values aren't lessons. The thing that turns strength into teaching is a *concept vocabulary* mapped onto the champion's behavior — and that's what the W42 campaign accidentally built. You have 64 formalized claims with detectors, a methodology for supported/contradicted/context-limited, and at-risk points as the bidding frame. Run the champion through that battery and you get the real product: **where the book is right, where it's wrong, and what to do instead — with receipts.** "Roberson says double ahead of your off; the champion agrees 84% of the time, and the 16% has a pattern, and here it is." That's not a strategy guide, that's *Winning 42, second edition* — validated against something close to optimal play, written in your family's idiom through Burl, who already has the voice, the belief trajectory tool, and the post-commit Q&A surface.

And the domino-stack image you used is literally the architecture: **oracle → champion → Gus → Burl → LEM** is a distillation chain, and a distillation chain *is* a pedagogy chain — each level explains the one above to the one below. The strongest player and the best teacher aren't two projects. They're one object viewed from two sides.

If I were holding the wrench: arena this weekend, auction v0 the next, and belief-weighted sampling as the first real ML milestone. Each is a finished, enjoyable artifact on its own — and the third one is the moment the whole stack starts compounding. Want me to spec the arena?
