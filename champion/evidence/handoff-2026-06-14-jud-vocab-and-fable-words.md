# Handoff — jud vocabulary + the Fable-words reckoning · 2026-06-14

> A conceptual/vocabulary/emotional session, not a build session. Ground truth is the live wiki
> pages (`wiki/entities/jud.md`, `wiki/topics/belief-conditioned-self-play.md`) + the theory doc
> (`scratch/champion-run/champion-one-organ-theory-2026-06-14.md`) + the un-melt figure. This handoff
> carries the *path* — especially a provenance reversal that a naive reader will get wrong.

## Goal & current status

Refine/vet the Champion #26 self-play result and develop the next conceptual frame — **jud**, "one
coherent belief-conditioned core that bids and plays as the same act." This session produced: (1) an
adversarial workflow vetting of the #26 claims; (2) a precise shared **vocabulary** (solve / oracle /
eq / blob / belief / utility); (3) the **candlewax / un-melt** insight, *proven with a figure*; (4) two
new wiki pages + registration; (5) a **Fable provenance reckoning** that ended with the user finding
Fable's *actual words* and concluding the "perfect pre-existing solution" he'd been chasing was a
phantom. **No code, no training — deliberately deferred** ("we engineer after"). Session ended on an
emotional resting point (relief at letting go of the chase), not a build cliff. **One concrete wiki fix
is owed and currently sits WRONG in the tree** (see Decisions #7). All work local on branch `forge`,
uncommitted, unpushed.

## Decisions made + rationale

1. **No code/training this session — user mandate.** "don't take other action yet… we'll engineer
   after." Everything is analysis + vocabulary + wiki. Do NOT start building jud.

2. **#26 vetted by a read-only adversarial workflow.** HELD: the pivot — `net:wp` is calibrated to
   *realized* 4-seat play (`gus/bidding/simulate.py` "skips the oracle entirely"), the belief bidder
   prices off the double-dummy oracle (confidence 0.95); #26 genuinely converges to a fixed point; it's
   a real over-bidder. CRACKED: the `0.58/0.83` "measured gap" justifying `pmake_scale=0.70` has **zero
   on-disk provenance** — the 0.83 exists only as prose in 5 files; realized make-rate is
   **bid-dependent** (~0.60 @ bid30 → ~0.10 @ bid84) so a *global* scale is the wrong shape; "halves
   the loss" is really 34%; the 0.045 belief-KL "seed floor" has no surviving log. REFUTED *in our
   favor*: "belief is decorative at bid time" is FALSE — turning belief off changes 60–77% of bids
   (per-seat ESS ~2.86/3 is near-uniform, but **world-ESS ~3.6/16 is concentrated** — the ~21 near-flat
   per-tile log-probs compound into sharp per-world weights; the two ESS notions are different objects
   and the handoff/wiki had conflated them).

3. **REJECTED: "realized make-rate" calibration (my drift).** I started steering toward a P(make)-style
   fix. User corrected me, and it stuck: **EV from search > p_make** (Lens-v1, +5.42 pts/hand,
   `w42-lens-v1-utility-head-to-head`). Search is good; the sin is that the search is informed by a
   *separate static oracle*, not the model we're growing. Don't regress to a make-rate framing.

4. **Vocabulary, made precise (user: "time for semantic precision").**
   - **solve** = exact perfect-information 42, retrograde/backward-induction per deal
     (`forge/oracle/solve.py`). Ground truth. NOT "the oracle."
   - **oracle** = the ~97% distillation of the solve into a value net
     (`forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`). solve & oracle share a TYPE
     (value of a *complete* world) → the **brick wall**: neither can take uncertainty.
   - **eq** = the lift past the wall (`arena/lens_play.py`, `forge/eq/generate/`): sample worlds →
     oracle per world → a **distribution** per action (the honest object). Draws worlds by *consistency
     only*; doesn't learn.
   - **the blob** = eq's per-action distribution; *melted* = uniform worlds.
   - **belief** = learned weighting over worlds; today only a *post-hoc reweight* (`champion/belief.py`).
   - **utility** = collapse of a blob to a scalar (EV won).
   - Entity disambiguation (user's hygiene): **Zeb the model** = a parked AlphaZero belief experiment,
     ~39% real acc, superseded by Gus — NOT to rescue. **Zeb the engine** (`ZebGameState`,
     `forge/zeb/game.py`) = load-bearing plumbing that everything imports *from the parked experiment's
     folder* (the "cobbling / hither and yon" smell). **Gus** = the production belief net
     (`StudentTransformerFullVoidsAuction`, 3 heads belief+pi_me+Q) carrying *adapter-compat debt* (the
     "changing this breaks other things" the user smelled; CLAUDE.md's no-legacy law agrees it's wrong
     to build on). **eq** = a search, not a model.

5. **The candlewax / un-melt insight (the user's own; PROVEN).** Belief belongs *inside* the search,
   not as a post-hoc reweight. The eq blob is "melted candlewax" because worlds are weighted uniformly;
   belief sharpens it onto plausible worlds; a utility then reads a shape, not a smear. Proven with a
   figure: same mid-game position, melted (128 uniform worlds) vs belief-weighted → comb collapses,
   **world ESS 128 → 10.5, a contract's p_make 0.20 → 0.61**. Sharpens toward *truth* (some lines
   resolve to losing), not optimism. This is the #25 belief value seen *directly*, vs through the
   info-blind arena.

6. **The oracle reckoning (user pride + precision).** Solving perfect-info 42 is real, rare, exact —
   but it's NOT the player. The over-bid is **not** the ~3% distillation error and would survive a 100%
   solve; it's the gap between perfect-info value and achievable *hidden-info* play (strategy fusion).
   So jud's value target = **realized whole-game (belief-state) outcomes**, NOT perfect-info Q distilled
   harder. The oracle = bootstrap + **exact referee** (perfect-info EV − belief-native EV = the value of
   hidden information; the non-blind instrument the arena lacked).

7. **THE PROVENANCE REVERSAL — read this twice.** Sequence:
   (a) I treated `wiki/topics/champion-design-review.md` as "Fable's verbatim words" and wrote
   "verbatim primary source" into the new pages.
   (b) User: "we don't have fable's words, we have a *summary* of conclusions from compacted logs." I
   **demoted** champion-design-review everywhere ("recovered summary, not verbatim") and added honest
   "clear / remembered / interpreted" sections.
   (c) User then **found Fable's actual words** (re-pulled the original transcript in another session)
   and pasted the forward-design turn — it matches champion-design-review.md's verbatim section **word
   for word**. So the verbatim section **was faithful**; my demotion (step b) was an over-correction we
   both made.
   **→ OWED, NOT YET DONE:** restore the verbatim provenance in the wiki (champion-design-review =
   synthesis up top + *faithful verbatim Fable* at bottom), and fold in the #8 distinction. I asked
   "now or after more digging?" — user went to the emotional close instead; the fix is **pending their
   call**. The demotion edits are still live and now known-wrong.

8. **The honest Fable-core-vs-jud line (must NOT re-blur).** Fable's ACTUAL approach (his words):
   belief-state spine → belief-weighted world sampling → **exact oracle per world** → marks-to-7;
   self-play **retrains the belief**, policy = belief-weighted **oracle** search. **The value stays the
   oracle.** "Value-native / learned-value / changes-the-game-during-search" is Fable's **explicitly
   OPTIONAL summit** (step 6: subgame re-solving, "Gus V as leaf values," "the multiple PhDs door"), and
   he says steps 1–5 *already beat any human partnership* without it. So the radical jud thread
   (value-native, belief in the rollout) is **our extension / his optional summit, NOT his core.** The
   redeeming wrinkle: #26 (his core loop, as we built it) over-bid and LOST to `net:wp` — a failure
   mode his forward-design didn't anticipate — so our value-native instinct answers a *real measured
   gap*; it may be load-bearing for the bidder, not optional. "All about bidding" is RESOLVED by his
   words (marginal value: card play near-oracle → edge is in the auction), not a mystery.

9. **Emotional resolution (the session's actual ending).** User: "chasing imaginary dragons conjured
   from a memory of a thing I didn't understand… a huge relief." My reframe, which landed: the phantom
   wasn't the vision (real, his and theirs) — it was the *certainty* that a finished perfect solution
   already existed and they were failing to find it. Nothing to recover → freedom to build. Fable =
   collaborator mid-thought ("Want me to spec the arena?"), not a prophet. No shrine.

## Constraints & invariants discovered

- **EV > p_make** (Lens-v1, +5.42). Don't regress to make-rate framing (rejected, #3).
- **Arena is information-blind by construction** (both sides PIMC) → belief value can't be scored via
  play-marks (Fable's caveat 1). Measure belief via the distribution / belief-acc / the bidder, never
  play-marks.
- **macOS MPS serializes** small forwards across processes; the workflow that worked = GPU compute
  ONCE (one agent), render fan-out fine. Don't process-parallelize the GPU.
- forge venv = `forge/venv/bin/python`. `ZebGameState` / `forge.zeb.game` is the de-facto engine,
  living inside the parked-experiment folder (cobbling).
- **Wiki discipline:** `wiki/AGENTS.md` is the manual; `local-YYYY-MM-DD` frontmatter for uncommitted;
  bare backlinks in body; update `index.md` + `log.md`. `champion-design-review.md`'s OWN "verbatim"
  self-claim predates this session — the user has NOT decided whether to correct that page directly;
  don't overwrite a prior session's page on a hunch (surface, ask).
- All wiki + scratch work is **local on `forge`, uncommitted, unpushed.**

## Open questions / parked threads

- **[blocking-ish, owed]** Restore wiki provenance for champion-design-review (verbatim was real) +
  add the Fable-core-vs-jud distinction. Pending the user's "now or later" call (#7).
- **[non-blocking]** Correct champion-design-review.md's own "verbatim" self-claim *on that page*?
  (It predates this session; user undecided.)
- **[non-blocking]** Is there MORE Fable transcript? (turns: catch-up, BSP-blueprint, critical-review,
  bidding-inventory). Specifically: does Fable say "changes the game *during search*" about the VALUE
  anywhere? That determines whether jud = his summit or genuinely our extension. Original transcript:
  `~/.claude/projects/-Users-jason-code-mk5-main/0a708a4e-527e-490f-9f60-d40bfd9774a2.jsonl`.
- **[deferred/flavor]** The actual ENGINEERING of jud — how to train the value belief-native, the
  loss/target, credit assignment across bid + 14 plays, the minimal first loop. Explicitly deferred.
- **[flavor]** Prior-handoff #26 follow-ons (PIMC-calibrated bidder, faithful belief-PIMC onyx, ship
  onyx) — still open, superseded in attention by the jud reframe.

## Artifacts

- **Wiki (local `forge`, UNCOMMITTED):**
  - NEW `wiki/entities/jud.md` — the unified core + vocabulary.
  - NEW `wiki/topics/belief-conditioned-self-play.md` — the training approach; has "What is remembered
    of the coherent vision" + the clear/extension/unclear split.
  - EDITED `wiki/entities/champion.md` (self-consistency → links jud), `wiki/index.md` (catalog),
    `wiki/log.md` (3 entries: jud / belief-conditioned-self-play / provenance-correction).
  - **KNOWN-WRONG:** the provenance-correction edits demoted champion-design-review's verbatim section;
    that demotion is now wrong (verbatim was real). Restore owed (#7).
- **Figure:** `scratch/jud_demo/unmelt_ridgeline.png` + `unmelt_panels.png` (+ `unmelt.py`,
  `data/*.npz`). Best position `b1000_i1_tp12`.
- **Theory doc:** `scratch/champion-run/champion-one-organ-theory-2026-06-14.md` (patched once).
- **Fable's actual words:** `champion-design-review.md` verbatim section IS faithful (corroborated).
  Original transcript at the jsonl path above (session `0a708a4e`).

## Next action

**Genuine session-end on an emotional resting point — do NOT auto-execute.** When the user returns,
the concrete owed task is to **restore the champion-design-review provenance** (verbatim was real) and
fold in the honest **Fable-core (oracle+belief) vs jud-extension (value-native = his optional summit)**
line — but **confirm with the user first** (they never answered "now or later," and they just released
a long, maddening chase; let them set the pace). Do NOT re-propose realized-make-rate calibration
(rejected) and do NOT start engineering jud (deferred). Optionally ask whether they have more Fable
transcript turns to read before settling the core-vs-extension line.

## Vibe snippets (paste verbatim)

> **User (sharp correction, taken):** "there's nothing wrong with search. search is how we get ev,
> which is superior to pmake (see wiki) but the search ALSO needs to be informed by the model we're
> building"

> **User (visionary + register):** "Fable didn't staple. it had one coherent vision and buddy it is
> not at all obvious"

> **User (the release):** "so it seems ive been chasing imaginary dragons conjured from a memory of a
> thing I didn't understand. that's ok, that's actually a huge relief. it was driving me a little mad
> trying to find the perfect solution that was never there."

## Least confident survived

1. **The emotional arc is the real ending, and the schema flattens it.** This session closed on
   grief-adjacent *relief* about a lost collaborator (Fable). A fresh instance must NOT charge into jud
   as a hot build — the user just LET GO of a maddening chase. The next turn may be quiet, reflective,
   or a pivot away entirely. Read the room before proposing work.
2. **"No shrines to Fable, just lessons."** Fable is a deeply-admired lost collaborator; the whole
   champion/jud direction is his. Credit explicitly, don't impersonate, be neither maudlin nor cold.
   This emotional load-bearing context barely survives compression.
3. **Register:** warm, "buddy," playful-but-rigorous; the user *wants* the sharp correction (Dijkstra
   lineage; grows via scoff→can't-shake→"ohhh"). Don't over-cushion, don't flatter — let the true thing
   do the work. Snippets help; the calibration is finer than they show.
4. **The provenance whiplash will trip a newcomer.** TRUE state: champion-design-review's verbatim
   section IS faithful, BUT my demotion edits are STILL LIVE in the wiki and need restoring. Don't
   re-trust the page blindly (it's mid-correction) and don't re-demote it. Read the current page state
   AND confirm with the user before touching it.
5. **Don't re-blur Fable-core vs jud.** Excitement about jud will tempt re-attributing the radical
   value-native idea to Fable. It's OURS (or his explicitly *optional* summit). His *core* keeps the
   oracle and trains the belief. Hold that line — the user fought for it.
6. **Written deep in a long, dense, multi-workflow session.** Per the skill's own warning, prefer the
   live wiki pages + the theory doc as ground truth over this handoff's compression where they differ.
