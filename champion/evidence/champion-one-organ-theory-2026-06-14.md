# The one organ — Champion theory, coherent with Fable's framework

> Written 2026-06-14 (UTC ~15:00) as Jason processes the direction. This is the synthesis
> from the working session: Jason's restatement of the vision, laid over Fable's verbatim
> design review (`wiki/topics/champion-design-review.md`, recovered from transcript
> `0a708a4e`), the epic (#29), and the build ladder (`wiki/entities/champion.md`).
> Captured as-is. Not yet a spec — the theory.

---

The dig settles one thing first: **the understanding Fable "wrote down" is the verbatim design
review in the wiki** (`wiki/topics/champion-design-review.md`) — recovered from transcript
`0a708a4e`, the five turns reproduced exactly because they'd been compacted out of live context.
There's no other Fable write-up hiding in scratch — only MISSION.md (which points *at* the design
review and logs "DIRECTION LOCKED — Fable's vision"), the handoffs, and `run_26_selfplay.sh` ("the
self-play fixed point (Fable's vision)"). So the framework to be coherent with is that one page.

## Yes — more accurate. And one clause is a correction.

The drift last turn toward "realized make-rate" is just p_make wearing a playout costume.
"Search is how we get EV, superior to p_make" is not only correct, it's *the* settled result of
this project: Lens v1, EV beats p_make by **+5.42 pts/hand, every CI excluding zero**
(`w42-lens-v1-utility-head-to-head`), and Fable's spine step 4 says it in words — "not points, not
even p_make, but marks-to-7 win probability." The signal is the whole value distribution from
search, not a made/missed bit. Search stays. The sin was never search — it was *what informs the
search*.

## Jason's words, laid over Fable's

- **"not one system here that predicts another there"** → Fable's *self-consistency*: "a bid is only
  information if the policy that makes bids is the same policy the belief model was trained on."
  Today's stack violates it three ways — the bidder evaluates worlds with a *separate* static
  oracle, `net:wp` is a *separate* distilled net, the belief is a *third* thing. One predicts
  another. The vision collapses them.
- **"the search ALSO needs to be informed by the model we're building"** → Fable step 3, "swap
  uniform world sampling for belief-weighted sampling *inside* the oracle player," extended: not
  just reweight the worlds, but let the model *be* the evaluator.
- **"trained on whole games / self-play iteration until convergence / that's the fixed point it was
  referring to"** → Fable, exactly: "play full games → retrain the belief on those games →
  re-derive the policy via belief-weighted search → repeat. Two or three rounds and your partnership
  has an actual bidding system — conventions that emerged because they're optimal, not because
  someone wrote them down." That sentence *is* the emergent signaling. The "fixed point" was always
  pointing here.

Four of the clauses sit directly on Fable's framework. It's coherent because it *is* the framework,
stated more tightly than the ladder ever managed.

## Two places the restatement goes *past* what Fable wrote down

**1. "including in the rollouts!" promotes the summit to the spine.** Fable's written step 3
evaluates each world with the *perfect-info* oracle — that's PIMC, and Fable knew it: "the only
value PIMC structurally misses is information-set value: concealment, partner inference, strategy
fusion." He parked the fix at step 6, the *optional* summit (re-solving with "Gus V as leaf
values"). Moving the belief *into* the rollout and making it the center is not a contradiction of
Fable — it's the resolution of his own **caveat 1**. The convention-value lives *entirely* in the
info-set terrain PIMC is blind to, so it can't be a capstone; it has to be in the rollout from the
first ply. The summit becomes load-bearing.

**2. #26's fixed point was a shadow of the real one.** #26 genuinely converged — but it converged
the *belief head* while the *value* stayed double-dummy oracle. So it found the fixed point of a
crippled loop: a perfectly calibrated **over-bidder**. The fixed point Fable meant is where the
*value itself* is belief-native and self-trained on whole-game marks. At *that* fixed point the
over-bid can't exist — it isn't a calibrated-optimism equilibrium, it's the convention equilibrium.
#26 reached for it and caught the smaller one.

## The one real divergence — the oracle — and the refinement

"throw out our existing model which nobody needs anyway." This is where the vision and Fable part.
Fable called forge "the crown jewel — almost nobody building a hobby game AI has ground truth," and
used it as the per-world evaluator. The honest read: it's right that the oracle **leaves the
decision loop** — the perfect-info crutch is exactly what causes the over-bid *and* blinds the agent
to signaling. But don't delete it; **promote it from player to referee.** Two jobs it's uniquely
good at once it stops playing:

- **Bootstrap** the value head so self-play isn't cold-start tabula-rasa (with ground truth in hand,
  warm-starting the value is free).
- **The instrument the project never had.** The arena is info-blind (Fable's caveat 1: "the harness
  can't reward concealment"). But *double-dummy EV minus belief-native EV is the value of hidden
  information* — the strategy-fusion gap, the exact thing measured as 0.83→0.58. As self-play learns
  to infer and signal, realistic EV climbs toward double-dummy and that gap closes. **The oracle,
  retired from the table, becomes the ruler that finally measures the signaling it could never use.**
  That's how to watch conventions emerge with receipts instead of faith.

So: out of the loop, yes. Out of the project, no — it's the bootstrap and the scorekeeper.

## The one theory, said simply, coherent with the framework

> One network maps a belief-state to three coupled things — what's hidden, what I'd do, what it's
> worth. Search produces EV by rolling out *with that same network, beliefs and all, in the
> rollout*. Self-play over whole games to 7 marks trains all three heads together until they stop
> moving. Bidding and play are the same decision at different depths of one belief-game. The oracle
> bootstraps the value and referees the gap. The fixed point is Fable's self-consistency — bids mean
> something, so conventions emerge because they're optimal.

Nothing stapled. One core, all the way through. The bones are unusually complete for it — but as
*prior art to learn from, not foundations to inherit*. The one-trunk-three-heads pattern (belief +
policy + value) is already proven trainable here by Gus (`StudentTransformerFullVoidsAuction`), but
the clean organ is built **fresh**, with no obligation to old adapters/tokenizers — the oracle's
strength can warm-start it; its debt does not come along. The arena exists; the self-play bridge
exists; 42 is small for this class — it's a real build, the genuine frontier of the field (ReBeL /
Student-of-Games territory: learn a value over *public belief states*, search on belief states,
self-play in the loop), but not a moonshot.

---

## Appendix — Fable's framework, for reference

**The spine (forward-design turn), at every decision, bid or play:**
1. Maintain a posterior over the 21 hidden tiles, conditioned on *all* evidence (auction, every
   play, every failure to follow suit).
2. Sample worlds from that posterior, not uniformly over consistent worlds.
3. Evaluate each world with the exact solver, marginalize.
4. Choose under marks-to-7 win probability given the score — not points, not p_make.

**Marginal-value ranking:** auction ≫ belief-weighted worlds > score-conditioned utility ≫
card-play polish. (Empirically refined #20–#28: the ranking holds *for the auction*; both play-side
levers — score-conditioned play #27, belief-weighted play #25 — measured dead, *because the arena is
information-blind*, not because the levers are worthless.)

**Self-consistency (the separator from "strong engine" to "wins tournaments"):** a bid is information
only if the policy that makes bids is the policy the belief was trained on → self-play fixed point →
conventions emerge as equilibrium artifacts.

**The summit (step 6, optional in Fable's plan; the spine in this restatement):** depth-limited
subgame re-solving on late tricks with learned leaf values — the fix for PIMC's information-set
flaw.

**The teaching half (one object, two sides):** a maximally strong player is mute; the W42 detector
vocabulary makes its behavior legible. oracle → champion → gus → burl → lem is a distillation chain
*and* a pedagogy chain. Target artifact: *Winning 42, second edition* — validated against
near-optimal play, in the family idiom.

**Two load-bearing caveats that were distilled out and steered later sessions wrong:**
1. The arena is information-blind by construction (both sides PIMC) → belief/concealment value
   cannot be rewarded via play-marks; show it through belief accuracy or the bidder/defense.
2. Score-conditioning was tested in the wrong phase (#27 tested *play*, −1.20); Fable located
   mark-state value at the *auction*, still unrun.
