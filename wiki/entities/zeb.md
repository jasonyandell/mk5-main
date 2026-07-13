---
title: Zeb — AlphaZero-style self-play player, later Burl's parked belief primitive
kind: entity
first_seen: 62e3b53
last_updated: d858781
status: superseded
---

## What it is

Zeb is a self-play-trained transformer for Texas 42, built AlphaZero-style: MCTS search
over a policy/value net, bootstrapped from nothing but the rules (no oracle in the loop
at first, [[forge]]'s [[expected-q-value|E[Q]]] tables brought in later as a teacher
signal). It lives at `forge/zeb/` inside [[forge]]. Three model sizes were trained across
its build window — 75K, 557K, and 3.3M parameters — the 3.3M "large-belief" checkpoint
being the one still referenced by Burl's tool wrapper today.

Zeb's name was chosen deliberately for a PR headline: "high wtf factor," workshopped from
"a cute character name like from a retrofuturistic sharecropper in Texas in the 1930s"
(2026-02-01, [[sources/claude/era4-zeb-era|conversation digest]]).

## Precursor: the actual first commit (2026-01-31, era 3)

`forge/zeb/` did not begin on Feb 1. Its literal first commit is `62e3b53`
(2026-01-31T21:43:59, "implement AlphaZero-style self-play learning"), on the last night of
[[eq-genesis]] (era 3) — 10 files, 1,690 insertions, explicitly reusing `forge/oracle/tables`
for trick resolution and the Stage-2 tokenization format from [[expected-q-value]]. This version
had no MCTS: it was a REINFORCE-with-value-baseline self-play loop (`module.py`), skip-bidding
v1, no trained result claimed in-window. Two same-day follow-on commits added W&B integration
(`9225532`) and baseline evaluation (`bdcb503`, adding `forge/zeb/BASELINES.md`). No user-turn
conversation in the era 3 window discusses this commit or the name "Zeb" — the commit message is
the only contemporaneous artifact naming the system.

The next day's `b711311` ("add MCTS + ZebModel training pipeline," 2026-02-01, era 4) is a
rewrite/expansion onto real MCTS search, not the origin of `forge/zeb/` — that distinction was
missed by the initial era-4 pass on this page and is corrected here. The AlphaZero *question*
(the framing below) is genuinely an era-4 event; the *artifact* it built on top of is one day
older.

## Origin: the AlphaZero question (era 4, 2026-02-01 .. 02-16)

Zeb opened a new line of attack on [[candlewax|the wall]] — [[expected-q-value|E[Q]]]
computes p(make)/EV with no plan and no judgment — by asking whether a policy that
actually plays, rather than a value function that only scores, could be learned the way
AlphaGo Zero learned Go: from self-play alone. The question was posed the moment a prior
analysis thread was closed as a dead end (*"there are a lot of interesting discoveries in
here about 42 but ultimately they do not lead anywhere"*, 2026-02-01T01:31) and answered
with *"ok left turn"* → *"why is alphago zero style approach not appropriate for 42
again?"* (02:42) → *"yes prepare a statement for Claude code, let's do this buddy"* (02:59,
[[sources/claude/era4-zeb-era|conversation digest]]). Twenty-five commits landed the first
day, seven of them from parallel build agents standing up types/game/observation/model/
self-play/module/evaluate at once (`b711311`, 1888 lines).

Underneath the build sat a doubt that recurs through the whole era and was never fully
resolved: *"nobody knows what 'better' really is... there simply is no spoon because it's
imperfect information and 'best' move has a random component and always will.. right?"*
(2026-02-02T00:02). Twice, mid-build, the project deliberately checked its own drift — once
rejecting an external RL library (Sample Factory) after an 85-minute evaluation in favor of
the home-grown pipeline, and once naming the pattern outright: *"my gut says we haven't
actually tried the alphazero approach, we've tried proxies and standins and
maybe-instead-ofs"* (2026-02-02T21:28). The scar tissue underneath the vigilance was named
too: *"I don't want to be disappointed again, I admit it. I thought PIMC was gonna be sweet.
it wasn't. I've had a blast making this and now I'm scared to find out if it stinks"*
(2026-02-05T04:32, [[sources/claude/era4-zeb-era|conversation digest]]).

## Build arc

BUILT, over 16 calendar days, 129 commits, solo (`git log --oneline`, era4 range). Key
milestones:

- **`b711311`** (2026-02-01) — determinized-UCT MCTS + transformer policy/value model,
  first commit.
- **`fa4de59`** (2026-02-03) — shift from oracle-guided to true AlphaZero self-play with
  policy priors.
- **`8e6e5ed`** (2026-02-05) — CUDA-graph kernel optimization, measured on a personal
  3050 Ti after MCTS-on-GPU proved *"completely bewildering"* and a rented B200 gave only
  ~2.2× over the 3050 Ti (*"shockingly not that fast"*). Diagnosis: *"MCTS is
  latency-bound, not compute-bound. Algorithm doesn't fit hardware."* Crammed fully onto
  GPU anyway (~9×), params doubled in response.
- **`a7d6b5b`** (2026-02-06) — **1,010,176 self-play games in 5 days from first commit**,
  557K-param model, 70.9% win rate vs random, policy loss 0.35, checkpoint
  `zeb-557k-1m.pt` (2.25MB) still on disk. Worker/learner split onto separate processes,
  Hugging Face Hub wired as the weight-exchange bus between them: *"those 67 downloads ..
  are ME buddy. I have distributed workers that pick up the latest model and use that to
  generate new games, alpha zero style"* (2026-02-07). Jason's own read of the milestone:
  *"I am unqualified as an ML engineer, but this warrants a LinkedIn section, yeah? I mean
  it's a real thing"* (2026-02-07T16:09).
- **`93b7037`** (2026-02-08) — a belief head added to `ZebModel`, predicting which seat
  holds each domino. This head is the one that later becomes Burl's `get_belief` tool (see
  below) — and, per the era's own closing note, the one that never fed the policy (see
  "What the era left open").
- **Feb 6-9** — the [[zeb-fleet-ops|Vast.ai fleet]] stood up: reputation-scored spot GPUs,
  a CQRS monitor, HF rate-limit engineering. Jason's framing of the spend: *"I'd never do
  this for work. ever. but for the hobby project? let's gooo"* (2026-02-08T19:41).
- **`3a77bb6`/`477ff00`** (2026-02-09) — [[eval-matrix-bradley-terry]], a Bradley-Terry Elo
  ranking anchored on E[Q], fanned out over Modal T4 GPUs. Snapshot: `zeb-large-belief` =
  **1579**, neck-and-neck with `eq:n=100` at 1600 (N=100, explicitly flagged noisy).
- **Feb 11-14** — four calendar days, zero commits.
- **`0a9c171`** (2026-02-15) — large-belief recap doc: W&B run `waxffg2j`, 3,743 training
  cycles, 1.7M self-play games, **76.3% peak vs random at cycle 2730**, 3.3M params.
- **`3e6ad2b`** (2026-02-15) — a real shape bug fixed: E[Q]'s `[N,7,85]` outcome PDF was
  being fed to the policy head as a `[N,7]` target.
- **`6081420`** (2026-02-16) — the [[full-teacher-eq-experiment]] closeout: E[Q]-as-teacher
  does not move play past the ceiling. See that page for the full ledger.

`forge/zeb/` is not a shelved prototype — it survives on disk today and received a commit
as recently as 2026-07-06 (`838f48d`, arena-perf work touching the oracle decision path).

## What self-play alone bought, and where it capped

Pure self-play/MCTS climbed from ~50% (random-baseline noise) to **70.9%** vs random at
557K params in 5 days (`a7d6b5b`), then to a **76.3%** peak at 3.3M params
(`large-belief-recap.md`, W&B `waxffg2j`). Six-times-larger params bought roughly five
points — a curve shaped like an information/architecture ceiling, not a capacity one, even
though the closing commit calls it "capacity" (see [[full-teacher-eq-experiment]] for the
naming caveat). The recurring joke that turned out approximately true: *"I bet it's gonna
cap at 74 lol .. it's always 74 with this game"* (2026-02-09T03:36) — though this ~74%
vs-random win-rate plateau is a different quantity from era 3's [[argmax-q-ceiling]] (73.96%
argmax-vs-oracle action match); the numeric echo is coincidence. By Feb 10 the read was
explicit: *"I think our boy Zeb large-belief has peaked. I'll leave it training because it
costs me fewer dollars than my curiosity."*

**Zeb has never beaten [[expected-q-value|E[Q]]] n=10 at pure play — not in this window,
not since.** Its best measured position was neck-and-neck: Elo 1579 vs E[Q]'s anchor 1600
(N=100, [[eval-matrix-bradley-terry]]), and a direct head-to-head captured "~75% of the
random-to-E[Q] gap" (`0d8f336`). The 2026-07-06 [[w42-jud-v1|jud v1]] verdict (`afd4802`)
names E[Q] n=10 *"the play champion, as it has against every learned challenger since
Zeb."* Neck-and-neck is not beat. See [[vs-random-eval-is-suspect]] for a caveat on what
these percentages actually measure.

## What the era left open

Two things surfaced mid-era and were waved past rather than fixed:

- **Scoring rule vs. win condition** (2026-02-09): *"we're just doing 'winner = more
  points'. but more points does not necessarily win the game! ... there is no bid anywhere
  where getting <30 is a win."* Every vs-random percentage on this page — 70.9%, 76.3%, the
  ~74% "ceiling" — is graded against total points, not marks-to-7. See
  [[vs-random-eval-is-suspect]].
- **Belief feeding policy, never wired.** The belief head landed 2026-02-08 (`93b7037`) and
  predicted opponent ownership correctly enough to be worth publishing — but it never fed a
  single move choice; it sat beside the policy head, connected only through a shared
  training loss. The era's own last note names this directly: *"so how do we get beliefs
  feeding policy? right now we kind of have 2 things side by side only connected via loss
  essentially"* (2026-02-18T03:45). That question is answered, months later and by a
  different project, in [[belief-trajectory]] and [[gus]] — see
  [[belief-feeding-policy]].

The era's own closing reframe, offered as a question rather than a claim: *"what if this is
vaguely the limit of alphago in 42? it's not really the right target anyway. and we've
learned a TON. maybe the finding is 'alphago under imperfect information can carry you
pretty far, actually!'"* (2026-02-16T15:38). See
[[alphazero-under-imperfect-information]] for the full era narrative.

## Output (belief tool wrapper)

For each of 28 dominoes: `P(opponent ∈ {L, partner, R} | visible state)` — a probability
vector over the three non-self seats. (burl/OVERVIEW.md @ 8d26e0d)

## Training (belief-prediction accuracy — distinct metric from vs-random play strength)

Trained via self-play + oracle distillation. 72% top-1 accuracy on opponent-of-each-domino
prediction, over *all* 28 dominoes including already-played ones. (burl/OVERVIEW.md @
8d26e0d) This number is inflated — see "Parked for Burl" below.

## Role for Burl

[[burl]] calls Zeb via a tool wrapper (`get_belief(player, domino) → [P_L, P_P, P_R]`).
Zeb is the authority on hidden-state beliefs; the [[engine]] is the authority on visible
facts. The composite tool `conditional_outcome(play, assume)` uses Zeb to compute
P(assume | visible), then the E[Q] framework evaluates outcome conditional on that
assumption. (burl/OVERVIEW.md @ 8d26e0d)

## Role for LEM

LEM did not use Zeb directly. The forge pipeline (`generate_eq_games_gpu`) used an E[Q]
policy at N=10 for game generation; Zeb's specific belief model was not a LEM dependency.
Zeb is Burl-first infrastructure, though it is part of [[forge]].

## Parked for Burl (calibration eval, 2026-04-18)

Zeb's advertised 72% top-1 was calculated over all 28 dominoes, including dominoes already
played (trivially known from visible state). These inflate the number.

Hidden-only calibration (dominoes actually uncertain at time of prediction):

| Metric | Value |
|---|---|
| Top-1 accuracy | ~39% |
| Brier score | 0.224 |
| ECE | 0.067 |

39% top-1 on the hidden dominoes (the only ones that matter for belief-based reasoning) is
not reliable enough to anchor decisions on. See [[experiments/zeb-calibration-eval]] and
[[decisions/zeb-parked-eq-primitive]]. (commit message @ d9baf3b)

**Status**: Zeb's tool wrapper (`get_belief`) is shipped in `burl/tools/zeb.py` but parked
behind a flag in Burl's default tool list. The default checkpoint was also corrected from
`large-belief-bootstrap.pt` (untrained, std 0.036) to `lb-v-eq-3740-bootstrap.pt`. Both
fixes are available if Zeb is re-enabled once a better belief model is trained.

**Replacement at d9baf3b**: E[Q] N=10 outcome PDF (`burl/tools/eq_distribution.py`) became
Burl's belief primitive. Counterfactual shift validated: Δmean +15, p_make 0.6→1.0 on seed 900013.

## Superseded for Burl (2026-04-23, commit d858781)

Burl's production belief primitive is now [[belief-trajectory]], which exposes [[gus]]'s
`v3_consistency_10000g` adapter: per-domino posterior, shift-since-last, V (state value),
CLS attention. Gus provides a higher-quality, calibrated belief model.

Zeb remains parked — not deleted. The `get_belief` tool wrapper in `burl/tools/zeb.py` is
still present and re-enableable. Gus is the production path. (commit message @ d858781)

**Naming disambiguation (2026-07-06).** [[w42-jud-v1|jud v1]]'s evidence bundle
(`champion/evidence/jud_v1/zeb_protocol_judplay_summary.json`) reuses "Zeb-protocol" as
the name of an eq-n=10-vs-judsearch/judplay A/B naming convention. This is unrelated to
the Zeb belief model on this page — no retraining of Zeb occurred, and the jud usage is
purely a naming coincidence.

## Potential role for [[book-strategy-player]] (2026-05-03)

The book-strategy-player architecture (designed 2026-05-03; build pending) opens two
candidate roles for Zeb:

1. **Training-data generator.** Zeb's self-play infrastructure could supply structured
   training data for the strategy-selector model (Model A) or end-to-end policy (Model C).
2. **Model C target.** Zeb's policy head, retrained on strategy-labeled data, has a much
   smaller structured output space (15-30 named strategies vs raw 7-domino choice).
   Faster to train, more interpretable, natively produces strategy attributions.

Speculative until the book-strategy framework is built and recording starts.

## Corrections to the received story (era 4)

- **The founding commit is `62e3b53` (2026-01-31, era 3), not `b711311` (2026-02-01, era 4).**
  See "Precursor" above. The era-4 narrative below is accurate about the AlphaZero *question*
  being asked and answered starting Feb 1 — but `forge/zeb/` as a repo path, and the name
  "Zeb," predate it by one day.
- **"Reinforce-to-flywheel" / "six weeks of RL training" was never an experiment, module,
  or bead.** It is the title of one 2026-02-05 conversation plus a single recap sentence.
  Zero commit, bead, or file carries the name. The six-week arc it describes continued
  entirely under Zeb — do not cite it as a separate run.
- **Sample Factory was never built.** One 85-minute conversation, evaluated and rejected
  in-session in favor of the home-grown pipeline. No commit, no file, no bead.
- **The "Crystal Palace word cloud" was a 3-minute nostalgic aside, not a course of work.**
  "Crystal Palace" itself predates era 4 (Nov-Dec 2025 name for the suit-algebra/rules-base
  foundation) and was not coined or renamed here.
- **Beads tracked zero activity in this window.** All 652 records scanned on all three
  timestamp fields: nothing falls in Feb 1-16. The `t42-*` IDs in commit messages
  (`t42-rvhp`, `t42-1xpp`, `t42-4xvg`, ...) are commit-message annotations with no matching
  bead records.
- **`lb-v-eq-1920` and `lb-v-eq-3740` are two different runs**, not one under two names — a
  later full-teacher run vs. an earlier Feb-15 large-belief bootstrap.

See [[sources/claude/era4-zeb-era|conversation digest]] and
[[alphazero-under-imperfect-information]] for the full era treatment.
