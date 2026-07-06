# jud v1 build report — the unified belief-conditioned value organ

Built 2026-07-06 on branch `worktree-agent-a2449b66c23008542` (base `68fda7b`).

One net, two consumers. `champion/jud_net.py` extends the v0 value-native
mechanism (`margin_net` + `ValueBidder`) from bid-time to every decision:
info-state (own hand + canonical auction + play history so far) → 43-bin
categorical over the declaring team's realized points, same target/loss/
readouts as margin_net. Bid-time is play-time with an empty history — one
featurization, one net, no special cases.

## 1. Play-decision snapshot emission (arena)

**Design: complete + compact, not sampled.** `HandRecord` gains one field,
`plays` — the terminal state's full `play_history`, 28 `(seat, domino)` pairs.
`snapshot_rows` emits it per hand. Every play decision is a *prefix* of it:

- mover at step k = `plays[k][0]`; current trick = `plays[4*(k//4):k]`
- trick winners are implicit: the leader of trick t+1 (`plays[4*(t+1)][0]`)
  won trick t (asserted in tests via `resolve_trick` replay)
- all ~28 per-decision info-states — offense AND defense — ride in one row,
  each sharing the hand's realized-outcome stamp (`bidder_team_pts`)

This works identically in sequential and `fast_batching` modes (the field is
read off the terminal state, not captured per tick). The bid-snapshot fields
are unchanged; `MarginDataset` consumes the new corpora as before. The CLI
snapshot dump is now compact JSON (`indent=2` would be ~30× the lines with
play histories). Cost: ~230 bytes/hand.

## 2. Featurization spec (350 dims)

`x = [hand 63 | auction 28 | play 259]`, all from
`champion.jud_net.featurize(hand, bids, bidder, dealer, decl_id, plays, seat)`.

- **hand [0:63]** — `bid_net.featurize_hand` on the observing seat's ORIGINAL
  7 dominoes; constant through the hand (which of them are gone is in the
  play block).
- **auction [63:91]** — `margin_net.canonical_auction` reused verbatim: the
  level-blind, later-seat-masked encoding (bidder's own level := 30, seats
  after the bidder := pass) — the v0 selection-leakage defense, kept EXACTLY.
  (Seats after the winner necessarily passed, so the masking is lossless;
  only the level erasure discards anything.) The one v1 difference:
  `auction_feature_vector`'s POV is the DECISION seat, so `is_winner` at
  relative seat r tells the net where the declarer sits — the offense/defense
  channel. At seat = bidder this is byte-identical to margin_net (tested).
- **play [91:350]** — per-domino map + global tail, POV-relative, derived
  from the play list alone:
  - per domino d (9 dims at `91 + 9*d`): played-by relative seat one-hot
    [0:4] (zero if unplayed), position-in-trick one-hot [4:8] (0 = led),
    trick index /6 [8]
  - global tail (7 dims at 343): declaring-team pts so far /42, defending-team
    pts so far /42 (complete tricks via `resolve_trick`), n_played /28,
    current-trick fill one-hot (n_played % 4)

  Who played what, in what order, trick winners, and the running score are
  all recoverable; nothing else is present. Belief stays implicit exactly as
  v0 made it for the auction: conditioning on everyone's actions.

Entry points: `featurize_snapshot(snap, step, seat)` (train) and
`featurize_state(zeb_state, seat=None)` (serve) are thin unpackers of the one
`featurize` — byte-identity between them is tested by replaying real match
hands through the zeb engine and comparing every pre-move and post-move
info-state (`champion/test_jud_net.py::test_featurize_state_matches_snapshot_prefix_over_real_match`).

**Model**: MLP 350 → 512 → 512 → 43 (~470K params). The margin_net family one
notch wider for the 3.8× input. A transformer is unjustified while the play
block is a fixed structured map, not a token sequence; the loss curve (see
§6) says the current binding constraint is regularization, not architecture.

**Dataset** (`JudDataset`): each hand expands to the exact query set the two
consumers issue — for each play step k (mover m), the info-state BEFORE the
move from m's POV (the decision; k=0's mover is the bidder, so the bid root
is a plain row) and AFTER it (the child JudPlay prices). ~56 rows/hand, one
Monte Carlo label per hand (no bootstrapping — 7-trick horizon). Exact dedup
(paired-half identical auctions collapse at step 0; a deterministic A==B
self-play collapses entirely); 90/5/5 split by `margin_net.split_of` deal
hash, so a hand's rows never straddle splits.

## 3. Consumers + registry specs

- **Bidding** — `JudNet.pmake_table` has `MarginNet.pmake_table`'s exact
  signature, so `champion.value_bidder.ValueBidder` consumes it with zero
  adapter. Spec: `jud[:wp][,pass[<q>]][,model=<path>]` (defaults:
  MarksToSeven, `champion/jud_net.pt`), mirroring `margin`.
- **Play** — `arena/jud_play.py::JudPlay`: at each decision, price every legal
  move by querying the net on the info-state AFTER that move (mover's POV —
  exactly the "evaluation" rows in the corpus), argmax E[pts]; a defender
  flips the sign (the net predicts the DECLARING team's points; margin =
  2·E[pts]−42 is affine, so argmax E[pts] = argmax margin-EV). All legal-move
  queries across the whole tick batch go through ONE forward pass. No world
  sampling, no oracle at runtime. Spec: `judplay[:model=<path>]`.

## 4. Commands

All from repo root with `forge/venv/bin/python -u`.

(a) **Corpus generation** (any arena match now emits play-decision corpora):

    python -u -m arena.cli --team-a margin:wp+lens:ev --team-b margin:wp+lens:ev \
        --n-games 256 --device mps --base-seed 9000000 \
        --out-dir scratch/jud-v1/sp_results \
        --emit-snapshots scratch/jud-v1/corpus_lens/sp_00.json

(b) **Training / eval**:

    python -u -m champion.jud_net train --corpus 'scratch/jud-v1/corpus_lens/*.json' \
        --epochs 60 --lr 3e-4 --device cpu          # saves champion/jud_net.pt
    python -u -m champion.jud_net eval  --corpus 'scratch/jud-v1/corpus_lens/*.json'
    # eval reports overall reliability + per-trick CE/MAE/ECE slices

(c) **Round-0 A/B**:

    python -u -m arena.cli --team-a jud+judplay --team-b net:wp+lens:ev \
        --n-games 64 --device mps --base-seed 4243 \
        --out-dir scratch/jud-v1/ab_round0b \
        --emit-snapshots scratch/jud-v1/ab_round0b/snaps.json

A jud loop round = (a) with `jud+judplay` self-play → (b) on the cumulative
corpus → (c); v0's `scratch/jud-v0/loop/run_loop.py` is the template
(swap `ValueBidder(load_margin_net(...))`/`LensPlay` for the `jud`/`judplay`
registry entries and `MN.train` for `champion.jud_net.train`).

## 5. Test status

`forge/venv/bin/pytest arena champion` — **133 passed, 3 skipped** (the 3 are
the pre-existing device-gated skips; baseline at 68fda7b was 111+3). New:

- `arena/test_snapshots.py` — plays shape/rotation/ownership; replay via
  `resolve_trick` reproduces the stamped outcome exactly; CLI round-trip.
- `champion/test_jud_net.py` (12) — train/serve byte-identity over a real
  match replay (pre- AND post-move rows); empty-history bid root byte-equal
  to `margin_net.featurize`; level-blindness; hand-built trick encoding
  checks; POV rotation; dataset expansion/dedup/splits; bid-only-corpus
  rejection; training smoke; pmake_table bounds/monotonicity; ValueBidder
  consumption.
- `arena/test_jud_play.py` (9) — defender-sign (stub head with a readable
  preference: offense plays it, defense avoids it, both in one batch);
  legal-move masking (illegal favorite never chosen; forced-follow states);
  batched == single choices; full CPU matches through the registry.

## 6. Round-0 measurements (smoke, honest)

Head: `champion/jud_net.pt` — 4,277 hands (227K rows) of `margin:wp+lens:ev`
play, lr 3e-4, best-val checkpoint. Val CE 2.564, test ECE(p30) 0.023.
Per-trick MAE(mean pts): 8.6 → 3.5 from root to terminal — the value sharpens
as evidence accrues, the jud signature.

64-game A/Bs vs `net:wp+lens:ev` (MPS, fast-batching):

| team A                | margin/game | offense share | A made | note |
|-----------------------|------------:|--------------:|-------:|------|
| `jud+judplay`         | −6.09       | 92.9%         | 10.2%  | combined |
| `jud+lens:ev`         | −4.08       | 92.9%         | 28.0%  | bidder isolated |
| `margin:wp+judplay`   | −5.53       | 73.6%         | 19.0%  | play isolated; B made 87.4% |
| `heuristic+judplay` vs `heuristic+random` | **+1.80** | 49.9% | 52.8% | floor: beats random play |

Reading: plumbing is proven end-to-end and judplay has real signal (beats
random comfortably, sign/masking verified), but round 0 loses hard to the
near-ceiling oracle player — as a searchless value net trained on 4K hands
should. The loss decomposes: the jud round-0 BIDDER over-bids worse than v0's
round 0 (93% offense vs 75%; the winner's-curse channel v0's loop closed,
plus root rows being only 2/56 of the training mix), and judplay's DEFENSE is
the largest single channel (net:wp converts 87% of contracts against it).
Per jud.md, defense is where information-set value concentrates — that is
where the loop has the most to win.

## 7. Unresolved / flags

1. **Overfitting is the binding constraint**: at lr 1e-3 the 512-wide MLP
   memorizes 200K rows within ~1 epoch (train CE → 0.3 while val CE rises);
   lr 3e-4 helps (best val 2.564 vs 2.637). Best-val checkpointing guards
   correctness, but real loop rounds want weight decay / dropout / a smaller
   trunk / more data. Play histories are near-unique fingerprints of a hand,
   so memorization pressure is structural.
2. **Root-row dilution**: bid-root rows are 2/56 of each hand's samples; root
   ECE is the worst slice (0.095 vs 0.02–0.05 mid-hand). If the loop's bidder
   stays over-optimistic, consider up-weighting step-0 rows — measure first.
3. **Round-0 winner's curse, again**: v0's P1 lesson replays exactly (over-bid
   at round 0, loop dissolves it). Registered expectation: the jud loop's
   offense share falls from 93% round-over-round as self-play data
   accumulates; if it does not, the root encoding needs another look.
4. **judplay is greedy depth-1**: argmax over post-move values, no search.
   The jud.md v1 ladder (value at leaves of shallow belief-state search) is
   the designed next step; this build supplies the leaf evaluator and its
   corpus machinery.
5. **Bid-level blindness at play time**: judplay maximizes E[pts] and never
   sees the contract level (canonical auction is level-blind by law). A
   defender that should risk points to set a 42-bid cannot express that.
   Deliberate v1 simplification; a marks-aware play utility over the pmf
   (exceedance at `contract_points(bid)`) is a one-line extension of JudPlay
   when wanted — the distribution survives to decision time.
6. **Old corpora**: pre-v1 snapshot files (no `plays`) still train
   `margin_net`, but `JudDataset` rejects them with a clear error —
   regenerate rather than mix.
