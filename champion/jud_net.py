"""jud_net (jud v1): the unified belief-conditioned value organ.

One net, two consumers. Info-state (own hand + canonical auction + play
history so far) → categorical distribution over the DECLARING team's realized
points (0..42) — the same target, loss, and readouts as `champion.margin_net`,
extended from bid-time to EVERY decision. A bid-time query is a play-time
query with an empty play history: ONE featurization, one net, no special
cases. Belief stays implicit, exactly as v0 made it for the auction: a net
conditioned on everyone's actions (bids AND plays) learns what the belief
head would say about the worlds those actions select.

Consumers
---------
* **Bidding** — `JudNet.pmake_table` has `MarginNet.pmake_table`'s exact
  signature, so `champion.value_bidder.ValueBidder` prices contracts through
  it unchanged (arena spec ``jud[:...]``). The hypothetical-completed-auction
  root is the empty-history info-state at seat = bidder.
* **Play** — `arena.jud_play.JudPlay` evaluates every legal move by querying
  this net on the info-state AFTER the move (still the mover's own POV) and
  picks argmax E[points] — sign-flipped for defenders, who minimize the
  declaring team's points. No world sampling, no oracle at runtime.

Featurization — x = [hand 63 | auction 28 | play 259] = 350 dims
----------------------------------------------------------------
* Hand: `champion.bid_net.featurize_hand` on the seat's ORIGINAL 7 dominoes —
  constant through the hand; which of them are gone is in the play block.
* Auction: `champion.margin_net.canonical_auction` reused verbatim — the
  level-blind, later-seat-masked encoding (bidder's own level := CANON_BID,
  seats after the bidder := pass) that kills selection leakage. The only
  difference from margin_net: `auction_feature_vector`'s POV is the DECISION
  seat, so relative-seat features (is_winner at seat r) tell the net where the
  declarer sits — the offense/defense channel. At seat = bidder this reduces
  byte-identically to margin_net's encoding (tested).
* Play history (the load-bearing block): a per-domino map + a small global
  summary, all POV-relative and derived from the (seat, domino) play list
  alone — so training rows (snapshot prefixes) and serving rows
  (ZebGameState) featurize byte-identically (tested).

    per domino d in 0..27, 9 dims at offset 9*d:
      [0:4] played-by relative seat one-hot  (r = (player - seat) % 4; zero if unplayed)
      [4:8] position-in-trick one-hot        (0 = led the trick)
      [8]   trick index / 6
    global tail, 7 dims:
      [252] declaring-team points so far / 42   (complete tricks, via resolve_trick)
      [253] defending-team points so far / 42
      [254] plays so far / 28
      [255:259] current-trick fill one-hot      (n_played % 4)

  Who played what, in what order (trick index × position), trick winners
  (position-0 seats), and the running score are all present; nothing else is.

Dataset
-------
Each snapshot hand expands to the exact query set the two consumers issue:
for every play step k (mover = plays[k][0]), the mover's info-state BEFORE
the move (step k — the decision) and AFTER it (step k+1 — the evaluation
JudPlay prices). Step 0's mover is the bidder, so the bid-time root is the
k = 0 decision row — no special case. Defense rows are ~half of everything,
which is where information-set value concentrates (jud v1 note). All rows of
a hand share its realized outcome: Monte Carlo targets, no bootstrapping —
at a 7-trick horizon TD machinery is pointless.

Model
-----
MLP: 350 → 512 → 512 → 43 logits (~470K params). The margin_net family
scaled up one notch for the 3.8x input; a transformer is unjustified when
the play block is already a fixed structured map rather than a sequence.
Start simple; the loss curve says when to reach for attention.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from arena.auction import MIN_BID, ONE_MARK, contract_points
from champion.bid_net import FEATURE_DIM as HAND_DIM  # 63
from champion.bid_net import featurize_hand
from champion.margin_net import (
    AUCTION_DIM,
    CANON_BID,
    N_POINTS,
    THRESHOLDS,
    _metrics,
    _reliability,
    _resolve_paths,
    canonical_auction,
    exceedance,
    mean_points,
    split_of,
)
from forge.bidding.schema import EVAL_DECLS
from forge.oracle.tables import resolve_trick
from forge.zeb.game import current_player
from forge.zeb.types import ZebGameState
from gus.model.auction import auction_feature_vector

# --------------------------------------------------------------------- #
#  Constants                                                              #
# --------------------------------------------------------------------- #

N_DOMINOES = 28
N_TRICKS = 7
PER_DOMINO = 9                                   # seat one-hot 4 + pos one-hot 4 + trick idx
GLOBAL_DIM = 7                                   # pts x2 + n_played + trick-fill one-hot 4
PLAY_DIM = N_DOMINOES * PER_DOMINO + GLOBAL_DIM  # 259
FEATURE_DIM = HAND_DIM + AUCTION_DIM + PLAY_DIM  # 350

# Optional per-legal-action auxiliary head (Lane B — dense E[Q] ranking).
N_ACTIONS = 7                        # hand slots 0..6 — forge e_q [7]/legal_actions slots
MARGIN_SCALE = float(ONE_MARK)       # 42; acting-seat point margin lives in [-42, +42]

_DEFAULT_MODEL = Path("champion/jud_net.pt")


# --------------------------------------------------------------------- #
#  Featurization                                                          #
# --------------------------------------------------------------------- #

def featurize_play(
    plays: Sequence[tuple[int, int]], seat: int, bidder: int, decl_id: int,
) -> Tensor:
    """[259] play-history feature from the (player, domino) list, POV ``seat``.

    Points-so-far are reconstructed from complete tricks with the engine's own
    `resolve_trick` — the emission contract (arena.test_snapshots) guarantees
    this replay matches the recorded outcome exactly.
    """
    feat = torch.zeros(PLAY_DIM, dtype=torch.float32)
    pts = [0, 0]  # (declaring team, defending team)
    bid_team = int(bidder) % 2
    for k, (player, domino) in enumerate(plays):
        base = PER_DOMINO * int(domino)
        feat[base + (int(player) - int(seat)) % 4] = 1.0
        feat[base + 4 + k % 4] = 1.0
        feat[base + 8] = (k // 4) / (N_TRICKS - 1)
        if k % 4 == 3:  # trick complete — resolve and score it
            trick = tuple(int(d) for _, d in plays[k - 3:k + 1])
            out = resolve_trick(trick[0], trick, int(decl_id))
            winner = (int(plays[k - 3][0]) + out.winner_offset) % 4
            pts[0 if winner % 2 == bid_team else 1] += out.points
    n = len(plays)
    g = N_DOMINOES * PER_DOMINO
    feat[g + 0] = pts[0] / 42.0
    feat[g + 1] = pts[1] / 42.0
    feat[g + 2] = n / float(N_DOMINOES)
    feat[g + 3 + n % 4] = 1.0
    return feat


def featurize(
    hand: Sequence[int],
    bids: Sequence[int],
    bidder: int,
    dealer: int,
    decl_id: int,
    plays: Sequence[tuple[int, int]] = (),
    seat: int | None = None,
) -> Tensor:
    """350-dim info-state: own hand (63) ⊕ canonical auction (28) ⊕ play (259).

    ``seat`` is the observing seat (whose hand, whose POV); ``None`` means the
    bidder — with the default empty ``plays`` that is exactly the bid-time
    hypothetical root, and the first 91 dims equal `margin_net.featurize`.
    """
    pov = int(bidder) if seat is None else int(seat)
    canon = canonical_auction(bids, bidder, dealer)
    return torch.cat([
        featurize_hand(tuple(hand)),
        auction_feature_vector(
            canon, bidder=int(bidder), bid_value=CANON_BID,
            decl_id=int(decl_id), current_player=pov,
        ),
        featurize_play(plays, pov, bidder, decl_id),
    ])


def featurize_snapshot(snap: Mapping, step: int = 0, seat: int | None = None) -> Tensor:
    """350-dim input from a snapshot row at play step ``step``, POV ``seat``
    (default: the bidder — step 0 then matches `margin_net.featurize_snapshot`
    on the shared 91 dims)."""
    pov = int(snap["bidder"]) if seat is None else int(seat)
    return featurize(
        tuple(int(t) for t in snap["hands"][pov]),
        [int(b) for b in snap["bids"]],
        int(snap["bidder"]),
        int(snap["dealer"]),
        int(snap["decl_id"]),
        [(int(p), int(d)) for p, d in snap["plays"][:step]],
        seat=pov,
    )


def featurize_state(state: ZebGameState, seat: int | None = None) -> Tensor:
    """350-dim input from a live engine state — the serving path.

    ``seat`` defaults to the player to act; pass it explicitly for post-move
    (or terminal) states, where the mover's POV is the one being priced. Only
    ``state.hands[seat]`` and public information enter the features.
    """
    pov = current_player(state) if seat is None else int(seat)
    return featurize(
        state.hands[pov],
        state.bid_state.bids,
        state.bidder,
        state.dealer,
        state.decl_id,
        state.play_history,
        seat=pov,
    )


# --------------------------------------------------------------------- #
#  Acting-seat orientation — the aux target's #1 foot-gun                 #
# --------------------------------------------------------------------- #
#
# JudNet's 43-bin head predicts the DECLARING team's realized points in
# [0, 42] (margin_net orientation). `mean_points` gives E[declaring pts];
# the declaring-team point margin is `2*E[pts] - 42` in [-42, +42]. A
# defender sits on the other side, so `arena.jud_play` flips the sign at
# consume time:  value_to_mover = sign * mean_points,  sign = +1 offense /
# -1 defense.
#
# The forge E[Q] tensor (`arena.lens_play` e_q; forge DecisionRecordGPU.e_q)
# is a DIFFERENT orientation: `query_model` already POV-corrects Q to the
# ACTING seat ("higher is better" — w42/lens_v1/lens.py), so e_q is the
# acting seat's OWN point margin in [-42, +42], one entry per legal hand
# slot 0..6 (the same slot indexing `arena.jud_play` argmaxes over). A
# defender's e_q is ALREADY sign-correct; there is nothing to flip.
#
# The aux head is therefore defined in ACTING-SEAT orientation to match e_q
# directly: it predicts the acting seat's margin / MARGIN_SCALE per slot,
# needs NO sign flip at consume time (unlike the main head), and its target
# is simply `e_q / MARGIN_SCALE`. `acting_seat_margin` is the explicit
# bridge that lets a test hand-check the two orientations agree.

def acting_seat_margin(decl_pts: int, seat: int, bidder: int) -> int:
    """Declaring-team realized points (0..42) → the ACTING seat's point margin
    in [-42, 42]. Offense (seat on the bidder's team) reads the declaring-team
    margin ``2*decl_pts - 42``; defense negates it. This is the orientation the
    aux head and forge e_q share; the main 43-bin head is in declaring-team
    orientation and `arena.jud_play` applies exactly this sign to it."""
    sign = 1 if int(seat) % 2 == int(bidder) % 2 else -1
    return sign * (2 * int(decl_pts) - (N_POINTS - 1))  # N_POINTS - 1 == 42


def aux_target_from_eq(e_q: Tensor) -> Tensor:
    """forge acting-seat E[Q] (margin in [-42, 42], per slot) → the aux head's
    target in [-1, 1]. e_q is ALREADY acting-seat oriented (query_model
    POV-corrects it), so the only transform is the scale ``1 / MARGIN_SCALE``."""
    return e_q.float() / MARGIN_SCALE


def child_value_target(v_act: float, seat: int, bidder: int) -> float:
    """Acting-seat per-move E[Q] margin ``v_act`` in [-42, 42] — the value of the
    CHILD state the mover reaches by playing the taken domino — → the DECLARING
    team's expected realized points in [0, 42], the space `mean_points` predicts
    and `arena.jud_play` argmaxes.

    This is the CHILD-side per-move arm (HC): where the aux head (HP) predicts
    all seven parent-side consequences in acting-seat orientation, the child
    target drops onto the MAIN head at the post-move state in DECLARING-team
    orientation, so no separate head is needed. It is the exact inverse of
    `acting_seat_margin`: ``sign = +1`` if the mover is on the bidding team else
    ``-1`` (a defender's e_q is already sign-correct), and
    ``E[pts_declaring] = (sign * v_act + 42) / 2``. At the terminal ply this
    reduces to the realized declaring points, agreeing with the row's CE label."""
    sign = 1.0 if int(seat) % 2 == int(bidder) % 2 else -1.0
    return (sign * float(v_act) + (N_POINTS - 1)) / 2.0  # N_POINTS - 1 == 42


# --------------------------------------------------------------------- #
#  Network                                                                #
# --------------------------------------------------------------------- #

class JudNet(nn.Module):
    """MLP: 350 → 512 → 512 → 43 logits (categorical over declaring-team points).

    ``aux_per_action`` (default False) adds ONE extra linear head (512 → 7) off
    the shared trunk that predicts the acting-seat per-slot value (Lane B dense
    E[Q] ranking auxiliary). With the flag OFF the module holds exactly the
    ``net.0/2/4`` parameters of every existing checkpoint — same keys, same
    shapes, byte-compatible load — and `forward` is byte-identical to
    ``self.net(x)``. The aux head is a training-time trunk-shaping signal only;
    the runtime consumers (pmake_table, JudPlay, JudSearch) still read the
    43-bin head via `forward`, so they are untouched whether the flag is on.
    """

    def __init__(self, in_dim: int = FEATURE_DIM, aux_per_action: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, N_POINTS),
        )
        self.aux_per_action = bool(aux_per_action)
        if self.aux_per_action:
            self.aux = nn.Linear(512, N_ACTIONS)  # acting-seat per-slot value

    def _trunk(self, x: Tensor) -> Tensor:
        """[B, 512] penultimate activation — the first four layers of ``net``
        (Linear→ReLU→Linear→ReLU), the input both heads read."""
        return self.net[:4](x)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, in_dim] → [B, 43] logits. Byte-identical to ``self.net(x)``."""
        return self.net[4](self._trunk(x))

    def forward_aux(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """(logits [B, 43], aux [B, 7]) sharing one trunk pass — the training
        path when ``aux_per_action`` is on. aux is in acting-seat / MARGIN_SCALE
        orientation (see `aux_target_from_eq`)."""
        if not self.aux_per_action:
            raise RuntimeError("forward_aux requires aux_per_action=True")
        h = self._trunk(x)
        return self.net[4](h), self.aux(h)

    def pmake_table(
        self, hand: Sequence[int], bids: Sequence[int], bidder: int, dealer: int
    ) -> dict[int, dict[int, float]]:
        """{decl: {threshold: P(pts ≥ threshold)}} at the empty-history root —
        `MarginNet.pmake_table`'s exact signature, so the ValueBidder consumes
        this net unchanged. The bid-time query IS the play-time query with no
        plays: one featurization, no special case."""
        decls = list(EVAL_DECLS)
        feats = torch.stack([
            featurize(hand, bids, bidder, dealer, d) for d in decls
        ])  # [9, 350]
        was_training = self.training
        self.eval()
        with torch.no_grad():
            exc = exceedance(self(feats))  # [9, 43]
        if was_training:
            self.train()
        return {
            d: {t: float(exc[i, contract_points(t)]) for t in THRESHOLDS}
            for i, d in enumerate(decls)
        }


def load_jud_net(model_path: str | Path = _DEFAULT_MODEL, device: str = "cpu") -> JudNet:
    """Load a trained JudNet checkpoint (mirrors ``value_bidder.load_margin_net``)."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = JudNet(
        in_dim=ckpt.get("feature_dim", FEATURE_DIM),
        aux_per_action=ckpt.get("aux_per_action", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# --------------------------------------------------------------------- #
#  Dataset                                                                #
# --------------------------------------------------------------------- #

def hand_samples(snap: Mapping) -> Iterable[tuple[int, int]]:
    """(step, pov_seat) sample coordinates for one snapshot hand — the exact
    query set the consumers issue: for each play step k the mover's info-state
    before the move (a decision, incl. the k=0 bid-time root: step 0's mover
    is the bidder) and after it (the child state JudPlay prices)."""
    for k, (player, _) in enumerate(snap["plays"]):
        yield k, int(player)
        yield k + 1, int(player)


def _deal_key(hands, decl_id, bidder, bids) -> tuple:
    """Deal identity shared by a snapshot row and its E[Q] `GameRecordGPU`.

    The bridge (`forge.cli.generate_eq_from_snapshots`) runs the oracle on the
    snapshot's seat-ordered deal and stamps back the same ``bids``/``bidder``,
    so the four fields reconstruct identically on both sides. Note two
    paired-half hands can share a deal_key — the (deal, step, seat) join below
    tolerates that (identical deals produce identical oracle labels)."""
    h = tuple(tuple(int(t) for t in seat) for seat in hands)
    b = tuple(int(x) for x in bids) if bids is not None else ()
    return (h, int(decl_id), int(bidder), b)


def load_aux_table(
    paths: str | Path | Iterable[str | Path],
) -> dict[tuple, tuple[Tensor, Tensor]]:
    """Bridge E[Q] corpus/corpora (`{"results": [GameRecordGPU, ...]}`) → aux-label
    table keyed by ``(deal_key, decision_index, acting_seat)`` →
    (e_q [7], legal_mask [7]).

    ``paths`` is a single ``.pt`` path, a glob string, or an iterable of paths
    (resolved by `margin_net._resolve_paths`, the same expander the corpus arg
    uses); the per-file tables are MERGED into one. The 3-corpus HC/HP round
    labels each self-play corpus into its own ``eq_*.pt``, so the join needs
    their union — distinct deals never collide, and identical deals (paired
    halves) produce identical labels, so a key collision is a harmless overwrite.

    ``decision_index`` is the 0-based ply the oracle produced the record at
    (``enumerate(record.decisions)``); ``acting_seat`` is ``decision.player``.
    Padded/illegal e_q slots (``-inf`` per DecisionRecordGPU) are zeroed under
    the mask so they can never poison the masked MSE. e_q stays in forge
    acting-seat orientation (margin, [-42, 42]); the scale to [-1, 1] happens at
    join time via `aux_target_from_eq` (parent-side aux head), and the child arm
    reads the taken slot and converts to declaring points via `child_value_target`.

    ALIGNMENT (the load-bearing caveat): the bridge replays the ORACLE's own
    greedy line from the initial deal, so ``decision_index`` k lands on the
    arena's play step k only where the two lines coincide. The ``acting_seat``
    component of the key is the guard — a row whose mover disagrees with the
    oracle's ply-k mover simply misses and stays unsupervised. Measure join
    coverage before trusting arm HP/HC; for guaranteed alignment, generate the
    E[Q] corpus with ``generate_eq_from_snapshots --teacher-forced`` (decision k
    is then exactly arena play step k, 100% decision-row coverage).
    """
    files = _resolve_paths(paths)
    if not files:
        raise FileNotFoundError(f"no aux-label .pt files matched {paths!r}")
    table: dict[tuple, tuple[Tensor, Tensor]] = {}
    for pt_path in files:
        blob = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        records = blob["results"] if isinstance(blob, dict) else blob
        for rec in records:
            dkey = _deal_key(rec.hands, rec.decl_id, rec.bidder, rec.bids)
            for k, dec in enumerate(rec.decisions):
                mask = dec.legal_mask.detach().cpu().float()
                e_q = dec.e_q.detach().cpu().float()
                e_q = torch.where(mask > 0, e_q, torch.zeros_like(e_q))  # kill -inf pads
                table[(dkey, int(k), int(dec.player))] = (e_q, mask)
    return table


class JudDataset(torch.utils.data.Dataset):
    """Snapshot-JSON corpus → (350-dim x, realized-points y) per-decision samples.

    Same corpus files as `MarginDataset` (``arena.cli --emit-snapshots``), but
    each hand expands via `hand_samples` into ~56 info-state rows, every one
    labeled with the hand's realized ``bidder_team_pts`` (one label per hand —
    the Monte Carlo target). Rows are exact-deduped: paired halves replay
    identical seeds, so identical auctions collapse at step 0, and a
    deterministic A==B self-play collapses entirely.

    ``aux_table`` (optional, from `load_aux_table`) attaches a per-legal-action
    E[Q] target + legal mask to each DECISION row — the row whose ``(step, pov)``
    is a real decision point (``pov == plays[step][0]``), i.e. the "before-move"
    half of each `hand_samples` pair. It is joined by ``(deal_key, step, pov)``.
    Rows that are child/eval states, or decision rows with no matching E[Q]
    label, get an all-zero mask so they contribute nothing to the aux loss.

    ``child_values`` (arm HC, requires ``aux_table``) additionally attaches a
    scalar CHILD target to each post-move (eval) row — the ``(step, pov)`` whose
    ``pov == plays[step-1][0]``, i.e. the "after-move" half. The target is the
    parent decision's e_q at the TAKEN slot converted to declaring-team points
    (`child_value_target`), which is exactly what `arena.jud_play` argmaxes over
    child states — so it supervises the MAIN head with no extra head. The k=0
    root and pure decision rows carry weight 0. This is orthogonal to the aux
    head: the two can coexist, but the pure HC arm builds the net with
    ``aux_per_action=False``.

    With ``aux_table=None`` (and ``child_values=False``) the dataset is
    byte-for-byte its pre-Lane-B self — `__getitem__`/`tensors` return the same
    (x, y) pairs and the row set is never perturbed by either label join.
    """

    def __init__(
        self,
        paths: str | Path | Iterable[str | Path],
        split: str = "all",
        aux_table: dict[tuple, tuple[Tensor, Tensor]] | None = None,
        child_values: bool = False,
    ) -> None:
        self.split = split
        self.aux = aux_table is not None
        self.child = bool(child_values)
        if self.child and aux_table is None:
            raise ValueError(
                "child_values=True requires an aux_table — the E[Q] table supplies "
                "the per-move child targets (pass load_aux_table(...) )."
            )
        self.samples: list[tuple[Tensor, Tensor]] = []
        self.keys: list[tuple] = []
        self.aux_targets: list[Tensor] = []   # [7] each, in [-1, 1] (0 where unsupervised)
        self.aux_masks: list[Tensor] = []      # [7] each, 1.0 on supervised legal slots
        self.child_targets: list[Tensor] = []  # [] each, declaring pts [0,42] (0 unsupervised)
        self.child_weights: list[Tensor] = []  # [] each, 1.0 on supervised child rows
        seen: set[tuple] = set()

        # The taken-slot lookup for child values reuses the canonical
        # snapshot→slot map (single source of truth with the E[Q] bridge). Import
        # lazily so the serving path (arena.jud_play → jud_net) never pays for the
        # forge.eq.generate import; child_values is a training-only join.
        if self.child:
            from forge.cli.generate_eq_from_snapshots import forced_actions_from_snapshot

        for f in _resolve_paths(paths):
            payload = json.loads(Path(f).read_text())
            snaps = payload["snapshots"] if isinstance(payload, dict) else payload
            for snap in snaps:
                if "plays" not in snap:
                    raise ValueError(
                        f"{f}: snapshot rows lack 'plays' — regenerate the corpus "
                        "with a post-jud-v1 arena (--emit-snapshots)."
                    )
                if split != "all" and split_of(snap["seed"], snap["hand_idx"]) != split:
                    continue
                base = (
                    int(snap["seed"]),
                    int(snap["hand_idx"]),
                    tuple(int(b) for b in snap["bids"]),
                    int(snap["bidder"]),
                    int(snap["decl_id"]),
                )
                y = torch.tensor(int(snap["bidder_team_pts"]), dtype=torch.long)
                plays = snap["plays"]
                dkey = (
                    _deal_key(snap["hands"], snap["decl_id"], snap["bidder"], snap["bids"])
                    if (self.aux or self.child) else None
                )
                forced = forced_actions_from_snapshot(snap) if self.child else None
                bidder = int(snap["bidder"])
                for step, pov in hand_samples(snap):
                    key = base + (
                        step, pov,
                        tuple((int(p), int(d)) for p, d in plays[:step]),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    self.samples.append((featurize_snapshot(snap, step, pov), y))
                    self.keys.append(key)
                    if self.aux:
                        at, am = self._aux_for(aux_table, dkey, step, pov, plays)
                        self.aux_targets.append(at)
                        self.aux_masks.append(am)
                    if self.child:
                        ct, cw = self._child_value_for(
                            aux_table, dkey, step, pov, plays, forced, bidder
                        )
                        self.child_targets.append(ct)
                        self.child_weights.append(cw)

    @staticmethod
    def _aux_for(aux_table, dkey, step, pov, plays) -> tuple[Tensor, Tensor]:
        """(target [7] in [-1,1], mask [7]) for one row. Supervised iff the row
        is the decision point ``pov == plays[step][0]`` and the E[Q] table holds
        a matching (deal, step, seat) label; otherwise an all-zero mask."""
        is_decision = step < len(plays) and int(pov) == int(plays[step][0])
        hit = aux_table.get((dkey, int(step), int(pov))) if is_decision else None
        if hit is None:
            return torch.zeros(N_ACTIONS), torch.zeros(N_ACTIONS)
        e_q, mask = hit
        return aux_target_from_eq(e_q), mask

    @staticmethod
    def _child_value_for(aux_table, dkey, step, pov, plays, forced, bidder):
        """(target scalar in [0,42], weight scalar in {0,1}) for the CHILD row
        ``(step, pov)`` — the post-move state `arena.jud_play` prices. Supervised
        iff the row is the child of ``pov``'s decision at ``step-1``
        (``pov == plays[step-1][0]``) AND the E[Q] table holds that parent
        decision's label. The target is the parent's e_q at the TAKEN slot
        (``forced[step-1]``, the domino→slot map shared with the bridge),
        converted to declaring-team E[pts] by `child_value_target`. Weight 0
        (target 0) otherwise, so the k=0 root and pure decision rows never carry
        a child value."""
        if step < 1:
            return torch.zeros(()), torch.zeros(())
        k = step - 1
        if k >= len(plays) or int(pov) != int(plays[k][0]):
            return torch.zeros(()), torch.zeros(())
        hit = aux_table.get((dkey, int(k), int(pov)))
        if hit is None:
            return torch.zeros(()), torch.zeros(())
        e_q, mask = hit
        slot = int(forced[k])
        if not (0 <= slot < N_ACTIONS) or float(mask[slot]) <= 0.0:
            return torch.zeros(()), torch.zeros(())
        target = child_value_target(float(e_q[slot]), int(pov), int(bidder))
        return torch.tensor(float(target)), torch.ones(())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.child:
            x, y = self.samples[idx]
            return (x, y, self.aux_targets[idx], self.aux_masks[idx],
                    self.child_targets[idx], self.child_weights[idx])
        if self.aux:
            x, y = self.samples[idx]
            return x, y, self.aux_targets[idx], self.aux_masks[idx]
        return self.samples[idx]

    def tensors(self) -> tuple[Tensor, Tensor]:
        """Stacked (X [N, 350], Y [N]) — for whole-split metric passes."""
        if not self.samples:
            return torch.empty(0, FEATURE_DIM), torch.empty(0, dtype=torch.long)
        xs = torch.stack([x for x, _ in self.samples])
        ys = torch.stack([y for _, y in self.samples])
        return xs, ys

    def tensors_aux(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Stacked (X [N,350], Y [N], AUX_T [N,7], AUX_M [N,7]). Requires an
        aux_table; AUX_M rows are all-zero where the row was not aux-supervised."""
        if not self.aux:
            raise RuntimeError("tensors_aux requires the dataset to carry an aux_table")
        xs, ys = self.tensors()
        if not self.samples:
            return xs, ys, torch.empty(0, N_ACTIONS), torch.empty(0, N_ACTIONS)
        return xs, ys, torch.stack(self.aux_targets), torch.stack(self.aux_masks)

    def aux_coverage(self) -> float:
        """Fraction of rows carrying at least one supervised aux slot — the
        join-alignment health check (see `load_aux_table`)."""
        if not self.aux or not self.aux_masks:
            return 0.0
        return float(sum(1 for m in self.aux_masks if float(m.sum()) > 0) / len(self.aux_masks))

    def tensors_child(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Stacked (X [N,350], Y [N], CHILD_T [N], CHILD_W [N]). Requires
        child_values; CHILD_W rows are 0 where the row is not a supervised child
        state (the decision half, and any parent-label miss)."""
        if not self.child:
            raise RuntimeError("tensors_child requires the dataset to be built with child_values=True")
        xs, ys = self.tensors()
        if not self.samples:
            return xs, ys, torch.empty(0), torch.empty(0)
        return xs, ys, torch.stack(self.child_targets), torch.stack(self.child_weights)

    def child_coverage(self) -> float:
        """Fraction of rows carrying a supervised child value — the child-join
        health check. Sits near 0.5 by design: only the post-move (child) half of
        each `hand_samples` pair can carry a child target (the decision half and
        the k=0 root never do), so 100% teacher-forced coverage reads ~0.5 —
        slightly above when a winner-leads row is deduped onto its decision twin."""
        if not self.child or not self.child_weights:
            return 0.0
        return float(sum(1 for w in self.child_weights if float(w) > 0) / len(self.child_weights))


# --------------------------------------------------------------------- #
#  Training                                                               #
# --------------------------------------------------------------------- #

def _forward_all(model: JudNet, ds: JudDataset, device: str) -> tuple[Tensor, Tensor]:
    xs, ys = ds.tensors()
    if xs.numel() == 0:
        return xs, ys
    model.eval()
    with torch.no_grad():
        logits = model(xs.to(device)).cpu()
    return logits, ys


def masked_action_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Per-legal-action MSE, averaged over supervised (mask==1) slots only.

    ``pred``/``target``/``mask`` are [B, 7]. Both pred and target live in the
    acting-seat / MARGIN_SCALE space ([-1, 1]); the normalization is what makes
    a default ``aux_lambda=1.0`` weigh this term comparably to the CE term (both
    O(1)) — CE is ~2–3 nats, and a [-1,1] MSE is O(0.1–1). ``mask.sum()`` is
    clamped so an all-unsupervised batch yields a finite 0 gradient."""
    se = (pred - target) ** 2 * mask
    return se.sum() / mask.sum().clamp(min=1.0)


def masked_value_mse(mean_pts: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    """Per-row MSE of the MAIN head's E[declaring pts] against the child-state
    per-move oracle value (arm HC), over supervised rows (weight==1) only.

    ``mean_pts`` is `mean_points(logits)` and ``target`` the child value, both in
    declaring-team points [0, 42]; both are normalized by MARGIN_SCALE (42, the
    max points) before squaring — the same scale discipline as
    `masked_action_mse`, so a default ``child_lambda=1.0`` weighs this term
    comparably to the CE (~2–3 nats; a /42 MSE is O(0.1–1)). ``weight.sum()`` is
    clamped so an all-unsupervised batch yields a finite 0 gradient. This is
    exactly the quantity `arena.jud_play` argmaxes over post-move child states —
    consumer-aligned supervision straight onto the head the player reads."""
    se = ((mean_pts - target) / MARGIN_SCALE) ** 2 * weight
    return se.sum() / weight.sum().clamp(min=1.0)


def train(
    corpus: str | Path | Iterable[str | Path],
    out_model: Path = _DEFAULT_MODEL,
    epochs: int = 60,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 8,
    device: str = "cpu",
    aux_labels: str | Path | dict | None = None,
    aux_lambda: float = 1.0,
    weight_decay: float = 0.0,
    seed: int | None = None,
    child_values: bool = False,
    child_lambda: float = 1.0,
) -> dict:
    """Train JudNet on the snapshot corpus; save best-val weights, return metrics.

    Three Lane B arms select through the E[Q] table + two flags:

    * **H** — ``aux_labels=None`` — the pre-Lane-B path, unchanged.
    * **HP** — ``aux_labels=<table>`` alone: a per-legal-action aux head is
      added and the loss becomes ``CE + aux_lambda * masked_action_mse`` (aux
      target/pred scale-normalized to [-1, 1]).
    * **HC** — ``aux_labels=<table>`` + ``child_values=True``: the SAME table
      supplies per-move CHILD targets, but NO head is added — the main head's
      ``mean_points`` on each post-move row is regressed toward the child value
      (`child_value_target`), the space `arena.jud_play` argmaxes. The loss is
      ``CE + child_lambda * masked_value_mse``. The aux head is suppressed here
      so HC stays the pure child arm (`aux_per_action=False`).

    ``aux_labels`` is a `load_aux_table` dict, a bridge E[Q] ``.pt`` path, or a
    glob/iterable of them (merged). Early stopping reads the main-head val CE, so
    all arms are graded on the same yardstick.
    """
    aux_table = (
        load_aux_table(aux_labels) if isinstance(aux_labels, (str, Path)) else aux_labels
    )
    child_on = bool(child_values)
    if child_on and aux_table is None:
        raise ValueError(
            "child_values=True requires aux_labels — its E[Q] table supplies the "
            "per-move child targets."
        )
    # The parent-side aux HEAD (arm HP) is added only when we are NOT running the
    # child arm: HC reads the same table for targets but supervises the MAIN head
    # on child states and adds no head, so the runtime consumers are byte-identical.
    aux_head_on = aux_table is not None and not child_on
    train_ds = JudDataset(corpus, split="train", aux_table=aux_table, child_values=child_on)
    val_ds = JudDataset(corpus, split="val", aux_table=aux_table, child_values=child_on)
    test_ds = JudDataset(corpus, split="test")
    if not len(train_ds):
        raise FileNotFoundError(
            f"No training samples from {corpus!r} — generate arena snapshots first."
        )
    print(
        f"Dataset: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test "
        f"decision rows (deduped, 90/5/5 by deal hash)",
        flush=True,
    )
    if aux_head_on:
        print(
            f"Aux head ON (lambda={aux_lambda}): train aux coverage "
            f"{train_ds.aux_coverage():.1%} of rows supervised",
            flush=True,
        )
    if child_on:
        print(
            f"Child values ON (lambda={child_lambda}): train child coverage "
            f"{train_ds.child_coverage():.1%} of rows supervised "
            f"(~0.5 is full teacher-forced coverage)",
            flush=True,
        )

    # Seed so the two Lane B arms (H vs HP) match everywhere but the aux path.
    # `JudNet` builds `self.net` before the optional `self.aux`, so seeding
    # here gives BOTH arms byte-identical trunk init (HP's extra aux draws come
    # after). A dedicated loader generator keeps the shuffle order independent
    # of those extra draws, so batch order matches too. weight_decay is the
    # named regularizer for the overfitting binding constraint (jud v1 §7).
    if seed is not None:
        torch.manual_seed(seed)
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=gen,
    )
    model = JudNet(aux_per_action=aux_head_on).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    val_logits, val_ys = _forward_all(model, val_ds, device)
    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            if child_on:
                # Child dataset yields the 6-tuple (x, y, aux_t, aux_m, child_t,
                # child_w); the aux slots ride along unused (no aux head in HC).
                x_b, y_b, _at_b, _am_b, cv_b, cw_b = batch
                logits = model(x_b.to(device))
                loss = loss_fn(logits, y_b.to(device)) + child_lambda * masked_value_mse(
                    mean_points(logits), cv_b.to(device), cw_b.to(device)
                )
            elif aux_head_on:
                x_b, y_b, at_b, am_b = batch
                logits, aux_pred = model.forward_aux(x_b.to(device))
                loss = loss_fn(logits, y_b.to(device)) + aux_lambda * masked_action_mse(
                    aux_pred, at_b.to(device), am_b.to(device)
                )
            else:
                x_b, y_b = batch
                logits = model(x_b.to(device))
                loss = loss_fn(logits, y_b.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x_b)
        train_ce = total / len(train_ds)

        val_logits, _ = _forward_all(model, val_ds, device)
        vm = _metrics(val_logits, val_ys)
        print(
            f"  epoch {epoch:3d}/{epochs}  train_ce={train_ce:.4f}  "
            f"val_ce={vm['ce']:.4f}  val_mae={vm['mae_mean_pts']:.3f}  "
            f"val_ece30={vm['ece_p30']:.4f}",
            flush=True,
        )

        # Early-stop on val CE (falls through cleanly if there is no val split).
        if vm["ce"] == vm["ce"] and vm["ce"] < best_val - 1e-5:
            best_val = vm["ce"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if len(val_ds) and bad >= patience:
                print(f"  early stop at epoch {epoch} (no val improvement in {patience})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "feature_dim": FEATURE_DIM,
         "aux_per_action": aux_head_on},
        out_model,
    )
    print(f"Saved model → {out_model}", flush=True)

    test_logits, test_ys = _forward_all(model, test_ds, device)
    tm = _metrics(test_logits, test_ys)
    metrics = {
        "best_val_ce": best_val if best_val != float("inf") else float("nan"),
        "test_ce": tm["ce"],
        "test_mae_mean_pts": tm["mae_mean_pts"],
        "test_ece_p30": tm["ece_p30"],
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "test_rows": len(test_ds),
        "epochs_run": epoch,
        "feature_dim": FEATURE_DIM,
    }
    print(
        f"\nBest val CE: {metrics['best_val_ce']:.4f}   "
        f"Test CE: {tm['ce']:.4f}   MAE(mean-pts): {tm['mae_mean_pts']:.3f}   "
        f"ECE P(pts≥30): {tm['ece_p30']:.4f}",
        flush=True,
    )
    return metrics


# --------------------------------------------------------------------- #
#  Evaluation                                                             #
# --------------------------------------------------------------------- #

def evaluate(
    corpus: str | Path | Iterable[str | Path],
    model_path: Path = _DEFAULT_MODEL,
    out_json: Path = Path("scratch/jud-v1/jud_net_eval.json"),
    device: str = "cpu",
) -> dict:
    """Test-split report: overall + per-phase (by trick) reliability.

    Where `margin_net.evaluate` reads one root exceedance curve, the play-time
    head should sharpen as evidence accrues — so the report also slices MAE and
    ECE by trick index, the calibration-vs-depth curve jud v1 cares about.
    """
    test_ds = JudDataset(corpus, split="test")
    if not len(test_ds):
        raise FileNotFoundError(f"No test samples from {corpus!r}.")
    model = load_jud_net(model_path, device)

    logits, ys = _forward_all(model, test_ds, device)
    exc = exceedance(logits).cpu().numpy()          # [N, 43]
    y_np = ys.cpu().numpy()

    rel = _reliability(exc[:, MIN_BID], (y_np >= MIN_BID).astype(float))
    thr = list(THRESHOLDS)
    result = {
        "n_test": len(y_np),
        "reliability_p30": rel,
        "ece_p30": rel["ece"],
        "exceedance_thresholds": thr,
        "predicted_exceedance": [float(exc[:, t].mean()) for t in thr],
        "empirical_exceedance": [float((y_np >= t).mean()) for t in thr],
        "mae_mean_pts": float((mean_points(logits) - ys.float()).abs().mean().item()),
        "test_ce": float(nn.functional.cross_entropy(logits, ys).item()),
        "model_path": str(model_path),
    }

    # Per-trick slices: sample k's trick index is n_played // 4, read back from
    # the global n_played feature (dim g+2) — no need to re-walk the corpus.
    xs, _ = test_ds.tensors()
    n_played = (xs[:, HAND_DIM + AUCTION_DIM + N_DOMINOES * PER_DOMINO + 2]
                * N_DOMINOES).round().long().numpy()
    by_trick = {}
    for trick in range(N_TRICKS + 1):  # 7 = terminal rows (all 28 played)
        m = (n_played // 4 == trick) if trick < N_TRICKS else (n_played == 28)
        if not m.any():
            continue
        sl = _metrics(logits[m], ys[m])
        by_trick[str(trick)] = {
            "n": int(m.sum()), "ce": sl["ce"],
            "mae_mean_pts": sl["mae_mean_pts"], "ece_p30": sl["ece_p30"],
        }
    result["by_trick"] = by_trick

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_json}", flush=True)
    print(
        "  trick   n      ce    mae   ece30\n" + "\n".join(
            f"  {t:>5} {v['n']:>5}  {v['ce']:.3f}  {v['mae_mean_pts']:5.2f}  "
            f"{v['ece_p30']:.3f}" for t, v in by_trick.items()
        ),
        flush=True,
    )
    return result


# --------------------------------------------------------------------- #
#  CLI                                                                    #
# --------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="JudNet — unified realized-value organ (jud v1)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tp = sub.add_parser("train", help="train on the snapshot corpus")
    tp.add_argument("--corpus", required=True,
                    help="glob of snapshot JSON files, e.g. 'scratch/jud-v1/corpus/*.json'")
    tp.add_argument("--out-model", type=Path, default=_DEFAULT_MODEL)
    tp.add_argument("--epochs", type=int, default=60)
    tp.add_argument("--batch-size", type=int, default=512)
    tp.add_argument("--lr", type=float, default=1e-3)
    tp.add_argument("--patience", type=int, default=8)
    tp.add_argument("--device", type=str, default="cpu")
    tp.add_argument("--aux-labels", type=str, default=None,
                    help="Lane B: bridge E[Q] .pt (single path, glob, or space of "
                         "them) — the per-action aux head (arm HP) with --aux-labels "
                         "alone, or the child-value TABLE (arm HC) with --child-values. "
                         "Omit for arm H (no E[Q] signal).")
    tp.add_argument("--aux-lambda", type=float, default=1.0,
                    help="weight on the masked-MSE aux term (default 1.0, scale-normalized)")
    tp.add_argument("--child-values", action="store_true",
                    help="Lane B arm HC: supervise the MAIN head's mean_points on "
                         "post-move CHILD rows with per-move E[Q] child values "
                         "(declaring-team points), the space arena.jud_play argmaxes. "
                         "Requires --aux-labels for the table; adds no head.")
    tp.add_argument("--child-lambda", type=float, default=1.0,
                    help="weight on the child-value MSE term (default 1.0, scale-normalized)")
    tp.add_argument("--weight-decay", type=float, default=0.0,
                    help="Adam weight decay (Lane B: set identically across both arms)")
    tp.add_argument("--seed", type=int, default=None,
                    help="seed trunk init + batch order (Lane B: identical across both arms)")

    ep = sub.add_parser("eval", help="test-split reliability report (overall + by trick)")
    ep.add_argument("--corpus", required=True, help="glob of snapshot JSON files")
    ep.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    ep.add_argument("--out-json", type=Path, default=Path("scratch/jud-v1/jud_net_eval.json"))
    ep.add_argument("--device", type=str, default="cpu")

    args = ap.parse_args()
    if args.cmd == "train":
        train(
            corpus=args.corpus, out_model=args.out_model, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, patience=args.patience,
            device=args.device, aux_labels=args.aux_labels, aux_lambda=args.aux_lambda,
            weight_decay=args.weight_decay, seed=args.seed,
            child_values=args.child_values, child_lambda=args.child_lambda,
        )
    else:
        evaluate(
            corpus=args.corpus, model_path=args.model,
            out_json=args.out_json, device=args.device,
        )
