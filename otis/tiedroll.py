"""Otis tied-strategy rollout — pricing the junk-retention economy (issue #49, P7).

This module builds the *information-honest evaluator* the count-fate ledger
(`wiki/topics/count-fate-ledger.md`) says no current tool performs: a rollout
where **every seat plays a deterministic function of its OWN information state**
(own hand + public history only). Such a rollout is **tied-by-construction** —
across sampled worlds the actor's actions are identical exactly until its
observations diverge, so guards (junk kept to deny rows) and walkers (junk that
catches sloughed count late) are priced at their real, one-strategy-across-worlds
value. No explicit divergence bookkeeping is needed, only batching: two worlds
with an identical actor info-state produce identical actor tokens, hence an
identical argmax.

The clairvoyant leg is the fusion-gap baseline: the SAME worlds, every seat
playing the oracle's full-deal argmax (`forge.eq.oracle.Stage1Oracle` via
`forge.eq.generate.model.query_model`). Strategy fusion zeroes guard/walker value
there, because the clairvoyant never buys insurance nor lottery tickets. The
retention price is the tied outcome difference between two commitments on common
random worlds; the **fusion gap** is (tied delta − clairvoyant delta).

Both legs share one driver. Each world is reconstructed as a full 4×7 INITIAL
deal (P's real hand + the opponents' played-in-the-prefix dominoes ∪ the corpus
`world_hands` remaining), then `forge.eq.game_tensor.GameStateTensor` — the exact
engine — replays the recorded prefix and rolls the tail. Completed 28-play
trajectories are refereed by `otis.fates.parse_game_fates` (the P1 identity is
asserted inside the parser), yielding per-team points and the count-tile fates.

Info-honest π_me: the gus student's `pi_me` head reads only `state_emb`
(`cls_h + voids_encoder(voids)`), never the world assignment — so querying it with
a ZERO world (as `gus/eval/lamir1.py:direct_decision` does) makes it a pure
function of the actor's info-state. That is the tied policy. The clairvoyant
policy is the oracle's per-world Q argmax over the full deal.

CPU by default; pass ``--device mps`` for the probe. otis does not own the GPU —
build and smoke on CPU only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from forge.oracle.declarations import has_trump_power
from forge.oracle.tables import (
    can_follow,
    is_in_called_suit,
    led_suit_for_lead_domino,
)
from otis.fates import COUNT_TILE_IDS, PLAYED_MODES, NeutralGame, parse_game_fates

# ---------------------------------------------------------------------------
# A decision stub the gus tokenizer understands (.player / .action_taken).
# ---------------------------------------------------------------------------


@dataclass
class PlayStub:
    """Minimal (player, slot) play record for the gus tokenizer + fate replay."""

    player: int
    action_taken: int  # slot index into that player's world-specific initial hand


# ---------------------------------------------------------------------------
# Commitments (the root lever)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Commitment:
    """A root commitment applied to the ACTOR (P) only.

    Exactly one flavour is active:

    * ``root_domino`` — force the actor to play this domino at the root ply
      (a fixed root action), then roll π_me/oracle for the rest.
    * ``keep_id`` / ``release_id`` — a *retention constraint*: throughout the
      rollout, whenever the actor is void and about to slough a non-trump tile,
      it dumps ``release_id`` first (if still held and legal) and never sloughs
      ``keep_id`` unless it is the only legal discard. This is the junk-retention
      economy with two signs (guard = keep, walker = release).

    ``label`` names the commitment for receipts. ``NONE`` is the unconstrained
    rollout (pure π_me / pure oracle).
    """

    label: str = "none"
    root_domino: int | None = None
    keep_id: int | None = None
    release_id: int | None = None


NONE = Commitment(label="none")


# ---------------------------------------------------------------------------
# Suit-algebra helpers (reuse the forge tables — no reimplemented rules)
# ---------------------------------------------------------------------------


def _is_trump(domino_id: int, decl_id: int) -> bool:
    """True if the domino is a power trump under this declaration."""
    return has_trump_power(decl_id) and is_in_called_suit(domino_id, decl_id)


def _actor_is_void(state, m: int, P: int, decl_id: int) -> bool:
    """Is actor P void in the current trick's led suit in world m (i.e. discarding)?

    False when leading (no led suit) or when P can still follow suit.
    """
    trick_len = int((state.trick_plays[m] >= 0).sum().item())
    if trick_len == 0:
        return False  # leading — not a slough
    lead_id = int(state.trick_plays[m, 0].item())
    led_suit = led_suit_for_lead_domino(lead_id, decl_id)
    for i in range(7):
        d = int(state.hands[m, P, i].item())
        if d >= 0 and can_follow(d, led_suit, decl_id):
            return False  # can follow — not void
    return True


# ---------------------------------------------------------------------------
# Policies (score functions): [M,7] score for each world's CURRENT player,
# indexed by that player's GameStateTensor hand slot. Higher = preferred.
# ---------------------------------------------------------------------------


class TiedPolicy:
    """Info-honest π_me: the gus student queried with a ZERO world assignment.

    The pi_me head depends only on ``state_emb`` (cls token + voids), so a zero
    world makes the argmax a pure function of the actor's info-state — the
    tied-by-construction property.
    """

    def __init__(self, student, is_voids: bool):
        self.student = student
        self.is_voids = is_voids
        self.device = next(student.parameters()).device

    def scores(self, state, deals, dec_lists, cp, decl_id) -> torch.Tensor:
        from gus.eval.lamir1 import _build_tokens_voids

        M = len(deals)
        toks, msks, vds = [], [], []
        for m in range(M):
            c = int(cp[m].item())
            t, am, v = _build_tokens_voids(
                deals[m], decl_id, dec_lists[m], len(dec_lists[m]), [], c
            )
            toks.append(t)
            msks.append(am)
            vds.append(v)
        # rollout stepping stays on CPU; only the model forward runs on self.device
        tb = torch.stack(toks).to(self.device)
        mb = torch.stack(msks).to(self.device)
        zw = torch.zeros(M, 28, 3, device=self.device)
        with torch.no_grad():
            if self.is_voids:
                out = self.student(tb, mb, zw, torch.stack(vds).to(self.device))
            else:
                out = self.student(tb, mb, zw)
        return out["pi_me_logits"].float().cpu()


class ClairvoyantPolicy:
    """Full-deal oracle: Stage-1 Q per action, tokenized from the live state.

    Each world is a full deal, so the oracle plays clairvoyantly — the fusion
    baseline against which the tied evaluator's guard/walker value is measured.
    """

    def __init__(self, oracle_model, tokenizer, device: str):
        self.model = oracle_model
        self.tokenizer = tokenizer  # CPU tokenizer (matches the CPU rollout state)
        self.device = device        # oracle model device (cpu or mps for the probe)

    def scores(self, state, deals, dec_lists, cp, decl_id) -> torch.Tensor:
        from forge.eq.generate.model import query_model
        from forge.eq.generate.tokenization import tokenize_batched

        # states.hands are the current REMAINING hands (played slots = -1);
        # build_remaining_bitmasks reads the played_mask, so passing them as the
        # "deal" is exactly what generation does mid-hand. Tokenize on CPU (cheap);
        # query_model moves the tokens to the oracle model's device.
        tokens, masks = tokenize_batched(state, state.hands.unsqueeze(1), self.tokenizer)
        q = query_model(self.model, tokens, masks, state, 1, self.device)
        return q.float().cpu()


# ---------------------------------------------------------------------------
# Deal reconstruction
# ---------------------------------------------------------------------------


def _decision_context(game, d_idx: int):
    """Shared prefix bookkeeping for a decision: (P, decl_id, bidder,
    played_by_seat, prefix_plays)."""
    dec = game.decisions[d_idx]
    P = int(dec.player)
    decl_id = int(game.decl_id)
    bidder = int(game.decisions[0].player)
    played_by_seat: dict[int, list[int]] = {s: [] for s in range(4)}
    prefix: list[tuple[int, int]] = []
    for j in range(d_idx):
        dj = game.decisions[j]
        p = int(dj.player)
        dom = int(game.hands[p][int(dj.action_taken)])
        played_by_seat[p].append(dom)
        prefix.append((p, dom))
    return P, decl_id, bidder, played_by_seat, prefix


def _reconstruct_one(game, P, played_by_seat, world_hands_m):
    """Reconstruct one full 4×7 INITIAL deal, or None if the sampled world is
    inconsistent (a domino repeated or leaked into the actor's hand)."""
    hands: list[list[int]] = [[] for _ in range(4)]
    hands[P] = [int(x) for x in game.hands[P]]
    for rel in range(3):
        s = (P + rel + 1) % 4
        remaining = [int(d) for d in world_hands_m[rel].tolist() if d >= 0]
        hand = played_by_seat[s] + remaining
        if len(hand) != 7:
            return None
        hands[s] = hand
    flat = [d for row in hands for d in row]
    if len(set(flat)) != 28:
        return None  # duplicate / leak — a known sampler quirk at some root worlds
    return hands


def valid_world_indices(game, d_idx: int, n_want: int, seed: int):
    """Deterministically collect ``n_want`` world indices whose reconstruction is a
    valid 28-domino permutation.

    The eq corpus's opening-lead (d_idx=0) worlds contain a fraction of
    inconsistent samples (a domino the actor holds leaks into an opponent, or two
    opponents share a tile). We filter those out so every priced world is a real
    full deal. Returns (indices sorted, n_full, n_scanned).
    """
    dec = game.decisions[d_idx]
    P, _, _, played_by_seat, _ = _decision_context(game, d_idx)
    wh = dec.world_hands
    n_full = wh.shape[0]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_full, generator=g).tolist()
    valid: list[int] = []
    scanned = 0
    for idx in order:
        scanned += 1
        if _reconstruct_one(game, P, played_by_seat, wh[idx]) is not None:
            valid.append(int(idx))
            if len(valid) >= n_want:
                break
    return sorted(valid), int(n_full), scanned


def reconstruct_initial_deals(game, d_idx: int, world_hands: torch.Tensor):
    """Reconstruct full 4×7 INITIAL deals, one per sampled world.

    ``world_hands`` is [M,3,7] relative to the actor P = game.decisions[d_idx].player,
    row ``rel`` → opponent ``(P+rel+1)%4`` (matching ``gus/eval/lamir1.py`` and
    ``otis/analysis/worldbank.py``). Each opponent's initial hand =
    {dominoes it played in the recorded prefix} ∪ {its sampled remaining tiles}.
    P keeps its real deal. Callers must pass only VALID worlds (see
    ``valid_world_indices``); an inconsistent world raises.
    Returns (deals, prefix_plays, P, decl_id, bidder, M).
    """
    P, decl_id, bidder, played_by_seat, prefix = _decision_context(game, d_idx)
    M = world_hands.shape[0]
    deals: list[list[list[int]]] = []
    for m in range(M):
        hands = _reconstruct_one(game, P, played_by_seat, world_hands[m])
        if hands is None:
            raise ValueError(f"world {m}: inconsistent sampled deal (filter with valid_world_indices)")
        deals.append(hands)
    return deals, prefix, P, decl_id, bidder, M


# ---------------------------------------------------------------------------
# The shared rollout driver
# ---------------------------------------------------------------------------


def _choose_actions(scores, legal, cp, P, deals, state, decl_id, commitment, root_ply):
    """Legal argmax, then apply the actor's commitment. Returns slot tensor [M]."""
    M = scores.shape[0]
    masked = scores.masked_fill(~legal, float("-inf"))
    chosen = masked.argmax(dim=1)

    if commitment.root_domino is None and commitment.keep_id is None and commitment.release_id is None:
        return chosen

    for m in range(M):
        if int(cp[m].item()) != P:
            continue

        # domino -> current slot for P in this world
        remaining: dict[int, int] = {}
        for i in range(7):
            d = int(state.hands[m, P, i].item())
            if d >= 0:
                remaining[d] = i

        # (a) fixed root action
        if root_ply and commitment.root_domino is not None:
            s = remaining.get(commitment.root_domino)
            if s is not None and bool(legal[m, s]):
                chosen[m] = s
            continue  # root action overrides; retention does not fire at root

        if commitment.keep_id is None and commitment.release_id is None:
            continue

        # (b) retention constraint — only when P is void AND defaulting to a slough
        if not _actor_is_void(state, m, P, decl_id):
            continue
        chosen_dom = int(state.hands[m, P, int(chosen[m].item())].item())
        if _is_trump(chosen_dom, decl_id):
            continue  # trumping in, not sloughing — leave it

        legal_slots = [i for i in range(7) if bool(legal[m, i])]

        # release preference: dump the release tile first if still held & sloughable
        if commitment.release_id is not None and commitment.release_id in remaining:
            rs = remaining[commitment.release_id]
            if bool(legal[m, rs]) and not _is_trump(commitment.release_id, decl_id):
                chosen[m] = rs
                continue

        # keep protection: never slough the keep tile unless it is the only discard
        if commitment.keep_id is not None and commitment.keep_id in remaining:
            ks = remaining[commitment.keep_id]
            if int(chosen[m].item()) == ks:
                alts = [
                    s
                    for s in legal_slots
                    if s != ks and not _is_trump(int(state.hands[m, P, s].item()), decl_id)
                ]
                if alts:
                    chosen[m] = max(alts, key=lambda s: float(scores[m, s].item()))
    return chosen


def roll_worlds(deals, decl_id, bidder, prefix, P, policy, commitment=NONE):
    """Roll every world to hand-end under ``policy`` + the actor's ``commitment``.

    Returns a list of length-28 trajectories, each ``[(seat, domino_id), ...]`` in
    play order. The recorded prefix is force-replayed (deterministic, identical
    public sequence across worlds); the tail follows the policy.
    """
    from forge.eq.game_tensor import GameStateTensor

    M = len(deals)
    state = GameStateTensor.from_deals(deals, [decl_id] * M, device="cpu", bidders=[bidder] * M)
    dec_lists: list[list[PlayStub]] = [[] for _ in range(M)]

    # --- force-replay the recorded prefix (uniform current player) ---
    for (p, dom) in prefix:
        slots = torch.tensor([deals[m][p].index(dom) for m in range(M)], dtype=torch.long)
        for m in range(M):
            dec_lists[m].append(PlayStub(p, int(slots[m].item())))
        state = state.apply_actions(slots)

    # --- roll the tail ---
    root_ply = True
    while bool(state.active_games().any().item()):
        cp = state.current_player.long()
        legal = state.legal_actions()
        scores = policy.scores(state, deals, dec_lists, cp, decl_id)
        chosen = _choose_actions(scores, legal, cp, P, deals, state, decl_id, commitment, root_ply)
        for m in range(M):
            dec_lists[m].append(PlayStub(int(cp[m].item()), int(chosen[m].item())))
        state = state.apply_actions(chosen)
        root_ply = False

    trajectories: list[list[tuple[int, int]]] = []
    for m in range(M):
        plays = [(stub.player, deals[m][stub.player][stub.action_taken]) for stub in dec_lists[m]]
        if len(plays) != 28:
            raise ValueError(f"world {m}: rollout produced {len(plays)} plays, expected 28")
        trajectories.append(plays)
    return trajectories


# ---------------------------------------------------------------------------
# Fates + aggregation
# ---------------------------------------------------------------------------

N_FATE_CLASSES = 8  # (captured_by_my_team{0,1}) × (played_mode {led,followed,trumped_in,sloughed})


def _fate_class(tile_fate, P: int) -> int:
    """8-class index: (captured_by_my_team) * 4 + mode_index."""
    captured_by_my = int(tile_fate.winner_seat % 2 == P % 2)
    mode_idx = PLAYED_MODES.index(tile_fate.played_mode)
    return captured_by_my * 4 + mode_idx


def fate_class_name(idx: int) -> str:
    side = "my_team" if idx < 4 else "their_team"
    return f"{side}/{PLAYED_MODES[idx % 4]}"


@dataclass
class WorldOutcome:
    world_idx: int
    my_points: int
    their_points: int
    fate_class: dict[int, int]  # tile_id -> 8-class index
    plays: list[tuple[int, int]] = field(default=None, repr=False)


def outcomes_from_trajectories(trajectories, deals, decl_id, bidder, P, world_indices):
    """Referee each world's trajectory via the fate parser → WorldOutcome list."""
    my_team = P % 2
    outcomes: list[WorldOutcome] = []
    for local_m, plays in enumerate(trajectories):
        ng = NeutralGame(
            game_id=f"tiedroll:w{world_indices[local_m]}",
            hands=deals[local_m],
            decl_id=decl_id,
            bidder=bidder,
            bid_value=42,
            plays=plays,
        )
        gf = parse_game_fates(ng)  # asserts the P1 identity internally
        my_points = gf.team0_points if my_team == 0 else gf.team1_points
        their_points = gf.team1_points if my_team == 0 else gf.team0_points
        fate_class = {tf.tile_id: _fate_class(tf, P) for tf in gf.tiles}
        outcomes.append(
            WorldOutcome(
                world_idx=int(world_indices[local_m]),
                my_points=my_points,
                their_points=their_points,
                fate_class=fate_class,
                plays=plays,
            )
        )
    return outcomes


@dataclass
class Aggregate:
    mean_my_points_belief: float
    mean_my_points_uniform: float
    ess: float
    n_worlds: int
    # per tile_id -> [8] belief-weighted fate probability (and uniform)
    fate_dist_belief: dict[int, list[float]]
    fate_dist_uniform: dict[int, list[float]]


def aggregate_outcomes(outcomes, w_belief, ess) -> Aggregate:
    M = len(outcomes)
    w_b = np.asarray(w_belief, dtype=np.float64)
    w_u = np.full(M, 1.0 / M)
    my = np.array([o.my_points for o in outcomes], dtype=np.float64)

    fate_b: dict[int, list[float]] = {}
    fate_u: dict[int, list[float]] = {}
    for tid in COUNT_TILE_IDS:
        hb = np.zeros(N_FATE_CLASSES)
        hu = np.zeros(N_FATE_CLASSES)
        for m, o in enumerate(outcomes):
            c = o.fate_class[tid]
            hb[c] += w_b[m]
            hu[c] += w_u[m]
        fate_b[tid] = hb.tolist()
        fate_u[tid] = hu.tolist()

    return Aggregate(
        mean_my_points_belief=float((my * w_b).sum()),
        mean_my_points_uniform=float((my * w_u).sum()),
        ess=float(ess),
        n_worlds=M,
        fate_dist_belief=fate_b,
        fate_dist_uniform=fate_u,
    )


# ---------------------------------------------------------------------------
# Belief weights over the sampled worlds (reuse the world-bank instrument)
# ---------------------------------------------------------------------------


def belief_weights_for_decision(student, is_voids, game, d_idx, world_hands):
    """Belief posterior weight per sampled world (and ESS), via the gus belief head.

    Mirrors ``otis/analysis/worldbank.py`` — the belief head is world-independent,
    so it is queried with a zero world assignment.
    """
    from gus.model.features import extract_belief_target, reconstruct_prior_plays
    from gus.model.tokenize import tokenize_decision
    from gus.model.voids import voids_feature_vector

    from otis.analysis.worldbank import belief_weights, world_seat_matrix

    actor = int(game.decisions[d_idx].player)
    prior = reconstruct_prior_plays(game.hands, game.decisions, d_idx)
    _, mask = extract_belief_target(game.hands, prior, actor)
    hidden = [d for d in range(28) if bool(mask[d])]

    tokens, attn = tokenize_decision(game.hands, int(game.decl_id), game.decisions, d_idx)
    voids = voids_feature_vector(prior, int(game.decl_id), actor)
    zw = torch.zeros(28, 3)
    with torch.no_grad():
        if is_voids:
            out = student(tokens.unsqueeze(0), attn.unsqueeze(0), zw.unsqueeze(0), voids.unsqueeze(0))
        else:
            out = student(tokens.unsqueeze(0), attn.unsqueeze(0), zw.unsqueeze(0))
    belief_logits = out["belief_logits"][0]

    seat_of = world_seat_matrix(world_hands)
    w, ess = belief_weights(belief_logits, seat_of, hidden)
    return w, ess


# ---------------------------------------------------------------------------
# Top-level: run one leg on one decision
# ---------------------------------------------------------------------------


@dataclass
class LegResult:
    commitment: str
    aggregate: Aggregate
    P: int = -1
    decl_id: int = -1
    bidder: int = -1
    outcomes: list[WorldOutcome] = field(default=None, repr=False)
    deals: list = field(default=None, repr=False)


def run_leg(game, d_idx, policy, commitment, world_indices, w_belief, ess):
    """Roll one leg (policy + commitment) on the chosen worlds → LegResult."""
    wh = game.decisions[d_idx].world_hands[world_indices]  # [M,3,7]
    deals, prefix, P, decl_id, bidder, M = reconstruct_initial_deals(game, d_idx, wh)
    trajectories = roll_worlds(deals, decl_id, bidder, prefix, P, policy, commitment)
    outcomes = outcomes_from_trajectories(trajectories, deals, decl_id, bidder, P, world_indices)
    agg = aggregate_outcomes(outcomes, w_belief, ess)
    return LegResult(
        commitment=commitment.label, aggregate=agg, P=P, decl_id=decl_id,
        bidder=bidder, outcomes=outcomes, deals=deals,
    )


def slough_profile(leg: LegResult):
    """Count how often the actor plays each domino as a void-slough (a non-trump
    discard) across the leg's worlds — the guard/walker candidates the retention
    lever can act on."""
    from collections import Counter

    from forge.eq.game_tensor import GameStateTensor

    ct: Counter = Counter()
    for local_m, o in enumerate(leg.outcomes):
        deal = leg.deals[local_m]
        state = GameStateTensor.from_deals([deal], [leg.decl_id], device="cpu", bidders=[leg.bidder])
        for (seat, dom) in o.plays:
            slot = deal[seat].index(dom)
            if (
                seat == leg.P
                and _actor_is_void(state, 0, leg.P, leg.decl_id)
                and not _is_trump(dom, leg.decl_id)
            ):
                ct[dom] += 1
            state = state.apply_actions(torch.tensor([slot], dtype=torch.long))
    return ct


# ---------------------------------------------------------------------------
# Retention pricing (item 4) — the fusion gap
# ---------------------------------------------------------------------------


@dataclass
class RetentionPrice:
    keep_id: int
    release_id: int
    tied_delta: float       # value of keeping keep_id vs keeping release_id (belief-weighted)
    clairvoyant_delta: float
    fusion_gap: float       # tied_delta - clairvoyant_delta
    tied_delta_uniform: float
    clairvoyant_delta_uniform: float


def price_retention(
    game, d_idx, keep_id, release_id, tied_policy, clair_policy, world_indices, w_belief, ess
):
    """Price keeping ``keep_id`` vs ``release_id`` on common random worlds.

    Runs two paired commitments on each leg:
      c1 = (keep=keep_id, release=release_id)   — protect keep_id, dump release_id
      c2 = (keep=release_id, release=keep_id)   — protect release_id, dump keep_id
    delta = mean_my_points(c1) − mean_my_points(c2). The tied delta prices the
    guard/walker under information honesty; the clairvoyant delta is the fusion
    baseline; fusion_gap = tied_delta − clairvoyant_delta.
    """
    c1 = Commitment(label=f"keep{keep_id}_rel{release_id}", keep_id=keep_id, release_id=release_id)
    c2 = Commitment(label=f"keep{release_id}_rel{keep_id}", keep_id=release_id, release_id=keep_id)

    tied_1 = run_leg(game, d_idx, tied_policy, c1, world_indices, w_belief, ess).aggregate
    tied_2 = run_leg(game, d_idx, tied_policy, c2, world_indices, w_belief, ess).aggregate
    clair_1 = run_leg(game, d_idx, clair_policy, c1, world_indices, w_belief, ess).aggregate
    clair_2 = run_leg(game, d_idx, clair_policy, c2, world_indices, w_belief, ess).aggregate

    tied_delta = tied_1.mean_my_points_belief - tied_2.mean_my_points_belief
    clair_delta = clair_1.mean_my_points_belief - clair_2.mean_my_points_belief
    return RetentionPrice(
        keep_id=keep_id,
        release_id=release_id,
        tied_delta=tied_delta,
        clairvoyant_delta=clair_delta,
        fusion_gap=tied_delta - clair_delta,
        tied_delta_uniform=tied_1.mean_my_points_uniform - tied_2.mean_my_points_uniform,
        clairvoyant_delta_uniform=clair_1.mean_my_points_uniform - clair_2.mean_my_points_uniform,
    )


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

DEFAULT_STUDENT = "gus/adapters/v3_consistency_10000g.pt"
DEFAULT_ORACLE = "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"


def load_tied_policy(adapter_path: str, device: str) -> TiedPolicy:
    from gus.model.load import load_student

    student, is_voids = load_student(adapter_path, device)
    student.eval()
    return TiedPolicy(student, is_voids)


def load_clairvoyant_policy(ckpt_path: str, device: str, max_batch: int) -> ClairvoyantPolicy:
    from forge.eq.oracle import Stage1Oracle
    from forge.eq.tokenize_gpu import GPUTokenizer

    oracle = Stage1Oracle(
        ckpt_path, device=device, compile=False, use_async=False, use_gpu_tokenizer=False
    )
    # Tokenizer stays on CPU to match the CPU rollout state; query_model moves the
    # tokens onto the oracle model's device (cpu or mps for the probe).
    tokenizer = GPUTokenizer(max_batch=max_batch, device="cpu")
    return ClairvoyantPolicy(oracle.model, tokenizer, device)


# ---------------------------------------------------------------------------
# Smoke CLI
# ---------------------------------------------------------------------------


def _find_void_capable_decisions(games, n_want, min_worlds, max_trick=1):
    """Pick trick-0/1 decisions where the actor holds >= 2 non-count, non-trump
    junk tiles (guard/walker candidates for the retention lever)."""
    picks = []
    for gi, game in enumerate(games):
        if len(game.decisions) != 28:
            continue
        decl_id = int(game.decl_id)
        for d_idx in range(4 * (max_trick + 1)):
            dec = game.decisions[d_idx]
            if dec.world_hands is None or dec.q_per_world is None:
                continue
            if dec.world_hands.shape[0] < min_worlds:
                continue
            P = int(dec.player)
            played_by_P = {
                int(game.hands[int(game.decisions[j].player)][int(game.decisions[j].action_taken)])
                for j in range(d_idx)
                if int(game.decisions[j].player) == P
            }
            remaining = [int(x) for x in game.hands[P] if int(x) >= 0 and int(x) not in played_by_P]
            junk = [d for d in remaining if d not in COUNT_TILE_IDS and not _is_trump(d, decl_id)]
            if len(junk) >= 2:
                picks.append((gi, d_idx, P))
                break
        if len(picks) >= n_want:
            break
    return picks


def run_smoke(
    corpus_path: str,
    device: str = "cpu",
    n_worlds: int = 20,
    n_decisions: int = 2,
    seed: int = 0,
    student_path: str = DEFAULT_STUDENT,
    oracle_path: str = DEFAULT_ORACLE,
    out_json: str = "scratch/otis-night/w6_smoke.json",
    max_probe_tiles: int = 3,
    log_every_s: float = 30.0,
):
    """CPU smoke: trick-0/1 void-capable decisions, both legs, retention pricing.

    Per decision: roll unconstrained tied + clairvoyant baselines, profile which
    junk tiles the actor sloughs, then price every unordered pair among the
    top-``max_probe_tiles`` sloughed junk tiles (tied vs clairvoyant → fusion gap).
    """
    t0 = time.time()
    print(f"[tiedroll] loading corpus {corpus_path}", flush=True)
    blob = torch.load(corpus_path, map_location="cpu", weights_only=False)
    games = blob["results"]

    print("[tiedroll] loading policies", flush=True)
    tied_policy = load_tied_policy(student_path, device)
    clair_policy = load_clairvoyant_policy(oracle_path, device, max_batch=n_worlds)

    picks = _find_void_capable_decisions(games, n_decisions, min_worlds=n_worlds)
    print(f"[tiedroll] {len(picks)} void-capable decisions selected", flush=True)

    results = []
    for (gi, d_idx, P) in picks:
        game = games[gi]
        dec = game.decisions[d_idx]
        world_indices, n_full, scanned = valid_world_indices(game, d_idx, n_worlds, seed)
        if len(world_indices) < n_worlds:
            print(
                f"[tiedroll] skip game {gi} d{d_idx}: only {len(world_indices)} valid "
                f"worlds among {scanned} scanned",
                flush=True,
            )
            continue
        wh_sel = dec.world_hands[world_indices]

        w_belief, ess = belief_weights_for_decision(
            tied_policy.student, tied_policy.is_voids, game, d_idx, wh_sel
        )

        t_dec = time.time()
        # unconstrained baselines (both legs)
        tied_base = run_leg(game, d_idx, tied_policy, NONE, world_indices, w_belief, ess)
        clair_base = run_leg(game, d_idx, clair_policy, NONE, world_indices, w_belief, ess)
        base_secs = time.time() - t_dec

        # slough profile → guard/walker candidates
        profile = slough_profile(tied_base)
        probe_tiles = [d for d, _ in profile.most_common()][:max_probe_tiles]

        # price every unordered pair among the probe tiles (tied vs clairvoyant)
        import itertools

        t_price = time.time()
        cells = []
        for keep_id, release_id in itertools.combinations(probe_tiles, 2):
            price = price_retention(
                game, d_idx, keep_id, release_id, tied_policy, clair_policy,
                world_indices, w_belief, ess,
            )
            is_gap_cell = (
                abs(price.tied_delta) >= 2.0 and abs(price.clairvoyant_delta) <= 0.5
            )
            cells.append(
                {
                    "keep_id": keep_id,
                    "release_id": release_id,
                    "tied_delta": round(price.tied_delta, 3),
                    "clairvoyant_delta": round(price.clairvoyant_delta, 3),
                    "fusion_gap": round(price.fusion_gap, 3),
                    "tied_delta_uniform": round(price.tied_delta_uniform, 3),
                    "clairvoyant_delta_uniform": round(price.clairvoyant_delta_uniform, 3),
                    "p7_gap_cell": bool(is_gap_cell),
                }
            )
        n_cells = max(len(cells), 1)
        price_secs = time.time() - t_price
        dec_secs = time.time() - t_dec

        rec = {
            "game_idx": gi,
            "d_idx": d_idx,
            "trick": d_idx // 4,
            "actor": P,
            "decl_id": int(game.decl_id),
            "n_worlds": len(world_indices),
            "n_full_worlds": int(n_full),
            "n_worlds_scanned_for_valid": int(scanned),
            "ess": round(ess, 2),
            "tied_base_my_points_belief": round(tied_base.aggregate.mean_my_points_belief, 3),
            "tied_base_my_points_uniform": round(tied_base.aggregate.mean_my_points_uniform, 3),
            "clairvoyant_base_my_points_belief": round(clair_base.aggregate.mean_my_points_belief, 3),
            "clairvoyant_base_my_points_uniform": round(clair_base.aggregate.mean_my_points_uniform, 3),
            "slough_profile": {str(k): int(v) for k, v in profile.most_common()},
            "probe_tiles": probe_tiles,
            "retention_cells": cells,
            "n_gap_cells": sum(1 for c in cells if c["p7_gap_cell"]),
            "tied_base_fate_dist_belief": {
                str(tid): [round(x, 4) for x in tied_base.aggregate.fate_dist_belief[tid]]
                for tid in COUNT_TILE_IDS
            },
            # timing granularities for probe planning
            "secs_baseline_two_legs": round(base_secs, 2),
            "secs_pricing_all_cells": round(price_secs, 2),
            "secs_per_cell": round(price_secs / n_cells, 2),
            "secs_per_decision": round(dec_secs, 2),
            "secs_per_cell_M50_extrapolated": round((price_secs / n_cells) * (50.0 / len(world_indices)), 2),
        }
        results.append(rec)
        best_gap = max((c["fusion_gap"] for c in cells), default=0.0)
        print(
            f"[tiedroll] game {gi} d{d_idx} actor{P} decl{int(game.decl_id)}: "
            f"tied_base={rec['tied_base_my_points_belief']:.2f} "
            f"clair_base={rec['clairvoyant_base_my_points_belief']:.2f} "
            f"cells={len(cells)} gap_cells={rec['n_gap_cells']} best_gap={best_gap:+.2f} "
            f"({dec_secs:.1f}s, {rec['secs_per_cell']:.2f}s/cell)",
            flush=True,
        )

    wall = time.time() - t0
    payload = {
        "smoke": True,
        "note": "SMOKE ONLY — probe pending on MPS. CPU timings are upper bounds.",
        "device": device,
        "corpus": corpus_path,
        "n_worlds": n_worlds,
        "seed": seed,
        "student": student_path,
        "oracle": oracle_path,
        "fate_classes": [fate_class_name(i) for i in range(N_FATE_CLASSES)],
        "count_tile_ids": list(COUNT_TILE_IDS),
        "wall_s": round(wall, 1),
        "decisions": results,
    }
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(payload, indent=2))
    print(f"[tiedroll] wrote {out} ({wall:.1f}s total)", flush=True)
    return payload


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Otis tied-strategy rollout (P7)")
    ap.add_argument("--corpus", default="gus/data/corpus_v2_eval.pt")
    ap.add_argument("--device", default="cpu", help="cpu (default) or mps for the probe")
    ap.add_argument("--n-worlds", type=int, default=20)
    ap.add_argument("--n-decisions", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--student", default=DEFAULT_STUDENT)
    ap.add_argument("--oracle", default=DEFAULT_ORACLE)
    ap.add_argument("--out", default="scratch/otis-night/w6_smoke.json")
    args = ap.parse_args()
    run_smoke(
        corpus_path=args.corpus,
        device=args.device,
        n_worlds=args.n_worlds,
        n_decisions=args.n_decisions,
        seed=args.seed,
        student_path=args.student,
        oracle_path=args.oracle,
        out_json=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
