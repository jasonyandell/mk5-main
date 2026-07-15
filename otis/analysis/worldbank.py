"""Otis world-bank analysis — the playout-free rung of issue #49 (items 1+2, P6).

Everything here derives from the stored joint-world tensors in the eq corpus:
``world_hands`` [M,3,7] (sampled opponent hands, relative to the actor) and
``q_per_world`` [M,7] (oracle Q per action per world). Schema:
``forge/eq/generate/types.py:54-104``. Loader precedent:
``gus/model/dataset_seq_world.py``.

The instrument answers three questions with NO new game playouts:

A) **Belief weights per world.** The canonical gus student
   (``gus/adapters/v3_consistency_10000g.pt``, a voids transformer whose belief
   head is world-independent) produces ``belief_logits`` [28,3] =
   log-scores of P(relative-seat | domino). For a sampled world we weight it by
   ``prod_{hidden d} P(assigned_seat_d | d)``, normalized over the decision's M
   worlds, and record ESS. A uniform-weight variant (w = 1/M) runs alongside
   everywhere.

B) **P6 — is the 3-2's context-clustered value bimodal?** For trick-0 decisions
   where the actor holds the 3-2, cluster the M worlds by the joint placement of
   the OTHER four count tiles (5-5, 6-4, 5-0, 4-1) across the three hidden
   relative seats / the actor's own hand, then test whether the belief-weighted
   cluster means of ``q_per_world[:, a*]`` (a* = argmax E[Q] over legal actions)
   split into two modes each carrying >= 20% belief mass, separated by >= 10
   points. Band (``wiki/experiments/otis-v0.md`` P6): >= 30% of qualifying
   decisions bimodal = PASS; < 10% = falsifier.

C) **Interaction structure (issue item 1).** Over trick-0/trick-1 decisions,
   first-order effect delta(d@s) and pairwise lift(d1@s1, d2@s2) of count tiles
   at relative seats, per-decision centered and pooled; export top cells and
   flag "count tile at opponent hurts UNLESS partner holds X" (lift sign
   opposing the first-order sum).

Relative-seat convention (matches ``gus/model/features.py:86-146``): seat index
0 = left_opp = (actor+1)%4, 1 = partner = (actor+2)%4, 2 = right_opp =
(actor+3)%4. The belief head's seat axis, the ``world_hands`` row axis, and this
placement code all share that ordering — so no absolute<->relative remap is
needed to weight a world.

CPU only. otis does not own the GPU.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Domino / count-tile constants
# ---------------------------------------------------------------------------

def pips_to_id(hi: int, lo: int) -> int:
    """Triangular domino id: id = hi*(hi+1)/2 + lo (0..27), hi >= lo."""
    if lo > hi:
        hi, lo = lo, hi
    return hi * (hi + 1) // 2 + lo


def id_to_pips(did: int) -> tuple[int, int]:
    hi = 0
    while (hi + 1) * (hi + 2) // 2 <= did:
        hi += 1
    return hi, did - hi * (hi + 1) // 2


# The five count tiles and their point values (35 count points total).
THREE_TWO = pips_to_id(3, 2)  # 8   — the P6 object of interest
COUNT_TILES = {
    pips_to_id(5, 5): 10,  # 20
    pips_to_id(6, 4): 10,  # 25
    pips_to_id(5, 0): 5,   # 15
    pips_to_id(4, 1): 5,   # 11
    pips_to_id(3, 2): 5,   # 8
}
# The "other four" for the P6 context vector (all count tiles except the 3-2).
OTHER_COUNT_TILES = tuple(t for t in COUNT_TILES if t != THREE_TWO)  # (20, 25, 15, 11)

SEAT_WORDS = {0: "left opp", 1: "partner", 2: "right opp", -1: "your hand"}


def tile_name(did: int) -> str:
    hi, lo = id_to_pips(did)
    return f"{hi}-{lo}"


# ---------------------------------------------------------------------------
# Pure, unit-tested core: clustering + bimodality split
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    key: tuple           # context code tuple, or ("other",)
    mass_belief: float   # summed belief weight
    mass_uniform: float  # summed uniform weight
    mean_belief: float   # belief-weighted mean q within cluster
    mean_uniform: float  # uniform-weighted mean q within cluster
    n_worlds: int
    q_min: float
    q_max: float
    world_idx: np.ndarray = field(default=None, repr=False)


def cluster_worlds(
    context_codes: list[tuple],
    q: np.ndarray,
    w_belief: np.ndarray,
    w_uniform: np.ndarray,
    merge_threshold: float = 0.02,
) -> list[Cluster]:
    """Group worlds by exact context code; merge clusters under ``merge_threshold``
    belief mass into a single "other" cluster.

    Args:
        context_codes: length-M list of hashable context tuples (one per world).
        q:             [M] value per world (q_per_world[:, a*]).
        w_belief:      [M] belief weights (should sum to ~1).
        w_uniform:     [M] uniform weights (should sum to ~1).
        merge_threshold: clusters below this belief mass fold into "other".

    Returns list of Cluster, sorted by ascending belief-weighted mean. The
    "other" bucket (if any) is included as a normal cluster with key ("other",).
    """
    q = np.asarray(q, dtype=np.float64)
    w_belief = np.asarray(w_belief, dtype=np.float64)
    w_uniform = np.asarray(w_uniform, dtype=np.float64)
    M = len(context_codes)

    groups: dict[tuple, list[int]] = {}
    for m in range(M):
        groups.setdefault(context_codes[m], []).append(m)

    kept: list[Cluster] = []
    other_idx: list[int] = []
    for key, idx in groups.items():
        idx_arr = np.array(idx, dtype=np.int64)
        mb = float(w_belief[idx_arr].sum())
        if mb < merge_threshold:
            other_idx.extend(idx)
            continue
        kept.append(_make_cluster(key, idx_arr, q, w_belief, w_uniform))

    if other_idx:
        idx_arr = np.array(sorted(other_idx), dtype=np.int64)
        kept.append(_make_cluster(("other",), idx_arr, q, w_belief, w_uniform))

    kept.sort(key=lambda c: c.mean_belief)
    return kept


def _make_cluster(key, idx_arr, q, w_belief, w_uniform) -> Cluster:
    qb = q[idx_arr]
    wb = w_belief[idx_arr]
    wu = w_uniform[idx_arr]
    mb = float(wb.sum())
    mu = float(wu.sum())
    mean_b = float((qb * wb).sum() / mb) if mb > 0 else float("nan")
    mean_u = float((qb * wu).sum() / mu) if mu > 0 else float("nan")
    return Cluster(
        key=key,
        mass_belief=mb,
        mass_uniform=mu,
        mean_belief=mean_b,
        mean_uniform=mean_u,
        n_worlds=int(len(idx_arr)),
        q_min=float(qb.min()),
        q_max=float(qb.max()),
        world_idx=idx_arr,
    )


@dataclass
class BimodalResult:
    is_bimodal: bool
    gap: float          # high group mean - low group mean at the chosen split
    low_mass: float
    high_mass: float
    low_mean: float
    high_mean: float
    n_clusters: int
    split_k: int        # number of clusters in the low group (chosen split)


def bimodality_split(
    means: np.ndarray,
    masses: np.ndarray,
    min_group_mass: float = 0.20,
    min_gap: float = 10.0,
) -> BimodalResult:
    """Split clusters (by mean) into low/high groups maximizing group-mean
    separation, then decide bimodality.

    Method: sort clusters by mean; for each of the C-1 threshold splits compute
    the mass-weighted mean of each group and their gap. A decision is bimodal
    iff there EXISTS a split where both groups carry >= ``min_group_mass`` belief
    mass AND the group-mean gap >= ``min_gap``. Among the mass-valid splits we
    report the one with the largest gap; if none is mass-valid we report the
    globally-max-gap split (is_bimodal=False, mass condition failed).

    (Reading the P6 band's "maximizing group-mean separation ... iff both groups
    >= 20% mass AND gap >= 10 points" as an existence test over the mass-valid
    splits — the operational choice for detecting two real modes; documented in
    the methods section.)
    """
    means = np.asarray(means, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    C = len(means)
    if C < 2:
        return BimodalResult(False, 0.0, float(masses.sum()) if C else 0.0,
                             0.0, float(means[0]) if C else float("nan"),
                             float("nan"), C, 0)

    order = np.argsort(means)
    sm = means[order]
    sw = masses[order]
    total = sw.sum()
    if total <= 0:
        return BimodalResult(False, 0.0, 0.0, 0.0, float("nan"), float("nan"), C, 0)
    sw = sw / total

    valid: list[dict] = []
    allcand: list[dict] = []
    for k in range(1, C):
        lw = float(sw[:k].sum())
        hw = float(sw[k:].sum())
        lmean = float((sm[:k] * sw[:k]).sum() / lw)
        hmean = float((sm[k:] * sw[k:]).sum() / hw)
        cand = dict(k=k, gap=hmean - lmean, low_mass=lw, high_mass=hw,
                    low_mean=lmean, high_mean=hmean)
        allcand.append(cand)
        if lw >= min_group_mass and hw >= min_group_mass:
            valid.append(cand)

    pool = valid if valid else allcand
    chosen = max(pool, key=lambda c: c["gap"])
    is_bimodal = bool(valid) and chosen["gap"] >= min_gap
    return BimodalResult(
        is_bimodal=is_bimodal,
        gap=chosen["gap"],
        low_mass=chosen["low_mass"],
        high_mass=chosen["high_mass"],
        low_mean=chosen["low_mean"],
        high_mean=chosen["high_mean"],
        n_clusters=C,
        split_k=chosen["k"],
    )


# ---------------------------------------------------------------------------
# Belief weighting over sampled worlds
# ---------------------------------------------------------------------------

def world_seat_matrix(world_hands: torch.Tensor) -> torch.Tensor:
    """[M,3,7] world_hands -> [M,28] long: relative seat holding each domino in
    each world, or -1 if the domino is not assigned in that world (own hand,
    already played, or a -1 pad slot)."""
    M = world_hands.shape[0]
    seat_of = torch.full((M, 28), -1, dtype=torch.long)
    wh = world_hands.long()
    rows = torch.arange(M)
    for seat in range(3):
        ids = wh[:, seat, :]  # [M,7]
        for j in range(7):
            col = ids[:, j]
            valid = col >= 0
            if valid.any():
                seat_of[rows[valid], col[valid]] = seat
    return seat_of


def belief_weights(
    belief_logits: torch.Tensor,   # [28,3]
    seat_of: torch.Tensor,         # [M,28] from world_seat_matrix
    hidden_ids: list[int],
) -> tuple[np.ndarray, float]:
    """weight(world) proportional to prod_{hidden d} P(assigned_seat_d | d).

    Returns (normalized weights [M], ess). ESS = (sum w)^2 / sum w^2 on the
    normalized weights (== 1/sum w_norm^2).
    """
    logp = torch.log_softmax(belief_logits, dim=-1)  # [28,3]
    if not hidden_ids:
        M = seat_of.shape[0]
        w = np.full(M, 1.0 / M)
        return w, float(M)
    H = torch.tensor(hidden_ids, dtype=torch.long)
    sel = seat_of[:, H]                       # [M, |H|] seat per hidden domino
    lpH = logp[H]                             # [|H|, 3]
    M = sel.shape[0]
    lpH_exp = lpH.unsqueeze(0).expand(M, -1, -1)          # [M,|H|,3]
    sel_clamped = sel.clamp(min=0).unsqueeze(-1)          # [M,|H|,1]
    gathered = lpH_exp.gather(2, sel_clamped).squeeze(-1) # [M,|H|]
    gathered = torch.where(sel >= 0, gathered, torch.zeros_like(gathered))
    logw = gathered.sum(dim=1)                # [M]
    logw = logw - logw.max()
    w = torch.softmax(logw, dim=0).numpy().astype(np.float64)
    ess = float(1.0 / np.square(w).sum())
    return w, ess


# ---------------------------------------------------------------------------
# Decision-level plumbing (features -> belief -> per-world placements)
# ---------------------------------------------------------------------------

@dataclass
class DecisionView:
    game_id: str
    decl_id: int
    actor: int
    d_idx: int
    trick: int
    M: int
    hidden_ids: list[int]
    own_ids: set
    a_star: int
    action_taken: int
    q_astar: np.ndarray        # [M] q_per_world[:, a*]
    seat_of: torch.Tensor      # [M,28]
    w_belief: np.ndarray       # [M]
    w_uniform: np.ndarray      # [M]
    ess: float
    # placement per count tile: dict tile_id -> np.ndarray[M] of codes in {-1,0,1,2}
    placements: dict


def _build_decision_view(model, game, d_idx: int, game_id: str) -> DecisionView | None:
    """Run belief, compute per-world weights and count-tile placements for one
    decision. Returns None if the decision carries no joint-world tensor."""
    from gus.model.features import reconstruct_prior_plays, extract_belief_target
    from gus.model.tokenize import tokenize_decision
    from gus.model.voids import voids_feature_vector

    dec = game.decisions[d_idx]
    if dec.world_hands is None or dec.q_per_world is None:
        return None
    actor = int(dec.player)
    wh = dec.world_hands                      # [M,3,7]
    qpw = dec.q_per_world.float().numpy()     # [M,7]
    M = wh.shape[0]

    prior_plays = reconstruct_prior_plays(game.hands, game.decisions, d_idx)
    _, belief_mask = extract_belief_target(game.hands, prior_plays, actor)
    hidden_ids = [d for d in range(28) if bool(belief_mask[d])]

    tokens, attn = tokenize_decision(game.hands, int(game.decl_id), game.decisions, d_idx)
    voids = voids_feature_vector(prior_plays, int(game.decl_id), actor)
    world_assign = torch.zeros(28, 3)  # belief head is world-independent
    with torch.no_grad():
        out = model(tokens.unsqueeze(0), attn.unsqueeze(0),
                    world_assign.unsqueeze(0), voids.unsqueeze(0))
    belief_logits = out["belief_logits"][0]   # [28,3]

    seat_of = world_seat_matrix(wh)
    w_belief, ess = belief_weights(belief_logits, seat_of, hidden_ids)
    w_uniform = np.full(M, 1.0 / M)

    # a* = argmax E[Q] over legal actions
    e_q = dec.e_q.float().clone()
    legal = dec.legal_mask.bool()
    e_q[~legal] = float("-inf")
    a_star = int(torch.argmax(e_q).item())
    action_taken = int(dec.action_taken)
    q_astar = qpw[:, a_star]

    own_ids = {int(x) for x in game.hands[actor] if int(x) >= 0}
    played_placement: dict[int, int] = {}
    for (p, d) in prior_plays:
        played_placement[int(d)] = ((int(p) - actor) % 4) - 1  # -> seat 0/1/2

    placements: dict[int, np.ndarray] = {}
    seat_np = seat_of.numpy()
    for t in COUNT_TILES:
        if t in own_ids:
            placements[t] = np.full(M, -1, dtype=np.int64)  # own hand
        elif t in played_placement:
            placements[t] = np.full(M, played_placement[t], dtype=np.int64)
        else:
            placements[t] = seat_np[:, t].astype(np.int64)  # per-world hidden seat

    return DecisionView(
        game_id=game_id, decl_id=int(game.decl_id), actor=actor, d_idx=d_idx,
        trick=d_idx // 4, M=M, hidden_ids=hidden_ids, own_ids=own_ids,
        a_star=a_star, action_taken=action_taken, q_astar=q_astar,
        seat_of=seat_of, w_belief=w_belief, w_uniform=w_uniform, ess=ess,
        placements=placements,
    )


# ---------------------------------------------------------------------------
# P6 per-decision analysis
# ---------------------------------------------------------------------------

@dataclass
class P6Row:
    game_id: str
    decl_id: int
    actor: int
    d_idx: int
    n_worlds: int
    ess: float
    a_star: int
    action_taken: int
    n_clusters_belief: int
    bimodal_belief: bool
    gap_belief: float
    low_mass_belief: float
    high_mass_belief: float
    n_clusters_uniform: int
    bimodal_uniform: bool
    gap_uniform: float
    low_mass_uniform: float
    high_mass_uniform: float
    clusters: list = field(default=None, repr=False)


def _context_codes(view: DecisionView, tiles=OTHER_COUNT_TILES) -> list[tuple]:
    M = view.M
    per_tile = [view.placements[t] for t in tiles]
    return [tuple(int(pt[m]) for pt in per_tile) for m in range(M)]


def analyze_p6_decision(view: DecisionView, merge_threshold: float = 0.02) -> P6Row:
    codes = _context_codes(view)

    cl_b = cluster_worlds(codes, view.q_astar, view.w_belief, view.w_uniform,
                          merge_threshold)
    means_b = np.array([c.mean_belief for c in cl_b])
    mass_b = np.array([c.mass_belief for c in cl_b])
    bim_b = bimodality_split(means_b, mass_b)

    # Uniform variant: recluster by uniform mass (merge threshold on uniform),
    # value = uniform-weighted mean.
    cl_u = cluster_worlds(codes, view.q_astar, view.w_uniform, view.w_uniform,
                          merge_threshold)
    means_u = np.array([c.mean_belief for c in cl_u])  # here w_belief arg == uniform
    mass_u = np.array([c.mass_belief for c in cl_u])
    bim_u = bimodality_split(means_u, mass_u)

    return P6Row(
        game_id=view.game_id, decl_id=view.decl_id, actor=view.actor, d_idx=view.d_idx,
        n_worlds=view.M, ess=view.ess, a_star=view.a_star, action_taken=view.action_taken,
        n_clusters_belief=len(cl_b), bimodal_belief=bim_b.is_bimodal,
        gap_belief=bim_b.gap, low_mass_belief=bim_b.low_mass, high_mass_belief=bim_b.high_mass,
        n_clusters_uniform=len(cl_u), bimodal_uniform=bim_u.is_bimodal,
        gap_uniform=bim_u.gap, low_mass_uniform=bim_u.low_mass, high_mass_uniform=bim_u.high_mass,
        clusters=cl_b,
    )


def is_p6_qualifying(game, d_idx: int, min_worlds: int = 100) -> bool:
    """trick 0, actor holds the 3-2, >= min_worlds sampled worlds."""
    if d_idx >= 4:  # trick 0 == decisions 0..3
        return False
    dec = game.decisions[d_idx]
    if dec.world_hands is None:
        return False
    if dec.world_hands.shape[0] < min_worlds:
        return False
    actor = int(dec.player)
    return THREE_TWO in {int(x) for x in game.hands[actor] if int(x) >= 0}


# ---------------------------------------------------------------------------
# Interaction structure (issue item 1)
# ---------------------------------------------------------------------------

class InteractionAccumulator:
    """Per-decision-centered, belief-weighted first-order + pairwise effects of
    count tiles at relative seats, pooled across decisions.

    For each decision, base = weighted E[Q(a*)]; every world contributes its
    centered value (q - base) with its belief weight (weights sum to 1 per
    decision, so decisions contribute equally). Cell = (tile, seat) active in a
    world iff that count tile is placed at that relative seat (0/1/2) in the
    world. Global E[Q] is 0 after centering.
    """

    def __init__(self, tiles=tuple(COUNT_TILES.keys())):
        self.tiles = tiles
        self.n_decisions = 0
        self.cell_mass: dict[tuple, float] = {}
        self.cell_val: dict[tuple, float] = {}
        self.cell_n: dict[tuple, int] = {}
        self.pair_mass: dict[tuple, float] = {}
        self.pair_val: dict[tuple, float] = {}
        self.pair_n: dict[tuple, int] = {}

    def add(self, view: DecisionView):
        w = view.w_belief
        q = view.q_astar
        base = float((w * q).sum())
        cq = q - base
        M = view.M
        # active cells per world
        placements = {t: view.placements[t] for t in self.tiles}
        # accumulate first-order
        for t in self.tiles:
            pt = placements[t]
            for seat in (0, 1, 2):
                mask = pt == seat
                if not mask.any():
                    continue
                cell = (t, seat)
                self.cell_mass[cell] = self.cell_mass.get(cell, 0.0) + float(w[mask].sum())
                self.cell_val[cell] = self.cell_val.get(cell, 0.0) + float((w[mask] * cq[mask]).sum())
                self.cell_n[cell] = self.cell_n.get(cell, 0) + int(mask.sum())
        # pairwise (unordered, distinct tiles)
        tl = list(self.tiles)
        for i in range(len(tl)):
            for j in range(i + 1, len(tl)):
                t1, t2 = tl[i], tl[j]
                p1, p2 = placements[t1], placements[t2]
                for s1 in (0, 1, 2):
                    m1 = p1 == s1
                    if not m1.any():
                        continue
                    for s2 in (0, 1, 2):
                        both = m1 & (p2 == s2)
                        if not both.any():
                            continue
                        # canonical order of the two cells
                        c1, c2 = (t1, s1), (t2, s2)
                        key = (c1, c2) if c1 <= c2 else (c2, c1)
                        self.pair_mass[key] = self.pair_mass.get(key, 0.0) + float(w[both].sum())
                        self.pair_val[key] = self.pair_val.get(key, 0.0) + float((w[both] * cq[both]).sum())
                        self.pair_n[key] = self.pair_n.get(key, 0) + int(both.sum())
        self.n_decisions += 1

    def first_order(self) -> dict[tuple, float]:
        return {c: self.cell_val[c] / self.cell_mass[c]
                for c in self.cell_mass if self.cell_mass[c] > 0}

    def results(self, min_worlds: int = 30) -> list[dict]:
        delta = self.first_order()
        nd = max(self.n_decisions, 1)
        rows: list[dict] = []
        for key, n in self.pair_n.items():
            if n < min_worlds:
                continue
            mass = self.pair_mass[key]
            if mass <= 0:
                continue
            mean_pair = self.pair_val[key] / mass
            c1, c2 = key
            d1 = delta.get(c1, 0.0)
            d2 = delta.get(c2, 0.0)
            lift = mean_pair - d1 - d2
            fo_sum = d1 + d2
            avg_mass = mass / nd  # avg belief mass per decision
            flag = ""
            if lift != 0.0 and fo_sum != 0.0 and (np.sign(lift) != np.sign(fo_sum)):
                # sign-opposing interaction. Flag the "hurts UNLESS partner holds X"
                # shape when the first-order sum is negative but together it helps,
                # and one of the cells sits at the partner seat (seat 1).
                partner_involved = (c1[1] == 1) or (c2[1] == 1)
                if fo_sum < 0 and lift > 0 and partner_involved:
                    flag = "conditional_partner_rescue"
                else:
                    flag = "sign_opposing"
            rows.append(dict(
                tile1=tile_name(c1[0]), seat1=c1[1], tile2=tile_name(c2[0]), seat2=c2[1],
                lift=lift, delta1=d1, delta2=d2, fo_sum=fo_sum,
                mass=avg_mass, n_worlds=n, flag=flag,
                rank_score=abs(lift) * avg_mass,
            ))
        rows.sort(key=lambda r: r["rank_score"], reverse=True)
        return rows


# ---------------------------------------------------------------------------
# Corpus streaming driver
# ---------------------------------------------------------------------------

def _default_chunks(data_dir: Path) -> list[Path]:
    v2 = sorted(data_dir.glob("corpus_v2_train_*_d0-9.pt"))
    legacy = sorted(data_dir.glob("corpus_train_chunk_*.pt"))
    return v2 + legacy


def run_worldbank(
    adapter_path: str,
    data_dir: str,
    out_dir: str,
    report_dir: str,
    min_qualifying: int = 100,
    min_games: int = 20,
    max_interaction: int = 2000,
    wall_cap_s: float = 45 * 60,
    log_every_s: float = 30.0,
    seed: int = 0,
) -> dict:
    from gus.model.load import load_student

    t0 = time.time()
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    report_path = Path(report_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    print(f"[worldbank] loading student {adapter_path} on cpu", flush=True)
    model, _ = load_student(adapter_path, "cpu")
    model.eval()

    rng = np.random.default_rng(seed)
    chunks = _default_chunks(data_path)
    print(f"[worldbank] {len(chunks)} candidate chunks", flush=True)

    p6_rows: list[P6Row] = []
    p6_games: set = set()
    inter = InteractionAccumulator()
    inter_seen = 0  # reservoir counter over trick-0/1 decisions
    chunks_used: list[str] = []
    notes: list[str] = []
    last_log = time.time()

    for chunk in chunks:
        if time.time() - t0 > wall_cap_s:
            notes.append(f"wall cap {wall_cap_s:.0f}s hit before processing {chunk.name}")
            break
        # Stop condition: enough P6 decisions across enough games AND interaction full.
        if len(p6_rows) >= min_qualifying and len(p6_games) >= min_games \
                and inter.n_decisions >= max_interaction:
            break

        print(f"[worldbank] loading chunk {chunk.name}", flush=True)
        blob = torch.load(str(chunk), weights_only=False)
        games = blob["results"]
        seeds = blob.get("seeds", list(range(len(games))))
        chunks_used.append(chunk.name)

        for gi, game in enumerate(games):
            if len(game.decisions) != 28:
                continue
            gid = f"{chunk.stem}:g{gi}:s{seeds[gi] if gi < len(seeds) else gi}:d{int(game.decl_id)}"
            # P6 qualifying decisions
            for d_idx in range(4):
                if is_p6_qualifying(game, d_idx):
                    view = _build_decision_view(model, game, d_idx, gid)
                    if view is not None:
                        p6_rows.append(analyze_p6_decision(view))
                        p6_games.add(gid)
            # Interaction sample over trick-0/1 (reservoir up to max_interaction)
            for d_idx in range(8):
                dec = game.decisions[d_idx]
                if dec.world_hands is None or dec.world_hands.shape[0] < 30:
                    continue
                inter_seen += 1
                if inter.n_decisions < max_interaction:
                    view = _build_decision_view(model, game, d_idx, gid)
                    if view is not None:
                        inter.add(view)
                # (past the cap we simply stop adding — a deterministic prefix
                # sample; documented in notes.)

            if time.time() - last_log > log_every_s:
                print(f"[worldbank] t={time.time()-t0:6.1f}s  p6={len(p6_rows)}"
                      f" games={len(p6_games)}  inter={inter.n_decisions}"
                      f"  chunk={chunk.name} g{gi}/{len(games)}", flush=True)
                last_log = time.time()

        del blob, games
        print(f"[worldbank] chunk done {chunk.name}: p6={len(p6_rows)}"
              f" games={len(p6_games)} inter={inter.n_decisions}"
              f" t={time.time()-t0:.1f}s", flush=True)

    if len(p6_rows) < min_qualifying:
        notes.append(f"only {len(p6_rows)} qualifying decisions (< {min_qualifying}); "
                     f"exhausted {len(chunks_used)} chunks")
    if inter.n_decisions >= max_interaction and inter_seen > inter.n_decisions:
        notes.append(f"interaction sample truncated to first {inter.n_decisions} of "
                     f"{inter_seen} trick-0/1 decisions (deterministic prefix)")

    wall_s = time.time() - t0

    # --- Aggregate P6 numbers ---
    n_q = len(p6_rows)
    bim_b = sum(1 for r in p6_rows if r.bimodal_belief)
    bim_u = sum(1 for r in p6_rows if r.bimodal_uniform)
    frac_b = bim_b / n_q if n_q else float("nan")
    frac_u = bim_u / n_q if n_q else float("nan")
    ess_vals = np.array([r.ess for r in p6_rows]) if n_q else np.array([])
    ess_median = float(np.median(ess_vals)) if n_q else float("nan")

    inter_rows = inter.results(min_worlds=30)

    # --- Write CSVs ---
    _write_p6_csv(out_path / "w5_p6.csv", p6_rows)
    _write_interactions_csv(out_path / "w5_interactions.csv", inter_rows)

    top_cells = [
        dict(tile1=r["tile1"], seat1=SEAT_WORDS[r["seat1"]], tile2=r["tile2"],
             seat2=SEAT_WORDS[r["seat2"]], lift=round(r["lift"], 3),
             mass=round(r["mass"], 4), n_worlds=r["n_worlds"], flag=r["flag"])
        for r in inter_rows[:15]
    ]

    summary = dict(
        qualifying_n=n_q,
        n_games=len(p6_games),
        bimodal_frac_belief=frac_b,
        bimodal_frac_uniform=frac_u,
        bimodal_count_belief=bim_b,
        bimodal_count_uniform=bim_u,
        ess_median=ess_median,
        ess_p25=float(np.percentile(ess_vals, 25)) if n_q else float("nan"),
        ess_p75=float(np.percentile(ess_vals, 75)) if n_q else float("nan"),
        interaction_n_decisions=inter.n_decisions,
        interaction_top_cells=top_cells,
        chunks_used=chunks_used,
        adapter=adapter_path,
        wall_s=round(wall_s, 1),
        notes=notes,
    )
    with open(out_path / "w5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Ledger cards + report ---
    _write_ledger_cards(report_path / "w5_ledger_cards.md", p6_rows, model_hint=adapter_path)
    _write_report(report_path / "w5_worldbank.md", summary, p6_rows, inter_rows,
                  ess_vals)

    print(f"[worldbank] DONE n_q={n_q} frac_belief={frac_b:.3f} "
          f"frac_uniform={frac_u:.3f} ess_med={ess_median:.1f} wall={wall_s:.1f}s",
          flush=True)
    return summary


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_p6_csv(path: Path, rows: list[P6Row]):
    import csv
    cols = ["game_id", "decl_id", "actor", "d_idx", "n_worlds", "ess", "a_star",
            "action_taken", "n_clusters_belief", "bimodal_belief", "gap_belief",
            "low_mass_belief", "high_mass_belief", "n_clusters_uniform",
            "bimodal_uniform", "gap_uniform", "low_mass_uniform", "high_mass_uniform"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.game_id, r.decl_id, r.actor, r.d_idx, r.n_worlds,
                        f"{r.ess:.3f}", r.a_star, r.action_taken,
                        r.n_clusters_belief, int(r.bimodal_belief), f"{r.gap_belief:.3f}",
                        f"{r.low_mass_belief:.4f}", f"{r.high_mass_belief:.4f}",
                        r.n_clusters_uniform, int(r.bimodal_uniform), f"{r.gap_uniform:.3f}",
                        f"{r.low_mass_uniform:.4f}", f"{r.high_mass_uniform:.4f}"])


def _write_interactions_csv(path: Path, rows: list[dict]):
    import csv
    cols = ["rank", "tile1", "seat1", "tile2", "seat2", "lift", "delta1", "delta2",
            "fo_sum", "mass", "n_worlds", "rank_score", "flag"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, r in enumerate(rows[:50], 1):
            w.writerow([i, r["tile1"], SEAT_WORDS[r["seat1"]], r["tile2"],
                        SEAT_WORDS[r["seat2"]], f"{r['lift']:.4f}", f"{r['delta1']:.4f}",
                        f"{r['delta2']:.4f}", f"{r['fo_sum']:.4f}", f"{r['mass']:.5f}",
                        r["n_worlds"], f"{r['rank_score']:.5f}", r["flag"]])


def _cluster_context_words(key: tuple) -> str:
    if key == ("other",):
        return "other (pooled minor contexts)"
    parts = []
    for tile, code in zip(OTHER_COUNT_TILES, key):
        parts.append(f"{tile_name(tile)}={SEAT_WORDS.get(int(code), '?')}")
    return ", ".join(parts)


def _write_ledger_cards(path: Path, rows: list[P6Row], model_hint: str, top: int = 10):
    bimodal = [r for r in rows if r.bimodal_belief]
    # highest-mass = most belief mass explained by the two modes (low+high)
    bimodal.sort(key=lambda r: (r.low_mass_belief + r.high_mass_belief, r.gap_belief),
                 reverse=True)
    cards = bimodal[:top]
    lines: list[str] = []
    lines.append("# Otis W5 — 3-2 fate ledger cards")
    lines.append("")
    lines.append(f"The {len(cards)} highest-mass bimodal P6 decisions (belief-weighted), "
                 f"from the world-bank instrument (`otis/analysis/worldbank.py`). Belief "
                 f"posterior: `{model_hint}`. Each card shows the public context, the "
                 f"context clusters of sampled worlds, and why the 3-2's fate value "
                 f"(oracle Q at a* = argmax E[Q]) differs across contexts.")
    lines.append("")
    if not cards:
        lines.append("_No bimodal P6 decisions found._")
        path.write_text("\n".join(lines))
        return
    for i, r in enumerate(cards, 1):
        lines.append(f"## Card {i} — {r.game_id}")
        lines.append("")
        lines.append(f"- Declaration id: {r.decl_id} · trick {r.d_idx // 4} "
                     f"(decision idx {r.d_idx}) · actor seat {r.actor}")
        lines.append(f"- Worlds M={r.n_worlds} · ESS={r.ess:.1f} · a*=slot {r.a_star} "
                     f"· action taken=slot {r.action_taken}")
        lines.append(f"- Bimodal split: low mass {r.low_mass_belief:.0%} / high mass "
                     f"{r.high_mass_belief:.0%} · gap {r.gap_belief:.1f} points")
        lines.append("")
        lines.append("| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |")
        lines.append("|---|---|---|---|---|---|")
        for c in sorted(r.clusters, key=lambda c: c.mean_belief):
            lines.append(f"| {_cluster_context_words(c.key)} | {c.mass_belief:.1%} | "
                         f"{c.mass_uniform:.1%} | {c.mean_belief:.1f} | {c.q_min:.1f} | "
                         f"{c.q_max:.1f} |")
        lines.append("")
        low = sorted(r.clusters, key=lambda c: c.mean_belief)[0]
        high = sorted(r.clusters, key=lambda c: c.mean_belief)[-1]
        lines.append(f"> Story: the 3-2 line is worth ~{low.mean_belief:.0f} pts when "
                     f"[{_cluster_context_words(low.key)}] but ~{high.mean_belief:.0f} pts "
                     f"when [{_cluster_context_words(high.key)}] — a "
                     f"{high.mean_belief - low.mean_belief:.0f}-point swing the "
                     f"belief-averaged marginal cannot see.")
        lines.append("")
    path.write_text("\n".join(lines))


def _write_report(path: Path, summary: dict, p6_rows: list[P6Row],
                  inter_rows: list[dict], ess_vals: np.ndarray):
    L: list[str] = []
    L.append("# Otis W5 — world-bank analysis (P6 + interaction structure)")
    L.append("")
    L.append("Playout-free rung of issue #49 (items 1+2, prediction P6). Everything "
             "below derives from stored `world_hands` [M,3,7] and `q_per_world` [M,7] "
             "in the eq corpus — no new game playouts. Instrument: "
             "`otis/analysis/worldbank.py`. CPU only.")
    L.append("")
    L.append("## Method")
    L.append("")
    L.append(f"- **Belief posterior**: gus student `{summary['adapter']}` "
             "(voids transformer, belief head world-independent). For each decision "
             "`belief_logits[28,3]` -> `log P(rel-seat | domino)`; a sampled world's "
             "weight = `prod over hidden dominoes P(assigned seat | domino)`, "
             "normalized over the decision's M worlds. ESS = 1 / sum(w_norm^2). A "
             "uniform variant (w = 1/M) runs alongside.")
    L.append("- **Relative seats**: 0=left opp, 1=partner, 2=right opp "
             "(`gus/model/features.py:86`); belief axis, `world_hands` rows, and the "
             "placement code share this ordering.")
    L.append("- **P6 qualifying decision**: trick 0, actor holds the 3-2 (id 8), "
             ">= 100 sampled worlds.")
    L.append("- **Context vector**: joint placement of the other four count tiles "
             "(5-5, 6-4, 5-0, 4-1) over {left opp, partner, right opp, your hand}. "
             "Worlds clustered by exact context; clusters < 2% belief mass folded "
             "into `other`.")
    L.append("- **Cluster value**: belief-weighted mean of `q_per_world[:, a*]`, "
             "a* = argmax E[Q] over legal actions.")
    L.append("- **Bimodality**: clusters sorted by mean; among splits where both "
             "low/high groups carry >= 20% belief mass, take the one with the largest "
             "group-mean gap; bimodal iff that gap >= 10 points. (Existence reading of "
             "the P6 band; falls back to global-max-gap split when no split is "
             "mass-valid, which then reads as not-bimodal.)")
    L.append("")
    L.append("## P6 result")
    L.append("")
    band = "PASS (>=30%)" if (summary["bimodal_frac_belief"] >= 0.30) else \
           ("FALSIFIER (<10%)" if summary["bimodal_frac_belief"] < 0.10 else "MID (10-30%)")
    L.append(f"- Qualifying decisions N = **{summary['qualifying_n']}** across "
             f"**{summary['n_games']}** games.")
    L.append(f"- **Bimodal fraction (belief-weighted) = {summary['bimodal_frac_belief']:.1%}** "
             f"({summary['bimodal_count_belief']}/{summary['qualifying_n']}) -> **{band}** "
             "against the registered P6 band (>=30% PASS, <10% falsifier).")
    L.append(f"- Bimodal fraction (uniform weights) = {summary['bimodal_frac_uniform']:.1%} "
             f"({summary['bimodal_count_uniform']}/{summary['qualifying_n']}).")
    L.append(f"- ESS distribution: median **{summary['ess_median']:.1f}**, "
             f"p25 {summary['ess_p25']:.1f}, p75 {summary['ess_p75']:.1f} "
             f"(of M={p6_rows[0].n_worlds if p6_rows else '?'} worlds).")
    L.append("")
    L.append("## Interaction structure (issue item 1)")
    L.append("")
    L.append(f"First-order delta(d@s) and pairwise lift(d1@s1, d2@s2) over the five "
             f"count tiles at relative seats, per-decision centered and pooled over "
             f"{summary['interaction_n_decisions']} trick-0/1 decisions. Cells require "
             f">= 30 pooled worlds. Full top-50 in `w5_interactions.csv`.")
    L.append("")
    n_flag = sum(1 for r in inter_rows if r["flag"] == "conditional_partner_rescue")
    L.append(f"- Sign-opposing 'count tile hurts UNLESS partner holds X' cells flagged: "
             f"**{n_flag}** (of {len(inter_rows)} qualifying pairwise cells).")
    L.append("")
    L.append("Top 15 cells by |lift| x mass:")
    L.append("")
    L.append("| # | tile1 @ seat | tile2 @ seat | lift | delta1 | delta2 | mass | n | flag |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(inter_rows[:15], 1):
        L.append(f"| {i} | {r['tile1']} @ {SEAT_WORDS[r['seat1']]} | "
                 f"{r['tile2']} @ {SEAT_WORDS[r['seat2']]} | {r['lift']:.2f} | "
                 f"{r['delta1']:.2f} | {r['delta2']:.2f} | {r['mass']:.4f} | "
                 f"{r['n_worlds']} | {r['flag'] or '-'} |")
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append(f"- Chunks streamed: {', '.join(summary['chunks_used'])}")
    L.append(f"- Wall time: {summary['wall_s']}s")
    if summary["notes"]:
        L.append("- Notes:")
        for n in summary["notes"]:
            L.append(f"  - {n}")
    L.append("")
    L.append("Ledger cards for the 10 highest-mass bimodal decisions: "
             "`otis/reports/w5_ledger_cards.md`. Machine summary: "
             "`scratch/otis-night/w5_summary.json`.")
    path.write_text("\n".join(L))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Otis world-bank analysis (P6 + interactions)")
    ap.add_argument("--adapter", default="gus/adapters/v3_consistency_10000g.pt")
    ap.add_argument("--data-dir", default="gus/data")
    ap.add_argument("--out-dir", default="scratch/otis-night")
    ap.add_argument("--report-dir", default="otis/reports")
    ap.add_argument("--min-qualifying", type=int, default=100)
    ap.add_argument("--min-games", type=int, default=20)
    ap.add_argument("--max-interaction", type=int, default=2000)
    ap.add_argument("--wall-cap-s", type=float, default=45 * 60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_worldbank(
        adapter_path=args.adapter, data_dir=args.data_dir, out_dir=args.out_dir,
        report_dir=args.report_dir, min_qualifying=args.min_qualifying,
        min_games=args.min_games, max_interaction=args.max_interaction,
        wall_cap_s=args.wall_cap_s, seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
