"""hoyt/profiles.py — frozen-profile objects + the one net call.

Two profile kinds per CONTRACTS.md, both over the domain of reachable
(seat, hand_mask, public node) info sets:

- **SigmaTable** — deterministic, built ONLY by `compile_sigma` (jud via
  FieldOracle: the single net-touching call) or `compile_rule_sigma`
  (net-free rules, for toys). Because a deterministic profile makes the
  whole hero-full-width reachable tree a fixed object, the table stores the
  mapping POSITIONALLY — per-wave move arrays aligned with the walk's slot
  order — plus the walk's captured structure (waves, leaf partition). That
  cache is what makes `br_solve` a backward-pass-only operation: payoffs
  AND belief weights can be swapped per re-solve for free (weights are
  deliberately not part of the table's identity). The table is therefore
  domain-bound to the hero=root.me walk over the exact world set it was
  compiled for (digest-checked); it cannot price a hidden seat's BR.

- **StochasticProfile** — (seat, hand_mask, node) → probability vector
  over legal moves, node identity = 128-bit rolling path hash
  (`subgame.path_hash`). Content-keyed, so it works across walks: BR of
  any hero seat, CFR average-strategy export, `profile_value`. Missing
  entries: single-legal decisions are always forced (prob 1); otherwise
  `uniform_fallback=True` plays uniform-over-legal (an empty table with
  the flag IS the uniform stress profile, with a fully vectorized path).

Zero torch at module level; `compile_sigma` lazily imports walt.field
(FieldOracle featurization constants + NodeCtx) at call time only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hoyt.subgame import (
    _ALL28,
    _AR28,
    Subgame,
    build_subgame,
    path_hash,
    run_engine,
)


def _bits_list(mask: int) -> list[int]:
    out = []
    m = int(mask)
    while m:
        b = m & -m
        out.append(b.bit_length() - 1)
        m ^= b
    return out


# --------------------------------------------------------------------- #
#  SigmaTable                                                            #
# --------------------------------------------------------------------- #

@dataclass
class SigmaTable:
    """Deterministic profile + captured hero-full-width walk structure."""

    root_key: tuple
    worlds_digest: str
    p0: int
    n_worlds: int
    # positional mapping: per wave, the profile move of every non-hero slot
    # (slot order of the canonical walk); dec[w] = (node, hand, seat, move)
    # unique-decision arrays — the CONTRACTS mapping, for audit/consumers.
    mv: list
    dec: list
    # captured structure (compact dtypes): waves for the backward pass,
    # leaf partition for weight->value evaluation.
    waves: list
    leaf_node: np.ndarray      # int32 [S_L]  leaf slot -> leaf node
    leaf_world: np.ndarray     # int32 [S_L]  leaf slot -> root world index
    leaf_pts: np.ndarray       # int16 [M_L]  declaring points per leaf node
    n_leaf_nodes: int
    n_nodes: int
    meta: dict = field(default_factory=dict)

    @property
    def n_infosets(self) -> int:
        """Unique (seat, hand, node) decisions captured (contract domain)."""
        return sum(len(d[0]) for d in self.dec if d is not None)


def _table_from_run(sub: Subgame, res: dict, meta: dict) -> SigmaTable:
    waves_c = []
    for w in res["waves"]:
        waves_c.append({
            "parent": w["parent"].astype(np.int32),
            "tile": w["tile"].astype(np.int8),
            "pseat": w["pseat"].astype(np.int8),
            "hero": w["hero"],
        })
    leaf = res["leaf"]
    dec_c = []
    for d in res["cap"]["dec"]:
        if d is None:
            dec_c.append(None)
        else:
            n, h, s, m = d
            dec_c.append((n.astype(np.int32), h.astype(np.uint32),
                          s.astype(np.int8), m.astype(np.int8)))
    return SigmaTable(
        root_key=sub.root_key,
        worlds_digest=sub.worlds_digest,
        p0=sub.p0,
        n_worlds=int(sub.worlds.shape[0]),
        mv=res["cap"]["mv"],
        dec=dec_c,
        waves=waves_c,
        leaf_node=leaf["snode"].astype(np.int32),
        leaf_world=leaf["sworld"].astype(np.int32),
        leaf_pts=leaf["ptsvd"].astype(np.int16),
        n_leaf_nodes=int(leaf["M"]),
        n_nodes=int(res["n_nodes"]),
        meta=meta,
    )


def compile_sigma(root, worlds, oracle) -> SigmaTable:
    """THE net-touching call: walk the hero-full-width × σ-partitioned
    reachable set (exactly walt.solver's wavefront shape), capturing jud's
    decision at every unique reachable (seat, hand_mask, node). Batched
    through FieldOracle.ev_rows — one chunked forward per wave, one net
    decision per unique info set. Everything downstream is net-free."""
    from walt.field import NodeCtx  # lazy: keeps kernel imports torch-free

    worlds = np.asarray(worlds, dtype=np.uint32).reshape(-1, 3)
    sub = build_subgame(root, worlds,
                        np.ones(worlds.shape[0], dtype=np.float64))
    ctx0 = NodeCtx.from_history(sub.hist0, sub.bidder, sub.luts)
    prov = _NetSigmaProvider(oracle, sub, ctx0.blocks)
    rows0, fwd0 = oracle.n_rows, oracle.n_forward
    res = run_engine(sub, prov, hero=sub.me, capture=True,
                     net_state={"ptsf0": ctx0.pts, "played0": ctx0.played})
    meta = {"net_rows": oracle.n_rows - rows0,
            "net_forwards": oracle.n_forward - fwd0}
    return _table_from_run(sub, res, meta)


def compile_rule_sigma(root, worlds, rule) -> SigmaTable:
    """Net-free SigmaTable from a deterministic rule
    ``rule(seat, hand_mask, legal_mask, decl_id) -> domino id`` (must be a
    bit of legal_mask). Toy/test twin of compile_sigma."""
    worlds = np.asarray(worlds, dtype=np.uint32).reshape(-1, 3)
    sub = build_subgame(root, worlds,
                        np.ones(worlds.shape[0], dtype=np.float64))
    prov = _RuleProvider(rule)
    res = run_engine(sub, prov, hero=sub.me, capture=True)
    return _table_from_run(sub, res, {"rule": getattr(rule, "__name__", "?")})


# --------------------------------------------------------------------- #
#  StochasticProfile                                                     #
# --------------------------------------------------------------------- #

# 64-bit odd mixing constants folding (seat, hand) and the second hash
# lane into one sortable key. Collisions are handled by exact (k0, h1, h2)
# verification at every lookup — the mix only has to be a good router.
_MIX_K0 = np.uint64(0xD6E8FEB86659FD93)
_MIX_H2 = np.uint64(0xCA5A826395121157)


class _Cols:
    """Frozen columnar profile: entries sorted by a mixed 64-bit key with
    exact key verification. moves/probs are flat with per-entry offsets;
    zero-probability moves are dropped at freeze time (dict-path parity:
    lookups there filtered probs > 0 per query)."""

    __slots__ = ("key", "k0", "h1", "h2", "off", "cnt", "mv", "pr",
                 "mvmask", "n")

    def __init__(self, k0, h1, h2, moff, mv, pr):
        n = len(k0)
        if n == 0:
            self.key = np.empty(0, dtype=np.uint64)
            self.k0 = np.empty(0, dtype=np.int64)
            self.h1 = self.h2 = self.key
            self.off = self.cnt = self.mvmask = np.empty(0, dtype=np.int64)
            self.mv = np.empty(0, dtype=np.int8)
            self.pr = np.empty(0, dtype=np.float64)
            self.n = 0
            return
        # drop zero-probability moves (dict-path parity: lookups there
        # filtered probs > 0 per query), still in original entry order
        keep = pr > 0.0
        cnt_k = np.add.reduceat(keep.astype(np.int64), moff[:-1])
        mv_k = mv[keep]
        pr_k = pr[keep]
        koff = np.concatenate(([0], np.cumsum(cnt_k)))[:-1]
        # sort entries by the mixed key
        key = h1 ^ (k0.astype(np.uint64) * _MIX_K0) ^ (h2 * _MIX_H2)
        perm = np.argsort(key, kind="stable")
        self.key = key[perm]
        self.k0 = k0[perm]
        self.h1 = h1[perm]
        self.h2 = h2[perm]
        cnt = cnt_k[perm]
        off = np.concatenate(([0], np.cumsum(cnt)))[:-1]
        total = int(cnt.sum())
        ramp = np.arange(total, dtype=np.int64) - np.repeat(off, cnt)
        src = np.repeat(koff[perm], cnt) + ramp
        self.mv = mv_k[src].astype(np.int8)
        self.pr = pr_k[src]
        self.off = off
        self.cnt = cnt
        self.mvmask = np.bitwise_or.reduceat(
            np.int64(1) << self.mv.astype(np.int64), off) \
            if total else np.zeros(n, dtype=np.int64)
        self.n = n

    def lookup(self, qk0, qh1, qh2):
        """Vectorized exact lookup. Returns (found bool[D], idx int64[D])
        with idx valid only where found."""
        if self.n == 0:
            return np.zeros(len(qk0), dtype=bool), \
                np.zeros(len(qk0), dtype=np.int64)
        qkey = qh1 ^ (qk0.astype(np.uint64) * _MIX_K0) ^ (qh2 * _MIX_H2)
        lo = np.searchsorted(self.key, qkey, side="left")
        hi = np.searchsorted(self.key, qkey, side="right")
        idx = np.minimum(lo, self.n - 1)
        run = hi - lo
        found = (run == 1) & (self.k0[idx] == qk0) & \
            (self.h1[idx] == qh1) & (self.h2[idx] == qh2)
        multi = np.flatnonzero(run > 1)
        for qi in multi.tolist():           # mixed-key collision: rare scan
            for j in range(int(lo[qi]), int(hi[qi])):
                if (self.k0[j] == qk0[qi] and self.h1[j] == qh1[qi]
                        and self.h2[j] == qh2[qi]):
                    idx[qi] = j
                    found[qi] = True
                    break
        return found, idx


class StochasticProfile:
    """(seat, hand_mask, node) -> probability vector over legal moves.

    Node keys are 128-bit rolling path hashes of the public tile path from
    the subgame root (`subgame.path_hash`); `set`/`get` take the path tuple
    and hash internally, so CFR exporters never touch hash internals.
    Single-legal decisions never need an entry (forced, prob 1).

    Two build paths, one frozen form: `set` stages python-dict entries
    (toys, reference-compatible export); `set_bulk` loads the whole CFR
    export as columnar arrays in one call (no python objects per entry).
    Consumers read through `_frozen()` — vectorized sorted-key lookup."""

    def __init__(self, uniform_fallback: bool = False):
        self.table: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self.uniform_fallback = bool(uniform_fallback)
        self._cols: _Cols | None = None

    def set(self, seat, hand_mask, path, moves, probs) -> None:
        if self._cols is not None:
            raise ValueError("profile was bulk-loaded; set() would shadow")
        moves = np.asarray(moves, dtype=np.int64).reshape(-1)
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        if moves.shape != probs.shape:
            raise ValueError("moves/probs length mismatch")
        h1, h2 = path_hash(path)
        self.table[(int(seat), int(hand_mask), h1, h2)] = (moves, probs)

    def set_bulk(self, seat, hand_mask, h1, h2, moff, moves, probs) -> None:
        """One-call columnar load: entry i is (seat[i], hand_mask[i],
        h1[i], h2[i]) with moves/probs[moff[i]:moff[i+1]] ascending.
        Probabilities must sum to 1 per entry; zero-prob moves are dropped.
        The profile must be empty (bulk load replaces, never merges)."""
        if self.table or self._cols is not None:
            raise ValueError("set_bulk on a non-empty profile")
        seat = np.asarray(seat, dtype=np.int64).reshape(-1)
        hand_mask = np.asarray(hand_mask, dtype=np.int64).reshape(-1)
        h1 = np.asarray(h1, dtype=np.uint64).reshape(-1)
        h2 = np.asarray(h2, dtype=np.uint64).reshape(-1)
        moff = np.asarray(moff, dtype=np.int64).reshape(-1)
        moves = np.asarray(moves, dtype=np.int64).reshape(-1)
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        if len(moff) != len(seat) + 1 or int(moff[-1]) != len(moves) \
                or len(moves) != len(probs):
            raise ValueError("set_bulk: misaligned columnar arrays")
        if len(seat):
            segsum = np.add.reduceat(probs, moff[:-1])
            bad = np.abs(segsum - 1.0) > 1e-9
            if bad.any():
                i = int(np.flatnonzero(bad)[0])
                raise ValueError(
                    f"set_bulk: probs sum {segsum[i]} != 1 at entry {i}")
        k0 = (seat << 28) | hand_mask
        self._cols = _Cols(k0, h1, h2, moff, moves, probs)

    def _frozen(self) -> _Cols:
        """The columnar view, built from staged dict entries on demand."""
        if self._cols is None:
            items = self.table.items()
            E = len(self.table)
            k0 = np.empty(E, dtype=np.int64)
            h1 = np.empty(E, dtype=np.uint64)
            h2 = np.empty(E, dtype=np.uint64)
            cnt = np.empty(E, dtype=np.int64)
            mv_parts, pr_parts = [], []
            for i, ((s, h, a, b), (mvs, prs)) in enumerate(items):
                k0[i] = (s << 28) | h
                h1[i] = a
                h2[i] = b
                cnt[i] = len(mvs)
                mv_parts.append(mvs)
                pr_parts.append(prs)
            moff = np.concatenate(([0], np.cumsum(cnt)))
            mv = np.concatenate(mv_parts) if E else np.empty(0, np.int64)
            pr = np.concatenate(pr_parts) if E else np.empty(0, np.float64)
            self._cols = _Cols(k0, h1, h2, moff, mv, pr)
        return self._cols

    def get(self, seat, hand_mask, path):
        h1, h2 = path_hash(path)
        got = self.table.get((int(seat), int(hand_mask), h1, h2))
        if got is not None or self._cols is None:
            return got
        c = self._cols
        q0 = np.array([(int(seat) << 28) | int(hand_mask)], dtype=np.int64)
        found, idx = c.lookup(q0, np.array([h1], dtype=np.uint64),
                              np.array([h2], dtype=np.uint64))
        if not found[0]:
            return None
        i = int(idx[0])
        sl = slice(int(c.off[i]), int(c.off[i]) + int(c.cnt[i]))
        return c.mv[sl].astype(np.int64), c.pr[sl].copy()

    def __len__(self) -> int:
        return len(self.table) if self._cols is None else self._cols.n


# --------------------------------------------------------------------- #
#  Providers (run_engine σ-step plugins)                                 #
# --------------------------------------------------------------------- #

class _TableProvider:
    """Replay a SigmaTable positionally (rewalk / self-consistency path)."""

    def __init__(self, table: SigmaTable):
        self.table = table

    def expand(self, eng, p, pos, sel, nsl, seats, hands):
        mv = self.table.mv[p - self.table.p0]
        if len(mv) != len(sel):
            raise ValueError(
                f"SigmaTable misaligned at p={p}: {len(mv)} stored moves "
                f"vs {len(sel)} profile slots — table compiled for a "
                "different (root, worlds)?")
        return sel, mv.astype(np.int64), None, None


class _RuleProvider:
    """Deterministic rule σ, deduped per unique (node, hand)."""

    def __init__(self, rule):
        self.rule = rule

    def expand(self, eng, p, pos, sel, nsl, seats, hands):
        sub = eng.sub
        uq, inv = np.unique((nsl << 28) | hands, return_inverse=True)
        u_n = uq >> 28
        u_h = uq & _ALL28
        ls = eng.led[u_n]
        fb = np.where(ls >= 0, sub.CFB[ls], _ALL28)
        lm = u_h & fb
        lm = np.where(lm != 0, lm, u_h)
        seats_u = eng.actor[u_n]
        mv_dec = np.empty(len(uq), dtype=np.int64)
        s_l, h_l, lm_l = seats_u.tolist(), u_h.tolist(), lm.tolist()
        for i in range(len(uq)):
            m = int(self.rule(s_l[i], h_l[i], lm_l[i], sub.decl))
            if not (lm_l[i] >> m) & 1:
                raise ValueError(f"rule returned illegal move {m}")
            mv_dec[i] = m
        return sel, mv_dec[inv], None, (u_n, u_h, seats_u, mv_dec)


class _UniformProvider:
    """Uniform-over-legal for every profile slot — the expectimax stress
    profile, fully vectorized (no dict, no path ids)."""

    def expand(self, eng, p, pos, sel, nsl, seats, hands):
        sub = eng.sub
        ls = eng.led[nsl]
        fb = np.where(ls >= 0, sub.CFB[ls], _ALL28)
        lm = hands & fb
        lm = np.where(lm != 0, lm, hands)
        bm = ((lm[:, None] >> _AR28) & 1).astype(bool)
        si, mv = np.nonzero(bm)                 # ascending (slot, move)
        cnt = np.bitwise_count(lm.astype(np.uint64)).astype(np.float64)
        return sel[si], mv.astype(np.int64), 1.0 / cnt[si], None


class _FullWidthProvider:
    """All legal moves per slot, weights unscaled — the CFR structural
    walk (reachability only; profile probabilities are the CFR lane's)."""

    def expand(self, eng, p, pos, sel, nsl, seats, hands):
        sub = eng.sub
        ls = eng.led[nsl]
        fb = np.where(ls >= 0, sub.CFB[ls], _ALL28)
        lm = hands & fb
        lm = np.where(lm != 0, lm, hands)
        bm = ((lm[:, None] >> _AR28) & 1).astype(bool)
        si, mv = np.nonzero(bm)
        return sel[si], mv.astype(np.int64), None, None


class _DictProfileProvider:
    """StochasticProfile lookups, deduped per unique (node, hand), fully
    vectorized through the profile's frozen columnar view. Per unique
    decision the emitted (moves, probs) composition matches the historic
    dict loop exactly: forced -> single legal; table hit -> stored entry
    (zero-prob moves dropped at freeze); miss -> uniform-over-legal when
    uniform_fallback, else KeyError."""

    def __init__(self, profile: StochasticProfile):
        self.profile = profile
        self.cols = profile._frozen()

    def expand(self, eng, p, pos, sel, nsl, seats, hands):
        sub = eng.sub
        uq, first, inv = np.unique(
            (nsl << 28) | hands, return_index=True, return_inverse=True)
        u_n = uq >> 28
        u_h = uq & _ALL28
        ls = eng.led[u_n]
        fb = np.where(ls >= 0, sub.CFB[ls], _ALL28)
        lm = u_h & fb
        lm = np.where(lm != 0, lm, u_h)
        seats_u = seats[first]
        D = len(uq)
        c = self.cols

        forced = (lm & (lm - 1)) == 0
        found = np.zeros(D, dtype=bool)
        idx = np.zeros(D, dtype=np.int64)
        nf = np.flatnonzero(~forced)
        if len(nf) and c.n:
            fnd, ix = c.lookup((seats_u[nf] << 28) | u_h[nf],
                               eng.ph1[u_n[nf]], eng.ph2[u_n[nf]])
            found[nf] = fnd
            idx[nf] = ix
            bad = fnd & ((c.mvmask[ix] & ~lm[nf]) != 0)
            if bad.any():
                i = int(nf[np.flatnonzero(bad)[0]])
                raise ValueError(
                    f"profile plays illegal move at seat={int(seats_u[i])} "
                    f"hand={int(u_h[i]):#x} at p={p}")
        miss = ~forced & ~found
        if miss.any() and not self.profile.uniform_fallback:
            i = int(np.flatnonzero(miss)[0])
            raise KeyError(
                f"profile has no entry for seat={int(seats_u[i])} "
                f"hand={int(u_h[i]):#x} at p={p} and "
                "uniform_fallback is off")

        nleg = np.bitwise_count(lm.astype(np.uint64)).astype(np.int64)
        lens = np.where(forced, 1, np.where(found, c.cnt[idx], nleg))
        starts_u = np.zeros(D, dtype=np.int64)
        np.cumsum(lens[:-1], out=starts_u[1:])
        total_u = int(lens.sum())
        mv_u = np.empty(total_u, dtype=np.int64)
        pr_u = np.empty(total_u, dtype=np.float64)

        f = np.flatnonzero(forced)
        if len(f):
            mv_u[starts_u[f]] = np.bitwise_count(
                (lm[f] - 1).astype(np.uint64)).astype(np.int64)
            pr_u[starts_u[f]] = 1.0
        fd = np.flatnonzero(found)
        if len(fd):
            cf = c.cnt[idx[fd]]
            tf = int(cf.sum())
            ramp = np.arange(tf, dtype=np.int64) - np.repeat(
                np.concatenate(([0], np.cumsum(cf)))[:-1], cf)
            src = np.repeat(c.off[idx[fd]], cf) + ramp
            dst = np.repeat(starts_u[fd], cf) + ramp
            mv_u[dst] = c.mv[src]
            pr_u[dst] = c.pr[src]
        mi = np.flatnonzero(miss)
        if len(mi):
            bmm = ((lm[mi, None] >> _AR28) & 1).astype(bool)
            _, tt = np.nonzero(bmm)             # row-major: ascending/row
            cb = nleg[mi]
            ramp = np.arange(len(tt), dtype=np.int64) - np.repeat(
                np.concatenate(([0], np.cumsum(cb)))[:-1], cb)
            dst = np.repeat(starts_u[mi], cb) + ramp
            mv_u[dst] = tt
            pr_u[dst] = np.repeat(1.0 / cb, cb)

        cnt_slot = lens[inv]
        tot = int(cnt_slot.sum())
        rep = np.repeat(sel, cnt_slot)
        csl = np.zeros(len(sel), dtype=np.int64)
        np.cumsum(cnt_slot[:-1], out=csl[1:])
        posn = np.repeat(starts_u[inv], cnt_slot) \
            + (np.arange(tot) - np.repeat(csl, cnt_slot))
        return rep, mv_u[posn], pr_u[posn], None


class _NetSigmaProvider:
    """jud argmax σ — verbatim port of walt.solver's per-wave decision
    block (dedup, one-legal shortcut, vectorized feature scatters, ONE
    chunked ev_rows forward, sign-argmax with first-max ties). Only used
    by compile_sigma; carries the walk's net-state (blocks path replay)."""

    # jud featurization layout (walt.field mirrors champion.jud_net)
    _F91 = 91
    _G = 252
    _FEAT = 350
    _NT = 7

    def __init__(self, oracle, sub: Subgame, root_blocks):
        self.oracle = oracle
        self.sub = sub
        self.root_blocks = root_blocks          # (4, 259) float32
        self.h91_local: dict[tuple, np.ndarray] = {}

    def expand(self, eng, p, pos, sel, nsl, seats, hands):
        sub = self.sub
        RANK, CNT = sub.RANK, sub.CNT
        bid_team = sub.bid_team
        uq, inv = np.unique((nsl << 28) | hands, return_inverse=True)
        u_n = uq >> 28
        u_h = uq & _ALL28
        ls = eng.led[u_n]
        fb = np.where(ls >= 0, sub.CFB[ls], _ALL28)
        lm = u_h & fb
        lm = np.where(lm != 0, lm, u_h)
        bm = ((lm[:, None] >> _AR28) & 1).astype(bool)
        ncand = bm.sum(axis=1)
        mv_dec = np.empty(len(uq), dtype=np.int64)
        one = ncand == 1
        if one.any():
            mv_dec[one] = np.argmax(bm[one], axis=1)
        multi = np.flatnonzero(~one)

        if len(multi):
            actor = eng.actor
            waves = eng.waves
            m_n = u_n[multi]
            m_h = u_h[multi]
            m_act = actor[m_n]
            bn, binv = np.unique(m_n, return_inverse=True)
            U = len(bn)
            povs = actor[bn]
            blocks = self.root_blocks[povs].copy()      # (U, 259)
            arU = np.arange(U)
            anc = bn
            for j in range(eng.cw, 0, -1):
                wj = waves[j]
                tj = wj["tile"][anc].astype(np.int64)
                sj = wj["pseat"][anc].astype(np.int64)
                k = eng.p0 + j - 1
                base = 9 * tj
                blocks[arU, base + (sj - povs) % 4] = 1.0
                blocks[arU, base + 4 + (k & 3)] = 1.0
                blocks[arU, base + 8] = np.float32((k // 4) / (self._NT - 1))
                anc = wj["parent"][anc]
            blocks[:, self._G + 3:self._G + 7] = 0.0
            blocks[:, self._G] = (eng.ptsfd[bn] / 42.0).astype(np.float32)
            blocks[:, self._G + 1] = (eng.ptsff[bn] / 42.0).astype(np.float32)

            # child-invariant 91-dim head per decision (cached)
            D2 = len(multi)
            H91 = np.empty((D2, self._F91), dtype=np.float32)
            orig_l = (m_h | eng.played[m_n, m_act]).tolist()
            act_l = m_act.tolist()
            for i in range(D2):
                hk = (act_l[i], orig_l[i])
                row = self.h91_local.get(hk)
                if row is None:
                    row = self.oracle._h91(hk[0], hk[1], sub.bids,
                                           sub.bidder, sub.dealer, sub.decl)
                    self.h91_local[hk] = row
                H91[i] = row

            bm2 = bm[multi]
            di, mv2 = np.nonzero(bm2)           # ascending (decision, move)
            R = len(di)
            starts = np.zeros(D2, dtype=np.int64)
            np.cumsum(ncand[multi][:-1], out=starts[1:])
            X = np.empty((R, self._FEAT), dtype=np.float32)
            X[:, :self._F91] = H91[di]
            X[:, self._F91:] = blocks[binv[di]]
            arR = np.arange(R)
            mb = self._F91 + 9 * mv2
            X[arR, mb] = 1.0                    # mover is POV
            X[arR, mb + 4 + pos] = 1.0
            X[arR, mb + 8] = np.float32((p // 4) / (self._NT - 1))
            X[:, self._F91 + self._G + 2] = np.float32((p + 1) / 28.0)
            X[:, self._F91 + self._G + 3 + ((p + 1) & 3)] = 1.0
            if pos == 3:                        # move completes a trick
                nrow = m_n[di]
                r_m = RANK[eng.led[nrow], mv2]
                win = np.where(r_m > eng.brank[nrow], m_act[di],
                               eng.bseat[nrow])
                padd = eng.tcnt[nrow] + CNT[mv2] + 1
                dwin = (win % 2) == bid_team
                nd = eng.ptsfd[nrow] + np.where(dwin, padd, 0)
                nf = eng.ptsff[nrow] + np.where(dwin, 0, padd)
                X[:, self._F91 + self._G] = (nd / 42.0).astype(np.float32)
                X[:, self._F91 + self._G + 1] = (nf / 42.0).astype(np.float32)

            ev = self.oracle.ev_rows(X)
            sig = np.where(m_act % 2 == bid_team,
                           np.float32(1.0), np.float32(-1.0))
            sev = ev * sig[di]
            smax = np.maximum.reduceat(sev, starts)
            posr = np.where(sev == smax[di], arR, R)    # first-max ties
            first = np.minimum.reduceat(posr, starts)
            mv_dec[multi] = mv2[first]

        return sel, mv_dec[inv], None, (u_n, u_h, eng.actor[u_n], mv_dec)
