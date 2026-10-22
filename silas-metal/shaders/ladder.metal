// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

// The focal-horizon ladder's gather on Metal (Exact-T1's `silas::player::trick1`, built for
// the GPU by Exact-GPU, 2026-10-22). One thread per world: the world is unranked from its
// shard (the first hidden slot's hand) and its index among the second slot's hands; the
// record is replayed under the seat laws (a 0/1 likelihood); then the viewer's next k + 1
// decisions are walked, and each horizon state is valued by every lawful tail and by the
// clairvoyant bound, exactly as `Ladder::visit` does, play for play.
//
// STRATEGY-FUSION FENCE: `god` is clairvoyant (the viewer sees this one world's hands). It
// is the ladder's upper bound only, the fence page's first sanctioned use; nothing chooses
// by it.
//
// Leaves are keyed by the public history from the root, Exact-T1's `Path` bits (5-bit
// symbols, tile + 1, the first play highest; 125 bits in four words). They are summed in a
// device hash table: each key word is claimed by compare-and-swap from EMPTY, so a slot's
// words are written in order by threads whose keys agree on every earlier word, and every
// thread with one key reaches the same slot. Counts are 32-bit atomics; plays are counted
// per thread and added with a carry into 64-bit pairs.

#include <metal_stdlib>
using namespace metal;

// The problem, as words.
constant uint L_LED = 0;          // 28: the led context of each tile
constant uint L_FOLLOW = 28;      // 8: the tiles that follow each context
constant uint L_KEY = 36;         // 8 * 28: the trick key of each tile in each context
constant uint L_COUNT = 260;      // 28: count points
constant uint L_VIEWER = 288;
constant uint L_LAWS = 289;       // 4: 0 lowest, 1 highest
constant uint L_DECLARING = 293;
constant uint L_THRESHOLD = 294;
constant uint L_TOTAL = 295;
constant uint L_K = 296;
constant uint L_HANDS = 297;      // 4: the start hands (the viewer's; the rest zero)
constant uint L_TRICK = 301;      // 4
constant uint L_N = 305;
constant uint L_LEADER = 306;
constant uint L_BANKED = 307;     // 2
constant uint L_POOL = 309;
constant uint L_ALLOWED = 310;    // 3, by hidden slot
constant uint L_CAP = 313;        // 3
constant uint L_SEATS = 316;      // 3
constant uint L_POSTERIOR = 319;
constant uint L_PLAYED = 320;     // 4: each seat's plays so far
constant uint L_RECLEN = 324;
constant uint L_RECORD = 325;     // 28: seat * 32 + tile
constant uint L_BINOM = 353;      // 29 * 8: C(n, r), r < 8
constant uint L_ROOTS = 585;      // the root tiles gathered (the rest of the root's legal tiles skipped)
constant uint L_REPLY = 586;      // the replies gathered at depth 1 (the next seat's play after a root tile)

// The chunk: [threads, shards, table mask].
constant uint EMPTY = 0xffffffffu;
constant uint MAXK = 6;
constant uint ERR_TABLE = 1;
constant uint ERR_DEPTH = 2;
constant uint ERR_PATH = 3;

struct St {
    uint hands[4];
    uchar trick[4];
    uint n;
    uint leader;
    uint banked[2];
};

struct Path {
    uint w[4];  // w[3] holds bits 96..127
    uint len;
};

static inline uint actor(thread const St& s) { return (s.leader + s.n) & 3u; }

static inline uint legal(device const uint* p, uint hand, thread const St& s) {
    if (s.n == 0) return hand;
    uint f = hand & p[L_FOLLOW + p[L_LED + s.trick[0]]];
    return f == 0 ? hand : f;
}

static inline uint law_play(uint law, uint legal) {
    return law == 0 ? ctz(legal) : 31u - clz(legal);
}

// -1 undecided, 0 the viewer's side fails, 1 it has its way.
static inline int decided(device const uint* p, thread const St& s) {
    uint d = p[L_DECLARING];
    bool mine = (p[L_VIEWER] & 1u) == d;
    if (s.banked[d] >= p[L_THRESHOLD]) return mine ? 1 : 0;
    if (s.banked[1 - d] > p[L_TOTAL] - p[L_THRESHOLD]) return mine ? 0 : 1;
    return -1;
}

static inline void play(device const uint* p, thread St& s, uint tile, thread uint& work) {
    work += 1;
    uint seat = actor(s);
    s.hands[seat] &= ~(1u << tile);
    s.trick[s.n] = uchar(tile);
    s.n += 1;
    if (s.n == 4) {
        uint ctx = p[L_LED + s.trick[0]];
        uint best = 0;
        for (uint i = 1; i < 4; i++) {
            if (p[L_KEY + ctx * 28 + s.trick[i]] > p[L_KEY + ctx * 28 + s.trick[best]]) best = i;
        }
        uint winner = (s.leader + best) & 3u;
        uint pts = 1;
        for (uint i = 0; i < 4; i++) pts += p[L_COUNT + s.trick[i]];
        s.banked[winner & 1u] += pts;
        s.leader = winner;
        s.n = 0;
    }
}

static inline void push(thread Path& path, uint tile, thread uint& err) {
    if (path.len >= 25) { err = ERR_PATH; return; }
    uint v = tile + 1;
    uint sh = 5 * (24 - path.len);
    uint word = sh >> 5, off = sh & 31u;
    path.w[word] |= v << off;
    if (off > 27) path.w[word + 1] |= v >> (32 - off);
    path.len += 1;
}

// The other seats play their laws until the viewer is to play or the hand is decided.
static inline void advance(device const uint* p, thread St& s, thread Path& path, bool keep, thread uint& work,
                           thread uint& err) {
    uint viewer = p[L_VIEWER];
    while (decided(p, s) < 0 && actor(s) != viewer) {
        uint seat = actor(s);
        uint tile = law_play(p[L_LAWS + seat], legal(p, s.hands[seat], s));
        play(p, s, tile, work);
        if (keep) push(path, tile, err);
    }
}

static inline uint tail(device const uint* p, St s, uint law, thread uint& work) {
    uint viewer = p[L_VIEWER];
    while (true) {
        int v = decided(p, s);
        if (v >= 0) return uint(v);
        uint seat = actor(s);
        uint l = seat == viewer ? law : p[L_LAWS + seat];
        play(p, s, law_play(l, legal(p, s.hands[seat], s)), work);
    }
}

// STRATEGY-FUSION FENCE: the clairvoyant bound, depth first, tiles ascending, stopping at
// the first line that has its way (Exact-T1's `god`, unrolled).
static inline uint god(device const uint* p, St s, thread uint& work, thread uint& err) {
    uint viewer = p[L_VIEWER];
    Path sink = {{0, 0, 0, 0}, 0};
    advance(p, s, sink, false, work, err);
    int v = decided(p, s);
    if (v >= 0) return uint(v);
    St st[8];
    uint rem[8];
    uint sp = 0;
    st[0] = s;
    rem[0] = legal(p, s.hands[viewer], s);
    while (true) {
        if (rem[sp] == 0) {
            if (sp == 0) return 0;
            sp -= 1;
            continue;
        }
        uint tile = ctz(rem[sp]);
        rem[sp] &= rem[sp] - 1;
        St nx = st[sp];
        play(p, nx, tile, work);
        advance(p, nx, sink, false, work, err);
        int d = decided(p, nx);
        if (d == 1) return 1;
        if (d == 0) continue;
        if (sp + 1 >= 8) { err = ERR_DEPTH; return 0; }
        sp += 1;
        st[sp] = nx;
        rem[sp] = legal(p, nx.hands[viewer], nx);
    }
}

static inline uint mix(uint h) {
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h;
}

static inline void add64(device atomic_uint* pair, uint x) {
    uint old = atomic_fetch_add_explicit(&pair[0], x, memory_order_relaxed);
    if (old + x < old) atomic_fetch_add_explicit(&pair[1], 1u, memory_order_relaxed);
}

static inline void insert(device atomic_uint* keys, device atomic_uint* vals, device atomic_uint* out, uint mask,
                          thread const Path& path, uint t0, uint t1, uint g, thread uint& err) {
    uint h = mix(path.w[0] ^ mix(path.w[1] ^ mix(path.w[2] ^ mix(path.w[3]))));
    for (uint probe = 0; probe <= mask; probe++) {
        uint slot = (h + probe) & mask;
        bool match = true;
        for (uint w = 0; w < 4 && match; w++) {
            device atomic_uint* cell = &keys[slot * 4 + 3 - w];
            uint want = path.w[3 - w];
            while (true) {
                uint expected = EMPTY;
                if (atomic_compare_exchange_weak_explicit(cell, &expected, want, memory_order_relaxed,
                                                          memory_order_relaxed)) {
                    if (w == 3) atomic_fetch_add_explicit(&out[8], 1u, memory_order_relaxed);
                    break;
                }
                if (expected == EMPTY) continue;  // a spurious failure
                match = expected == want;
                break;
            }
        }
        if (match) {
            atomic_fetch_add_explicit(&vals[slot * 4 + 0], 1u, memory_order_relaxed);
            if (t0) atomic_fetch_add_explicit(&vals[slot * 4 + 1], 1u, memory_order_relaxed);
            if (t1) atomic_fetch_add_explicit(&vals[slot * 4 + 2], 1u, memory_order_relaxed);
            if (g) atomic_fetch_add_explicit(&vals[slot * 4 + 3], 1u, memory_order_relaxed);
            return;
        }
    }
    err = ERR_TABLE;
}

// The j-th c-subset of `set`, its elements taken ascending (the combinatorial number system).
static inline uint unrank(device const uint* p, uint set, uint c, uint j) {
    uint out = 0, need = c, m = popcount(set);
    while (need > 0) {
        uint e = ctz(set);
        set &= set - 1;
        m -= 1;
        uint with = p[L_BINOM + m * 8 + need - 1];
        if (j < with) { out |= 1u << e; need -= 1; } else { j -= with; }
    }
    return out;
}

static inline bool consistent(device const uint* p, thread const uint* hands) {
    if (p[L_POSTERIOR] == 0) return true;
    uint held[4];
    for (uint s = 0; s < 4; s++) held[s] = hands[s] | p[L_PLAYED + s];
    uint viewer = p[L_VIEWER];
    uint lead = 0;
    for (uint i = 0; i < p[L_RECLEN]; i++) {
        uint s = p[L_RECORD + i] >> 5, t = p[L_RECORD + i] & 31u;
        bool leading = i % 4 == 0;
        if (s != viewer) {
            uint hand = held[s];
            uint lg = hand;
            if (!leading) {
                uint f = hand & p[L_FOLLOW + p[L_LED + lead]];
                lg = f == 0 ? hand : f;
            }
            if (law_play(p[L_LAWS + s], lg) != t) return false;
        }
        held[s] &= ~(1u << t);
        if (leading) lead = t;
    }
    return true;
}

// chunk: [threads, shards, mask]; shards: the chunk's first-slot hands; prefix: worlds before
// each shard (shards + 1 entries); out: [worlds lo, hi, tree lo, hi, tails lo, hi, god lo, hi, slots claimed].
kernel void ladder_gather(device const uint* p [[buffer(0)]],
                          device const uint* chunk [[buffer(1)]],
                          device const uint* shards [[buffer(2)]],
                          device const uint* prefix [[buffer(3)]],
                          device atomic_uint* keys [[buffer(4)]],
                          device atomic_uint* vals [[buffer(5)]],
                          device atomic_uint* out [[buffer(6)]],
                          device atomic_uint* errw [[buffer(7)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= chunk[0]) return;
    uint lo = 0, hi = chunk[1];
    while (hi - lo > 1) {
        uint mid = (lo + hi) / 2;
        if (prefix[mid] <= id) lo = mid; else hi = mid;
    }
    uint j = id - prefix[lo];
    uint h0 = shards[lo];
    uint rest = p[L_POOL] & ~h0;
    uint h1 = unrank(p, rest & p[L_ALLOWED + 1], p[L_CAP + 1], j);
    uint h2 = rest & ~h1;
    if ((h2 & ~p[L_ALLOWED + 2]) != 0) return;
    uint hands[4];
    for (uint s = 0; s < 4; s++) hands[s] = p[L_HANDS + s];
    hands[p[L_SEATS + 0]] = h0;
    hands[p[L_SEATS + 1]] = h1;
    hands[p[L_SEATS + 2]] = h2;
    if (!consistent(p, hands)) return;

    uint err = 0;
    uint tree = 0, tails = 0, gwork = 0;
    bool counted = false;
    uint mask = chunk[2];
    uint k = p[L_K];
    if (k >= MAXK) { atomic_fetch_max_explicit(errw, ERR_DEPTH, memory_order_relaxed); return; }
    uint viewer = p[L_VIEWER];
    St start;
    for (uint s = 0; s < 4; s++) start.hands[s] = hands[s];
    for (uint i = 0; i < 4; i++) start.trick[i] = uchar(p[L_TRICK + i]);
    start.n = p[L_N];
    start.leader = p[L_LEADER];
    start.banked[0] = p[L_BANKED];
    start.banked[1] = p[L_BANKED + 1];

    St st[MAXK + 1];
    Path path[MAXK + 1];
    uint rem[MAXK + 1];
    uint sp = 0;
    st[0] = start;
    path[0] = Path{{0, 0, 0, 0}, 0};
    rem[0] = legal(p, start.hands[viewer], start) & p[L_ROOTS];
    while (err == 0) {
        if (rem[sp] == 0) {
            if (sp == 0) break;
            sp -= 1;
            continue;
        }
        uint tile = ctz(rem[sp]);
        rem[sp] &= rem[sp] - 1;
        St nx = st[sp];
        Path px = path[sp];
        uint tree0 = tree;
        play(p, nx, tile, tree);
        push(px, tile, err);
        advance(p, nx, px, true, tree, err);
        if (sp == 0) {
            // A reply outside the piece: this root's work and leaves belong to another piece.
            uint reply = (px.w[3] >> 19) & 31u;
            // (No reply, the hand decided at the root play: bit 31.)
            uint bit = reply == 0 ? 31u : reply - 1;
            if (((p[L_REPLY] >> bit) & 1u) == 0) {
                tree = tree0;
                continue;
            }
            counted = true;
        }
        uint depth = sp + 1;
        int d = decided(p, nx);
        if (d >= 0) {
            insert(keys, vals, out, mask, px, uint(d), uint(d), uint(d), err);
            continue;
        }
        if (depth > k) {
            uint t0 = tail(p, nx, 0, tails);
            uint t1 = tail(p, nx, 1, tails);
            uint g = god(p, nx, gwork, err);
            insert(keys, vals, out, mask, px, t0, t1, g, err);
            continue;
        }
        sp += 1;
        st[sp] = nx;
        path[sp] = px;
        rem[sp] = legal(p, nx.hands[viewer], nx);
    }
    if (err != 0) {
        atomic_fetch_max_explicit(errw, err, memory_order_relaxed);
        return;
    }
    if (counted) add64(&out[0], 1u);
    add64(&out[2], tree);
    add64(&out[4], tails);
    add64(&out[6], gwork);
}
