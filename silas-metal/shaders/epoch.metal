// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

// The field epoch on Metal. One thread per work item (a segment: one information state and
// one focal action; and one particle: a hidden world and its table type, with an exact mass).
// The thread plays the focal action, then every play the declared field can make, in
// ascending tile order, until the focal seat must decide again or the hand ends, and emits
// one child per line: the field plays seen, the hands left, and the mass times the line's
// likelihood at scale 840 per field play. Integers only; the rules come from Silas's tables.
// The layouts are fixed in src/epoch.rs (the word offsets below).

#include <metal_stdlib>
using namespace metal;

constant uint T_INC = 0;     // incidence[8]: the tiles following each context
constant uint T_LED = 8;     // led[28]: the context each tile leads
constant uint T_RANK = 36;   // rank[8 * 28]: a tile's trick key in a led context
constant uint T_LAW = 260;   // law[type * 4 + seat]

constant uint SEG_WORDS = 12;
constant uint PART_WORDS = 16;
constant uint CHILD_WORDS = 16;
constant uint MAX_FIELD = 6;  // field plays between two focal plays: at most 3 + 3

constant uint LAW_LOWEST = 0;
constant uint LAW_HIGHEST = 1;
constant uint LAW_UNIFORM = 2;
constant uint LAW_LEAN_LOW = 3;
constant uint LAW_LEAN_HIGH = 4;

constant uint ERR_OVERFLOW = 1;
constant uint ERR_DEPTH = 2;
constant uint ERR_COUNT = 3;
constant uint ERR_ILLEGAL = 4;
constant uint ERR_LAW = 5;

struct State {
    uint hands[4];
    uint leader;
    uint n;          // plays in the open trick
    uint t[4];       // the open trick's plays
    uint done;       // completed tricks
    uint obs;        // field plays seen, five bits each, 31 = none
    uint nobs;
};

static uint lowest_bit(uint m) { return ctz(m); }
static uint highest_bit(uint m) { return 31 - clz(m); }

static uint legal(device const uint* tab, thread const State& s, uint seat) {
    uint hand = s.hands[seat];
    if (s.n == 0) return hand;
    uint follow = hand & tab[T_INC + tab[T_LED + s.t[0]]];
    return follow != 0 ? follow : hand;
}

// Play `tile` for the seat to act; resolve the trick when it is the fourth play.
static void play(device const uint* tab, thread State& s, uint tile) {
    uint seat = (s.leader + s.n) & 3;
    s.hands[seat] &= ~(1u << tile);
    s.t[s.n] = tile;
    s.n += 1;
    if (s.n == 4) {
        uint led = tab[T_LED + s.t[0]];
        uint best = 0;
        for (uint i = 1; i < 4; i++) {
            if (tab[T_RANK + led * 28 + s.t[i]] > tab[T_RANK + led * 28 + s.t[best]]) best = i;
        }
        s.leader = (s.leader + best) & 3;
        s.n = 0;
        s.done += 1;
    }
}

// The law's weight on `tile` at scale 840, given the legal set; zero off its support.
static uint weight(uint law, uint legal_set, uint tile, thread uint& err) {
    uint n = popcount(legal_set);
    uint lo = lowest_bit(legal_set);
    uint hi = highest_bit(legal_set);
    switch (law) {
        case LAW_LOWEST: return tile == lo ? 840 : 0;
        case LAW_HIGHEST: return tile == hi ? 840 : 0;
        case LAW_UNIFORM: return 840 / n;
        case LAW_LEAN_LOW: return tile == lo ? 420 * (n + 1) / n : 420 / n;
        case LAW_LEAN_HIGH: return tile == hi ? 420 * (n + 1) / n : 420 / n;
        default: err = ERR_LAW; return 0;
    }
}

// The tiles the law can play.
static uint support(uint law, uint legal_set) {
    switch (law) {
        case LAW_LOWEST: return 1u << lowest_bit(legal_set);
        case LAW_HIGHEST: return 1u << highest_bit(legal_set);
        default: return legal_set;
    }
}

// mass *= f, exactly, eight 32-bit limbs; false on overflow.
static bool mul_small(thread uint* m, uint f) {
    ulong carry = 0;
    for (uint i = 0; i < 8; i++) {
        ulong p = (ulong)m[i] * (ulong)f + carry;
        m[i] = (uint)p;
        carry = p >> 32;
    }
    return carry == 0;
}

static void fail(device atomic_uint* err, uint code) {
    atomic_fetch_max_explicit(err, code, memory_order_relaxed);
}

// The epoch for one work item. With `emit` false it only counts the lines; with `emit` true
// it writes them at `out + offset`, and refuses when the count differs from the first pass.
static void epoch(device const uint* tab, device const uint* segs, device const uint* parts,
                  device const uint* items, uint item, bool emit, device uint* counts,
                  uint base, uint expect, device uint* out, uint out_cap,
                  device atomic_uint* err) {
    uint seg = items[2 * item];
    uint part = items[2 * item + 1];
    device const uint* g = segs + seg * SEG_WORDS;
    device const uint* p = parts + part * PART_WORDS;
    uint focal = g[0];
    State s0;
    for (uint i = 0; i < 4; i++) s0.hands[i] = p[i];
    s0.hands[focal] = g[8];
    s0.leader = g[1];
    s0.n = g[2];
    for (uint i = 0; i < 4; i++) s0.t[i] = g[3 + i];
    s0.done = g[7];
    s0.obs = 0x3fffffff;
    s0.nobs = 0;
    uint action = g[9];
    if (((legal(tab, s0, focal) >> action) & 1) == 0 || ((s0.leader + s0.n) & 3) != focal) {
        fail(err, ERR_ILLEGAL);
        return;
    }
    play(tab, s0, action);
    uint kind = p[4];
    uint world = p[5];

    State st[MAX_FIELD + 1];
    uint cand[MAX_FIELD + 1];
    uint fac[MAX_FIELD + 1];
    st[0] = s0;
    int d = 0;
    bool fresh = true;
    uint lines = 0;
    uint e = 0;
    while (true) {
        if (fresh) {
            fresh = false;
            thread State& s = st[d];
            uint actor = (s.leader + s.n) & 3;
            if (s.done == 7 || actor == focal) {
                if (emit) {
                    if (lines >= expect || base + lines >= out_cap) { fail(err, ERR_COUNT); return; }
                    device uint* c = out + (base + lines) * CHILD_WORDS;
                    uint m[8];
                    for (uint i = 0; i < 8; i++) m[i] = p[8 + i];
                    for (int k = 1; k <= d; k++) {
                        if (!mul_small(m, fac[k])) { fail(err, ERR_OVERFLOW); return; }
                    }
                    c[0] = item;
                    c[1] = s.obs;
                    for (uint i = 0; i < 4; i++) c[2 + i] = s.hands[i];
                    c[2 + focal] = 0;
                    c[6] = kind;
                    c[7] = world;
                    for (uint i = 0; i < 8; i++) c[8 + i] = m[i];
                }
                lines += 1;
                if (d == 0) break;
                d -= 1;
                continue;
            }
            if (d == (int)MAX_FIELD) { fail(err, ERR_DEPTH); return; }
            uint law = tab[T_LAW + kind * 4 + actor];
            cand[d] = support(law, legal(tab, s, actor));
        }
        if (cand[d] == 0) {
            if (d == 0) break;
            d -= 1;
            continue;
        }
        uint tile = lowest_bit(cand[d]);
        cand[d] &= cand[d] - 1;
        thread State& s = st[d];
        uint actor = (s.leader + s.n) & 3;
        uint law = tab[T_LAW + kind * 4 + actor];
        fac[d + 1] = weight(law, legal(tab, s, actor), tile, e);
        if (e != 0) { fail(err, e); return; }
        st[d + 1] = s;
        play(tab, st[d + 1], tile);
        uint shift = 5 * st[d + 1].nobs;
        st[d + 1].obs = (st[d + 1].obs & ~(31u << shift)) | (tile << shift);
        st[d + 1].nobs += 1;
        d += 1;
        fresh = true;
    }
    if (!emit) counts[item] = lines;
    else if (lines != expect) fail(err, ERR_COUNT);
}

kernel void epoch_count(device const uint* tab [[buffer(0)]],
                        device const uint* segs [[buffer(1)]],
                        device const uint* parts [[buffer(2)]],
                        device const uint* items [[buffer(3)]],
                        device uint* counts [[buffer(4)]],
                        device atomic_uint* err [[buffer(5)]],
                        constant uint& n_items [[buffer(6)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= n_items) return;
    epoch(tab, segs, parts, items, id, false, counts, 0, 0, counts, 0, err);
}

kernel void epoch_emit(device const uint* tab [[buffer(0)]],
                       device const uint* segs [[buffer(1)]],
                       device const uint* parts [[buffer(2)]],
                       device const uint* items [[buffer(3)]],
                       device const uint* counts [[buffer(4)]],
                       device const uint* offsets [[buffer(5)]],
                       device uint* out [[buffer(6)]],
                       device atomic_uint* err [[buffer(7)]],
                       constant uint* params [[buffer(8)]],
                       uint id [[thread_position_in_grid]]) {
    uint n_items = params[0];
    uint out_cap = params[1];
    uint item = params[2] + id;
    if (id >= n_items) return;
    epoch(tab, segs, parts, items, item, true, (device uint*)counts, offsets[id], counts[item], out, out_cap, err);
}
