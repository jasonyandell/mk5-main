// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! Constellations: the canonical form of a position over the live tiles.
//!
//! The remaining game reads a tile only through relations among the tiles
//! still in it: which suit it leads, which led suits it follows, where it
//! stands among the followers of each suit that can still be led, where it
//! stands among the trumps, its count, and who holds it. A constellation is
//! that pattern with the names rubbed out: the pip suits are colors with no
//! order of their own, the dead tiles are gone, and two positions with the
//! same constellation are the same remaining game under a tile bijection
//! (texas-42's Lemma E, `walt/CENSUS-RULINGS.md`; its idea page is
//! `wiki/idea-retrograde-rank.md` there). Any value that is a fold over the
//! remaining game with an operator that reads only those relations is equal
//! at the two.
//!
//! Two forms:
//!
//! - [`seat_form`] — a seat's information state: its own tiles, the tiles
//!   it cannot see (and which seats may hold each, from the voids shown),
//!   the open trick, who leads, and what the contract still needs. This is
//!   the key for a table of *information-state* values (Zeb's plan under a
//!   field that itself reads only constellations). It never names a world.
//! - [`world_form`] — every holder known. A census object only (class
//!   counts); it carries no value here.
//!
//! What it is not: a key for the deployed field net, which reads pips and
//! the dead tiles, so a value planned against that net is not a function of
//! the constellation (`scenario/constellation.py` measures how often).
//!
//! Canonical labeling: tiles carry a label-free code (holder class, the
//! seats provably void of it, count, trump standing); each color carries an
//! invariant (its followers' codes in standing order); colors are sorted by
//! invariant and only ties are permuted, the lexicographically least tile
//! list over those permutations is the form, and its order is the labeling.

use crate::planner::Rec;
use crate::rules::Rules;

/// The called suit's index among the led suits.
const CALLED: usize = 7;
/// "No standing here" in a 4-bit standing slot.
const NONE: u64 = 15;
/// The most tiles a form can hold: 28 live less nothing, plus a 3-tile open trick.
pub const MAX_TILES: usize = 31;

/// A canonical form: a header and the tiles' codes, sorted. Equal forms
/// are isomorphic remaining games; `order[i]` is the tile at canonical
/// index i.
#[derive(Clone, Debug)]
pub struct Canon {
    pub form: Form,
    pub order: Vec<u8>,
    /// Color orders that reach the form: the position's automorphisms. Above
    /// one, some tiles are interchangeable and `order` is one labeling of several.
    pub automorphisms: u32,
}

/// The form itself, hashable: `words[0]` is the header, then one code per tile.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Form {
    pub words: Vec<u64>,
}

impl Canon {
    /// The canonical index of `tile`, if it is in the form.
    pub fn index_of(&self, tile: u8) -> Option<usize> {
        self.order.iter().position(|&t| t == tile)
    }
}

/// What the value still depends on beyond the tiles: under `make`, the
/// points the bidding side still needs, clipped to [0, P + 1] (P the
/// points still to be won), and the viewer's side; under `points`, only the
/// viewer's side (the table then holds the margin still to come).
#[derive(Clone, Copy, Debug)]
pub enum Contract {
    Make {
        bid: i32,
        bidder_team: u8,
    },
    Points {
        bidder_team: u8,
    },
    /// No count and no score: Lemma E's own object (texas-42's census
    /// counted these). Sound for count-free values only.
    CountFree,
}

/// The seat-level form: `viewer`'s information state at `rec`, holding `own`.
pub fn seat_form(rules: &Rules, viewer: u8, own: u32, rec: &Rec, contract: Contract) -> Canon {
    let holder = |t: u8, live: bool| -> u64 {
        if !live {
            4 + rel(viewer, trick_seat(rec, t)) as u64
        } else if own >> t & 1 == 1 {
            1
        } else {
            2
        }
    };
    canonize(rules, viewer, rec, contract, true, &holder)
}

/// The world-level form: every holder known (`hands` are the remaining
/// hands, absolute seats), seats relative to `focal`. A census object; the
/// voids the record shows are dropped, since with every hand known they
/// change nothing in the remaining game.
pub fn world_form(rules: &Rules, focal: u8, hands: &[u32; 4], rec: &Rec, contract: Contract) -> Canon {
    let holder = |t: u8, live: bool| -> u64 {
        if !live {
            return 4 + rel(focal, trick_seat(rec, t)) as u64;
        }
        let seat = (0..4u8).find(|&s| hands[s as usize] >> t & 1 == 1).expect("a live tile is held");
        8 + rel(focal, seat) as u64
    };
    canonize(rules, focal, rec, contract, false, &holder)
}

#[inline]
fn rel(from: u8, seat: u8) -> u8 {
    (seat + 4 - from) % 4
}

fn trick_seat(rec: &Rec, t: u8) -> u8 {
    let i = rec.trick[..rec.n as usize].iter().position(|&x| x == t).expect("a table tile is in the trick");
    (rec.leader + i as u8) % 4
}

fn bits(mut m: u32) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if m == 0 {
            return None;
        }
        let t = m.trailing_zeros() as u8;
        m &= m - 1;
        Some(t)
    })
}

/// Tiles of `set` in descending order of `rank`, written to `out`; returns how many.
#[inline]
fn standings(set: u32, rank: &[u8; 28], out: &mut [u8; 28]) -> usize {
    let mut n = 0;
    for t in bits(set) {
        out[n] = t;
        n += 1;
    }
    out[..n].sort_unstable_by(|a, b| rank[*b as usize].cmp(&rank[*a as usize]));
    n
}

/// A color's invariant: whether the open trick was led in it, then its
/// followers' codes, standing and lead bit, in standing order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Sig {
    len: u8,
    words: [u64; 12],
}

fn canonize(
    rules: &Rules,
    focal: u8,
    rec: &Rec,
    contract: Contract,
    with_voids: bool,
    holder: &dyn Fn(u8, bool) -> u64,
) -> Canon {
    let all: u32 = (1 << 28) - 1;
    let live = !rec.played & all;
    let table: u32 = rec.trick[..rec.n as usize].iter().fold(0, |m, &t| m | 1 << t);
    let universe = live | table;
    let led = if rec.n > 0 { Some(rules.led_suit[rec.trick[0] as usize] as usize) } else { None };

    // Trumps: rank 32 and up under every led suit. With none, the called
    // suit (nello's doubles) is a color like the pips.
    let mut trumps = 0u32;
    for t in 0..28u8 {
        if rules.rank[0][t as usize] >= 32 {
            trumps |= 1 << t;
        }
    }
    let has_trump = trumps != 0;

    // The suits that can still be led: some live tile leads them, or the open trick does.
    let mut leadable = 0u32;
    for t in bits(live) {
        leadable |= 1 << rules.led_suit[t as usize];
    }
    if let Some(c) = led {
        leadable |= 1 << c;
    }
    let mut colors = [0usize; 8];
    let mut m = 0;
    for c in 0..8 {
        if leadable >> c & 1 == 1 && !(has_trump && c == CALLED) {
            colors[m] = c;
            m += 1;
        }
    }

    // Label-free codes: holder (5 bits), void seats (4), count (2), trump standing (4).
    let mut buf = [0u8; 28];
    let mut tstand = [NONE; 28];
    let nt = standings(trumps & universe, &rules.rank[CALLED], &mut buf);
    for (i, &t) in buf[..nt].iter().enumerate() {
        tstand[t as usize] = i as u64;
    }
    let mut base = [0u64; 28];
    for t in bits(universe) {
        let is_live = live >> t & 1 == 1;
        let mut voids = 0u64;
        if with_voids && is_live {
            for r in 0..4u8 {
                voids |= ((rec.forbid[((focal + r) % 4) as usize] >> t & 1) as u64) << r;
            }
        }
        let count = match (rules.count[t as usize], contract) {
            (_, Contract::CountFree) => 0,
            (10, _) => 2,
            (5, _) => 1,
            _ => 0,
        };
        base[t as usize] = holder(t, is_live) << 10 | voids << 6 | count << 4 | tstand[t as usize];
    }

    // Each color's followers in standing order: the open trick's tiles count
    // only in the suit it was led in; elsewhere only the live tiles stand.
    let mut stand = [[NONE; 28]; 8];
    let mut sigs = [Sig { len: 0, words: [0; 12] }; 8];
    for j in 0..m {
        let c = colors[j];
        let pool = if Some(c) == led { universe } else { live };
        let n = standings(rules.follow_bits[c] & pool, &rules.rank[c], &mut buf);
        let sig = &mut sigs[j];
        sig.words[0] = (Some(c) == led) as u64;
        for (s, &t) in buf[..n].iter().enumerate() {
            stand[j][t as usize] = s as u64;
            let leads = (is_set(live, t) && rules.led_suit[t as usize] as usize == c) as u64;
            sig.words[1 + s] = base[t as usize] << 5 | (s as u64) << 1 | leads;
        }
        sig.len = 1 + n as u8;
    }
    let mut by_sig = [0usize; 8];
    for (j, x) in by_sig[..m].iter_mut().enumerate() {
        *x = j;
    }
    by_sig[..m].sort_unstable_by(|a, b| sigs[*a].cmp(&sigs[*b]).then(a.cmp(b)));

    let mut lead_of = [NONE; 28];
    for t in bits(live) {
        let c = rules.led_suit[t as usize] as usize;
        lead_of[t as usize] =
            if has_trump && c == CALLED { 14 } else { colors[..m].iter().position(|&x| x == c).unwrap() as u64 };
    }
    let led_j = led.and_then(|c| colors[..m].iter().position(|&x| x == c));

    // The header: who leads and how far the trick is, relative to the focal
    // seat; the led suit (a canonical color, 14 trump, 15 none); the contract.
    let tricks_left = (live.count_ones() + rec.n as u32).div_ceil(4) as i32;
    let pool_points = tricks_left + bits(universe).map(|t| rules.count[t as usize] as i32).sum::<i32>();
    let contract_bits: u64 = match contract {
        Contract::Make { bid, bidder_team } => {
            let need = (bid - rec.banked[bidder_team as usize] as i32).clamp(0, pool_points + 1) as u64;
            1 << 8 | ((focal % 2 == bidder_team) as u64) << 7 | need
        }
        Contract::Points { bidder_team } => ((focal % 2 == bidder_team) as u64) << 7,
        Contract::CountFree => 0,
    };

    let mut tiles = [0u8; MAX_TILES];
    let mut nt = 0;
    for t in bits(universe) {
        tiles[nt] = t;
        nt += 1;
    }
    let mut best_words = [u64::MAX; MAX_TILES + 1];
    let mut best_order = [0u8; MAX_TILES];
    let mut automorphisms = 0u32;
    let mut eval = |order: &[usize]| {
        let mut slot = [0u64; 8]; // color index -> canonical position
        for (pos, &j) in order.iter().enumerate() {
            slot[j] = pos as u64;
        }
        let led_code = match (led, led_j) {
            (None, _) => NONE,
            (_, Some(j)) => slot[j],
            (Some(_), None) => 14,
        };
        let mut words = [0u64; MAX_TILES + 1];
        words[0] = (rel(focal, rec.leader) as u64) << 40 | (rec.n as u64) << 36 | led_code << 32 | contract_bits;
        let mut coded = [(0u64, 0u8); MAX_TILES];
        for (i, &t) in tiles[..nt].iter().enumerate() {
            let lead = match lead_of[t as usize] {
                _ if !is_set(live, t) => NONE,
                14 => 14,
                j => slot[j as usize],
            };
            let mut code = base[t as usize] << 32 | lead << 28;
            for (pos, &j) in order.iter().enumerate() {
                code |= stand[j][t as usize] << (4 * pos);
            }
            coded[i] = (code, t);
        }
        coded[..nt].sort_unstable();
        for i in 0..nt {
            words[1 + i] = coded[i].0;
        }
        match words[..=nt].cmp(&best_words[..=nt]) {
            std::cmp::Ordering::Equal => automorphisms += 1,
            std::cmp::Ordering::Greater => {}
            std::cmp::Ordering::Less => {
                best_words = words;
                for i in 0..nt {
                    best_order[i] = coded[i].1;
                }
                automorphisms = 1;
            }
        }
    };

    // Permute only colors whose invariants tie; almost always none do.
    let tied = (1..m).any(|k| sigs[by_sig[k - 1]] == sigs[by_sig[k]]);
    if !tied {
        eval(&by_sig[..m]);
    } else {
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for k in 0..m {
            if k > 0 && sigs[by_sig[k - 1]] == sigs[by_sig[k]] {
                groups.last_mut().unwrap().push(by_sig[k]);
            } else {
                groups.push(vec![by_sig[k]]);
            }
        }
        each_order(&groups, 0, &mut Vec::with_capacity(m), &mut |o: &[usize]| eval(o));
    }
    Canon { form: Form { words: best_words[..=nt].to_vec() }, order: best_order[..nt].to_vec(), automorphisms }
}

#[inline]
fn is_set(m: u32, t: u8) -> bool {
    m >> t & 1 == 1
}

/// Every color order that keeps the invariant classes in sequence and
/// permutes within each.
fn each_order(groups: &[Vec<usize>], g: usize, prefix: &mut Vec<usize>, f: &mut dyn FnMut(&[usize])) {
    if g == groups.len() {
        f(prefix);
        return;
    }
    permute(&groups[g], &mut Vec::new(), &mut vec![false; groups[g].len()], &mut |p: &[usize]| {
        let n = prefix.len();
        prefix.extend_from_slice(p);
        each_order(groups, g + 1, prefix, f);
        prefix.truncate(n);
    });
}

fn permute(items: &[usize], acc: &mut Vec<usize>, used: &mut Vec<bool>, f: &mut dyn FnMut(&[usize])) {
    if acc.len() == items.len() {
        f(acc);
        return;
    }
    for i in 0..items.len() {
        if !used[i] {
            used[i] = true;
            acc.push(items[i]);
            permute(items, acc, used, f);
            acc.pop();
            used[i] = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{tile, Rules};

    /// Relabel pips by `p` (a permutation of 0..7): a tile maps to the tile of its relabeled ends.
    fn relabel(p: &[u8; 7], t: u8) -> u8 {
        let (h, l) = crate::rules::pips(t);
        let (a, b) = (p[h as usize], p[l as usize]);
        tile(a.max(b), a.min(b))
    }

    fn hands_of(deal: &[u8; 28]) -> [u32; 4] {
        let mut h = [0u32; 4];
        for (i, &t) in deal.iter().enumerate() {
            h[i / 7] |= 1 << t;
        }
        h
    }

    fn shuffled(seed: u64) -> [u8; 28] {
        let mut d: [u8; 28] = std::array::from_fn(|i| i as u8);
        let mut s = crate::planner::SplitMix::new(seed);
        for i in (1..28).rev() {
            let j = (s.next_u64() % (i as u64 + 1)) as usize;
            d.swap(i, j);
        }
        d
    }

    /// A random legal play-out of `plays` tiles from a shuffled deal.
    fn played_out(rules: &Rules, seed: u64, plays: usize) -> ([u32; 4], Rec) {
        let mut hands = hands_of(&shuffled(seed));
        let mut rec = Rec::opening((seed % 4) as u8);
        let mut s = crate::planner::SplitMix::new(seed ^ 0xABCD);
        for _ in 0..plays {
            let seat = rec.seat() as usize;
            let legal = rules.legal(hands[seat], rec.led());
            let ts: Vec<u8> = bits(legal).collect();
            let t = ts[(s.next_u64() % ts.len() as u64) as usize];
            hands[seat] &= !(1 << t);
            rec = rec.advance(t, rules);
        }
        (hands, rec)
    }

    #[test]
    fn the_seven_trump_declarations_fold_to_one_form() {
        // Relabeling the pips so trump p becomes trump q, keeping the other
        // pips in order, carries every relation but count (texas-42's
        // Lemma S-fold is count-free). Where every tile still in play keeps
        // its count it is a game isomorphism, and the forms agree.
        let mut checked = 0;
        for seed in 0..2000u64 {
            let p = (seed % 7) as u8;
            let q = ((seed / 7) % 7) as u8;
            let rp = Rules::new(p).unwrap();
            let rq = Rules::new(q).unwrap();
            let (hands, rec) = played_out(&rp, seed, 16 + (seed % 9) as usize);
            // pip map: p -> q, the others in order
            let mut others_from: Vec<u8> = (0..7).filter(|&x| x != p).collect();
            let others_to: Vec<u8> = (0..7).filter(|&x| x != q).collect();
            let mut map = [0u8; 7];
            map[p as usize] = q;
            for (a, b) in others_from.drain(..).zip(others_to) {
                map[a as usize] = b;
            }
            let mv = |m: u32| bits(m).fold(0u32, |acc, t| acc | 1 << relabel(&map, t));
            let mut trick = [0u8; 3];
            for (slot, &t) in trick.iter_mut().zip(&rec.trick[..rec.n as usize]) {
                *slot = relabel(&map, t);
            }
            let rec2 = Rec {
                played: mv(rec.played),
                leader: rec.leader,
                n: rec.n,
                trick,
                banked: rec.banked,
                forbid: [mv(rec.forbid[0]), mv(rec.forbid[1]), mv(rec.forbid[2]), mv(rec.forbid[3])],
            };
            let hands2 = [mv(hands[0]), mv(hands[1]), mv(hands[2]), mv(hands[3])];
            let in_play =
                !rec.played & ((1 << 28) - 1) | rec.trick[..rec.n as usize].iter().fold(0, |m, &t| m | 1 << t);
            if bits(in_play).any(|t| rp.count[t as usize] != rq.count[relabel(&map, t) as usize]) {
                continue;
            }
            checked += 1;
            let seat = rec.seat();
            let c = Contract::Make { bid: 30, bidder_team: 0 };
            let a = seat_form(&rp, seat, hands[seat as usize], &rec, c);
            let b = seat_form(&rq, seat, hands2[seat as usize], &rec2, c);
            assert_eq!(a.form, b.form, "seed {seed}");
            for (x, y) in a.order.iter().zip(&b.order) {
                assert_eq!(relabel(&map, *x), *y, "the labeling transports, seed {seed}");
            }
            assert_eq!(world_form(&rp, seat, &hands, &rec, c).form, world_form(&rq, seat, &hands2, &rec2, c).form);
        }
        assert!(checked >= 50, "only {checked} count-preserving cases");
    }

    #[test]
    fn the_form_ignores_the_dead_and_reads_the_living() {
        // Last trick, sixes trump: 6-6 led into three sloughs, and 6-5 led into
        // the same three: one live trump on top either way.
        let r = Rules::new(6).unwrap();
        let c = Contract::Points { bidder_team: 0 };
        let mk = |top: u8| {
            let rest = [tile(1, 0), tile(2, 0), tile(3, 0)];
            let live = 1u32 << top | rest.iter().fold(0, |m, &t| m | 1 << t);
            let rec = Rec {
                played: !live & ((1 << 28) - 1),
                leader: 0,
                n: 0,
                trick: [0; 3],
                banked: [20, 18],
                forbid: [0; 4],
            };
            let hands = [1 << top, 1 << rest[0], 1 << rest[1], 1 << rest[2]];
            world_form(&r, 0, &hands, &rec, c)
        };
        assert_eq!(mk(tile(6, 6)).form, mk(tile(6, 5)).form);
    }

    #[test]
    fn different_standings_are_different_forms() {
        // The leader's trump is boss in one and not the other.
        let r = Rules::new(6).unwrap();
        let c = Contract::Points { bidder_team: 0 };
        let mk = |a: u8, b: u8| {
            let live = 1u32 << a | 1 << b | 1 << tile(2, 0) | 1 << tile(3, 0);
            let rec =
                Rec { played: !live & ((1 << 28) - 1), leader: 0, n: 0, trick: [0; 3], banked: [0, 0], forbid: [0; 4] };
            let hands = [1 << a, 1 << b, 1 << tile(2, 0), 1 << tile(3, 0)];
            world_form(&r, 0, &hands, &rec, c)
        };
        assert_ne!(mk(tile(6, 6), tile(6, 5)).form, mk(tile(6, 5), tile(6, 6)).form);
    }

    #[test]
    fn the_labeling_is_a_bijection_onto_the_live_and_table_tiles() {
        let r = Rules::new(3).unwrap();
        for seed in 0..100u64 {
            let (hands, rec) = played_out(&r, seed, 12 + (seed % 13) as usize);
            let seat = rec.seat();
            let f = seat_form(&r, seat, hands[seat as usize], &rec, Contract::Make { bid: 30, bidder_team: 1 });
            let mut want: Vec<u8> =
                bits(!rec.played & ((1 << 28) - 1)).chain(rec.trick[..rec.n as usize].iter().copied()).collect();
            let mut got = f.order.clone();
            want.sort();
            got.sort();
            assert_eq!(got, want);
            assert_eq!(f.form.words.len(), f.order.len() + 1);
        }
    }
}
