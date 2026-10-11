//! The mold — the shape of one deal's perfect-information tree: how many
//! distinct positions each layer holds (layer d = d tiles played), with
//! transposition on the whole public record, as `Record.key()` defines it.
//! `scenario/atlas.py`'s `mold` is the referee.
//!
//! The middle layers of a deal hold hundreds of millions of positions, so
//! only two layers ever exist, each as one 128-bit key per position split
//! into hash-chosen shards: the current layer as plain vectors, the next as
//! sets being filled by every thread at once.
//!
//! A key is the record's tape key (played, leader, open trick, banked) in
//! the low word and each seat's void closure above it — the led suits whose
//! every follower the seat provably lacks. Forbid only ever grows by whole
//! follow sets, so a forbid mask is the union of its closure's follow sets
//! and the closure is a function of the mask: keys are equal exactly when
//! records are.

use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::planner::{splitmix, Rec};
use crate::rules::Rules;

const SHARDS: usize = 512;
const FLUSH: usize = 1024; // children a thread buffers per shard before taking its lock

#[derive(Clone, Copy, PartialEq, Eq)]
struct Key(u128);

impl Key {
    #[inline]
    fn spread(self) -> u64 {
        splitmix(self.0 as u64 ^ splitmix((self.0 >> 64) as u64))
    }
    #[inline]
    fn shard(self) -> usize {
        // Middle bits: the table indexes by the low bits and tags by the top seven.
        (self.spread() >> 32) as usize % SHARDS
    }
}

impl Hash for Key {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.spread());
    }
}

/// The key is mixed already; the table takes it as is.
#[derive(Default)]
struct Pass(u64);

impl Hasher for Pass {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("keys hash as one u64")
    }
    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

type Set = HashSet<Key, BuildHasherDefault<Pass>>;

fn encode(rec: &Rec, rules: &Rules) -> Key {
    let mut voids = 0u128;
    for (seat, &f) in rec.forbid.iter().enumerate() {
        let closure = (0..8).filter(|&s| rules.follow_bits[s] & !f == 0).fold(0u8, |c, s| c | 1 << s);
        voids |= (closure as u128) << (8 * seat);
    }
    Key(rec.pack() as u128 | voids << 64)
}

fn decode(key: Key, rules: &Rules) -> Rec {
    let lo = key.0 as u64;
    let n = (lo >> 30 & 3) as u8;
    let mut trick = [0u8; 3];
    for (i, t) in trick.iter_mut().enumerate().take(n as usize) {
        *t = (lo >> (32 + 5 * i) & 31) as u8;
    }
    let mut forbid = [0u32; 4];
    for (seat, f) in forbid.iter_mut().enumerate() {
        let closure = (key.0 >> (64 + 8 * seat)) as u8;
        *f = (0..8).filter(|&s| closure >> s & 1 == 1).fold(0, |m, s| m | rules.follow_bits[s]);
    }
    Rec {
        played: (lo & 0x0FFF_FFFF) as u32,
        leader: (lo >> 28 & 3) as u8,
        n,
        trick,
        banked: [(lo >> 47 & 63) as u8, (lo >> 53 & 63) as u8],
        forbid,
    }
}

/// Positions per layer for layers 0..=last of the deal `hands` (absolute
/// seats) with `leader` to lead, on `threads` OS threads.
pub fn mold(rules: &Rules, hands: [u32; 4], leader: u8, last: usize, threads: usize) -> Vec<u64> {
    let root = Rec { played: 0, leader, n: 0, trick: [0; 3], banked: [0; 2], forbid: [0; 4] };
    let mut layer: Vec<Vec<Key>> = vec![Vec::new(); SHARDS];
    layer[0].push(encode(&root, rules));
    let mut counts = Vec::with_capacity(last + 1);
    for d in 0..=last {
        counts.push(layer.iter().map(|s| s.len() as u64).sum());
        if d == last {
            break;
        }
        let next: Vec<Mutex<Set>> = (0..SHARDS).map(|_| Mutex::new(Set::default())).collect();
        let claim = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut bufs: Vec<Vec<Key>> = (0..SHARDS).map(|_| Vec::with_capacity(FLUSH)).collect();
                    let flush = |i: usize, buf: &mut Vec<Key>| {
                        next[i].lock().unwrap().extend(buf.drain(..));
                    };
                    loop {
                        let i = claim.fetch_add(1, Ordering::Relaxed);
                        if i >= SHARDS {
                            break;
                        }
                        for &key in &layer[i] {
                            let rec = decode(key, rules);
                            let hand = hands[rec.seat() as usize] & !rec.played;
                            let mut legal = rules.legal(hand, rec.led());
                            while legal != 0 {
                                let t = legal.trailing_zeros() as u8;
                                legal &= legal - 1;
                                let child = encode(&rec.advance(t, rules), rules);
                                let j = child.shard();
                                bufs[j].push(child);
                                if bufs[j].len() == FLUSH {
                                    flush(j, &mut bufs[j]);
                                }
                            }
                        }
                    }
                    for (j, buf) in bufs.iter_mut().enumerate() {
                        flush(j, buf);
                    }
                });
            }
        });
        // The expanded layer goes before the next one compacts from sets to vectors.
        drop(std::mem::take(&mut layer));
        layer = next.into_iter().map(|m| m.into_inner().unwrap().into_iter().collect()).collect();
    }
    counts
}
