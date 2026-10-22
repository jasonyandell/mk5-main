// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The focal-horizon ladder's gather on Metal: Exact-T1's `silas::player::trick1::Ladder`
//! (`gather`, `visit`, `tail`, `god`), one GPU thread per world, its output in Exact-T1's
//! `Gathered::to_bytes` format byte for byte, so its `--load` and `rung` read it unchanged.
//!
//! The host here builds the problem the way `Ladder::new` does (the same shards in the same
//! order, so a shard range means the same worlds), dispatches the worlds in chunks, and
//! drains the device's leaf table to a sorted map. Counts are exact integers; plays are
//! counted where Exact-T1 counts them (FAIR CLOCKS: the budget is plays, never time).
//!
//! STRATEGY-FUSION FENCE: the gather's `god` column is the ladder's clairvoyant upper
//! bound, the fence page's first sanctioned use (a bound that refutes); nothing here
//! chooses by it.

use std::collections::BTreeMap;

use silas::player::{Root, SeatLaw};
use silas::{Context, Domino, Frame, HIDDEN_SEATS, Seat};

use crate::epoch::EpochError;
use crate::metal::Gpu;

const L_WORDS: usize = 353 + 29 * 8;
const EMPTY: u32 = u32::MAX;
/// Worlds per command, at most; also at most a quarter of the leaf table's slots, the
/// table being drained once a quarter full. A command that fills it anyway refuses the gather.
pub const CHUNK_WORLDS: u64 = 1 << 21;
/// Lawful tails, as Exact-T1's `TAILS`: lowest, highest.
pub const NT: usize = 2;

/// Exact-T1's `Leaf`: the worlds reaching one public history, the success counts of each
/// lawful tail, the clairvoyant count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Leaf {
    pub mass: u64,
    pub tails: [u64; NT],
    pub god: u64,
}

impl Leaf {
    fn add(&mut self, o: &Leaf) {
        self.mass += o.mass;
        self.god += o.god;
        for i in 0..NT {
            self.tails[i] += o.tails[i];
        }
    }
}

/// Exact-T1's `Work`: plays laid walking the tree, in the tails, in the bound.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Work {
    pub tree: u64,
    pub tails: u64,
    pub god: u64,
}

impl Work {
    pub fn total(&self) -> u64 {
        self.tree + self.tails + self.god
    }
}

/// Exact-T1's `Gathered`, with the same bytes. Leaves are keyed by (path bits, length).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Gathered {
    pub k: usize,
    pub ranges: Vec<(usize, usize)>,
    pub leaves: BTreeMap<(u128, u8), Leaf>,
    pub worlds: u64,
    pub work: Work,
}

/// A path's length: its symbols run from the top while nonzero.
fn path_len(bits: u128) -> u8 {
    (0..25u8).take_while(|i| (bits >> (5 * (24 - i))) & 31 != 0).count() as u8
}

impl Gathered {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut put = |x: u64| out.extend_from_slice(&x.to_le_bytes());
        put(self.k as u64);
        put(self.ranges.len() as u64);
        for &(a, b) in &self.ranges {
            put(a as u64);
            put(b as u64);
        }
        put(self.worlds);
        put(self.work.tree);
        put(self.work.tails);
        put(self.work.god);
        put(self.leaves.len() as u64);
        for ((bits, len), l) in &self.leaves {
            put(*bits as u64);
            put((bits >> 64) as u64);
            put(u64::from(*len));
            put(l.mass);
            l.tails.iter().for_each(|t| put(*t));
            put(l.god);
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Gathered, String> {
        if !bytes.len().is_multiple_of(8) {
            return Err("a gather is whole words".into());
        }
        let mut words = bytes.chunks(8).map(|c| u64::from_le_bytes(c.try_into().expect("eight bytes")));
        let mut get = || words.next().ok_or_else(|| "the gather is cut short".to_string());
        let k = get()? as usize;
        let ranges = (0..get()?).map(|_| Ok((get()? as usize, get()? as usize))).collect::<Result<Vec<_>, String>>()?;
        let worlds = get()?;
        let work = Work { tree: get()?, tails: get()?, god: get()? };
        let mut leaves = BTreeMap::new();
        for _ in 0..get()? {
            let bits = u128::from(get()?) | (u128::from(get()?) << 64);
            let len = get()? as u8;
            let mut l = Leaf { mass: get()?, ..Leaf::default() };
            for t in l.tails.iter_mut() {
                *t = get()?;
            }
            l.god = get()?;
            leaves.insert((bits, len), l);
        }
        if get().is_ok() {
            return Err("the gather runs long".into());
        }
        Ok(Gathered { k, ranges, leaves, worlds, work })
    }
}

/// Exact-T1's `Rung`: per root action, the lawful lower count and the clairvoyant upper count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rung {
    pub k: usize,
    pub actions: Vec<(Domino, u64, u64)>,
    pub mass: u64,
    pub leaves: usize,
    pub work: Work,
}

impl Rung {
    pub fn bar(&self) -> u64 {
        self.actions.iter().map(|a| a.1).max().unwrap_or(0)
    }

    /// The executable root tile: the lowest tile of greatest lower count.
    pub fn choice(&self) -> Domino {
        let bar = self.bar();
        self.actions.iter().find(|a| a.1 == bar).expect("a root has an action").0
    }

    pub fn survivors(&self) -> Vec<Domino> {
        let bar = self.bar();
        self.actions.iter().filter(|a| a.2 >= bar).map(|a| a.0).collect()
    }

    pub fn gamma(&self) -> u64 {
        self.actions.iter().map(|a| a.2).max().unwrap_or(0) - self.bar()
    }
}

/// A public position for the backward pass: whose turn it is, read from the plays alone.
#[derive(Clone)]
struct Public {
    leader: Seat,
    trick: Vec<Domino>,
}

/// Every `k`-subset of the bits of `set`, in Exact-T1's order (`trick1::subsets`).
fn subsets(set: u32, k: usize) -> Vec<u32> {
    let tiles: Vec<u32> = (0..32).filter(|i| set >> i & 1 == 1).map(|i| 1u32 << i).collect();
    let mut out = Vec::new();
    fn go(tiles: &[u32], k: usize, acc: u32, out: &mut Vec<u32>) {
        if k == 0 {
            out.push(acc);
            return;
        }
        for i in 0..tiles.len() {
            if tiles.len() - i < k {
                break;
            }
            go(&tiles[i + 1..], k - 1, acc | tiles[i], out);
        }
    }
    go(&tiles, k, 0, &mut out);
    out
}

fn binomial(n: u64, r: u64) -> u64 {
    if r > n {
        return 0;
    }
    (0..r).fold(1, |acc, i| acc * (n - i) / (i + 1))
}

fn law_code(law: SeatLaw) -> Result<u32, String> {
    match law {
        SeatLaw::Lowest => Ok(0),
        SeatLaw::Highest => Ok(1),
        other => Err(format!("the ladder seats lowest and highest only, not {other:?}")),
    }
}

/// One root of the ladder, as the kernel reads it.
pub struct Problem {
    words: Vec<u32>,
    shards: Vec<u32>,
    pool: u32,
    allowed: [u32; HIDDEN_SEATS],
    capacity: [usize; HIDDEN_SEATS],
}

impl Problem {
    /// As `Ladder::new(frame, laws, root)`.
    pub fn new(frame: &Frame, laws: [SeatLaw; 4], root: Root) -> Result<Problem, String> {
        let viewer = frame.viewer().index();
        if frame.next_actor() != Some(frame.viewer()) {
            return Err("the ladder starts at the viewer's turn".into());
        }
        let mut codes = [0u32; 4];
        for s in 0..4 {
            if s != viewer {
                codes[s] = law_code(laws[s])?;
            }
        }
        let physics = frame.physics();
        let mut w = vec![0u32; L_WORDS];
        for d in Domino::ALL {
            w[d.index()] = physics.led_context(d).index() as u32;
            w[260 + d.index()] = physics.count(d);
        }
        for q in Context::ALL {
            w[28 + q.index()] = physics.incidence(q).bits();
            for d in Domino::ALL {
                w[36 + q.index() * 28 + d.index()] = u32::from(physics.trick_key(d, q));
            }
        }
        let mut played_by = [0u32; 4];
        let mut record = Vec::new();
        let mut banked = [0u32; 2];
        for trick in frame.tricks() {
            for (s, d) in trick {
                played_by[s.index()] |= 1 << d.index();
                record.push((s.index() as u32) * 32 + d.index() as u32);
            }
            if let Ok(t) = <[(Seat, Domino); 4]>::try_from(trick) {
                let winner = physics.trick_winner(t[0].0, t.map(|(_, d)| d));
                banked[winner.team().index()] += 1 + t.iter().map(|(_, d)| physics.count(*d)).sum::<u32>();
            }
        }
        let contract = frame.contract();
        let kernel = frame.kernel();
        w[288] = viewer as u32;
        w[289..293].copy_from_slice(&codes);
        w[293] = contract.declarer().team().index() as u32;
        w[294] = contract.threshold();
        w[295] = physics.total_points();
        w[297 + viewer] = kernel.viewer_hand().bits();
        for (i, d) in frame.prefix().iter().enumerate() {
            w[301 + i] = d.index() as u32;
        }
        w[305] = frame.prefix().len() as u32;
        w[306] = frame.leader().index() as u32;
        w[307..309].copy_from_slice(&banked);
        let pool = kernel.pool().bits();
        w[309] = pool;
        let hidden = kernel.hidden();
        let allowed: [u32; HIDDEN_SEATS] = std::array::from_fn(|i| kernel.allowed(physics, i).bits());
        let capacity: [usize; HIDDEN_SEATS] = std::array::from_fn(|i| hidden[i].capacity);
        for i in 0..HIDDEN_SEATS {
            w[310 + i] = allowed[i];
            w[313 + i] = capacity[i] as u32;
            w[316 + i] = hidden[i].seat.index() as u32;
        }
        w[319] = u32::from(root == Root::Posterior);
        w[320..324].copy_from_slice(&played_by);
        if record.len() > 28 {
            return Err("a record of more than 28 plays".into());
        }
        w[324] = record.len() as u32;
        w[325..325 + record.len()].copy_from_slice(&record);
        for n in 0..29u64 {
            for r in 0..8u64 {
                w[353 + (n * 8 + r) as usize] = binomial(n, r) as u32;
            }
        }
        Ok(Problem { words: w, shards: subsets(allowed[0], capacity[0]), pool, allowed, capacity })
    }

    pub fn shards(&self) -> usize {
        self.shards.len()
    }

    /// The rung of a gather covering every shard (Exact-T1's `Ladder::rung`): Bellman over
    /// the sorted leaves, the viewer's nodes taking each side's maximum separately, every
    /// other node summing.
    pub fn rung(&self, frame: &Frame, g: &Gathered) -> Result<Rung, String> {
        let covered: usize = g.ranges.iter().map(|(a, b)| b - a).sum();
        if covered != self.shards.len() {
            return Err(format!("the gather covers {covered} of {} shards", self.shards.len()));
        }
        let leaves: Vec<(u128, u8, Leaf)> = g.leaves.iter().map(|((b, l), x)| (*b, *l, *x)).collect();
        let at = |bits: u128, i: u8| Domino::from_index((((bits >> (5 * (24 - i))) & 31) - 1) as usize).expect("a tile");
        let start = Public { leader: frame.leader(), trick: frame.prefix().to_vec() };
        let viewer = frame.viewer();
        let physics = frame.physics();
        let play = |s: &Public, t: Domino| {
            let mut n = s.clone();
            n.trick.push(t);
            if let Ok(four) = <[Domino; 4]>::try_from(n.trick.as_slice()) {
                n.leader = physics.trick_winner(n.leader, four);
                n.trick.clear();
            }
            n
        };
        fn reduce(
            leaves: &[(u128, u8, Leaf)],
            depth: u8,
            s: &Public,
            viewer: Seat,
            at: &dyn Fn(u128, u8) -> Domino,
            play: &dyn Fn(&Public, Domino) -> Public,
        ) -> Result<(u64, u64, u64), String> {
            if leaves.len() == 1 && leaves[0].1 == depth {
                let l = &leaves[0].2;
                return Ok((*l.tails.iter().max().expect("a tail"), l.god, l.mass));
            }
            if leaves.iter().any(|x| x.1 <= depth) {
                return Err(format!("a leaf is a prefix of another at depth {depth}"));
            }
            let focal = s.leader.plus(s.trick.len()) == viewer;
            let (mut lo, mut up, mut mass) = (0u64, 0u64, None::<u64>);
            let mut i = 0;
            while i < leaves.len() {
                let t = at(leaves[i].0, depth);
                let j = i + leaves[i..].iter().take_while(|x| at(x.0, depth) == t).count();
                let (l, u, m) = reduce(&leaves[i..j], depth + 1, &play(s, t), viewer, at, play)?;
                if focal {
                    lo = lo.max(l);
                    up = up.max(u);
                    if mass.is_some_and(|x| x != m) {
                        return Err("a viewer's play changed the mass".into());
                    }
                    mass = Some(m);
                } else {
                    lo += l;
                    up += u;
                    mass = Some(mass.unwrap_or(0) + m);
                }
                i = j;
            }
            Ok((lo, up, mass.expect("a node has a child")))
        }
        let mut actions = Vec::new();
        let mut i = 0;
        while i < leaves.len() {
            let a = at(leaves[i].0, 0);
            let j = i + leaves[i..].iter().take_while(|x| at(x.0, 0) == a).count();
            let (lo, up, m) = reduce(&leaves[i..j], 1, &play(&start, a), viewer, &at, &play)?;
            if m != g.worlds {
                return Err(format!("root action {a} carries {m} of {} worlds", g.worlds));
            }
            actions.push((a, lo, up));
            i = j;
        }
        Ok(Rung { k: g.k, actions, mass: g.worlds, leaves: leaves.len(), work: g.work })
    }

    /// Worlds of shard `i` before the record's likelihood: the second slot's hands.
    fn candidates(&self, i: usize) -> u64 {
        let rest = self.pool & !self.shards[i];
        binomial(u64::from((rest & self.allowed[1]).count_ones()), self.capacity[1] as u64)
    }

    /// The leaves of horizon `k` over shards `from..to`, on the GPU, with a leaf table of
    /// `2^table_bits` slots (drained to the host whenever it passes a quarter full).
    pub fn gather(&self, gpu: &mut Gpu, k: usize, from: usize, to: usize, table_bits: u32) -> Result<Gathered, EpochError> {
        assert!(from <= to && to <= self.shards.len(), "shards {from}..{to} of {}", self.shards.len());
        if k >= 6 {
            return Err(EpochError::Depth);
        }
        let mut words = self.words.clone();
        words[296] = k as u32;
        let p = gpu.upload(&words)?;
        let slots = 1usize << table_bits;
        let keys = gpu.zeroed(slots * 4)?;
        Gpu::fill(&keys, slots * 4, EMPTY);
        let vals = gpu.zeroed(slots * 4)?;
        let out = gpu.zeroed(9)?;
        let err = gpu.zeroed(1)?;
        let mut leaves: BTreeMap<(u128, u8), Leaf> = BTreeMap::new();
        let drain = |leaves: &mut BTreeMap<(u128, u8), Leaf>| {
            let kw = Gpu::read(&keys, slots * 4);
            let vw = Gpu::read(&vals, slots * 4);
            for s in 0..slots {
                let key = &kw[4 * s..4 * s + 4];
                if key.contains(&EMPTY) {
                    continue;
                }
                let bits = key.iter().enumerate().fold(0u128, |a, (i, x)| a | (u128::from(*x) << (32 * i)));
                let v = &vw[4 * s..4 * s + 4];
                let leaf = Leaf { mass: v[0].into(), tails: [v[1].into(), v[2].into()], god: v[3].into() };
                leaves.entry((bits, path_len(bits))).or_default().add(&leaf);
            }
            Gpu::fill(&keys, slots * 4, EMPTY);
            Gpu::fill(&vals, slots * 4, 0);
        };
        let chunk_worlds = CHUNK_WORLDS.min(slots as u64 / 4);
        let mut claimed_at_drain = 0u32;
        let mut i = from;
        while i < to {
            let mut shards = Vec::new();
            let mut prefix = vec![0u32];
            let mut total = 0u64;
            while i < to && (shards.is_empty() || total + self.candidates(i) <= chunk_worlds) {
                total += self.candidates(i);
                shards.push(self.shards[i]);
                prefix.push(total as u32);
                i += 1;
            }
            let chunk = gpu.upload(&[total as u32, shards.len() as u32, (slots - 1) as u32])?;
            let (shards_b, prefix_b) = (gpu.upload(&shards)?, gpu.upload(&prefix)?);
            let ladder = gpu.ladder.clone();
            gpu.run(&ladder, &[&p, &chunk, &shards_b, &prefix_b, &keys, &vals, &out, &err], total as usize)?;
            match Gpu::read(&err, 1)[0] {
                0 => {}
                1 => return Err(EpochError::Device(format!("the leaf table of 2^{table_bits} slots is full"))),
                2 => return Err(EpochError::Depth),
                e => return Err(EpochError::Device(format!("the ladder kernel refused with code {e}"))),
            }
            let claimed = Gpu::read(&out, 9)[8];
            if (claimed - claimed_at_drain) as usize > slots / 4 {
                drain(&mut leaves);
                claimed_at_drain = claimed;
            }
        }
        drain(&mut leaves);
        let o = Gpu::read(&out, 8);
        let pair = |i: usize| u64::from(o[i]) | (u64::from(o[i + 1]) << 32);
        Ok(Gathered {
            k,
            ranges: vec![(from, to)],
            leaves,
            worlds: pair(0),
            work: Work { tree: pair(2), tails: pair(4), god: pair(6) },
        })
    }
}
