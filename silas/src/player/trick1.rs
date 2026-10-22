// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The focal-horizon ladder at any decision, built for trick 1 (Exact-T1, 2026-10-22;
//! `scratch/desks/family-2026-10-22/design-t1.md`, texas-42 `walt/math/focal_horizon_sandwich_v0.1.md`).
//!
//! STRATEGY-FUSION FENCE: the upper tail here is clairvoyant. [`god`] lets the viewer see
//! every hand of one world and asks whether some line of its own plays then makes (the
//! others play their declared laws, which see only their own hands). It is used as a bound
//! that refutes, the fence page's first sanctioned use: it supplies the upper end `U` of a
//! root action's interval and nothing else. No choice is ever read off it; every choice, at
//! the root and inside the horizon, is the argmax of the lawful lower side. The ruling:
//! `wiki/topics/strategy-fusion-fence.md`.
//!
//! The model: one deterministic seat law per other seat (a [`SeatLaw`] that is a pure
//! function of the seat's own hand and the trick's lead). The belief is the exact counted
//! posterior over the fiber: every deal of the unseen tiles that fits the seats' counts and
//! shown voids, kept when each law replays every play the other seats made (a 0/1
//! likelihood), or every deal of the fiber when the root reads the record through its
//! voids alone ([`Root::FiberUniform`]). Masses are counts of worlds.
//!
//! The ladder (focal-horizon sandwich, Theorems 1 to 4): from the root, the viewer's next
//! `k + 1` decisions are made exact, one choice per public history (the merge before the
//! max), and below that horizon a world is valued twice: below by the better of a library
//! of lawful tails, chosen once per public history after the sum over its worlds; above by
//! the clairvoyant continuation. For root action `a`, `L[a] <= Q[a] <= U[a]`; both nest in
//! `k`; at `k + 1` at least the viewer's decisions left, `L = Q = U`.
//!
//! Every count is an integer; the work is counted in plays (every tile any simulation
//! lays), a deterministic budget (FAIR CLOCKS).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{Root, SeatLaw};
use crate::frame::{Frame, HIDDEN_SEATS};
use crate::nouns::{Context, Domino, Seat};
use crate::physics::Physics;
use crate::{Result, error};

/// One declaration's rules as flat tables, read once from [`Physics`].
#[derive(Clone, Debug)]
pub struct Rules {
    led: [u8; Domino::COUNT],
    follow: [u32; Context::COUNT],
    key: [[u8; Domino::COUNT]; Context::COUNT],
    count: [u8; Domino::COUNT],
}

impl Rules {
    pub fn new(physics: &Physics) -> Rules {
        let led = Domino::ALL.map(|d| physics.led_context(d).index() as u8);
        let follow = Context::ALL.map(|q| physics.incidence(q).bits());
        let key = Context::ALL.map(|q| Domino::ALL.map(|d| physics.trick_key(d, q)));
        let count = Domino::ALL.map(|d| physics.count(d) as u8);
        Rules { led, follow, key, count }
    }

    /// The legal plays from `hand` when `lead` led the trick (anything when leading).
    #[inline]
    pub fn legal(&self, hand: u32, lead: Option<u8>) -> u32 {
        match lead {
            None => hand,
            Some(t) => match hand & self.follow[self.led[t as usize] as usize] {
                0 => hand,
                f => f,
            },
        }
    }

    /// The offset from the leader of the play that takes the trick: the first of the
    /// greatest keys in the led context.
    #[inline]
    pub fn winner(&self, plays: [u8; 4]) -> usize {
        let keys = &self.key[self.led[plays[0] as usize] as usize];
        let mut best = 0;
        for i in 1..4 {
            if keys[plays[i] as usize] > keys[plays[best] as usize] {
                best = i;
            }
        }
        best
    }

    #[inline]
    pub fn points(&self, plays: [u8; 4]) -> u8 {
        1 + plays.iter().map(|t| self.count[*t as usize]).sum::<u8>()
    }
}

/// The tile a deterministic law plays from `legal`.
#[inline]
fn law_play(law: SeatLaw, legal: u32) -> u8 {
    match law {
        SeatLaw::Lowest => legal.trailing_zeros() as u8,
        SeatLaw::Highest => (31 - legal.leading_zeros()) as u8,
        _ => unreachable!("the ladder seats deterministic laws only"),
    }
}

fn deterministic(law: SeatLaw) -> bool {
    matches!(law, SeatLaw::Lowest | SeatLaw::Highest)
}

/// A lawful tail for the viewer below the horizon: a deterministic law read from its own
/// hand and the lead alone, so one tile per information state.
pub const TAILS: [SeatLaw; 2] = [SeatLaw::Lowest, SeatLaw::Highest];
const NT: usize = TAILS.len();

/// The state of one world mid-hand.
#[derive(Clone, Copy, Debug)]
struct St {
    hands: [u32; 4],
    trick: [u8; 4],
    n: u8,
    leader: u8,
    banked: [u8; 2],
}

impl St {
    #[inline]
    fn actor(&self) -> usize {
        (self.leader as usize + self.n as usize) % 4
    }

    #[inline]
    fn lead(&self) -> Option<u8> {
        if self.n == 0 { None } else { Some(self.trick[0]) }
    }
}

/// What the viewer wants and how the hand is played: the rules, the other seats' laws, the
/// contract's arithmetic.
struct Game {
    rules: Rules,
    viewer: usize,
    laws: [SeatLaw; 4],
    /// The declaring team's index and its threshold; the defenders set past `42 - threshold`.
    declaring: usize,
    threshold: u8,
    total: u8,
}

impl Game {
    /// Whether the viewer's side has its way, once decided: the declarers make, or the
    /// defenders set.
    #[inline]
    fn decided(&self, st: &St) -> Option<bool> {
        let mine = self.viewer % 2 == self.declaring;
        if st.banked[self.declaring] >= self.threshold {
            Some(mine)
        } else if st.banked[1 - self.declaring] > self.total - self.threshold {
            Some(!mine)
        } else {
            None
        }
    }

    #[inline]
    fn play(&self, st: &mut St, tile: u8, work: &mut u64) {
        *work += 1;
        let seat = st.actor();
        st.hands[seat] &= !(1u32 << tile);
        st.trick[st.n as usize] = tile;
        st.n += 1;
        if st.n == 4 {
            let w = self.rules.winner(st.trick);
            let winner = (st.leader as usize + w) % 4;
            st.banked[winner % 2] += self.rules.points(st.trick);
            st.leader = winner as u8;
            st.n = 0;
        }
    }

    /// The other seats play their laws until the viewer is to play or the hand is decided.
    /// Each play is appended to the path.
    #[inline]
    fn advance(&self, st: &mut St, path: &mut Path, work: &mut u64) {
        while self.decided(st).is_none() && st.actor() != self.viewer {
            let seat = st.actor();
            let tile = law_play(self.laws[seat], self.rules.legal(st.hands[seat], st.lead()));
            self.play(st, tile, work);
            path.push(tile);
        }
    }

    /// The value of the lawful tail `law` for the viewer from `st`, every seat its law.
    fn tail(&self, mut st: St, law: SeatLaw, work: &mut u64) -> bool {
        loop {
            if let Some(v) = self.decided(&st) {
                return v;
            }
            let seat = st.actor();
            let l = if seat == self.viewer { law } else { self.laws[seat] };
            let tile = law_play(l, self.rules.legal(st.hands[seat], st.lead()));
            self.play(&mut st, tile, work);
        }
    }
}

/// STRATEGY-FUSION FENCE: the clairvoyant continuation, a bound only. Whether some line of
/// the viewer's own plays, chosen seeing this one world, has its way against the others'
/// laws. An upper bound on every lawful continuation's value in this world.
fn god(game: &Game, st: St, work: &mut u64) -> bool {
    let mut st = st;
    let mut sink = Path::default();
    game.advance(&mut st, &mut sink, work);
    if let Some(v) = game.decided(&st) {
        return v;
    }
    let hand = st.hands[game.viewer];
    let mut legal = game.rules.legal(hand, st.lead());
    while legal != 0 {
        let tile = legal.trailing_zeros() as u8;
        legal &= legal - 1;
        let mut next = st;
        game.play(&mut next, tile, work);
        if god(game, next, work) {
            return true;
        }
    }
    false
}

/// A public history from the root: plays as 5-bit symbols (tile + 1), the first in the
/// highest bits, so numeric order is lexicographic order with a prefix first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Path {
    bits: u128,
    len: u8,
}

const PATH_MAX: u8 = 25;

impl Path {
    #[inline]
    fn push(&mut self, tile: u8) {
        assert!(self.len < PATH_MAX, "a path holds at most {PATH_MAX} plays");
        self.bits |= u128::from(tile + 1) << (5 * (PATH_MAX - 1 - self.len));
        self.len += 1;
    }

    #[inline]
    fn at(&self, i: u8) -> u8 {
        ((self.bits >> (5 * (PATH_MAX - 1 - i))) & 31) as u8 - 1
    }
}

/// A leaf of the horizon tree: the worlds reaching one public history, the success counts
/// of each lawful tail, and the clairvoyant upper count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Leaf {
    mass: u64,
    tails: [u64; NT],
    god: u64,
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

/// The ladder at one horizon: per root action, the lawful lower count and the clairvoyant
/// upper count, over the root's worlds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rung {
    pub k: usize,
    /// Root tile, lower count, upper count, ascending by tile.
    pub actions: Vec<(Domino, u64, u64)>,
    /// The worlds of the root (the posterior's mass).
    pub mass: u64,
    /// Public histories at the horizon (leaves of the tree).
    pub leaves: usize,
    /// Plays laid by every simulation.
    pub work: Work,
}

impl Rung {
    /// The best lower count: the executable policy's value (its root tile is the argmax).
    pub fn bar(&self) -> u64 {
        self.actions.iter().map(|a| a.1).max().unwrap_or(0)
    }

    /// The executable policy's root tile: the lowest tile of greatest lower count.
    pub fn choice(&self) -> Domino {
        let bar = self.bar();
        self.actions.iter().find(|a| a.1 == bar).expect("a root has an action").0
    }

    /// The actions whose upper count reaches the bar.
    pub fn survivors(&self) -> Vec<Domino> {
        let bar = self.bar();
        self.actions.iter().filter(|a| a.2 >= bar).map(|a| a.0).collect()
    }

    /// The certified regret count: `max U - L(executable)`.
    pub fn gamma(&self) -> u64 {
        self.actions.iter().map(|a| a.2).max().unwrap_or(0) - self.bar()
    }

    /// One action is proved best: its lower count exceeds every other upper count.
    pub fn settled(&self) -> Option<Domino> {
        let bar = self.bar();
        let c = self.choice();
        self.actions.iter().all(|a| a.0 == c || a.2 < bar).then_some(c)
    }
}

/// One root: the start state, the worlds' hidden hands (by slot), and the game.
pub struct Ladder {
    game: Game,
    start: St,
    hidden_seats: [usize; HIDDEN_SEATS],
    /// The hands of the first hidden slot, the shard unit; the rest are enumerated per shard.
    shards: Vec<u32>,
    pool: u32,
    capacity: [usize; HIDDEN_SEATS],
    allowed: [u32; HIDDEN_SEATS],
    /// Each seat's plays so far, and the record, for the posterior's replay.
    played_by: [u32; 4],
    record: Vec<(usize, u8)>,
    root: Root,
}

/// Every `k`-subset of the bits of `set`, ascending.
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

impl Ladder {
    /// The ladder at the viewer's decision in `frame`, the other seats playing `laws`
    /// (deterministic ones), the belief read as `root` says.
    pub fn new(frame: &Frame, laws: [SeatLaw; 4], root: Root) -> Result<Ladder> {
        let viewer = frame.viewer().index();
        if frame.next_actor() != Some(frame.viewer()) {
            return Err(error("the ladder starts at the viewer's turn"));
        }
        if let Some(s) = (0..4).find(|s| *s != viewer && !deterministic(laws[*s])) {
            return Err(error(format!(
                "seat {s}'s law is not deterministic; the ladder seats lowest and highest only"
            )));
        }
        let physics = frame.physics();
        let rules = Rules::new(physics);
        let contract = frame.contract();
        let mut played_by = [0u32; 4];
        let mut record = Vec::new();
        let mut banked = [0u8; 2];
        for trick in frame.tricks() {
            for (s, d) in trick {
                played_by[s.index()] |= 1 << d.index();
                record.push((s.index(), d.index() as u8));
            }
            if let Ok(t) = <[(Seat, Domino); 4]>::try_from(trick) {
                let plays = t.map(|(_, d)| d.index() as u8);
                let winner = t[rules.winner(plays)].0;
                banked[winner.team().index()] += rules.points(plays);
            }
        }
        let kernel = frame.kernel();
        let mut hands = [0u32; 4];
        hands[viewer] = kernel.viewer_hand().bits();
        let mut trick = [0u8; 4];
        for (i, d) in frame.prefix().iter().enumerate() {
            trick[i] = d.index() as u8;
        }
        let start = St { hands, trick, n: frame.prefix().len() as u8, leader: frame.leader().index() as u8, banked };
        let game = Game {
            rules,
            viewer,
            laws,
            declaring: contract.declarer().team().index(),
            threshold: contract.threshold() as u8,
            total: physics.total_points() as u8,
        };
        let hidden = kernel.hidden();
        let pool = kernel.pool().bits();
        let allowed: [u32; HIDDEN_SEATS] = core::array::from_fn(|i| kernel.allowed(physics, i).bits());
        let capacity = core::array::from_fn(|i| hidden[i].capacity);
        Ok(Ladder {
            game,
            start,
            hidden_seats: core::array::from_fn(|i| hidden[i].seat.index()),
            shards: subsets(allowed[0], capacity[0]),
            pool,
            capacity,
            allowed,
            played_by,
            record,
            root,
        })
    }

    /// Shards: the first hidden slot's possible hands.
    pub fn shards(&self) -> usize {
        self.shards.len()
    }

    /// Whether the laws replay every play the other seats made in this world (the 0/1
    /// likelihood of the record), each hand being its remaining tiles plus its plays.
    fn consistent(&self, hands: &[u32; 4]) -> bool {
        if self.root == Root::FiberUniform {
            return true;
        }
        let mut held: [u32; 4] = core::array::from_fn(|s| hands[s] | self.played_by[s]);
        let mut lead = None;
        for (i, &(s, t)) in self.record.iter().enumerate() {
            if i % 4 == 0 {
                lead = None;
            }
            if s != self.game.viewer && law_play(self.game.laws[s], self.game.rules.legal(held[s], lead)) != t {
                return false;
            }
            held[s] &= !(1 << t);
            if i % 4 == 0 {
                lead = Some(t);
            }
        }
        true
    }

    /// Every world of shard `i`, as the four remaining hands.
    fn worlds(&self, i: usize, mut f: impl FnMut(&[u32; 4])) {
        let h0 = self.shards[i];
        let rest = self.pool & !h0;
        for h1 in subsets(rest & self.allowed[1], self.capacity[1]) {
            let h2 = rest & !h1;
            if h2 & !self.allowed[2] != 0 {
                continue;
            }
            let mut hands = self.start.hands;
            hands[self.hidden_seats[0]] = h0;
            hands[self.hidden_seats[1]] = h1;
            hands[self.hidden_seats[2]] = h2;
            if self.consistent(&hands) {
                f(&hands);
            }
        }
    }

    /// One world's leaves: the viewer's next `k + 1` decisions branched (the root's
    /// included), each horizon state valued by every lawful tail and by the bound.
    #[allow(clippy::too_many_arguments)]
    fn visit(
        &self,
        st: St,
        depth: usize,
        k: usize,
        roots: u32,
        path: Path,
        acc: &mut HashMap<Path, Leaf>,
        work: &mut Work,
    ) {
        let game = &self.game;
        // The root is branched even when the hand is already decided, so every root action
        // has its leaves.
        if let Some(v) = game.decided(&st).filter(|_| depth > 0) {
            let leaf = Leaf { mass: 1, tails: [v as u64; NT], god: v as u64 };
            acc.entry(path).or_default().add(&leaf);
            return;
        }
        if depth > k {
            let mut leaf = Leaf { mass: 1, tails: [0; NT], god: 0 };
            for (i, law) in TAILS.iter().enumerate() {
                leaf.tails[i] = game.tail(st, *law, &mut work.tails) as u64;
            }
            leaf.god = god(game, st, &mut work.god) as u64;
            acc.entry(path).or_default().add(&leaf);
            return;
        }
        let mut legal = game.rules.legal(st.hands[game.viewer], st.lead()) & roots;
        while legal != 0 {
            let tile = legal.trailing_zeros() as u8;
            legal &= legal - 1;
            let mut next = st;
            let mut p = path;
            game.play(&mut next, tile, &mut work.tree);
            p.push(tile);
            game.advance(&mut next, &mut p, &mut work.tree);
            self.visit(next, depth + 1, k, u32::MAX, p, acc, work);
        }
    }

    /// The viewer's legal tiles at the root, as bits.
    pub fn menu(&self) -> u32 {
        self.game.rules.legal(self.start.hands[self.game.viewer], self.start.lead())
    }

    /// The leaves of horizon `k` over the shards `from..to` for the root tiles in `roots`
    /// (bits; [`Ladder::menu`] for all), on `threads` threads. Gathers over disjoint ranges
    /// add ([`Gathered::merge`]), so a rung spans as many runs as its cap needs, and root
    /// tiles never meet below the root's maximum, so each can be gathered alone; what a
    /// gather costs does not depend on the threads or the machine.
    pub fn gather(&self, k: usize, roots: u32, from: usize, to: usize, threads: usize) -> Gathered {
        assert!(from <= to && to <= self.shards.len(), "shards {from}..{to} of {}", self.shards.len());
        let next = AtomicUsize::new(from);
        let total = Mutex::new((HashMap::<Path, Leaf>::new(), 0u64, Work::default()));
        std::thread::scope(|scope| {
            for _ in 0..threads.max(1) {
                scope.spawn(|| {
                    let mut acc: HashMap<Path, Leaf> = HashMap::new();
                    let (mut worlds, mut work) = (0u64, Work::default());
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= to {
                            break;
                        }
                        self.worlds(i, |hands| {
                            worlds += 1;
                            let mut st = self.start;
                            st.hands = *hands;
                            self.visit(st, 0, k, roots, Path::default(), &mut acc, &mut work);
                        });
                    }
                    let mut t = total.lock().expect("the merge is unpoisoned");
                    for (p, l) in acc {
                        t.0.entry(p).or_default().add(&l);
                    }
                    t.1 += worlds;
                    t.2.add(&work);
                });
            }
        });
        let (leaves, worlds, work) = total.into_inner().expect("the merge is unpoisoned");
        Gathered { k, ranges: vec![(from, to)], leaves, worlds, work }
    }

    /// The rung of gathers of one horizon, each over every shard, whose root tiles are
    /// together the menu, each tile once; refused past `budget` plays in all. The gathers are
    /// consumed one at a time, so a rung whose tiles do not fit in memory together still reduces.
    pub fn rung(&self, gathers: impl IntoIterator<Item = Gathered>, budget: u64) -> Result<Rung> {
        let (mut actions, mut leaves, mut work, mut mass, mut k) = (Vec::new(), 0, Work::default(), None, None);
        for g in gathers {
            let covered: usize = g.ranges.iter().map(|(a, b)| b - a).sum();
            if covered != self.shards.len() {
                return Err(error(format!("a gather covers {covered} of {} shards", self.shards.len())));
            }
            if k.is_some_and(|k| k != g.k) || mass.is_some_and(|m| m != g.worlds) {
                return Err(error("the gathers are of different horizons or roots"));
            }
            (k, mass) = (Some(g.k), Some(g.worlds));
            work.add(&g.work);
            let mut sorted: Vec<(Path, Leaf)> = g.leaves.into_iter().collect();
            sorted.sort_unstable_by_key(|(p, _)| *p);
            leaves += sorted.len();
            let mut i = 0;
            while i < sorted.len() {
                let a = sorted[i].0.at(0);
                let j = i + sorted[i..].iter().take_while(|(p, _)| p.at(0) == a).count();
                let mut st = self.start;
                st.hands = [u32::MAX; 4];
                let mut sink = 0;
                self.game.play(&mut st, a, &mut sink);
                let (lo, up, m) = self.reduce(&sorted[i..j], 1, st);
                assert_eq!(m, g.worlds, "every root action carries the root's whole mass");
                actions.push((Domino::from_index(a as usize).expect("a tile"), lo, up));
                i = j;
            }
        }
        actions.sort_unstable_by_key(|a| a.0);
        let k = k.ok_or_else(|| error("a rung needs a gather"))?;
        let tiles: Vec<Domino> = actions.iter().map(|a| a.0).collect();
        let menu: Vec<Domino> = Domino::ALL.into_iter().filter(|d| self.menu() >> d.index() & 1 == 1).collect();
        if tiles != menu {
            return Err(error(format!("the gathers hold root tiles {tiles:?}, not the menu {menu:?} once each")));
        }
        if work.total() > budget {
            return Err(error(format!("refused: horizon {k} took {} plays, past {budget}", work.total())));
        }
        Ok(Rung { k, actions, mass: mass.expect("a gather"), leaves, work })
    }

    /// The rung at horizon `k` in one run.
    pub fn solve(&self, k: usize, threads: usize, budget: u64) -> Result<Rung> {
        self.rung([self.gather(k, self.menu(), 0, self.shards.len(), threads)], budget)
    }

    /// Bellman over the sorted leaves sharing a prefix of length `at`, the public state
    /// `st` after it (hands unread): the viewer's node takes the maximum of each side,
    /// separately; another seat's node sums. Returns (lower, upper, mass).
    fn reduce(&self, leaves: &[(Path, Leaf)], at: u8, st: St) -> (u64, u64, u64) {
        if leaves.len() == 1 && leaves[0].0.len == at {
            let l = &leaves[0].1;
            return (*l.tails.iter().max().expect("a tail"), l.god, l.mass);
        }
        assert!(
            leaves.iter().all(|(p, _)| p.len > at),
            "a leaf is never a prefix of another: at {at}, {:?}",
            leaves.iter().map(|(p, l)| ((0..p.len).map(|i| p.at(i)).collect::<Vec<_>>(), l.mass)).collect::<Vec<_>>()
        );
        let focal = st.actor() == self.game.viewer;
        let (mut lo, mut up, mut mass) = (0u64, 0u64, None::<u64>);
        let mut i = 0;
        while i < leaves.len() {
            let t = leaves[i].0.at(at);
            let j = i + leaves[i..].iter().take_while(|(p, _)| p.at(at) == t).count();
            let mut next = st;
            let mut sink = 0;
            self.game.play(&mut next, t, &mut sink);
            let (l, u, m) = self.reduce(&leaves[i..j], at + 1, next);
            if focal {
                lo = lo.max(l);
                up = up.max(u);
                assert!(mass.is_none_or(|x| x == m), "a viewer's play keeps the mass");
                mass = Some(m);
            } else {
                lo += l;
                up += u;
                mass = Some(mass.unwrap_or(0) + m);
            }
            i = j;
        }
        (lo, up, mass.expect("a node has a child"))
    }
}

/// Plays laid, by where: walking the horizon tree, the lawful tails below it, the
/// clairvoyant bound below it.
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

    fn add(&mut self, o: &Work) {
        self.tree += o.tree;
        self.tails += o.tails;
        self.god += o.god;
    }
}

/// The horizon's leaves over some shards of a root: what a capped run hands the next.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gathered {
    pub k: usize,
    /// The shard ranges covered, disjoint.
    pub ranges: Vec<(usize, usize)>,
    leaves: HashMap<Path, Leaf>,
    pub worlds: u64,
    pub work: Work,
}

impl Gathered {
    /// Two gathers of the same horizon over disjoint ranges, added.
    pub fn merge(mut self, o: Gathered) -> Result<Gathered> {
        if self.k != o.k {
            return Err(error(format!("horizons {} and {} do not merge", self.k, o.k)));
        }
        if self.roots() != o.roots() && !self.leaves.is_empty() && !o.leaves.is_empty() {
            return Err(error("gathers of different root tiles do not merge; a rung joins them"));
        }
        for &(a, b) in &o.ranges {
            if self.ranges.iter().any(|&(c, d)| a < d && c < b) {
                return Err(error(format!("shards {a}..{b} are gathered twice")));
            }
        }
        self.ranges.extend(o.ranges);
        self.ranges.sort_unstable();
        // Adjacent ranges coalesce, so pieces merged write the bytes of the whole.
        self.ranges.dedup_by(|b, a| {
            let touch = a.1 == b.0;
            if touch {
                a.1 = b.1;
            }
            touch
        });
        for (p, l) in o.leaves {
            self.leaves.entry(p).or_default().add(&l);
        }
        self.worlds += o.worlds;
        self.work.add(&o.work);
        Ok(self)
    }

    /// The same leaves, masses and bounds over the same shards: the parity another engine
    /// must meet. Work is left out; it follows the search order, so each engine reports its own.
    pub fn agrees(&self, o: &Gathered) -> bool {
        (self.k, &self.ranges, &self.leaves, self.worlds) == (o.k, &o.ranges, &o.leaves, o.worlds)
    }

    /// The root tiles gathered, as bits.
    pub fn roots(&self) -> u32 {
        self.leaves.keys().fold(0, |r, p| r | 1 << p.at(0))
    }

    /// Public histories gathered so far.
    pub fn leaves(&self) -> usize {
        self.leaves.len()
    }

    /// The gather as bytes (little-endian words; leaves in path order, so equal gathers
    /// write equal bytes).
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
        let mut leaves: Vec<_> = self.leaves.iter().collect();
        leaves.sort_unstable_by_key(|(p, _)| **p);
        put(leaves.len() as u64);
        for (p, l) in leaves {
            put(p.bits as u64);
            put((p.bits >> 64) as u64);
            put(p.len as u64);
            put(l.mass);
            l.tails.iter().for_each(|t| put(*t));
            put(l.god);
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Gathered> {
        if !bytes.len().is_multiple_of(8) {
            return Err(error("a gather is whole words"));
        }
        let mut words = bytes.chunks(8).map(|c| u64::from_le_bytes(c.try_into().expect("eight bytes")));
        let mut get = || words.next().ok_or_else(|| error("the gather is cut short"));
        let k = get()? as usize;
        let ranges = (0..get()?).map(|_| Ok((get()? as usize, get()? as usize))).collect::<Result<Vec<_>>>()?;
        let worlds = get()?;
        let work = Work { tree: get()?, tails: get()?, god: get()? };
        let n = get()?;
        let mut leaves = HashMap::with_capacity((n as usize).min(bytes.len() / 56));
        for _ in 0..n {
            let bits = u128::from(get()?) | u128::from(get()?) << 64;
            let p = Path { bits, len: get()? as u8 };
            let mut l = Leaf { mass: get()?, ..Leaf::default() };
            for t in l.tails.iter_mut() {
                *t = get()?;
            }
            l.god = get()?;
            leaves.insert(p, l);
        }
        if get().is_ok() {
            return Err(error("the gather runs long"));
        }
        Ok(Gathered { k, ranges, leaves, worlds, work })
    }
}
