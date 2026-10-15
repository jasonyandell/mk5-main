// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! STRATEGY-FUSION FENCE. An exact perfect-information (double-dummy,
//! clairvoyant) solver: minimax over every remaining play when all four hands
//! are face up. It sees all four hands, so its value used as the value of a
//! move for a seat that cannot see them, per world and then averaged, is
//! strategy fusion: "strong but wrong" (Jason, 2026-10-13). It is behind the
//! non-default `clairvoyant` cargo feature, and a policy cannot read it.
//! Allowed instead: information-state solving, one choice per information
//! set and never one per world (`forces`, `lay_down`, `(certain-make)`,
//! `(certain-set)`; Zeb's plan). The ruling, the inventory and the sanctioned
//! uses: `wiki/topics/strategy-fusion-fence.md`.
//!
//! **This is a reference, never a player.** A value it reports is what the
//! seat could achieve if it could see the hands too. It explains a book move
//! only where the book's own argument is about a fully known position;
//! everywhere else it is an upper bound on what any real seat can achieve, and
//! a move it calls a mistake may be the right play for a seat that cannot tell
//! the worlds apart.
//!
//! The solver works over any [`Physics`], from an [`OpenPosition`]: the four
//! hands, the contract (whose declaration is the physics), the trick's leader
//! and plays so far, and the points each side has banked. It has two objectives ([`Objective`]): the points
//! margin for the declaring side (theirs minus the defenders', the declaring
//! side maximizing and the defenders minimizing), and make-or-set against the
//! contract (one if the declaring side reaches its threshold, zero if not), which a
//! null-window search answers without the exact count. It returns the value of
//! every legal move at the root, so a given move can be graded against the
//! best.
//!
//! The search is integer alpha-beta with a transposition table at each
//! trick's start, holding proven lower and upper bounds on the declaring
//! side's points still to come. With tile ownership fixed, the remaining pool,
//! leader and declaring side are an exact 31-bit key; changing an owner clears
//! the table. Scores and bids do not enter the key, so one table serves both
//! objectives and every position of a deal. Interchangeable plays are searched
//! once, and the forced final trick is resolved directly.

use crate::frame::{Contract, Frame, World};
use crate::nouns::{Context, Domino, DominoSet, Seat};
use crate::physics::Physics;
use crate::{Result, error};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

/// A position with every hand face up: the contract, the four remaining
/// hands, the current trick's leader and plays, and the points each side has
/// banked in closed tricks (by team index).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OpenPosition {
    contract: Contract,
    hands: [DominoSet; Seat::COUNT],
    leader: Seat,
    trick: Vec<Domino>,
    banked: [u32; 2],
}

impl OpenPosition {
    /// The opening position of a deal: seven tiles to each seat, the declarer
    /// to lead, nothing banked.
    pub fn deal(contract: Contract, hands: [DominoSet; Seat::COUNT]) -> Result<OpenPosition> {
        if hands.iter().any(|h| h.len() != Domino::COUNT / Seat::COUNT) {
            return Err(error("a deal gives each seat seven tiles"));
        }
        let leader = contract.declarer();
        OpenPosition::new(contract, hands, leader, Vec::new(), [0, 0])
    }

    /// A position of the play. Checked for coherence: the hands and the
    /// trick's plays are disjoint, at most three tiles are on the table, the
    /// seats that have played to the trick hold one tile fewer than those
    /// still to play. Whether the trick's plays followed suit is not checked:
    /// the hands they came from are not part of the position.
    pub fn new(
        contract: Contract,
        hands: [DominoSet; Seat::COUNT],
        leader: Seat,
        trick: Vec<Domino>,
        banked: [u32; 2],
    ) -> Result<OpenPosition> {
        let mut seen = DominoSet::EMPTY;
        for d in hands.iter().flat_map(|h| h.iter()).chain(trick.iter().copied()) {
            if !seen.insert(d) {
                return Err(error(format!("{d} is in two places")));
            }
        }
        if trick.len() >= Seat::COUNT {
            return Err(error("a trick on the table holds at most three plays"));
        }
        let held = hands[leader.plus(trick.len()).index()].len();
        let coherent =
            (0..Seat::COUNT).all(|i| hands[leader.plus(i).index()].len() + usize::from(i < trick.len()) == held);
        if !coherent || (held == 0 && !trick.is_empty()) {
            return Err(error("the hands' sizes disagree with the trick's plays"));
        }
        Ok(OpenPosition { contract, hands, leader, trick, banked })
    }

    /// A frame's position in one of its worlds: the world's four hands, and
    /// the frame's contract, open trick and banked points.
    pub fn at(frame: &Frame, world: &World) -> Result<OpenPosition> {
        if !frame.contains(world) {
            return Err(error("the world is outside this frame's support"));
        }
        OpenPosition::of(frame, Seat::ALL.map(|s| world.hand(s)))
    }

    /// The frame's position with these four hands, which the caller has read
    /// from one of its worlds.
    pub(crate) fn of(frame: &Frame, hands: [DominoSet; Seat::COUNT]) -> Result<OpenPosition> {
        let trick = frame.prefix().to_vec();
        OpenPosition::new(frame.contract().clone(), hands, frame.leader(), trick, frame.banked())
    }

    pub fn contract(&self) -> &Contract {
        &self.contract
    }

    /// The declaration's rules.
    pub fn physics(&self) -> &Physics {
        self.contract.declaration()
    }

    pub fn hand(&self, seat: Seat) -> DominoSet {
        self.hands[seat.index()]
    }

    pub fn hands(&self) -> [DominoSet; Seat::COUNT] {
        self.hands
    }

    pub fn leader(&self) -> Seat {
        self.leader
    }

    /// The plays to the open trick, the leader's first.
    pub fn trick(&self) -> &[Domino] {
        &self.trick
    }

    /// The points each side has banked in closed tricks, by team index.
    pub fn banked(&self) -> [u32; 2] {
        self.banked
    }

    /// The seat to play, or `None` once every tile is played.
    pub fn next_actor(&self) -> Option<Seat> {
        let seat = self.leader.plus(self.trick.len());
        (!self.hand(seat).is_empty()).then_some(seat)
    }

    /// The next actor's legal plays; empty once every tile is played.
    pub fn legal(&self) -> DominoSet {
        self.next_actor()
            .map_or(DominoSet::EMPTY, |seat| self.physics().legal_plays(self.hand(seat), self.trick.first().copied()))
    }

    /// The position after the next actor plays `tile`, which must be legal.
    /// The fourth play closes the trick: its winner banks its points and leads.
    pub fn play(&self, tile: Domino) -> Result<OpenPosition> {
        let Some(seat) = self.next_actor() else {
            return Err(error("every tile is played"));
        };
        let physics = self.physics();
        if !self.legal().contains(tile) {
            return Err(error(format!("{seat} cannot play {tile}")));
        }
        let mut next = self.clone();
        next.hands[seat.index()].remove(tile);
        next.trick.push(tile);
        if let Ok(plays) = <[Domino; 4]>::try_from(next.trick.as_slice()) {
            next.leader = physics.trick_winner(self.leader, plays);
            next.banked[next.leader.team().index()] += physics.trick_points(plays);
            next.trick.clear();
        }
        Ok(next)
    }

    /// The points not yet banked: a point for each trick still open or to come
    /// and the count on every tile in the hands or on the table.
    pub fn unbanked(&self) -> u32 {
        let physics = self.physics();
        let tiles = self.hands.iter().map(|h| h.len()).sum::<usize>() + self.trick.len();
        let count: u32 =
            self.hands.iter().flat_map(|h| h.iter()).chain(self.trick.iter().copied()).map(|d| physics.count(d)).sum();
        (tiles / Seat::COUNT) as u32 + count
    }

    /// The declaring side's team index.
    fn declaring(&self) -> usize {
        self.contract.declarer().team().index()
    }
}

/// What the solver optimizes, always from the declaring side's point of view:
/// the declaring side maximizes it and the defenders minimize it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Objective {
    /// The declaring side's final points less the defenders', banked points
    /// included: an even number from -42 to 42 over a whole deal.
    Margin,
    /// One if the declaring side's final points reach the contract's
    /// threshold, zero if not.
    Make,
}

/// The exact value of every legal move at a position, under one objective.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootValues {
    pub objective: Objective,
    pub mover: Seat,
    /// Whether the mover is on the declaring side, and so maximizes.
    pub declaring: bool,
    /// Each legal move and its value, in domino order.
    pub moves: Vec<(Domino, i32)>,
}

impl RootValues {
    /// The value of `tile`, if it is a legal move.
    pub fn value(&self, tile: Domino) -> Option<i32> {
        self.moves.iter().find(|(d, _)| *d == tile).map(|(_, v)| *v)
    }

    /// The mover's best value: the greatest if it declares, the least if not.
    pub fn best(&self) -> i32 {
        let values = self.moves.iter().map(|(_, v)| *v);
        let best = if self.declaring { values.max() } else { values.min() };
        best.expect("a root has a legal move")
    }

    /// The moves whose value is the best.
    pub fn tied_best(&self) -> DominoSet {
        let best = self.best();
        self.moves.iter().filter(|(_, v)| *v == best).map(|(d, _)| *d).collect()
    }
}

/// STRATEGY-FUSION FENCE: the exact double-dummy (clairvoyant) solver for one
/// physics. It sees all four hands; its value is never a seat's value, and a
/// value averaged over worlds is strategy fusion. A reference for tests and
/// showcases only (`wiki/topics/strategy-fusion-fence.md`); a player solves
/// information states instead. Its transposition table persists across calls,
/// so solving the positions of one deal in turn reuses the work. When a tile
/// changes hands between calls, the old bounds are discarded.
pub struct ClairvoyantSolver {
    physics: Physics,
    rules: Arc<Rules>,
    table: HashMap<u32, Bounds, BuildHasherDefault<Mix>>,
    owners: [u32; Seat::COUNT],
    nodes: u64,
}

impl ClairvoyantSolver {
    /// STRATEGY-FUSION FENCE: a clairvoyant solver, which sees all four hands.
    /// Never a player's value (`wiki/topics/strategy-fusion-fence.md`).
    pub fn new(physics: &Physics) -> ClairvoyantSolver {
        ClairvoyantSolver {
            physics: physics.clone(),
            rules: Arc::new(Rules::new(physics)),
            table: HashMap::default(),
            owners: [0; Seat::COUNT],
            nodes: 0,
        }
    }

    /// An independent search table under the same precomputed rules.
    pub fn fresh(&self) -> ClairvoyantSolver {
        ClairvoyantSolver {
            physics: self.physics.clone(),
            rules: self.rules.clone(),
            table: HashMap::default(),
            owners: [0; Seat::COUNT],
            nodes: 0,
        }
    }

    pub fn physics(&self) -> &Physics {
        &self.physics
    }

    /// The positions searched so far.
    pub fn nodes(&self) -> u64 {
        self.nodes
    }

    /// The value of every legal move at `position`. Refused once every tile is
    /// played, and for a position under another physics.
    pub fn solve(&mut self, position: &OpenPosition, objective: Objective) -> Result<RootValues> {
        let mover = position.next_actor().ok_or_else(|| error("every tile is played; there is no move to value"))?;
        self.check(position)?;
        self.scope(position);
        let moves = position
            .legal()
            .iter()
            .map(|tile| Ok((tile, self.value(&position.play(tile)?, objective))))
            .collect::<Result<Vec<_>>>()?;
        Ok(RootValues { objective, mover, declaring: mover.team() == position.contract.declarer().team(), moves })
    }

    fn check(&self, position: &OpenPosition) -> Result<()> {
        if *position.physics() != self.physics {
            return Err(error(format!(
                "the solver is for {} and the position is under {}",
                self.physics.name(),
                position.physics().name()
            )));
        }
        Ok(())
    }

    /// With tile ownership fixed, the remaining pool determines all four
    /// hands. Extend that ownership as new tiles appear; if a tile changes
    /// owner, discard the old bounds before admitting the new world. A tile
    /// on the open trick need not be owned: no table entry includes it.
    fn scope(&mut self, position: &OpenPosition) {
        let hands = position.hands.map(|h| h.bits());
        let known = self.owners.iter().fold(0, |all, h| all | h);
        if (0..Seat::COUNT).any(|s| hands[s] & known & !self.owners[s] != 0) {
            self.table.clear();
            self.owners = hands;
        } else {
            for (owner, hand) in self.owners.iter_mut().zip(hands) {
                *owner |= hand;
            }
        }
    }

    /// The value of `position` under best play by both sides. The position
    /// must be under the solver's physics.
    pub fn value(&mut self, position: &OpenPosition, objective: Objective) -> i32 {
        assert!(self.check(position).is_ok(), "a position under the solver's physics");
        self.scope(position);
        let side = position.declaring();
        let banked = [position.banked[side], position.banked[1 - side]].map(|p| p as i32);
        let unbanked = position.unbanked() as i32;
        let node = Node::new(&self.rules, position);
        match objective {
            Objective::Margin => {
                // Each null-window query proves an upper or lower bound on
                // future points. Bisect the integer interval until it is a
                // singleton, reusing the table between queries.
                let (mut lower, mut upper) = (0, unbanked);
                while lower < upper {
                    let threshold = (lower + upper + 1) / 2;
                    let v = self.search(node, threshold - 1, threshold);
                    if v >= threshold {
                        lower = v;
                    } else {
                        upper = v;
                    }
                }
                let future = lower;
                banked[0] - banked[1] + 2 * future - unbanked
            }
            Objective::Make => {
                let need = position.contract.threshold() as i32 - banked[0];
                let made = need <= 0 || (need <= unbanked && self.search(node, need - 1, need) >= need);
                i32::from(made)
            }
        }
    }

    /// The declaring side's points still to come from `node`, by fail-soft
    /// alpha-beta: a result at or below `alpha` is an upper bound on the true
    /// value, one at or above `beta` a lower bound, and one between them exact.
    fn search(&mut self, node: Node, mut alpha: i32, mut beta: i32) -> i32 {
        self.nodes += 1;
        let leader = usize::from(node.leader);
        let n = usize::from(node.n);
        let tricks = node.hands[(leader + n) % Seat::COUNT].count_ones() as i32;
        if tricks == 0 {
            return 0;
        }
        let most = tricks + i32::from(node.unbanked_count);
        if most <= alpha {
            return most;
        }
        if beta <= 0 {
            return 0;
        }
        // The last trick has no choice left. Resolve it once, including any
        // prefix already on the table, without four more search frames.
        if tricks == 1 {
            let mut trick = node.trick;
            for (i, tile) in trick.iter_mut().enumerate().skip(n) {
                *tile = node.hands[(leader + i) % Seat::COUNT].trailing_zeros() as u8;
            }
            let (offset, points) = self.rules.resolve(trick);
            return if (leader + offset) % 2 == usize::from(node.side) { i32::from(points) } else { 0 };
        }
        let mut first = NO_TILE;
        let mut key = 0;
        if n == 0 {
            key = node.key();
            if let Some(b) = self.table.get(&key) {
                let (lo, hi) = (i32::from(b.lo), i32::from(b.hi));
                if lo >= beta || lo == hi {
                    return lo;
                }
                if hi <= alpha {
                    return hi;
                }
                alpha = alpha.max(lo);
                beta = beta.min(hi);
                first = b.best;
            }
        }
        let seat = (leader + n) % Seat::COUNT;
        let hand = node.hands[seat];
        let legal = if n == 0 {
            hand
        } else {
            let follow = hand & self.rules.follow[usize::from(self.rules.led[usize::from(node.trick[0])])];
            if follow == 0 { hand } else { follow }
        };
        let maximizing = seat % 2 == usize::from(node.side);
        let (window_alpha, window_beta) = (alpha, beta);
        let mut best = if maximizing { i32::MIN } else { i32::MAX };
        let mut best_tile = NO_TILE;
        for tile in self.order(&node, seat, legal, first) {
            let v = self.after(node, seat, tile, alpha, beta);
            if maximizing && v > best || !maximizing && v < best {
                best = v;
                best_tile = tile;
            }
            if maximizing {
                alpha = alpha.max(v);
            } else {
                beta = beta.min(v);
            }
            if alpha >= beta {
                break;
            }
        }
        if n == 0 {
            let most = node.hands[leader].count_ones() as i8 + node.unbanked_count as i8;
            let b = self.table.entry(key).or_insert(Bounds { lo: 0, hi: most, best: NO_TILE });
            let v = best as i8;
            if best <= window_alpha {
                b.hi = b.hi.min(v);
            } else if best >= window_beta {
                b.lo = b.lo.max(v);
            } else {
                (b.lo, b.hi) = (v, v);
            }
            b.best = best_tile;
        }
        best
    }

    /// The value after `seat` plays `tile`: the trick's points, if the play
    /// closes it and the declaring side takes it, and the search beyond.
    fn after(&mut self, mut node: Node, seat: usize, tile: u8, alpha: i32, beta: i32) -> i32 {
        node.hands[seat] &= !(1 << tile);
        node.trick[usize::from(node.n)] = tile;
        if node.n < 3 {
            node.n += 1;
            return self.search(node, alpha, beta);
        }
        let (offset, points) = self.rules.resolve(node.trick);
        let winner = (usize::from(node.leader) + offset) % Seat::COUNT;
        let gained = if winner % 2 == usize::from(node.side) { i32::from(points) } else { 0 };
        node.unbanked_count -= points - 1;
        node.leader = winner as u8;
        node.n = 0;
        gained + self.search(node, alpha - gained, beta - gained)
    }

    /// The order to try `legal` in: the table's best move first, then the
    /// rest by a cheap guess at their worth to the mover.
    fn order(&self, node: &Node, seat: usize, legal: u32, first: u8) -> Moves {
        if legal.is_power_of_two() {
            return Moves { tiles: [legal.trailing_zeros() as u8; 7], len: 1, at: 0 };
        }
        let mut moves = Moves { tiles: [0; 7], len: 0, at: 0 };
        let mut scored = [(0i32, 0u8); 7];
        let mut bits = legal;
        while bits != 0 {
            let tile = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            let score = if tile == first { i32::MAX } else { self.rules.guess(node, seat, tile) };
            scored[moves.len] = (score, tile);
            moves.len += 1;
        }
        scored[..moves.len].sort_unstable_by(|a, b| b.cmp(a));
        for (slot, (_, tile)) in moves.tiles.iter_mut().zip(scored) {
            *slot = tile;
        }
        self.rules.distinct(node, seat, moves)
    }
}

const NO_TILE: u8 = u8::MAX;

/// Proven bounds on the declaring side's points still to come from a trick's
/// start, and the move that last proved them.
#[derive(Clone, Copy)]
struct Bounds {
    lo: i8,
    hi: i8,
    best: u8,
}

/// The search's state: the four hands as tile bitmasks, the trick's leader and
/// plays, the declaring side, and the count not yet banked.
#[derive(Clone, Copy)]
struct Node {
    hands: [u32; Seat::COUNT],
    leader: u8,
    n: u8,
    trick: [u8; 4],
    side: u8,
    unbanked_count: u8,
}

impl Node {
    fn new(rules: &Rules, position: &OpenPosition) -> Node {
        let mut trick = [0; 4];
        for (slot, d) in trick.iter_mut().zip(&position.trick) {
            *slot = d.index() as u8;
        }
        let tiles = position.hands.iter().flat_map(|h| h.iter()).chain(position.trick.iter().copied());
        Node {
            hands: position.hands.map(|h| h.bits()),
            leader: position.leader.index() as u8,
            n: position.trick.len() as u8,
            trick,
            side: position.declaring() as u8,
            unbanked_count: tiles.map(|d| rules.count[d.index()]).sum(),
        }
    }

    /// Exact within the table's fixed tile ownership: the remaining pool,
    /// leader and declaring side. The count is a function of the pool.
    fn key(&self) -> u32 {
        let pool = self.hands.iter().fold(0, |all, h| all | h);
        pool | u32::from(self.leader) << 28 | u32::from(self.side) << 30
    }
}

/// One declaration's tables, packed for the search from the physics. Silas
/// derives legality and the winner from the same tables, so these agree.
struct Rules {
    led: [u8; Domino::COUNT],
    follow: [u32; Context::COUNT],
    key: [[u8; Domino::COUNT]; Context::COUNT],
    count: [u8; Domino::COUNT],
    interchangeable: [u32; Domino::COUNT],
    distinguishes: [[u32; Domino::COUNT]; Domino::COUNT],
}

impl Rules {
    fn new(physics: &Physics) -> Rules {
        let count = Domino::ALL.map(|d| u8::try_from(physics.count(d)).expect("a tile counts under 256 points"));
        assert!(count.iter().map(|c| u32::from(*c)).sum::<u32>() + 7 <= 127, "a hand's points fit the table's bounds");
        let mut rules = Rules {
            led: Domino::ALL.map(|d| physics.led_context(d).index() as u8),
            follow: Context::ALL.map(|q| physics.incidence(q).bits()),
            key: Context::ALL.map(|q| Domino::ALL.map(|d| physics.trick_key(d, q))),
            count,
            interchangeable: [0; Domino::COUNT],
            distinguishes: [[0; Domino::COUNT]; Domino::COUNT],
        };
        for a in 0..Domino::COUNT {
            for b in 0..a {
                if rules.count[a] != rules.count[b]
                    || rules.led[a] != rules.led[b]
                    || rules.follow.iter().any(|f| (f >> a & 1) != (f >> b & 1))
                {
                    continue;
                }
                rules.interchangeable[a] |= 1 << b;
                rules.interchangeable[b] |= 1 << a;
                let mut mask = 0;
                for t in 0..Domino::COUNT {
                    if rules.key.iter().any(|k| k[a].cmp(&k[t]) != k[b].cmp(&k[t])) {
                        mask |= 1 << t;
                    }
                }
                rules.distinguishes[a][b] = mask;
                rules.distinguishes[b][a] = mask;
            }
        }
        rules
    }

    /// Two tiles of one hand are interchangeable when they have identical
    /// count, led context and follow membership, and compare alike against
    /// every tile outside that hand (including the open trick). Swapping
    /// their names preserves every legal continuation and trick outcome:
    /// only one tile of their owner can appear in any trick.
    fn distinct(&self, node: &Node, seat: usize, ordered: Moves) -> Moves {
        let mut outside = 0;
        for s in 0..Seat::COUNT {
            if s != seat {
                outside |= node.hands[s];
            }
        }
        for t in &node.trick[..usize::from(node.n)] {
            outside |= 1 << t;
        }
        let mut kept = Moves { tiles: [0; 7], len: 0, at: 0 };
        let mut seen = 0;
        for tile in ordered {
            let t = usize::from(tile);
            let mut candidates = self.interchangeable[t] & seen;
            let mut redundant = false;
            while candidates != 0 {
                let other = candidates.trailing_zeros() as usize;
                candidates &= candidates - 1;
                if self.distinguishes[t][other] & outside == 0 {
                    redundant = true;
                    break;
                }
            }
            if !redundant {
                kept.tiles[kept.len] = tile;
                kept.len += 1;
                seen |= 1 << tile;
            }
        }
        kept
    }

    /// The winner's offset from the leader (the first of the greatest keys)
    /// and the trick's points.
    fn resolve(&self, trick: [u8; 4]) -> (usize, u8) {
        let keys = &self.key[usize::from(self.led[usize::from(trick[0])])];
        let mut best = 0;
        for i in 1..4 {
            if keys[usize::from(trick[i])] > keys[usize::from(trick[best])] {
                best = i;
            }
        }
        (best, 1 + trick.iter().map(|t| self.count[usize::from(*t)]).sum::<u8>())
    }

    /// A move-ordering guess, larger first; it changes the speed, never the
    /// value. A leader tries its strongest tiles first. A follower whose side
    /// is taking the trick gives count; otherwise it tries to take the trick
    /// as cheaply as it can, and failing that throws its least.
    fn guess(&self, node: &Node, seat: usize, tile: u8) -> i32 {
        let t = usize::from(tile);
        let count = i32::from(self.count[t]);
        if node.n == 0 {
            return i32::from(self.key[usize::from(self.led[t])][t]);
        }
        let n = usize::from(node.n);
        let keys = &self.key[usize::from(self.led[usize::from(node.trick[0])])];
        let (mut top, mut taker) = (keys[usize::from(node.trick[0])], 0);
        for i in 1..n {
            if keys[usize::from(node.trick[i])] > top {
                (top, taker) = (keys[usize::from(node.trick[i])], i);
            }
        }
        let ours = (usize::from(node.leader) + taker) % 2 == seat % 2;
        let key = i32::from(keys[t]);
        if ours {
            if key > i32::from(top) { -100 + count - key } else { 100 + count * 4 - key }
        } else if key > i32::from(top) {
            200 - key + count
        } else {
            -count * 4 - key
        }
    }
}

/// Moves in the order to try them.
struct Moves {
    tiles: [u8; 7],
    len: usize,
    at: usize,
}

impl Iterator for Moves {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        (self.at < self.len).then(|| {
            self.at += 1;
            self.tiles[self.at - 1]
        })
    }
}

/// A deterministic hash for the table's 31-bit keys.
#[derive(Default)]
struct Mix(u64);

impl Hasher for Mix {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.write_u64(u64::from(*b));
        }
    }

    fn write_u32(&mut self, k: u32) {
        self.0 = u64::from(k).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 ^= self.0 >> 32;
    }

    fn write_u64(&mut self, k: u64) {
        self.0 = (self.0 ^ k).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

/// A deal with every hand known: the contract (its declaration the physics)
/// and the four hands by seat.
#[derive(Clone, Debug)]
pub struct Deal {
    pub contract: Contract,
    pub hands: [DominoSet; Seat::COUNT],
}

/// A move's value set against the best, under one objective, from the
/// declaring side's point of view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Judged {
    /// The value of the move graded.
    pub given: i32,
    /// The mover's best value.
    pub best: i32,
    /// The moves that reach the best.
    pub tied: DominoSet,
}

impl Judged {
    /// Whether the graded move reaches the best.
    pub fn optimal(&self) -> bool {
        self.given == self.best
    }

    /// How much the graded move gives up against the best, in the objective's
    /// units: zero when it is optimal.
    pub fn shortfall(&self) -> i32 {
        (self.best - self.given).abs()
    }
}

/// One play of a line, graded double-dummy at the position it was made from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grade {
    /// The play's index in the line, from zero.
    pub ply: usize,
    pub seat: Seat,
    pub tile: Domino,
    /// The seat's legal plays; a single one means the play was forced.
    pub legal: DominoSet,
    pub margin: Judged,
    pub make: Judged,
}

/// STRATEGY-FUSION FENCE: grade every play of `line`, from the deal's opening
/// lead, under both objectives, by the clairvoyant solver: a play it calls a
/// mistake may be right for a seat that cannot see the hands
/// (`wiki/topics/strategy-fusion-fence.md`). Refused at the first illegal play.
pub fn clairvoyant_grade_line(deal: &Deal, line: &[Domino]) -> Result<Vec<Grade>> {
    let mut solver = ClairvoyantSolver::new(deal.contract.declaration());
    let mut position = OpenPosition::deal(deal.contract.clone(), deal.hands)?;
    let mut grades = Vec::with_capacity(line.len());
    for (ply, tile) in line.iter().copied().enumerate() {
        let legal = position.legal();
        let next = position.play(tile).map_err(|e| error(format!("play {ply}: {e}")))?;
        let judge = |values: RootValues| Judged {
            given: values.value(tile).expect("a legal move has a value"),
            best: values.best(),
            tied: values.tied_best(),
        };
        let margin = judge(solver.solve(&position, Objective::Margin)?);
        let make = judge(solver.solve(&position, Objective::Make)?);
        let seat = position.next_actor().expect("a legal play has an actor");
        grades.push(Grade { ply, seat, tile, legal, margin, make });
        position = next;
    }
    Ok(grades)
}
