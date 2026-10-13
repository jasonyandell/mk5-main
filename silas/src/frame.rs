//! What a seat knows (a [`Kernel`] and the public trick residue, together a
//! [`Frame`], with the hand's public [`Record`] when it is known), the hidden
//! [`World`]s consistent with it, and [`Holdings`], the only window through
//! which a query reads a world.

use std::sync::Arc;

use crate::nouns::{Context, ContextSet, Domino, DominoSet, Seat, Team};
use crate::physics::{Physics, total_points};
use crate::{Result, error};

/// Three seats are hidden from the one viewer.
pub const HIDDEN_SEATS: usize = 3;

/// The most tiles a seat can hold.
pub const MAX_CAPACITY: usize = 7;

/// One hidden seat: how many tiles it still holds, and the contexts it has
/// publicly failed to follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Hidden {
    pub seat: Seat,
    pub capacity: usize,
    pub voids: ContextSet,
}

/// The viewer's information, independent of any rules: its own hand, the
/// hidden pool, and each hidden seat's capacity and shown voids.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Kernel {
    viewer: Seat,
    viewer_hand: DominoSet,
    pool: DominoSet,
    hidden: [Hidden; HIDDEN_SEATS],
}

impl Kernel {
    pub fn new(
        viewer: Seat,
        viewer_hand: DominoSet,
        pool: DominoSet,
        hidden: [Hidden; HIDDEN_SEATS],
    ) -> Result<Kernel> {
        if !pool.is_disjoint(viewer_hand) {
            return Err(error(format!(
                "the hidden pool meets the viewer's hand at {:?}",
                pool.intersection(viewer_hand)
            )));
        }
        for (i, h) in hidden.iter().enumerate() {
            if h.seat == viewer || hidden[..i].iter().any(|g| g.seat == h.seat) {
                return Err(error("the hidden seats must be the three seats other than the viewer"));
            }
            if h.capacity > MAX_CAPACITY {
                return Err(error(format!("capacity {} exceeds {MAX_CAPACITY}", h.capacity)));
            }
        }
        let capacity: usize = hidden.iter().map(|h| h.capacity).sum();
        if capacity != pool.len() {
            return Err(error(format!("capacities total {capacity} but the hidden pool holds {}", pool.len())));
        }
        Ok(Kernel { viewer, viewer_hand, pool, hidden })
    }

    pub fn viewer(&self) -> Seat {
        self.viewer
    }

    pub fn viewer_hand(&self) -> DominoSet {
        self.viewer_hand
    }

    /// The tiles the viewer cannot see.
    pub fn pool(&self) -> DominoSet {
        self.pool
    }

    pub fn hidden(&self) -> &[Hidden; HIDDEN_SEATS] {
        &self.hidden
    }

    /// The tiles not yet played: the viewer's hand and the hidden pool.
    pub fn live(&self) -> DominoSet {
        self.viewer_hand.union(self.pool)
    }

    /// How many tiles `seat` still holds.
    pub fn capacity(&self, seat: Seat) -> usize {
        if seat == self.viewer { self.viewer_hand.len() } else { self.hidden[self.slot(seat)].capacity }
    }

    fn slot(&self, seat: Seat) -> usize {
        self.hidden.iter().position(|h| h.seat == seat).expect("every other seat is hidden")
    }

    /// The pool tiles hidden slot `slot` may hold: those that follow none of
    /// its shown voids.
    pub fn allowed(&self, physics: &dyn Physics, slot: usize) -> DominoSet {
        self.hidden[slot].voids.iter().fold(self.pool, |out, q| out.difference(physics.incidence(q)))
    }

    /// Assemble a world from the hidden slots' hands. Membership is
    /// [`Kernel::contains`]'s question, not this constructor's.
    pub fn world(&self, hidden_hands: [DominoSet; HIDDEN_SEATS]) -> World {
        let mut hands = [DominoSet::EMPTY; Seat::COUNT];
        hands[self.viewer.index()] = self.viewer_hand;
        for (h, hand) in self.hidden.iter().zip(hidden_hands) {
            hands[h.seat.index()] = hand;
        }
        World { hands }
    }

    /// Whether `world` is consistent with this kernel: the viewer's hand is its
    /// own, each hidden seat holds exactly its capacity of tiles it may hold,
    /// and the hidden hands partition the pool.
    pub fn contains(&self, physics: &dyn Physics, world: &World) -> bool {
        let mut seen = DominoSet::EMPTY;
        for (slot, h) in self.hidden.iter().enumerate() {
            let hand = world.hand(h.seat);
            if hand.len() != h.capacity || !hand.is_subset_of(self.allowed(physics, slot)) || !hand.is_disjoint(seen) {
                return false;
            }
            seen = seen.union(hand);
        }
        world.hand(self.viewer) == self.viewer_hand && seen == self.pool
    }
}

/// The remaining hands of all four seats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct World {
    hands: [DominoSet; Seat::COUNT],
}

impl World {
    pub fn hand(&self, seat: Seat) -> DominoSet {
        self.hands[seat.index()]
    }
}

/// What the auction settled: the seat that won it, and the points its side
/// must take. A marks bid (84 and up) is the whole 42 here; marks are
/// scoring, which Silas does not have.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Contract {
    pub declarer: Seat,
    pub bid: u32,
}

/// The public record of a hand: its contract and every play so far, each
/// attributed to its seat, in play order. The declarer leads the first trick,
/// and each trick is a run of four plays. What the record determines (the
/// banked points, each trick's leader and plays, the shown voids) is derived
/// from it, never supplied beside it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Record {
    contract: Contract,
    plays: Vec<(Seat, Domino)>,
    banked: [u32; 2],
}

impl Record {
    pub fn contract(&self) -> Contract {
        self.contract
    }

    /// Every play so far, in order.
    pub fn plays(&self) -> &[(Seat, Domino)] {
        &self.plays
    }

    /// The points each team has taken in the closed tricks, by team index.
    pub fn banked(&self) -> [u32; 2] {
        self.banked
    }

    /// The tricks so far, the open one last: runs of at most four plays.
    pub fn tricks(&self) -> std::slice::Chunks<'_, (Seat, Domino)> {
        self.plays.chunks(Seat::COUNT)
    }

    /// The number of the trick being played or about to be led, from one;
    /// `None` once all seven have closed.
    pub fn trick_number(&self) -> Option<usize> {
        (self.plays.len() < Domino::COUNT).then_some(self.plays.len() / Seat::COUNT + 1)
    }

    /// The contexts `seat` has failed to follow, read off the plays.
    pub fn shown_voids(&self, physics: &dyn Physics, seat: Seat) -> ContextSet {
        let mut voids = ContextSet::EMPTY;
        for trick in self.tricks() {
            let led = physics.led_context(trick[0].1);
            for (actor, tile) in &trick[1..] {
                if *actor == seat && !physics.follows(*tile, led) {
                    voids.insert(led);
                }
            }
        }
        voids
    }

    /// The points the declaring side still needs to make its bid (zero once
    /// made), and the points the other side still needs to set it (zero once
    /// set), under the physics's count.
    pub fn to_make_and_set(&self, physics: &dyn Physics) -> (u32, u32) {
        let declaring = self.contract.declarer.team();
        let setting = if declaring == Team::T0 { Team::T1 } else { Team::T0 };
        let to_make = self.contract.bid.saturating_sub(self.banked[declaring.index()]);
        let to_set = (total_points(physics) + 1 - self.contract.bid).saturating_sub(self.banked[setting.index()]);
        (to_make, to_set)
    }
}

/// A kernel under a physics, with the public residue of the current trick
/// (its leader, the plays so far, and every tile explicitly known played)
/// and, when the hand's public record is known, that record.
/// [`Frame::new`] checks coherence, not reachability from a deal;
/// [`Frame::from_record`] derives everything public from the record.
#[derive(Clone, Debug)]
pub struct Frame {
    physics: Arc<dyn Physics>,
    kernel: Kernel,
    leader: Seat,
    prefix: Vec<Domino>,
    played: DominoSet,
    record: Option<Record>,
}

impl PartialEq for Frame {
    fn eq(&self, other: &Frame) -> bool {
        self.physics.identity() == other.physics.identity()
            && (&self.kernel, self.leader, &self.prefix, self.played, &self.record)
                == (&other.kernel, other.leader, &other.prefix, other.played, &other.record)
    }
}
impl Eq for Frame {}

impl Frame {
    /// A frame without a record. The predicates that read the record (the
    /// contract, the score, earlier tricks, the viewer's own shown voids) are
    /// undefined on it.
    pub fn new(
        physics: Arc<dyn Physics>,
        kernel: Kernel,
        leader: Seat,
        prefix: Vec<Domino>,
        played: DominoSet,
    ) -> Result<Frame> {
        let unique: DominoSet = prefix.iter().copied().collect();
        if prefix.len() > 3
            || unique.len() != prefix.len()
            || !unique.is_subset_of(played)
            || !played.is_disjoint(kernel.live())
        {
            return Err(error(
                "incoherent public residue: the prefix must be distinct played tiles, at most three, and no played tile is live",
            ));
        }
        Ok(Frame { physics, kernel, leader, prefix, played, record: None })
    }

    /// `viewer`'s frame from the public record: the contract, the plays so
    /// far, and the viewer's remaining `hand`. The leader, the open trick, the
    /// played set, each hidden seat's capacity and shown voids, and the banked
    /// points are read off the plays. Refused when the plays break turn order
    /// (the declarer leads the first trick, each trick's winner the next),
    /// repeat a tile, give a seat more than seven, follow a context their seat
    /// has shown void in, or include an illegal play by the viewer; when the
    /// bid exceeds the points in the hand; and when no deal of the hidden
    /// tiles fits.
    pub fn from_record(
        physics: Arc<dyn Physics>,
        viewer: Seat,
        hand: DominoSet,
        contract: Contract,
        plays: Vec<(Seat, Domino)>,
    ) -> Result<Frame> {
        if contract.bid > total_points(physics.as_ref()) {
            return Err(error(format!("a bid of {} exceeds the points in a hand", contract.bid)));
        }
        let dealt =
            plays.iter().filter(|(seat, _)| *seat == viewer).fold(hand, |h, (_, d)| h.union(DominoSet::single(*d)));
        let mut held = dealt;
        let mut played = DominoSet::EMPTY;
        let mut remaining = [MAX_CAPACITY; Seat::COUNT];
        let mut voids = [ContextSet::EMPTY; Seat::COUNT];
        let mut banked = [0u32; 2];
        let mut leader = contract.declarer;
        let mut prefix: Vec<Domino> = Vec::new();
        for (seat, tile) in plays.iter().copied() {
            if seat != leader.plus(prefix.len()) || remaining[seat.index()] == 0 || !played.insert(tile) {
                return Err(error(format!(
                    "the record's play {seat} {tile} is out of turn, repeated, or one too many"
                )));
            }
            let lead = prefix.first().copied();
            if seat == viewer {
                if !physics.legal_plays(held, lead).contains(tile) {
                    return Err(error(format!("the viewer's play {tile} in the record is illegal")));
                }
                held.remove(tile);
            }
            if let Some(q) = lead.map(|d| physics.led_context(d)) {
                if !physics.follows(tile, q) {
                    voids[seat.index()].insert(q);
                } else if voids[seat.index()].contains(q) {
                    return Err(error(format!("{seat} follows {q} with {tile} after showing void in it")));
                }
            }
            remaining[seat.index()] -= 1;
            prefix.push(tile);
            if let Ok(trick) = <[Domino; 4]>::try_from(prefix.as_slice()) {
                leader = physics.trick_winner(leader, trick);
                banked[leader.team().index()] += physics.trick_points(trick);
                prefix.clear();
            }
        }
        if !hand.is_disjoint(played) || remaining[viewer.index()] != hand.len() {
            return Err(error("the viewer's hand disagrees with its plays in the record"));
        }
        let hidden: [Hidden; HIDDEN_SEATS] = std::array::from_fn(|i| {
            let seat = viewer.plus(i + 1);
            Hidden { seat, capacity: remaining[seat.index()], voids: voids[seat.index()] }
        });
        let kernel = Kernel::new(viewer, hand, DominoSet::FULL.difference(hand.union(played)), hidden)?;
        if physics.count_worlds(&kernel) == 0 {
            return Err(error("no deal of the hidden tiles fits the record"));
        }
        let record = Some(Record { contract, plays, banked });
        Ok(Frame { physics, kernel, leader, prefix, played, record })
    }

    /// The hand's public record, when the frame was built from one.
    pub fn record(&self) -> Option<&Record> {
        self.record.as_ref()
    }

    /// The frame after one play whose successor kernel, leader and prefix the
    /// caller has computed. A frame with a record is rebuilt from its record
    /// extended by the play, and must agree with them.
    pub(crate) fn after(
        &self,
        kernel: Kernel,
        leader: Seat,
        prefix: Vec<Domino>,
        play: (Seat, Domino),
    ) -> Result<Frame> {
        let Some(record) = &self.record else {
            return Frame::new(
                self.physics.clone(),
                kernel,
                leader,
                prefix,
                self.played.union(DominoSet::single(play.1)),
            );
        };
        let mut plays = record.plays.clone();
        plays.push(play);
        let next = Frame::from_record(self.physics.clone(), kernel.viewer, kernel.viewer_hand, record.contract, plays)?;
        if (&next.kernel, next.leader, &next.prefix) != (&kernel, leader, &prefix) {
            return Err(error("the stepped frame disagrees with its record"));
        }
        Ok(next)
    }

    pub fn physics(&self) -> &dyn Physics {
        self.physics.as_ref()
    }

    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    pub fn leader(&self) -> Seat {
        self.leader
    }

    pub fn prefix(&self) -> &[Domino] {
        &self.prefix
    }

    /// Tiles explicitly supplied as played; never inferred from `live`.
    pub fn played(&self) -> DominoSet {
        self.played
    }

    /// The tile that led the current trick.
    pub fn lead(&self) -> Option<Domino> {
        self.prefix.first().copied()
    }

    pub fn led_context(&self) -> Option<Context> {
        self.lead().map(|d| self.physics.led_context(d))
    }

    /// The seat to play next, unless it has no tiles left.
    pub fn next_actor(&self) -> Option<Seat> {
        let seat = self.leader.plus(self.prefix.len());
        (self.kernel.capacity(seat) > 0).then_some(seat)
    }

    /// The seat currently taking the trick: the first of the greatest keys.
    pub fn current_winner(&self) -> Option<Seat> {
        let led = self.led_context()?;
        let key = |i: usize| self.physics.trick_key(self.prefix[i], led);
        let best = (1..self.prefix.len()).fold(0, |best, i| if key(i) > key(best) { i } else { best });
        Some(self.leader.plus(best))
    }

    pub fn contains(&self, world: &World) -> bool {
        self.kernel.contains(self.physics(), world)
    }

    pub fn count_worlds(&self) -> u128 {
        self.physics.count_worlds(&self.kernel)
    }

    pub fn worlds(&self) -> Box<dyn Iterator<Item = World> + '_> {
        self.physics.worlds(&self.kernel)
    }
}

/// Read access to one world, limited to what a belief resolves. A belief may
/// leave some pool tiles unresolved: each of its points then stands for every
/// world that agrees on the resolved tiles. A read that could tell those
/// worlds apart is refused, so an answer is exact for every world it stands
/// for.
#[derive(Clone, Copy, Debug)]
pub struct Holdings<'a> {
    frame: &'a Frame,
    world: &'a World,
    unresolved: DominoSet,
}

impl<'a> Holdings<'a> {
    pub(crate) fn new(frame: &'a Frame, world: &'a World, unresolved: DominoSet) -> Holdings<'a> {
        debug_assert!(frame.contains(world) && unresolved.is_subset_of(frame.kernel.pool));
        Holdings { frame, world, unresolved }
    }

    pub fn frame(&self) -> &'a Frame {
        self.frame
    }

    /// The tiles of `tiles` that `seat` holds.
    pub fn hand_within(&self, seat: Seat, tiles: DominoSet) -> Result<DominoSet> {
        let kernel = &self.frame.kernel;
        if seat != kernel.viewer {
            let slot = kernel.slot(seat);
            let open = if kernel.hidden[slot].capacity == 0 {
                DominoSet::EMPTY
            } else {
                tiles.intersection(self.unresolved).intersection(kernel.allowed(self.frame.physics(), slot))
            };
            if !open.is_empty() {
                return Err(error(format!("reading {seat} needs the unresolved tiles {open:?}; resolve them")));
            }
        }
        Ok(self.world.hand(seat).intersection(tiles))
    }

    pub fn holds(&self, seat: Seat, tile: Domino) -> Result<bool> {
        Ok(!self.hand_within(seat, DominoSet::single(tile))?.is_empty())
    }

    /// The seat's whole remaining hand.
    pub fn hand(&self, seat: Seat) -> Result<DominoSet> {
        self.hand_within(seat, DominoSet::FULL)
    }
}
