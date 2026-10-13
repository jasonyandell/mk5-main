//! What a seat knows (a [`Kernel`] and the public trick residue, together a
//! [`Frame`]), the hidden [`World`]s consistent with it, and [`Holdings`], the
//! only window through which a query reads a world.

use std::sync::Arc;

use crate::nouns::{Context, ContextSet, Domino, DominoSet, Seat};
use crate::physics::Physics;
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

/// A kernel under a physics, with the public residue of the current trick:
/// its leader, the plays so far, and every tile explicitly known played.
/// The constructor checks coherence, not reachability from a deal.
#[derive(Clone, Debug)]
pub struct Frame {
    physics: Arc<dyn Physics>,
    kernel: Kernel,
    leader: Seat,
    prefix: Vec<Domino>,
    played: DominoSet,
}

impl PartialEq for Frame {
    fn eq(&self, other: &Frame) -> bool {
        self.physics.identity() == other.physics.identity()
            && (&self.kernel, self.leader, &self.prefix, self.played)
                == (&other.kernel, other.leader, &other.prefix, other.played)
    }
}
impl Eq for Frame {}

impl Frame {
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
        Ok(Frame { physics, kernel, leader, prefix, played })
    }

    pub fn physics(&self) -> &dyn Physics {
        self.physics.as_ref()
    }

    pub(crate) fn physics_handle(&self) -> &Arc<dyn Physics> {
        &self.physics
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

    pub fn led_context(&self) -> Option<Context> {
        self.prefix.first().map(|d| self.physics.led_context(*d))
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
