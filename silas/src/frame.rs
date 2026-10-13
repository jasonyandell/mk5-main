//! What a seat knows. A [`Frame`] is the hand's public record as one seat
//! sees it: the [`Contract`] (whose declaration is the [`Physics`]), the
//! viewer, the hand it was dealt, and every play so far. Everything else (the
//! leader, the open trick, the played set, the banked points, each seat's
//! shown voids and remaining size, the viewer's [`Kernel`]) is derived from
//! those by one replay, never supplied. A [`World`] is one deal of the hidden
//! tiles consistent with the frame, and [`Holdings`] the only window through
//! which a query reads one.

use std::cell::Cell;
use std::fmt;
use std::str::FromStr;

use crate::nouns::{Context, ContextSet, Domino, DominoSet, Seat, Team};
use crate::physics::Physics;
use crate::{Error, Result, error};

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
/// hidden pool, and each hidden seat's capacity and shown voids. A frame
/// derives one; a belief over a part of the pool builds others.
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

    pub(crate) fn slot(&self, seat: Seat) -> usize {
        self.hidden.iter().position(|h| h.seat == seat).expect("every other seat is hidden")
    }

    /// The pool tiles hidden slot `slot` may hold: those that follow none of
    /// its shown voids.
    pub fn allowed(&self, physics: &Physics, slot: usize) -> DominoSet {
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
    pub fn contains(&self, physics: &Physics, world: &World) -> bool {
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

/// A bid as the table says it: a points bid (30 to 41), or a marks bid (42
/// for one mark, 84 for two, and so on). Range legality is the auction's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Bid {
    Points(u32),
    Marks(u32),
}

impl Bid {
    /// The bid the table calls `n`: below 42 a points bid, a multiple of 42 a
    /// marks bid.
    pub fn number(n: u32) -> Result<Bid> {
        match n {
            0..42 => Ok(Bid::Points(n)),
            _ if n.is_multiple_of(42) => Ok(Bid::Marks(n / 42)),
            _ => Err(error(format!("a bid of {n} exceeds the points in a hand, and is not a marks bid (42 per mark)"))),
        }
    }

    /// The number the table calls the bid: `n` points, or 42 per mark.
    pub fn value(self) -> u32 {
        match self {
            Bid::Points(n) => n,
            Bid::Marks(m) => 42 * m,
        }
    }
}

impl fmt::Display for Bid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value())
    }
}

impl FromStr for Bid {
    type Err = Error;

    fn from_str(s: &str) -> Result<Bid> {
        Bid::number(s.parse().map_err(|_| error(format!("not a bid: {s}")))?)
    }
}

/// What the auction settled: the seat that won it, its bid, and the
/// declaration, whose rules are the hand's [`Physics`]. The points the
/// declaring side must take (the threshold) and the marks at stake are
/// derived: a points bid needs its points and pays one mark; a marks bid
/// needs every point and pays its marks.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Contract {
    declarer: Seat,
    bid: Bid,
    declaration: Physics,
}

impl Contract {
    /// Refused for a points bid of every point or more (that is a marks bid)
    /// and for a marks bid of no marks.
    pub fn new(declarer: Seat, bid: Bid, declaration: Physics) -> Result<Contract> {
        match bid {
            Bid::Points(n) if n >= declaration.total_points() => {
                Err(error(format!("a bid of {n} exceeds the points in a hand; bid every point as marks")))
            }
            Bid::Marks(0) => Err(error("a marks bid is for one mark or more")),
            _ => Ok(Contract { declarer, bid, declaration }),
        }
    }

    pub fn declarer(&self) -> Seat {
        self.declarer
    }

    pub fn bid(&self) -> Bid {
        self.bid
    }

    /// The declaration's rules.
    pub fn declaration(&self) -> &Physics {
        &self.declaration
    }

    /// The points the declaring side must take.
    pub fn threshold(&self) -> u32 {
        match self.bid {
            Bid::Points(n) => n,
            Bid::Marks(_) => self.declaration.total_points(),
        }
    }

    /// The marks the hand is played for.
    pub fn stake(&self) -> u32 {
        match self.bid {
            Bid::Points(_) => 1,
            Bid::Marks(m) => m,
        }
    }

    /// The team that receives the stake when the declaring side ends with
    /// `points`.
    pub fn award(&self, points: u32) -> (Team, u32) {
        let declaring = self.declarer.team();
        let team = if points >= self.threshold() { declaring } else { declaring.other() };
        (team, self.stake())
    }
}

/// The hand's public record as one seat sees it: the contract, the viewer,
/// the hand it was dealt, and every play so far, in order. Each play's seat,
/// each trick's leader, the open trick, the banked points, the shown voids
/// and the viewer's kernel are derived from these by one replay when the
/// frame is built, and are never supplied.
#[derive(Clone)]
pub struct Frame {
    contract: Contract,
    viewer: Seat,
    dealt: DominoSet,
    plays: Vec<Domino>,
    replay: Replay,
}

/// What the record determines, computed once by [`Frame::new`].
#[derive(Clone, Debug)]
struct Replay {
    history: Vec<(Seat, Domino)>,
    /// The leader of the open trick, or of the next.
    leader: Seat,
    banked: [u32; 2],
    voids: [ContextSet; Seat::COUNT],
    kernel: Kernel,
}

impl PartialEq for Frame {
    fn eq(&self, other: &Frame) -> bool {
        (&self.contract, self.viewer, self.dealt, &self.plays)
            == (&other.contract, other.viewer, other.dealt, &other.plays)
    }
}

impl Eq for Frame {}

impl fmt::Debug for Frame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Frame")
            .field("physics", &self.contract.declaration)
            .field("declarer", &self.contract.declarer)
            .field("bid", &self.contract.bid)
            .field("viewer", &self.viewer)
            .field("dealt", &self.dealt)
            .field("history", &self.replay.history)
            .finish()
    }
}

impl Frame {
    /// `viewer`'s frame: dealt `dealt` (seven tiles) under `contract`, after
    /// `plays`. Each play's seat is read off the record: the declarer leads
    /// the first trick, each trick's winner the next. Refused when a tile is
    /// played twice, when the viewer plays a tile it was not dealt or plays
    /// illegally, when another seat plays one of the viewer's tiles, when a
    /// seat plays a tile of a context it has shown void in, and when no deal
    /// of the hidden tiles fits.
    pub fn new(contract: Contract, viewer: Seat, dealt: DominoSet, plays: Vec<Domino>) -> Result<Frame> {
        Frame::replayed(contract, viewer, dealt, plays, None)
    }

    /// The same frame from the record as a scorekeeper writes it: the
    /// viewer's remaining `hand`, and each play attributed to its seat. The
    /// attribution is checked against the replay.
    pub fn from_history(
        contract: Contract,
        viewer: Seat,
        hand: DominoSet,
        history: &[(Seat, Domino)],
    ) -> Result<Frame> {
        let mine: DominoSet = history.iter().filter(|(s, _)| *s == viewer).map(|(_, d)| *d).collect();
        let played: DominoSet = history.iter().map(|(_, d)| *d).collect();
        if !hand.is_disjoint(played) || hand.len() + mine.len() != MAX_CAPACITY {
            return Err(error("the viewer's hand disagrees with its plays in the record"));
        }
        let seats: Vec<Seat> = history.iter().map(|(s, _)| *s).collect();
        let plays = history.iter().map(|(_, d)| *d).collect();
        Frame::replayed(contract, viewer, hand.union(mine), plays, Some(&seats))
    }

    fn replayed(
        contract: Contract,
        viewer: Seat,
        dealt: DominoSet,
        plays: Vec<Domino>,
        attributed: Option<&[Seat]>,
    ) -> Result<Frame> {
        if dealt.len() != MAX_CAPACITY {
            return Err(error(format!(
                "the viewer's hand disagrees with the record: it was dealt {} tiles, not seven",
                dealt.len()
            )));
        }
        if plays.len() > Domino::COUNT {
            return Err(error("a record holds at most 28 plays"));
        }
        let physics = contract.declaration();
        let mut held = dealt;
        let mut played = DominoSet::EMPTY;
        let mut remaining = [MAX_CAPACITY; Seat::COUNT];
        let mut voids = [ContextSet::EMPTY; Seat::COUNT];
        let mut banked = [0u32; 2];
        let mut leader = contract.declarer();
        let mut history = Vec::with_capacity(plays.len());
        let mut start = 0;
        for (i, tile) in plays.iter().copied().enumerate() {
            let seat = leader.plus(i - start);
            if let Some(claimed) = attributed.map(|seats| seats[i])
                && claimed != seat
            {
                return Err(error(format!("the record's play {claimed} {tile} is out of turn: {seat} is to play")));
            }
            if !played.insert(tile) {
                return Err(error(format!("the record repeats {tile}")));
            }
            let lead = (i > start).then(|| plays[start]);
            if seat == viewer {
                if !held.contains(tile) {
                    return Err(error(format!("the record gives the viewer {tile}, which it was not dealt")));
                }
                if !physics.legal_plays(held, lead).contains(tile) {
                    return Err(error(format!("the viewer's play {tile} in the record is illegal")));
                }
                held.remove(tile);
            } else if dealt.contains(tile) {
                return Err(error(format!("the record gives {seat} the viewer's {tile}")));
            }
            if let Some(q) = voids[seat.index()].iter().find(|q| physics.follows(tile, *q)) {
                return Err(error(format!("{seat} plays {tile}, which follows {q}, after showing void in it")));
            }
            if let Some(q) = lead.map(|d| physics.led_context(d))
                && !physics.follows(tile, q)
            {
                voids[seat.index()].insert(q);
            }
            remaining[seat.index()] -= 1;
            history.push((seat, tile));
            if let Ok(trick) = <[Domino; 4]>::try_from(&plays[start..=i]) {
                leader = physics.trick_winner(leader, trick);
                banked[leader.team().index()] += physics.trick_points(trick);
                start = i + 1;
            }
        }
        let hidden: [Hidden; HIDDEN_SEATS] = std::array::from_fn(|i| {
            let seat = viewer.plus(i + 1);
            Hidden { seat, capacity: remaining[seat.index()], voids: voids[seat.index()] }
        });
        let kernel = Kernel::new(viewer, held, DominoSet::FULL.difference(held.union(played)), hidden)?;
        if physics.count_worlds(&kernel) == 0 {
            return Err(error("no deal of the hidden tiles fits the record"));
        }
        let replay = Replay { history, leader, banked, voids, kernel };
        Ok(Frame { contract, viewer, dealt, plays, replay })
    }

    /// The frame after one more play.
    pub fn after(&self, tile: Domino) -> Result<Frame> {
        let mut plays = self.plays.clone();
        plays.push(tile);
        Frame::new(self.contract.clone(), self.viewer, self.dealt, plays)
    }

    pub fn contract(&self) -> &Contract {
        &self.contract
    }

    /// The declaration's rules.
    pub fn physics(&self) -> &Physics {
        self.contract.declaration()
    }

    pub fn viewer(&self) -> Seat {
        self.viewer
    }

    /// The seven tiles the viewer was dealt.
    pub fn dealt(&self) -> DominoSet {
        self.dealt
    }

    /// Every play so far, in order.
    pub fn plays(&self) -> &[Domino] {
        &self.plays
    }

    /// Every play so far with the seat that made it.
    pub fn history(&self) -> &[(Seat, Domino)] {
        &self.replay.history
    }

    /// The tricks so far, the open one last: runs of at most four plays.
    pub fn tricks(&self) -> std::slice::Chunks<'_, (Seat, Domino)> {
        self.replay.history.chunks(Seat::COUNT)
    }

    pub fn kernel(&self) -> &Kernel {
        &self.replay.kernel
    }

    /// The seat that led the open trick, or leads the next.
    pub fn leader(&self) -> Seat {
        self.replay.leader
    }

    /// The open trick's plays.
    pub fn prefix(&self) -> &[Domino] {
        &self.plays[self.plays.len() - self.plays.len() % Seat::COUNT..]
    }

    /// The tiles played so far.
    pub fn played(&self) -> DominoSet {
        self.plays.iter().copied().collect()
    }

    /// The tile that led the open trick.
    pub fn lead(&self) -> Option<Domino> {
        self.prefix().first().copied()
    }

    pub fn led_context(&self) -> Option<Context> {
        self.lead().map(|d| self.physics().led_context(d))
    }

    /// The seat to play next; none once every tile is played.
    pub fn next_actor(&self) -> Option<Seat> {
        (self.plays.len() < Domino::COUNT).then(|| self.leader().plus(self.prefix().len()))
    }

    /// The seat currently taking the open trick: the first of the greatest
    /// keys.
    pub fn current_winner(&self) -> Option<Seat> {
        let led = self.led_context()?;
        let prefix = self.prefix();
        let key = |i: usize| self.physics().trick_key(prefix[i], led);
        let best = (1..prefix.len()).fold(0, |best, i| if key(i) > key(best) { i } else { best });
        Some(self.leader().plus(best))
    }

    /// The points each team has taken in the closed tricks, by team index.
    pub fn banked(&self) -> [u32; 2] {
        self.replay.banked
    }

    /// The contexts `seat` has failed to follow.
    pub fn shown_voids(&self, seat: Seat) -> ContextSet {
        self.replay.voids[seat.index()]
    }

    /// The number of the trick being played or about to be led, from one;
    /// `None` once all seven have closed.
    pub fn trick_number(&self) -> Option<usize> {
        (self.plays.len() < Domino::COUNT).then_some(self.plays.len() / Seat::COUNT + 1)
    }

    /// The points the declaring side still needs to reach its threshold
    /// (zero once made), and the points the other side still needs to set it
    /// (zero once set).
    pub fn to_make_and_set(&self) -> (u32, u32) {
        let declaring = self.contract.declarer().team();
        let threshold = self.contract.threshold();
        let to_make = threshold.saturating_sub(self.replay.banked[declaring.index()]);
        let to_set = (self.physics().total_points() + 1 - threshold)
            .saturating_sub(self.replay.banked[declaring.other().index()]);
        (to_make, to_set)
    }

    pub fn contains(&self, world: &World) -> bool {
        self.kernel().contains(self.physics(), world)
    }

    pub fn count_worlds(&self) -> u128 {
        self.physics().count_worlds(self.kernel())
    }

    pub fn worlds(&self) -> impl Iterator<Item = World> + '_ {
        self.physics().worlds(self.kernel())
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
    /// When set, a refused read records the tiles it needed here.
    needed: Option<&'a Cell<DominoSet>>,
}

impl<'a> Holdings<'a> {
    pub(crate) fn new(frame: &'a Frame, world: &'a World, unresolved: DominoSet) -> Holdings<'a> {
        debug_assert!(frame.contains(world) && unresolved.is_subset_of(frame.kernel().pool));
        Holdings { frame, world, unresolved, needed: None }
    }

    /// The same window, recording into `needed` the tiles any refused read
    /// needed.
    pub(crate) fn probing(self, needed: &'a Cell<DominoSet>) -> Holdings<'a> {
        Holdings { needed: Some(needed), ..self }
    }

    pub fn frame(&self) -> &'a Frame {
        self.frame
    }

    /// The tiles of `tiles` that `seat` holds.
    pub fn hand_within(&self, seat: Seat, tiles: DominoSet) -> Result<DominoSet> {
        let kernel = self.frame.kernel();
        if seat != kernel.viewer {
            let slot = kernel.slot(seat);
            let open = if kernel.hidden[slot].capacity == 0 {
                DominoSet::EMPTY
            } else {
                tiles.intersection(self.unresolved).intersection(kernel.allowed(self.frame.physics(), slot))
            };
            if !open.is_empty() {
                if let Some(needed) = self.needed {
                    needed.set(needed.get().union(open));
                }
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
