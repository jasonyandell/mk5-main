//! Silas over Zeb (the `silas` feature): [`ZebPhysics`], one declaration's
//! [`Rules`] and Zeb's [`Fiber`] behind Silas's `Physics` trait.
//!
//! The two crates number the game the same way, and the maps below say so
//! explicitly rather than by coincidence:
//!
//! - **tiles**: Zeb's tile id `t` is Silas's domino of index `t`. Both number
//!   the double-six set triangularly, (high, low) ↦ high·(high+1)/2 + low, so
//!   0-0 is 0 and 6-6 is 27; a tile mask is a `DominoSet`'s bits.
//! - **contexts**: Zeb's led suit `k` in 0..=6 is the pip context `q{k}`, and
//!   its led suit 7, the called suit, is `q*`.
//! - **seats**: Zeb's seat `s` is `S{s}`. Both play clockwise from the leader
//!   (`(leader + i) % 4` is `leader.plus(i)`) and pair `s` with `s + 2`
//!   (Zeb banks a trick to team `seat % 2`, Silas's `Team` is `T{s % 2}`).
//!
//! Every rule comes from Zeb's own code: the four tables of [`Rules`] are
//! the required methods, and `legal_plays`, `trick_winner` and
//! `trick_points` are overridden with [`Rules::legal`],
//! [`Rules::winner_offset`] and [`Rules::trick_points`], so Silas reads Zeb's
//! rules rather than re-deriving them. The fiber of hidden worlds is Zeb's
//! counting table: `count_worlds` is [`Fiber::size`] and `worlds` reads
//! [`Fiber::world_at`] in [`Fiber::worlds`]'s order, lazily.

use std::fmt;

use silas::{Context, Domino, DominoSet, Kernel, Physics, Seat, World};

use crate::fiber::Fiber;
use crate::rules::{Rules, N_DECLS};

/// Each declaration's name in a physics identity (`zeb/P5`): the pip suits
/// trump `P0`..`P6` (the names `silas::reference::Straight42` uses), doubles
/// trump `DT`, the doubles as their own unpowered suit `DS` (nello's
/// algebra), and no trump `NT`.
pub const DECL_NAMES: [&str; N_DECLS as usize] = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "DT", "DS", "NT"];

/// The domino with Zeb's tile id `t`, or None outside 0..28.
pub fn domino(t: u8) -> Option<Domino> {
    Domino::from_index(t as usize)
}

/// Zeb's tile id of a domino.
pub fn tile(d: Domino) -> u8 {
    d.index() as u8
}

/// The dominoes of a Zeb tile mask, or None if it has a bit above 27.
pub fn dominoes(mask: u32) -> Option<DominoSet> {
    DominoSet::from_bits(mask)
}

/// Zeb's tile mask of a set of dominoes.
pub fn mask(set: DominoSet) -> u32 {
    set.bits()
}

/// The context of Zeb's led suit `k`, or None outside 0..8.
pub fn context(k: u8) -> Option<Context> {
    Context::from_index(k as usize)
}

/// Zeb's led suit of a context.
pub fn led_suit(q: Context) -> u8 {
    q.index() as u8
}

/// The seat of Zeb's seat `s`, or None outside 0..4.
pub fn seat(s: u8) -> Option<Seat> {
    Seat::from_index(s as usize)
}

/// Zeb's seat of a seat.
pub fn seat_id(s: Seat) -> u8 {
    s.index() as u8
}

/// One declaration of Zeb's rules as a Silas physics.
pub struct ZebPhysics {
    rules: Rules,
}

impl ZebPhysics {
    /// The physics of declaration `decl`, or None outside 0..N_DECLS.
    pub fn new(decl: u8) -> Option<ZebPhysics> {
        Rules::new(decl).map(|rules| ZebPhysics { rules })
    }

    pub fn rules(&self) -> &Rules {
        &self.rules
    }

    /// Zeb's fiber of the kernel: the played tiles are those not live, each
    /// hidden seat's size is its capacity, and its forbidden tiles are the
    /// followers of its shown voids.
    fn fiber(&self, kernel: &Kernel) -> Fiber {
        let viewer = seat_id(kernel.viewer());
        let mut sizes = [0u8; 4];
        let mut forbid = [0u32; 4];
        sizes[viewer as usize] = kernel.viewer_hand().len() as u8;
        for h in kernel.hidden() {
            sizes[h.seat.index()] = h.capacity as u8;
            forbid[h.seat.index()] = h.voids.iter().fold(0, |m, q| m | self.rules.follow_bits[q.index()]);
        }
        let played = mask(DominoSet::FULL.difference(kernel.live()));
        Fiber::new(viewer, mask(kernel.viewer_hand()), played, sizes, forbid)
    }
}

impl fmt::Debug for ZebPhysics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ZebPhysics").field(&DECL_NAMES[self.rules.decl as usize]).finish()
    }
}

/// A world of the kernel from Zeb's four remaining hands, absolute seats.
fn world(kernel: &Kernel, hands: [u32; 4]) -> World {
    kernel.world(kernel.hidden().map(|h| dominoes(hands[h.seat.index()]).expect("Zeb's hands are tile masks")))
}

impl Physics for ZebPhysics {
    fn identity(&self) -> String {
        format!("zeb/{}", DECL_NAMES[self.rules.decl as usize])
    }

    fn led_context(&self, d: Domino) -> Context {
        context(self.rules.led_suit[d.index()]).expect("Zeb's led suits are 0..8")
    }

    fn incidence(&self, q: Context) -> DominoSet {
        dominoes(self.rules.follow_bits[q.index()]).expect("Zeb's follow sets are tile masks")
    }

    fn trick_key(&self, d: Domino, led: Context) -> u8 {
        self.rules.rank[led.index()][d.index()]
    }

    fn count(&self, d: Domino) -> u32 {
        u32::from(self.rules.count[d.index()])
    }

    fn count_worlds(&self, kernel: &Kernel) -> u128 {
        u128::try_from(self.fiber(kernel).size()).expect("a count is nonnegative")
    }

    fn worlds<'a>(&'a self, kernel: &'a Kernel) -> Box<dyn Iterator<Item = World> + 'a> {
        let mut fiber = self.fiber(kernel);
        Box::new((0..fiber.size()).map(move |k| world(kernel, fiber.world_at(k))))
    }

    fn legal_plays(&self, hand: DominoSet, lead: Option<Domino>) -> DominoSet {
        dominoes(self.rules.legal(mask(hand), lead.map(tile))).expect("a legal set is part of the hand")
    }

    fn trick_winner(&self, leader: Seat, plays: [Domino; 4]) -> Seat {
        leader.plus(self.rules.winner_offset(&plays.map(tile)) as usize)
    }

    fn trick_points(&self, plays: [Domino; 4]) -> u32 {
        u32::from(self.rules.trick_points(&plays.map(tile)))
    }
}
