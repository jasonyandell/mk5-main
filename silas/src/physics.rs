//! The Physics boundary: everything that depends on the declaration or on a
//! hidden world. Silas owns the nouns and the language; a `Physics` owns the
//! rules. The required methods are the four rule tables of one declaration
//! (led context, follow sets, trick rank, count) and the fiber of hidden
//! worlds (count and enumerate). Everything else the language needs is derived
//! from those tables by the provided methods, which an implementation may
//! override only with an equal, faster computation.

use std::fmt;

use crate::frame::{Kernel, World};
use crate::nouns::{Context, Domino, DominoSet, Seat};

pub trait Physics: fmt::Debug + Send + Sync {
    /// A stable name for these rules, such as `straight42/P6`: one word, with
    /// no whitespace, parentheses or semicolons, so policy text can carry it.
    /// Two physics with the same identity must be the same rules: frames,
    /// summaries and policy keys are compared through it.
    fn identity(&self) -> String;

    /// The context a tile leads.
    fn led_context(&self, tile: Domino) -> Context;

    /// The tiles that follow `context` (its effective incidence): the tiles a
    /// seat must play, if it holds any, when `context` is led.
    fn incidence(&self, context: Context) -> DominoSet;

    /// The tile's strength in a trick led in `led`. Only the order matters:
    /// the greatest key among the four plays takes the trick.
    fn trick_key(&self, tile: Domino, led: Context) -> u8;

    /// The count points on a tile.
    fn count(&self, tile: Domino) -> u32;

    /// The number of hidden worlds consistent with `kernel`: assignments of
    /// its pool to the hidden seats that meet every capacity and respect every
    /// shown void.
    fn count_worlds(&self, kernel: &Kernel) -> u128;

    /// Those worlds, each exactly once, lazily.
    fn worlds<'a>(&'a self, kernel: &'a Kernel) -> Box<dyn Iterator<Item = World> + 'a>;

    /// Whether `tile` follows `context`.
    fn follows(&self, tile: Domino, context: Context) -> bool {
        self.incidence(context).contains(tile)
    }

    /// The tiles that would take a trick led in `led` from `tile`.
    fn beats(&self, led: Context, tile: Domino) -> DominoSet {
        let key = self.trick_key(tile, led);
        Domino::ALL.into_iter().filter(|e| self.trick_key(*e, led) > key).collect()
    }

    /// The tiles that beat `tile` when `tile` itself is led.
    fn threat(&self, tile: Domino) -> DominoSet {
        self.beats(self.led_context(tile), tile)
    }

    /// The legal plays from `hand` in a trick led by the tile `lead`:
    /// anything when leading (`lead` is `None`), otherwise the tiles that
    /// follow the context `lead` leads, or anything when the hand holds none
    /// of them. Keyed by the led tile, not its context: every trick is led by
    /// a tile, and a context no tile leads has no legal plays to ask about.
    fn legal_plays(&self, hand: DominoSet, lead: Option<Domino>) -> DominoSet {
        match lead.map(|t| hand.intersection(self.incidence(self.led_context(t)))) {
            Some(follows) if !follows.is_empty() => follows,
            _ => hand,
        }
    }

    /// The seat that takes a trick of four distinct plays, in play order from
    /// `leader`: the first play with the greatest key in the led context.
    fn trick_winner(&self, leader: Seat, plays: [Domino; 4]) -> Seat {
        let led = self.led_context(plays[0]);
        let mut best = 0;
        for i in 1..4 {
            if self.trick_key(plays[i], led) > self.trick_key(plays[best], led) {
                best = i;
            }
        }
        leader.plus(best)
    }

    /// One point for the trick plus the count on its tiles.
    fn trick_points(&self, plays: [Domino; 4]) -> u32 {
        1 + plays.iter().map(|d| self.count(*d)).sum::<u32>()
    }
}

/// Every point a hand can bank: a point per trick plus all the count.
pub(crate) fn total_points(physics: &dyn Physics) -> u32 {
    (Domino::COUNT / Seat::COUNT) as u32 + Domino::ALL.iter().map(|d| physics.count(*d)).sum::<u32>()
}
