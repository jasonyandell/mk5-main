//! STRATEGY-FUSION FENCE: the clairvoyant solver, for `tests/book_leads.rs`'s
//! double-dummy rows only. The ruling: wiki/topics/strategy-fusion-fence.md.
//!
//! Helpers for `tests/book_leads.rs`: the book's eight full deals (Hands
//! 28-35) as open positions along their narrated lines, every root move valued
//! double-dummy under both objectives, and the bidder's side's points.

use silas::*;

use super::fixture::{Role, book};

/// One full deal and its narrated line, under the line's declaration and bid
/// (a marks bid needs all 42). The bidder sits in S0.
pub struct Narrated {
    pub physics: Physics,
    pub opening: OpenPosition,
    /// Every named play of the line's tricks, then of its tail.
    pub plays: Vec<Domino>,
}

pub fn narrated(id: &str) -> Narrated {
    let b = book();
    let line = b.line(&format!("{id}/prose"));
    let physics = super::physics(line.declaration.as_deref().expect("a straight declaration"));
    let contract = super::contract(&physics, Seat::S0, line.bid.number());
    let opening = OpenPosition::deal(contract, b.example(id).deal()).unwrap();
    let mut plays = line.plays();
    if let Some(tail) = &line.tail {
        plays.extend(tail.plays.iter().map(|(_, d)| d.expect("a named tail play")));
    }
    Narrated { physics, opening, plays }
}

impl Narrated {
    pub fn physics(&self) -> &Physics {
        &self.physics
    }

    /// The position after the first `n` narrated plays.
    pub fn after(&self, n: usize) -> OpenPosition {
        self.plays[..n].iter().fold(self.opening.clone(), |p, d| p.play(*d).unwrap())
    }

    /// The position at which `role` decides in trick `trick` (from one) of the
    /// narrated line.
    pub fn at(&self, trick: usize, role: Role) -> OpenPosition {
        let mut position = self.opening.clone();
        let mut tricks = 1;
        for d in &self.plays {
            if tricks == trick && position.next_actor() == Some(seat(role)) {
                return position;
            }
            position = position.play(*d).unwrap();
            tricks += usize::from(position.trick().is_empty());
        }
        if tricks == trick && position.next_actor() == Some(seat(role)) {
            return position;
        }
        panic!("the narrated line never has {role:?} to play in trick {trick}")
    }

    /// Every legal move at `position`, valued.
    pub fn values(&self, position: &OpenPosition) -> Values {
        values(self.physics(), position)
    }

    /// The value of a position: the bidder's side's points and whether the
    /// bid is made, under best play by all four seats.
    pub fn worth(&self, position: &OpenPosition) -> (u32, i32) {
        let mut solver = ClairvoyantSolver::new(self.physics());
        (points(solver.value(position, Objective::Margin)), solver.value(position, Objective::Make))
    }

    /// Follow a line from `position`: a named tile must be legal and keep the
    /// margin-best value; `None` lets the mover take its least margin-best
    /// tile. Returns the position reached and the tiles played.
    pub fn follow(&self, position: &OpenPosition, script: &[Option<&str>]) -> (OpenPosition, Vec<Domino>) {
        let mut solver = ClairvoyantSolver::new(self.physics());
        let mut position = position.clone();
        let mut played = Vec::new();
        for step in script {
            let root = solver.solve(&position, Objective::Margin).unwrap();
            let tile = match step {
                Some(s) => {
                    let d: Domino = s.parse().unwrap();
                    assert!(root.tied_best().contains(d), "{d} is not margin-best at {position:?}: {:?}", root.moves);
                    d
                }
                None => root.tied_best().iter().next().unwrap(),
            };
            played.push(tile);
            position = position.play(tile).unwrap();
        }
        (position, played)
    }
}

pub fn seat(role: Role) -> Seat {
    [Seat::S0, Seat::S1, Seat::S2, Seat::S3][role.index()]
}

/// The bidder's side's final points from a margin over a whole deal.
pub fn points(margin: i32) -> u32 {
    u32::try_from((42 + margin) / 2).unwrap()
}

/// Each legal move: the bidder's side's points (margin objective) and
/// whether the bid is made (make objective), both under best play after it.
pub struct Values {
    pub moves: Vec<(Domino, u32, i32)>,
}

pub fn values(physics: &Physics, position: &OpenPosition) -> Values {
    let mut solver = ClairvoyantSolver::new(physics);
    let margin = solver.solve(position, Objective::Margin).unwrap();
    let make = solver.solve(position, Objective::Make).unwrap();
    let moves = margin.moves.iter().map(|(d, v)| (*d, points(*v), make.value(*d).unwrap())).collect();
    Values { moves }
}

impl Values {
    /// The points and make value of `tile`.
    pub fn of(&self, tile: &str) -> (u32, i32) {
        let d: Domino = tile.parse().unwrap();
        let (_, p, m) = self.moves.iter().find(|(e, _, _)| *e == d).unwrap_or_else(|| panic!("{d} is not legal"));
        (*p, *m)
    }

    /// The legal moves other than `tiles`.
    pub fn others(&self, tiles: &[&str]) -> Vec<(u32, i32)> {
        let named: Vec<Domino> = tiles.iter().map(|t| t.parse().unwrap()).collect();
        self.moves.iter().filter(|(d, _, _)| !named.contains(d)).map(|(_, p, m)| (*p, *m)).collect()
    }

    /// Assert each listed tile's points and make value.
    pub fn assert(&self, expected: &[(&str, u32, i32)]) {
        for (tile, p, m) in expected {
            assert_eq!(self.of(tile), (*p, *m), "{tile}");
        }
    }
}

/// The same position under another bid.
pub fn rebid(position: &OpenPosition, bid: u32) -> OpenPosition {
    let old = position.contract();
    let contract = Contract::new(old.declarer(), Bid::number(bid).unwrap(), old.declaration().clone()).unwrap();
    OpenPosition::new(contract, position.hands(), position.leader(), position.trick().to_vec(), position.banked())
        .unwrap()
}
