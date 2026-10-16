// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The arena's ledger (`docs/cross-game.md` § 2.5, slice 4): a comparison
//! Silas cannot play out itself (a field with a seat only the arena can
//! play), imported as data. The arena writes one row per deal, seating and
//! arm: the deal, the auction, the contract, the outcome and the plays.
//! Outcomes, never utilities: Silas replays every row under its own rules,
//! settles it through [`Contract::settle`] and values it under its own
//! [`Utility`], so the units are Silas's by construction. The answer is a
//! [`Paired`], `exact_given(field)` met with the sample: never `exact`.

use std::collections::BTreeMap;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;

use crate::field::{Paired, Seats, Settlement, Table, check, range, summarize};
use crate::frame::{Contract, World};
use crate::measure::{Authority, Sample};
use crate::nouns::{Domino, DominoSet, Seat};
use crate::scoring::{Outcome, Scoring, Utility};
use crate::{Result, error};

/// Which player sat in the chair under test: the first, the second, or the
/// chance player run alongside.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Arm {
    A,
    B,
    Chance,
}

impl Arm {
    pub const ALL: [Arm; 3] = [Arm::A, Arm::B, Arm::Chance];
}

/// One hand of a ledger: the deal it belongs to (from zero), the hands as
/// dealt, the shaker, the chair under test, the arm, and how it ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LedgerRow {
    pub deal: usize,
    pub world: World,
    pub shaker: Seat,
    pub chair: Seat,
    pub arm: Arm,
    pub settlement: Settlement,
}

/// A comparison as the arena played it: the field's identity (its model:
/// net content hash, eps, seed), the three players' names (a, b, chance),
/// the sample of deals, the seatings, the table, and every hand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ledger {
    pub field: String,
    pub players: [String; 3],
    pub sample: Sample,
    pub seats: Seats,
    pub table: Table,
    pub rows: Vec<LedgerRow>,
}

impl Paired {
    /// The paired comparison a ledger records, the chance arm alongside:
    /// each deal's difference is its mean over the seatings, the interval
    /// the empirical Bernstein bound at `confidence`, as
    /// [`crate::Trial::compare`]. Every row is replayed under Silas's rules
    /// and settled by [`Contract::settle`]; the ledger's outcome must be the
    /// replay's. Refused when the utility's unit is not the ledger's scoring
    /// (a marks ledger read as points), for a sample of fewer than two deals
    /// or a confidence outside (0, 1), when a deal does not hold every
    /// seating of every arm exactly once on one deal, and for any row whose
    /// auction is not the table's, whose contract is not its auction's, whose
    /// declaration is not offered, or whose plays are illegal or out of turn.
    pub fn from_ledger(ledger: &Ledger, utility: &Utility, confidence: &BigRational) -> Result<Paired> {
        let state = &ledger.table.state;
        let scoring = state.scoring();
        let unit_fits = match utility {
            Utility::Marks => matches!(scoring, Scoring::Marks { .. }),
            Utility::Points => matches!(scoring, Scoring::Points { .. }),
            Utility::MatchWin(value) => value.scoring() == scoring,
        };
        if !unit_fits {
            return Err(error(format!(
                "the ledger's hands were played for {scoring}; read them in that unit, not as {}",
                match utility {
                    Utility::Marks => "marks".to_owned(),
                    Utility::Points => "points".to_owned(),
                    Utility::MatchWin(value) => format!("a match-win value for {}", value.scoring()),
                }
            )));
        }
        let seatings = ledger.seats.seatings();
        check(utility, confidence, Some(ledger.sample), &seatings)?;
        let mut deals: BTreeMap<usize, BTreeMap<(Seat, Seat, Arm), &LedgerRow>> = BTreeMap::new();
        for row in &ledger.rows {
            if row.deal >= ledger.sample.n {
                return Err(error(format!("a row of deal {} in a sample of {}", row.deal, ledger.sample.n)));
            }
            verify(row, &ledger.table)?;
            let cell = deals.entry(row.deal).or_default();
            if cell.values().next().is_some_and(|other| other.world != row.world) {
                return Err(error(format!("deal {} is dealt two ways", row.deal)));
            }
            if cell.insert((row.chair, row.shaker, row.arm), row).is_some() {
                return Err(error(format!(
                    "deal {} has two rows for the chair {} at the shaker {} in arm {:?}",
                    row.deal, row.chair, row.shaker, row.arm
                )));
            }
        }
        let k = BigRational::from_integer(BigInt::from(seatings.len()));
        let (mut ab, mut ac) = (Vec::new(), Vec::new());
        for deal in 0..ledger.sample.n {
            let cell = deals.get(&deal).ok_or_else(|| error(format!("the ledger has no rows for deal {deal}")))?;
            if cell.len() != seatings.len() * Arm::ALL.len() {
                return Err(error(format!(
                    "deal {deal} has {} rows, not one for each of {} seatings and 3 arms",
                    cell.len(),
                    seatings.len()
                )));
            }
            let (mut x, mut y) = (BigRational::zero(), BigRational::zero());
            for &(chair, shaker) in &seatings {
                let value = |arm| -> Result<BigRational> {
                    let row = cell.get(&(chair, shaker, arm)).ok_or_else(|| {
                        error(format!("deal {deal} has no row for the chair {chair} at the shaker {shaker} in {arm:?}"))
                    })?;
                    row.settlement.value(utility, chair.team(), state)
                };
                let va = value(Arm::A)?;
                x += &va - value(Arm::B)?;
                y += va - value(Arm::Chance)?;
            }
            ab.push(x / &k);
            ac.push(y / &k);
        }
        let authority = Authority::given(ledger.field.clone())?
            .meet(&Authority::sampled(ledger.sample.n, ledger.sample.seed))
            .meet(&utility.authority());
        let width = range(&ledger.table, utility);
        let chance = summarize(ac, true, &width, confidence, authority.clone(), None);
        Ok(summarize(ab, true, &width, confidence, authority, Some(Box::new(chance))))
    }
}

/// A row is a hand Silas's rules allow, ending as it says.
fn verify(row: &LedgerRow, table: &Table) -> Result<()> {
    let at = |message: String| {
        error(format!("deal {}, chair {}, shaker {}, {:?}: {message}", row.deal, row.chair, row.shaker, row.arm))
    };
    let auction = row.settlement.auction();
    if auction.config() != table.config || auction.shaker() != row.shaker || !auction.is_complete() {
        return Err(at(format!(
            "the auction {auction} is not a complete auction of the table, shaken by {}",
            row.shaker
        )));
    }
    match &row.settlement {
        Settlement::PassedOut { .. } if auction.result().is_none() => Ok(()),
        Settlement::PassedOut { .. } => Err(at(format!("{auction} is not passed out"))),
        Settlement::Played { contract, outcome, plays, .. } => {
            if !table.offered.contains(contract.declaration()) {
                return Err(at(format!("{} is not offered", contract.declaration().name())));
            }
            if auction.contract(contract.declaration().clone())? != *contract {
                return Err(at(format!("the contract is not the auction's: {auction}")));
            }
            let replayed = replay(&row.world, contract, plays).map_err(|e| at(e.0))?;
            if replayed != *outcome {
                return Err(at(format!("the ledger says {outcome}; the plays make {replayed}")));
            }
            Ok(())
        }
    }
}

/// The outcome of `plays` from the deal `world` under `contract`: each play
/// by the seat to play, a tile it holds and may play; seven tricks, the
/// declarer leading the first and each trick's winner the next.
fn replay(world: &World, contract: &Contract, plays: &[(Seat, Domino)]) -> Result<Outcome> {
    if plays.len() != Domino::COUNT {
        return Err(error(format!("a hand is 28 plays, not {}", plays.len())));
    }
    let physics = contract.declaration();
    let mut hands: [DominoSet; Seat::COUNT] = Seat::ALL.map(|s| world.hand(s));
    let (mut leader, mut banked) = (contract.declarer(), [0u32; 2]);
    for trick in plays.chunks(Seat::COUNT) {
        for (k, &(seat, tile)) in trick.iter().enumerate() {
            if seat != leader.plus(k) {
                return Err(error(format!("{seat} plays {tile} out of turn: {} is to play", leader.plus(k))));
            }
            if !physics.legal_plays(hands[seat.index()], (k > 0).then_some(trick[0].1)).contains(tile) {
                return Err(error(format!("{seat} may not play {tile}")));
            }
            hands[seat.index()].remove(tile);
        }
        let tiles: [Domino; 4] = std::array::from_fn(|k| trick[k].1);
        leader = physics.trick_winner(leader, tiles);
        banked[leader.team().index()] += physics.trick_points(tiles);
    }
    let side = contract.declarer().team();
    Ok(Outcome::new(banked[side.index()], banked[side.other().index()]))
}

#[cfg(feature = "ledger")]
mod json {
    //! The ledger as the arena writes it (`arena/ledger.py`): one JSON
    //! object, every key required and no other allowed, so a ledger that
    //! carries a utility (marks won, a transfer, a value) is refused, not
    //! ignored.

    use std::sync::Arc;

    use serde_json::{Map, Value};

    use super::{Arm, Ledger, LedgerRow};
    use crate::auction::{Auction, AuctionConfig, Call};
    use crate::field::{Seats, Settlement, Table};
    use crate::frame::{Bid, Contract, World};
    use crate::measure::Sample;
    use crate::nouns::{Domino, DominoSet, Seat};
    use crate::physics::Physics;
    use crate::reference::Straight42;
    use crate::scoring::{MatchState, Outcome, Scoring};
    use crate::{Result, error};

    /// The format this reader reads.
    pub const FORMAT: &str = "silas-ledger-1";

    fn object<'a>(v: &'a Value, what: &str, keys: &[&str]) -> Result<&'a Map<String, Value>> {
        let map = v.as_object().ok_or_else(|| error(format!("{what} is an object")))?;
        if let Some(extra) = map.keys().find(|k| !keys.contains(&k.as_str())) {
            return Err(error(format!(
                "{what} has the field {extra:?}: a ledger carries outcomes, never utilities, and only the fields {}",
                keys.join(", ")
            )));
        }
        if let Some(missing) = keys.iter().find(|k| !map.contains_key(**k)) {
            return Err(error(format!("{what} lacks the field {missing:?}")));
        }
        Ok(map)
    }

    fn uint(v: &Value, what: &str) -> Result<u64> {
        v.as_u64().ok_or_else(|| error(format!("{what} is a nonnegative integer, not {v}")))
    }

    fn text<'a>(v: &'a Value, what: &str) -> Result<&'a str> {
        v.as_str().ok_or_else(|| error(format!("{what} is a string, not {v}")))
    }

    fn list<'a>(v: &'a Value, what: &str) -> Result<&'a [Value]> {
        v.as_array().map(Vec::as_slice).ok_or_else(|| error(format!("{what} is a list, not {v}")))
    }

    fn seat(v: &Value, what: &str) -> Result<Seat> {
        let n = uint(v, what)?;
        usize::try_from(n)
            .ok()
            .and_then(Seat::from_index)
            .ok_or_else(|| error(format!("{what} is a seat 0 to 3, not {n}")))
    }

    fn declaration(v: &Value, what: &str) -> Result<Physics> {
        Ok(text(v, what)?.parse::<Straight42>()?.physics())
    }

    impl Ledger {
        /// Read a ledger the arena wrote (format `silas-ledger-1`): its
        /// header, then one row per deal, seating and arm. Structure only;
        /// [`crate::Paired::from_ledger`] replays and checks every row.
        pub fn from_json(source: &str) -> Result<Ledger> {
            let v: Value = serde_json::from_str(source).map_err(|e| error(format!("not a ledger: {e}")))?;
            let top = object(
                &v,
                "the ledger",
                &["ledger", "field", "players", "sample", "seats", "auction", "offered", "scoring", "score", "rows"],
            )?;
            if top["ledger"] != FORMAT {
                return Err(error(format!("the ledger's format is {}, not {FORMAT}", top["ledger"])));
            }
            let players = object(&top["players"], "players", &["a", "b", "chance"])?;
            let name = |k: &str| text(&players[k], k).map(str::to_owned);
            let sample = object(&top["sample"], "sample", &["n", "seed"])?;
            let sample = Sample {
                n: usize::try_from(uint(&sample["n"], "sample.n")?).map_err(|_| error("sample.n is too large"))?,
                seed: uint(&sample["seed"], "sample.seed")?,
            };
            let seats = match &top["seats"] {
                Value::String(s) if s == "rotate" => Seats::Rotate,
                other => Seats::Fixed(seat(other, "seats")?),
            };
            let auction = object(&top["auction"], "auction", &["max_marks", "forced"])?;
            let forced = auction["forced"].as_bool().ok_or_else(|| error("auction.forced is true or false"))?;
            let config = AuctionConfig::new(
                u32::try_from(uint(&auction["max_marks"], "auction.max_marks")?)
                    .map_err(|_| error("too many marks"))?,
                forced,
            )?;
            let offered: Arc<[Physics]> = list(&top["offered"], "offered")?
                .iter()
                .map(|d| declaration(d, "an offered declaration"))
                .collect::<Result<_>>()?;
            let scoring = object(&top["scoring"], "scoring", &["kind", "target"])?;
            let target = u32::try_from(uint(&scoring["target"], "scoring.target")?)
                .map_err(|_| error("the target is too large"))?;
            let scoring = match text(&scoring["kind"], "scoring.kind")? {
                "marks" => Scoring::Marks { target },
                "points" => Scoring::Points { target },
                other => return Err(error(format!("scoring.kind is marks or points, not {other}"))),
            };
            let score: Vec<u32> = list(&top["score"], "score")?
                .iter()
                .map(|s| uint(s, "a score").and_then(|n| u32::try_from(n).map_err(|_| error("a score is too large"))))
                .collect::<Result<_>>()?;
            let score: [u32; 2] = score.try_into().map_err(|_| error("score is two numbers, by team"))?;
            let table = Table { config, offered, state: MatchState::new(scoring, score)? };
            let rows = list(&top["rows"], "rows")?
                .iter()
                .enumerate()
                .map(|(i, r)| row(r, &table).map_err(|e| error(format!("row {i}: {}", e.0))))
                .collect::<Result<_>>()?;
            Ok(Ledger {
                field: text(&top["field"], "field")?.to_owned(),
                players: [name("a")?, name("b")?, name("chance")?],
                sample,
                seats,
                table,
                rows,
            })
        }
    }

    fn row(v: &Value, table: &Table) -> Result<LedgerRow> {
        let r =
            object(v, "a row", &["deal", "hands", "shaker", "chair", "arm", "calls", "contract", "outcome", "plays"])?;
        let tile = |v: &Value| -> Result<Domino> { text(v, "a tile")?.parse() };
        let hands = list(&r["hands"], "hands")?
            .iter()
            .map(|h| list(h, "a hand")?.iter().map(tile).collect::<Result<DominoSet>>())
            .collect::<Result<Vec<_>>>()?;
        let world = World::deal(hands.try_into().map_err(|_| error("hands are four, by seat"))?)?;
        let shaker = seat(&r["shaker"], "shaker")?;
        let arm = match text(&r["arm"], "arm")? {
            "a" => Arm::A,
            "b" => Arm::B,
            "chance" => Arm::Chance,
            other => return Err(error(format!("arm is a, b or chance, not {other}"))),
        };
        let calls =
            list(&r["calls"], "calls")?.iter().map(|c| text(c, "a call")?.parse()).collect::<Result<Vec<Call>>>()?;
        let auction = Auction::new(table.config, shaker, calls)?;
        let settlement = match (&r["contract"], &r["outcome"]) {
            (Value::Null, Value::Null) if list(&r["plays"], "plays")?.is_empty() => Settlement::PassedOut { auction },
            (Value::Null, _) | (_, Value::Null) => {
                return Err(error(
                    "a passed-out row has no contract, no outcome and no plays; a played row has all three",
                ));
            }
            (contract, outcome) => {
                let c = object(contract, "contract", &["declarer", "bid", "declaration"])?;
                let bid = Bid::number(
                    u32::try_from(uint(&c["bid"], "contract.bid")?).map_err(|_| error("a bid too large"))?,
                )?;
                let contract = Contract::new(
                    seat(&c["declarer"], "contract.declarer")?,
                    bid,
                    declaration(&c["declaration"], "contract.declaration")?,
                )?;
                let o = object(outcome, "outcome", &["declaring", "other"])?;
                let points =
                    |k: &str| uint(&o[k], k).and_then(|n| u32::try_from(n).map_err(|_| error("too many points")));
                let outcome = Outcome::new(points("declaring")?, points("other")?);
                let plays = list(&r["plays"], "plays")?
                    .iter()
                    .map(|p| match list(p, "a play")? {
                        [s, t] => Ok((seat(s, "a play's seat")?, tile(t)?)),
                        _ => Err(error("a play is [seat, tile]")),
                    })
                    .collect::<Result<_>>()?;
                Settlement::Played { auction, contract, outcome, plays }
            }
        };
        Ok(LedgerRow {
            deal: usize::try_from(uint(&r["deal"], "deal")?).map_err(|_| error("deal too large"))?,
            world,
            shaker,
            chair: seat(&r["chair"], "chair")?,
            arm,
            settlement,
        })
    }
}
