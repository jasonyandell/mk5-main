//! Fixtures shared by the integration tests. Expected answers in the tests are
//! built directly from hands and the reference rules, independently of the
//! query compiler, joins, projection and aggregation.
#![allow(dead_code)]

pub mod fixture;

use std::num::NonZeroU64;
use std::sync::Arc;

use num_bigint::BigInt;
use num_rational::BigRational;
use silas::reference::Straight42;
use silas::*;

pub fn r(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

pub fn w(n: u64, d: u64) -> Weight {
    Weight::ratio(n, NonZeroU64::new(d).unwrap())
}

pub fn tile(s: &str) -> Domino {
    s.parse().unwrap()
}

pub fn tiles(ss: &[&str]) -> DominoSet {
    ss.iter().map(|s| tile(s)).collect()
}

pub fn physics(decl: &str) -> Arc<dyn Physics> {
    Arc::new(decl.parse::<Straight42>().unwrap())
}

pub fn open_hidden(capacities: [usize; 3]) -> [Hidden; 3] {
    let seats = [Seat::S1, Seat::S2, Seat::S3];
    std::array::from_fn(|i| Hidden { seat: seats[i], capacity: capacities[i], voids: ContextSet::EMPTY })
}

/// Viewer S0 holds 6-6 and 0-0; six tiles are hidden, two per seat: 90 worlds.
pub fn small_frame(physics: Arc<dyn Physics>) -> Frame {
    let kernel = Kernel::new(
        Seat::S0,
        tiles(&["6-6", "0-0"]),
        tiles(&["6-5", "5-5", "3-2", "6-4", "2-1", "1-0"]),
        open_hidden([2, 2, 2]),
    )
    .unwrap();
    Frame::new(physics, kernel, Seat::S0, vec![], DominoSet::EMPTY).unwrap()
}

pub fn uniform() -> Belief {
    Belief::uniform(small_frame(physics("P6")), 90).unwrap()
}

pub fn compile(s: &str) -> CompiledQuery {
    s.parse::<Query>().unwrap().compile(&Registry::standard()).unwrap()
}

pub fn budget() -> Budget {
    Budget::new(20_000_000)
}

/// The value of an estimate that must be exact.
pub fn exact(e: Estimate) -> BigRational {
    assert_eq!(e.authority(), &Authority::Exact);
    e.value().clone()
}

pub fn given(model: &str) -> Authority {
    Authority::ExactGiven { model: model.to_owned() }
}

pub const COUNTS: &str = include_str!("../../examples/partner-count.silas");
pub const HAS_COUNT: &str = include_str!("../../examples/partner-has-count.silas");
pub const MASTER: &str = include_str!("../../examples/partner-master.silas");
pub const FORCED: &str = include_str!("../../examples/current-forced-play.silas");
pub const SIX_FIVE: &str = include_str!("../../examples/six-five-guarded.silas");
