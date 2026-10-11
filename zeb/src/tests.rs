//! The player natively: the embedded net, the plans, determinism, and
//! parity with the Python referee on `tests/fixtures/parity.json`, down to
//! every scenario's outcome.

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::*;

/// Seat 0's hand in the tests: 6-6 6-5 6-4 5-5 5-4 4-4 3-3.
const HAND: [u8; 7] = [27, 26, 25, 20, 19, 14, 9];

#[test]
fn the_embedded_net_parses_and_its_id_is_its_digest() {
    let net = FieldNet::from_bytes(NET).unwrap();
    assert!((0..=6).all(|d| net.plays(d)) && !net.plays(7) && !net.plays(9));
    let digest: String = Sha256::digest(NET).iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(net_id(), digest);
}

#[test]
fn worlds_are_bounded() {
    assert!(Player::new(0, 0).is_err());
    assert!(Player::new(MAX_WORLDS + 1, 0).is_err());
    assert!(Player::new(MAX_WORLDS, 0).is_ok());
}

#[test]
fn a_forecast_is_a_tail_and_is_deterministic() {
    let zeb = Player::new(24, 7).unwrap();
    let f = zeb.forecast(&HAND, &[6, 7, 9]).unwrap();
    assert_eq!(f.iter().map(|f| f.decl).collect::<Vec<_>>(), vec![6, 7, 9]);
    for x in &f {
        assert!(x.pmake.windows(2).all(|w| w[0] >= w[1]), "a tail never rises");
        assert!((-42.0..=42.0).contains(&x.mean));
    }
    assert_eq!(f, zeb.forecast(&HAND, &[6, 7, 9]).unwrap());
    assert_ne!(f, Player::new(24, 8).unwrap().forecast(&HAND, &[6, 7, 9]).unwrap());
}

#[test]
fn a_forecast_refuses_what_it_cannot_plan() {
    let zeb = Player::new(8, 0).unwrap();
    assert!(zeb.forecast(&HAND, &[6, 6]).unwrap_err().0.contains("twice"));
    assert!(zeb.forecast(&HAND, &[8]).unwrap_err().0.contains("nello"));
    assert!(zeb.forecast(&HAND[..6], &[]).unwrap_err().0.contains("7 tiles"));
    assert_eq!(zeb.forecast(&HAND, &[]).unwrap(), vec![]);
}

#[test]
fn a_forced_move_is_not_planned() {
    // Seat 3 leads 6-0; seat 0 holds 6-6 6-5 6-4 and must follow with one of them.
    let p = Position::new(6, 3, 30, 0, &HAND, &[(3, 21)]).unwrap();
    assert_eq!(p.legal().count_ones(), 3);
    // Seat 1 leads 3-0 into seat 0, whose only three is 3-3.
    let p = Position::new(6, 1, 30, 0, &HAND, &[(1, 6), (2, 7), (3, 8)]).unwrap();
    let zeb = Player::new(16, 0).unwrap();
    assert_eq!(zeb.play(&p), Move { tile: 9, forced: true, q: vec![] });
    assert_eq!(zeb.analyze(&p), Hint { tile: 9, forced: true, q: vec![], pmake: vec![] });
}

#[test]
fn play_and_analyze_agree() {
    let zeb = Player::new(32, 3).unwrap();
    for decl in [0, 7] {
        let p = Position::new(decl, 1, 34, 0, &HAND, &[(1, 0), (2, 1), (3, 2)]).unwrap();
        let (m, h) = (zeb.play(&p), zeb.analyze(&p));
        assert_eq!((m.tile, m.forced, &m.q), (h.tile, h.forced, &h.q));
        // q is the tail at the contract.
        for ((t, q), (u, tails)) in h.q.iter().zip(&h.pmake) {
            assert_eq!(t, u);
            assert_eq!(*q, tails[34 - 30]);
        }
    }
}

// -- parity with the referee ------------------------------------------------

#[derive(Deserialize)]
struct Fixtures {
    net: String,
    positions: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    request: Request,
    referee: Referee,
}

#[derive(Deserialize)]
struct Request {
    decl: u8,
    bidder: u8,
    bid: u8,
    seat: u8,
    hand: Vec<u8>,
    plays: Vec<(u8, u8)>,
    worlds: usize,
    seed: String,
}

#[derive(Deserialize)]
struct Referee {
    tile: u8,
    forced: bool,
    #[serde(default)]
    q: Vec<(u8, f64)>,
    #[serde(default)]
    weights: Vec<f64>,
    #[serde(default)]
    outcomes: Vec<(u8, Vec<f64>)>,
    #[serde(default)]
    pmake: Vec<(u8, [f64; 13])>,
    forecast: Option<ForecastWant>,
}

#[derive(Deserialize)]
struct ForecastWant {
    decl: u8,
    mean: f64,
    pmake: [f64; 13],
}

#[test]
fn every_scenario_ends_where_the_referee_says() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/parity.json");
    let text = std::fs::read_to_string(path).expect("the fixtures: python -u -m zeb.scripts.make_fixtures");
    let fixtures: Fixtures = serde_json::from_str(&text).unwrap();
    assert_eq!(fixtures.net, net_id(), "the fixtures were made with another net");
    assert!(fixtures.positions.len() >= 24);
    for (i, Case { request: r, referee: want }) in fixtures.positions.iter().enumerate() {
        let p = Position::new(r.decl, r.bidder, r.bid, r.seat, &r.hand, &r.plays).unwrap();
        let zeb = Player::new(r.worlds, r.seed.parse().unwrap()).unwrap();
        let hint = zeb.analyze(&p);
        assert_eq!(hint, Hint { tile: want.tile, forced: want.forced, q: want.q.clone(), pmake: want.pmake.clone() });
        assert_eq!(zeb.play(&p), Move { tile: want.tile, forced: want.forced, q: want.q.clone() }, "case {i}");
        if want.forced {
            continue;
        }
        let a = zeb.analysis(&p);
        assert_eq!((&a.weights, &a.outcomes), (&want.weights, &want.outcomes), "case {i}");
        if let Some(f) = &want.forecast {
            let got = zeb.forecast(&r.hand, &[f.decl]).unwrap();
            assert_eq!(got, vec![Forecast { decl: f.decl, mean: f.mean, pmake: f.pmake }], "case {i}");
        }
    }
}
