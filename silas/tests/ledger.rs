// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The arena's ledger (`docs/cross-game.md`, slice 4), held to Silas's own
//! play-out. `data/ledger-*.json` are written by the arena's auction and play
//! engine for a field of Silas policies (`arena/test_ledger.py`): every row of
//! the two players under test is the hand Silas's `play_out` plays from that
//! deal and shaker, call for call and tile for tile, and settles the same, so
//! the arena and Silas agree on the auction, legality, the tricks and the
//! settlement across the boundary. Read as a comparison, each ledger gives the
//! arena's own paired mean to the rational and its flips; a ledger that
//! carries a utility, a ledger read in another unit, and a ledger whose hands
//! Silas's rules refuse are all refused.

use std::sync::Arc;

use num_rational::BigRational;
use silas::*;

fn data(name: &str) -> String {
    std::fs::read_to_string(format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

fn ledger(name: &str) -> Ledger {
    Ledger::from_json(&data(name)).unwrap()
}

/// A seat of `arena/test_ledger.py` (`silas:<name>`, `SilasSeat`): a policy of
/// the book's bidding, or a path under `examples/`.
fn player(spec: &str) -> Player {
    let name = spec.strip_prefix("silas:").unwrap();
    let path = if name.contains('/') {
        format!("{}/examples/{name}.silas", env!("CARGO_MANIFEST_DIR"))
    } else {
        format!("{}/examples/book/bidding/{name}.silas", env!("CARGO_MANIFEST_DIR"))
    };
    let program = PolicyProgram::parse(&std::fs::read_to_string(&path).unwrap(), &path).unwrap();
    let physics = "P6".parse::<reference::Straight42>().unwrap().physics();
    Player::Policy(Arc::new(program.compile(&Registry::standard(), &physics).unwrap()))
}

fn field(ledger: &Ledger) -> [Player; 4] {
    let specs: Vec<&str> = ledger.field.split(", ").map(|s| s.split_once(' ').unwrap().1).collect();
    std::array::from_fn(|s| player(specs[s]))
}

const LEDGERS: [&str; 2] = ["ledger-marks.json", "ledger-points.json"];

fn half() -> BigRational {
    "19/20".parse().unwrap()
}

#[test]
fn every_arena_hand_of_a_silas_field_is_silas_own_play_out() {
    for name in LEDGERS {
        let ledger = ledger(name);
        let seats = field(&ledger);
        let arms = [player(&ledger.players[0]), player(&ledger.players[1])];
        let (mut played, mut passed) = (0, 0);
        for row in &ledger.rows {
            let arm = match row.arm {
                Arm::A => &arms[0],
                Arm::B => &arms[1],
                Arm::Chance => continue,
            };
            let mut sat = seats.clone();
            sat[row.chair.index()] = arm.clone();
            let ours =
                play_out(&row.world, row.shaker, &Field::new(sat), &ledger.table, &mut Budget::new(u64::MAX)).unwrap();
            assert_eq!(ours, row.settlement, "{name}: deal {}, chair {}, shaker {}", row.deal, row.chair, row.shaker);
            match ours {
                Settlement::Played { .. } => played += 1,
                Settlement::PassedOut { .. } => passed += 1,
            }
        }
        assert!(played > 0, "{name}");
        if name == "ledger-marks.json" {
            assert!(passed > 0, "the straight auction passes hands out");
        } else {
            assert_eq!(passed, 0, "a forced shaker never passes out");
        }
    }
}

#[test]
fn a_ledger_reads_to_the_arenas_paired_mean_and_flips() {
    for line in data("ledger-arena-means.txt").lines().filter(|l| !l.starts_with('#')) {
        let words: Vec<&str> = line.split_whitespace().collect();
        let [name, unit, mean, w, l, t] = words[..] else { panic!("{line}") };
        let utility = if unit == "marks" { Utility::Marks } else { Utility::Points };
        let ledger = ledger(name);
        let paired = Paired::from_ledger(&ledger, &utility, &half()).unwrap();
        assert_eq!(paired.delta().value(), &mean.parse::<BigRational>().unwrap(), "{name}");
        // The sign test's counts are the flips.
        assert_eq!(paired.flips(), (w.parse().unwrap(), l.parse().unwrap(), t.parse().unwrap()), "{name}");
        assert_eq!(paired.n(), ledger.sample.n);
        let authority = paired.delta().authority();
        assert!(!authority.is_exact());
        assert_eq!(authority.models().iter().collect::<Vec<_>>(), vec![&ledger.field]);
        assert_eq!(authority.samples().iter().copied().collect::<Vec<_>>(), vec![ledger.sample]);
        let chance = paired.chance().expect("the chance arm");
        assert_eq!(chance.delta().authority(), authority);
        let (lo, hi) = paired.interval();
        assert!(lo <= paired.delta().value() && paired.delta().value() <= hi);
    }
}

#[test]
fn a_ledger_that_carries_a_utility_is_refused() {
    let mut v: serde_json::Value = serde_json::from_str(&data("ledger-marks.json")).unwrap();
    v["rows"][0]["marks"] = 1.into();
    let refused = Ledger::from_json(&v.to_string()).unwrap_err();
    assert!(refused.0.contains("outcomes, never utilities"), "{refused}");
    let mut v: serde_json::Value = serde_json::from_str(&data("ledger-marks.json")).unwrap();
    v["utility"] = "marks".into();
    assert!(Ledger::from_json(&v.to_string()).is_err());
}

#[test]
fn a_ledger_read_in_another_unit_is_refused() {
    let marks = ledger("ledger-marks.json");
    let refused = Paired::from_ledger(&marks, &Utility::Points, &half()).unwrap_err();
    assert!(refused.0.contains("played for marks to 7"), "{refused}");
    assert!(Paired::from_ledger(&ledger("ledger-points.json"), &Utility::Marks, &half()).is_err());
}

#[test]
fn a_ledger_whose_hands_silas_refuses_is_refused() {
    let marks = ledger("ledger-marks.json");
    let at = marks.rows.iter().position(|r| matches!(r.settlement, Settlement::Played { .. })).unwrap();
    // An outcome the plays do not make.
    let mut wrong = marks.clone();
    if let Settlement::Played { outcome, .. } = &mut wrong.rows[at].settlement {
        *outcome = Outcome::new(outcome.declaring() + 1, outcome.other() - 1);
    }
    let refused = Paired::from_ledger(&wrong, &Utility::Marks, &half()).unwrap_err();
    assert!(refused.0.contains("the plays make"), "{refused}");
    // Two plays swapped: out of turn.
    let mut wrong = marks.clone();
    if let Settlement::Played { plays, .. } = &mut wrong.rows[at].settlement {
        plays.swap(0, 1);
    }
    assert!(Paired::from_ledger(&wrong, &Utility::Marks, &half()).is_err());
    // A seating missing.
    let mut wrong = marks.clone();
    wrong.rows.remove(at);
    assert!(Paired::from_ledger(&wrong, &Utility::Marks, &half()).is_err());
    // A deal dealt two ways.
    let mut wrong = marks.clone();
    let other = wrong.rows.iter().find(|r| r.deal != wrong.rows[at].deal).unwrap().world;
    wrong.rows[at].world = other;
    assert!(Paired::from_ledger(&wrong, &Utility::Marks, &half()).is_err());
    // The ledger itself reads.
    assert!(Paired::from_ledger(&marks, &Utility::Marks, &half()).is_ok());
}

#[test]
fn silas_compare_reads_a_ledger() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_silas"))
        .args(["compare", "--ledger", "tests/data/ledger-marks.json"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    let text = String::from_utf8(out.stdout).unwrap() + &String::from_utf8(out.stderr).unwrap();
    assert!(out.status.success(), "{text}");
    assert!(text.contains("silas:risk-budget - silas:never-32-or-33: 0.1250 marks a deal"), "{text}");
    assert!(text.contains("flips 2 for, 1 against, 1 tied"), "{text}");
    assert!(text.contains("and sampled(4, 20261015, arena)"), "{text}");
}

#[test]
fn a_menu_ledger_replays_its_doubles_trump_and_no_trump_hands() {
    // The book's full menu in the chair (#189): the table offers all nine
    // declarations, and every doubles-trump and no-trump row is replayed under
    // Silas's own rules and settles as the arena says (the means test above
    // reads it to the arena's paired mean).
    let menu = ledger("ledger-menu.json");
    let names: Vec<String> = menu.table.offered.iter().map(|d| d.name().to_owned()).collect();
    assert!(names.contains(&"straight42/DT".to_owned()) && names.contains(&"straight42/NT".to_owned()), "{names:?}");
    let declared: Vec<String> = menu
        .rows
        .iter()
        .filter_map(|r| match &r.settlement {
            Settlement::Played { contract, .. } => Some(contract.declaration().name().to_owned()),
            Settlement::PassedOut { .. } => None,
        })
        .collect();
    for want in ["straight42/DT", "straight42/NT"] {
        assert!(declared.iter().any(|d| d == want), "{want} in {declared:?}");
    }
    assert!(Paired::from_ledger(&menu, &Utility::Marks, &half()).is_ok());
    // The same rows at a table that offers only the pip trumps are refused.
    let mut v: serde_json::Value = serde_json::from_str(&data("ledger-menu.json")).unwrap();
    v["offered"] = serde_json::json!(["P6", "P5", "P4", "P3", "P2", "P1", "P0"]);
    let pips = Ledger::from_json(&v.to_string()).unwrap();
    let refused = Paired::from_ledger(&pips, &Utility::Marks, &half()).unwrap_err();
    assert!(refused.0.contains("is not offered"), "{refused}");
}
