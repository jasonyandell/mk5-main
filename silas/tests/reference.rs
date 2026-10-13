//! The reference physics against facts stated independently of it.
mod common;

use std::collections::BTreeSet;

use common::*;
use silas::reference::{Decl, Straight42};
use silas::*;

#[test]
fn nine_straight_declarations_by_name() {
    let names: Vec<String> = Straight42::ALL.iter().map(|p| p.identity()).collect();
    let expected = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "DT", "NT"].map(|d| format!("straight42/{d}"));
    assert_eq!(names, expected);
    for rules in Straight42::ALL {
        assert_eq!(rules.decl().to_string().parse::<Straight42>().unwrap(), rules);
    }
    assert!("DS".parse::<Decl>().is_err() && "P7".parse::<Decl>().is_err());
}

#[test]
fn the_count_is_35_points_on_five_tiles() {
    for rules in Straight42::ALL {
        let counted: Vec<(Domino, u32)> =
            Domino::ALL.into_iter().map(|d| (d, rules.count(d))).filter(|(_, c)| *c > 0).collect();
        let expected = [("5-0", 5), ("4-1", 5), ("3-2", 5), ("6-4", 10), ("5-5", 10)];
        assert_eq!(
            counted.into_iter().collect::<BTreeSet<_>>(),
            expected.map(|(d, c)| (tile(d), c)).into_iter().collect()
        );
    }
}

#[test]
fn every_tile_follows_the_context_it_leads_and_trumps_are_absorbed() {
    for rules in Straight42::ALL {
        for d in Domino::ALL {
            assert!(rules.follows(d, rules.led_context(d)), "{rules:?} {d}");
        }
        let trumps = rules.incidence(Context::Called);
        for q in Context::ALL.into_iter().filter(|q| *q != Context::Called) {
            assert!(rules.incidence(q).is_disjoint(trumps));
        }
    }
    let p6: Straight42 = "P6".parse().unwrap();
    assert_eq!(p6.incidence(Context::Called).len(), 7);
    assert_eq!(p6.led_context(tile("6-1")), Context::Called);
    assert_eq!(p6.led_context(tile("5-1")), "q5".parse().unwrap());
    let nt: Straight42 = "NT".parse().unwrap();
    assert!(nt.incidence(Context::Called).is_empty());
    assert!(nt.follows(tile("6-4"), "q4".parse().unwrap()) && nt.follows(tile("6-4"), "q6".parse().unwrap()));
}

#[test]
fn trumps_take_led_doubles_and_doubles_top_their_suits() {
    let trick = [tile("5-5"), tile("5-1"), tile("6-0"), tile("5-2")];
    let p6: Straight42 = "P6".parse().unwrap();
    let nt: Straight42 = "NT".parse().unwrap();
    assert_eq!(p6.trick_winner(Seat::S0, trick), Seat::S2);
    assert_eq!(nt.trick_winner(Seat::S0, trick), Seat::S0);
    assert_eq!(nt.trick_points(trick), 1 + 10);
    let dt: Straight42 = "DT".parse().unwrap();
    // Under doubles trump the doubles rank by pip; elsewhere a double tops its suit.
    assert_eq!(dt.trick_winner(Seat::S1, [tile("1-1"), tile("6-6"), tile("6-5"), tile("0-0")]), Seat::S2);
    assert_eq!(p6.threat(tile("6-6")), DominoSet::EMPTY);
    assert_eq!(nt.threat(tile("6-5")), tiles(&["6-6"]));
}

#[test]
fn the_fiber_count_matches_its_enumeration() {
    let mut voided = open_hidden([2, 3, 2]);
    voided[0].voids.insert(Context::Called);
    voided[2].voids.insert("q1".parse().unwrap());
    let kernels = [
        Kernel::new(
            Seat::S0,
            tiles(&["6-6"]),
            tiles(&["6-5", "5-5", "3-2", "6-4", "2-1", "1-0"]),
            open_hidden([2, 2, 2]),
        )
        .unwrap(),
        Kernel::new(Seat::S0, tiles(&["0-0"]), tiles(&["6-5", "5-5", "3-2", "6-4", "2-1", "1-0", "4-4"]), voided)
            .unwrap(),
        Kernel::new(Seat::S0, tiles(&["0-0"]), DominoSet::EMPTY, open_hidden([0, 0, 0])).unwrap(),
    ];
    for rules in Straight42::ALL {
        for kernel in &kernels {
            let worlds: Vec<World> = rules.worlds(kernel).collect();
            assert_eq!(worlds.len() as u128, rules.count_worlds(kernel));
            assert_eq!(worlds.iter().collect::<BTreeSet<_>>().len(), worlds.len());
            assert!(worlds.iter().all(|w| kernel.contains(&rules, w)));
        }
    }
    // With no voids the count is the multinomial 6! / (2! 2! 2!).
    assert_eq!(Straight42::ALL[0].count_worlds(&kernels[0]), 90);
    // An exhaustive check: every assignment of the pool that the kernel
    // admits is enumerated.
    let rules = Straight42::ALL[6];
    let kernel = &kernels[1];
    let pool: Vec<Domino> = kernel.pool().iter().collect();
    let mut admitted = 0;
    for code in 0..3usize.pow(pool.len() as u32) {
        let mut hands = [DominoSet::EMPTY; 3];
        let mut c = code;
        for d in &pool {
            hands[c % 3].insert(*d);
            c /= 3;
        }
        admitted += usize::from(kernel.contains(&rules, &kernel.world(hands)));
    }
    assert_eq!(admitted as u128, rules.count_worlds(kernel));
}
