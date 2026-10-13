//! The reference physics against facts stated independently of it.
mod common;

use std::collections::BTreeSet;

use common::*;
use silas::reference::{Decl, Straight42};
use silas::*;

#[test]
fn nine_straight_declarations_by_name() {
    let names: Vec<String> = Straight42::ALL.iter().map(|p| p.name()).collect();
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
    for rules in Straight42::ALL.map(Straight42::physics) {
        for d in Domino::ALL {
            assert!(rules.follows(d, rules.led_context(d)), "{rules:?} {d}");
        }
        let trumps = rules.incidence(Context::Called);
        for q in Context::ALL.into_iter().filter(|q| *q != Context::Called) {
            assert!(rules.incidence(q).is_disjoint(trumps));
        }
    }
    let p6 = physics("P6");
    assert_eq!(p6.incidence(Context::Called).len(), 7);
    assert_eq!(p6.led_context(tile("6-1")), Context::Called);
    assert_eq!(p6.led_context(tile("5-1")), "q5".parse().unwrap());
    let nt = physics("NT");
    assert!(nt.incidence(Context::Called).is_empty());
    assert!(nt.follows(tile("6-4"), "q4".parse().unwrap()) && nt.follows(tile("6-4"), "q6".parse().unwrap()));
}

#[test]
fn trumps_take_led_doubles_and_doubles_top_their_suits() {
    let trick = [tile("5-5"), tile("5-1"), tile("6-0"), tile("5-2")];
    let p6 = physics("P6");
    let nt = physics("NT");
    assert_eq!(p6.trick_winner(Seat::S0, trick), Seat::S2);
    assert_eq!(nt.trick_winner(Seat::S0, trick), Seat::S0);
    assert_eq!(nt.trick_points(trick), 1 + 10);
    let dt = physics("DT");
    // Under doubles trump the doubles rank by pip; elsewhere a double tops its suit.
    assert_eq!(dt.trick_winner(Seat::S1, [tile("1-1"), tile("6-6"), tile("6-5"), tile("0-0")]), Seat::S2);
    assert_eq!(p6.threat(tile("6-6")), DominoSet::EMPTY);
    assert_eq!(nt.threat(tile("6-5")), tiles(&["6-6"]));
}

/// The tournament rules of play (Winning 42 p. 163): at no trump the doubles
/// are no suit of their own (S-0209-g), and under doubles trump a double led
/// calls for a double from each player who holds one, any tile from one who
/// does not (S-0209-h). The variation that a player without a double must
/// follow the double's pip (p. 80) is not a straight declaration.
#[test]
fn the_books_follow_rules_at_no_trump_and_doubles_trump() {
    let nt = physics("NT");
    let sixes_led = Some(tile("6-6"));
    assert_eq!(nt.legal_plays(tiles(&["6-1", "2-2", "5-5"]), sixes_led), tiles(&["6-1"]));
    assert_eq!(nt.legal_plays(tiles(&["1-1", "2-2"]), Some(tile("2-1"))), tiles(&["2-2"]));
    let dt = physics("DT");
    let double_led = Some(tile("5-5"));
    assert_eq!(dt.legal_plays(tiles(&["3-3", "5-1", "6-5"]), double_led), tiles(&["3-3"]));
    assert_eq!(dt.legal_plays(tiles(&["5-1", "6-5", "2-0"]), double_led), tiles(&["5-1", "6-5", "2-0"]));
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
    for rules in Straight42::ALL.map(Straight42::physics) {
        for kernel in &kernels {
            let worlds: Vec<World> = rules.worlds(kernel).collect();
            assert_eq!(worlds.len() as u128, rules.count_worlds(kernel));
            assert_eq!(worlds.iter().collect::<BTreeSet<_>>().len(), worlds.len());
            assert!(worlds.iter().all(|w| kernel.contains(&rules, w)));
        }
    }
    // With no voids the count is the multinomial 6! / (2! 2! 2!).
    assert_eq!(Straight42::ALL[0].physics().count_worlds(&kernels[0]), 90);
    // An exhaustive check: every assignment of the pool that the kernel
    // admits is enumerated.
    let rules = Straight42::ALL[6].physics();
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

#[test]
fn off_suit_plays_tie_at_the_bottom() {
    // Fours trump, fives led: the 3-1 and the 6-2 neither follow nor trump.
    let p4 = physics("P4");
    let led = p4.led_context(tile("5-1"));
    assert_eq!(p4.trick_key(tile("3-1"), led), 0);
    assert_eq!(p4.trick_key(tile("6-2"), led), 0);
    assert!(!p4.beats(led, tile("3-1")).contains(tile("6-2")));
    assert!(p4.beats(led, tile("3-1")).contains(tile("5-0")));
}

/// A declaration under test: the reference's tables, renamed, with every
/// trick key shifted, or with the called suit stripped of its power.
struct Variant {
    base: Straight42,
    name: &'static str,
    shift: u8,
    powerless: bool,
}

impl Declaration for Variant {
    fn name(&self) -> String {
        self.name.to_owned()
    }

    fn led_context(&self, tile: Domino) -> Context {
        self.base.led_context(tile)
    }

    fn incidence(&self, context: Context) -> DominoSet {
        self.base.incidence(context)
    }

    fn trick_key(&self, tile: Domino, led: Context) -> u8 {
        let called = self.base.incidence(Context::Called).contains(tile);
        if self.powerless && called && led != Context::Called {
            return 0;
        }
        self.base.trick_key(tile, led) + self.shift
    }

    fn count(&self, tile: Domino) -> u32 {
        self.base.count(tile)
    }
}

#[test]
fn the_identity_is_the_tables_and_the_trump_is_derived() {
    let identities: BTreeSet<String> = Straight42::ALL.iter().map(|p| p.physics().identity().to_owned()).collect();
    assert_eq!(identities.len(), 9);
    for base in Straight42::ALL {
        let reference = base.physics();
        // Another name, and keys with the same order: the same tables.
        let renamed = Physics::new(&Variant { base, name: "elsewhere", shift: 3, powerless: false }).unwrap();
        assert_eq!((renamed.identity(), &renamed), (reference.identity(), &reference));
        assert_eq!(renamed.name(), "elsewhere");
        assert!(reference.identity().starts_with("tables-") && !reference.identity().contains(char::is_whitespace));
        // The trump is the called suit wherever it has power.
        let expected = (base.decl() != Decl::NoTrump).then_some(Context::Called);
        assert_eq!(reference.trump(), expected, "{base:?}");
        // Stripped of its power (the doubles as their own suit, say), the
        // called suit is no trump, and the tables differ.
        let powerless = Physics::new(&Variant { base, name: "powerless", shift: 0, powerless: true }).unwrap();
        if expected.is_some() {
            assert_eq!(powerless.trump(), None);
            assert_ne!(powerless.identity(), reference.identity());
        }
    }
    // A tile must follow the context it leads.
    struct Broken;
    impl Declaration for Broken {
        fn name(&self) -> String {
            "broken".into()
        }
        fn led_context(&self, _: Domino) -> Context {
            Context::Called
        }
        fn incidence(&self, _: Context) -> DominoSet {
            DominoSet::EMPTY
        }
        fn trick_key(&self, _: Domino, _: Context) -> u8 {
            0
        }
        fn count(&self, _: Domino) -> u32 {
            0
        }
    }
    assert!(Physics::new(&Broken).is_err());
}
