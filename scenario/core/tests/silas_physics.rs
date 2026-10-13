//! Zeb against Silas. `ZebPhysics` supplies only Zeb's four rule tables;
//! Silas derives legality, the trick's winner and points, and the fiber from
//! them. Zeb's tables must be the reference's (`silas::reference::Straight42`)
//! for the nine straight declarations, so the two physics are one (equal
//! identities); and Zeb's own legality, winner, points and fiber (the code its
//! planner runs) must agree with Silas's on everything Silas can ask. The
//! tables are compared exhaustively; plays, tricks and fibers on seeded random
//! positions. The tenth declaration, the doubles as their own unpowered suit,
//! has no reference and is held to its facts. (Built only with the `silas`
//! feature: `required-features` in Cargo.toml.)

use std::collections::BTreeSet;

use scenario_core::physics::{context, domino, dominoes, led_suit, mask, seat, seat_id, tile, ZebPhysics, DECL_NAMES};
use scenario_core::planner::SplitMix;
use scenario_core::rules::{self, Rules, DOUBLES_SUIT, DOUBLES_TRUMP, NO_TRUMP, N_DECLS};
use silas::reference::{Decl, Straight42};
use silas::{
    Auction, Belief, Bid, Budget, Context, ContextSet, Declaration, Domino, DominoSet, Frame, Hidden, Kernel, Physics,
    Pip, Query, Registry, Seat, Team, World,
};

/// Silas's reference for a Zeb declaration; none for the doubles suit.
fn reference(decl: u8) -> Option<Straight42> {
    let decl = match decl {
        0..=6 => Decl::PipTrump(Pip::ALL[decl as usize]),
        DOUBLES_TRUMP => Decl::DoublesTrump,
        NO_TRUMP => Decl::NoTrump,
        _ => return None,
    };
    Some(Straight42::new(decl))
}

/// The nine declarations both implement, as (Zeb's, Silas's physics of
/// Zeb's tables, the reference's physics).
fn shared() -> Vec<(ZebPhysics, Physics, Physics)> {
    (0..N_DECLS)
        .filter_map(|d| {
            let zeb = ZebPhysics::new(d).unwrap();
            let physics = zeb.physics();
            Some((zeb, physics, reference(d)?.physics()))
        })
        .collect()
}

fn d(s: &str) -> Domino {
    s.parse().unwrap()
}

fn set(ss: &[&str]) -> DominoSet {
    ss.iter().map(|s| d(s)).collect()
}

fn below(rng: &mut SplitMix, n: usize) -> usize {
    (rng.next_u64() % n as u64) as usize
}

fn shuffled(rng: &mut SplitMix) -> [Domino; 28] {
    let mut deck = Domino::ALL;
    for i in (1..28).rev() {
        deck.swap(i, below(rng, i + 1));
    }
    deck
}

fn random_member(rng: &mut SplitMix, s: DominoSet) -> Domino {
    s.iter().nth(below(rng, s.len())).unwrap()
}

// -- the maps -------------------------------------------------------------------

#[test]
fn tile_ids_are_silas_domino_indices() {
    for t in 0..28u8 {
        let d = domino(t).unwrap();
        assert_eq!((d.hi().value(), d.lo().value()), rules::pips(t), "tile {t}");
        assert_eq!(tile(d), t);
    }
    assert_eq!((domino(0), domino(27)), (Some(self::d("0-0")), Some(self::d("6-6"))));
    assert_eq!(rules::tile(6, 5), tile(self::d("5-6")));
    assert_eq!(domino(28), None);
    let mut rng = SplitMix::new(1);
    for _ in 0..1000 {
        let m = (rng.next_u64() as u32) & ((1 << 28) - 1);
        let s = dominoes(m).unwrap();
        assert_eq!(mask(s), m);
        assert!((0..28u8).all(|t| (m >> t & 1 == 1) == s.contains(domino(t).unwrap())));
    }
    assert_eq!(dominoes(1 << 28), None);
}

#[test]
fn led_suits_are_contexts_with_seven_the_called_suit() {
    for k in 0..7u8 {
        assert_eq!(context(k), Some(Context::Natural(Pip::new(k).unwrap())));
    }
    assert_eq!(context(7), Some(Context::Called));
    assert_eq!(context(8), None);
    assert!(Context::ALL.iter().all(|q| context(led_suit(*q)) == Some(*q)));
    // A tile leads Zeb's suit 7 exactly when it is in the called set.
    for decl in 0..N_DECLS {
        let rules = Rules::new(decl).unwrap();
        for t in 0..28u8 {
            let (high, low) = rules::pips(t);
            let called = match decl {
                DOUBLES_TRUMP | DOUBLES_SUIT => high == low,
                NO_TRUMP => false,
                p => high == p || low == p,
            };
            assert_eq!(rules.led_suit[t as usize] == 7, called, "decl {decl} tile {t}");
            if !called {
                assert_eq!(context(rules.led_suit[t as usize]), Some(Context::Natural(Pip::new(high).unwrap())));
            }
        }
    }
}

#[test]
fn seats_play_clockwise_and_pair_across() {
    for s in 0..4u8 {
        let x = seat(s).unwrap();
        assert_eq!(seat_id(x), s);
        for i in 0..4u8 {
            assert_eq!(seat_id(x.plus(i as usize)), (s + i) % 4);
        }
        assert_eq!(seat_id(x.partner()), (s + 2) % 4);
        assert_eq!(x.team().index() as u8, s % 2);
    }
    assert_eq!(seat(4), None);
    assert_eq!(Seat::S1.team(), Team::T1);
}

#[test]
fn ten_declarations_ten_identities() {
    let ids: BTreeSet<String> =
        (0..N_DECLS).map(|d| ZebPhysics::new(d).unwrap().physics().identity().to_owned()).collect();
    assert_eq!(ids.len(), 10);
    assert!(ZebPhysics::new(N_DECLS).is_none());
    assert_eq!(ZebPhysics::new(5).unwrap().name(), "zeb/P5");
    assert_eq!(ZebPhysics::new(DOUBLES_SUIT).unwrap().physics().name(), "zeb/DS");
    assert!(ids.iter().all(|id| !id.contains(|c: char| c.is_whitespace() || "();".contains(c))));
    // The nine shared names are the reference's own, and so are the tables:
    // Zeb's physics and the reference's are one physics, by identity.
    for (zeb, physics, reference) in shared() {
        assert_eq!(zeb.name().strip_prefix("zeb/"), reference.name().strip_prefix("straight42/"));
        assert_eq!((physics.identity(), &physics), (reference.identity(), &reference));
        assert_eq!(physics.trump(), reference.trump());
    }
    assert_eq!(shared().len(), 9);
    assert_eq!(DECL_NAMES[DOUBLES_SUIT as usize], "DS");
    // The doubles as their own suit have no trump.
    assert_eq!(ZebPhysics::new(DOUBLES_SUIT).unwrap().physics().trump(), None);
}

// -- the tables, exhaustively ---------------------------------------------------

#[test]
fn the_tables_agree_on_every_tile_and_context() {
    let mut checked = [0usize; 5];
    for (zeb, physics, reference) in shared() {
        let name = zeb.name();
        for t in Domino::ALL {
            assert_eq!(zeb.led_context(t), reference.led_context(t), "{name} led_context {t}");
            assert_eq!(zeb.count(t), reference.count(t), "{name} count {t}");
            assert_eq!(physics.threat(t), reference.threat(t), "{name} threat {t}");
            checked[0] += 1;
        }
        for q in Context::ALL {
            assert_eq!(zeb.incidence(q), reference.incidence(q), "{name} incidence {q}");
            checked[1] += 1;
            for a in Domino::ALL {
                assert_eq!(physics.beats(q, a), reference.beats(q, a), "{name} beats {q} {a}");
                checked[2] += 1;
                for b in Domino::ALL {
                    // The keys' values may differ; their order may not.
                    let z = zeb.trick_key(a, q).cmp(&zeb.trick_key(b, q));
                    assert_eq!(z, reference.trick_key(a, q).cmp(&reference.trick_key(b, q)), "{name} {q} {a} {b}");
                    assert_eq!(z, physics.trick_key(a, q).cmp(&physics.trick_key(b, q)), "{name} {q} {a} {b}");
                    checked[3] += 1;
                }
            }
        }
    }
    assert_eq!(checked, [9 * 28, 9 * 8, 9 * 8 * 28, 9 * 8 * 28 * 28, 0]);
}

// -- plays and tricks, seeded at random -----------------------------------------

/// Zeb's own legal plays, through its tile masks.
fn zeb_legal(zeb: &ZebPhysics, hand: DominoSet, lead: Option<Domino>) -> DominoSet {
    dominoes(zeb.rules().legal(mask(hand), lead.map(tile))).unwrap()
}

#[test]
fn legal_plays_agree_on_random_hands_and_leads() {
    let mut rng = SplitMix::new(42);
    for (zeb, physics, reference) in shared() {
        for _ in 0..20_000 {
            let deck = shuffled(&mut rng);
            let hand: DominoSet = deck[..1 + below(&mut rng, 7)].iter().copied().collect();
            let lead = (below(&mut rng, 8) > 0).then(|| deck[27]);
            let silas = physics.legal_plays(hand, lead);
            assert_eq!(zeb_legal(&zeb, hand, lead), silas, "{zeb:?} {hand:?} {lead:?}");
            assert_eq!(silas, reference.legal_plays(hand, lead), "{zeb:?} {hand:?} {lead:?}");
        }
    }
}

#[test]
fn whole_random_hands_agree_trick_by_trick() {
    let mut rng = SplitMix::new(7);
    let mut tricks = 0;
    for (zeb, physics, reference) in shared() {
        for _ in 0..2_000 {
            let deck = shuffled(&mut rng);
            let mut hands: [DominoSet; 4] = std::array::from_fn(|s| deck[7 * s..7 * s + 7].iter().copied().collect());
            let mut leader = Seat::ALL[below(&mut rng, 4)];
            for _ in 0..7 {
                let mut plays = Vec::new();
                for i in 0..4 {
                    let s = leader.plus(i);
                    let legal = zeb_legal(&zeb, hands[s.index()], plays.first().copied());
                    let t = random_member(&mut rng, legal);
                    hands[s.index()].remove(t);
                    plays.push(t);
                }
                let plays: [Domino; 4] = plays.try_into().unwrap();
                let winner = leader.plus(zeb.rules().winner_offset(&plays.map(tile)) as usize);
                assert_eq!(winner, physics.trick_winner(leader, plays), "{zeb:?} {leader} {plays:?}");
                assert_eq!(winner, reference.trick_winner(leader, plays), "{zeb:?} {leader} {plays:?}");
                let points = u32::from(zeb.rules().trick_points(&plays.map(tile)));
                assert_eq!(points, physics.trick_points(plays), "{zeb:?} {plays:?}");
                assert_eq!(points, reference.trick_points(plays), "{zeb:?} {plays:?}");
                leader = winner;
                tricks += 1;
            }
        }
    }
    assert_eq!(tricks, 9 * 2_000 * 7);
}

// -- the fiber --------------------------------------------------------------------

/// A kernel from a random hand played `plays` tiles deep under `physics`,
/// seen by a random seat: its hand, the unseen pool, each other seat's
/// remaining count and the voids its plays have shown.
fn played_kernel(rng: &mut SplitMix, physics: &Physics, plays: usize) -> Kernel {
    let deck = shuffled(rng);
    let mut hands: [DominoSet; 4] = std::array::from_fn(|s| deck[7 * s..7 * s + 7].iter().copied().collect());
    let mut voids = [ContextSet::EMPTY; 4];
    let mut leader = Seat::ALL[below(rng, 4)];
    let mut trick: Vec<Domino> = Vec::new();
    for _ in 0..plays {
        let s = leader.plus(trick.len());
        let t = random_member(rng, physics.legal_plays(hands[s.index()], trick.first().copied()));
        if let Some(lead) = trick.first() {
            let q = physics.led_context(*lead);
            if !physics.follows(t, q) {
                voids[s.index()].insert(q);
            }
        }
        hands[s.index()].remove(t);
        trick.push(t);
        if let Ok(full) = <[Domino; 4]>::try_from(trick.as_slice()) {
            leader = physics.trick_winner(leader, full);
            trick.clear();
        }
    }
    let viewer = Seat::ALL[below(rng, 4)];
    let others: Vec<Seat> = Seat::ALL.into_iter().filter(|s| *s != viewer).collect();
    let pool = others.iter().fold(DominoSet::EMPTY, |p, s| p.union(hands[s.index()]));
    let hidden = std::array::from_fn(|i| {
        let s = others[i];
        Hidden { seat: s, capacity: hands[s.index()].len(), voids: voids[s.index()] }
    });
    Kernel::new(viewer, hands[viewer.index()], pool, hidden).unwrap()
}

/// A kernel with uneven capacities and voids that need not come from play
/// (so some have no worlds at all).
fn arbitrary_kernel(rng: &mut SplitMix) -> Kernel {
    let deck = shuffled(rng);
    let capacities: [usize; 3] = std::array::from_fn(|_| below(rng, 6));
    let pool_len: usize = capacities.iter().sum();
    let viewer = Seat::ALL[below(rng, 4)];
    let viewer_hand: DominoSet = deck[pool_len..pool_len + below(rng, 8)].iter().copied().collect();
    let pool: DominoSet = deck[..pool_len].iter().copied().collect();
    let others: Vec<Seat> = Seat::ALL.into_iter().filter(|s| *s != viewer).collect();
    let hidden = std::array::from_fn(|i| Hidden {
        seat: others[i],
        capacity: capacities[i],
        voids: Context::ALL.into_iter().filter(|_| below(rng, 6) == 0).collect(),
    });
    Kernel::new(viewer, viewer_hand, pool, hidden).unwrap()
}

/// Every world of a small kernel by brute force: each pool tile to each
/// hidden seat, kept when the kernel contains it.
fn brute_force(kernel: &Kernel, physics: &Physics) -> BTreeSet<World> {
    let tiles: Vec<Domino> = kernel.pool().iter().collect();
    let mut out = BTreeSet::new();
    for code in 0..3usize.pow(tiles.len() as u32) {
        let mut hands = [DominoSet::EMPTY; 3];
        let mut c = code;
        for t in &tiles {
            hands[c % 3].insert(*t);
            c /= 3;
        }
        let world = kernel.world(hands);
        if kernel.contains(physics, &world) {
            out.insert(world);
        }
    }
    out
}

/// Zeb's own worlds of a kernel (its `Fiber`), checked to be distinct members
/// of it and as many as Silas counts over Zeb's tables.
fn zeb_worlds(zeb: &ZebPhysics, kernel: &Kernel) -> BTreeSet<World> {
    let physics = zeb.physics();
    let mut fiber = zeb.fiber(kernel);
    let size = fiber.size();
    let listed: Vec<World> = (0..size)
        .map(|k| {
            let hands = fiber.world_at(k);
            kernel.world(kernel.hidden().map(|h| dominoes(hands[h.seat.index()]).unwrap()))
        })
        .collect();
    let worlds: BTreeSet<World> = listed.iter().copied().collect();
    assert_eq!(worlds.len(), listed.len(), "Zeb lists a world twice");
    assert_eq!(listed.len() as u128, physics.count_worlds(kernel));
    assert!(worlds.iter().all(|w| kernel.contains(&physics, w)));
    worlds
}

#[test]
fn the_fibers_agree_on_played_kernels() {
    let mut rng = SplitMix::new(2026);
    let (mut counted, mut listed, mut voided) = (0, 0, 0);
    for (zeb, physics, reference) in shared() {
        for _ in 0..150 {
            let plays = below(&mut rng, 29);
            let kernel = played_kernel(&mut rng, &physics, plays);
            let n = u128::try_from(zeb.fiber(&kernel).size()).unwrap();
            assert_eq!(n, physics.count_worlds(&kernel), "{zeb:?} {kernel:?}");
            assert!(n > 0, "a played kernel has its own deal");
            assert_eq!(n, reference.count_worlds(&kernel), "{zeb:?} {kernel:?}");
            counted += 1;
            voided += kernel.hidden().iter().any(|h| !h.voids.is_empty()) as usize;
            if n <= 40_000 {
                let worlds: BTreeSet<World> = reference.worlds(&kernel).collect();
                assert_eq!(zeb_worlds(&zeb, &kernel), worlds, "{zeb:?} {kernel:?}");
                listed += 1;
            }
        }
    }
    assert_eq!(counted, 9 * 150);
    assert!(listed > 500 && voided > 500, "listed {listed}, with voids {voided}");
}

#[test]
fn the_fibers_agree_on_arbitrary_kernels() {
    let mut rng = SplitMix::new(31);
    let mut empty = 0;
    for (zeb, physics, reference) in shared() {
        for _ in 0..200 {
            let kernel = arbitrary_kernel(&mut rng);
            let worlds: BTreeSet<World> = reference.worlds(&kernel).collect();
            assert_eq!(physics.worlds(&kernel).collect::<BTreeSet<_>>(), worlds, "{zeb:?} {kernel:?}");
            assert_eq!(zeb_worlds(&zeb, &kernel), worlds, "{zeb:?} {kernel:?}");
            if kernel.pool().len() <= 9 {
                assert_eq!(brute_force(&kernel, &physics), worlds);
            }
            empty += worlds.is_empty() as usize;
        }
    }
    assert!(empty > 20, "only {empty} empty fibers");
}

#[test]
fn the_doubles_suit_fiber_is_its_brute_force() {
    let zeb = ZebPhysics::new(DOUBLES_SUIT).unwrap();
    let physics = zeb.physics();
    let mut rng = SplitMix::new(8);
    let mut voided = 0;
    for _ in 0..300 {
        let kernel = arbitrary_kernel(&mut rng);
        if kernel.pool().len() <= 9 {
            let truth = brute_force(&kernel, &physics);
            assert_eq!(zeb_worlds(&zeb, &kernel), truth);
            assert_eq!(physics.worlds(&kernel).collect::<BTreeSet<_>>(), truth);
            voided += kernel.hidden().iter().any(|h| !h.voids.is_empty()) as usize;
        }
    }
    for _ in 0..100 {
        let plays = 20 + below(&mut rng, 9);
        let kernel = played_kernel(&mut rng, &physics, plays);
        assert_eq!(zeb_worlds(&zeb, &kernel), brute_force(&kernel, &physics));
        voided += kernel.hidden().iter().any(|h| !h.voids.is_empty()) as usize;
    }
    assert!(voided > 100);
}

// -- the doubles as their own suit --------------------------------------------------

#[test]
fn the_doubles_form_their_own_suit() {
    let ds = ZebPhysics::new(DOUBLES_SUIT).unwrap().physics();
    let doubles: DominoSet = Domino::ALL.into_iter().filter(|t| t.is_double()).collect();
    assert_eq!(ds.incidence(Context::Called), doubles);
    for t in Domino::ALL {
        let q = if t.is_double() { Context::Called } else { Context::Natural(t.hi()) };
        assert_eq!(ds.led_context(t), q, "{t}");
    }
    for p in Pip::ALL {
        let suit: DominoSet = Domino::ALL.into_iter().filter(|t| t.has(p) && !t.is_double()).collect();
        assert_eq!(ds.incidence(Context::Natural(p)), suit);
        assert_eq!(suit.len(), 6);
    }
    // Led doubles rank by pip: 6-6 is boss, 0-0 takes nothing.
    assert_eq!(ds.threat(d("6-6")), DominoSet::EMPTY);
    assert_eq!(ds.threat(d("0-0")), doubles.difference(set(&["0-0"])));
    assert_eq!(ds.threat(d("3-3")), set(&["4-4", "5-5", "6-6"]));
    // A non-double never beats a led double.
    assert!(Domino::ALL.iter().all(|t| t.is_double() || doubles.is_subset_of(ds.beats(Context::Called, *t))));
    assert_eq!(ds.count(d("5-5")), 10);
}

#[test]
fn the_doubles_have_no_trump_power() {
    let zeb = ZebPhysics::new(DOUBLES_SUIT).unwrap();
    let ds = zeb.physics();
    for p in Pip::ALL {
        let q = Context::Natural(p);
        for t in Domino::ALL.into_iter().filter(|t| t.is_double()) {
            // Played to a pip lead, a double is a slough: it beats nothing.
            assert!(!ds.follows(t, q));
            assert!(Domino::ALL.iter().all(|u| !ds.beats(q, *u).contains(t)), "{t} beats in {q}");
        }
    }
    // 6-6 and 5-5 to a led 5-4 take nothing; the 5-1 that follows does not
    // take it either; the leader keeps the trick.
    let trick = [d("5-4"), d("6-6"), d("5-5"), d("5-1")];
    assert_eq!(ds.trick_winner(Seat::S2, trick), Seat::S2);
    assert_eq!(ds.trick_points(trick), 1 + 10);
    // Holding the 5-5 and a five, a seat must follow fives with the five;
    // holding only doubles, anything goes.
    for (hand, lead, legal) in [
        (set(&["5-5", "5-2", "1-1"]), d("5-4"), set(&["5-2"])),
        (set(&["5-5", "1-1"]), d("5-4"), set(&["5-5", "1-1"])),
        (set(&["5-5", "5-2"]), d("1-1"), set(&["5-5"])),
    ] {
        assert_eq!(ds.legal_plays(hand, Some(lead)), legal);
        assert_eq!(zeb_legal(&zeb, hand, Some(lead)), legal);
    }
    // Under doubles trump, by contrast, the 6-6 takes the same trick.
    let dt = ZebPhysics::new(DOUBLES_TRUMP).unwrap().physics();
    assert_eq!(dt.trick_winner(Seat::S2, trick), Seat::S3);
}

// -- Silas, end to end on Zeb's physics ------------------------------------------

#[test]
fn the_four_fives_eighty_four_on_zeb_physics() {
    let physics = ZebPhysics::new(5).unwrap().physics();
    let bidder = set(&["5-5", "5-4", "5-2", "5-0", "4-4", "4-2", "2-2"]);
    let auction = Auction::won_by(Seat::S0, Bid::Marks(2)).unwrap();
    let root = Frame::new(auction, physics, Seat::S0, bidder, vec![]).unwrap();
    assert_eq!(root.count_worlds(), 399_072_960);
    let query = include_str!("../../../silas/examples/six-five-guarded.silas")
        .parse::<Query>()
        .unwrap()
        .compile(&Registry::standard())
        .unwrap();
    let belief = Belief::uniform_resolving(root, set(&["6-5", "5-1", "5-3"]), 1_000).unwrap();
    let p = query.summarize(&belief, &mut Budget::new(10_000_000)).unwrap().event_probability();
    assert_eq!((p.value().to_string(), p.authority()), ("33/95".to_string(), &silas::Authority::exact()));
}
