// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The field and the paired comparison (`docs/cross-game.md`, slice 3): a
//! hand played out by four seats, each by its own player at its own frame,
//! conserves the hand; the uniform deals and every comparison are
//! deterministic in their seed; a comparison is never `exact`, is zero for a
//! player against itself on every deal, and negates when its players swap;
//! and the chance player against a reseeded self is indistinguishable on a
//! 500-deal sample.

mod common;

use std::sync::Arc;

use common::*;
use num_rational::BigRational;
use num_traits::Zero;
use silas::reference::Straight42;
use silas::*;

fn compiled(path: &str) -> CompiledPolicy {
    let path = format!("{}/examples/book/{path}", env!("CARGO_MANIFEST_DIR"));
    let program = PolicyProgram::parse(&std::fs::read_to_string(&path).unwrap(), &path).unwrap();
    program.compile(&Registry::standard(), &physics("P6")).unwrap()
}

fn player(path: &str) -> Player {
    Player::Policy(Arc::new(compiled(path)))
}

fn table(config: AuctionConfig, scoring: Scoring) -> Table {
    Table {
        config,
        offered: Straight42::ALL.iter().map(|d| d.physics()).collect::<Arc<[Physics]>>(),
        state: MatchState::new(scoring, [0, 0]).unwrap(),
    }
}

/// Bidders of three kinds, two of whom also play by a book rule.
fn field() -> Field {
    Field::new([
        player("bidding/bid-only-enough.silas"),
        player("pounce.silas"),
        player("bidding/risk-budget.silas"),
        Player::Chance { seed: 7 },
    ])
}

fn trial(deals: Deals, seats: Seats, utility: Utility) -> Trial {
    Trial {
        field: field(),
        deals,
        seats,
        table: table(AuctionConfig::STRAIGHT, Scoring::MARKS),
        utility,
        confidence: r(19, 20),
    }
}

fn budget() -> Budget {
    Budget::new(u64::MAX)
}

#[test]
fn every_play_out_plays_seven_tricks_of_every_tile_for_the_hands_42_points() {
    let field = field();
    let worlds = Deals::Uniform(Sample { n: 30, seed: 11 }).worlds();
    let mut passed_out = 0;
    for (config, scoring) in
        [(AuctionConfig::STRAIGHT, Scoring::MARKS), (AuctionConfig::new(2, true).unwrap(), Scoring::POINTS)]
    {
        let table = table(config, scoring);
        for world in &worlds {
            for shaker in Seat::ALL {
                let settlement = play_out(world, shaker, &field, &table, &mut budget()).unwrap();
                assert_eq!(settlement.auction().shaker(), shaker);
                assert!(settlement.auction().is_complete());
                let Settlement::Played { auction, contract, outcome, plays } = settlement else {
                    assert!(!config.forced(), "a forced shaker never passes the hand out");
                    passed_out += 1;
                    continue;
                };
                assert_eq!(auction.contract(contract.declaration().clone()).unwrap(), contract);
                assert_eq!(plays.len(), 28);
                for seat in Seat::ALL {
                    let played: DominoSet = plays.iter().filter(|(s, _)| *s == seat).map(|(_, d)| *d).collect();
                    assert_eq!(played, world.hand(seat), "{seat} plays exactly its seven tiles");
                }
                // Lean's terminal_scores: seven tricks, each led by the last
                // one's winner, their points the hand's 42.
                let physics = contract.declaration();
                let (mut leader, mut banked) = (contract.declarer(), [0u32; 2]);
                let tricks: Vec<_> = plays.chunks(4).collect();
                assert_eq!(tricks.len(), 7);
                for trick in tricks {
                    assert_eq!(trick[0].0, leader);
                    let tiles: [Domino; 4] = std::array::from_fn(|k| trick[k].1);
                    leader = physics.trick_winner(leader, tiles);
                    banked[leader.team().index()] += physics.trick_points(tiles);
                }
                let side = contract.declarer().team();
                assert_eq!(outcome, Outcome::new(banked[side.index()], banked[side.other().index()]));
                assert_eq!(outcome.declaring() + outcome.other(), 42);
            }
        }
    }
    assert!(passed_out > 0, "the straight auction passes some of these hands out");
}

#[test]
fn a_passed_out_hand_moves_no_score() {
    let auction = Auction::new(AuctionConfig::STRAIGHT, Seat::S3, vec![Call::Pass; 4]).unwrap();
    let state = MatchState::new(Scoring::MARKS, [3, 5]).unwrap();
    let passed = Settlement::PassedOut { auction };
    for utility in [Utility::Marks, Utility::Points] {
        assert_eq!(passed.value(&utility, Team::T0, &state).unwrap(), BigRational::zero());
    }
}

#[test]
fn a_deal_gives_each_seat_seven_tiles_of_its_own() {
    let deck = Domino::ALL;
    let hands: [DominoSet; 4] = std::array::from_fn(|s| deck[7 * s..7 * s + 7].iter().copied().collect());
    assert!(World::deal(hands).is_ok());
    let mut short = hands;
    short[0].remove(deck[0]);
    assert!(World::deal(short).is_err());
    let mut twice = hands;
    twice[1] = hands[0];
    assert!(World::deal(twice).is_err());
}

#[test]
fn the_uniform_deals_and_the_comparison_are_determined_by_their_seed() {
    let deals = |seed| Deals::Uniform(Sample { n: 12, seed }).worlds();
    assert_eq!(deals(5), deals(5));
    assert_ne!(deals(5), deals(6));
    let t = trial(Deals::Uniform(Sample { n: 12, seed: 5 }), Seats::Fixed(Seat::S1), Utility::Marks);
    let (a, b) = (player("bidding/risk-budget.silas"), player("bidding/never-32-or-33.silas"));
    assert_eq!(t.compare(&a, &b, &mut budget()).unwrap(), t.compare(&a, &b, &mut budget()).unwrap());
    // The chance player is a function of its seed and its frame.
    let c = Player::Chance { seed: 3 };
    assert_eq!(t.compare(&c, &b, &mut budget()).unwrap(), t.compare(&c, &b, &mut budget()).unwrap());
}

#[test]
fn a_comparison_rests_on_its_field_and_sample_and_is_never_exact() {
    let (a, b) = (player("bidding/risk-budget.silas"), player("bidding/bid-only-enough.silas"));
    let id = field().id();
    assert!(id.starts_with("S0 bid-only-enough#") && id.ends_with("S3 chance#7"), "{id}");
    let listed = trial(Deals::Listed(Deals::Uniform(Sample { n: 3, seed: 1 }).worlds()), Seats::Rotate, Utility::Marks);
    let sampled = trial(Deals::Uniform(Sample { n: 3, seed: 1 }), Seats::Rotate, Utility::Points);
    for (t, samples) in [(&listed, vec![]), (&sampled, vec![Sample { n: 3, seed: 1 }])] {
        let paired = t.compare(&a, &b, &mut budget()).unwrap();
        for p in [&paired, paired.chance().expect("the chance arm")] {
            let authority = p.delta().authority();
            assert!(!authority.is_exact());
            assert_eq!(authority.models().iter().collect::<Vec<_>>(), vec![&id]);
            assert_eq!(authority.samples().iter().copied().collect::<Vec<_>>(), samples);
        }
        assert!(paired.chance().unwrap().chance().is_none());
    }
    // Over a listed set the answer is exact given the field: the interval is
    // the mean, and the verdict its sign.
    let paired = listed.compare(&a, &b, &mut budget()).unwrap();
    assert_eq!(paired.interval(), &(paired.delta().value().clone(), paired.delta().value().clone()));
}

#[test]
fn a_player_against_itself_is_zero_on_every_deal_and_swapping_negates() {
    let t = trial(Deals::Uniform(Sample { n: 24, seed: 2 }), Seats::Rotate, Utility::Marks);
    let (a, b) = (player("bidding/risk-budget.silas"), player("bidding/never-32-or-33.silas"));
    let same = t.compare(&a, &a, &mut budget()).unwrap();
    assert!(same.per_deal().iter().all(Zero::is_zero));
    assert!(same.delta().value().is_zero());
    assert_eq!((same.flips(), same.verdict()), ((0, 0, 24), Verdict::Indistinguishable));
    let (ab, ba) = (t.compare(&a, &b, &mut budget()).unwrap(), t.compare(&b, &a, &mut budget()).unwrap());
    assert!(ab.per_deal().iter().any(|d| !d.is_zero()), "the two bid differently somewhere");
    assert_eq!(ba.delta().value(), &-ab.delta().value());
    assert_eq!(ba.per_deal(), ab.per_deal().iter().map(|d| -d).collect::<Vec<_>>());
    assert_eq!(ba.interval(), &(-&ab.interval().1, -&ab.interval().0));
    let (w, l, tie) = ab.flips();
    assert_eq!(ba.flips(), (l, w, tie));
}

#[test]
fn the_chance_player_against_a_reseeded_self_is_indistinguishable_on_500_deals() {
    let t = trial(Deals::Uniform(Sample { n: 500, seed: 42 }), Seats::Fixed(Seat::S0), Utility::Marks);
    let paired = t.compare(&Player::Chance { seed: 1 }, &Player::Chance { seed: 2 }, &mut budget()).unwrap();
    let (lo, hi) = paired.interval();
    assert!(*lo < BigRational::zero() && BigRational::zero() < *hi, "[{lo}, {hi}]");
    assert_eq!(paired.verdict(), Verdict::Indistinguishable);
    let (w, l, _) = paired.flips();
    assert!(w > 0 && l > 0, "the two seeds call differently");
}

#[test]
fn a_comparison_refuses_what_it_cannot_answer() {
    let a = Player::CHANCE;
    let mut t = trial(Deals::Uniform(Sample { n: 4, seed: 0 }), Seats::Rotate, Utility::Marks);
    t.confidence = r(1, 1);
    assert!(t.compare(&a, &a, &mut budget()).is_err());
    let t = trial(Deals::Uniform(Sample { n: 1, seed: 0 }), Seats::Rotate, Utility::Marks);
    assert!(t.compare(&a, &a, &mut budget()).is_err());
    let t = trial(Deals::Listed(vec![]), Seats::Rotate, Utility::Marks);
    assert!(t.compare(&a, &a, &mut budget()).is_err());
    // A match-win value is one side's; a chair on the other is refused.
    let value = WinValue::solve(Scoring::MARKS, Team::T0, "toy: even", |_| {
        let k = Contract::new(Seat::S0, Bid::Points(30), physics("P6")).unwrap();
        Ok(vec![(k.clone(), Outcome::new(30, 12), w(1, 2)), (k, Outcome::new(0, 42), w(1, 2))])
    })
    .unwrap();
    let t = trial(Deals::Uniform(Sample { n: 2, seed: 0 }), Seats::Rotate, Utility::MatchWin(value.clone()));
    assert!(t.compare(&a, &a, &mut budget()).is_err());
    let t = trial(Deals::Uniform(Sample { n: 2, seed: 0 }), Seats::Fixed(Seat::S2), Utility::MatchWin(value));
    let paired = t.compare(&a, &a, &mut budget()).unwrap();
    assert!(paired.delta().authority().models().contains("toy: even"));
}
