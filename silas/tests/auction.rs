// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The auction (`docs/cross-game.md`, slice 2): legality held to the arena's
//! and the engine's, the contract derived from the auction, the passed-out
//! hand, the forced bid, and the auction frame a bid rule reads.
//!
//! The authority for legality is `wiki/topics/rules-of-42.md` § Bidding and
//! texas-42's Lean `AuctionState.legalBid`. `data/auction-legal.txt` holds the
//! arena's legal calls after every legal prefix (written and checked by
//! `arena/test_silas_fixtures.py`); the TypeScript engine is held to the same
//! file (`src/tests/rules/bidding/silas-auction-fixture.test.ts`), five marks
//! included.

mod common;

use std::sync::Arc;

use common::*;
use silas::*;

/// Each line of the fixture: the prefix and the calls legal after it.
fn fixture() -> Vec<(Vec<Call>, Vec<Call>)> {
    let calls =
        |s: &str| -> Vec<Call> { s.split_whitespace().filter(|w| *w != "-").map(|w| w.parse().unwrap()).collect() };
    include_str!("data/auction-legal.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let (prefix, legal) = l.split_once(':').unwrap();
            (calls(prefix), calls(legal))
        })
        .collect()
}

#[test]
fn the_legal_calls_are_the_arenas_after_every_prefix_at_every_shaker() {
    let lines = fixture();
    // Every legal prefix of fewer than four calls: the walk of the arena's rules.
    assert_eq!(lines.len(), 834);
    for shaker in Seat::ALL {
        for (prefix, legal) in &lines {
            let auction = Auction::new(AuctionConfig::STRAIGHT, shaker, prefix.clone()).unwrap();
            assert_eq!(&auction.legal_calls(), legal, "after {auction}");
            assert_eq!(auction.actor(), Some(shaker.plus(1 + prefix.len())));
            // A call outside the legal set is refused.
            for call in std::iter::once(Call::Pass)
                .chain((25..=45).map(|n| Call::Bid(Bid::Points(n))))
                .chain((0..=6).map(|m| Call::Bid(Bid::Marks(m))))
                .filter(|c| !legal.contains(c))
            {
                assert!(auction.after(call).is_err(), "{call} after {auction}");
            }
        }
    }
}

#[test]
fn the_rules_of_42_hold_on_every_prefix() {
    // rules-of-42.md § Bidding, clause by clause, on the fixture's prefixes.
    for (prefix, legal) in fixture() {
        let high = prefix.iter().rev().find_map(|c| if let Call::Bid(b) = c { Some(*b) } else { None });
        // Each seat calls once: a pass is always legal under the straight rules.
        assert!(legal.contains(&Call::Pass));
        for call in &legal {
            let Call::Bid(bid) = call else { continue };
            // Every bid exceeds the one before; points run 30 to 41.
            assert!(high.is_none_or(|h| h < *bid), "{bid} after {high:?}");
            if let Bid::Points(n) = bid {
                assert!((30..=41).contains(n));
            }
        }
        match high {
            // Any player may bid up to two marks when two marks has not been bid.
            None | Some(Bid::Points(_)) => {
                assert!(legal.contains(&Call::Bid(Bid::Marks(1))) && legal.contains(&Call::Bid(Bid::Marks(2))));
                assert!(!legal.contains(&Call::Bid(Bid::Marks(3))));
            }
            // After a marks bid, only one mark more.
            Some(Bid::Marks(m)) => assert_eq!(legal, vec![Call::Pass, Call::Bid(Bid::Marks(m + 1))]),
        }
    }
}

#[test]
fn a_marks_cap_is_a_config_and_leans_ceiling_is_five() {
    let four = AuctionConfig::new(4, false).unwrap();
    let raise = |config, calls: &[u32]| {
        let calls = calls.iter().map(|n| Call::Bid(Bid::number(*n).unwrap())).collect();
        Auction::new(config, Seat::S3, calls).map(|a| a.legal_calls())
    };
    // The straight rules (the arena, the engine) let three raises reach five
    // marks; a cap of four, a config of its own, refuses the fifth.
    assert_eq!(raise(AuctionConfig::STRAIGHT, &[84, 126, 168]).unwrap(), vec![Call::Pass, "210".parse().unwrap()]);
    assert_eq!(raise(four, &[84, 126, 168]).unwrap(), vec![Call::Pass]);
    assert!(raise(four, &[84, 126, 168, 210]).is_err());
    // Four calls reach at most five marks under any cap (Lean's mark_le_ceiling):
    // a first marks bid is one or two, and each overcall adds one.
    let big = AuctionConfig::new(9, false).unwrap();
    assert!(raise(big, &[84, 126, 168, 210]).is_ok());
    assert!(raise(big, &[126]).is_err());
    assert!(AuctionConfig::new(0, false).is_err());
}

#[test]
fn the_contract_is_the_last_bidder_at_his_bid() {
    let p = physics("P4");
    let calls = |s: &str| s.split_whitespace().map(|w| w.parse().unwrap()).collect::<Vec<Call>>();
    // S3 shook; S0 opens 30, S1 raises to 31, S2 passes, S3 bids 84.
    let auction = Auction::new(AuctionConfig::STRAIGHT, Seat::S3, calls("30 31 pass 84")).unwrap();
    assert_eq!(auction.result(), Some((Seat::S3, Bid::Marks(2))));
    let contract = auction.contract(p.clone()).unwrap();
    assert_eq!((contract.declarer(), contract.bid(), contract.stake()), (Seat::S3, Bid::Marks(2), 2));
    // A frame derives its contract from its auction, and reads the auction.
    let dealt = tiles(&["4-4", "6-4", "5-4", "4-3", "6-6", "0-0", "6-5"]);
    let frame = Frame::new(auction.clone(), p.clone(), Seat::S3, dealt, vec![]).unwrap();
    assert_eq!(frame.contract(), &contract);
    for (query, holds) in [
        ("(shaker S3)", true),
        ("(bid-of S0 30)", true),
        ("(bid-of S1 31)", true),
        ("(passed S2)", true),
        ("(bid-of S3 84)", true),
        ("(high-bid 84)", true),
        ("(high-bidder S3)", true),
        ("(declarer S3)", true),
        ("(no-bid)", false),
        ("(passed S0)", false),
    ] {
        let answers = compile(&format!("(query {query})")).evaluate_viewer(&frame, &mut budget()).unwrap();
        assert_eq!(!answers.is_empty(), holds, "{query}");
    }
    // Under way, or passed out, there is no contract and no frame of the play.
    let under_way = Auction::new(AuctionConfig::STRAIGHT, Seat::S3, calls("30 31")).unwrap();
    assert!(under_way.contract(p.clone()).unwrap_err().0.contains("under way"));
    let passed = Auction::new(AuctionConfig::STRAIGHT, Seat::S3, calls("pass pass pass pass")).unwrap();
    assert_eq!(passed.result(), None);
    assert!(passed.contract(p.clone()).unwrap_err().0.contains("passed out"));
    assert!(Frame::new(passed, p, Seat::S0, dealt, vec![]).is_err());
}

#[test]
fn the_forced_bid_takes_the_shakers_pass_away_only_over_three_passes() {
    let forced = AuctionConfig::new(5, true).unwrap();
    let calls = |s: &str| s.split_whitespace().map(|w| w.parse().unwrap()).collect::<Vec<Call>>();
    for shaker in Seat::ALL {
        let dead = Auction::new(forced, shaker, calls("pass pass pass")).unwrap();
        assert_eq!(dead.actor(), Some(shaker));
        assert!(!dead.legal(Call::Pass));
        assert_eq!(dead.legal_calls().first(), Some(&Call::Bid(Bid::Points(30))));
        assert!(dead.after(Call::Pass).is_err());
        // Over a bid the shaker may pass as ever; and under the straight
        // rules four passes are a passed-out hand, reshaken by the next seat.
        assert!(Auction::new(forced, shaker, calls("pass 30 pass")).unwrap().legal(Call::Pass));
        let out = Auction::new(AuctionConfig::STRAIGHT, shaker, calls("pass pass pass pass")).unwrap();
        assert!(out.is_complete() && out.result().is_none());
    }
}

#[test]
fn won_by_is_the_shortest_auction_its_declarer_wins() {
    for declarer in Seat::ALL {
        for n in [30, 41, 42, 84, 126, 168, 210] {
            let bid = Bid::number(n).unwrap();
            let auction = Auction::won_by(declarer, bid).unwrap();
            assert_eq!(auction.result(), Some((declarer, bid)), "{auction}");
            let bids = auction.calls().iter().filter(|c| **c != Call::Pass).count();
            assert_eq!(bids, 1 + (n / 42).saturating_sub(2) as usize, "{auction}");
        }
        assert!(Auction::won_by(declarer, Bid::Marks(6)).is_err());
        assert!(Auction::won_by(declarer, Bid::Points(29)).is_err());
    }
}

fn menu() -> Arc<[Physics]> {
    ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "DT", "NT"].iter().map(|d| physics(d)).collect()
}

#[test]
fn the_auction_frame_reads_the_hand_the_auction_and_the_match() {
    let calls = |s: &str| s.split_whitespace().map(|w| w.parse().unwrap()).collect::<Vec<Call>>();
    let auction = Auction::new(AuctionConfig::STRAIGHT, Seat::S1, calls("31 pass")).unwrap();
    let dealt = tiles(&["5-5", "5-4", "5-2", "5-0", "4-4", "2-2", "4-2"]);
    let state = MatchState::new(Scoring::MARKS, [4, 6]).unwrap();
    let frame = AuctionFrame::new(auction, Seat::S0, dealt, state, menu()).unwrap();
    let at = |query: &str| -> Option<bool> {
        let q = compile(&format!("(query {query})"));
        q.evaluate_auction(&frame, &mut budget()).ok().map(|a| !a.is_empty())
    };
    for (query, holds) in [
        ("(viewer S0)", true),
        ("(mine 5-5)", true),
        ("(mine 6-6)", false),
        ("(count 5-5 10)", true),
        ("(bears 5-0 q0)", true),
        ("(bears 5-0 q*)", false),
        ("(high-pip 5-0 q5)", true),
        ("(high-pip 5-0 q0)", false),
        ("(shaker S1)", true),
        ("(bid-of S2 31)", true),
        ("(passed S3)", true),
        ("(no-bid)", false),
        ("(high-bid 31)", true),
        ("(high-bidder S2)", true),
        ("(least-raise 32)", true),
        ("(legal-bid 31)", false),
        ("(legal-bid 84)", true),
        ("(legal-bid 126)", false),
        ("(scoring-marks)", true),
        ("(scoring-points)", false),
        ("(score S0 4)", true),
        ("(short S0 3)", true),
        ("(short S1 1)", true),
        ("(short-at-most S0 3)", true),
        ("(short-at-most S0 2)", false),
        ("(behind S0 2)", true),
        ("(behind S0 3)", false),
        ("(behind S1 1)", false),
    ] {
        assert_eq!(at(query), Some(holds), "{query}");
    }
    // A predicate of the play is refused at the auction, before any work, and
    // a predicate of the auction alone is refused in the play.
    assert!(at("(declarer S0)").is_none());
    assert!(at("(trump q5)").is_none());
    assert!(at("(holds S1 6-6)").is_none());
    let play = framed(&contract(&physics("P5"), Seat::S0, 84), Seat::S0, dealt, vec![]).unwrap();
    let refused = compile("(query (least-raise 30))").evaluate_viewer(&play, &mut budget()).unwrap_err();
    assert!(refused.0.contains("least-raise is a predicate of the auction"), "{refused}");
    // Bids are legal only for the seat to call: undefined for another seat.
    let other = AuctionFrame::new(frame.auction().clone(), Seat::S1, dealt, state, menu()).unwrap();
    let q = compile("(query (least-raise 32))");
    let not = compile("(query (not (least-raise 32)))");
    assert!(q.evaluate_auction(&other, &mut budget()).unwrap().is_empty());
    assert!(not.evaluate_auction(&other, &mut budget()).unwrap().is_empty());
}

#[test]
fn a_suits_face_order_is_every_declarations_where_it_is_not_trump() {
    // (ranks-above a b q): a and b bear q's pip and a ranks above b by their
    // faces, the double first and then the other pip down. Under each of the
    // nine straight declarations, on the tiles that follow a pip suit q that
    // is not trump there (and are not trump themselves), it is exactly the
    // trick key's order in q: the suit's order before trump is named is the
    // order of every declaration that leaves the suit alone.
    let order = compile("(query (roles (domino a) (domino b) (context q)) (out a b q) (ranks-above a b q))");
    let dealt = tiles(&["5-5", "5-4", "5-2", "5-0", "4-4", "2-2", "4-2"]);
    let auction = Auction::new(AuctionConfig::STRAIGHT, Seat::S3, vec![]).unwrap();
    let state = MatchState::new(Scoring::MARKS, [0, 0]).unwrap();
    let at_auction = AuctionFrame::new(auction, Seat::S0, dealt, state, menu()).unwrap();
    let faces = order.evaluate_auction(&at_auction, &mut budget()).unwrap();
    let ranked = |a: Domino, b: Domino, q: Context| {
        faces.contains(&Answer(vec![Value::Domino(a), Value::Domino(b), Value::Context(q)]))
    };
    let mut compared = 0;
    for declaration in menu().iter() {
        // A tile fact: the same answers in the play, under any declaration.
        let play = framed(&contract(declaration, Seat::S0, 30), Seat::S0, dealt, vec![]).unwrap();
        assert_eq!(order.evaluate_viewer(&play, &mut budget()).unwrap(), faces, "{}", declaration.name());
        let trump = declaration.trump();
        for p in Pip::ALL {
            let q = Context::Natural(p);
            if trump == Some(q) {
                continue;
            }
            let suit: Vec<Domino> = Domino::ALL
                .into_iter()
                .filter(|d| d.has(p) && declaration.follows(*d, q) && trump.is_none_or(|t| !declaration.follows(*d, t)))
                .collect();
            for &a in &suit {
                for &b in &suit {
                    let keys = declaration.trick_key(a, q) > declaration.trick_key(b, q);
                    assert_eq!(ranked(a, b, q), keys, "{a} over {b} in {q} under {}", declaration.name());
                    compared += 1;
                }
            }
        }
    }
    // The suit's tiles off trump: 7 a suit under no trump, 6 under a pip
    // trump (the tile bearing both), 6 under doubles (the double).
    assert_eq!(compared, 7 * 49 + 7 * 6 * 6 * 6 + 7 * 36);
    // Off the suit, nothing ranks: a tile must bear the pip, and q* has none.
    assert!(!ranked(Domino::ALL[27], Domino::ALL[0], Context::Natural(Pip::ALL[3])));
    assert!(faces.iter().all(|a| a.0[2] != Value::Context(Context::Called)));
    assert_eq!(faces.len(), 7 * 21);
}

#[test]
fn an_auction_frame_refuses_what_no_table_deals() {
    let auction = Auction::new(AuctionConfig::STRAIGHT, Seat::S3, vec![]).unwrap();
    let state = MatchState::new(Scoring::POINTS, [0, 0]).unwrap();
    let six = tiles(&["5-5", "5-4", "5-2", "5-0", "4-4", "2-2"]);
    assert!(AuctionFrame::new(auction.clone(), Seat::S0, six, state, menu()).is_err());
    let seven = six.union(tiles(&["4-2"]));
    assert!(AuctionFrame::new(auction.clone(), Seat::S0, seven, state, Arc::from(vec![])).is_err());
    let twice: Arc<[Physics]> = vec![physics("P5"), physics("P5")].into();
    assert!(AuctionFrame::new(auction.clone(), Seat::S0, seven, state, twice).is_err());
    let won = MatchState::new(Scoring::MARKS, [7, 2]).unwrap();
    assert!(AuctionFrame::new(auction, Seat::S0, seven, won, menu()).is_err());
}
