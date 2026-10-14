//! The auction read as evidence (`GAPS.md` 4, #134): a belief conditioned on
//! each read seat's calls being what its bidding policy makes. Held to small
//! cases by hand, to its two limits (a policy that says nothing, a policy
//! that says everything about one tile), and to brute force: every world of a
//! small frame, each read seat's dealt hand run through its policy.
mod common;

use std::sync::Arc;

use common::*;
use silas::*;

fn policy(text: &str) -> CompiledPolicy {
    let program = PolicyProgram::parse(text, "test").unwrap_or_else(|e| panic!("{e}"));
    program.compile(&Registry::standard(), &physics("P6")).unwrap()
}

fn book_policy(file: &str) -> CompiledPolicy {
    let path = format!("{}/examples/book/bidding/{file}", env!("CARGO_MANIFEST_DIR"));
    policy(&std::fs::read_to_string(path).unwrap())
}

/// The nine straight declarations, the frame's own (P6) first.
fn offered() -> Arc<[Physics]> {
    ["P6", "P5", "P4", "P3", "P2", "P1", "P0", "DT", "NT"].iter().map(|d| physics(d)).collect()
}

fn marks() -> MatchState {
    MatchState::new(Scoring::MARKS, [0, 0]).unwrap()
}

fn reading(readers: &[(Seat, &CompiledPolicy)]) -> Reading {
    Reading::calls(readers.iter().map(|(s, p)| (*s, (*p).clone())).collect(), marks(), offered()).unwrap()
}

fn probability(text: &str, belief: &Belief) -> Estimate {
    let query = format!("(include book) {text}").parse::<Query>().unwrap().compile(&Registry::standard()).unwrap();
    query.summarize(belief, &mut budget()).unwrap().event_probability()
}

/// The posterior over `frame`'s worlds conditioned on the auction, as the
/// reading counts it, resolved on what `query` reads.
fn read(frame: &Frame, readers: &[(Seat, &CompiledPolicy)], query: &str) -> Result<Belief> {
    let evidence = reading(readers).evidence(frame, &mut budget())?;
    let compiled = format!("(include book) {query}").parse::<Query>().unwrap().compile(&Registry::standard()).unwrap();
    Belief::given_for(&evidence, &compiled, 100_000, &mut budget())
}

/// The same posterior by brute force: every world, each read seat's dealt
/// hand (its hand in the world and the tiles it played) put to its policy
/// at its turn in the auction, and its declaration when it won.
fn brute(frame: &Frame, readers: &[(Seat, &CompiledPolicy)]) -> Result<Belief> {
    let uniform = Belief::uniform(frame.clone(), 100_000).unwrap();
    let auction = frame.auction();
    let agrees = |holdings: Holdings<'_>| -> Result<Weight> {
        for (seat, policy) in readers {
            let played: DominoSet = frame.history().iter().filter(|(s, _)| s == seat).map(|(_, d)| *d).collect();
            let dealt = holdings.hand(*seat)?.union(played);
            let k = (0..4).find(|k| auction.seat_of(*k) == *seat).unwrap();
            let before = Auction::new(auction.config(), auction.shaker(), auction.calls()[..k].to_vec())?;
            let at = AuctionFrame::new(before, *seat, dealt, marks(), offered())?;
            if policy.bid(&at, &mut budget())?.choice != auction.calls()[k] {
                return Ok(Weight::zero());
            }
            if *seat == frame.contract().declarer() {
                let won = AuctionFrame::new(auction.clone(), *seat, dealt, marks(), offered())?;
                if policy.declare(&won, &mut budget())?.choice != *frame.physics() {
                    return Ok(Weight::zero());
                }
            }
        }
        Ok(Weight::one())
    };
    uniform.condition_likelihood("brute", "brute force", agrees).map(|(posterior, _)| posterior)
}

/// The small frame (S0 holds 6-6 and 0-0 and leads to trick six; 6-5 5-5
/// 3-2 6-4 2-1 1-0 lie two to a seat: 90 worlds) under another auction S0
/// wins.
fn small(shaker: Seat, calls: &[&str]) -> Frame {
    let frame = small_frame(physics("P6"));
    let calls = calls.iter().map(|c| c.parse().unwrap()).collect();
    let auction = Auction::new(AuctionConfig::STRAIGHT, shaker, calls).unwrap();
    Frame::new(auction, physics("P6"), Seat::S0, frame.dealt(), frame.plays().to_vec()).unwrap()
}

#[test]
fn a_pass_without_the_tile_puts_it_in_the_other_two_hands_by_hand() {
    // S0 opened 30 and the others passed. Under "raise holding the 6-5" S1's
    // pass says he lacks it: of the 90 worlds, the 60 where S1 does not hold
    // it remain, and the other two seats hold it in 30 each.
    let frame = small_frame(physics("P6"));
    let six_five = policy("(policy six-five (bidding (rule r (bid least-raise) (mine 6-5))))");
    let belief = read(&frame, &[(Seat::S1, &six_five)], "(query (holds S1 6-5))").unwrap();
    assert_eq!(belief.total_mass(), &Weight::integer(60));
    assert_eq!(exact_given(probability("(query (holds S1 6-5))", &belief)), r(0, 1));
    assert_eq!(exact_given(probability("(query (holds S2 6-5))", &belief)), r(1, 2));
    assert_eq!(exact_given(probability("(query (holds S3 6-5))", &belief)), r(1, 2));
    // Two passes read: the 6-5 is S3's, in the 30 worlds that put it there.
    let both = read(&frame, &[(Seat::S1, &six_five), (Seat::S2, &six_five)], "(query (holds S3 6-5))").unwrap();
    assert_eq!(both.total_mass(), &Weight::integer(30));
    assert_eq!(exact_given(probability("(query (holds S3 6-5))", &both)), r(1, 1));
    // All three: no seat may hold it, and it is in the pool: impossible.
    let all = [(Seat::S1, &six_five), (Seat::S2, &six_five), (Seat::S3, &six_five)];
    let refused = read(&frame, &all, "(query (holds S3 6-5))").unwrap_err();
    assert!(refused.0.contains("zero-mass"), "{refused}");
}

/// An estimate's value, asserting it rests on a model and no sample.
fn exact_given(e: Estimate) -> num_rational::BigRational {
    assert!(!e.authority().is_exact() && !e.authority().is_sampled(), "{}", e.authority());
    e.value().clone()
}

#[test]
fn a_policy_that_always_passes_leaves_the_belief_unchanged() {
    // The fallback is the least legal call, a pass: every hand agrees with a
    // pass, so reading it removes no world, under every set of seats read.
    let quiet = policy("(policy quiet)");
    let frame = small_frame(physics("P6"));
    let uniform = Belief::uniform(frame.clone(), 90).unwrap();
    for readers in [vec![Seat::S1], vec![Seat::S2, Seat::S3], vec![Seat::S1, Seat::S2, Seat::S3]] {
        let readers: Vec<(Seat, &CompiledPolicy)> = readers.into_iter().map(|s| (s, &quiet)).collect();
        for q in [
            "(query (holds S1 6-5))",
            "(query (some (roles d) (double d) (holds S2 d)))",
            "(query (holds S3 3-2) (holds S3 2-1))",
        ] {
            let belief = read(&frame, &readers, q).unwrap();
            assert_eq!(probability(q, &belief).value(), probability(q, &uniform).value(), "{q}");
            assert_eq!(belief.total_mass(), &Weight::integer(90));
        }
    }
    // A bid it would never make is impossible: S0, who bid, is the viewer,
    // so put the bid in S2's mouth.
    let frame = small(Seat::S1, &["30", "31", "35", "pass"]);
    let refused = read(&frame, &[(Seat::S2, &quiet)], "(query (holds S2 6-5))").unwrap_err();
    assert!(refused.0.contains("zero-mass"), "{refused}");
}

#[test]
fn a_policy_that_bids_on_the_six_six_places_it_for_certain() {
    // The opening: S0 holds no 6-6; S0 and S1 pass, S2 bids 30, S3 passes,
    // and S2 declares sixes. Under "bid holding the 6-6" (declaring the first
    // offered, sixes), S2 holds it.
    let six_six = policy("(policy six-six (bidding (rule r (bid least-raise) (mine 6-6))))");
    let hand = tiles(&["5-5", "5-4", "4-4", "3-2", "2-0", "1-1", "1-0"]);
    let auction = Auction::new(
        AuctionConfig::STRAIGHT,
        Seat::S3,
        vec![Call::Pass, Call::Pass, "30".parse().unwrap(), Call::Pass],
    );
    let frame = Frame::new(auction.unwrap(), physics("P6"), Seat::S0, hand, vec![]).unwrap();
    let uniform = Belief::uniform_for(frame.clone(), &compile("(query (holds S2 6-6))"), 1000, &mut budget()).unwrap();
    assert_eq!(probability("(query (holds S2 6-6))", &uniform).value(), &r(1, 3));
    let belief = read(&frame, &[(Seat::S2, &six_six)], "(query (holds S2 6-6))").unwrap();
    assert_eq!(exact_given(probability("(query (holds S2 6-6))", &belief)), r(1, 1));
    // C(20,6) of S2's 116,280 hands hold it; each completes C(14,7) ways.
    let evidence = reading(&[(Seat::S2, &six_six)]).evidence(&frame, &mut budget()).unwrap();
    assert_eq!((evidence.seats()[0].agreeing, evidence.seats()[0].candidates), (38_760, 116_280));
    assert_eq!(belief.total_mass(), &Weight::integer(38_760 * 3_432));
    // S1's pass alone: not his, so S2's or S3's, evenly.
    let belief = read(&frame, &[(Seat::S1, &six_six)], "(query (holds S2 6-6))").unwrap();
    assert_eq!(exact_given(probability("(query (holds S2 6-6))", &belief)), r(1, 2));
    assert_eq!(exact_given(probability("(query (holds S1 6-6))", &belief)), r(0, 1));
}

#[test]
fn the_reading_is_brute_force_over_every_world_of_a_small_frame() {
    // Three auctions S0 wins (the frame's plays keep him the declarer), the
    // book's two bidding policies and a toy one, every set of read seats,
    // and queries of one tile, of a double and of a sum: the counted
    // posterior is brute force's, exactly, and impossible together.
    let only = book_policy("bid-only-enough.silas");
    let risk = book_policy("risk-budget.silas");
    let toy =
        policy("(policy toy (bidding (rule r (bid least-raise) (mine 5-5)) (rule s (bid least-raise) (mine 6-4))))");
    let auctions: [(Seat, &[&str]); 3] = [
        (Seat::S3, &["30", "pass", "pass", "pass"]),
        (Seat::S0, &["30", "pass", "31", "32"]),
        (Seat::S1, &["30", "31", "35", "pass"]),
    ];
    let queries = [
        "(query (holds S1 6-5))",
        "(query (some (roles d) (double d) (holds S2 d)))",
        "(query (<= 10 (points (roles d) (holds S3 d))))",
        "(query (holds S2 3-2) (holds S3 1-0))",
    ];
    let (mut compared, mut impossible, mut informative) = (0, 0, 0);
    for (shaker, calls) in auctions {
        let frame = small(shaker, calls);
        for policies in [[&only, &only, &only], [&risk, &risk, &risk], [&toy, &only, &risk]] {
            for mask in 1..8u32 {
                let readers: Vec<(Seat, &CompiledPolicy)> =
                    (0..3).filter(|i| mask >> i & 1 == 1).map(|i| (Seat::ALL[i + 1], policies[i])).collect();
                let truth = brute(&frame, &readers);
                for q in queries {
                    match (read(&frame, &readers, q), &truth) {
                        (Ok(belief), Ok(truth)) => {
                            assert_eq!(belief.total_mass(), truth.total_mass(), "{calls:?} {mask} {q}");
                            assert_eq!(
                                exact_given(probability(q, &belief)),
                                probability(q, truth).value().clone(),
                                "{calls:?} {mask} {q}"
                            );
                            compared += 1;
                            informative += usize::from(belief.total_mass() != &Weight::integer(90));
                        }
                        (Err(e), Err(_)) => {
                            assert!(e.0.contains("zero-mass"), "{e}");
                            impossible += 1;
                        }
                        (a, b) => panic!("{calls:?} {mask} {q}: {:?} against {:?}", a.err(), b.as_ref().err()),
                    }
                }
            }
        }
    }
    assert!(
        compared > 150 && impossible > 0 && informative > 100,
        "{compared} compared ({informative} informative), {impossible} impossible"
    );
}

#[test]
fn the_reading_names_its_policies_and_refuses_what_it_cannot_read() {
    let only = book_policy("bid-only-enough.silas");
    let frame = small_frame(physics("P6"));
    let belief = read(&frame, &[(Seat::S2, &only)], "(query (holds S2 6-5))").unwrap();
    let models = belief.authority().models();
    assert_eq!(models.len(), 1);
    assert!(models.iter().next().unwrap().starts_with("S2 bids by bid-only-enough#"), "{models:?}");
    assert_eq!(belief.authority().to_string(), format!("exact_given({})", models.iter().next().unwrap()));
    // The viewer's own call is no evidence to him; a seat is read once.
    let refused = reading(&[(Seat::S0, &only)]).evidence(&frame, &mut budget()).err().unwrap();
    assert!(refused.0.contains("viewer"), "{refused}");
    let twice = Reading::calls(vec![(Seat::S1, only.clone()), (Seat::S1, only.clone())], marks(), offered());
    assert!(twice.err().unwrap().0.contains("twice"));
    // The declarer's declaration is read only when it was offered: S2 won
    // the opening's auction and declared sixes.
    let auction = Auction::new(
        AuctionConfig::STRAIGHT,
        Seat::S3,
        vec![Call::Pass, Call::Pass, "30".parse().unwrap(), Call::Pass],
    );
    let hand = tiles(&["5-5", "5-4", "4-4", "3-2", "2-0", "1-1", "1-0"]);
    let opening = Frame::new(auction.unwrap(), physics("P6"), Seat::S0, hand, vec![]).unwrap();
    let narrow: Arc<[Physics]> = [physics("P5")].into_iter().collect();
    let reading = Reading::calls(vec![(Seat::S2, only.clone())], marks(), narrow).unwrap();
    let message = reading.evidence(&opening, &mut budget()).err().unwrap().0;
    assert!(message.contains("not among those offered"), "{message}");
}
