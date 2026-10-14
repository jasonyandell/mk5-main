// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The plays read as evidence (#128 for the play): a belief weighed by each
//! read seat's past plays under its play policy, every play checked at that
//! seat's own frame of the time. Held to its two limits (a policy that allows
//! every play, a policy that says everything about one tile), to brute force
//! (every world of small frames, each read seat's frames built from the world
//! and the record, under both likelihoods, every set of read seats, with and
//! without the calls), and to its refusals.
mod common;

use std::sync::Arc;

use common::fixture::Role;
use common::rules::book_at;
use common::*;
use silas::*;

fn policy(text: &str) -> CompiledPolicy {
    let program = PolicyProgram::parse(text, "test").unwrap_or_else(|e| panic!("{e}"));
    program.compile(&Registry::standard(), &physics("P6")).unwrap()
}

fn book_policy(file: &str) -> CompiledPolicy {
    let path = format!("{}/examples/book/{file}", env!("CARGO_MANIFEST_DIR"));
    policy(&std::fs::read_to_string(path).unwrap())
}

fn query(text: &str) -> CompiledQuery {
    format!("(include book) {text}").parse::<Query>().unwrap().compile(&Registry::standard()).unwrap()
}

fn probability(text: &str, belief: &Belief) -> Estimate {
    query(text).summarize(belief, &mut budget()).unwrap().event_probability()
}

fn owned(readers: &[(Seat, &CompiledPolicy)]) -> Vec<(Seat, CompiledPolicy)> {
    readers.iter().map(|(s, p)| (*s, (*p).clone())).collect()
}

/// The posterior at `frame` with the plays of `readers` read under
/// `likelihood`, resolved on what `text` reads.
fn read(frame: &Frame, readers: &[(Seat, &CompiledPolicy)], likelihood: Likelihood, text: &str) -> Result<Belief> {
    let evidence = Reading::plays(owned(readers), likelihood)?.evidence(frame, &mut budget())?;
    Belief::given_for(&evidence, &query(text), 100_000, &mut budget())
}

/// What `policy` allows `seat` at `then` as the reading takes it: the
/// fallback stands for silence, every legal tile.
fn allowed(policy: &CompiledPolicy, then: &Frame) -> DominoSet {
    let allowed = policy.allowed(then, &mut budget()).unwrap();
    match allowed.provenance {
        DecisionProvenance::Fallback => then.physics().legal_plays(then.kernel().viewer_hand(), then.lead()),
        _ => allowed.tiles,
    }
}

/// The same posterior by brute force: every world of the frame, and for
/// each read seat its dealt hand (its hand in the world and the tiles it
/// played) put through its policy at each of its past plays, the frame
/// built from that hand and the record before the play. With `calls`, each
/// read seat's call (and declaration) is put to the same policy.
fn brute(frame: &Frame, readers: &[(Seat, &CompiledPolicy)], likelihood: Likelihood, calls: bool) -> Result<Belief> {
    let uniform = Belief::uniform(frame.clone(), 100_000).unwrap();
    let auction = frame.auction();
    let weigh = |holdings: Holdings<'_>| -> Result<Weight> {
        let mut weight = Weight::one();
        for (seat, policy) in readers {
            let mine: Vec<usize> = (0..frame.plays().len()).filter(|i| frame.history()[*i].0 == *seat).collect();
            let dealt = mine.iter().fold(holdings.hand(*seat)?, |h, i| h.union(DominoSet::single(frame.plays()[*i])));
            if calls {
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
            for i in mine {
                let then =
                    Frame::new(auction.clone(), frame.physics().clone(), *seat, dealt, frame.plays()[..i].to_vec())?;
                let tiles = allowed(policy, &then);
                if !tiles.contains(frame.plays()[i]) {
                    return Ok(Weight::zero());
                }
                if likelihood == Likelihood::Uniform {
                    weight = &weight * &w(1, tiles.len() as u64);
                }
            }
        }
        Ok(weight)
    };
    uniform.condition_likelihood("brute", "brute force", weigh).map(|(posterior, _)| posterior)
}

fn marks() -> MatchState {
    MatchState::new(Scoring::MARKS, [0, 0]).unwrap()
}

/// The nine straight declarations.
fn offered() -> Arc<[Physics]> {
    ["P6", "P5", "P4", "P3", "P2", "P1", "P0", "DT", "NT"].iter().map(|d| physics(d)).collect()
}

/// An estimate's value, asserting it rests on a model and no sample.
fn exact_given(e: Estimate) -> num_rational::BigRational {
    assert!(!e.authority().is_exact() && !e.authority().is_sampled(), "{}", e.authority());
    e.value().clone()
}

const ANYTHING: &str = "(policy anything (rule any (play d) (own-legal d)))";
const SIX_SIX: &str = "(policy six-six (rule six (play 6-6) (own-legal 6-6)))";
/// Plays no count when it can.
const KEEP_COUNT: &str = "(policy keep-count (rule plain (play d) (own-legal d) (count d 0)))";

#[test]
fn a_policy_that_allows_every_play_leaves_the_belief_unchanged() {
    // Under the consistent likelihood a policy that allows every legal tile,
    // by a rule or by its fallback, removes no world: the plays say nothing
    // about the hands. Every set of seats, on the small frame (five tricks
    // played) and on Hand 31 from the left setter's seat at trick six.
    let quiet = policy("(policy quiet)");
    let anything = policy(ANYTHING);
    let frames = [small_frame(physics("P6")), book_at("hand-31", 20, Role::Opponent1).frame(Seat::S0)];
    for frame in &frames {
        let uniform = Belief::uniform(frame.clone(), 100_000).unwrap();
        let hidden: Vec<Seat> = (1..4).map(|i| frame.viewer().plus(i)).collect();
        for p in [&quiet, &anything] {
            for mask in 1..8u32 {
                let readers: Vec<(Seat, &CompiledPolicy)> =
                    (0..3).filter(|i| mask >> i & 1 == 1).map(|i| (hidden[i], p)).collect();
                for q in [
                    "(query (holds S1 6-5))",
                    "(query (some (roles d) (double d) (holds S2 d)))",
                    "(query (<= 10 (points (roles d) (holds S3 d))))",
                ] {
                    let belief = read(frame, &readers, Likelihood::Consistent, q).unwrap();
                    assert_eq!(exact_given(probability(q, &belief)), exact(probability(q, &uniform)), "{q}");
                    assert_eq!(belief.total_mass(), uniform.total_mass());
                }
            }
        }
    }
    // Under the uniform likelihood the same policy is a seat choosing at
    // random among its legal tiles, which is evidence: a seat that followed
    // suit is likelier to have held few followers. On Hand 31 the left
    // setter weighs the right setter's 3-2 on the bidder's 2-2: likelier
    // his only trump.
    let frame = &frames[1];
    let q = "(query (bidder b) (trump q) (none (roles t) (in t q) (live t) (not (mine t)) (not (holds b t))))";
    let uniform = exact(probability(q, &Belief::uniform(frame.clone(), 100).unwrap()));
    let random = exact_given(probability(q, &read(frame, &[(Seat::S3, &anything)], Likelihood::Uniform, q).unwrap()));
    assert_eq!((uniform, random), (r(2, 15), r(3, 14)));
}

#[test]
fn a_policy_that_plays_the_six_six_whenever_it_can_places_it_elsewhere_when_it_did_not() {
    // Treys. S0 leads the 2-0, S1 follows with the 2-1, S2 takes it with the
    // 2-2, S3 follows with the 4-2; S2 leads the 4-4 and S3 follows with the
    // 4-1. S0, to play, cannot see the 6-6: uniformly S2's in 5 of 16. A
    // seat that leads the 6-6 whenever it holds it did not lead it, so it is
    // not S2's, under either likelihood; it is S1's in 6 of 11 and S3's in 5.
    // No follow says anything about it, since the 6-6 follows neither deuces
    // nor fours: the rule is silent there.
    let hand = tiles(&["2-0", "4-0", "6-1", "5-5", "3-3", "1-0", "5-1"]);
    let plays = ["2-0", "2-1", "2-2", "4-2", "4-4", "4-1"].iter().map(|d| tile(d)).collect();
    let frame = framed(&contract(&physics("P3"), Seat::S0, 30), Seat::S0, hand, plays).unwrap();
    let six_six = policy(SIX_SIX);
    let uniform = Belief::uniform_for(frame.clone(), &query("(query (holds S2 6-6))"), 100, &mut budget()).unwrap();
    assert_eq!(exact(probability("(query (holds S2 6-6))", &uniform)), r(5, 16));
    let s2 = vec![(Seat::S2, &six_six)];
    let all = vec![(Seat::S1, &six_six), (Seat::S2, &six_six), (Seat::S3, &six_six)];
    // Under the uniform likelihood with every seat read, the follows weigh
    // too: a follow is likelier from a seat with fewer followers, so S3,
    // who followed deuces and then fours, is likelier to hold the tiles that
    // are neither, the 6-6 among them.
    for (likelihood, readers, s1) in [
        (Likelihood::Consistent, &s2, r(6, 11)),
        (Likelihood::Consistent, &all, r(6, 11)),
        (Likelihood::Uniform, &s2, r(6, 11)),
        (Likelihood::Uniform, &all, r(5_639, 10_710)),
    ] {
        let belief = read(&frame, readers, likelihood, "(query (holds S2 6-6))").unwrap();
        assert_eq!(exact_given(probability("(query (holds S2 6-6))", &belief)), r(0, 1));
        let s3 = r(1, 1) - &s1;
        assert_eq!(exact_given(probability("(query (holds S1 6-6))", &belief)), s1, "{likelihood:?}");
        assert_eq!(exact_given(probability("(query (holds S3 6-6))", &belief)), s3, "{likelihood:?}");
    }
    // The evidence keeps S2's hands without the 6-6: C(15,5) of C(16,5).
    let evidence = Reading::plays(owned(&[(Seat::S2, &six_six)]), Likelihood::Consistent)
        .unwrap()
        .evidence(&frame, &mut budget())
        .unwrap();
    let seat = &evidence.seats()[0];
    assert_eq!((seat.agreeing, seat.candidates), (3_003, 4_368));
    let plays = seat.plays.as_ref().unwrap();
    assert_eq!((plays.policy.as_str(), plays.tiles.as_slice()), ("six-six", &[tile("2-2"), tile("4-4")][..]));
}

#[test]
fn the_play_reading_is_brute_force_over_every_world_of_small_frames() {
    // Every world of each frame, each read seat's frames built from the
    // world and the record: the counted posterior is brute force's, mass and
    // probability, under both likelihoods and every set of read seats, and
    // impossible together. The small frame under three declarations (five
    // tricks played, two tiles a seat) and Hand 31 from each seat that is to
    // play in tricks five and six (three tiles a seat, then two).
    let policies = [
        policy(ANYTHING),
        policy(SIX_SIX),
        policy(KEEP_COUNT),
        book_policy("protect-the-count.silas"),
        book_policy("donate-when-safe.silas"),
        book_policy("follow-low.silas"),
    ];
    let mut frames: Vec<Frame> = ["P6", "P2", "NT"].iter().map(|d| small_frame(physics(d))).collect();
    for (plays, role) in [(16, Role::Opponent2), (17, Role::Bidder), (18, Role::Opponent1), (20, Role::Opponent1)] {
        frames.push(book_at("hand-31", plays, role).frame(Seat::S0));
    }
    let queries = [
        "(query (holds S1 6-5))",
        "(query (some (roles d) (double d) (holds S2 d)))",
        "(query (<= 10 (points (roles d) (holds S3 d))))",
        "(query (trump q) (some (roles t) (in t q) (holds S2 t)))",
    ];
    let (mut compared, mut impossible, mut informative, mut weighed) = (0, 0, 0, 0);
    for frame in &frames {
        let hidden: Vec<Seat> = (1..4).map(|i| frame.viewer().plus(i)).collect();
        let uniform = Belief::uniform(frame.clone(), 100_000).unwrap();
        for k in 0..policies.len() {
            // The seats take the policies in turn, so mixed readings occur.
            let at = |i: usize| &policies[(k + i) % policies.len()];
            for likelihood in [Likelihood::Consistent, Likelihood::Uniform] {
                for mask in 1..8u32 {
                    let readers: Vec<(Seat, &CompiledPolicy)> =
                        (0..3).filter(|i| mask >> i & 1 == 1).map(|i| (hidden[i], at(i))).collect();
                    let truth = brute(frame, &readers, likelihood, false);
                    for q in queries {
                        match (read(frame, &readers, likelihood, q), &truth) {
                            (Ok(belief), Ok(truth)) => {
                                assert_eq!(belief.total_mass(), truth.total_mass(), "{frame:?} {mask} {q}");
                                assert_eq!(
                                    exact_given(probability(q, &belief)),
                                    probability(q, truth).value().clone(),
                                    "{frame:?} {mask} {q}"
                                );
                                compared += 1;
                                informative +=
                                    usize::from(probability(q, &belief).value() != probability(q, &uniform).value());
                                weighed += usize::from(!belief.total_mass().rational().is_integer());
                            }
                            (Err(e), Err(_)) => {
                                assert!(e.0.contains("zero-mass"), "{e}");
                                impossible += 1;
                            }
                            (a, b) => panic!("{frame:?} {mask} {q}: {:?} against {:?}", a.err(), b.as_ref().err()),
                        }
                    }
                }
            }
        }
    }
    assert!(
        compared > 1_000 && impossible > 0 && informative > 300 && weighed > 300,
        "{compared} compared ({informative} informative, {weighed} weighed), {impossible} impossible"
    );
}

#[test]
fn the_calls_and_the_plays_are_read_from_one_hand() {
    // A seat read on both is read from the one hand it was dealt: its call
    // and its plays narrow the same hands, and the reading is brute force's
    // reading of both. The small frame under the auctions S0 wins, and a
    // policy that bids holding the 5-5 and plays no count when it can.
    let both = policy(
        "(policy both (bidding (rule r (bid least-raise) (mine 5-5))) (rule plain (play d) (own-legal d) (count d 0)))",
    );
    let only = policy("(policy only (bidding (rule r (bid least-raise) (mine 5-5))))");
    let (mut compared, mut narrowed) = (0, 0);
    for (shaker, calls) in [(Seat::S3, ["30", "pass", "pass", "pass"]), (Seat::S1, ["30", "31", "35", "pass"])] {
        let small = small_frame(physics("P6"));
        let calls = calls.iter().map(|c| c.parse().unwrap()).collect();
        let auction = Auction::new(AuctionConfig::STRAIGHT, shaker, calls).unwrap();
        let frame = Frame::new(auction, physics("P6"), Seat::S0, small.dealt(), small.plays().to_vec()).unwrap();
        for likelihood in [Likelihood::Consistent, Likelihood::Uniform] {
            for mask in 1..8u32 {
                let readers: Vec<(Seat, &CompiledPolicy)> =
                    (0..3).filter(|i| mask >> i & 1 == 1).map(|i| (Seat::ALL[i + 1], &both)).collect();
                let truth = brute(&frame, &readers, likelihood, true);
                let reading = Reading::calls(owned(&readers), marks(), offered())
                    .unwrap()
                    .with_plays(owned(&readers), likelihood)
                    .unwrap();
                let calls_alone = Reading::calls(owned(&readers), marks(), offered()).unwrap();
                for q in ["(query (holds S1 5-5))", "(query (holds S2 3-2) (holds S3 1-0))"] {
                    let counted = reading
                        .evidence(&frame, &mut budget())
                        .and_then(|e| Belief::given_for(&e, &query(q), 1000, &mut budget()));
                    match (counted, &truth) {
                        (Ok(belief), Ok(truth)) => {
                            assert_eq!(belief.total_mass(), truth.total_mass());
                            assert_eq!(exact_given(probability(q, &belief)), probability(q, truth).value().clone());
                            let models = belief.authority().models();
                            assert_eq!(models.len(), 2 * readers.len(), "{models:?}");
                            let alone = calls_alone.evidence(&frame, &mut budget()).unwrap();
                            let alone = Belief::given_for(&alone, &query(q), 1000, &mut budget()).unwrap();
                            narrowed += usize::from(alone.total_mass() != belief.total_mass());
                            compared += 1;
                        }
                        (Err(e), Err(_)) => assert!(e.0.contains("zero-mass"), "{e}"),
                        (a, b) => panic!("{mask} {q}: {:?} against {:?}", a.err(), b.as_ref().err()),
                    }
                }
            }
        }
        // A policy with no play rules reads the calls alone: its plays all
        // fall back, and the fallback is silence.
        let readers = [(Seat::S1, &only), (Seat::S2, &only), (Seat::S3, &only)];
        let with = Reading::calls(owned(&readers), marks(), offered())
            .unwrap()
            .with_plays(owned(&readers), Likelihood::Consistent)
            .unwrap();
        let without = Reading::calls(owned(&readers), marks(), offered()).unwrap();
        let q = query("(query (holds S1 5-5))");
        let with = with.evidence(&frame, &mut budget()).and_then(|e| Belief::given_for(&e, &q, 1000, &mut budget()));
        let without =
            without.evidence(&frame, &mut budget()).and_then(|e| Belief::given_for(&e, &q, 1000, &mut budget()));
        assert_eq!(with.map(|b| b.total_mass().clone()), without.map(|b| b.total_mass().clone()));
    }
    assert_eq!((compared, narrowed), (20, 16), "compared, and narrowed by the plays");
}

#[test]
fn the_play_reading_names_its_policies_and_refuses_what_it_cannot_read() {
    let keep = policy(KEEP_COUNT);
    let frame = small_frame(physics("P6"));
    let q = "(query (holds S2 6-5))";
    for (likelihood, how) in [(Likelihood::Consistent, "plays within"), (Likelihood::Uniform, "plays uniformly within")]
    {
        let belief = read(&frame, &[(Seat::S2, &keep)], likelihood, q).unwrap();
        let models = belief.authority().models();
        assert_eq!(models.len(), 1);
        let model = models.iter().next().unwrap();
        assert!(model.starts_with(&format!("S2 {how} keep-count#")), "{model}");
        assert_eq!(belief.authority().to_string(), format!("exact_given({model})"));
    }
    // The viewer's own plays are no evidence to him; a seat is read once,
    // and the plays once.
    let refused = read(&frame, &[(Seat::S0, &keep)], Likelihood::Consistent, q).unwrap_err();
    assert!(refused.0.contains("viewer"), "{refused}");
    let twice = Reading::plays(owned(&[(Seat::S1, &keep), (Seat::S1, &keep)]), Likelihood::Consistent);
    assert!(twice.err().unwrap().0.contains("twice"));
    let again = Reading::plays(owned(&[(Seat::S1, &keep)]), Likelihood::Consistent)
        .unwrap()
        .with_plays(owned(&[(Seat::S2, &keep)]), Likelihood::Consistent);
    assert!(again.err().unwrap().0.contains("already"));
}
