//! The book's readings of a play, exactly (#128 for the play; README, The
//! record as evidence): a seat's belief conditioned on another seat's past
//! plays under a named policy. Each number is exact given that policy and
//! stands beside the uniform deal's, which the play cannot move. Every
//! reading is asked with the bidder in each of the four chairs.
mod common;

use common::rules::{B, O1, P, book_at, dealt, prose};
use common::*;
use num_rational::BigRational;
use silas::*;

fn compiled_query(file: &str) -> CompiledQuery {
    let path = format!("{}/examples/book/{file}", env!("CARGO_MANIFEST_DIR"));
    Query::parse(&std::fs::read_to_string(&path).unwrap(), &path).unwrap().compile(&Registry::standard()).unwrap()
}

fn policy_text(text: &str, frame: &Frame) -> CompiledPolicy {
    PolicyProgram::parse(text, "test").unwrap().compile(&Registry::standard(), frame.physics()).unwrap()
}

fn book_policy(file: &str, frame: &Frame) -> CompiledPolicy {
    let path = format!("{}/examples/book/{file}", env!("CARGO_MANIFEST_DIR"));
    policy_text(&std::fs::read_to_string(path).unwrap(), frame)
}

/// P(query) at `frame`: uniform, and with the plays of `readers` read under
/// each likelihood, consistent then uniform.
fn uniform_and_read(
    frame: &Frame,
    readers: &[(Seat, CompiledPolicy)],
    q: &CompiledQuery,
) -> (BigRational, BigRational, BigRational) {
    let mut budget = Budget::new(u64::MAX);
    let uniform = Belief::uniform_for(frame.clone(), q, 100_000, &mut budget).unwrap();
    let before = q.summarize(&uniform, &mut budget).unwrap().event_probability();
    assert!(before.authority().is_exact());
    let [consistent, weighed] = [Likelihood::Consistent, Likelihood::Uniform].map(|likelihood| {
        let reading = Reading::plays(readers.to_vec(), likelihood).unwrap();
        let evidence = reading.evidence(frame, &mut budget).unwrap();
        let read = Belief::given_for(&evidence, q, 100_000, &mut budget).unwrap();
        let after = q.summarize(&read, &mut budget).unwrap().event_probability();
        assert_eq!(after.authority().models().len(), readers.len(), "exact given each read policy");
        after.value().clone()
    });
    (before.value().clone(), consistent, weighed)
}

#[test]
fn s0140e_the_left_setter_knows_the_bidder_holds_both_trumps_out() {
    // Hand 31 trick 6 (deuces, 30): the left setter, on lead with the 1-1 and
    // the 5-2, knows the bidder holds both trumps still out, the top one among
    // them. Over the deals he cannot tell apart that is 2 in 15: the right
    // setter has not shown out of trump. Read under protect-the-count, the
    // right setter's trick-one 3-2 (a count trump on the bidder's 2-2) was his
    // only trump, and the partner showed out on the same trick: certain.
    let q = compiled_query("reading/bidder-holds-the-trumps-out.silas");
    let at = book_at("hand-31", 20, O1);
    for declarer in Seat::ALL {
        let frame = at.frame(declarer);
        let right = (declarer.plus(3), book_policy("protect-the-count.silas", &frame));
        assert_eq!(uniform_and_read(&frame, &[right], &q), (r(2, 15), r(1, 1), r(1, 1)), "the bidder in {declarer}");
    }
}

#[test]
fn s0138e_the_bidders_two_readings_after_trick_one() {
    // Hand 31 trick 2, the bidder to lead. First: the right setter would not
    // have played the 3-2 had he another trump (protect-the-count), so both
    // trumps out are the left setter's: 5/22 uniformly, certain read. Second:
    // the partner, giving count on the top trump, would have given the 5-5
    // before the 5-0 (donate-when-safe, the biggest first), so a setter holds
    // it: 5/8 uniformly, certain read.
    let at = book_at("hand-31", 4, B);
    let trumps = compiled_query("reading/left-setter-holds-the-trumps-out.silas");
    let five_five = compiled_query("odds/c02-opponent-double-five.silas");
    for declarer in Seat::ALL {
        let frame = at.frame(declarer);
        let right = (declarer.plus(3), book_policy("protect-the-count.silas", &frame));
        let partner = (declarer.plus(2), book_policy("donate-when-safe.silas", &frame));
        assert_eq!(uniform_and_read(&frame, std::slice::from_ref(&right), &trumps), (r(5, 22), r(1, 1), r(1, 1)));
        assert_eq!(uniform_and_read(&frame, std::slice::from_ref(&partner), &five_five), (r(5, 8), r(1, 1), r(1, 1)));
        assert_eq!(uniform_and_read(&frame, &[partner, right], &five_five), (r(5, 8), r(1, 1), r(1, 1)));
    }
}

#[test]
fn c21_the_bait_read_under_the_crisis_rule_is_evidence_not_knowledge() {
    // Hand 31 trick 2: the bidder leads the 6-5 and the partner, to play
    // after the left setter's 6-3, knows the bidder holds the 6-4, or he
    // would not have led the 6-5 (p. 107). Uniformly 5/16. Read under the
    // crisis rule, whose bait leads the 6-5 holding the 6-4, it is 91/167:
    // where the rule is silent the bidder might lead anything, the 6-5
    // included. The book's certainty needs a bidder who never leads the 6-5
    // without the 6-4, which no rule of the book says. Under the uniform
    // likelihood the silent lead is one tile in six, the bait one in one, so
    // the 6-5 is six times likelier from a hand that baits: 1549/1777. Here
    // the two likelihoods differ, and each says what it assumes.
    let at = book_at("hand-31", 6, P);
    let six_four = "(include book) (query (bidder b) (holds b 6-4))";
    let six_four = six_four.parse::<Query>().unwrap().compile(&Registry::standard()).unwrap();
    for declarer in Seat::ALL {
        let frame = at.frame(declarer);
        let bidder = (declarer, book_policy("crisis.silas", &frame));
        assert_eq!(uniform_and_read(&frame, &[bidder], &six_four), (r(5, 16), r(91, 167), r(1_549, 1_777)));
    }
}

#[test]
fn c02_the_partners_top_trump_reads_as_the_sender_plays_it() {
    // Hand 30 trick 1 (sixes, 30), the bidder's chair once the partner has
    // played the 6-6 on his 6-5. Is the partner out of trump? 26/57
    // uniformly. Read under the worked hand's sender (partner-plays-the-top-
    // trump: the top trump whenever he holds it), the play says nothing about
    // his other trumps: still 26/57. Read under p. 25's sender (the top trump
    // only when it is his only trump), certain. The reading is the sender's
    // policy, and the book's two passages give two numbers.
    let at = dealt("hand-30", B, &prose("hand-30")[..3]);
    let singleton = compiled_query("gates/c02-reading-singleton.silas");
    let alone = "(include book)
        (policy top-trump-alone
          (rule keep-the-top-trump (play low)
            (viewer me) (bidders-partner me) (bidders-lead) (trump-led)
            (own-legal low) (trump-tile low) (not (boss-trump low))))";
    for declarer in Seat::ALL {
        let frame = at.frame(declarer);
        let partner = declarer.plus(2);
        let worked = (partner, book_policy("partner-plays-the-top-trump.silas", &frame));
        let page = (partner, policy_text(alone, &frame));
        assert_eq!(uniform_and_read(&frame, &[worked], &singleton), (r(26, 57), r(26, 57), r(26, 57)));
        assert_eq!(uniform_and_read(&frame, &[page], &singleton), (r(26, 57), r(1, 1), r(1, 1)));
    }
}
