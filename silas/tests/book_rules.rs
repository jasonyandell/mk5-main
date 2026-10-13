//! The book's rules as the book means them (Roberson, *Winning 42*): the
//! policies of `examples/book/*.silas`, checked on the book's own positions
//! from `examples/book/hands.json`. Where the book plays a rule, the book's
//! tile is allowed; where it declines, the rule does not fire. Every check
//! runs at all four seatings, the bidder in each chair.
//!
//! Each position names the catalog entry it comes from (`S-NNNN-x`, the
//! private study's catalog). Where the 2026-10-12 study's gate encoding
//! (`examples/book/gates/`) misreads the book there, the test asserts the gate
//! too: it fails the book's play and the rule passes. "Ours" marks plays the
//! book does not name, filled in consistently with what it states.
mod common;

use common::fixture::{Position, book};
use common::rules::*;
use common::*;
use silas::*;

const BIDDERS_PLAN: &str = include_str!("../examples/book/bidders-plan.silas");
const CRISIS: &str = include_str!("../examples/book/crisis.silas");
const DEAD_DOUBLE: &str = include_str!("../examples/book/dead-double.silas");
const DONATE: &str = include_str!("../examples/book/donate-when-safe.silas");
const FIRST_LEAD: &str = include_str!("../examples/book/first-lead.silas");
const HELPER_STRETCH: &str = include_str!("../examples/book/helper-stretch.silas");
const LOW_FROM_A_PAIR: &str = include_str!("../examples/book/low-from-a-pair.silas");
const PARTNER_LEADS: &str = include_str!("../examples/book/partner-leads.silas");
const TOP_TRUMP: &str = include_str!("../examples/book/partner-plays-the-top-trump.silas");
const POUNCE: &str = include_str!("../examples/book/pounce.silas");
const PROTECT: &str = include_str!("../examples/book/protect-the-count.silas");
const SETTER_LEADS: &str = include_str!("../examples/book/setter-leads.silas");
const SHED: &str = include_str!("../examples/book/shed-for-a-void.silas");
const TRUMP_SET: &str = include_str!("../examples/book/trump-set.silas");
const DEFEND_84: &str = include_str!("../examples/book/defend-the-84.silas");
const FOLLOW_LOW: &str = include_str!("../examples/book/follow-low.silas");
const FOLLOW_OFF: &str = include_str!("../examples/book/follow-with-the-off.silas");
const TAKE: &str = include_str!("../examples/book/take-the-trick.silas");

const RULES: [&str; 18] = [
    BIDDERS_PLAN,
    CRISIS,
    DEAD_DOUBLE,
    DEFEND_84,
    DONATE,
    FIRST_LEAD,
    FOLLOW_LOW,
    FOLLOW_OFF,
    HELPER_STRETCH,
    LOW_FROM_A_PAIR,
    PARTNER_LEADS,
    TOP_TRUMP,
    POUNCE,
    PROTECT,
    SETTER_LEADS,
    SHED,
    TAKE,
    TRUMP_SET,
];

const C01A: &str = include_str!("../examples/book/gates/c01a-top-trump-for-the-donation.silas");
const C01B: &str = include_str!("../examples/book/gates/c01b-partner-count-on-a-trump-lead.silas");
const C03: &str = include_str!("../examples/book/gates/c03-biggest-count-first.silas");
const C07: &str = include_str!("../examples/book/gates/c07-pounce.silas");
const C08: &str = include_str!("../examples/book/gates/c08-no-count-on-the-double.silas");
const C09: &str = include_str!("../examples/book/gates/c09-partner-never-leads-trump.silas");
const C10: &str = include_str!("../examples/book/gates/c10-off-on-trick-one.silas");
const C11: &str = include_str!("../examples/book/gates/c11-double-ahead-of-the-off.silas");
const C12A: &str = include_str!("../examples/book/gates/c12a-four-trumps-lead-the-double.silas");
const C12B: &str = include_str!("../examples/book/gates/c12b-four-trumps-lead-low.silas");
const C13A: &str = include_str!("../examples/book/gates/c13a-shed-a-singleton.silas");
const C13B: &str = include_str!("../examples/book/gates/c13b-protect-the-count.silas");
const C13C: &str = include_str!("../examples/book/gates/c13c-low-from-a-pair.silas");
const C13D: &str = include_str!("../examples/book/gates/c13d-lead-a-singleton.silas");
const C14: &str = include_str!("../examples/book/gates/c14-crisis-cash-the-double.silas");
const C15: &str = include_str!("../examples/book/gates/c15-setter-leads-call-for-count.silas");
const C21: &str = include_str!("../examples/book/gates/c21-six-five-bait.silas");
const C22: &str = include_str!("../examples/book/gates/c22-trump-the-double-from-the-left.silas");
const C23: &str = include_str!("../examples/book/gates/c23-partner-leads-doubles.silas");

/// The bidder's view of the book's hand `id` after `plays`.
fn hand(id: &str, plays: &[&str]) -> Position {
    book().bidders(id, &plays.iter().map(|d| tile(d)).collect::<Vec<_>>())
}

/// Whether the policy, firing by any rule, allows `d`.
fn allows_tile(source: &str, at: &Position, d: &str) -> bool {
    fires(source, at).is_some_and(|(_, allowed)| allowed.contains(tile(d)))
}

#[test]
fn every_rule_parses_round_trips_names_no_chair_and_compiles_under_every_declaration() {
    for source in RULES {
        let parsed = program(source);
        assert_eq!(parsed.to_string().parse::<PolicyProgram>().unwrap(), parsed);
        for chair in ["S0", "S1", "S2", "S3"] {
            assert!(!source.contains(chair), "a book rule names no chair");
        }
        for rules in reference::Straight42::ALL {
            parsed.compile(&Registry::standard(), &rules.physics()).unwrap();
        }
    }
}

// -- donate-when-safe (replaces C1b and C3)

#[test]
fn donate_on_the_top_trump_lead_s0138d_s0142f_s0142i() {
    assert_eq!(allows(DONATE, &book_at("hand-31", 2, P), "give-a-five"), tiles(&["5-0"]));
    assert_eq!(allows(DONATE, &book_at("hand-32", 2, P), "give-a-five"), tiles(&["4-1"]));
    // Hand 32 trick 2: no count left to give; the book throws the 4-2.
    assert_eq!(fires(DONATE, &book_at("hand-32", 6, P)), None);
}

#[test]
fn s0137b_s0057c_donate_on_the_bidders_double_once_the_setters_show_out_of_trump() {
    // Hand 30 trick 5: the bidder leads the 2-2 (sixes trump); both setters
    // have shown out of sixes, and the 2-2 is the top deuce. The partner, void
    // in deuces, gives the 5-0. The gates read trump leads only.
    let at = book_at("hand-30", 18, P);
    assert_eq!(allows(DONATE, &at, "give-a-five"), tiles(&["5-0"]));
    assert_eq!(fires(C01B, &at), None);
    assert_eq!(fires(C03, &at), None);
}

#[test]
fn s0144l_donate_on_the_last_trump_in_hand_33() {
    // Hand 33 trick 5: the bidder's 4-0 is the last four; the partner gives
    // the 3-2. The gate agrees here; the catalog's miss was a missing test.
    let at = book_at("hand-33", 18, P);
    assert_eq!(allows(DONATE, &at, "give-a-five"), tiles(&["3-2"]));
    assert_eq!(allows(C01B, &at, "top-lead-give-five"), tiles(&["3-2"]));
}

#[test]
fn s0133d_the_last_setter_gives_count_to_his_partners_trick() {
    // Hand 29 trick 1: the left setter's 6-4 takes the bidder's 4-0; the right
    // setter, void in fours and last, gives the 5-0 and the bid is set.
    let at = book_at("hand-29", 3, O2);
    assert_eq!(allows(DONATE, &at, "give-a-five"), tiles(&["5-0"]));
    assert_eq!(fires(C01B, &at), None);
    assert_eq!(fires(C03, &at), None);
}

#[test]
fn s0058b_s0057d_a_low_trump_lead_is_safe_once_both_setters_show_out() {
    // Ours (the book states it, p. 38): fours. Trick 1 the bidder's 4-4 finds
    // all three others void; trick 2 he leads the 4-0, and from the partner's
    // chair the 6-4 and the rest are live, but only the bidder can hold them.
    // The gate's not-top branch withholds the count; the book gives it.
    let at =
        position("P4", P, &["5-5", "5-0", "6-2", "3-1", "2-0", "0-0"], &["4-4", "3-3", "2-1", "1-1", "4-0", "6-1"]);
    assert_eq!(allows(DONATE, &at, "give-a-ten"), tiles(&["5-5"]));
    assert_eq!(allows(C01B, &at, "not-top-lead-no-count"), tiles(&["6-2", "3-1", "2-0", "0-0"]));
}

#[test]
fn s0144i_no_count_on_the_bidders_off() {
    // Hand 33 trick 3: the bidder leads the 2-0 off; the partner follows with
    // the 5-2 and keeps the 3-2. The gates are silent.
    let at = book_at("hand-33", 10, P);
    assert_eq!(allows(DONATE, &at, "no-count-on-a-trick-that-can-be-lost"), tiles(&["5-2", "6-2"]));
    assert_eq!(fires(C01B, &at), None);
}

// -- pounce (replaces C7)

#[test]
fn pounce_in_hand_32_s0142l_s0064b_s0143d() {
    // The bidder's off, beatable; the left setter loads it with a ten. The
    // book's "more urgent against 35 and 36" changes no tile.
    let at = book_at("hand-32", 9, O1);
    assert_eq!(allows(POUNCE, &at, "play-a-ten-now"), tiles(&["6-4", "5-5"]));
    assert_eq!(allows(C07, &at, "play-count-now"), tiles(&["6-4", "5-5"]));
}

#[test]
fn s0130e_s0136a_pounce_while_able_to_follow_a_double_cannot_win_your_own_count() {
    // Hand 28 trick 3: the left setter holds 3-3 and 3-2 on the bidder's 3-1;
    // the only trey count is his own, so he plays the 3-2 and hopes his
    // partner wins. Hand 30 trick 3 is the same play from the seat after the
    // bidder's partner. The gate needs a void and the left seat.
    let at = book_at("hand-28", 9, O1);
    assert_eq!(allows(POUNCE, &at, "play-a-five-now"), tiles(&["3-2"]));
    assert_eq!(fires(C07, &at), None);
    let at = book_at("hand-30", 9, O2);
    assert_eq!(allows(POUNCE, &at, "play-a-five-now"), tiles(&["3-2"]));
    assert_eq!(fires(C07, &at), None);
}

#[test]
fn s0066e_s0079b_no_pounce_on_a_lead_nothing_can_beat() {
    // Hand 35 tricks 1 and 2: the bidder leads the 4-4 and then the 6-4, the
    // top trumps; the left setter, void, holds the 5-0. The gate dumps it;
    // the book throws the dead 0-0 and then the 1-0.
    for plays in [0, 4] {
        let at = book_at("hand-35", plays + 1, O1);
        assert_eq!(fires(POUNCE, &at), None);
        assert_eq!(allows(C07, &at, "play-count-now"), tiles(&["5-0"]));
    }
}

#[test]
fn s0064f_no_count_once_the_trick_is_lost() {
    // Hand 34 tricks 2 and 3: the right setter plays last to the bidder's
    // trump; his partner has followed low and he cannot win: no count (he
    // keeps the 3-2). The gate has no rule for this seat.
    let at = book_at("hand-34", 7, O2);
    assert!(allows(POUNCE, &at, "no-count-on-a-lost-trick").contains(tile("6-0")));
    assert!(!allows(POUNCE, &at, "no-count-on-a-lost-trick").contains(tile("3-2")));
    assert_eq!(fires(C07, &at), None);
    let at = book_at("hand-34", 11, O2);
    assert_eq!(allows(POUNCE, &at, "no-count-on-a-lost-trick"), tiles(&["2-1", "3-1", "6-1", "6-3"]));
}

#[test]
fn s0064e_play_the_double_when_count_may_still_fall() {
    // Ours (p. 43): treys trump; the bidder leads the 6-2, the left setter
    // holds the 6-6 and the 6-1 and the 6-4 is out. He plays the double,
    // hoping his partner adds count. The gate needs a void.
    let at = position("P3", O1, &["6-6", "6-1", "5-5", "2-1", "1-1", "0-0", "4-4"], &["6-2"]);
    assert_eq!(allows(POUNCE, &at, "win-with-the-double"), tiles(&["6-6"]));
    assert_eq!(fires(C07, &at), None);
}

// -- protect-the-count (replaces C8 and C13b)

#[test]
fn s0144b_protect_the_count_on_the_top_trump() {
    // Hand 33 trick 1: the 4-4 leads; the left setter follows with the 4-2
    // and keeps the 4-1. The gates take non-trump doubles only.
    let at = book_at("hand-33", 1, O1);
    assert_eq!(allows(PROTECT, &at, "follow-the-master-without-count"), tiles(&["4-2"]));
    assert_eq!(fires(C08, &at), None);
    assert_eq!(fires(C13B, &at), None);
}

#[test]
fn s0071a_follow_the_double_with_the_guard_with_two_followers_or_three() {
    // Ours (p. 49): treys trump, the bidder leads the 6-6.
    let at = position("P3", O1, &["6-4", "6-0", "5-5", "2-1", "1-1", "5-1", "2-0"], &["6-6"]);
    assert_eq!(allows(PROTECT, &at, "follow-the-master-without-count"), tiles(&["6-0"]));
    let at = position("P3", O1, &["6-4", "6-0", "5-5", "2-1", "1-1", "5-1", "6-1"], &["6-6"]);
    assert_eq!(allows(PROTECT, &at, "follow-the-master-without-count"), tiles(&["6-0", "6-1"]));
    assert_eq!(fires(C13B, &at), None);
}

#[test]
fn s0078c_s0073c_in_a_trump_set_the_count_goes_on_the_double() {
    // Ours (pp. 50, 54): deuces. Trick 1 only the right setter follows the
    // 2-2: he holds the trumps. The bidder leads the 6-6; the left setter,
    // with the 6-4 and the 6-0, feeds the count his partner may trump. The C8
    // gate keeps it back.
    let at = position("P2", O1, &["6-4", "6-0", "5-5", "3-1", "1-0", "4-3"], &["2-2", "6-1", "5-1", "2-0", "6-6"]);
    assert_eq!(fires(PROTECT, &at), None);
    assert_eq!(allows(TRUMP_SET, &at, "feed-the-trump-holder"), tiles(&["6-4"]));
    assert_eq!(allows(C08, &at, "follow-without-count"), tiles(&["6-0"]));
}

// -- shed-for-a-void (replaces C13a)

#[test]
fn shed_the_books_singletons_s0070d_s0142g_s0144d_s0135d() {
    assert_eq!(allows(SHED, &book_at("hand-32", 3, O2), "throw-a-lone-tile"), tiles(&["6-2"]));
    assert_eq!(allows(SHED, &book_at("hand-30", 3, O2), "throw-a-lone-tile"), tiles(&["4-0"]));
}

#[test]
fn s0065b_never_shed_a_lone_double() {
    // Hand 33 trick 1: the right setter's lone tiles are the 6-0 and the 2-2;
    // the book throws the 6-0 and keeps the double, which the gate would
    // throw.
    let at = book_at("hand-33", 3, O2);
    assert_eq!(allows(SHED, &at, "throw-a-lone-tile"), tiles(&["6-0"]));
    assert_eq!(allows(C13A, &at, "throw-a-lone-tile"), tiles(&["6-0", "2-2"]));
}

#[test]
fn s0065b_s0071a_never_shed_the_last_guard_of_a_count_tile() {
    // Ours (p. 49): treys trump; the 6-0 is the setter's only blank but the
    // last spare of his 6-4. The gate throws it; the book keeps it, and
    // throws a plain tile instead.
    let at = position("P3", O1, &["6-4", "6-0", "5-5", "2-1", "1-1", "5-1", "4-2"], &["3-3"]);
    assert_eq!(allows(SHED, &at, "throw-what-you-need-not-keep"), tiles(&["2-1", "5-1"]));
    assert_eq!(allows(C13A, &at, "throw-a-lone-tile"), tiles(&["6-0"]));
}

#[test]
fn s0071b_keep_a_spare_of_each_suit_of_a_count_tile() {
    // Ours (p. 49): treys trump; the 4-2 is the setter's only deuce and the
    // last spare four behind his 6-4. The gate throws it; the rule keeps it.
    let at = position("P3", O1, &["6-4", "6-0", "4-2", "5-5", "5-1", "1-1", "1-0"], &["3-3"]);
    assert_eq!(allows(SHED, &at, "throw-what-you-need-not-keep"), tiles(&["5-1", "1-0"]));
    assert_eq!(allows(C13A, &at, "throw-a-lone-tile"), tiles(&["4-2"]));
}

#[test]
fn s0108a_under_doubles_keep_the_high_tiles() {
    // Ours (p. 79): doubles trump; the bidder leads the 6-6 and the setter,
    // void, holds the 6-5, the top six, alone. The gate throws it; the rule
    // keeps it.
    let at = position("DT", O1, &["6-5", "4-3", "4-2", "3-1", "2-0", "1-0", "5-1"], &["6-6"]);
    assert!(!allows(SHED, &at, "throw-what-you-need-not-keep").contains(tile("6-5")));
    assert_eq!(allows(C13A, &at, "throw-a-lone-tile"), tiles(&["6-5"]));
}

#[test]
fn s0065c_shed_on_the_bidders_double_too() {
    // Ours (p. 44): treys trump; the bidder leads the 5-5 and the setter, void
    // in fives and holding one trump, sheds a lone tile. The gate reads trump
    // leads only.
    let at = position("P3", O1, &["3-1", "6-6", "4-0", "2-1", "1-1", "6-2", "6-1"], &["5-5"]);
    assert_eq!(allows(SHED, &at, "throw-a-lone-tile"), tiles(&["4-0"]));
    assert_eq!(fires(C13A, &at), None);
}

#[test]
fn s0150c_against_84_the_book_throws_a_dead_double_not_a_singleton() {
    // Hand 35 trick 1 (84): the left setter throws the dead 0-0; the gate
    // sheds a lone tile, and the shed rule is for points.
    let at = book_at("hand-35", 1, O1);
    assert_eq!(fires(SHED, &at), None);
    assert_eq!(allows(DEAD_DOUBLE, &at, "throw-a-dead-double"), tiles(&["0-0"]));
    assert!(!allows(C13A, &at, "throw-a-lone-tile").contains(tile("0-0")));
}

// -- low-from-a-pair (replaces C13c)

#[test]
fn low_from_a_pair_in_hand_35_s0151e() {
    let at = book_at("hand-35", 19, O2);
    assert_eq!(allows(LOW_FROM_A_PAIR, &at, "follow-low"), tiles(&["5-1"]));
    assert_eq!(allows(C13C, &at, "follow-low"), tiles(&["5-1"]));
}

#[test]
fn s0097c_s0071c_against_84_low_from_a_pair_with_count() {
    // Ours (p. 70): treys trump, an 84; the bidder leads the 6-6 and the
    // setter holds the 6-5 and the 6-4. He plays the 6-4 and keeps the 6-5
    // for the off. The gate needs a pair without count.
    let at = position("P3", O1, &["6-5", "6-4", "5-5", "2-1", "1-1", "0-0", "4-4"], &["6-6"]).bid(84);
    assert_eq!(allows(LOW_FROM_A_PAIR, &at, "keep-the-higher-for-the-off"), tiles(&["6-4"]));
    assert_eq!(fires(C13C, &at), None);
}

// -- partner-leads (replaces C9 and C23)

#[test]
fn the_partners_leads_in_hands_29_30_31() {
    // S-0135-e, S-0135-h, S-0139-d, S-0139-e, S-0134-a.
    assert_eq!(allows(PARTNER_LEADS, &book_at("hand-30", 4, P), "lead-a-double-or-the-top-tile"), tiles(&["0-0"]));
    assert_eq!(allows(PARTNER_LEADS, &book_at("hand-30", 8, P), "lead-what-calls-only-a-five"), tiles(&["3-1"]));
    assert_eq!(allows(PARTNER_LEADS, &book_at("hand-31", 8, P), "lead-a-double-or-the-top-tile"), tiles(&["0-0"]));
    assert_eq!(allows(PARTNER_LEADS, &book_at("hand-31", 12, P), "lead-what-calls-no-count"), tiles(&["3-1"]));
    assert_eq!(allows(PARTNER_LEADS, &book_at("hand-29", 12, P), "lead-what-calls-no-count"), tiles(&["3-0", "6-3"]));
}

#[test]
fn s0059c_s0060b_the_partner_leads_the_top_tile_of_a_suit() {
    // Ours (p. 39): deuces trump. The 4-4, 5-4 and 6-4 are played; the
    // partner, on lead with no double and no trump, holds the 4-3, now the
    // top four. The gates lead doubles only and step aside for it.
    let at = position(
        "P2",
        P,
        &["4-3", "5-0", "6-1", "6-0", "5-1"],
        &["4-4", "5-4", "6-4", "4-0", "3-1", "3-0", "3-3", "5-3"],
    );
    assert_eq!(allows(PARTNER_LEADS, &at, "lead-a-double-or-the-top-tile"), tiles(&["4-3"]));
    assert_eq!(fires(C23, &at), None);
    assert_eq!(fires(C09, &at), None);
}

#[test]
fn s0060c_never_trump_and_a_six_before_a_four_or_a_five() {
    // Ours (p. 40): treys trump; the partner trumps in on trick 1 and leads
    // holding a trump and only fours, fives and sixes, every one calling a
    // ten. The book leads a six (it calls the 6-4 alone) and not trump; the
    // gate allows every non-trump. The exception that leads trump reads the
    // bidder's trumps (GAP likelihood).
    let at = position("P3", P, &["3-1", "5-4", "6-0", "4-0", "5-1", "6-1"], &["2-2", "2-0", "3-2", "2-1"]);
    assert_eq!(allows(PARTNER_LEADS, &at, "lead-what-calls-only-a-ten"), tiles(&["6-0", "6-1"]));
    assert_eq!(allows(C09, &at, "lead-anything-but-trump"), tiles(&["5-4", "6-0", "4-0", "5-1", "6-1"]));
}

// -- partner-plays-the-top-trump (the C2 handoff, the worked hand's play)

#[test]
fn s0135c_s0041a_the_partner_plays_the_top_trump_in_hand_30() {
    assert_eq!(allows(TOP_TRUMP, &book_at("hand-30", 2, P), "play-the-top-trump"), tiles(&["6-6"]));
    assert_eq!(fires(TOP_TRUMP, &book_at("hand-29", 2, P)), None);
}

#[test]
fn s0043b_the_singleton_reading_does_not_apply_to_the_lowest_trump() {
    // Hand 12 trick 1: the partner follows the 6-6 with the 6-0, the lowest
    // six out, not the highest (the 6-2 is out). The book draws no inference,
    // and the reading's trigger fails in every world.
    let reading = include_str!("../examples/book/gates/c02-reading-singleton.silas");
    let reading = reading.parse::<Query>().unwrap().compile(&Registry::standard()).unwrap();
    let at = hand("hand-12", &["6-6", "6-1", "6-0"]);
    for declarer in Seat::ALL {
        let belief = Belief::uniform_resolving(at.frame(declarer), tiles(&["6-2"]), 100).unwrap();
        let p = exact(reading.summarize(&belief, &mut budget()).unwrap().event_probability());
        assert_eq!(p, r(0, 1), "the bidder in {declarer}");
    }
}

// -- first-lead (replaces C10, C12a and C12b)

#[test]
fn s0040c_s0043a_s0042c_s0045a_one_off_with_four_trumps_may_go_first_but_the_book_leads_trump() {
    // Hands 12 and 14: the gate makes the off the only tile; the book leads
    // the top trump, which the rule allows with the off.
    for (id, top, off) in [("hand-12", "6-6", "2-0"), ("hand-14", "4-4", "5-2")] {
        let at = hand(id, &[]);
        let allowed = allows(FIRST_LEAD, &at, "one-off-may-go-first");
        assert!(allowed.contains(tile(top)) && allowed.contains(tile(off)), "{id}");
        assert_eq!(allows(C10, &at, "lead-the-off"), tiles(&[off]), "{id}");
    }
}

#[test]
fn s0084b_s0086a_in_an_84_the_off_never_goes_first() {
    // Hands 17 and 18: one off and four trumps, bid 84. The gate leads the
    // off; the book leads the high trumps.
    for (id, off, top) in [("hand-17", "1-0", "5-5"), ("hand-18", "5-1", "2-2")] {
        let at = hand(id, &[]);
        let allowed = allows(FIRST_LEAD, &at, "lead-the-top-trumps");
        assert!(allowed.contains(tile(top)) && !allowed.contains(tile(off)), "{id}");
        assert_eq!(allows(C10, &at, "lead-the-off"), tiles(&[off]), "{id}");
    }
}

#[test]
fn s0105a_hand_23_leads_a_low_double_under_doubles_trump() {
    // Hand 23 (doubles): four trumps, the 6-6 without the 5-5 and 4-4. The
    // book leads the 1-1; C10 leads the 6-1 off, C12b the lowest, the 0-0.
    let at = hand("hand-23", &[]);
    assert!(allows(FIRST_LEAD, &at, "double-up-lead-low").contains(tile("1-1")));
    assert_eq!(allows(C10, &at, "lead-the-off"), tiles(&["6-1"]));
    assert_eq!(allows(C12B, &at, "lead-the-lowest-trump"), tiles(&["0-0"]));
}

#[test]
fn s0050c_s0133a_three_trumps_without_the_second_lead_low_not_the_count() {
    // Hands 16 and 29: three trumps with the top, not the second. The book
    // leads the low trump that is not count; the gates take four trumps only.
    for (id, low) in [("hand-16", "3-1"), ("hand-29", "4-0")] {
        let at = if id == "hand-29" { book_at(id, 0, B) } else { hand(id, &[]) };
        assert_eq!(allows(FIRST_LEAD, &at, "double-up-lead-low"), tiles(&[low]), "{id}");
        assert_eq!(fires(C12B, &at), None, "{id}");
    }
}

#[test]
fn s0232d_with_a_trump_count_tile_out_the_double_may_go_first() {
    // Ours (p. 183): fours, the 4-4 with the 4-3, 4-2, 4-0; the 6-4 and 4-1
    // are out. The gate leads only the lowest.
    let at = position("P4", B, &["4-4", "4-3", "4-2", "4-0", "3-3", "5-5", "6-6"], &[]);
    assert_eq!(allows(FIRST_LEAD, &at, "double-up-lead-low"), tiles(&["4-4", "4-3", "4-2", "4-0"]));
    assert_eq!(allows(C12B, &at, "lead-the-lowest-trump"), tiles(&["4-0"]));
}

#[test]
fn s0146b_s0150b_hands_34_and_35_lead_the_top_trump() {
    // Hand 34: four trumps, the top and third: the double (the gate agrees;
    // double-dummy the lead sets a made 84). Hand 35: the top two held, where
    // C12A is silent and the book leads the 4-4.
    let at = book_at("hand-34", 0, B);
    assert_eq!(allows(FIRST_LEAD, &at, "double-up-lead-the-double"), tiles(&["5-5"]));
    assert_eq!(allows(C12A, &at, "lead-the-double"), tiles(&["5-5"]));
    let at = book_at("hand-35", 0, B);
    assert!(allows(FIRST_LEAD, &at, "lead-the-top-trumps").contains(tile("4-4")));
    assert_eq!(fires(C12A, &at), None);
}

#[test]
fn s0125c_with_two_trumps_lead_the_straight_off() {
    // Hand 27: two trumps; the 6-2 calls no count (the 6-4 is the bidder's
    // trump) and the 3-1 has the 3-3 behind it. The gate is silent.
    let at = hand("hand-27", &[]);
    assert_eq!(allows(FIRST_LEAD, &at, "two-trumps-lead-the-straight-off"), tiles(&["6-2"]));
    assert_eq!(fires(C10, &at), None);
}

#[test]
fn the_first_lead_allows_the_books_tile_in_every_full_deal() {
    // S-0129-d (the 6-0, holding the 0-0), S-0135-b, S-0138-b, S-0142-d,
    // S-0144-a. In Hand 32 the C10 gate leads the 3-1 off; the book the 0-0.
    for (id, lead) in [
        ("hand-28", "6-0"),
        ("hand-29", "4-0"),
        ("hand-30", "6-5"),
        ("hand-31", "2-2"),
        ("hand-32", "0-0"),
        ("hand-33", "4-4"),
        ("hand-34", "5-5"),
        ("hand-35", "4-4"),
    ] {
        assert!(allows_tile(FIRST_LEAD, &book_at(id, 0, B), lead), "{id}");
    }
    assert_eq!(allows(C10, &book_at("hand-32", 0, B), "lead-the-off"), tiles(&["3-1"]));
}

#[test]
fn s0022c_without_the_top_trump_any_trump() {
    // Hand 30: sixes without the 6-6; the book leads the 6-5. Neither C12
    // gate looks without the top trump.
    let at = book_at("hand-30", 0, B);
    assert_eq!(allows(FIRST_LEAD, &at, "lead-a-trump"), tiles(&["6-2", "6-4", "6-5"]));
    assert_eq!(fires(C12A, &at), None);
    assert_eq!(fires(C12B, &at), None);
}

// -- bidders-plan (replaces C1a and C11)

#[test]
fn s0130a_draw_the_last_trump_with_the_partner_not_shown_void() {
    // Hand 28 trick 2: the 2-0 is out; the book leads the 0-0. The gate needs
    // the partner shown void.
    let at = book_at("hand-28", 4, B);
    assert_eq!(allows(BIDDERS_PLAN, &at, "draw-the-trumps"), tiles(&["0-0"]));
    assert_eq!(fires(C01A, &at), None);
}

#[test]
fn s0136c_draw_with_the_top_trump_in_hand_30() {
    let at = book_at("hand-30", 12, B);
    assert_eq!(allows(BIDDERS_PLAN, &at, "draw-the-trumps"), tiles(&["6-4"]));
    assert_eq!(fires(C01A, &at), None);
}

#[test]
fn s0142h_draw_in_hand_32_where_the_gate_agrees() {
    let at = book_at("hand-32", 4, B);
    assert_eq!(allows(BIDDERS_PLAN, &at, "draw-the-trumps"), tiles(&["6-0"]));
    assert_eq!(allows(C01A, &at, "lead-the-top-trump"), tiles(&["6-0"]));
}

#[test]
fn s0144k_the_donation_lead_with_no_trump_out() {
    // Hand 33 trick 5: no four is out; the partner followed both trump leads
    // and has had no trick to give count on. The book leads the last trump.
    let at = book_at("hand-33", 16, B);
    assert_eq!(allows(BIDDERS_PLAN, &at, "the-donation-lead"), tiles(&["4-0"]));
    assert_eq!(fires(C01A, &at), None);
}

#[test]
fn s0047d_the_donation_lead_in_hand_15() {
    // Hand 15 trick 2 (ours: trick 1's three trumps): every trump is in, but
    // the partner cannot know it; the book leads the 6-4.
    let at = hand("hand-15", &["4-4", "4-0", "4-2", "5-4"]);
    assert_eq!(allows(BIDDERS_PLAN, &at, "the-donation-lead"), tiles(&["6-4"]));
    assert_eq!(fires(C01A, &at), None);
}

#[test]
fn s0043c_s0043e_hand_12_draws_then_dumps_the_off() {
    // Hand 12 trick 2: the 6-2 is out; the 6-5 is the only lead that tells
    // the partner the trick is safe. Trick 3 (ours: the bidder's 6-5): the
    // partner has given count; the off goes now.
    let at = hand("hand-12", &["6-6", "6-1", "6-0", "1-0"]);
    assert_eq!(allows(BIDDERS_PLAN, &at, "draw-the-trumps"), tiles(&["6-5"]));
    assert_eq!(fires(C01A, &at), None);
    let at = hand("hand-12", &["6-6", "6-1", "6-0", "1-0", "6-5", "6-2", "5-0", "3-0"]);
    assert_eq!(allows(BIDDERS_PLAN, &at, "dump-a-bare-off-calling-a-five"), tiles(&["2-0"]));
}

#[test]
fn s0105c_hand_23_draws_the_last_trump_with_the_6_6() {
    let at = hand("hand-23", &["1-1", "5-5", "4-4", "3-1", "5-3", "5-1", "5-0", "5-4"]);
    assert_eq!(allows(BIDDERS_PLAN, &at, "draw-the-trumps"), tiles(&["6-6"]));
    assert_eq!(fires(C01A, &at), None);
}

#[test]
fn s0105d_hand_23_leads_trump_again_with_none_out() {
    // Hand 23 trick 4 (ours: trick 3's other tiles; the partner follows the
    // 6-6 with the 2-2): no trump is out and the partner has had no trick to
    // give on; the book leads the 3-3 and the 0-0 for his 6-4.
    let at = hand("hand-23", &["1-1", "5-5", "4-4", "3-1", "5-3", "5-1", "5-0", "5-4", "6-6", "5-2", "2-2", "6-2"]);
    assert_eq!(allows(BIDDERS_PLAN, &at, "the-donation-lead"), tiles(&["3-3"]));
    assert_eq!(fires(C01A, &at), None);
}

#[test]
fn s0122b_hand_26_draws_with_the_6_4() {
    // Hand 26 trick 2 (ours: trick 1's tiles): the partner followed the 4-4,
    // so the gate is silent; the book leads the 6-4.
    let at = hand("hand-26", &["4-4", "4-0", "4-1", "4-3"]);
    assert_eq!(allows(BIDDERS_PLAN, &at, "draw-the-trumps"), tiles(&["6-4"]));
    assert_eq!(fires(C01A, &at), None);
}

#[test]
fn s0046c_s0104a_a_double_of_another_suit_may_pull_the_offs_count() {
    // Ours (Hand 14's idea, p. 28): fours drawn, the partner has given; the
    // bidder holds the 5-2 off and the 0-0, which may pull the 5-0 the off
    // calls. The book allows the double first; the gate takes only the off's
    // own suit.
    let plays = ["4-4", "4-0", "4-1", "5-4", "6-4", "1-0", "3-2", "6-5"];
    let at = position("P4", B, &["4-3", "4-2", "0-0", "5-2", "6-6"], &plays);
    assert_eq!(allows(BIDDERS_PLAN, &at, "dump-a-bare-off"), tiles(&["0-0", "5-2"]));
    assert_eq!(fires(C11, &at), None);
}

#[test]
fn s0087a_hand_19_saves_the_3_3_for_next_to_last() {
    // Hand 19, read as sixes (the book's "one less trump"), an 84 (ours: the
    // trump tricks). The book cashes its trump and doubles and saves the 3-3;
    // the gate leads it at once.
    let plays = ["6-6", "6-4", "6-3", "6-1", "6-5", "6-0", "1-1", "2-1"];
    let at = position("P6", B, &["3-2", "4-4", "5-5", "6-2", "3-3"], &plays).bid(84);
    assert_eq!(allows(BIDDERS_PLAN, &at, "cash-the-other-winners"), tiles(&["4-4", "5-5", "6-2"]));
    assert_eq!(allows(C11, &at, "lead-the-double"), tiles(&["3-3"]));
}

#[test]
fn s0130d_dump_the_off_that_risks_least_and_keep_the_double_ahead_of_the_other() {
    // Hand 28 trick 3: offs 3-1 (calls a five), 5-3 (calls the 5-5) and 6-1
    // (behind the 6-6). The book leads the 3-1; the C11 gate the 6-6.
    let at = book_at("hand-28", 8, B);
    assert_eq!(allows(BIDDERS_PLAN, &at, "dump-a-bare-off-calling-a-five"), tiles(&["3-1"]));
    assert_eq!(allows(C11, &at, "lead-the-double"), tiles(&["6-6"]));
    // Trick 6: the 6-6 ahead of the 6-1 (S-0131-c; the gate agrees).
    let at = book_at("hand-28", 20, B);
    assert_eq!(allows(BIDDERS_PLAN, &at, "the-double-of-the-offs-suit"), tiles(&["6-6"]));
}

#[test]
fn s0142k_s0144g_dump_the_off_after_the_trumps() {
    // Hand 32 trick 3: the 3-1 (the 2-2 may pull its 3-2, a permission the
    // book does not take here). Hand 33 trick 3: the 2-0.
    assert!(allows_tile(BIDDERS_PLAN, &book_at("hand-32", 8, B), "3-1"));
    assert!(allows_tile(BIDDERS_PLAN, &book_at("hand-33", 8, B), "2-0"));
}

#[test]
fn s0137a_the_double_ahead_with_the_last_trump_in_the_partners_hand() {
    // Hand 30 trick 5: both setters have shown out of sixes; the 6-1 is the
    // partner's. The book leads the 2-2 and keeps its own trump. The gate
    // needs trumps drawn.
    let at = book_at("hand-30", 16, B);
    assert_eq!(allows(BIDDERS_PLAN, &at, "the-double-ahead-of-the-off"), tiles(&["2-2"]));
    assert_eq!(fires(C11, &at), None);
}

#[test]
fn s0052a_hand_16_strips_with_the_5_5_before_the_2_2() {
    // Hand 16 trick 5 (ours where the book names no tile): no trump is left
    // anywhere; the bidder holds 5-5, 2-2 and the 2-1. The book leads the 5-5;
    // the gate the 2-2.
    let at = hand(
        "hand-16",
        &[
            "3-1", "3-4", "3-0", "3-6", "1-1", "1-0", "1-4", "1-5", "0-0", "3-2", "0-2", "0-5", "3-3", "5-3", "6-6",
            "4-4",
        ],
    );
    assert_eq!(allows(BIDDERS_PLAN, &at, "cash-the-other-winners"), tiles(&["5-5"]));
    assert_eq!(allows(C11, &at, "lead-the-double"), tiles(&["2-2"]));
}

#[test]
fn s0085a_s0086a_in_an_84_the_offs_double_goes_next_to_last() {
    // Hands 17 and 18 trick 5 (ours: the trump tricks): the book leads the
    // 3-3 before the 1-1, and the 0-0 before the 5-5. The gate leads the off's
    // own double at once.
    let at = hand(
        "hand-17",
        &[
            "5-5", "5-4", "5-3", "5-1", "6-5", "6-6", "4-4", "2-1", "5-2", "6-4", "0-0", "6-2", "5-0", "2-2", "4-3",
            "6-3",
        ],
    );
    assert_eq!(allows(BIDDERS_PLAN, &at, "cash-the-other-winners"), tiles(&["3-3"]));
    assert_eq!(allows(C11, &at, "lead-the-double"), tiles(&["1-1"]));
    let at = hand(
        "hand-18",
        &[
            "2-2", "5-2", "4-2", "2-0", "6-2", "6-6", "1-1", "4-4", "3-2", "6-4", "3-3", "6-3", "2-1", "6-1", "4-3",
            "6-0",
        ],
    );
    assert_eq!(allows(BIDDERS_PLAN, &at, "cash-the-other-winners"), tiles(&["0-0"]));
    assert_eq!(allows(C11, &at, "lead-the-double"), tiles(&["5-5"]));
}

#[test]
fn s0110c_at_no_trump_the_offs_doubles_wait() {
    // Hand 25 (no trump): the book leads the 3-3, 1-1 and 4-4 and keeps the
    // 5-5 and 2-2 ahead of the 5-0 and 2-1. The gate leads them at once.
    let at = hand("hand-25", &[]);
    assert_eq!(allows(BIDDERS_PLAN, &at, "cash-the-other-winners"), tiles(&["1-1", "3-3", "4-4"]));
    assert_eq!(allows(C11, &at, "lead-the-double"), tiles(&["1-1", "2-2", "5-5"]));
}

#[test]
fn s0122c_s0122e_hand_26_strips_with_the_1_1_and_saves_the_5_5() {
    // Hand 26, first line (ours: the setters' tiles). Trick 3: no trump out,
    // the 3-2 and 5-0 out; the book leads the 1-1, the gate the 2-2 or the
    // 5-5 (the 5-2 follows deuces too). Trick 6:
    // the 5-5 ahead of the 5-2 (the gate agrees).
    let tricks = [
        "4-4", "4-0", "4-1", "4-3", "6-4", "5-4", "4-2", "6-6", "1-1", "1-0", "2-1", "6-1", "3-3", "3-0", "3-1", "3-2",
        "2-2", "6-2", "2-0", "5-3",
    ];
    let at = hand("hand-26", &tricks[..8]);
    assert_eq!(allows(BIDDERS_PLAN, &at, "cash-the-other-winners"), tiles(&["1-1", "2-2", "3-3"]));
    assert_eq!(allows(C11, &at, "lead-the-double"), tiles(&["2-2", "5-5"]));
    let at = hand("hand-26", &tricks);
    assert_eq!(allows(BIDDERS_PLAN, &at, "the-double-of-the-offs-suit"), tiles(&["5-5"]));
}

#[test]
fn s0040b_with_a_trump_to_come_back_the_bare_off_goes_before_an_unrelated_double() {
    // Ours (p. 24): fours trump, drawn in two tricks; the bidder holds the
    // 2-2, the 6-3 off and the 4-2. The 2-2 could pull a six only below the 6-3. The
    // gate is silent; the rule leads the off.
    let plays = ["4-4", "4-0", "4-1", "4-3", "6-4", "5-4", "1-0", "3-0"];
    let at = position("P4", B, &["2-2", "6-3", "4-2", "5-5", "1-1"], &plays);
    assert_eq!(allows(BIDDERS_PLAN, &at, "dump-a-bare-off-calling-no-count"), tiles(&["6-3"]));
    assert_eq!(fires(C11, &at), None);
}

#[test]
fn the_plan_in_the_84s_of_hands_34_and_35_s0148b() {
    for (id, plays, lead) in [
        ("hand-34", 8, "5-2"),
        ("hand-34", 12, "5-0"),
        ("hand-34", 16, "2-2"),
        ("hand-34", 20, "4-4"),
        ("hand-35", 8, "4-3"),
        ("hand-35", 12, "4-0"),
        ("hand-35", 16, "5-5"),
        ("hand-35", 20, "6-6"),
        ("hand-33", 20, "1-1"),
        ("hand-29", 16, "4-4"),
    ] {
        assert!(allows_tile(BIDDERS_PLAN, &book_at(id, plays, B), lead), "{id} after {plays}");
    }
}

// -- crisis (replaces C14 and C21)

#[test]
fn s0139a_the_six_five_bait_in_hand_31() {
    let at = book_at("hand-31", 4, B);
    assert_eq!(allows(CRISIS, &at, "the-six-five-bait"), tiles(&["6-5"]));
    assert_eq!(allows(C21, &at, "lead-the-six-five"), tiles(&["6-5"]));
    assert_eq!(fires(BIDDERS_PLAN, &at), None);
}

#[test]
fn s0167b_the_bait_is_a_crisis_lead() {
    // Hand 31 trick 1 (deuces): the bidder holds the 6-5 and 6-4 and leads
    // the 2-2. The C21 gate leads the 6-5 on any lead; the book, in crisis.
    let at = book_at("hand-31", 0, B);
    assert_eq!(fires(CRISIS, &at), None);
    assert_eq!(allows(C21, &at, "lead-the-six-five"), tiles(&["6-5"]));
}

#[test]
fn s0073d_s0123d_s0123e_s0124a_hand_26s_crisis() {
    // Hand 26, second line (ours: the tricks' other tiles): after two trump
    // tricks the left setter keeps the 4-2. The bidder cashes the 5-5, itself
    // a ten; then, the trump holder having followed with the 5-3, the 2-2
    // rather than the 3-3 (the 1-1 calls no count). The C14 gate's trigger is
    // one trump trick.
    let plays = ["4-4", "4-0", "4-1", "4-3", "6-4", "4-5", "6-6", "3-0", "5-5", "5-3", "5-1", "6-5"];
    let at = hand("hand-26", &plays[..8]);
    assert_eq!(allows(CRISIS, &at, "cash-a-double-that-is-count"), tiles(&["5-5"]));
    assert_eq!(fires(C14, &at), None);
    let at = hand("hand-26", &plays);
    assert_eq!(allows(CRISIS, &at, "a-count-double-away-from-his-suit"), tiles(&["2-2"]));
    assert_eq!(fires(C14, &at), None);
}

#[test]
fn s0124a_of_the_count_doubles_the_suit_with_the_most_tiles_out() {
    // Hand 26's crisis once the 5-5 is cashed (above): the bidder keeps the
    // 3-3, 2-2 and 1-1. Twos out: 6-2, 3-2, 2-1, 2-0 (the 4-2 is trump);
    // threes out: 6-3, 3-2, 3-1. The 2-2's suit has the most out, and the
    // rule that ignores the trump holder's suit allows it alone, where the
    // gate's encoding allowed every count-calling double.
    let plays = ["4-4", "4-0", "4-1", "4-3", "6-4", "4-5", "6-6", "3-0", "5-5", "5-3", "5-1", "6-5"];
    let at = hand("hand-26", &plays);
    let more_out = |t: &str, s: &str| {
        let text = format!("(include book) (query (more-out {t} {s}))");
        let query = text.parse::<Query>().unwrap().compile(&Registry::standard()).unwrap();
        !query.evaluate_viewer(&at.frame(Seat::S0), &mut budget()).unwrap().is_empty()
    };
    assert!(more_out("q2", "q3") && !more_out("q3", "q2") && !more_out("q2", "q2"));
    // The third rule set aside (it allows the bidder's 0-0 alone, which he
    // does not hold), the fourth decides.
    let without_the_holders_suit = CRISIS.replace(
        "(rule a-count-double-away-from-his-suit (play dd)",
        "(rule a-count-double-away-from-his-suit (play dd) (tile dd 0-0)",
    );
    assert_ne!(without_the_holders_suit, CRISIS);
    assert_eq!(allows(&without_the_holders_suit, &at, "a-count-double"), tiles(&["2-2"]));
}

#[test]
fn s0127a_the_gates_one_trump_trick_is_not_a_crisis() {
    // Ours (the gate's own trigger): fours, after one trump trick the right
    // setter has shown out, but the bidder's 6-4 still draws both trumps out.
    // The gate cashes a double; the book draws.
    let at = position("P4", B, &["3-3", "1-1", "2-2", "5-2", "6-4", "4-3"], &["4-4", "4-0", "4-1", "3-0"]);
    assert_eq!(fires(CRISIS, &at), None);
    assert_eq!(allows(BIDDERS_PLAN, &at, "draw-the-trumps"), tiles(&["6-4"]));
    assert_eq!(allows(C14, &at, "cash-a-double"), tiles(&["1-1", "2-2", "3-3"]));
}

// -- setter-leads (replaces C15 and C13d)

#[test]
fn s0130f_lead_the_count_that_sets_in_hand_28() {
    // Hand 28 trick 4: the setters have 6 and need 13. The 5-4 calls the 5-5:
    // eleven. The gates lead the 1-1.
    let at = book_at("hand-28", 12, O2);
    let allowed = allows(SETTER_LEADS, &at, "lead-the-count-that-sets");
    assert!(allowed.contains(tile("5-4")) && !allowed.contains(tile("1-1")));
    assert_eq!(allows(C15, &at, "a-double-that-calls-for-count"), tiles(&["1-1"]));
    assert_eq!(allows(C13D, &at, "lead-the-lone-tile"), tiles(&["1-1"]));
}

#[test]
fn s0140b_lead_the_count_that_sets_in_hand_31() {
    // Hand 31 trick 5: the setters have 1 and need 13. The 4-1 with the 6-4
    // makes sixteen; the 4-4 calls only the ten.
    let at = book_at("hand-31", 16, O2);
    assert_eq!(allows(SETTER_LEADS, &at, "lead-the-count-that-sets"), tiles(&["4-1"]));
    assert_eq!(allows(C15, &at, "a-double-that-calls-for-count"), tiles(&["4-4"]));
}

#[test]
fn s0131a_s0067c_after_the_set_lead_count_that_calls_for_more() {
    // Hand 28 trick 5: the bidder is set; the left setter leads the 4-1 for
    // fifteen with the 6-4. The gate leads the 4-4.
    let at = book_at("hand-28", 16, O1);
    assert_eq!(allows(SETTER_LEADS, &at, "lead-count-that-calls-for-more"), tiles(&["4-1"]));
    assert_eq!(allows(C15, &at, "a-double-that-calls-for-count"), tiles(&["4-4"]));
}

#[test]
fn the_setters_leads_where_the_gate_agrees_s0133e_s0133i_s0144j() {
    assert_eq!(allows(SETTER_LEADS, &book_at("hand-29", 4, O1), "a-double-that-calls-for-count"), tiles(&["2-2"]));
    assert_eq!(fires(SETTER_LEADS, &book_at("hand-29", 8, O1)), None);
    assert_eq!(allows(SETTER_LEADS, &book_at("hand-33", 12, O2), "lead-the-count-that-sets"), tiles(&["5-0"]));
}

#[test]
fn s0070c_one_point_sets_lead_a_double() {
    // Ours (p. 48): deuces, a bid of 31. The left setter trumps the bidder's
    // 5-5 for eleven; one point sets. He holds the 3-3, which calls no count,
    // and the 6-1 and 6-0, which call the 6-4. The book leads the double; the
    // gate the six.
    let at = position("P2", O1, &["3-3", "6-1", "4-3", "1-0", "6-0", "3-1"], &["5-5", "2-0", "5-4", "5-3"]).bid(31);
    assert_eq!(allows(SETTER_LEADS, &at, "one-point-sets-lead-a-double"), tiles(&["3-3"]));
    assert_eq!(allows(C15, &at, "a-tile-that-calls-for-count"), tiles(&["6-0", "6-1"]));
}

#[test]
fn s0069d_think_twice_about_a_low_six_against_35() {
    // Ours (p. 47): deuces, a bid of 35, already set. The left setter's only
    // count-calling leads are the low sixes, which call the 6-4; the gate
    // leads them as a matter of course, the book only after thinking twice.
    let at = position("P2", O1, &["6-1", "3-0", "4-0", "3-1", "1-0", "6-0"], &["5-5", "2-0", "5-4", "5-3"]).bid(35);
    assert_eq!(allows(SETTER_LEADS, &at, "think-twice-a-low-four-five-or-six"), tiles(&["6-0", "6-1"]));
    assert_eq!(allows(C15, &at, "a-tile-that-calls-for-count"), tiles(&["6-0", "6-1"]));
}

// -- trump-set (replaces C22)

/// Deuces: the bidder leads the 2-2 and only the left setter follows.
const ALONE: [&str; 4] = ["2-2", "2-0", "6-1", "5-1"];

#[test]
fn s0074a_the_trump_set_needs_the_bidders_partner_void_too() {
    // Ours (p. 51): the partner follows the 2-2 with the 2-1, so the left
    // setter is not alone with the trumps; the gate's trigger does not test
    // the bidder's partner.
    let hand = ["5-2", "4-2", "1-1", "3-3", "1-0", "4-0"];
    let at = position("P2", O1, &hand, &["2-2", "2-0", "2-1", "6-1", "6-6"]);
    assert_eq!(fires(TRUMP_SET, &at), None);
    assert_eq!(allows(C22, &at, "trump-in"), tiles(&["5-2", "4-2"]));
    let at = position("P2", O1, &hand, &[&ALONE[..], &["6-6"]].concat());
    assert_eq!(allows(TRUMP_SET, &at, "trump-a-big-double"), tiles(&["5-2", "4-2"]));
}

#[test]
fn s0074b_usually_and_above_all_the_big_doubles() {
    // Ours: the same trump set, the bidder leads the 3-3; the left setter,
    // void in treys, still trumps (usually), by the second rule.
    let at = position("P2", O1, &["5-2", "4-2", "1-1", "1-0", "4-0", "6-6"], &[&ALONE[..], &["3-3"]].concat());
    assert_eq!(allows(TRUMP_SET, &at, "trump-a-double"), tiles(&["5-2", "4-2"]));
}

#[test]
fn s0074c_the_last_setter_trumps_a_double_that_holds_count() {
    // Ours (p. 51): deuces, the right setter holds the trumps (the bidder's
    // partner void with the 6-1). The bidder leads the 6-6 and the left setter
    // puts the 6-4 on it: he trumps. With no count on the trick he does not.
    let hand = ["5-2", "4-2", "1-1", "3-3", "1-0", "4-0"];
    let alone = ["2-2", "6-0", "6-1", "2-0"];
    let at = position("P2", O2, &hand, &[&alone[..], &["6-6", "6-4", "6-3"]].concat());
    assert_eq!(allows(TRUMP_SET, &at, "trump-in-on-count"), tiles(&["5-2", "4-2"]));
    let at = position("P2", O2, &hand, &[&alone[..], &["6-6", "6-5", "6-3"]].concat());
    assert_eq!(fires(TRUMP_SET, &at), None);
}

#[test]
fn s0075a_the_setter_on_lead_leads_the_trump_that_beats_the_rest() {
    // Ours (p. 52): deuces. The bidder leads the 2-0 and the left setter, who
    // holds the 2-2, takes it with the others void: the trump set, and he is
    // on lead with the 2-6 the top trump still out.
    let hand = ["2-6", "2-5", "1-1", "3-3", "1-0", "4-0"];
    let at = position("P2", O1, &hand, &["2-0", "2-2", "6-1", "5-1"]);
    assert_eq!(allows(TRUMP_SET, &at, "lead-a-high-trump"), tiles(&["2-6"]));
    // Without it, a trump below the top still out is not a lead the rule makes.
    let hand = ["2-5", "2-4", "1-1", "3-3", "1-0", "4-0"];
    let at = position("P2", O1, &hand, &["2-0", "2-2", "6-1", "5-1"]);
    assert_eq!(fires(TRUMP_SET, &at), None);
}

#[test]
fn the_books_trump_sets_reach_the_ruff_with_no_trump_left_s0072c_s0074a() {
    // The book narrates three trump sets (Hands 32, 34, 35) and no hand for
    // the rules above (the diagram of p. 51 gives none). In each the bidder
    // keeps leading trump and draws the holder's last one before he leads a
    // double the holder is void in, so every narrated position that reaches
    // the setter's ruff has nothing to ruff with: Hand 34 trick 5 (the 2-2)
    // and trick 6 (the 4-4) from the left setter, Hand 35 trick 5 (the 5-5)
    // from the right. A rule that asks only for a legal trump (no second one)
    // after a trump set and a non-trump double stays silent at all three, and
    // answers the constructed set of p. 51.
    const ANY_TRUMP: &str = "(include book)
        (policy any-trump
          (rule r (play t)
            (viewer me)
            (trump-set-alone me)
            (bidder-led doubled)
            (double doubled)
            (not (trump-tile doubled))
            (own-legal t)
            (trump-tile t)))";
    for (id, plays, role) in [("hand-34", 17, O1), ("hand-34", 21, O1), ("hand-35", 19, O2)] {
        let at = book_at(id, plays, role);
        assert_eq!(fires(ANY_TRUMP, &at), None, "{id} after {plays} plays");
        assert_eq!(fires(TRUMP_SET, &at), None, "{id} after {plays} plays");
    }
    let hand = ["5-2", "4-2", "1-1", "3-3", "1-0", "4-0"];
    let at = position("P2", O1, &hand, &[&ALONE[..], &["6-6"]].concat());
    assert_eq!(allows(ANY_TRUMP, &at, "r"), tiles(&["5-2", "4-2"]));
}

#[test]
fn s0073c_no_feed_once_the_trump_holder_cannot_trump() {
    // The pour, s300001 deal 43342 ply 17: treys. On the bidder's 5-3 only
    // the right setter followed (the 6-3); every trump is now played. The
    // bidder leads the 0-0 and the left setter holds the 5-5: nobody can
    // trump the double, and the book's feed is to a holder who may yet trump.
    let plays = [
        "3-3", "4-3", "3-2", "3-0", "5-3", "1-1", "2-0", "6-3", "6-5", "6-1", "4-1", "6-6", "6-0", "6-4", "3-1", "5-4",
        "0-0",
    ];
    assert_eq!(fires(TRUMP_SET, &position("P3", O1, &["5-5", "5-1", "2-2"], &plays)), None);
    // s200001 deal 9437 ply 21: blanks, the right setter alone on the
    // bidder's 3-0 (with the 0-0), every trump played; the bidder leads the 4-4.
    let plays = [
        "2-0", "1-0", "4-0", "5-0", "2-2", "5-2", "6-2", "2-1", "6-5", "6-6", "6-1", "6-4", "3-0", "5-1", "4-3", "0-0",
        "1-1", "6-0", "3-1", "5-4", "4-4",
    ];
    assert_eq!(fires(TRUMP_SET, &position("P0", O1, &["6-3", "5-5"], &plays)), None);
}

#[test]
fn s0073c_the_feed_waits_on_a_trump_still_out() {
    // Ours (pp. 50, 54): deuces, the trump set of the 6-6 feed above, but the
    // right setter has since shown out of trump: no feed.
    let hand = ["6-4", "6-0", "5-5", "3-1", "1-0"];
    let plays = ["2-2", "6-1", "5-1", "2-0", "2-1", "4-3", "3-3", "1-1", "6-6"];
    assert_eq!(fires(TRUMP_SET, &position("P2", O1, &hand, &plays)), None);
}

// -- dead-double (new)

#[test]
fn throw_the_dead_double_s0131e_s0136f_s0137c_s0146d_s0151a_s0151g_s0148d() {
    for (id, plays, role, dead) in [
        ("hand-28", 23, O2, "1-1"),
        ("hand-30", 15, O2, "3-3"),
        ("hand-30", 19, O2, "4-4"),
        ("hand-34", 3, O2, "0-0"),
        ("hand-35", 11, O2, "1-1"),
        ("hand-35", 22, P, "3-3"),
        ("hand-34", 14, P, "1-1"),
    ] {
        assert!(allows(DEAD_DOUBLE, &book_at(id, plays, role), "throw-a-dead-double").contains(tile(dead)), "{id}");
    }
}

#[test]
fn s0147a_the_decoy_the_rule_cannot_see() {
    // Hand 34 trick 2: the partner throws the 6-2 and keeps the dead 1-1 for
    // a while, a decoy for an opponent's reading (GAP likelihood). The rule
    // throws the 1-1 at once.
    let at = book_at("hand-34", 6, P);
    assert_eq!(allows(DEAD_DOUBLE, &at, "throw-a-dead-double"), tiles(&["1-1"]));
}

/// The pour, s250001 deal 7003 to ply 15: fours; the bidder leads the 2-0 to
/// trick 4 and the right setter, void in deuces, is to play.
const RUFF_7003: [&str; 15] =
    ["2-2", "5-2", "3-2", "2-1", "6-1", "6-5", "6-6", "6-3", "5-5", "5-1", "4-1", "5-0", "2-0", "1-1", "6-0"];

#[test]
fn s0091b_a_throw_is_on_a_trick_you_do_not_take() {
    // With the 4-3 he takes the trick by trumping; the 0-0 is dead, but a
    // ruff is not a throw.
    assert_eq!(fires(DEAD_DOUBLE, &position("P4", O2, &["6-4", "4-3", "4-2", "0-0"], &RUFF_7003)), None);
    // Ours: the same trick with no trump to take it; the 0-0 goes.
    let at = position("P4", O2, &["5-3", "3-1", "3-0", "0-0"], &RUFF_7003);
    assert_eq!(allows(DEAD_DOUBLE, &at, "throw-a-dead-double"), tiles(&["0-0"]));
}

#[test]
fn the_book_throws_no_dead_ten() {
    // The pour, s200001 deal 25254 ply 19: treys; the 5-4, 5-2, 5-1 and 5-0
    // are played and the 5-3 is trump, so the right setter's 5-5 is dead. He
    // cannot follow the fours or take the trick; the book throws no count.
    let plays = [
        "2-0", "2-2", "4-2", "2-1", "1-0", "5-4", "5-1", "6-1", "3-3", "3-0", "3-1", "3-2", "1-1", "5-0", "5-2", "4-4",
        "4-1", "4-0", "4-3",
    ];
    assert_eq!(fires(DEAD_DOUBLE, &position("P3", O2, &["6-5", "6-0", "5-5"], &plays)), None);
}

// -- helper-stretch (new)

#[test]
fn s0105b_win_with_the_lowest_sure_tile() {
    // Hand 23 trick 2: the left setter leads the 5-3; the bidder, last, wins
    // with the 5-4 and keeps the 6-5.
    let at = hand("hand-23", &["1-1", "5-5", "4-4", "3-1", "5-3", "5-1", "5-0"]);
    assert_eq!(allows(HELPER_STRETCH, &at, "win-with-the-lowest-sure-tile"), tiles(&["5-4"]));
}

// -- pounce: the trick the setters may take is one the led suit can lose

#[test]
fn s0135f_s0139d_no_pounce_on_the_partners_double() {
    // Hands 30 and 31: the bidder's partner leads his 0-0, the top blank; the
    // right setter, void, throws the 5-1 and keeps the 5-5, though his partner
    // has not shown out of trump. A ruff is no trick to pounce on.
    for (id, plays) in [("hand-30", 5), ("hand-31", 9)] {
        let at = book_at(id, plays, O2);
        assert_eq!(fires(POUNCE, &at), None, "{id}");
        assert_eq!(allows(SHED, &at, "then-the-guard"), tiles(&["5-1"]), "{id}");
    }
}

#[test]
fn s0133b_the_left_setter_wins_with_the_ten() {
    // Hand 29 trick 1: the bidder leads the low 4-0; the left setter plays
    // the 6-4 at once for the trick and 11.
    assert_eq!(allows(POUNCE, &book_at("hand-29", 1, O1), "play-a-ten-now"), tiles(&["6-4"]));
}

// -- shed-for-a-void: the throwaways

#[test]
fn the_setters_throwaways_s0133g_s0136b_s0136d_s0139c_s0142j_s0144f_s0144h() {
    for (id, plays, role, thrown) in [
        ("hand-29", 6, O2, "6-0"),
        ("hand-30", 11, O1, "2-1"),
        ("hand-30", 13, O1, "4-2"),
        ("hand-31", 7, O2, "4-0"),
        ("hand-32", 7, O2, "5-3"),
        ("hand-33", 7, O2, "3-0"),
        ("hand-33", 9, O1, "5-1"),
    ] {
        let allowed = allows(SHED, &book_at(id, plays, role), "throw-what-you-need-not-keep");
        assert!(allowed.contains(tile(thrown)), "{id} after {plays}: {allowed:?}");
    }
    // Hand 31 trick 2: the 5-1 guards the 4-1, so the 4-0 goes, alone.
    assert_eq!(allows(SHED, &book_at("hand-31", 7, O2), "throw-what-you-need-not-keep"), tiles(&["4-0"]));
    // Hand 33 trick 2: the right setter keeps his 0-0 and his count.
    let allowed = allows(SHED, &book_at("hand-33", 7, O2), "throw-what-you-need-not-keep");
    assert_eq!(allowed, tiles(&["3-0", "1-0", "3-1"]));
}

#[test]
fn s0130b_with_only_doubles_and_count_besides_the_guard_goes() {
    // Hand 28 trick 2: the left setter, void in trump, holds the 5-1, the
    // 3-2, the 5-5, the 4-1, the 3-3 and the 4-4; the 5-1 guards the 4-1 but
    // is his one tile neither count nor double.
    assert_eq!(allows(SHED, &book_at("hand-28", 5, O1), "then-the-guard"), tiles(&["5-1"]));
}

#[test]
fn s0140a_the_left_setter_throws_the_5_4_under_his_partners_trick() {
    // Hand 31 trick 4: two trumps in hand, void in treys, his partner taking
    // the trick with the 3-3: the 5-4, his one plain tile (and the void in
    // fours and fives the book wants). He does not trump his partner's trick.
    let at = book_at("hand-31", 15, O1);
    assert_eq!(allows(SHED, &at, "throw-what-you-need-not-keep"), tiles(&["5-4"]));
    assert_eq!(fires(TAKE, &at), None);
}

#[test]
fn s0130c_the_book_against_its_void_guideline() {
    // Hand 28 trick 2: on the bidder's trump lead the right setter, void in
    // trump, throws the 2-1; the 6-3 is his only six and only trey. The book
    // gives no reason; the rule voids, as the book's guideline says.
    let at = book_at("hand-28", 7, O2);
    assert_eq!(allows(SHED, &at, "throw-a-lone-tile"), tiles(&["6-3"]));
}

// -- follow-low (new)

#[test]
fn follow_with_the_lowest_trump_s0129e_s0133c_s0136e_s0138c_s0142e_s0144c_s0146c() {
    for (id, plays, role, low) in [
        ("hand-28", 2, P, "1-0"),
        ("hand-29", 2, P, "4-2"),
        ("hand-30", 14, P, "6-0"),
        ("hand-31", 1, O1, "2-1"),
        ("hand-32", 1, O1, "2-0"),
        ("hand-33", 2, P, "4-3"),
        ("hand-34", 1, O1, "5-1"),
    ] {
        assert_eq!(allows(FOLLOW_LOW, &book_at(id, plays, role), "the-lowest-trump"), tiles(&[low]), "{id}");
    }
}

#[test]
fn s0133f_the_partner_follows_low_with_no_count_to_protect() {
    // Hand 29 trick 2: the left setter's 2-2; the partner, holding the 2-1 and
    // the 5-2, plays the lower.
    assert_eq!(allows(FOLLOW_LOW, &book_at("hand-29", 5, P), "the-partner-follows-low"), tiles(&["2-1"]));
}

#[test]
fn follow_low_leaves_count_to_the_count_rules() {
    // Hand 33 trick 1: the left setter's trumps are the 4-2 and the 4-1, a
    // five; protect-the-count keeps it (S-0144-b), and this rule is silent.
    let at = book_at("hand-33", 1, O1);
    assert_eq!(fires(FOLLOW_LOW, &at), None);
    assert_eq!(allows(PROTECT, &at, "follow-the-master-without-count"), tiles(&["4-2"]));
}

// -- take-the-trick (new)

#[test]
fn take_the_trick_s0134b_s0140c_s0143a() {
    // Hand 29 trick 4: void in sixes, the bidder trumps the 6-6 with the 4-1,
    // the lower of the 4-1 and the 4-4.
    assert_eq!(allows(TAKE, &book_at("hand-29", 14, B), "trump-it"), tiles(&["4-1"]));
    // Hand 31 trick 5: the left setter trumps the 6-4 with the 4-2 and sets.
    assert_eq!(allows(TAKE, &book_at("hand-31", 18, O1), "trump-it"), tiles(&["4-2"]));
    // Hand 32 trick 3: last, the right setter takes the off trick with the
    // 3-3, the one trey above the partner's 6-3, and sets.
    assert_eq!(allows(TAKE, &book_at("hand-32", 11, O2), "take-it-last"), tiles(&["3-3"]));
    // The same play: Hand 28 trick 3 (the 6-3), Hand 33 trick 3 (the 2-2),
    // Hand 35 trick 7 (the 6-5, setting the 84).
    for (id, plays, take) in [("hand-28", 11, "6-3"), ("hand-33", 11, "2-2"), ("hand-35", 27, "6-5")] {
        assert_eq!(allows(TAKE, &book_at(id, plays, O2), "take-it-last"), tiles(&[take]), "{id}");
    }
}

#[test]
fn s0051b_hand_16_trumps_the_0_0_with_the_3_2() {
    // Hand 16 trick 3 (ours: trick 1's 3-0 and 4-3): the right setter leads
    // the 0-0; the bidder, out of blanks, trumps with the 3-2 and keeps the
    // 3-3 to draw the last trump.
    let at = hand("hand-16", &["3-1", "3-0", "4-3", "6-3", "1-1", "1-0", "4-1", "5-1", "0-0"]);
    assert_eq!(allows(TAKE, &at, "trump-it"), tiles(&["3-2"]));
}

#[test]
fn s0126b_hand_27_trumps_the_5_5_with_the_6_4() {
    // Hand 27 trick 2 (ours: the setters' tiles but the 6-6 and the 5-5): the
    // bidder, last and out of fives, trumps with the 6-4, the lower trump.
    let at = hand("hand-27", &["6-2", "6-6", "6-1", "6-0", "5-5", "5-3", "5-1"]);
    assert_eq!(allows(TAKE, &at, "take-it-last"), tiles(&["6-4"]));
}

#[test]
fn s0074b_the_trump_holder_last_trumps_only_count() {
    // Ours (p. 51): deuces; only the right setter followed the bidder's 2-2.
    // On the bidder's 4-4 with no count on it he holds his trumps back; with
    // the partner's 4-1 on it he trumps, low.
    let hand = ["5-2", "4-2", "1-1", "1-0", "6-6", "3-1"];
    let alone = ["2-2", "6-1", "5-1", "2-0"];
    let at = position("P2", O2, &hand, &[&alone[..], &["4-4", "4-3", "4-0"]].concat());
    assert_eq!(fires(TAKE, &at), None);
    let at = position("P2", O2, &hand, &[&alone[..], &["4-4", "4-3", "4-1"]].concat());
    assert_eq!(allows(TAKE, &at, "take-it-last"), tiles(&["4-2"]));
}

#[test]
fn s0131b_the_bidder_trumps_his_partners_trick_and_no_rule_says_so() {
    // Hand 28 trick 5: the bidder, last, trumps his partner's 6-4 with the
    // 5-0. The trick was his side's; the rule takes only the other side's.
    assert_eq!(fires(TAKE, &book_at("hand-28", 19, B)), None);
}

// -- follow-with-the-off (new)

#[test]
fn s0135g_s0040a_the_bidder_follows_with_the_off_his_double_does_not_protect() {
    // Hand 30 trick 2: the partner's 0-0; the bidder plays the 1-0 and keeps
    // the 2-0 under his 2-2.
    assert_eq!(allows(FOLLOW_OFF, &book_at("hand-30", 6, B), "play-the-bare-off"), tiles(&["1-0"]));
    // Hand 16 trick 2 (ours: trick 1's 3-0 and 4-3): the setter's 1-1; the
    // bidder plays the 1-0 and keeps the 2-1 under his 2-2.
    let at = hand("hand-16", &["3-1", "3-0", "4-3", "6-3", "1-1"]);
    assert_eq!(allows(FOLLOW_OFF, &at, "play-the-bare-off"), tiles(&["1-0"]));
    // Hand 33 trick 4: the bidder can take the trick with the 5-5; no off.
    assert_eq!(fires(FOLLOW_OFF, &book_at("hand-33", 13, B)), None);
}

// -- defend-the-84 (new)

#[test]
fn the_84_throws_s0147b_s0147c_s0148a_s0148c_s0148e_s0150h_s0151d_s0151f() {
    for (id, plays, role, thrown) in [
        ("hand-34", 7, O2, "6-0"),
        ("hand-34", 9, O1, "2-0"),
        ("hand-34", 11, O2, "2-1"),
        ("hand-34", 13, O1, "3-0"),
        ("hand-34", 15, O2, "3-2"),
        ("hand-35", 9, O1, "6-0"),
        ("hand-35", 15, O2, "2-0"),
        ("hand-35", 21, O1, "3-0"),
    ] {
        let allowed = allows(DEFEND_84, &book_at(id, plays, role), "throw-what-wins-nothing");
        assert!(allowed.contains(tile(thrown)), "{id} after {plays}: {allowed:?}");
    }
    // Hand 34 trick 4: the right setter keeps the 6-1 and the 3-1 for the
    // 1-0, and the 6-3 that saves the 6-1 from the partner's 6-6: the 3-2 goes.
    assert_eq!(allows(DEFEND_84, &book_at("hand-34", 15, O2), "throw-what-wins-nothing"), tiles(&["3-2"]));
    // Hand 35 trick 4: the right setter's 5-1 guards his 6-5; the 2-0 goes.
    assert_eq!(allows(DEFEND_84, &book_at("hand-35", 15, O2), "throw-what-wins-nothing"), tiles(&["2-0"]));
}

#[test]
fn s0148f_a_double_before_a_pair() {
    // Hand 34 trick 5: the left setter holds the 3-3 (the 3-1 still out) and
    // the 4-3 and the 6-4 (the 4-2 and the 4-0 out); nothing else. He gives up
    // the fours and keeps the double.
    assert_eq!(allows(DEFEND_84, &book_at("hand-34", 17, O1), "a-pair-before-a-double"), tiles(&["4-3", "6-4"]));
}

#[test]
fn s0151b_the_book_against_its_pair_rule() {
    // Hand 35 trick 4: the left setter holds the 5-3, the 2-2, the 3-0 and
    // the 5-0. The partner threw the 3-2 on trick 3, the last trey his pair
    // could take; the 5-3 and the 5-0 still take the 5-2 or the 5-1. The book
    // throws the 5-0; the rule throws the 3-0, which guards the 5-3 against
    // the 3-3.
    let at = book_at("hand-35", 13, O1);
    assert_eq!(allows(DEFEND_84, &at, "then-a-protector"), tiles(&["3-0"]));
}

#[test]
fn s0111f_at_no_trump_the_setter_keeps_his_count_too() {
    // Ours (p. 82): Hand 25 at no trump; the bidder leads the 4-4 and the left
    // setter, out of fours, keeps his 6-6 (the 6-4, 6-2 and 6-1 out) and his
    // count, the 3-2; with three treys for the 3-1 he may spare one.
    let at = position("NT", O1, &["6-6", "6-0", "3-2", "6-3", "3-0", "1-0", "6-5"], &["4-4"]);
    let allowed = allows(DEFEND_84, &at, "throw-what-wins-nothing");
    for kept in ["6-6", "3-2"] {
        assert!(!allowed.contains(tile(kept)), "{kept}: {allowed:?}");
    }
    assert!(allowed.contains(tile("1-0")), "{allowed:?}");
}

// -- bidders-plan: the plan's other narrated leads

#[test]
fn s0022a_s0144e_plan_for_the_setters_holding_the_missing_trumps() {
    // Hand 33 trick 2: the right setter has shown out of fours, two fours are
    // out, and the partner may hold both (5/22, odds/c12-hand33-left-holds-
    // trump); the bidder draws as if a setter holds them.
    assert_eq!(allows(BIDDERS_PLAN, &book_at("hand-33", 4, B), "draw-the-trumps"), tiles(&["6-4"]));
}

#[test]
fn s0048a_hand_15_either_off_calls_no_count() {
    // Hand 15, line A, trick 3 (ours: trick 1's trumps): with fours trump the
    // 6-0 and the 1-0 call for no count; either may go, and the book leads the
    // 6-0.
    let at = hand("hand-15", &["4-4", "4-2", "4-0", "5-4", "6-4", "2-0", "2-1", "6-1"]);
    assert_eq!(allows(BIDDERS_PLAN, &at, "dump-a-bare-off-calling-no-count"), tiles(&["6-0", "1-0"]));
}

#[test]
fn s0107d_hand_24_leads_the_5_1_to_draw_the_6_5() {
    // Hand 24 trick 3, doubles trump, on the archive's reading of trick 2
    // (the 3-3, not the 1-1): trumps drawn, the bidder dumps an off at once,
    // and the 5-1 is one the rule allows; it draws the 6-5, which made the
    // 6-4 his worst off.
    let at = hand("hand-24", &["6-6", "0-0", "4-4", "5-3", "5-5", "3-3", "5-2", "1-0"]);
    let found = fires(BIDDERS_PLAN, &at);
    assert!(found.as_ref().is_some_and(|(_, allowed)| allowed.contains(tile("5-1"))), "{found:?}");
}

#[test]
fn s0052b_hand_16_the_2_1_walks_last() {
    // Hand 16 (ours where the book names no tile): after the 5-5 and the 2-2
    // draw the other deuces, the 2-1 cannot lose and goes last.
    let plays = [
        "3-1", "3-0", "4-3", "6-3", "1-1", "1-0", "4-1", "5-1", "0-0", "3-2", "2-0", "5-0", "3-3", "5-3", "6-0", "6-1",
        "5-5", "5-4", "5-2", "6-5", "2-2", "4-2", "6-2", "4-4",
    ];
    assert_eq!(allows(BIDDERS_PLAN, &hand("hand-16", &plays), "take-the-rest"), tiles(&["2-1"]));
}

#[test]
fn s0145a_hand_33_cashes_the_1_1_and_the_last_deuce() {
    // Hand 33 trick 6: the 1-1, a winner, before the 2-1; trick 7, on the
    // archive's reading of trick 6 (the right setter must follow with the
    // 3-1, not throw the 0-0), the 2-1 is the last deuce and walks.
    assert_eq!(allows(BIDDERS_PLAN, &book_at("hand-33", 20, B), "cash-the-other-winners"), tiles(&["1-1"]));
    let plays = then(&prose("hand-33")[..20], &["1-1", "6-3", "6-2", "3-1"]);
    let at = dealt("hand-33", B, &plays);
    assert_eq!(allows(BIDDERS_PLAN, &at, "take-the-rest"), tiles(&["2-1"]));
}

#[test]
fn s0146e_s0148i_hand_34_runs_trumps_then_the_doubles() {
    // Once the 6-5 has fallen the bidder leads his trumps and lets the
    // defenders guess; the 4-4 goes next-to-last, ahead of the 4-2.
    assert_eq!(allows(BIDDERS_PLAN, &book_at("hand-34", 4, B), "draw-the-trumps"), tiles(&["5-4"]));
    assert!(allows(BIDDERS_PLAN, &book_at("hand-34", 8, B), "cash-the-other-winners").contains(tile("5-2")));
    assert!(allows(BIDDERS_PLAN, &book_at("hand-34", 12, B), "cash-the-other-winners").contains(tile("5-0")));
    assert_eq!(allows(BIDDERS_PLAN, &book_at("hand-34", 16, B), "cash-the-other-winners"), tiles(&["2-2"]));
    assert_eq!(allows(BIDDERS_PLAN, &book_at("hand-34", 20, B), "the-double-of-the-offs-suit"), tiles(&["4-4"]));
}

// -- narrated plays the earlier rules already read

#[test]
fn s0143b_after_the_set_lead_what_calls_for_count() {
    // Hand 32 trick 4: the setters have set the 36; the right setter leads
    // the 5-4, which calls for the 5-5, so the bidder must trump rather than
    // throw another off.
    let at = dealt("hand-32", O2, &prose("hand-32")[..12]);
    assert!(fires(SETTER_LEADS, &at).is_some_and(|(_, allowed)| allowed.contains(tile("5-4"))));
}

#[test]
fn s0131d_either_dead_double_may_go() {
    // Hand 28 trick 6: no four or trey is left outside the left setter's
    // hand; both his doubles are dead, and the book says either may go.
    assert_eq!(allows(DEAD_DOUBLE, &book_at("hand-28", 21, O1), "throw-a-dead-double"), tiles(&["3-3", "4-4"]));
}

#[test]
fn s0151h_the_low_six_keeps_the_6_5() {
    // Hand 35 trick 6: on the 6-6 the right setter plays the 6-1 and keeps
    // the 6-5, now the top six, for the bidder's last tile.
    assert_eq!(allows(LOW_FROM_A_PAIR, &book_at("hand-35", 23, O2), "follow-low"), tiles(&["6-1"]));
}

#[test]
fn s0150e_no_count_on_the_bidders_top_trump() {
    // Hand 35 trick 1: the left setter has thrown; the right setter keeps the
    // 4-1 back and follows with the 4-2.
    assert_eq!(allows(POUNCE, &book_at("hand-35", 3, O2), "no-count-on-a-lost-trick"), tiles(&["4-2"]));
}
