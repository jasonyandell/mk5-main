//! The hand's public record in the frame, the predicates that read it, and
//! the `none` clause.
mod common;

use common::*;
use silas::*;

/// A complete deal: S0's hand first, then S1, S2, S3.
const DEAL: [[&str; 7]; 4] = [
    ["0-0", "1-1", "2-0", "3-2", "5-5", "6-4", "6-6"],
    ["1-0", "2-2", "3-0", "4-1", "5-1", "6-0", "6-5"],
    ["2-1", "3-1", "4-0", "4-4", "5-2", "6-1", "6-3"],
    ["3-3", "4-2", "4-3", "5-0", "5-3", "5-4", "6-2"],
];

/// The deal played lowest-legal for `n` plays from `declarer`'s lead, under
/// `decl`: the record's plays.
fn lowest_plays(physics: &Physics, declarer: Seat, n: usize) -> Vec<(Seat, Domino)> {
    let mut hands = DEAL.map(|hand| tiles(&hand));
    let mut leader = declarer;
    let mut trick: Vec<Domino> = Vec::new();
    let mut plays = Vec::new();
    while plays.len() < n {
        let seat = leader.plus(trick.len());
        let tile = physics.legal_plays(hands[seat.index()], trick.first().copied()).iter().next().unwrap();
        hands[seat.index()].remove(tile);
        plays.push((seat, tile));
        trick.push(tile);
        if let Ok(full) = <[Domino; 4]>::try_from(trick.as_slice()) {
            leader = physics.trick_winner(leader, full);
            trick.clear();
        }
    }
    plays
}

/// `viewer`'s frame after `n` lowest-legal plays, `declarer` bidding `bid`.
fn recorded(decl: &str, viewer: Seat, declarer: Seat, bid: u32, n: usize) -> Frame {
    let physics = physics(decl);
    let plays = lowest_plays(&physics, declarer, n);
    let hand = plays
        .iter()
        .filter(|(s, _)| *s == viewer)
        .fold(tiles(&DEAL[viewer.index()]), |h, (_, d)| h.difference(DominoSet::single(*d)));
    framed_history(&contract(&physics, declarer, bid), viewer, hand, &plays).unwrap()
}

fn holds(frame: &Frame, query: &str) -> bool {
    !compile(query).evaluate_viewer(frame, &mut budget()).unwrap().is_empty()
}

/// Whether `atom` holds, and whether its negation does.
fn both(frame: &Frame, atom: &str) -> (bool, bool) {
    (
        holds(frame, &format!("(query (roles) (out) (pattern {atom}))")),
        holds(frame, &format!("(query (roles) (out) (pattern (not {atom})))")),
    )
}

#[test]
fn a_record_frame_derives_the_residue_the_steps_produce() {
    let physics = physics("P6");
    let plays = lowest_plays(&physics, Seat::S0, 11);
    let contract = contract(&physics, Seat::S0, 30);
    let mut stepped = framed_history(&contract.clone(), Seat::S0, tiles(&DEAL[0]), &[]).unwrap();
    for (n, (actor, tile)) in plays.iter().copied().enumerate() {
        let class = match stepped.led_context() {
            None => PlayClass::Lead,
            Some(q) if physics.follows(tile, q) => PlayClass::Follow(q),
            Some(q) => PlayClass::Slough(q),
        };
        stepped = step_frame(&stepped, ObservedPlay { actor, tile, class }, &mut budget()).unwrap();
        let hand = stepped.kernel().viewer_hand();
        let direct = framed_history(&contract.clone(), Seat::S0, hand, &plays[..=n]).unwrap();
        assert_eq!(stepped, direct);
        // The same frame from the dealt hand and the tiles alone.
        let tiles_only = plays[..=n].iter().map(|(_, d)| *d).collect();
        assert_eq!(framed(&contract.clone(), Seat::S0, tiles(&DEAL[0]), tiles_only).unwrap(), direct);
    }
    assert_eq!(stepped.history(), plays.as_slice());
    assert_eq!(stepped.trick_number(), Some(3));
    assert_eq!(stepped.prefix().len(), 3);
    // Every closed trick's points are banked: a point a trick plus its count.
    let closed: u32 = stepped
        .tricks()
        .filter(|t| t.len() == 4)
        .map(|t| 1 + t.iter().map(|(_, d)| physics.count(*d)).sum::<u32>())
        .sum();
    assert_eq!(stepped.banked().iter().sum::<u32>(), closed);
}

#[test]
fn incoherent_records_are_refused() {
    let physics = physics("P6");
    let plays = lowest_plays(&physics, Seat::S0, 8);
    let hand = |plays: &[(Seat, Domino)]| {
        plays
            .iter()
            .filter(|(s, _)| *s == Seat::S0)
            .fold(tiles(&DEAL[0]), |h, (_, d)| h.difference(DominoSet::single(*d)))
    };
    let at = |plays: Vec<(Seat, Domino)>, bid| {
        let contract = Contract::new(Seat::S0, Bid::number(bid)?, physics.clone())?;
        framed_history(&contract, Seat::S0, hand(&plays), &plays)
    };
    assert!(at(plays.clone(), 42).is_ok());
    assert!(at(plays.clone(), 43).is_err(), "a bid above the hand's 42 points");
    assert!(at(plays.clone(), 84).is_ok(), "two marks");
    let mut wrong = plays.clone();
    wrong.swap(1, 2);
    assert!(at(wrong, 30).is_err(), "out of turn");
    let mut wrong = plays.clone();
    wrong[5].1 = wrong[1].1;
    assert!(at(wrong, 30).is_err(), "a repeated tile");
    // The declarer leads the first trick.
    let s1 = contract(&physics, Seat::S1, 30);
    assert!(framed_history(&s1, Seat::S0, hand(&plays), &plays).is_err());
    // The viewer's hand must be what its dealt hand left.
    assert!(framed_history(&contract(&physics, Seat::S0, 30), Seat::S0, tiles(&DEAL[0]), &plays).is_err());
    // Blanks trump: the other seats' plays are checked against their shown
    // voids only, the viewer's against its hand. S1 shows void in trump, then
    // follows trump.
    let p0 = common::physics("P0");
    let s0 = contract(&p0, Seat::S0, 30);
    let reneges = vec![
        (Seat::S0, tile("0-0")),
        (Seat::S1, tile("2-2")),
        (Seat::S2, tile("4-0")),
        (Seat::S3, tile("5-0")),
        (Seat::S0, tile("2-0")),
        (Seat::S1, tile("1-0")),
    ];
    assert!(framed_history(&s0.clone(), Seat::S0, hand(&reneges[..5]), &reneges[..5]).is_ok());
    assert!(framed_history(&s0.clone(), Seat::S0, hand(&reneges), &reneges).is_err());
    // A seat that has shown void in a context plays none of its tiles again,
    // whether it follows, sloughs or leads: S2 trumps S1's six and then leads
    // a six.
    let s1 = contract(&p0, Seat::S1, 30);
    let led_after_void = vec![
        (Seat::S1, tile("6-5")),
        (Seat::S2, tile("4-0")),
        (Seat::S3, tile("6-2")),
        (Seat::S0, tile("6-4")),
        (Seat::S2, tile("6-1")),
    ];
    let at = |plays: &[(Seat, Domino)]| framed_history(&s1.clone(), Seat::S0, hand(plays), plays);
    assert!(at(&led_after_void[..4]).is_ok());
    assert!(at(&led_after_void).is_err());
    // The viewer may not slough a trump lead while holding trump.
    let mut viewer =
        vec![(Seat::S1, tile("1-0")), (Seat::S2, tile("4-0")), (Seat::S3, tile("5-0")), (Seat::S0, tile("1-1"))];
    assert!(framed_history(&s1.clone(), Seat::S0, hand(&viewer), &viewer).is_err());
    viewer[3].1 = tile("2-0");
    assert!(framed_history(&s1, Seat::S0, hand(&viewer), &viewer).is_ok());
}

#[test]
fn the_contract_and_score_read_as_the_book_reads() {
    let frame = recorded("P6", Seat::S2, Seat::S2, 34, 12);
    let [t0, t1] = frame.banked();
    assert!(t0 > 0 && t1 > 0, "both sides have banked: {t0} {t1}");
    assert_eq!(both(&frame, "(declarer S2)"), (true, false));
    assert_eq!(both(&frame, "(declarer S0)"), (false, true));
    assert_eq!(both(&frame, "(bid 34)"), (true, false));
    assert_eq!(both(&frame, &format!("(banked S0 {t0})")), (true, false));
    assert_eq!(both(&frame, &format!("(banked S1 {t1})")), (true, false));
    assert_eq!(both(&frame, &format!("(to-make {})", 34u32.saturating_sub(t0))), (true, false));
    assert_eq!(both(&frame, &format!("(to-set {})", (43 - 34u32).saturating_sub(t1))), (true, false));
    // The declarer, relationally: the viewer's partner's partner.
    assert!(holds(&frame, "(query (roles (chair b) (chair p)) (out) (pattern (declarer b) (partner b p) (viewer b)))"));
    // A contract made reads zero to make; set, zero to set. A bid is 30 or
    // more, so the make is found further into the hand.
    let made = (12..=28)
        .flat_map(|n| Seat::ALL.map(|d| (d, n)))
        .map(|(d, n)| recorded("P6", d, d, 30, n))
        .find(|f| f.banked()[f.contract().declarer().team().index()] >= 30)
        .expect("some side banks 30 under lowest-legal play");
    assert_eq!(both(&made, "(to-make 0)"), (true, false));
    let set = (12..=28)
        .map(|n| recorded("P6", Seat::S2, Seat::S2, 41, n))
        .find(|f| f.banked()[Team::T1.index()] >= 2)
        .expect("the setters bank two points");
    assert_eq!(both(&set, "(to-set 0)"), (true, false));
}

#[test]
fn history_names_earlier_tricks_by_their_lead() {
    let frame = recorded("P6", Seat::S1, Seat::S3, 30, 9);
    let plays = frame.history().to_vec();
    assert_eq!(both(&frame, "(trick-number 3)"), (true, false));
    assert_eq!(both(&frame, "(trick-number 2)"), (false, true));
    let (first_leader, first_lead) = plays[0];
    assert_eq!(first_leader, Seat::S3, "the declarer leads the first trick");
    assert_eq!(both(&frame, &format!("(led {first_leader} {first_lead})")), (true, false));
    assert_eq!(both(&frame, &format!("(trick-of {first_lead} 1)")), (true, false));
    let (second_leader, second_lead) = plays[4];
    for (i, (seat, tile)) in plays.iter().enumerate() {
        let trick = i / 4 + 1;
        let lead = plays[(i / 4) * 4].1;
        assert_eq!(both(&frame, &format!("(played-to {lead} {seat} {tile})")), (true, false));
        assert_eq!(both(&frame, &format!("(trick-of {tile} {trick})")), (true, false));
        if trick != 2 {
            assert_eq!(both(&frame, &format!("(played-to {second_lead} {seat} {tile})")), (false, true));
        }
    }
    // The open trick's lead is a lead of the record, as in the frame.
    let (open_leader, open_lead) = plays[8];
    assert_eq!(open_leader, frame.leader());
    assert!(holds(
        &frame,
        &format!(
            "(query (roles) (out) (pattern (led {open_leader} {open_lead}) (trick-play {open_leader} {open_lead})))"
        )
    ));
    // Who led which trick, relationally.
    let leaders = compile("(query (roles (chair c) (domino d)) (out c d) (pattern (led c d)))")
        .evaluate_viewer(&frame, &mut budget())
        .unwrap();
    let expected = [(first_leader, first_lead), (second_leader, second_lead), (open_leader, open_lead)];
    assert_eq!(leaders, expected.iter().map(|(c, d)| Answer(vec![Value::Chair(*c), Value::Domino(*d)])).collect());
    // A tile not yet played belongs to no trick.
    let unplayed = frame.kernel().viewer_hand().iter().next().unwrap();
    assert_eq!(both(&frame, &format!("(trick-of {unplayed} 4)")), (false, true));
    assert_eq!(both(&frame, &format!("(trick-of {unplayed} 0)")), (false, true));
}

#[test]
fn every_frame_has_its_record_and_the_hand_ends_undefined() {
    // Every frame is built from a record, so its predicates are defined: an
    // atom or its negation holds.
    let frame = small_frame(physics("P6"));
    for atom in [
        "(declarer S0)",
        "(bid 30)",
        "(stake 1)",
        "(banked S0 0)",
        "(to-make 30)",
        "(to-set 13)",
        "(trick-number 6)",
        "(led S0 6-6)",
        "(played-to 6-6 S0 6-6)",
        "(trick-of 6-6 1)",
        "(shown-void S0 q6)",
        "(shown-void S1 q6)",
    ] {
        let (yes, no) = both(&frame, atom);
        assert!(yes != no, "{atom}");
    }
    // Once all seven tricks have closed, no trick is being played: the trick
    // number is undefined, neither n nor not n, and no seat is next.
    let done = recorded("P6", Seat::S0, Seat::S0, 30, 28);
    assert_eq!((done.trick_number(), done.next_actor()), (None, None));
    for n in [7, 8] {
        assert_eq!(both(&done, &format!("(trick-number {n})")), (false, false), "{n}");
    }
    assert_eq!(both(&done, "(trick-number 1)"), (false, false));
    assert!(!holds(&done, "(query (roles (chair c)) (out) (pattern (next-actor c)))"));
}

#[test]
fn the_contract_carries_kind_stake_and_declaration() {
    let p = physics("P5");
    let points = contract(&p, Seat::S1, 35);
    let one_mark = contract(&p, Seat::S1, 42);
    let two_marks = contract(&p, Seat::S1, 84);
    assert_eq!((points.bid(), one_mark.bid(), two_marks.bid()), (Bid::Points(35), Bid::Marks(1), Bid::Marks(2)));
    assert_eq!((points.threshold(), points.stake()), (35, 1));
    assert_eq!((one_mark.threshold(), one_mark.stake()), (42, 1));
    assert_eq!((two_marks.threshold(), two_marks.stake()), (42, 2));
    assert_eq!(two_marks.declaration(), &p);
    // Made, the stake goes to the declaring side; set, to the other.
    assert_eq!((two_marks.award(42), two_marks.award(41)), ((Team::T1, 2), (Team::T0, 2)));
    assert_eq!((points.award(35), points.award(34)), ((Team::T1, 1), (Team::T0, 1)));
    assert!(Bid::number(43).is_err() && Bid::number(85).is_err());
    assert_eq!("126".parse::<Bid>().unwrap(), Bid::Marks(3));
    assert!(Contract::new(Seat::S0, Bid::Points(42), p.clone()).is_err(), "every point is a marks bid");
    assert!(Contract::new(Seat::S0, Bid::Marks(0), p.clone()).is_err());
    // A rule can guard on the 84: the bid reads as the table says it.
    let plays = lowest_plays(&p, Seat::S1, 4);
    let hand = tiles(&DEAL[1]).difference(plays.iter().filter(|(s, _)| *s == Seat::S1).map(|(_, d)| *d).collect());
    let frame = framed_history(&two_marks, Seat::S1, hand, &plays).unwrap();
    assert_eq!(both(&frame, "(bid 84)"), (true, false));
    assert_eq!(both(&frame, "(bid 42)"), (false, true));
    assert_eq!(both(&frame, "(stake 2)"), (true, false));
    let [ours, theirs] = [frame.banked()[1], frame.banked()[0]];
    assert_eq!(both(&frame, &format!("(to-make {})", 42 - ours)), (true, false));
    assert_eq!(both(&frame, &format!("(to-set {})", 1u32.saturating_sub(theirs))), (true, false));
}

#[test]
fn shown_voids_are_the_kernels_and_imply_the_world_void() {
    for decl in ["P6", "P0", "DT", "NT"] {
        let frame = recorded(decl, Seat::S0, Seat::S0, 30, 17);
        for h in frame.kernel().hidden() {
            assert_eq!(h.voids, frame.shown_voids(h.seat), "{decl} {}", h.seat);
        }
        let shown = compile("(query (roles (chair c) (context q)) (out c q) (pattern (shown-void c q)))")
            .evaluate_viewer(&frame, &mut budget())
            .unwrap();
        let expected: Answers = Seat::ALL
            .into_iter()
            .flat_map(|c| frame.shown_voids(c).iter().map(move |q| Answer(vec![Value::Chair(c), Value::Context(q)])))
            .collect();
        assert_eq!(shown, expected, "{decl}");
        let belief = Belief::uniform(frame.clone(), 10_000_000).unwrap();
        let broken = compile(
            "(query (roles (chair c) (context q)) (out) (pattern (shown-void c q) (not (viewer c)) (not (void c q))))",
        );
        let p = exact(broken.summarize(&belief, &mut budget()).unwrap().event_probability());
        assert_eq!(p, r(0, 1), "{decl}: a shown void is a void in every world");
    }
}

#[test]
fn a_void_in_every_world_need_not_be_shown() {
    // S0 holds every six but the 6-5; once the 6-5 is played, no hidden seat
    // holds a six, yet only the seat that failed to follow has shown it.
    let p = physics("P0");
    let plays =
        vec![(Seat::S0, tile("6-6")), (Seat::S1, tile("6-5")), (Seat::S2, tile("2-2")), (Seat::S3, tile("3-3"))];
    let hand = tiles(&["6-4", "6-3", "6-2", "6-1", "6-0", "1-1"]);
    let frame = framed_history(&contract(&p, Seat::S0, 30), Seat::S0, hand, &plays).unwrap();
    assert_eq!(both(&frame, "(shown-void S2 q6)"), (true, false));
    assert_eq!(both(&frame, "(shown-void S1 q6)"), (false, true));
    let belief = Belief::uniform_resolving(frame, DominoSet::EMPTY, 10).unwrap();
    let void = compile("(query (roles) (out) (pattern (void S1 q6)))");
    assert_eq!(exact(void.summarize(&belief, &mut budget()).unwrap().event_probability()), r(1, 1));
}

#[test]
fn none_reads_no_other_and_matches_the_built_in_boss() {
    // `boss` is "live, follows, and no live tile beats it": with none, the
    // same relation is written out.
    for decl in ["P6", "P3", "DT", "NT"] {
        let frame = recorded(decl, Seat::S1, Seat::S0, 30, 10);
        let built_in = compile("(query (roles (domino d) (context q)) (out d q) (pattern (boss d q)))");
        let written = compile(
            "(query (roles (domino d) (context q)) (out d q)
               (pattern (live d) (in d q) (none (roles (domino e)) (live e) (beats e d q))))",
        );
        let a = built_in.evaluate_viewer(&frame, &mut budget()).unwrap();
        assert!(!a.is_empty());
        assert_eq!(a, written.evaluate_viewer(&frame, &mut budget()).unwrap(), "{decl}");
    }
    // A none's roles are not distinct from the roles around it: "no other"
    // is written out.
    let frame = recorded("P6", Seat::S0, Seat::S0, 30, 0);
    let ones =
        "(query (roles (domino d) (context q)) (out d) (pattern (own-legal d) (leads-context d q) (not (context q q*))
                  (none (roles (domino e)) (own-legal e) (in e q) (not (tile e d)))))";
    // S0 holds 0-0 1-1 2-0 3-2 5-5 6-4 6-6, sixes trump. The 1-1, 3-2 and 5-5
    // are alone in the suits they lead (ones, threes, fives); the 0-0 and 2-0
    // share the blanks. Without "not the same tile", every tile is its own
    // other, and none is alone.
    let alone = compile(ones).evaluate_viewer(&frame, &mut budget()).unwrap();
    assert_eq!(alone, ["1-1", "3-2", "5-5"].iter().map(|d| Answer(vec![Value::Domino(tile(d))])).collect());
    let not_other = ones.replace(" (not (tile e d))", "");
    assert!(compile(&not_other).evaluate_viewer(&frame, &mut budget()).unwrap().is_empty());
}

#[test]
fn none_is_three_valued_and_never_manufactures_a_fact() {
    // S0 is not to play: own-legal is undefined everywhere.
    let frame = recorded("P6", Seat::S0, Seat::S1, 30, 1);
    assert_ne!(frame.next_actor(), Some(Seat::S0));
    let exists = "(query (roles (domino d)) (out) (pattern (own-legal d) (double d)))";
    let none = "(query (roles) (out) (pattern (none (roles (domino d)) (own-legal d) (double d))))";
    assert!(!holds(&frame, exists), "undefined is not true");
    assert!(!holds(&frame, none), "nor is it false: none of an undefined body is undefined");
    // A body refuted by a defined clause is refuted, whatever the undefined one.
    let refuted = "(query (roles) (out) (pattern (none (roles (domino d)) (own-legal d) (double d) (not (double d)))))";
    assert!(holds(&frame, refuted));
    // Nested: a none inside a none reads both scopes. The doubles with no
    // other live tile of their suit that they fail to beat: every double.
    let nested = "(query (roles (domino d) (context q)) (out d)
                    (pattern (live d) (double d) (leads-context d q)
                      (none (roles (domino e)) (live e) (in e q) (not (tile e d))
                        (none (roles (domino f)) (tile f d) (beats f e q)))))";
    let frame = recorded("P0", Seat::S0, Seat::S0, 30, 0);
    let doubles: Answers =
        Domino::ALL.into_iter().filter(|d| d.is_double()).map(|d| Answer(vec![Value::Domino(d)])).collect();
    assert_eq!(compile(nested).evaluate_viewer(&frame, &mut budget()).unwrap(), doubles);
    let reversed = nested.replace("(beats f e q)", "(beats e f q)");
    assert!(compile(&reversed).evaluate_viewer(&frame, &mut budget()).unwrap().is_empty());
    let parsed: Query = nested.parse().unwrap();
    assert_eq!(parsed.to_string().parse::<Query>().unwrap(), parsed);
}

#[test]
fn none_is_checked_like_any_clause() {
    for s in [
        // A none may not redeclare a role in scope, nor join one with same.
        "(query (roles (domino d)) (out) (pattern (none (roles (domino d)) (live d))))",
        "(query (roles (domino d)) (out) (pattern (none (roles (domino e)) (same e d) (live e))))",
        "(query (roles (domino d)) (out) (pattern (none (roles (domino e) (domino e)) (live e))))",
        "(query (roles) (out) (pattern (none (roles (domino e)) (mystery e))))",
        "(query (roles) (out) (pattern (none (roles (domino e)) (live f))))",
        "(query (roles) (out) (pattern (none (roles (number n)) (bid n))))",
    ] {
        let refused = s.parse::<Query>().and_then(|q| q.compile(&Registry::standard()).map(drop));
        assert!(refused.is_err(), "{s}");
    }
    // Viewer-only exactly when the body is.
    let world = "(policy p (fallback lowest-legal) (rule r (select answer)
                   (guard (query (roles (domino d)) (out d) (pattern (own-legal d) (none (roles (chair c)) (holds c d)))))))";
    let p = physics("P6");
    assert!(world.parse::<PolicyProgram>().unwrap().compile(&Registry::standard(), &p).is_err());
    let viewer = world.replace("(holds c d)", "(shown-void c q6)");
    assert!(viewer.parse::<PolicyProgram>().unwrap().compile(&Registry::standard(), &p).is_ok());
    // `none` is reserved.
    struct Named;
    impl Predicate for Named {
        fn spec(&self) -> &PredicateSpec {
            static SPEC: std::sync::OnceLock<PredicateSpec> = std::sync::OnceLock::new();
            SPEC.get_or_init(|| PredicateSpec {
                name: "none".into(),
                version: "test".into(),
                parameters: vec![],
                horizon_plies: 0,
                access: Access::Viewer,
                phase: Phase::Play,
            })
        }
        fn evaluate(&self, _: PredicateContext<'_>, _: &[Value], _: &mut Budget) -> Result<Option<bool>> {
            Ok(Some(true))
        }
    }
    assert!(Registry::new().register(Named).is_err());
}
