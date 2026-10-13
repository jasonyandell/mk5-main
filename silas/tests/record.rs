//! The hand's public record in the frame, the predicates that read it, and
//! the `none` clause.
mod common;

use std::sync::Arc;

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
fn lowest_plays(physics: &dyn Physics, declarer: Seat, n: usize) -> Vec<(Seat, Domino)> {
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
    let plays = lowest_plays(physics.as_ref(), declarer, n);
    let hand = plays
        .iter()
        .filter(|(s, _)| *s == viewer)
        .fold(tiles(&DEAL[viewer.index()]), |h, (_, d)| h.difference(DominoSet::single(*d)));
    Frame::from_record(physics, viewer, hand, Contract { declarer, bid }, plays).unwrap()
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
    let plays = lowest_plays(physics.as_ref(), Seat::S0, 11);
    let contract = Contract { declarer: Seat::S0, bid: 30 };
    let mut stepped = Frame::from_record(physics.clone(), Seat::S0, tiles(&DEAL[0]), contract, vec![]).unwrap();
    for (n, (actor, tile)) in plays.iter().copied().enumerate() {
        let class = match stepped.led_context() {
            None => PlayClass::Lead,
            Some(q) if physics.follows(tile, q) => PlayClass::Follow(q),
            Some(q) => PlayClass::Slough(q),
        };
        stepped = step_frame(&stepped, ObservedPlay { actor, tile, class }, &mut budget()).unwrap();
        let hand = stepped.kernel().viewer_hand();
        let direct = Frame::from_record(physics.clone(), Seat::S0, hand, contract, plays[..=n].to_vec()).unwrap();
        assert_eq!(stepped, direct);
    }
    let record = stepped.record().unwrap();
    assert_eq!(record.plays(), plays.as_slice());
    assert_eq!(record.trick_number(), Some(3));
    assert_eq!(stepped.prefix().len(), 3);
    // Every closed trick's points are banked: a point a trick plus its count.
    let closed: u32 = record
        .tricks()
        .filter(|t| t.len() == 4)
        .map(|t| 1 + t.iter().map(|(_, d)| physics.count(*d)).sum::<u32>())
        .sum();
    assert_eq!(record.banked().iter().sum::<u32>(), closed);
}

#[test]
fn incoherent_records_are_refused() {
    let physics = physics("P6");
    let plays = lowest_plays(physics.as_ref(), Seat::S0, 8);
    let hand = |plays: &[(Seat, Domino)]| {
        plays
            .iter()
            .filter(|(s, _)| *s == Seat::S0)
            .fold(tiles(&DEAL[0]), |h, (_, d)| h.difference(DominoSet::single(*d)))
    };
    let at = |plays: Vec<(Seat, Domino)>, bid| {
        Frame::from_record(physics.clone(), Seat::S0, hand(&plays), Contract { declarer: Seat::S0, bid }, plays)
    };
    assert!(at(plays.clone(), 42).is_ok());
    assert!(at(plays.clone(), 43).is_err(), "a bid above the hand's 42 points");
    let mut wrong = plays.clone();
    wrong.swap(1, 2);
    assert!(at(wrong, 30).is_err(), "out of turn");
    let mut wrong = plays.clone();
    wrong[5].1 = wrong[1].1;
    assert!(at(wrong, 30).is_err(), "a repeated tile");
    // The declarer leads the first trick.
    let contract = Contract { declarer: Seat::S1, bid: 30 };
    assert!(Frame::from_record(physics.clone(), Seat::S0, hand(&plays), contract, plays.clone()).is_err());
    // The viewer's hand must be what its dealt hand left.
    assert!(
        Frame::from_record(
            physics.clone(),
            Seat::S0,
            tiles(&DEAL[0]),
            Contract { declarer: Seat::S0, bid: 30 },
            plays.clone()
        )
        .is_err()
    );
    // Blanks trump: the other seats' plays are checked against their shown
    // voids only, the viewer's against its hand. S1 shows void in trump, then
    // follows trump.
    let contract = Contract { declarer: Seat::S0, bid: 30 };
    let reneges = vec![
        (Seat::S0, tile("0-0")),
        (Seat::S1, tile("2-2")),
        (Seat::S2, tile("4-0")),
        (Seat::S3, tile("5-0")),
        (Seat::S0, tile("2-0")),
        (Seat::S1, tile("1-0")),
    ];
    assert!(
        Frame::from_record(common::physics("P0"), Seat::S0, hand(&reneges[..5]), contract, reneges[..5].to_vec())
            .is_ok()
    );
    assert!(Frame::from_record(common::physics("P0"), Seat::S0, hand(&reneges), contract, reneges).is_err());
    // The viewer may not slough a trump lead while holding trump.
    let mut viewer =
        vec![(Seat::S1, tile("1-0")), (Seat::S2, tile("4-0")), (Seat::S3, tile("5-0")), (Seat::S0, tile("1-1"))];
    let contract = Contract { declarer: Seat::S1, bid: 30 };
    assert!(Frame::from_record(common::physics("P0"), Seat::S0, hand(&viewer), contract, viewer.clone()).is_err());
    viewer[3].1 = tile("2-0");
    assert!(Frame::from_record(common::physics("P0"), Seat::S0, hand(&viewer), contract, viewer).is_ok());
}

#[test]
fn the_contract_and_score_read_as_the_book_reads() {
    let frame = recorded("P6", Seat::S2, Seat::S2, 34, 12);
    let record = frame.record().unwrap();
    let [t0, t1] = record.banked();
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
    // A contract made reads zero to make; set, zero to set.
    let made = recorded("P6", Seat::S2, Seat::S2, t0, 12);
    assert_eq!(both(&made, "(to-make 0)"), (true, false));
    let set = recorded("P6", Seat::S2, Seat::S2, 43 - t1, 12);
    assert_eq!(both(&set, "(to-set 0)"), (true, false));
}

#[test]
fn history_names_earlier_tricks_by_their_lead() {
    let frame = recorded("P6", Seat::S1, Seat::S3, 30, 9);
    let record = frame.record().unwrap();
    let plays = record.plays().to_vec();
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
fn without_a_record_its_predicates_are_undefined_not_false() {
    let frame = small_frame(physics("P6"));
    assert!(frame.record().is_none());
    for atom in [
        "(declarer S0)",
        "(bid 30)",
        "(banked S0 0)",
        "(to-make 30)",
        "(to-set 13)",
        "(trick-number 1)",
        "(led S0 6-6)",
        "(played-to 6-6 S0 6-6)",
        "(trick-of 6-6 1)",
        "(shown-void S0 q6)",
    ] {
        assert_eq!(both(&frame, atom), (false, false), "{atom}");
    }
    // A hidden seat's shown voids are the kernel's, record or not.
    assert_eq!(both(&frame, "(shown-void S1 q6)"), (false, true));
}

#[test]
fn shown_voids_are_the_kernels_and_imply_the_world_void() {
    for decl in ["P6", "P0", "DT", "NT"] {
        let frame = recorded(decl, Seat::S0, Seat::S0, 30, 17);
        let record = frame.record().unwrap();
        for h in frame.kernel().hidden() {
            assert_eq!(h.voids, record.shown_voids(frame.physics(), h.seat), "{decl} {}", h.seat);
        }
        let shown = compile("(query (roles (chair c) (context q)) (out c q) (pattern (shown-void c q)))")
            .evaluate_viewer(&frame, &mut budget())
            .unwrap();
        let expected: Answers = Seat::ALL
            .into_iter()
            .flat_map(|c| {
                record
                    .shown_voids(frame.physics(), c)
                    .iter()
                    .map(move |q| Answer(vec![Value::Chair(c), Value::Context(q)]))
            })
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
    let frame = Frame::from_record(p, Seat::S0, hand, Contract { declarer: Seat::S0, bid: 30 }, plays).unwrap();
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
    let world = "(policy p (initial a) (fallback lowest-legal) (rule r (in a) (next a) (select answer)
                   (guard (query (roles (domino d)) (out d) (pattern (own-legal d) (none (roles (chair c)) (holds c d)))))))";
    let p: Arc<dyn Physics> = physics("P6");
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
            })
        }
        fn evaluate(&self, _: PredicateContext<'_>, _: &[Value], _: &mut Budget) -> Result<Option<bool>> {
            Ok(Some(true))
        }
    }
    assert!(Registry::new().register(Named).is_err());
}
