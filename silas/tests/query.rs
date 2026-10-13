//! The query language on finite worlds, against the reference physics.
mod common;

use std::collections::BTreeSet;

use common::*;
use silas::reference::Straight42;
use silas::*;

fn one(v: Value) -> Answer {
    Answer(vec![v])
}

#[test]
fn examples_parse_compile_and_roundtrip() {
    for s in [COUNTS, HAS_COUNT, MASTER, FORCED, SIX_FIVE] {
        let query = s.parse::<Query>().unwrap();
        assert_eq!(query.to_string().parse::<Query>().unwrap(), query);
        assert_eq!(
            query.compile(&Registry::standard()).unwrap().identity(),
            query.clone().compile(&Registry::standard()).unwrap().identity()
        );
    }
}

#[test]
fn syntax_and_type_errors_are_refused_before_evaluation() {
    for s in [
        "",
        "(query",
        "(query (roles) (out)))",
        "(query (roles) (out)) extra",
        "(query (roles) (out) (pattern (not)))",
        "(query (roles (number a)) (out))",
    ] {
        assert!(s.parse::<Query>().is_err(), "{s}");
    }
    for s in [
        "(query (roles (domino a) (chair a)) (out))",
        "(query (roles (domino S0)) (out))",
        "(query (roles (domino a)) (out b))",
        "(query (roles (domino a)) (out a a))",
        "(query (roles (chair a)) (out) (pattern (live a)))",
        "(query (roles (domino a)) (out) (pattern (holds S0 a q*)))",
        "(query (roles (domino a)) (out) (pattern (mystery a)))",
        "(query (roles (domino a) (chair b)) (out) (pattern (same a b)))",
        "(query (roles (domino a)) (out) (pattern (same a missing)))",
        "(query (roles (domino a)) (out) (pattern (count a a)))",
    ] {
        assert!(s.parse::<Query>().unwrap().compile(&Registry::standard()).is_err(), "{s}");
    }
}

#[test]
fn empty_query_false_empty_pattern_true_and_hidden_witnesses_are_existential() {
    let b = uniform();
    let world = b.worlds().next().unwrap().0;
    assert!(compile("(query (roles) (out))").evaluate(b.frame(), world, &mut budget()).unwrap().is_empty());
    assert_eq!(
        compile("(query (roles) (out) (pattern))").evaluate(b.frame(), world, &mut budget()).unwrap(),
        BTreeSet::from([Answer(vec![])])
    );
    // 28 witnesses still produce one Boolean answer.
    let many = compile("(query (roles (domino hidden)) (out) (pattern))");
    assert_eq!(many.evaluate(b.frame(), world, &mut budget()).unwrap(), BTreeSet::from([Answer(vec![])]));
}

#[test]
fn equality_patterns_are_complete_and_quotient_before_injective_binding() {
    let b = uniform();
    let w = b.worlds().next().unwrap().0;
    let distinct = compile("(query (roles (domino a) (domino b)) (out a b) (pattern))");
    let alias = compile("(query (roles (domino a) (domino b)) (out a b) (pattern (same a b)))");
    let all = compile("(query (roles (domino a) (domino b)) (out a b) (pattern) (pattern (same a b)))");
    assert_eq!(distinct.evaluate(b.frame(), w, &mut budget()).unwrap().len(), 28 * 27);
    assert_eq!(alias.evaluate(b.frame(), w, &mut budget()).unwrap().len(), 28);
    let expected: Answers = Domino::ALL
        .into_iter()
        .flat_map(|a| Domino::ALL.map(|b| Answer(vec![Value::Domino(a), Value::Domino(b)])))
        .collect();
    assert_eq!(all.evaluate(b.frame(), w, &mut budget()).unwrap(), expected);
    let transitive =
        compile("(query (roles (chair a) (chair b) (chair c)) (out a b c) (pattern (same a b) (same c b)))");
    let expected = Seat::ALL.map(|s| Answer(vec![Value::Chair(s); 3])).into_iter().collect();
    assert_eq!(transitive.evaluate(b.frame(), w, &mut budget()).unwrap(), expected);
}

#[test]
fn joins_match_direct_relations_on_every_world_under_all_nine_declarations() {
    let query = compile(
        "(query (roles (chair me) (chair partner) (domino d) (context suit)) (out d suit)
        (pattern (viewer me) (partner me partner) (holds partner d) (in d suit)))",
    );
    for rules in Straight42::ALL {
        let b = Belief::uniform(small_frame(rules.physics()), 90).unwrap();
        for (world, _) in b.worlds() {
            let expected: Answers = world
                .hand(Seat::S2)
                .iter()
                .flat_map(|d| {
                    Context::ALL
                        .into_iter()
                        .filter(move |q| rules.incidence(*q).contains(d))
                        .map(move |q| Answer(vec![Value::Domino(d), Value::Context(q)]))
                })
                .collect();
            assert_eq!(query.evaluate(b.frame(), world, &mut budget()).unwrap(), expected, "{rules:?} {world:?}");
        }
    }
}

#[test]
fn overlapping_patterns_and_internal_witnesses_do_not_multiply_mass() {
    let b = uniform();
    let base = compile(COUNTS);
    let mut duplicate = base.source().clone();
    duplicate.patterns.extend(duplicate.patterns.clone());
    let duplicate = duplicate.compile(&Registry::standard()).unwrap();
    assert!(base.compare(&duplicate, &b, Comparison::Answers, &mut budget()).unwrap().is_none());
    let summary = base.summarize(&b, &mut budget()).unwrap();
    assert_eq!(exact(summary.event_probability()), r(4, 5)); // 1 - C(3,2)/C(6,2)
    for d in ["3-2", "6-4", "5-5"] {
        assert_eq!(exact(summary.presence_probability(&one(Value::Domino(tile(d))))), r(1, 3));
    }
    let boolean = compile(HAS_COUNT).summarize(&b, &mut budget()).unwrap();
    assert_eq!(exact(boolean.event_probability()), r(4, 5));
    assert_eq!(boolean.answer_presence_mass.len(), 1);
    assert_eq!(duplicate.summarize(&b, &mut budget()).unwrap().answer_set_mass, summary.answer_set_mass);
}

#[test]
fn answer_multiplicity_is_not_a_probability_distribution() {
    let full = uniform();
    let two = full.worlds().find(|(w, _)| w.hand(Seat::S2) == tiles(&["3-2", "6-4"])).unwrap().0;
    let one_count = full.worlds().find(|(w, _)| w.hand(Seat::S2) == tiles(&["5-5", "1-0"])).unwrap().0;
    let b = Belief::from_weights(
        full.frame().clone(),
        "two equally weighted worlds",
        given("test: two worlds, equal weight"),
        [(*two, w(1, 1)), (*one_count, w(1, 1))],
        2,
    )
    .unwrap();
    let summary = compile(COUNTS).summarize(&b, &mut budget()).unwrap();
    assert_eq!(summary.event_probability().value(), &r(1, 1));
    assert_eq!(summary.event_probability().authority(), &given("test: two worlds, equal weight"));
    let presence: Weight = summary.answer_presence_mass.values().sum();
    assert_eq!(presence.rational() / summary.total_mass.rational(), r(3, 2));
    assert!(!summary.certainty.world_functional);
    assert_eq!(summary.certainty.constant_multiplicity, None);
    let uniform_law = summary.select(Selection::UniformWithinWorld);
    let first = summary.select(Selection::LexicographicFirst);
    assert_eq!(uniform_law.probabilities[&one(Value::Domino(tile("3-2")))], w(1, 4));
    assert_eq!(first.probabilities[&one(Value::Domino(tile("3-2")))], w(1, 2));
    assert_eq!(uniform_law.probabilities[&one(Value::Domino(tile("5-5")))], w(1, 2));
    assert_eq!(uniform_law.probabilities.values().sum::<Weight>(), w(1, 1));
    assert_eq!(uniform_law.authority, given("test: two worlds, equal weight"));
}

#[test]
fn certainty_does_not_confuse_existence_with_identity() {
    let b = uniform();
    let holder = compile("(query (roles (chair holder)) (out holder) (pattern (holds holder 5-5)))");
    let s = holder.summarize(&b, &mut budget()).unwrap();
    assert!(s.certainty.event_certain && s.certainty.world_functional);
    assert_eq!(s.certainty.constant_multiplicity, Some(1));
    assert!(s.certainty.identity.is_none());
    let (posterior, p) = b
        .condition(&compile("(query (roles) (out) (pattern (holds S2 5-5)))"), "observed holder", &mut budget())
        .unwrap();
    assert_eq!(exact(p), r(1, 3));
    assert_eq!(
        holder.summarize(&posterior, &mut budget()).unwrap().certainty.identity,
        Some(one(Value::Chair(Seat::S2)))
    );
    assert_eq!(holder.summarize(&b, &mut budget()).unwrap(), s);
}

#[test]
fn conditioning_uses_events_and_exact_likelihoods_transactionally() {
    let b = uniform();
    let (posterior, p) = b.condition(&compile(HAS_COUNT), "partner has count", &mut budget()).unwrap();
    assert_eq!(posterior.len(), 72);
    assert_eq!(exact(p), r(4, 5));
    let (soft, p) = b
        .condition_likelihood("soft evidence", "test: half if partner holds 5-5, else a quarter", |h| {
            Ok(if h.holds(Seat::S2, tile("5-5"))? { w(1, 2) } else { w(1, 4) })
        })
        .unwrap();
    assert_eq!(p.value(), &r(1, 3));
    assert_eq!(p.authority(), &given("test: half if partner holds 5-5, else a quarter"));
    assert_eq!(soft.authority(), p.authority());
    assert!(b.condition(&compile("(query (roles) (out))"), "impossible", &mut budget()).is_err());
    assert!(b.condition_likelihood("invalid", "test: two", |_| Ok(w(2, 1))).is_err());
    assert!(b.condition_likelihood("unnamed", " ", |_| Ok(w(1, 2))).is_err());
    assert!(Weight::new(r(-1, 1)).is_err());
    assert_eq!(b.len(), 90);
}

#[test]
fn duplicate_worlds_merge_but_bad_measures_and_caps_are_refused() {
    let b = uniform();
    let w0 = *b.worlds().next().unwrap().0;
    let model = given("test: duplicate entries");
    let merged =
        Belief::from_weights(b.frame().clone(), "duplicates", model.clone(), [(w0, w(1, 3)), (w0, w(2, 3))], 1)
            .unwrap();
    assert_eq!(merged.len(), 1);
    assert_eq!(*merged.total_mass(), w(1, 1));
    assert!(Belief::from_weights(b.frame().clone(), "empty", model.clone(), [(w0, w(0, 1))], 1).is_err());
    assert!(Belief::from_weights(b.frame().clone(), "claims exact", Authority::exact(), [(w0, w(1, 1))], 1).is_err());
    assert!(Belief::uniform(b.frame().clone(), 89).is_err());
    let invalid = b.frame().kernel().world([DominoSet::EMPTY; 3]);
    assert!(Belief::from_weights(b.frame().clone(), "invalid world", model, [(invalid, w(1, 1))], 1).is_err());
    assert!(compile(COUNTS).evaluate(b.frame(), &invalid, &mut budget()).is_err());
}

#[test]
fn answer_equivalence_is_stronger_than_boolean_equivalence() {
    let a = compile("(query (roles (domino d)) (out d) (pattern (tile d 0-0)))");
    let b = compile("(query (roles (domino d)) (out d) (pattern (tile d 6-6)))");
    let measure = uniform();
    assert!(a.compare(&b, &measure, Comparison::Existence, &mut budget()).unwrap().is_none());
    let witness = a.compare(&b, &measure, Comparison::Answers, &mut budget()).unwrap().unwrap();
    assert_ne!(witness.left, witness.right);
    assert!(measure.frame().contains(&witness.world));
    assert_eq!(a.evaluate(measure.frame(), &witness.world, &mut budget()).unwrap(), witness.left);
}

#[test]
fn the_records_open_trick_drives_partial_trick_queries() {
    // S0 leads the 6-6 from the small frame.
    let f = small_frame(physics("P6")).after(tile("6-6")).unwrap();
    assert_eq!((f.kernel().viewer_hand(), f.prefix()), (tiles(&["0-0"]), &[tile("6-6")][..]));
    let b = Belief::uniform(f, 90).unwrap();
    let forced = compile(FORCED);
    let rules: Straight42 = "P6".parse().unwrap();
    for (w, _) in b.worlds() {
        // Sixes are trump: S1 must follow with a six if it holds one.
        let sixes = w.hand(Seat::S1).intersection(rules.incidence(Context::Called));
        let legal = if sixes.is_empty() { w.hand(Seat::S1) } else { sixes };
        let expected = match legal.iter().collect::<Vec<_>>().as_slice() {
            [only] => BTreeSet::from([Answer(vec![Value::Chair(Seat::S1), Value::Domino(*only)])]),
            _ => BTreeSet::new(),
        };
        assert_eq!(forced.evaluate(b.frame(), w, &mut budget()).unwrap(), expected);
    }
    let event = compile(
        "(query (roles) (out) (pattern (leader S0) (current-winner S0) (next-actor S1) (led-context q*) (played 6-6) (not (live 6-6))))",
    );
    assert_eq!(exact(event.summarize(&b, &mut budget()).unwrap().event_probability()), r(1, 1));
}

#[test]
fn undefined_is_not_false_and_never_satisfies_negation() {
    let b = uniform();
    for s in [
        "(query (roles) (out) (pattern (led-context q*)))",
        "(query (roles) (out) (pattern (not (led-context q*))))",
        "(query (roles) (out) (pattern (not (legal S1 5-5))))",
    ] {
        assert!(compile(s).summarize(&b, &mut budget()).unwrap().event_mass.is_zero());
    }
}

#[test]
fn work_exhaustion_refuses_a_partial_query_or_summary() {
    let b = uniform();
    let c = compile(COUNTS);
    assert!(c.summarize(&b, &mut Budget::new(1)).is_err());
    assert!(c.evaluate(b.frame(), b.worlds().next().unwrap().0, &mut Budget::new(0)).is_err());
    assert_eq!(exact(c.summarize(&b, &mut budget()).unwrap().event_probability()), r(4, 5));
}

struct ViewerProbe(PredicateSpec);

impl Predicate for ViewerProbe {
    fn spec(&self) -> &PredicateSpec {
        &self.0
    }

    fn evaluate(&self, ctx: PredicateContext<'_>, args: &[Value], budget: &mut Budget) -> Result<Option<bool>> {
        assert!(ctx.world().is_none(), "a viewer-only computation never sees a world");
        budget.spend(2)?;
        match args {
            [Value::Domino(d)] => Ok(Some(ctx.frame().kernel().viewer_hand().contains(*d))),
            _ => panic!("the compiler checked the signature"),
        }
    }
}

#[test]
fn extension_registry_freezes_semantics_and_enforces_information_access() {
    let mut registry = Registry::standard();
    let spec = PredicateSpec {
        name: "in-hand".to_owned(),
        version: "test-v1".to_owned(),
        parameters: vec![Sort::Domino],
        horizon_plies: 0,
        access: Access::Viewer,
    };
    registry.register(ViewerProbe(spec.clone())).unwrap();
    assert!(registry.register(ViewerProbe(spec)).is_err());
    let query = "(query (roles (domino d)) (out d) (pattern (in-hand d)))"
        .parse::<Query>()
        .unwrap()
        .compile(&registry)
        .unwrap();
    let s = query.summarize(&uniform(), &mut budget()).unwrap();
    assert_eq!(s.certainty.constant_multiplicity, Some(2));
    assert!(query.identity().contains("test-v1"));
    assert_eq!(
        s.certainty.constant_answers.unwrap(),
        tiles(&["6-6", "0-0"]).iter().map(|d| one(Value::Domino(d))).collect()
    );
}

#[test]
fn zero_weight_worlds_do_not_affect_belief_certainty() {
    let full = uniform();
    let worlds: Vec<_> = full.worlds().take(2).map(|(w, _)| *w).collect();
    let b = Belief::from_weights(
        full.frame().clone(),
        "point mass",
        given("test: point mass"),
        [(worlds[0], w(1, 1)), (worlds[1], w(0, 1))],
        2,
    )
    .unwrap();
    assert_eq!(b.len(), 1);
    let query = compile("(query (roles (chair c)) (out c) (pattern (holds c 5-5)))");
    assert!(query.summarize(&b, &mut budget()).unwrap().certainty.identity.is_some());
}

/// Seat 0's view of a recorded hand before its sixth trick (texas-42's
/// receipt hand 0, trick 6, reproduced here so the test stands alone). Threes
/// are trump; S1 led the 0-0, S2 and S3 have shown void in blanks and in
/// trumps, and S0 is to play. Three tiles are hidden, one per seat, and the
/// only live count tile is 4-1. The record: S0 draws trump three times (S2
/// and S3 show out on the first), S1 takes the fourth trick with the 6-6 and
/// the fifth with the last trump, and leads the 0-0.
fn receipt_coordinate() -> Frame {
    let plays = [
        "3-3", "3-0", "1-0", "5-4", // T1: S0 leads; S2 and S3 show out of trump
        "6-3", "5-3", "4-0", "5-5", // T2
        "4-3", "3-2", "5-0", "6-0", // T3
        "6-5", "6-6", "6-4", "6-2", // T4: S1 takes it
        "3-1", "5-1", "6-1", "1-1", // T5: S1's last trump
        "0-0", "5-2", "4-4", // T6: S2 and S3 show out of blanks
    ];
    let p = physics("P3");
    let dealt = tiles(&["2-0", "4-2", "3-3", "6-3", "4-3", "6-5", "1-1"]);
    let frame = Frame::new(contract(&p, Seat::S0, 30), Seat::S0, dealt, plays.map(tile).to_vec()).unwrap();
    let shown: ContextSet = ["q0", "q*"].iter().map(|q| q.parse::<Context>().unwrap()).collect();
    assert_eq!((frame.leader(), frame.prefix().len(), frame.next_actor()), (Seat::S1, 3, Some(Seat::S0)));
    assert_eq!(frame.kernel().hidden().map(|h| h.voids), [ContextSet::EMPTY, shown, shown]);
    assert_eq!(frame.kernel().pool(), tiles(&["2-1", "2-2", "4-1"]));
    frame
}

#[test]
fn receipt_coordinate_query_conditions_and_refuses_without_partial_results() {
    let frame = receipt_coordinate();
    assert_eq!(frame.count_worlds(), 6);
    let belief = Belief::uniform(frame.clone(), 40_000).unwrap();
    let summary = compile(COUNTS).summarize(&belief, &mut Budget::new(10_000_000)).unwrap();
    // Independently: the three hidden tiles go one to each seat, 3! = 6 ways,
    // and 4-1 sits with partner S2 in two of them.
    assert_eq!(summary.positive_worlds, 6);
    assert_eq!(exact(summary.event_probability()), r(1, 3));
    assert_eq!(exact(summary.presence_probability(&one(Value::Domino(tile("4-1"))))), r(1, 3));
    let (posterior, p) = belief.condition(&compile(HAS_COUNT), "partner has count", &mut budget()).unwrap();
    assert_eq!(exact(p), r(1, 3));
    let conditioned = compile(COUNTS).summarize(&posterior, &mut budget()).unwrap();
    assert_eq!(conditioned.positive_worlds, 2);
    assert_eq!(exact(conditioned.event_probability()), r(1, 1));
    assert!(compile(COUNTS).summarize(&belief, &mut Budget::new(1)).is_err());
    assert!(Belief::uniform(frame, 5).is_err());
}

#[test]
fn mastery_beating_and_negative_joins_match_direct_set_queries() {
    let masters = compile("(query (roles (domino d)) (out d) (pattern (master d)))");
    let beaters = compile(
        "(query (roles (domino a) (domino witness)) (out a)
        (pattern (holds S2 a) (holds S1 witness) (beats a witness q*) (not (double a))))",
    );
    for rules in Straight42::ALL {
        let key = |d: Domino, q: Context| rules.trick_key(d, q);
        let b = Belief::uniform(small_frame(rules.physics()), 90).unwrap();
        let live = b.frame().kernel().live();
        // A master is live and no live tile outranks it in the context it leads.
        let expected_masters: Answers = live
            .iter()
            .filter(|d| live.iter().all(|e| key(e, rules.led_context(*d)) <= key(*d, rules.led_context(*d))))
            .map(|d| one(Value::Domino(d)))
            .collect();
        for (world, _) in b.worlds() {
            assert_eq!(masters.evaluate(b.frame(), world, &mut budget()).unwrap(), expected_masters);
            let expected: Answers = world
                .hand(Seat::S2)
                .iter()
                .filter(|a| {
                    !a.is_double()
                        && world.hand(Seat::S1).iter().any(|w| key(*a, Context::Called) > key(w, Context::Called))
                })
                .map(|a| one(Value::Domino(a)))
                .collect();
            assert_eq!(beaters.evaluate(b.frame(), world, &mut budget()).unwrap(), expected);
        }
    }
}

#[test]
fn frames_reject_incoherent_records_and_played_is_the_record() {
    let f = small_frame(physics("NT"));
    let again = |plays: Vec<Domino>| Frame::new(f.contract().clone(), Seat::S0, f.dealt(), plays);
    // A tile played twice, or a viewer's tile played by another seat.
    let mut twice = f.plays().to_vec();
    twice.push(f.plays()[0]);
    assert!(again(twice).is_err());
    let mut stolen = f.plays()[..1].to_vec();
    stolen.push(tile("6-6"));
    assert!(again(stolen).is_err());
    // A tile is played exactly when it is not live: the record is the one
    // source of both.
    let b = Belief::uniform(f.clone(), 90).unwrap();
    for d in Domino::ALL {
        let both = compile(&format!("(query (roles) (out) (pattern (not (live {d})) (not (played {d}))))"));
        assert_eq!(exact(both.summarize(&b, &mut budget()).unwrap().event_probability()), r(0, 1), "{d}");
        assert_eq!(f.played().contains(d), !f.kernel().live().contains(d));
    }
}

#[test]
fn a_resolving_belief_agrees_with_the_full_support_and_refuses_unresolved_reads() {
    let full = uniform();
    let resolving = Belief::uniform_resolving(full.frame().clone(), tiles(&["5-5", "3-2", "6-4"]), 90).unwrap();
    assert_eq!(resolving.total_mass(), &Weight::integer(90));
    // Three tiles to three seats of two: 27 placements less the 3 that crowd one seat.
    assert_eq!(resolving.len(), 24);
    assert!(Belief::uniform_resolving(full.frame().clone(), tiles(&["5-5", "3-2", "6-4"]), 23).is_err());
    for source in [COUNTS, HAS_COUNT, "(query (roles (chair c)) (out c) (pattern (holds c 5-5)))"] {
        let q = compile(source);
        let (a, b) = (q.summarize(&full, &mut budget()).unwrap(), q.summarize(&resolving, &mut budget()).unwrap());
        assert_eq!((a.event_probability(), a.answer_presence_mass), (b.event_probability(), b.answer_presence_mass));
    }
    // 6-5 is unresolved: reading where it sits is refused, never guessed.
    assert!(compile("(query (roles) (out) (pattern (holds S1 6-5)))").summarize(&resolving, &mut budget()).is_err());
    // ...but the viewer's own hand and resolved tiles are always readable.
    assert!(
        compile("(query (roles) (out) (pattern (holds S0 6-6) (holds S1 5-5)))")
            .summarize(&resolving, &mut budget())
            .is_ok()
    );
    let (posterior, p) = resolving.condition(&compile(HAS_COUNT), "partner has count", &mut budget()).unwrap();
    assert_eq!(exact(p), r(4, 5));
    assert_eq!(posterior.unresolved(), tiles(&["6-5", "2-1", "1-0"]));
    let everything = Belief::uniform_resolving(full.frame().clone(), DominoSet::FULL, 90).unwrap();
    assert_eq!((everything.len(), everything.unresolved()), (90, DominoSet::EMPTY));
}
