//! Diagnostics for a rule that silently never fires: warnings read off the
//! core (`Query::lint`, `PolicyProgram::lint`), and the funnel that says
//! where a pattern stops and how often each step holds (`Funnel`,
//! `Coverage`, `why_not`).
mod common;

use std::path::PathBuf;

use common::*;
use silas::*;

fn lint(text: &str) -> Vec<String> {
    let text = format!("(include book)\n{text}");
    match Program::parse(&text, "lint").unwrap_or_else(|e| panic!("{e}")) {
        Program::Query(q) => q.lint(),
        Program::Policy(p) => p.lint(),
        Program::Dominance(_) => vec![],
    }
    .into_iter()
    .map(|w| w.0)
    .collect()
}

fn warns(text: &str, expected: &str) {
    let warnings = lint(text);
    assert!(warnings.iter().any(|w| w.contains(expected)), "{text}\n{warnings:#?}");
    assert!(warnings.iter().all(|w| w.contains("warning: ")), "{warnings:#?}");
}

#[test]
fn each_warning_names_what_makes_a_pattern_never_hold_or_a_clause_say_nothing() {
    warns("(query (roles (domino d) (domino e)) (out d) (pattern (mine d)))", "e is declared and never used");
    warns("(query (some (roles (domino x)) (bid 30)))", "x is bound and never used");
    warns(
        "(query (roles (chair a) (chair b) (chair c) (chair d) (chair e)) (out)
           (pattern (viewer a) (leader b) (next-actor c) (partner a d) (successor a e)))",
        "the pattern needs 5 distinct chairs and there are 4",
    );
    warns("(query (out a) (mine a) (live b) (tile a b))", "never true: a and b are declared together");
    warns("(query (out a) (mine a) (live b) (not (tile a b)))", "always true: a and b");
    warns("(query (out d) (mine d) (double d) (not (double d)))", "never holds: this clause and its negation");
    // The role a negation forgot to bind, written out and inferred.
    warns(
        "(query (roles (chair p) (domino his)) (out) (pattern (partner S0 p) (none (roles) (trick-play p his))))",
        "his is used only under a negation",
    );
    warns("(query (partner S0 p) (not (trick-play p his)))", "his is used only under a negation");
    // A rule's answer the viewer cannot play, and a rule with no pattern.
    warns("(policy p (rule r (play d) (live d)))", "rule r plays d without asking (own-legal d) or (mine d)");
    warns("(policy p (rule r (play 6-4)))", "rule r has no pattern, so it never fires");
    // What is not warned: a scope's own role under its own negation, a
    // literal against a role, an output used only negated, a double
    // negation.
    for quiet in [
        "(query (out d) (live d) (none (roles e) (live e) (beats e d q*)))",
        "(query (out d) (mine d) (not (tile d 6-4)))",
        "(query (out d) (not (mine d)))",
        "(query (partner S0 p) (none (roles) (none (roles) (trick-play p his))))",
        "(policy p (rule r (play d) (follower d)))",
    ] {
        assert_eq!(lint(quiet), Vec::<String>::new(), "{quiet}");
    }
    // Located as an error is: source, line and column, the line and a caret.
    let w = &lint("(query (out d)\n  (mine d) (double d) (not (double d)))")[0];
    assert!(w.starts_with("lint:3:23: warning: never holds"), "{w}");
    assert!(w.lines().last().unwrap().ends_with('^'), "{w}");
}

fn silas_files() -> Vec<PathBuf> {
    fn walk(dir: PathBuf, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(path, out);
            } else if path.extension().is_some_and(|x| x == "silas") {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    walk(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples"), &mut files);
    files.sort();
    files
}

#[test]
fn the_crates_files_warn_only_where_a_clause_is_redundant() {
    let mut warnings = Vec::new();
    for path in silas_files() {
        let text = std::fs::read_to_string(&path).unwrap();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        match Program::parse(&text, &name) {
            Ok(Program::Query(q)) => warnings.extend(q.lint()),
            Ok(Program::Policy(p)) => warnings.extend(p.lint()),
            _ => {}
        }
    }
    // The crisis's (not (tile his l)) restates what distinctness says: a
    // seat's own lead is not his follow. Redundant, and harmless.
    let warnings: Vec<String> = warnings.into_iter().map(|w| w.0).collect();
    assert_eq!(warnings.len(), 1, "{warnings:#?}");
    assert!(warnings[0].starts_with("crisis.silas:78:13: warning: always true: his and l"), "{warnings:#?}");
}

/// S0, the declarer under sixes, to lead after one trick in which S1 showed
/// out of trump.
fn led_frame() -> Frame {
    let p = physics("P6");
    let dealt = tiles(&["0-0", "1-1", "2-0", "3-2", "5-5", "6-4", "6-6"]);
    Frame::new(contract(&p, Seat::S0, 30), Seat::S0, dealt, ["6-4", "1-0", "6-1", "6-2"].map(tile).to_vec()).unwrap()
}

#[test]
fn a_funnel_steps_through_the_source_and_stops_where_the_pattern_fails() {
    let registry = Registry::standard();
    // A definition call is one step, however many primitives it expands to;
    // the steps keep the source's order though the core keeps atoms first.
    let q: Query = "(include book)
        (query (out d)
          (viewer b)
          (bidder b)
          (none (roles e) (mine e) (double e) (beats e d q*))
          (own-legal d)
          (double d)
          (trump-tile d))"
        .parse()
        .unwrap();
    let funnels = Funnel::of(&q, &registry).unwrap();
    assert_eq!(funnels.len(), 1);
    assert_eq!(
        funnels[0].labels(),
        [
            "(viewer b)",
            "(bidder b)",
            "(none (roles e) (mine e) (double e) (beats e d q*))",
            "(own-legal d)",
            "(double d)",
            "(trump-tile d)"
        ]
    );
    // S0 bids and leads: he holds the 6-6, a double trump; every step holds.
    let frame = led_frame();
    assert_eq!(funnels[0].depth(&frame, &mut budget()).unwrap(), 6);
    // Once he has led the 6-6 it is S1's turn: own-legal is undefined, and
    // the pattern stops at it.
    let after = frame.after(tile("6-6")).unwrap();
    assert_eq!(funnels[0].depth(&after, &mut budget()).unwrap(), 3);
}

#[test]
fn coverage_counts_each_rule_and_each_step_and_says_why_a_rule_passed() {
    let registry = Registry::standard();
    let p = physics("P6");
    // The first rule can never hold: a and b are distinct, so (tile a b)
    // fails; the second plays a double; the third any legal tile.
    let program = PolicyProgram::parse(
        "(include book)
         (policy p
           (rule never (play a) (own-legal a) (mine b) (tile a b))
           (rule doubles (play d) (own-legal d) (double d))
           (rule any (play d) (own-legal d)))",
        "p",
    )
    .unwrap();
    assert!(program.lint().iter().any(|w| w.0.contains("never true: a and b")));
    let policy = program.compile(&registry, &p).unwrap();
    let mut coverage = Coverage::new(&policy, &registry).unwrap();
    let mut rng = Rng(42);
    let mut frames = Vec::new();
    for _ in 0..20 {
        let deck = rng.shuffled();
        let deal: [DominoSet; 4] = std::array::from_fn(|s| deck[7 * s..7 * s + 7].iter().copied().collect());
        let contract = contract(&p, Seat::S0, 30);
        let mut record = Frame::new(contract.clone(), Seat::S0, deal[0], vec![]).unwrap();
        let mut hands = deal;
        while let Some(viewer) = record.next_actor() {
            frames.push(Frame::new(contract.clone(), viewer, deal[viewer.index()], record.plays().to_vec()).unwrap());
            let legal = p.legal_plays(hands[viewer.index()], record.lead());
            let tile = legal.iter().nth(rng.below(legal.len())).unwrap();
            hands[viewer.index()].remove(tile);
            record = record.after(tile).unwrap();
        }
    }
    for frame in &frames {
        coverage.observe(&policy, frame, &mut budget()).unwrap();
    }
    assert_eq!(coverage.frames, 560);
    let [never, doubles, any] = &coverage.rules[..] else { panic!() };
    // Every frame is decided once: by a rule, the exact table or the fallback.
    assert_eq!(never.decided + doubles.decided + any.decided + coverage.exact + coverage.fallback, coverage.frames);
    assert_eq!((never.answered, never.decided), (0, 0));
    // (mine b) wants a second tile in hand: not on the last trick's 80 frames.
    assert_eq!(never.patterns[0].iter().map(|(_, n)| *n).collect::<Vec<_>>(), [560, 480, 0]);
    assert_eq!(any.allowed, 560);
    assert_eq!(any.decided, 560 - doubles.decided);
    assert!(doubles.decided > 0 && doubles.decided == doubles.allowed);
    // The steps hold on a prefix: each count at most the one before, and the
    // last the frames where the guard answered.
    for rule in &coverage.rules {
        let counts: Vec<u64> = rule.patterns[0].iter().map(|(_, n)| *n).collect();
        assert!(counts.windows(2).all(|w| w[0] >= w[1]), "{}: {counts:?}", rule.name);
        assert_eq!(*counts.last().unwrap(), rule.answered, "{}", rule.name);
    }
    // At one frame, why each rule before the decider passed.
    let frame = frames
        .iter()
        .find(|f| {
            policy.decide(f, &mut budget()).unwrap().provenance
                == (DecisionProvenance::RelationalRule { name: "any".into() })
        })
        .unwrap();
    let decision = policy.decide(frame, &mut budget()).unwrap();
    let why = why_not(&policy, &decision.provenance, frame, &registry, &mut budget()).unwrap();
    assert_eq!(
        why,
        [
            ("never".to_owned(), "stops at (tile a b)".to_owned()),
            ("doubles".to_owned(), "stops at (double d)".to_owned())
        ]
    );
}
