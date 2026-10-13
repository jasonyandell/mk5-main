//! The surface sugar reads to the core it spells: inferred roles, a default
//! `(out)`, a query's one pattern without `(pattern ...)`, `and`, `or` and
//! negation of any clause, a rule's `(play ...)`, and an unsaid fallback.
//! Each short form is asked to equal its explicit form as syntax (the same
//! `Query`, printed the same) and as a compiled program (the same identity),
//! and the book's rules rewritten in it are asked to allow, at every point of
//! the book's narrated deals and of random ones, exactly what they allowed
//! as written before (`tests/data/explicit/`).
mod common;

use common::fixture::book;
use common::*;
use silas::*;

fn query(text: &str) -> Query {
    text.parse().unwrap_or_else(|e| panic!("{text}\n{e}"))
}

/// Two texts read to one query: equal as syntax, printed alike, and one
/// compiled identity.
fn same_query(short: &str, explicit: &str) {
    let (a, b) = (query(&format!("(include book) {short}")), query(&format!("(include book) {explicit}")));
    assert_eq!(a, b, "\n{short}\n{explicit}");
    assert_eq!(a.to_string(), b.to_string());
    let registry = Registry::standard();
    assert_eq!(a.compile(&registry).unwrap().identity(), b.compile(&registry).unwrap().identity());
}

fn error(text: &str) -> String {
    match Program::parse(text, "sugar") {
        Ok(Program::Query(q)) => q.compile(&Registry::standard()).map(|_| String::new()).unwrap_or_else(|e| e.0),
        Ok(Program::Policy(p)) => {
            p.compile(&Registry::standard(), &physics("P6")).map(|_| String::new()).unwrap_or_else(|e| e.0)
        }
        Ok(Program::Dominance(_)) => String::new(),
        Err(e) => e.0,
    }
}

#[test]
fn roles_take_their_sorts_from_their_uses_in_order_of_first_use() {
    // No (roles ...): every free name, in order of first use, sorted by the
    // slot it fills: a predicate's, a definition's, or a same partner's.
    same_query(
        "(query (out plain)
           (viewer me) (setter me) (partner me pard) (bidder-led led) (led-context q)
           (follower plain) (count plain 0) (follower counted) (count-tile counted))",
        "(query (roles (chair me) (chair pard) (domino led) (context q) (domino plain) (domino counted))
           (out plain)
           (pattern (viewer me) (setter me) (partner me pard) (bidder-led led) (led-context q)
             (follower plain) (count plain 0) (follower counted) (count-tile counted)))",
    );
    // A bare name in an explicit (roles ...), which keeps its order.
    same_query(
        "(query (roles d me) (out d) (pattern (viewer me) (own-legal d)))",
        "(query (roles (domino d) (chair me)) (out d) (pattern (viewer me) (own-legal d)))",
    );
    // A scope's roles, bare.
    same_query(
        "(query (out d) (live d) (none (roles e) (live e) (beats e d q*)))",
        "(query (roles (domino d)) (out d) (pattern (live d) (none (roles (domino e)) (live e) (beats e d q*))))",
    );
    // A same joins sorts: b is a chair because a is.
    same_query(
        "(query (roles a b) (out a b) (pattern (viewer a) (same a b)))",
        "(query (roles (chair a) (chair b)) (out a b) (pattern (viewer a) (same a b)))",
    );
    // A definition's parameters, bare, sorted by its body; a number too.
    same_query(
        "(define worth (roles d n) (count d n)) (query (worth 6-4 10))",
        "(define worth (roles (domino d) (number n)) (count d n)) (query (roles) (out) (pattern (worth 6-4 10)))",
    );
    // Over several patterns, first use runs across them.
    same_query(
        "(query (out d) (pattern (mine d) (double d)) (pattern (viewer c) (led c d)))",
        "(query (roles (domino d) (chair c)) (out d) (pattern (mine d) (double d)) (pattern (viewer c) (led c d)))",
    );
}

#[test]
fn out_and_pattern_may_go_unsaid() {
    // No (out): a yes-or-no question. One pattern's clauses, bare.
    same_query(
        "(query (viewer b) (bidder b) (lay-down))",
        "(query (roles (chair b)) (out) (pattern (viewer b) (bidder b) (lay-down)))",
    );
    // An empty query is still false, and an empty pattern still true.
    assert!(query("(query)").patterns.is_empty());
    assert_eq!(query("(query (pattern))").patterns.len(), 1);
    assert_eq!(query("(query)"), query("(query (roles) (out))"));
}

#[test]
fn and_or_and_not_are_kleenes_connectives_spelled_in_the_core() {
    // (or a b) is (none (roles) (not a) (not b)).
    same_query(
        "(query (or (bid 42) (bid 84)))",
        "(query (roles) (out) (pattern (none (roles) (not (bid 42)) (not (bid 84)))))",
    );
    // De Morgan, both ways, and double negation.
    same_query("(query (not (or (bid 42) (bid 84))))", "(query (pattern (not (bid 42)) (not (bid 84))))");
    same_query("(query (not (and (bid 42) (stake 1))))", "(query (pattern (none (roles) (bid 42) (stake 1))))");
    same_query("(query (not (not (bid 42))))", "(query (pattern (bid 42)))");
    same_query("(query (and (bid 42) (stake 1)))", "(query (pattern (bid 42) (stake 1)))");
    // (not (some ...)) is (none ...), and (not (none ...)) is (some ...).
    same_query(
        "(query (out c) (viewer c) (not (some (roles d) (mine d) (double d))))",
        "(query (roles (chair c)) (out c) (pattern (viewer c) (none (roles (domino d)) (mine d) (double d))))",
    );
    same_query(
        "(query (not (none (roles d) (mine d))))",
        "(query (roles) (out) (pattern (some (roles (domino d)) (mine d))))",
    );
    // The protect-the-count exception three ways: one core.
    let core = "(query (roles (chair pard)) (out) (pattern (partner S0 pard)
        (none (roles) (trump-set pard) (none (roles (domino his)) (trick-play pard his)))))";
    same_query("(query (partner S0 pard) (or (not (trump-set pard)) (some (roles his) (trick-play pard his))))", core);
    same_query("(query (partner S0 pard) (not (and (trump-set pard) (none (roles his) (trick-play pard his)))))", core);
    // A disjunct may be a conjunction.
    same_query(
        "(query (or (and (bid 42) (stake 1)) (bid 84)))",
        "(query (pattern (none (roles) (none (roles) (bid 42) (stake 1)) (not (bid 84)))))",
    );
    // And the connectives mean what they say: (or) over a frame.
    let frame = opening(&physics("P6"), tiles(&["0-0", "1-1", "2-0", "3-2", "5-5", "6-4", "6-6"]));
    let holds = |text: &str| {
        !query(&format!("(include book) {text}"))
            .compile(&Registry::standard())
            .unwrap()
            .evaluate_viewer(&frame, &mut budget())
            .unwrap()
            .is_empty()
    };
    assert!(holds("(query (or (bid 31) (bid 30)))"));
    assert!(!holds("(query (or (bid 31) (bid 32)))"));
    assert!(!holds("(query (or))"));
    assert!(holds("(query (and))"));
    assert!(holds("(query (not (and (bid 30) (bid 31))))"));
}

#[test]
fn a_rule_says_what_it_plays_and_the_fallback_goes_unsaid() {
    let registry = Registry::standard();
    let short = "(include book)
        (policy p
          (rule follow (play d) (follower d) (count d 0))
          (rule lead-the-six-four (play 6-4) (viewer b) (bidder b))
          (rule two (play d) (pattern (mine d) (double d)) (pattern (trump-tile d))))";
    let explicit = "(include book)
        (policy p
          (fallback lowest-legal)
          (rule follow (select answer) (guard (query (roles (domino d)) (out d) (pattern (follower d) (count d 0)))))
          (rule lead-the-six-four (select tile 6-4) (guard (query (roles (chair b)) (out) (pattern (viewer b) (bidder b)))))
          (rule two (select answer)
            (guard (query (roles (domino d)) (out d) (pattern (mine d) (double d)) (pattern (trump-tile d))))))";
    let (a, b) = (PolicyProgram::parse(short, "short").unwrap(), PolicyProgram::parse(explicit, "explicit").unwrap());
    assert_eq!(a, b);
    for rules in reference::Straight42::ALL {
        let p = rules.physics();
        assert_eq!(a.compile(&registry, &p).unwrap().identity(), b.compile(&registry, &p).unwrap().identity());
    }
    // (play ...) may be followed by (roles ...).
    let roles = "(policy p (rule r (play d) (roles (domino d)) (mine d)))";
    let plain = "(policy p (rule r (play d) (mine d)))";
    assert_eq!(PolicyProgram::parse(roles, "r").unwrap(), PolicyProgram::parse(plain, "p").unwrap());
}

#[test]
fn a_short_form_that_cannot_be_read_says_why_and_where() {
    for (text, expected) in [
        ("(query (viewer x) (mine x))", "sugar:1:25: x is a domino here and a chair at 1:16"),
        ("(query (out x) (viewer c))", "cannot infer the sort of x"),
        ("(query (roles x) (out) (pattern (viewer c)))", "sugar:1:15: cannot infer the sort of x"),
        ("(query (to-make n))", "n stands where a number goes"),
        ("(query (out d) (mine d) (pattern (double d)))", "not both"),
        ("(query (roles (domino d)) (pattern (mine d)) (out d))", "(out ...) is out of place"),
        ("(query (not (same a b)))", "same cannot be negated"),
        ("(policy p (rule r (play S0) (viewer c)))", "S0 is not a domino or a name to play"),
        ("(policy p (fallback lowest-legal) (fallback lowest-legal))", "one fallback"),
        ("(define or (roles (chair c)) (viewer c)) (query)", "cannot name a definition"),
    ] {
        let e = error(text);
        assert!(e.contains(expected), "{text}\n{e}");
    }
}

/// Every point of the eight full deals' narrated lines, from the seat to
/// play, with the bidder in each chair; then 200 seeded random deals under
/// the nine straight declarations, played out at random.
fn frames() -> Vec<Frame> {
    let mut out = Vec::new();
    let b = book();
    for id in ["hand-28", "hand-29", "hand-30", "hand-31", "hand-32", "hand-33", "hand-34", "hand-35"] {
        let example = b.example(id);
        let physics = example.physics();
        let deal = example.deal();
        let plays = b.line(&format!("{id}/prose")).plays();
        for declarer in Seat::ALL {
            let contract = contract(&physics, declarer, example.bid.number());
            for n in 0..plays.len() {
                let record = Frame::new(contract.clone(), declarer, deal[0], plays[..n].to_vec()).unwrap();
                let viewer = record.next_actor().unwrap();
                let role = (viewer.index() + 4 - declarer.index()) % 4;
                out.push(Frame::new(contract.clone(), viewer, deal[role], plays[..n].to_vec()).unwrap());
            }
        }
    }
    let mut rng = Rng(0x5eed);
    for game in 0..200 {
        let physics = reference::Straight42::ALL[game % 9].physics();
        let deck = rng.shuffled();
        let deal: [DominoSet; 4] = std::array::from_fn(|s| deck[7 * s..7 * s + 7].iter().copied().collect());
        let contract = contract(&physics, Seat::ALL[rng.below(4)], 30 + rng.below(13) as u32);
        let mut hands = deal;
        let mut record = Frame::new(contract.clone(), Seat::S0, deal[0], vec![]).unwrap();
        while let Some(viewer) = record.next_actor() {
            out.push(Frame::new(contract.clone(), viewer, deal[viewer.index()], record.plays().to_vec()).unwrap());
            let legal = physics.legal_plays(hands[viewer.index()], record.lead());
            let tile = legal.iter().nth(rng.below(legal.len())).unwrap();
            hands[viewer.index()].remove(tile);
            record = record.after(tile).unwrap();
        }
    }
    out
}

#[test]
fn the_book_rules_in_the_short_forms_allow_what_they_allowed_written_out() {
    let registry = Registry::standard();
    let frames = frames();
    assert!(frames.len() > 6_000);
    for name in ["protect-the-count", "bidders-plan", "low-from-a-pair"] {
        let read = |path: String| {
            let text = std::fs::read_to_string(&path).unwrap();
            PolicyProgram::parse(&text, &path).unwrap()
        };
        let root = env!("CARGO_MANIFEST_DIR");
        let explicit = read(format!("{root}/tests/data/explicit/{name}.silas"));
        let now = read(format!("{root}/examples/book/{name}.silas"));
        assert_eq!(now, explicit, "{name}");
        let mut compiled = std::collections::BTreeMap::new();
        let mut fired = 0;
        for frame in &frames {
            let id = frame.physics().identity().to_owned();
            let (a, b) = compiled.entry(id).or_insert_with(|| {
                (
                    now.compile(&registry, frame.physics()).unwrap(),
                    explicit.compile(&registry, frame.physics()).unwrap(),
                )
            });
            assert_eq!(a.identity(), b.identity());
            let allowed = a.allowed(frame, &mut budget()).unwrap();
            assert_eq!(allowed, b.allowed(frame, &mut budget()).unwrap(), "{name}");
            fired += usize::from(allowed.provenance != DecisionProvenance::Fallback);
        }
        assert!(fired > 0, "{name} never fires on these frames");
    }
}
