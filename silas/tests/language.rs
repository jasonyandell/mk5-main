// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The language's surface: `some`, named definitions and the book prelude,
//! located errors, and the canonical formatter.
mod common;

use std::path::PathBuf;

use common::*;
use silas::*;

/// S0, the declarer under sixes, to lead after one trick in which S1 showed
/// out of trump (the policy-test deal, played lowest).
fn led_frame() -> Frame {
    let p = physics("P6");
    let dealt = tiles(&["0-0", "1-1", "2-0", "3-2", "5-5", "6-4", "6-6"]);
    framed(&contract(&p, Seat::S0, 30), Seat::S0, dealt, ["6-4", "1-0", "6-1", "6-2"].map(tile).to_vec()).unwrap()
}

fn answers(frame: &Frame, query: &str) -> Answers {
    compile(query).evaluate_viewer(frame, &mut budget()).unwrap()
}

fn holds(frame: &Frame, query: &str) -> bool {
    !answers(frame, query).is_empty()
}

#[test]
fn some_is_kleenes_existential_and_none_its_negation() {
    let frame = led_frame();
    let s0_leads = "(query (roles) (out) (pattern (some (roles (domino d)) (own-legal d) (double d))))";
    assert!(holds(&frame, s0_leads));
    // Not S0's turn: own-legal is undefined, and so are both some and none.
    let s1 = frame.after(tile("6-6")).unwrap();
    assert!(!holds(&s1, s0_leads));
    assert!(!holds(&s1, &s0_leads.replace("(some", "(none")));
    // A defined, false clause refutes every binding: some is false, none true.
    let refuted = "(query (roles) (out) (pattern (none (roles (domino d)) (own-legal d) (double d) (not (double d)))))";
    assert!(holds(&s1, refuted));
    assert!(!holds(&s1, &refuted.replace("(none", "(some")));
    // (not (some ...)) is written (none ...): the same answers.
    let some =
        "(query (roles (domino d)) (out d) (pattern (live d) (some (roles (domino e)) (live e) (beats e d q*))))";
    let negated =
        "(query (roles (domino d)) (out d) (pattern (live d) (none (roles (domino e)) (live e) (beats e d q*))))";
    let boss = "(query (roles (domino d)) (out d) (pattern (boss d q*)))";
    let live: usize = frame.kernel().live().len();
    assert_eq!(answers(&frame, some).len() + answers(&frame, negated).len(), live);
    let bosses = answers(&frame, boss);
    assert!(bosses.is_subset(&answers(&frame, negated)));
}

#[test]
fn a_definition_is_its_body_with_its_arguments() {
    let frame = led_frame();
    let defined = "(define seats-partner (roles (chair c) (chair p)) (partner c p))
        (define bidders-mate (roles (chair p)) (some (roles (chair b)) (declarer b) (seats-partner b p)))
        (query (roles (chair p)) (out p) (pattern (bidders-mate p)))";
    let written = "(query (roles (chair p) (chair b)) (out p) (pattern (declarer b) (partner b p)))";
    assert_eq!(answers(&frame, defined), answers(&frame, written));
    assert_eq!(answers(&frame, defined), [Answer(vec![Value::Chair(Seat::S2)])].into());
    // The expansion is printed in primitives and reads back as itself.
    let query: Query = defined.parse().unwrap();
    assert!(!query.to_string().contains("bidders-mate"));
    assert_eq!(query.to_string().parse::<Query>().unwrap(), query);
    // A definition's own roles are renamed apart from the caller's: here the
    // caller's b is the viewer, and the definition's b is a role of its own.
    let apart = "(define bidders-mate (roles (chair p)) (some (roles (chair b)) (declarer b) (partner b p)))
        (query (roles (chair b)) (out b) (pattern (viewer b) (bidders-mate b)))";
    assert!(answers(&frame, apart).is_empty(), "S0's partner is S2, not S0");
    // Negated, a definition with roles of its own is a none.
    let not_mate = "(define bidders-mate (roles (chair p)) (some (roles (chair b)) (declarer b) (partner b p)))
        (query (roles (chair c)) (out c) (pattern (not (bidders-mate c))))";
    assert_eq!(answers(&frame, not_mate).len(), 3);
    // A literal argument, a number parameter, and a definition with none.
    let calls = "(define worth (roles (domino d) (number n)) (count d n))
        (query (roles) (out) (pattern (worth 6-4 10) (not (worth 6-4 5))))";
    assert!(holds(&frame, calls));
}

#[test]
fn definitions_are_checked_where_they_are_called() {
    for (source, expected) in [
        ("(define partner (roles (chair c)) (viewer c)) (query (roles) (out))", "cannot name a definition"),
        (
            "(define d (roles (chair c)) (viewer c)) (define d (roles (chair c)) (viewer c)) (query (roles) (out))",
            "already defined",
        ),
        (
            "(define d (roles (chair c)) (viewer c)) (query (roles) (out) (pattern (d S0 S1)))",
            "takes 1 argument(s), got 2",
        ),
        (
            "(define d (roles (chair c)) (viewer c)) (query (roles (domino x)) (out) (pattern (d x)))",
            "parameter c is a chair, got a domino",
        ),
        ("(include nowhere) (query (roles) (out))", "no library nowhere"),
        ("(define d (roles (chair c)) (same c c)) (query (roles) (out))", "no roles of its own"),
        ("(query (roles) (out)) (query (roles) (out))", "only (include ...) and (define ...)"),
    ] {
        let error = source.parse::<Query>().unwrap_err().0;
        assert!(error.contains(expected), "{source}: {error}");
    }
    // Including a library twice is including it once.
    let twice = "(include book) (include book) (query (roles (chair c)) (out c) (pattern (bidder c)))";
    assert!(twice.parse::<Query>().is_ok());
}

#[test]
fn the_prelude_names_the_book_and_says_trump_once_for_every_declaration() {
    for rules in reference::Straight42::ALL {
        let p = rules.physics();
        let frame = opening(&p, tiles(&["0-0", "1-1", "2-0", "3-2", "5-5", "6-4", "6-6"]));
        let trumps = answers(&frame, "(include book) (query (roles (domino d)) (out d) (pattern (trump-tile d)))");
        let expected: Answers = match p.trump() {
            Some(t) => p.incidence(t).iter().map(|d| Answer(vec![Value::Domino(d)])).collect(),
            None => Answers::new(),
        };
        assert_eq!(trumps, expected, "{rules:?}");
        let top = answers(&frame, "(include book) (query (roles (domino d)) (out d) (pattern (top-trump d)))");
        assert_eq!(top.len(), usize::from(p.trump().is_some()), "{rules:?}");
        let seats = "(include book) (query (roles (chair b) (chair p) (chair l) (chair r)) (out b p l r)
            (pattern (bidder b) (bidders-partner p) (left-setter l) (right-setter r)))";
        let row = [Seat::S0, Seat::S2, Seat::S1, Seat::S3].map(Value::Chair).to_vec();
        assert_eq!(answers(&frame, seats), [Answer(row)].into());
    }
    // Sixes: the 6-6 tops the trumps, the 6-5 is next, and the bidder holds
    // two of them; on lead, his doubles walk and his other tiles are offs.
    let frame = opening(&physics("P6"), tiles(&["0-0", "1-1", "2-0", "3-2", "5-5", "6-4", "6-6"]));
    let one = |q: &str| answers(&frame, &format!("(include book) {q}"));
    let d = |s: &str| Answer(vec![Value::Domino(tile(s))]);
    assert_eq!(one("(query (roles (domino d)) (out d) (pattern (next-trump 6-6 d)))"), [d("6-5")].into());
    assert_eq!(one("(query (roles (domino d)) (out d) (pattern (lowest-trump-held d)))"), [d("6-4")].into());
    assert_eq!(one("(query (roles (domino d)) (out d) (pattern (walker d)))"), ["0-0", "1-1", "5-5"].map(d).into());
    assert_eq!(one("(query (roles (domino d)) (out d) (pattern (off d)))"), ["2-0", "3-2"].map(d).into());
    assert!(one("(query (roles) (out) (pattern (four-trumps-or-more)))").is_empty());
}

#[test]
fn every_error_in_program_text_names_its_line_and_column_with_a_caret() {
    let cases = [
        ("(query (roles) (out)\n  (pattern (live 6-6))))", "query:2:24: unexpected closing parenthesis", "^"),
        ("(query (roles) (out)\n  (pattern (live 6-6)", "query:2:3: this parenthesis is never closed", "^"),
        (
            "(query (roles (chair c)) (out c)\n  (pattern\n    (mystery c)))",
            "query:3:5: unregistered predicate mystery",
            "^^^^^^^^^^^",
        ),
        (
            "(query (roles (domino a)) (out)\n  (pattern (count a a)))",
            "query:2:12: count expects number, got domino",
            "^",
        ),
        ("(query (roles (chair c) (chair c)) (out))", "query:1:25: duplicate role c", "^^^^^^^^^"),
        (
            "(query (roles (chair c)) (out c) (pattern (holds S1 q*)))",
            "query:1:43: holds expects domino, got context",
            "^",
        ),
    ];
    for (source, location, caret) in cases {
        let error = source.parse::<Query>().and_then(|q| q.compile(&Registry::standard()).map(drop)).unwrap_err().0;
        assert!(error.starts_with(location), "{source}\n{error}");
        let lines: Vec<&str> = error.lines().collect();
        assert_eq!(lines.len(), 4, "{error}");
        assert!(lines[3].trim_start_matches(|c: char| c.is_ascii_digit() || c == ' ').starts_with('|'));
        assert!(lines[3].ends_with(caret), "{error}");
    }
    // A policy's rule that reads a world is refused at its rule.
    let policy = "(policy p\n  (fallback lowest-legal)\n  (rule peek (select answer)\n    (guard (query (roles (domino d)) (out d) (pattern (holds S1 d))))))";
    let error = PolicyProgram::parse(policy, "peek.silas")
        .and_then(|p| p.compile(&Registry::standard(), &physics("P6")).map(drop))
        .unwrap_err()
        .0;
    assert!(error.starts_with("peek.silas:3:3: rule peek uses the world predicate holds"), "{error}");
    // A call of a definition with the wrong sort points at the call.
    let call = "(include book)\n(query (roles (domino d)) (out)\n  (pattern (bidder d)))";
    assert!(call.parse::<Query>().unwrap_err().0.starts_with("query:3:12: bidder's parameter b is a chair"));
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
    for dir in ["examples", "lib"] {
        walk(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(dir), &mut files);
    }
    files.sort();
    files
}

#[test]
fn the_formatter_is_canonical_idempotent_and_keeps_every_file_as_it_reads() {
    let files = silas_files();
    assert!(files.len() > 50);
    for path in files {
        let name = path.display().to_string();
        let text = std::fs::read_to_string(&path).unwrap();
        let formatted = format(&text, &name).unwrap();
        // Every file of the repository is already in canonical form.
        assert_eq!(formatted, text, "{name} is not formatted (silas fmt)");
        assert_eq!(format(&formatted, &name).unwrap(), formatted, "{name}");
    }
    // Messy text: formatted once, it stays put, and it reads the same.
    let messy = "; a note\n(include   book)\n\n\n(query (roles (chair c)\n (domino d)) ; trailing\n (out d)\n(pattern (viewer c) (bidder c)\n\n (own-legal d)  (double d) (not (trump-tile d)) (none (roles (domino e)) (own-legal e) (in e q*) (beats e d q*) (not (tile e d)))))";
    let once = format(messy, "messy").unwrap();
    assert_eq!(format(&once, "messy").unwrap(), once);
    assert_eq!(once.parse::<Query>().unwrap(), messy.parse::<Query>().unwrap());
    assert!(once.contains("; trailing\n") && once.contains("\n\n(query") && !once.contains("\n\n\n"));
    assert!(once.lines().all(|l| l.len() <= 100), "{once}");
}
