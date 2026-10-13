//! The core every file of the crate reads to, pinned. A Silas text is read
//! to its core: definitions expanded, sugar rewritten, roles and outputs
//! spelled out. The core prints as itself (`Display`), so equal prints are
//! equal programs. The pins were taken from the files as they stood before the
//! surface sugar (2026-10-13), and a file or prelude added since is pinned as
//! first written (the auction's: `examples/book/bidding/`, `lib/bidding.silas`);
//! any rewrite of a file, or of a prelude, into the sugar must read to the same
//! core, and these pins prove it. The auction's files were re-pinned once, when
//! the aggregate (`card`, `points`) replaced their literal bounds: a new core by
//! design, whose meaning `tests/bidding.rs` holds to the old on every hand.
//! `crisis.silas` was re-pinned then too, its meaning changed on purpose: the
//! aggregate states the "most tiles out" its encoding had left as a gap.
//!
//! Each prelude definition is pinned by the core of a query that calls it on
//! roles of its parameters' sorts, so a definition no file calls is pinned too.
//!
//! On a mismatch the test prints the whole table, to compare by eye. A pin
//! changes only when a file's meaning is meant to change.

use std::fmt::Write;
use std::path::PathBuf;

use silas::*;

/// FNV-1a over a core's text: a pin, not a security property.
fn digest(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
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

/// The preludes' definitions, each as a query calling it: `(define name
/// (roles (sort p) ...)` heads every one, on one line. Each is named by its
/// library's file.
fn prelude_calls() -> Vec<(String, String)> {
    [("book", include_str!("../lib/book.silas")), ("bidding", include_str!("../lib/bidding.silas"))]
        .into_iter()
        .flat_map(|(library, text)| library_calls(library, text))
        .collect()
}

fn library_calls(library: &str, text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|line| line.strip_prefix("(define "))
        .map(|rest| {
            let (name, params) = rest.split_once(" (roles").unwrap();
            let params: Vec<(&str, &str)> = params
                .split('(')
                .filter_map(|p| p.split(')').next())
                .filter_map(|p| p.trim().split_once(' '))
                .collect();
            let args: Vec<String> = params
                .iter()
                .map(|(sort, p)| match *sort {
                    "number" => "5".to_owned(),
                    "team" => "T0".to_owned(),
                    _ => format!("arg-{p}"),
                })
                .collect();
            let roles: Vec<String> = params
                .iter()
                .filter(|(sort, _)| !matches!(*sort, "number" | "team"))
                .map(|(sort, p)| format!("({sort} arg-{p})"))
                .collect();
            let query = format!(
                "(include {library}) (query (roles {}) (out) (pattern ({name} {})))",
                roles.join(" "),
                args.join(" ")
            );
            (format!("lib/{library}.silas:{name}"), query)
        })
        .collect()
}

fn table() -> String {
    let mut out = String::new();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for path in silas_files() {
        let name = path.strip_prefix(&root).unwrap().display().to_string();
        let text = std::fs::read_to_string(&path).unwrap();
        // A file of comments alone (an odds claim with no query) reads to nothing.
        let core = match Program::parse(&text, &name) {
            Ok(Program::Query(q)) => q.to_string(),
            Ok(Program::Policy(p)) => p.to_string(),
            Ok(Program::Dominance(d)) => d.to_string(),
            Err(e) if e.0.ends_with("found nothing") => String::new(),
            Err(e) => panic!("{e}"),
        };
        writeln!(out, "{name} {:016x}", digest(&core)).unwrap();
    }
    for (name, query) in prelude_calls() {
        let core = query.parse::<Query>().unwrap().to_string();
        writeln!(out, "{name} {:016x}", digest(&core)).unwrap();
    }
    out
}

#[test]
fn every_file_and_every_prelude_definition_reads_to_its_pinned_core() {
    let now = table();
    let pinned = include_str!("data/core-pins.txt");
    assert!(now.lines().count() > 140, "{now}");
    assert_eq!(now, pinned, "the cores changed; the table now:\n{now}");
}
