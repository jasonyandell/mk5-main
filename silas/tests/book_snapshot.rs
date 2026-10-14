// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! What the study's gate encodings allow, pinned: they are the record of what
//! the 2026-10-12 study priced. Every policy of `examples/book/gates/` is
//! asked, through [`CompiledPolicy::allowed`], for its deciding rule and the
//! tiles that rule allows: at every point of the eight full deals' narrated
//! lines, from the seat to play, with the bidder in each of the four chairs;
//! and at every point of 1,600 seeded random deals under all nine straight
//! declarations (the last 1,200 biased toward the book's shapes: trump leads,
//! doubles, trumping in, four-trump bidders).
//!
//! The pins were taken before the rules were rewritten with the book prelude
//! (`lib/book.silas`, 2026-10-13), from the rules as they stood, each written
//! out in primitives; the rewrite had to reproduce them exactly. The narrated
//! points where a rule fires are listed in `tests/data/book-allowed.txt`;
//! the rest, and the random deals, are pinned by a digest of the full table.
mod common;

use std::fmt::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

use common::fixture::book;
use common::*;
use silas::*;

/// FNV-1a over the table's text: a pin, not a security property.
fn digest(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}

/// Every policy of the book, by file name, in name order.
fn policies() -> &'static [(String, PolicyProgram)] {
    static POLICIES: OnceLock<Vec<(String, PolicyProgram)>> = OnceLock::new();
    POLICIES.get_or_init(read_policies)
}

fn read_policies() -> Vec<(String, PolicyProgram)> {
    let dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/book/gates"));
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "silas"))
        .collect();
    files.sort();
    files
        .iter()
        .filter_map(|f| {
            let text = std::fs::read_to_string(f).unwrap();
            let name = f.file_name().unwrap().to_string_lossy().into_owned();
            match Program::parse(&text, &name).unwrap() {
                Program::Policy(policy) => Some((name, policy)),
                Program::Query(_) | Program::Dominance(_) => None,
            }
        })
        .collect()
}

fn compiled(physics: &Physics) -> Vec<(String, CompiledPolicy)> {
    policies().iter().map(|(name, p)| (name.clone(), p.compile(&Registry::standard(), physics).unwrap())).collect()
}

fn allowed(policy: &CompiledPolicy, frame: &Frame) -> String {
    let allowed = policy.allowed(frame, &mut budget()).unwrap();
    match allowed.provenance {
        DecisionProvenance::RelationalRule { name } => format!("{name} {:?}", allowed.tiles),
        DecisionProvenance::Fallback => "fallback".into(),
        DecisionProvenance::Exact => panic!("the book's rules have no exact table"),
    }
}

/// The seat to play after `plays`, `declarer` leading the first trick.
fn next_actor(physics: &Physics, declarer: Seat, plays: &[Domino]) -> Seat {
    let seats = actors(physics, declarer, plays);
    match plays.len() % 4 {
        0 => {
            let last = plays.len().checked_sub(4).map(|start| <[Domino; 4]>::try_from(&plays[start..]).unwrap());
            last.map_or(declarer, |trick| physics.trick_winner(seats[plays.len() - 4], trick))
        }
        k => seats[plays.len() - k].plus(k),
    }
}

/// The narrated table: one line per point, seating and policy.
fn narrated() -> String {
    let mut out = String::new();
    let b = book();
    for id in ["hand-28", "hand-29", "hand-30", "hand-31", "hand-32", "hand-33", "hand-34", "hand-35"] {
        let example = b.example(id);
        let physics = example.physics();
        let policies = compiled(&physics);
        let deal = example.deal();
        let plays = b.line(&format!("{id}/prose")).plays();
        for n in 0..plays.len() {
            let prefix = &plays[..n];
            for declarer in Seat::ALL {
                let viewer = next_actor(&physics, declarer, prefix);
                let role = (viewer.index() + 4 - declarer.index()) % 4;
                let contract = contract(&physics, declarer, example.bid.number());
                let frame = framed(&contract, viewer, deal[role], prefix.to_vec()).unwrap();
                for (name, policy) in &policies {
                    writeln!(out, "{id} {n} {declarer} role{role} {name} {}", allowed(policy, &frame)).unwrap();
                }
            }
        }
    }
    out
}

/// The random table: 1,600 seeded deals played out.
fn random() -> String {
    let declarations = reference::Straight42::ALL.map(reference::Straight42::physics);
    let compiled_under: std::collections::BTreeMap<String, Vec<(String, CompiledPolicy)>> =
        declarations.iter().map(|p| (p.identity().to_owned(), compiled(p))).collect();
    let mut out = String::new();
    let mut rng = 0x5eed_u64;
    let mut next = |n: usize| {
        rng = rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) % n as u64) as usize
    };
    for game in 0..1600 {
        let biased = game >= 400;
        let physics = &declarations[next(9)];
        let policies = &compiled_under[physics.identity()];
        let mut deck = Domino::ALL.to_vec();
        for i in (1..28).rev() {
            deck.swap(i, next(i + 1));
        }
        let declarer = Seat::ALL[next(4)];
        let trumps_dealt = |deck: &[Domino]| {
            deck[7 * declarer.index()..7 * declarer.index() + 7]
                .iter()
                .filter(|d| physics.follows(**d, Context::Called))
                .count()
        };
        while game >= 800 && trumps_dealt(&deck) < 4 && !physics.incidence(Context::Called).is_empty() {
            for i in (1..28).rev() {
                deck.swap(i, next(i + 1));
            }
        }
        let deal: [DominoSet; 4] = std::array::from_fn(|s| deck[7 * s..7 * s + 7].iter().copied().collect());
        let bid = 30 + next(13) as u32;
        let mut hands = deal;
        let mut plays: Vec<Domino> = Vec::new();
        for _ in 0..28 {
            let viewer = next_actor(physics, declarer, &plays);
            let contract = contract(physics, declarer, bid);
            let frame = framed(&contract, viewer, deal[viewer.index()], plays.clone()).unwrap();
            for (name, policy) in policies {
                writeln!(out, "game{game} {} {} {name} {}", plays.len(), viewer, allowed(policy, &frame)).unwrap();
            }
            let legal = physics.legal_plays(hands[viewer.index()], frame.lead());
            let trumps = legal.intersection(physics.incidence(Context::Called));
            let doubles: DominoSet = legal.iter().filter(|d| d.is_double()).collect();
            let pick = if !biased {
                legal
            } else if frame.lead().is_none() && plays.len() < 4 && !trumps.is_empty() {
                trumps
            } else if frame.lead().is_none() && !doubles.is_empty() && next(3) > 0 {
                doubles
            } else if frame.lead().is_some() && !trumps.is_empty() && next(2) > 0 {
                trumps
            } else {
                legal
            };
            let tile = pick.iter().nth(next(pick.len())).unwrap();
            hands[viewer.index()].remove(tile);
            plays.push(tile);
        }
    }
    out
}

#[test]
fn the_book_rules_allow_on_the_narrated_deals_what_they_allowed_before_the_prelude() {
    let table = narrated();
    assert_eq!(table.lines().count(), 13_760);
    let firing: Vec<&str> = table.lines().filter(|l| !l.ends_with(" fallback")).collect();
    let pinned: Vec<&str> = include_str!("data/book-allowed.txt").lines().collect();
    assert_eq!(firing.len(), pinned.len());
    for (now, then) in firing.iter().zip(&pinned) {
        assert_eq!(now, then);
    }
    assert_eq!(digest(&table), 0x7da1_5588_1bfc_d361);
}

#[test]
fn the_book_rules_allow_on_random_deals_what_they_allowed_before_the_prelude() {
    let table = random();
    assert_eq!(table.lines().count(), 896_000);
    assert_eq!(digest(&table), 0xc788_6d6a_eadf_0c69);
}
