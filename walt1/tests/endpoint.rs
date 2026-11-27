// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The `walt1 play` contract (WALT.md §6, §8.1): exact fields, legality, purity, errors.

use walt::{decide, handle, parse};

const HEAD: &str = "walt1 play\ndecl 6\nbid 35\nbidder 0\nn 40\nn0 8\nseed 1000\n";

fn req(seat: u8, hand: &str, plays: &str) -> String {
    format!("{HEAD}seat {seat}\nhand {hand}\nplays {plays}\n")
}

fn field<'a>(json: &'a str, key: &str) -> &'a str {
    let at = json.find(&format!("\"{key}\":")).unwrap_or_else(|| panic!("{key} in {json}")) + key.len() + 3;
    let rest = &json[at..];
    let mut depth = 0;
    for (i, c) in rest.char_indices() {
        match c {
            '[' => depth += 1,
            ']' if depth > 0 => depth -= 1,
            ',' | '}' | ']' if depth == 0 => return &rest[..i],
            _ => {}
        }
    }
    rest
}

/// The four reference examples: the exact fields match and the options are the legal tiles.
#[test]
fn reference_examples_exact_fields() {
    let cases = [
        (0, "1 6 7 8 18 19 26", "", "0", "[0,0]", vec![1, 6, 7, 8, 18, 19, 26]),
        (1, "3 4 12 16 22 23 24", "0 19 1 16 2 17 3 15 0 8", "0", "[6,0]", vec![3, 4, 12, 22, 23, 24]),
        (
            1,
            "3 4 12 16 22 23 24",
            "0 19 1 16 2 17 3 15 0 8 1 3 2 13 3 9 3 5 0 26 1 4 2 21 0 6",
            "0",
            "[7,6]",
            vec![12, 22, 23, 24],
        ),
        (
            3,
            "5 9 11 15 20 25 27",
            "0 19 1 16 2 17 3 15 0 8 1 3 2 13 3 9 3 5 0 26 1 4 2 21 0 6 1 12 2 0 3 11 0 18 1 23 2 2 3 20 1 22 2 10",
            "1",
            "[13,17]",
            vec![25, 27],
        ),
    ];
    for (seat, hand, plays, leader, points, legal) in cases {
        let text = req(seat, hand, plays);
        let out = handle(&text);
        assert_eq!(field(&out, "leader"), leader, "{out}");
        assert_eq!(field(&out, "points"), points, "{out}");
        assert_eq!(field(&out, "forced"), "false");
        assert_eq!(field(&out, "raced"), "false");
        let d = decide(&parse(&text).unwrap()).unwrap();
        assert_eq!(d.opts.iter().map(|o| o.tile).collect::<Vec<_>>(), legal);
        assert!(legal.contains(&d.choice));
        // Purity: the same text, the same line.
        assert_eq!(handle(&text), out);
    }
}

/// The last example is already lost for the bidders (17 > 42 − 35): every tile is worth 0
/// and the tie that survives refinement goes to the lowest id.
#[test]
fn decided_hand_ties_to_lowest_id() {
    let out = handle(&req(
        3,
        "5 9 11 15 20 25 27",
        "0 19 1 16 2 17 3 15 0 8 1 3 2 13 3 9 3 5 0 26 1 4 2 21 0 6 1 12 2 0 3 11 0 18 1 23 2 2 3 20 1 22 2 10",
    ));
    assert_eq!(field(&out, "choice"), "25");
    assert_eq!(field(&out, "opts"), "[[25,0],[27,0]]");
}

#[test]
fn forced_play_is_not_evaluated() {
    let out = handle(&req(1, "3 4 12 16 22 23 24", "0 19"));
    assert_eq!(
        out,
        r#"{"v":1,"kind":"play","choice":16,"forced":true,"raced":false,"opts":[],"leader":0,"points":[0,0]}"#
    );
}

#[test]
fn defaults_and_other_declarations() {
    for decl in [0, 3, 7, 9] {
        let text = format!("walt1 play\ndecl {decl}\nbid 30\nseat 2\nbidder 2\nhand 0 5 9 14 20 21 27\n");
        let r = parse(&text).unwrap();
        assert_eq!((r.n, r.n0, r.seed), (40, 8, 0xB7E1_5162_8AED_2A6B));
        let d = decide(&r).unwrap();
        assert_eq!(d.opts.len(), 7);
        assert!(d.opts.iter().all(|o| o.makes <= o.worlds && o.bp() <= 10000));
        // A refined tile carries a larger sample; an unrefined one keeps n.
        assert!(d.opts.iter().all(|o| [40, 160, 640].contains(&o.worlds)));
    }
}

/// §5.4: when the top is tied after the first round, the tied tiles are re-valued on 160 and
/// then 640 fresh worlds, and the others keep their 40-world values.
#[test]
fn ties_are_refined_and_replaced() {
    let mut refined = 0;
    for seed in 0..12 {
        let text = format!(
            "walt1 play\ndecl 6\nbid 35\nseat 0\nbidder 0\nhand 1 6 7 8 18 19 26\nplays 0 19 1 16 2 17 3 15\nseed {seed}\n"
        );
        let d = decide(&parse(&text).unwrap()).unwrap();
        let max = d.opts.iter().map(|o| o.worlds).max().unwrap();
        if max > 40 {
            refined += 1;
            assert!(d.opts.iter().filter(|o| o.worlds > 40).count() >= 2 || max == 160);
        }
        // The choice is a best tile of the final values, lowest id among equals.
        let best = d.opts.iter().fold(d.opts[0], |b, o| {
            if o.makes as u64 * b.worlds as u64 > b.makes as u64 * o.worlds as u64 {
                *o
            } else {
                b
            }
        });
        assert_eq!(d.choice, best.tile);
    }
    assert!(refined > 0, "no seed of twelve produced a top tie at n = 40");
}

#[test]
fn bad_input_is_an_error_line() {
    let bad = [
        "walt2 play\n".to_string(),
        req(0, "1 6 7 8 18 19", ""),                    // six tiles
        req(0, "1 6 7 8 18 19 19", ""),                 // a repeat
        req(1, "3 4 12 16 22 23 24", ""),               // not seat 1's turn
        req(1, "3 4 12 16 22 23 24", "0 19 1 3"),       // seat 1 could follow with 16
        req(1, "3 4 12 16 22 23 24", "0 3"),            // seat 0 plays a tile seat 1 holds
        req(2, "0 2 10 13 14 17 21", "0 19 0 16"),      // out of turn
        req(2, "0 2 10 13 14 17 21", "0 19 1 19"),      // tile reused
        req(0, "1 6 7 8 18 19 28", ""),                 // not a tile
        req(0, "1 6 7 8 18 19 26", "0"),                // half a pair
        format!("{}bogus 1\n", req(0, "1 6 7 8 18 19 26", "")),
        "walt1 play\ndecl 8\nbid 35\nseat 0\nbidder 0\nhand 1 6 7 8 18 19 26\n".to_string(),
        "walt1 play\ndecl 6\nbid 29\nseat 0\nbidder 0\nhand 1 6 7 8 18 19 26\n".to_string(),
        "walt1 play\ndecl 6\nbid 35\nseat 0\nbidder 0\nhand 1 6 7 8 18 19 -26\n".to_string(),
        // Seat 1 trumps a five lead, then is recorded following fives: a contradiction.
        req(0, "1 6 7 8 18 19 26", "0 19 1 21 2 17 3 15 1 22 2 10 3 9 0 26 0 18 1 16"),
    ];
    for text in bad {
        let out = handle(&text);
        assert!(out.starts_with(r#"{"v":1,"error":""#), "{text:?} gave {out}");
    }
}

/// The fixture the wasm is held to (`tests/walt.test.js`): this build's own answers on the
/// corpus's first hand. A change to any sampled value shows up here before it ships.
#[test]
fn fixture_answers_are_reproduced() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/hand0.jsonl")).unwrap();
    let unquote = |s: &str| s.replace("\\n", "\n").replace("\\\"", "\"");
    let mut rows = 0;
    for line in text.lines() {
        let req = &line[line.find("\"req\":\"").unwrap() + 7..line.find("\",\"res\":\"").unwrap()];
        let res = &line[line.find("\"res\":\"").unwrap() + 7..line.len() - 2];
        assert_eq!(handle(&unquote(req)), unquote(res));
        rows += 1;
    }
    assert_eq!(rows, 28);
}
