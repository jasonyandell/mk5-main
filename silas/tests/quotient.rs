// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The decision quotient (`src/quotient.rs`) on courses whose answers are known: the
//! Lean witness that one block is not enough (`Stab/Residual.lean`), the local
//! hexagon of Jason's note (`Stab/Hexagon.lean`: four root outputs, thirteen
//! sequential classes), the shape and fork interfaces apart, and one real root of
//! the Gus contingent course against the Python reference (`lem/quotient.py`).

use std::collections::{BTreeMap, BTreeSet};

use silas::quotient::file::CourseFile;
use silas::quotient::{Block, Course, Decision, Interface};

type Labelled = Course<u32, &'static str>;

fn decision(menu: &[u32], best: &[u32], blocks: &[(u32, &'static str, usize)]) -> Decision<u32, &'static str> {
    Decision {
        menu: menu.to_vec(),
        best: best.to_vec(),
        blocks: blocks.iter().map(|&(play, seen, next)| Block { play, seen, next }).collect(),
    }
}

/// Two roots alike for one block; their second decisions accept different
/// observations; one final tile.
fn witness() -> Labelled {
    Course::new(vec![
        decision(&[1], &[1], &[(1, "a", 2)]),
        decision(&[1], &[1], &[(1, "a", 3)]),
        decision(&[2], &[2], &[(2, "b", 4)]),
        decision(&[2], &[2], &[(2, "c", 4)]),
        decision(&[3], &[3], &[]),
    ])
    .unwrap()
}

#[test]
fn one_block_is_not_enough_with_three_tiles() {
    let q = witness().quotient(Interface::Raw).unwrap();
    assert_eq!(q.ladder(), vec![3, 4, 5, 5]);
    assert_eq!(q.ladder_of(&[0, 1]), vec![1, 1, 2, 2]);
    assert_eq!(q.states, 5);
    // Without tile identity or observation labels every output is a one-tile menu, the
    // roots are one shape, and only depth separates: root, second, final.
    let s = witness().quotient(Interface::Shape).unwrap();
    assert_eq!((s.states, s.ladder()), (3, vec![1, 2, 3, 3]));
}

#[test]
fn fork_forgets_how_many_observations_lead_to_a_class() {
    let c: Labelled = Course::new(vec![
        decision(&[1], &[1], &[(1, "a", 2), (1, "b", 2)]),
        decision(&[1], &[1], &[(1, "a", 2)]),
        decision(&[3], &[3], &[]),
    ])
    .unwrap();
    assert_eq!(c.quotient(Interface::Shape).unwrap().states, 3);
    assert_eq!(c.quotient(Interface::Fork).unwrap().states, 2);
    assert_eq!(c.quotient(Interface::Raw).unwrap().states, 3);
}

#[test]
fn malformed_courses_are_refused() {
    let bad = |d: Vec<Decision<u32, &'static str>>, why: &str| {
        let e = Course::new(d).unwrap_err().0;
        assert!(e.contains(why), "{e}");
    };
    bad(vec![decision(&[2, 1], &[1], &[])], "strictly ascending");
    bad(vec![decision(&[1, 2], &[3], &[])], "leaves the menu");
    bad(vec![decision(&[1], &[1], &[(2, "a", 0)])], "outside the menu");
    bad(vec![decision(&[1], &[1], &[(1, "a", 5)])], "points nowhere");
    bad(vec![decision(&[1], &[1], &[(1, "a", 1), (1, "a", 1)]), decision(&[1], &[1], &[])], "share a label");
    bad(vec![decision(&[1], &[1], &[(1, "a", 1)]), decision(&[1], &[1], &[(1, "a", 0)])], "cycle");
}

/// The note's local course: tiles 3-1, 4-1, 5-1; an order's value is E[A], E[B] or
/// E[C] by the note's table, with `E[A] = x`, `E[B] = 0`, `E[C] = y`.
fn hexagon_course(points: &[(i64, i64)]) -> (Course<u32, &'static str>, Vec<usize>) {
    const T: [u32; 3] = [31, 41, 51];
    let vec_of = |first: u32, second: u32, x: i64, y: i64| match (first, second) {
        (31, 41) | (51, 41) => x, // A
        (31, 51) | (41, 51) => 0, // B
        _ => y,                   // C: (41, 31), (51, 31)
    };
    let argmax = |menu: &[u32], q: &[i64]| -> Vec<u32> {
        let top = *q.iter().max().unwrap();
        menu.iter().zip(q).filter(|(_, v)| **v == top).map(|(a, _)| *a).collect()
    };
    let mut ds = Vec::new();
    // The three final tiles.
    for t in T {
        ds.push(decision(&[t], &[t], &[]));
    }
    let mut roots = Vec::new();
    for &(x, y) in points {
        let mut root_blocks = Vec::new();
        let mut root_q = Vec::new();
        for first in T {
            let menu: Vec<u32> = T.iter().copied().filter(|&t| t != first).collect();
            let q: Vec<i64> = menu.iter().map(|&s| vec_of(first, s, x, y)).collect();
            let blocks: Vec<(u32, &'static str, usize)> = menu
                .iter()
                .map(|&s| {
                    let last = T.iter().position(|&t| t != first && t != s).unwrap();
                    (s, "", last)
                })
                .collect();
            ds.push(decision(&menu, &argmax(&menu, &q), &blocks));
            root_blocks.push((first, "", ds.len() - 1));
            root_q.push(*q.iter().max().unwrap());
        }
        roots.push(ds.len());
        ds.push(decision(&T, &argmax(&T, &root_q), &root_blocks));
    }
    (Course::new(ds).unwrap(), roots)
}

#[test]
fn the_hexagon_has_four_root_outputs_and_thirteen_sequential_classes() {
    // Every lattice point of the hexagon scaled by four: |x|, |y|, |x - y| <= 4.
    let points: Vec<(i64, i64)> = (-4..=4)
        .flat_map(|x| (-4..=4).map(move |y| (x, y)))
        .filter(|&(x, y): &(i64, i64)| (x - y).abs() <= 4)
        .collect();
    assert_eq!(points.len(), 61);
    let (course, roots) = hexagon_course(&points);
    let q = course.quotient(Interface::Raw).unwrap();
    assert_eq!(q.ladder_of(&roots), vec![4, 13, 13]);
    // Two roots share a class exactly when (sign x, sign y, sign (x - y)) agree.
    let pattern = |(x, y): (i64, i64)| (x.signum(), y.signum(), (x - y).signum());
    let mut by_class: BTreeMap<usize, BTreeSet<(i64, i64, i64)>> = BTreeMap::new();
    for (r, &p) in roots.iter().zip(&points) {
        by_class.entry(q.class[*r]).or_default().insert(pattern(p));
    }
    assert!(by_class.values().all(|s| s.len() == 1));
    assert_eq!(by_class.values().flatten().collect::<BTreeSet<_>>().len(), 13);
}

#[test]
fn one_real_root_matches_the_python_reference() {
    // Root test-0007 of the maskbreadth study (teacher.json sha256 1f467219…),
    // exported by `lem/quotient.py --export`; the counts are that tool's.
    let file = CourseFile::from_json(include_str!("data/quotient-test-0007.json")).unwrap();
    let r = file.report().unwrap();
    assert_eq!(r["nodes"], 52);
    assert_eq!(r["menu_sizes"], serde_json::json!({"1": 42, "2": 9, "3": 1}));
    assert_eq!(r["discretionary_unsettled"]["nodes"], 7);
    assert_eq!(r["discretionary_unsettled"]["nonflat"], 3);
    let expect = [("raw", 13, vec![9, 13, 13]), ("shape", 5, vec![4, 5, 5]), ("fork", 4, vec![4, 4])];
    for (name, states, ladder) in expect {
        let i = &r["interfaces"][name];
        assert_eq!(i["states"], states, "{name}");
        assert_eq!(i["ladder_by_block_depth"], serde_json::json!(ladder), "{name}");
    }
    assert_eq!(r["interfaces"]["raw"]["states_by_stage"], serde_json::json!({"root": 1, "second": 9, "third": 3}));
    assert_eq!(
        r["decision_classes"],
        serde_json::json!({
            "2": {"nodes": 6, "distinct_weak_orders": 2, "weak_orders": {"00": 4, "10": 2},
                  "tier_types": {"1/1": 2, "2": 4}},
            "3": {"nodes": 1, "distinct_weak_orders": 1, "weak_orders": {"100": 1}, "tier_types": {"2/1": 1}}})
    );
}

#[test]
fn a_broken_bellman_row_is_refused() {
    let text = include_str!("data/quotient-test-0007.json");
    let mut v: serde_json::Value = serde_json::from_str(text).unwrap();
    // The first decision with a real menu claims a value its blocks do not deliver.
    let nodes = v["nodes"].as_array_mut().unwrap();
    let row = nodes.iter_mut().find(|n| n["menu"].as_array().unwrap().len() > 1).unwrap();
    let k = row["q"].as_array().unwrap().len() - 1;
    let q = row["q"][k].as_str().unwrap().parse::<num_rational::BigRational>().unwrap();
    row["q"][k] = serde_json::json!((q + num_rational::BigRational::new(1.into(), 1000.into())).to_string());
    let best = row["menu"][k].clone();
    row["best"] = serde_json::json!([best]);
    let e = CourseFile::from_json(&v.to_string()).err().unwrap().0;
    assert!(e.contains("Bellman"), "{e}");
}
