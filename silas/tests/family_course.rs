// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The family course file (`silas-family-course-v1`, `silas family --courses`), the form
//! `lem/endgame.py` writes a Walt endgame in: read into a `Family`, solved exactly, and
//! refused when it is not a family.
use serde_json::Value;
use silas::family::file::FamilyFile;

/// Four worlds. Play 1 ends it, 3 of them good. Play 2 shows one of two tiles; behind
/// each, two worlds, and the right reply wins both.
fn course(stop: &str) -> String {
    format!(
        r#"{{"schema": "silas-family-course-v1", "id": "toy", "root": 0, "choices": [
            {{"menu": [1, 2], "mass": "4", "payoff": [{stop}, "0"],
              "blocks": [{{"play": 2, "seen": [[1, 5]], "next": 1}}, {{"play": 2, "seen": [[1, 6]], "next": 2}}]}},
            {{"menu": [3, 4], "mass": "2", "payoff": ["2", "0"], "blocks": []}},
            {{"menu": [3, 4], "mass": "2", "payoff": ["0", "2"], "blocks": []}}]}}"#
    )
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect()
}

#[test]
fn a_constant_course_has_one_class_and_the_exact_root_values() {
    let f = FamilyFile::from_json(&course(r#""3""#)).unwrap();
    assert!(f.constant);
    let r = f.report().unwrap();
    assert_eq!(r["id"], "toy");
    assert_eq!(r["choices"], 3);
    assert_eq!(r["reachable"], 3);
    let classes = r["classes"].as_array().unwrap();
    assert_eq!(classes.len(), 1);
    assert_eq!(strings(&classes[0]["value"]), ["3/4", "1"]);
    assert_eq!(classes[0]["best"], serde_json::json!([2]));
    for reach in ["every", "path", "plan"] {
        assert!(r["quotient"][reach]["raw"]["states"].as_u64().unwrap() >= 1, "{reach}");
    }
}

#[test]
fn a_parametric_payoff_crosses_at_its_threshold() {
    // Play 1 is worth 2 + 4p of 4: below p = 1/2 play 2 is best, above it play 1.
    let f = FamilyFile::from_json(&course(r#"["2", "6"]"#)).unwrap();
    assert!(!f.constant);
    let r = f.report().unwrap();
    assert_eq!(strings(&r["thresholds"]), ["1/2"]);
    let best: Vec<&Value> = r["classes"].as_array().unwrap().iter().map(|c| &c["best"]).collect();
    assert!(best.contains(&&serde_json::json!([2])));
    assert!(best.contains(&&serde_json::json!([1])));
}

#[test]
fn what_is_not_a_family_course_is_refused() {
    let wrong_schema = course(r#""3""#).replace("silas-family-course-v1", "silas-course-v1");
    assert!(FamilyFile::from_json(&wrong_schema).is_err());
    let no_root = course(r#""3""#).replace(r#""root": 0"#, r#""root": 9"#);
    assert!(FamilyFile::from_json(&no_root).is_err());
    // The blocks under play 2 must partition its mass: 2 + 3 is not 4.
    let leaky = course(r#""3""#).replacen(r#""mass": "2""#, r#""mass": "3""#, 1);
    assert!(FamilyFile::from_json(&leaky).is_err());
    let not_rational = course(r#""three""#);
    assert!(FamilyFile::from_json(&not_rational).is_err());
}
