//! The `silas` command on the book's hands.

use std::process::Command;

fn silas(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_silas"))
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("the silas command runs");
    let text = String::from_utf8(out.stdout).unwrap() + &String::from_utf8(out.stderr).unwrap();
    (out.status.success(), text)
}

#[test]
fn eval_prints_the_fired_rule_its_witnesses_and_the_allowed_set() {
    let policy = "examples/book/gates/c01a-top-trump-for-the-donation.silas";
    let (ok, out) = silas(&["eval", "--hand", "hand-32", "--trick", "2", "--seat", "bidder", "--policy", policy]);
    assert!(ok, "{out}");
    assert!(out.contains("hand-32: trick 2, the bidder (S0) to play"), "{out}");
    assert!(out.contains("fired: rule lead-the-top-trump"), "{out}");
    assert!(out.contains("6-0  <=  b=S0 p=S2 top=6-0"), "{out}");
    assert!(out.contains("allowed: 6-0\n  plays: 6-0"), "{out}");
    let (ok, out) = silas(&["eval", "--hand", "hand-30", "--trick", "3", "--seat", "left", "--policy", policy]);
    assert!(ok && out.contains("fired: no rule; the fallback"), "{out}");
}

#[test]
fn query_prints_the_exact_answer_with_its_authority() {
    let (ok, out) = silas(&["query", "examples/six-five-guarded.silas", "--hand", "hand-34"]);
    assert!(ok, "{out}");
    assert!(out.contains("399072960 worlds, as 27 points resolving 5-1 5-3 6-5"), "{out}");
    assert!(out.contains("P(answer) = 33/95 (exact)"), "{out}");
    assert!(out.contains("bids 84: 42 points for 2 mark(s)"), "{out}");
}

#[test]
fn query_states_the_lay_down_and_the_claim_exactly() {
    let (ok, out) = silas(&["query", "examples/book/lay-down.silas", "--hand", "hand-21"]);
    assert!(ok && out.contains("399072960 worlds, as 1 points resolving nothing"), "{out}");
    assert!(out.contains("P(answer) = 1 (exact)"), "{out}");
    let (ok, out) = silas(&["query", "examples/book/lay-down.silas", "--hand", "hand-34"]);
    assert!(ok && out.contains("P(answer) = 0 (exact)"), "{out}");
    // Hand 32's claim is in the narrated line's tail: the 5-4 led, the bidder to play.
    let args = ["--hand", "hand-32", "--trick", "4", "--seat", "bidder"];
    let (ok, out) = silas(&[&["query", "examples/book/claim-the-rest.silas"][..], &args].concat());
    assert!(ok && out.contains("S3:5-4\n") && out.contains("P(answer) = 1 (exact)"), "{out}");
    // The double-dummy bound reads every hidden tile: refused above the cap.
    std::fs::write(
        std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("dd.silas"),
        "(query (roles)\n  (out)\n  (pattern (dd-make)))\n",
    )
    .unwrap();
    let dd = concat!(env!("CARGO_TARGET_TMPDIR"), "/dd.silas");
    let (ok, out) = silas(&["query", dd, "--hand", "hand-21"]);
    assert!(!ok && out.contains("exceeds its cap of 100000 points"), "{out}");
}

#[test]
fn query_decides_a_dominance_exactly_without_a_hand() {
    let (ok, out) = silas(&["query", "examples/book/cross-game/last-mark-points.silas"]);
    assert!(ok && out.contains("10500 states, 43 outcomes"), "{out}");
    assert!(out.contains("bid alone (S-0082-a)") && out.contains("  holds (exact)"), "{out}");
    let (ok, out) = silas(&["query", "examples/book/cross-game/two-marks-out.silas"]);
    assert!(ok, "{out}");
    assert!(
        out.contains(
            "fails (exact): from 5-0, the hand ending 42 to 0: the first leaves 6-0 and the second 7-0 (won by T0)"
        ),
        "{out}"
    );
}

#[test]
fn check_compiles_every_example_and_fmt_checks_and_rewrites() {
    let (ok, out) = silas(&["check"]);
    assert!(ok && out.contains(" 0 failed"), "{out}");
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("messy.silas");
    std::fs::write(&path, "(query (roles)   (out))").unwrap();
    let file = path.to_str().unwrap();
    let (ok, out) = silas(&["fmt", "--check", file]);
    assert!(!ok && out.contains("not formatted"), "{out}");
    let (ok, _) = silas(&["fmt", file]);
    assert!(ok);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "(query (roles)\n  (out))\n");
    let (ok, out) = silas(&["fmt", "--check", file]);
    assert!(ok, "{out}");
    // An error in a file names its line and column.
    std::fs::write(&path, "(query (roles)\n  (out) (pattern (mystery)))").unwrap();
    let (ok, out) = silas(&["check", path.parent().unwrap().to_str().unwrap()]);
    assert!(!ok && out.contains("messy.silas:2:18: unregistered predicate mystery"), "{out}");
}
