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
fn bid_prints_the_call_its_rule_and_the_declaration() {
    let (ok, out) = silas(&["bid", "examples/book/bidding/risk-budget.silas", "--hand", "hand-1"]);
    assert!(ok, "{out}");
    assert!(out.contains("hand-1: the bidder (S0) to call; S3 shook: no calls; marks to 7"), "{out}");
    assert!(out.contains("calls 36, by rule bid-36"), "{out}");
    assert!(
        out.contains("if the rest pass, declares straight42/P6, by rule suit-at-risk-6\n    q6  <=  t=q6"),
        "{out}"
    );
    // Over another seat's 35 the partner of Hand 32 passes, by the fallback.
    let args = ["--hand", "hand-32", "--seat", "partner", "--calls", "35", "--scoring", "points"];
    let (ok, out) = silas(&[&["bid", "examples/book/bidding/bid-only-enough.silas"][..], &args].concat());
    assert!(ok && out.contains("S0 shook: S1 35; points to 250"), "{out}");
    assert!(out.contains("legal: pass 36 37") && out.contains("calls pass, by the fallback"), "{out}");
    let (ok, out) = silas(&["bid", "examples/book/bidding/risk-budget.silas", "--hand", "hand-1", "--seat", "left"]);
    assert!(!ok && out.contains("gives only the bidder's hand"), "{out}");
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

#[test]
fn eval_says_why_each_earlier_rule_passed_and_takes_the_policy_as_its_file() {
    let args = ["--hand", "hand-28", "--trick", "3", "--seat", "left"];
    let (ok, out) = silas(&[&["eval", "examples/book/pounce.silas"][..], &args].concat());
    assert!(ok, "{out}");
    assert!(
        out.contains("passed: rule no-count-on-a-lost-trick: stops at (some (roles his) (trick-play pard his))"),
        "{out}"
    );
    assert!(out.contains("passed: rule play-a-ten-now: stops at (count give 10)"), "{out}");
    assert!(out.contains("fired: rule play-a-five-now\n    3-2  <=  me=S1 l=S0 give=3-2"), "{out}");
    // --policy still names it; a query handed to eval says where it goes.
    let (ok, same) = silas(&[&["eval", "--policy", "examples/book/pounce.silas"][..], &args].concat());
    assert!(ok && same == out, "{same}");
    let (ok, out) = silas(&["eval", "examples/book/lay-down.silas", "--hand", "hand-21"]);
    assert!(!ok && out.contains("is not a policy: silas query examples/book/lay-down.silas"), "{out}");
}

#[test]
fn coverage_counts_each_rule_and_each_step_over_the_books_hands() {
    let (ok, out) = silas(&["coverage", "examples/book/protect-the-count.silas", "--hand", "hand-33"]);
    assert!(ok, "{out}");
    assert!(out.contains("over 24 frames of hand-33, and 0 random deals"), "{out}");
    assert!(out.contains("rule follow-the-master-without-count: decided 1, allowed 1, answered 1 of 24"), "{out}");
    assert!(out.contains("  12/24  (setter me)\n"), "{out}");
    assert!(out.contains("   1/24  (count-tile counted)\n  the fallback decided 23 of 24"), "{out}");
    // A step that never holds is flagged; so is a rule that never decides.
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("never.silas");
    let never =
        "(include book)\n(policy never\n  (rule r (play a)\n    (own-legal a)\n    (mine b)\n    (tile a b)))\n";
    std::fs::write(&path, never).unwrap();
    let (ok, out) = silas(&["coverage", path.to_str().unwrap(), "--hand", "hand-28"]);
    assert!(ok && out.contains("! rule r: decided 0") && out.contains("!      0/"), "{out}");
    assert!(out.lines().any(|l| l.starts_with("    !") && l.ends_with("(tile a b)")), "{out}");
    // A query is covered too, where it reads only the viewer's frame.
    let (ok, out) = silas(&["coverage", "examples/book/lay-down.silas", "--hand", "hand-21"]);
    assert!(ok && out.contains("held at 1 of 1") && out.contains("1/1  (certain-make)"), "{out}");
}

#[test]
fn the_trump_set_rules_are_rare_not_broken() {
    // The book narrates three trump sets and no position where its rules for
    // them apply: over its 205 frames no rule decides, and the frames that
    // reach the setter's ruff (one, two and one, by rule) have no trump to
    // ruff with.
    let (ok, out) = silas(&["coverage", "examples/book/trump-set.silas"]);
    assert!(ok, "{out}");
    for rule in ["trump-a-big-double", "trump-a-double", "trump-in-on-count", "feed-the-trump-holder", "lead-a-high-trump"]
    {
        assert!(out.contains(&format!("rule {rule}: decided 0, allowed 0, answered 0 of 205")), "{out}");
    }
    assert!(out.contains("   1/205  (own-legal ruff)\n    !      0/205  (trump-tile ruff)"), "{out}");
    assert!(out.contains("   2/205  (own-legal ruff)\n    !      0/205  (trump-tile ruff)"), "{out}");
    // Over 1,000 seeded random deals (28,205 frames with the book's) the ruff
    // rules stop at the setter's ruff, where no frame holds a trump (so the
    // clause for a second trump is not what silences them); the feed reaches
    // a holder who may yet trump and finds no count; the lead fires.
    let (ok, out) = silas(&["coverage", "examples/book/trump-set.silas", "--random", "1000"]);
    assert!(ok, "{out}");
    assert!(out.contains("rule trump-a-double: decided 0, allowed 0, answered 0 of 28205"), "{out}");
    assert!(out.contains("  14/28205  (own-legal ruff)\n    !      0/28205  (trump-tile ruff)"), "{out}");
    assert!(out.contains("   3/28205  (own-legal give)\n    !      0/28205  (count-tile give)"), "{out}");
    assert!(out.contains("rule lead-a-high-trump: decided 7, allowed 7, answered 7 of 28205"), "{out}");
}

#[test]
fn expand_prints_the_core_and_check_prints_warnings() {
    let (ok, out) = silas(&["expand", "examples/book/lay-down.silas"]);
    assert!(ok && out.contains("(roles (chair b))\n  (out)\n  (pattern\n    (viewer b)\n    (declarer b)"), "{out}");
    // check takes a file as well as a directory, and warns without failing.
    let (ok, out) = silas(&["check", "examples/book/crisis.silas"]);
    assert!(ok, "{out}");
    assert!(out.contains("crisis.silas:62:7: warning: always true: his and l"), "{out}");
    assert!(out.contains("1 compiled, 0 failed, 1 warnings"), "{out}");
}

#[test]
fn a_misspelt_option_an_unknown_hand_or_command_is_refused_with_the_choices() {
    let (ok, out) = silas(&["query", "examples/book/lay-down.silas", "--hand", "hand-21", "--trik", "2"]);
    assert!(
        !ok && out.contains("no option --trik here; the options are --hand, --trick, --seat, --cap, --book"),
        "{out}"
    );
    let (ok, out) = silas(&["query", "examples/book/lay-down.silas", "--hand", "hand-99"]);
    assert!(!ok && out.contains("no hand hand-99 in the book; the hands are hand-1,"), "{out}");
    let (ok, out) = silas(&["frobnicate"]);
    assert!(!ok && out.contains("no command frobnicate") && out.contains("silas coverage"), "{out}");
    let (ok, out) = silas(&["help"]);
    assert!(ok && out.starts_with("usage:"), "{out}");
}
