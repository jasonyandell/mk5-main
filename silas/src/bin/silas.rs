// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! `silas`: the language at the command line, over the book's hands.
//!
//! ```text
//! silas eval     POLICY --hand ID [--trick N] [--seat ROLE] [--book FILE]
//! silas bid      POLICY --hand ID [--seat ROLE] [--calls C,...] [--scoring S] [--book FILE]
//! silas query    QUERY --hand ID [--trick N] [--seat ROLE] [--cap N] [--book FILE] [--law ROLE]
//!                [--read ROLE=POLICY,... [--likelihood consistent|uniform]
//!                 [--auction C,C,C,C [--shaker ROLE] [--offered D,...] [--scoring S]]]
//! silas query    DOMINANCE
//! silas compare  --ledger FILE [--utility marks|points] [--confidence P/Q]
//! silas quotient --course FILE
//! silas family   --courses FILE
//! silas coverage POLICY-OR-QUERY [--hand ID] [--random N [--bidders POLICY [--offered D,...] [--scoring S]]
//!                [--play POLICY]] [--book FILE]
//! silas expand   FILE ...
//! silas fmt      [--check] FILE ...
//! silas check    [DIR-OR-FILE ...]
//! ```
//!
//! A position is a hand of `examples/book/hands.json` (or `--book`): a full
//! deal at the point of its narrated line (its tricks, then its tail, up to
//! the first play the book does not name) where `--seat` (bidder, left,
//! partner, right) is to play to trick `--trick` (default: the opening lead),
//! seen from that seat; or a hand the book gives alone, seen by its bidder
//! before the first lead. The bidder sits in S0.
//!
//! `eval` prints the deciding layer of the policy, where each rule before it
//! stops (the first clause, in source order, that fails), the fired rule's
//! witnesses (each answer and one binding of the guard's roles) and the
//! allowed set. `bid` puts the seat's hand at its turn in the auction after
//! `--calls` (the calls before it, in calling order; none by default) under
//! `--scoring` (marks or points, the match at 0-0), and prints the policy's
//! call, the rule that made it with its witnesses, and the declaration it
//! would make if the others then passed. `query` prints the query's exact probability over the worlds
//! the seat cannot tell apart, with its authority, resolving only the tiles
//! it reads. With `--read` (seats and the policies they act by), the worlds
//! are weighed by every read seat's past plays: kept when its policy allows
//! each (`--likelihood consistent`, the default), or weighed by the chance of
//! each among what it allows (`uniform`). With `--auction` too (the four
//! calls from the shaker's left; `--shaker`, by default the right, so the
//! bidder calls first) every read seat's policy must also make its call, and
//! declare the hand's declaration if it won, from its hand, with `--offered`
//! the declarations offered (by default the nine straight ones) and
//! `--scoring` the match (at 0-0). Each is exact given those policies. On a
//! `(dominates ...)` file it needs no hand: it decides the form
//! by enumerating the hand's outcomes, exactly, and prints the first
//! counterexample when there is one. `compare` reads an arena ledger (the
//! hands of a paired comparison, outcomes only), replays and settles every
//! row under Silas's rules, and prints the mean paired difference in the
//! ledger's unit (or `--utility`), its interval at `--confidence` (19/20 by
//! default), the flips, the verdict and the chance arm. `quotient` reads a
//! solved course (`lem/quotient.py --export`), audits it, and prints its
//! decision quotient under the raw, shape and fork interfaces as JSON: the
//! states, the block-depth ladders, and the counts beside them. `family` reads
//! family courses (`lem/endgame.py`), one JSON object a line, solves each
//! exactly and prints one JSON report a line: the root's values and maximizing
//! set in every parameter class, and the controller's quotient. `coverage` reads a policy (or a query of
//! the viewer's frame) at every point of the book's narrated lines, and of
//! `--random` seeded deals, and counts how often each rule answered, allowed a
//! legal tile and decided, and how often each step of each pattern held: a
//! zero is a rule or a clause that never holds there. With `--bidders` the
//! random deals' contracts are the auctions a bidding policy makes from them
//! (and the book's hands are left out), and with `--play` every seat plays
//! uniformly within what a play policy allows. `expand` prints a
//! file's core: definitions expanded, short forms spelled out. `fmt` formats
//! canonically, in place (or reports with `--check`). `check` compiles every
//! `.silas` file (policies under all nine straight declarations) and prints
//! its warnings. An option the command does not take is refused.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use serde_json::Value as Json;
use silas::reference::Straight42;
use silas::*;

const ROLES: [&str; 4] = ["bidder", "left", "partner", "right"];
const BOOK_ROLES: [&str; 4] = ["bidder", "opponent-1", "partner", "opponent-2"];

fn fail(message: impl Into<String>) -> Error {
    Error(message.into())
}

const USAGE: &str = "\
usage:
  silas eval     POLICY --hand ID [--trick N] [--seat ROLE] [--book FILE]
  silas bid      POLICY --hand ID [--seat ROLE] [--calls C,...] [--scoring S] [--book FILE]
  silas query    QUERY --hand ID [--trick N] [--seat ROLE] [--cap N] [--book FILE] [--law ROLE]
                 [--read ROLE=POLICY,... [--likelihood consistent|uniform]
                  [--auction C,C,C,C [--shaker ROLE] [--offered D,...] [--scoring S]]]
  silas query    DOMINANCE
  silas compare  --ledger FILE [--utility marks|points] [--confidence P/Q]
  silas quotient --course FILE
  silas family   --courses FILE
  silas coverage POLICY-OR-QUERY [--hand ID] [--random N [--bidders POLICY [--offered D,...] [--scoring S]]
                 [--play POLICY]] [--book FILE]
  silas expand   FILE ...
  silas fmt      [--check] FILE ...
  silas check    [DIR-OR-FILE ...]

A position is a hand of examples/book/hands.json at trick N (default 1),
with ROLE (bidder, left, partner, right) to play; the bidder sits in S0.
eval's POLICY may also be given as --policy FILE.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rest = args.get(1..).unwrap_or(&[]);
    let result = match args.first().map(String::as_str) {
        Some("eval") => eval(rest),
        Some("bid") => bid(rest),
        Some("query") => query(rest),
        Some("compare") => compare(rest),
        Some("quotient") => quotient(rest),
        Some("family") => family(rest),
        Some("coverage") => coverage(rest),
        Some("expand") => expand(rest),
        Some("fmt") => fmt(rest),
        Some("check") => check(rest),
        Some("help" | "--help" | "-h") => {
            println!("{USAGE}");
            Ok(true)
        }
        Some(other) => Err(fail(format!("silas: no command {other}\n{USAGE}"))),
        None => Err(fail(USAGE)),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

/// `--name value` options and the remaining words.
struct Options {
    named: Vec<(String, String)>,
    flags: Vec<String>,
    words: Vec<String>,
}

impl Options {
    /// The arguments, given the command's flags and its options that take a
    /// value. Anything else spelled `--name` is refused, so a misspelt option
    /// never runs silently on a default.
    fn parse(args: &[String], flags: &[&str], named: &[&str]) -> Result<Options> {
        let mut out = Options { named: vec![], flags: vec![], words: vec![] };
        let mut it = args.iter();
        while let Some(arg) = it.next() {
            match arg.strip_prefix("--") {
                Some(name) if flags.contains(&name) => out.flags.push(name.to_owned()),
                Some(name) if !named.contains(&name) => {
                    let known: Vec<String> = flags.iter().chain(named).map(|n| format!("--{n}")).collect();
                    let known = if known.is_empty() { "none".to_owned() } else { known.join(", ") };
                    return Err(fail(format!("no option --{name} here; the options are {known}")));
                }
                Some(name) => {
                    let value = it.next().ok_or_else(|| fail(format!("--{name} takes a value")))?;
                    out.named.push((name.to_owned(), value.clone()));
                }
                None => out.words.push(arg.clone()),
            }
        }
        Ok(out)
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.named.iter().rev().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }

    fn need(&self, name: &str) -> Result<&str> {
        self.get(name).ok_or_else(|| fail(format!("--{name} is required")))
    }

    fn number(&self, name: &str, default: usize) -> Result<usize> {
        self.get(name).map_or(Ok(default), |v| v.parse().map_err(|_| fail(format!("--{name} takes a number"))))
    }
}

/// `x` to four decimal places, rounded half away from zero; no float.
fn decimal(x: &num_rational::BigRational) -> String {
    let scaled = (x * num_rational::BigRational::from_integer(10_000.into())).round().to_integer();
    let sign = if scaled < 0.into() { "-" } else { "" };
    let digits = format!("{:05}", scaled.magnitude());
    let (whole, frac) = digits.split_at(digits.len() - 4);
    format!("{sign}{whole}.{frac}")
}

/// A solved course's decision quotient, every interface, as JSON.
fn quotient(args: &[String]) -> Result<bool> {
    let options = Options::parse(args, &[], &["course"])?;
    if !options.words.is_empty() {
        return Err(fail("silas quotient --course FILE"));
    }
    let path = options.need("course")?;
    let course = quotient::file::CourseFile::from_json(&read(path)?).map_err(|e| fail(format!("{path}: {e}")))?;
    let report = course.report()?;
    println!("{}", serde_json::to_string_pretty(&report).map_err(|e| fail(e.to_string()))?);
    Ok(true)
}

/// Family courses, one a line, each solved exactly; one report a line.
fn family(args: &[String]) -> Result<bool> {
    let options = Options::parse(args, &[], &["courses"])?;
    if !options.words.is_empty() {
        return Err(fail("silas family --courses FILE"));
    }
    let path = options.need("courses")?;
    for (i, line) in read(path)?.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let at = |e: Error| fail(format!("{path}:{}: {e}", i + 1));
        let course = family::file::FamilyFile::from_json(line).map_err(at)?;
        let report = course.report().map_err(at)?;
        println!("{}", serde_json::to_string(&report).map_err(|e| fail(e.to_string()))?);
    }
    Ok(true)
}

/// An arena ledger's paired comparison, replayed and settled by Silas.
fn compare(args: &[String]) -> Result<bool> {
    let options = Options::parse(args, &[], &["ledger", "utility", "confidence"])?;
    if !options.words.is_empty() {
        return Err(fail("silas compare --ledger FILE [--utility marks|points] [--confidence P/Q]"));
    }
    let path = options.need("ledger")?;
    let ledger = Ledger::from_json(&read(path)?).map_err(|e| fail(format!("{path}: {e}")))?;
    let unit = match ledger.table.state.scoring() {
        Scoring::Marks { .. } => "marks",
        Scoring::Points { .. } => "points",
    };
    let (unit, utility) = match options.get("utility").unwrap_or(unit) {
        "marks" => ("marks", Utility::Marks),
        "points" => ("points", Utility::Points),
        other => return Err(fail(format!("--utility is marks or points, not {other}"))),
    };
    let confidence: num_rational::BigRational = options
        .get("confidence")
        .unwrap_or("19/20")
        .parse()
        .map_err(|_| fail("--confidence is a fraction, such as 19/20"))?;
    let paired = Paired::from_ledger(&ledger, &utility, &confidence)?;
    let seats = match ledger.seats {
        Seats::Rotate => "every chair".to_owned(),
        Seats::Fixed(chair) => format!("the chair {chair}"),
    };
    let [a, b, chance] = &ledger.players;
    println!("compare {path}: {a} against {b}, in {seats} at every shaker, from {}", ledger.table.state);
    println!("  field: {}", ledger.field);
    let show = |name: &str, p: &Paired| {
        let (lo, hi) = p.interval();
        let (wins, losses, ties) = p.flips();
        println!(
            "  {name}: {} {unit} a deal [{}, {}] at {confidence}; flips {wins} for, {losses} against, {ties} tied; {:?}",
            decimal(p.delta().value()),
            decimal(lo),
            decimal(hi),
            p.verdict()
        );
    };
    show(&format!("{a} - {b}"), &paired);
    show(&format!("{a} - {chance}"), paired.chance().expect("the chance arm"));
    println!("  {} deals; {}", paired.n(), paired.delta().authority());
    Ok(true)
}

fn read(path: &str) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| fail(format!("{path}: {e}")))
}

/// One hand of the book: its auction, with the bidder in S0, and its
/// declaration; the four hands it gives (only the bidder's, for a hand the
/// book gives alone); and the plays its narrated line names, up to the first
/// it does not. The book gives a hand's contract, not its auction, so the
/// record is the shortest auction the bidder wins with the contract's bid.
struct Hand {
    id: String,
    auction: Auction,
    declaration: Physics,
    hands: [DominoSet; 4],
    full: bool,
    narrated: Vec<Domino>,
}

/// Every hand of the book's hands file (`--book`, by default
/// `examples/book/hands.json`) that has a declaration and a numeric bid.
fn book(options: &Options) -> Result<Vec<Hand>> {
    let path = options
        .get("book")
        .map_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/examples/book/hands.json").to_owned(), str::to_owned);
    let json: Json = serde_json::from_str(&read(&path)?).map_err(|e| fail(format!("{path}: {e}")))?;
    let empty = vec![];
    let lines = json["lines"].as_array().unwrap_or(&empty);
    let mut out = Vec::new();
    for example in json["examples"].as_array().unwrap_or(&empty) {
        let id = example["id"].as_str().unwrap_or("?").to_owned();
        let line = lines.iter().find(|l| l["example"] == id.as_str() && l["source"] == "prose");
        let text = |v: &Json| v.as_str().map(str::to_owned);
        let declaration = text(&example["declaration"]).or_else(|| line.and_then(|l| text(&l["declaration"])));
        let bid = example["bid"].as_u64().or_else(|| line.and_then(|l| l["bid"].as_u64()));
        let (Some(declaration), Some(bid)) = (declaration, bid) else { continue };
        let rules: Straight42 = declaration.parse()?;
        let auction = Auction::won_by(Seat::S0, Bid::number(bid as u32)?)?;
        let mut hands = [DominoSet::EMPTY; 4];
        for seat in example["seats"].as_array().unwrap_or(&empty) {
            let role = text(&seat["role"]).unwrap_or_default();
            let at = BOOK_ROLES.iter().position(|r| *r == role).ok_or_else(|| fail(format!("{id}: a role {role}")))?;
            let tiles = seat["tiles"].as_array().unwrap_or(&empty);
            hands[at] = tiles.iter().filter_map(Json::as_str).map(str::parse).collect::<Result<_>>()?;
        }
        // The narrated tricks' plays and then the tail's, up to the first the
        // book does not name.
        let mut narrated = Vec::new();
        let tricks = line.map_or(&empty, |l| l["tricks"].as_array().unwrap_or(&empty));
        let tail = line.map(|l| &l["tail"]).into_iter().filter(|t| t.is_object());
        'line: for t in tricks.iter().chain(tail) {
            for play in t["plays"].as_array().unwrap_or(&empty) {
                match play[1].as_str() {
                    Some(tile) => narrated.push(tile.parse::<Domino>()?),
                    None => break 'line,
                }
            }
        }
        out.push(Hand { id, auction, declaration: rules.physics(), hands, full: example["kind"] == "deal", narrated });
    }
    Ok(out)
}

impl Hand {
    /// The bidder's frame before the opening lead.
    fn opening(&self) -> Result<Frame> {
        Frame::new(self.auction.clone(), self.declaration.clone(), Seat::S0, self.hands[0], vec![])
    }

    /// Every point of the narrated line, seen by the seat to play: the
    /// number of plays before it and the frame. The line stops at the first
    /// play the record refuses (a hand the book gives in part has no record
    /// from the deal), and the refusal comes with the points before it.
    fn points(&self) -> (Vec<(usize, Frame)>, Option<Error>) {
        let mut out = Vec::new();
        let mut record = match self.opening() {
            Ok(frame) if !self.full => return (vec![(0, frame)], None),
            Ok(frame) => frame,
            Err(e) => return (out, Some(fail(format!("{}: {e}", self.id)))),
        };
        for i in 0..=self.narrated.len() {
            let Some(viewer) = record.next_actor() else { break };
            let (auction, declaration) = (self.auction.clone(), self.declaration.clone());
            match Frame::new(auction, declaration, viewer, self.hands[viewer.index()], self.narrated[..i].to_vec()) {
                Ok(frame) => out.push((i, frame)),
                Err(e) => return (out, Some(fail(format!("{}: after {i} narrated plays: {e}", self.id)))),
            }
            if i < self.narrated.len() {
                match record.after(self.narrated[i]) {
                    Ok(next) => record = next,
                    Err(e) => return (out, Some(fail(format!("{}: the narrated play {}: {e}", self.id, i + 1)))),
                }
            }
        }
        (out, None)
    }
}

/// The book hand `--hand` names.
fn hand(options: &Options) -> Result<Hand> {
    let id = options.need("hand")?;
    let hands = book(options)?;
    let known: Vec<String> = hands.iter().map(|h| h.id.clone()).collect();
    hands
        .into_iter()
        .find(|h| h.id == id)
        .ok_or_else(|| fail(format!("no hand {id} in the book; the hands are {}", known.join(", "))))
}

/// A book position: the frame, and a line describing it.
fn position(options: &Options) -> Result<(Frame, String)> {
    let hand = hand(options)?;
    let id = &hand.id;
    let trick = options.number("trick", 1)?;
    let seat = match options.get("seat") {
        Some(name) => Some(
            ROLES
                .iter()
                .position(|r| *r == name)
                .ok_or_else(|| fail(format!("--seat is one of {}", ROLES.join(", "))))?,
        ),
        None => None,
    };
    if !hand.full {
        if trick != 1 || seat.is_some_and(|s| s != 0) {
            return Err(fail(format!("{id} gives only the bidder's hand: its position is the opening lead")));
        }
        return Ok((hand.opening()?, format!("{id}: the bidder's hand before the opening lead")));
    }
    // The first point of the line where the seat is to play to the trick.
    let (points, refused) = hand.points();
    let (_, frame) = points
        .into_iter()
        .find(|(i, frame)| i / 4 + 1 == trick && seat.is_none_or(|r| frame.viewer().index() == r))
        .ok_or_else(|| {
            refused.unwrap_or_else(|| fail(format!("{id}'s narrated line does not reach that seat in trick {trick}")))
        })?;
    let viewer = frame.viewer();
    let who = ROLES[viewer.index()];
    Ok((frame, format!("{id}: trick {trick}, the {who} ({viewer}) to play")))
}

fn tiles(set: DominoSet) -> String {
    set.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" ")
}

fn describe(frame: &Frame, heading: &str) {
    let contract = frame.contract();
    println!("{heading}");
    println!(
        "  {} ({}); the bidder {} bids {}: {} points for {} mark(s)",
        frame.physics().name(),
        frame.physics().identity(),
        contract.declarer(),
        contract.bid(),
        contract.threshold(),
        contract.stake()
    );
    let tricks: Vec<String> =
        frame.tricks().map(|t| t.iter().map(|(s, d)| format!("{s}:{d}")).collect::<Vec<_>>().join(" ")).collect();
    if !tricks.is_empty() {
        println!("  record: {}", tricks.join(" | "));
    }
    let [a, b] = frame.banked();
    println!("  banked: T0 {a}, T1 {b}; hand: {}", tiles(frame.kernel().viewer_hand()));
}

/// The one file a command reads: its single word, or `--policy` for eval.
fn file<'a>(options: &'a Options, usage: &str) -> Result<&'a str> {
    match (options.words.as_slice(), options.get("policy")) {
        ([path], None) => Ok(path),
        ([], Some(path)) => Ok(path),
        _ => Err(fail(format!("usage: {usage}"))),
    }
}

/// A policy file, or a hint at the command that reads what the file holds.
fn policy_program(path: &str) -> Result<PolicyProgram> {
    match Program::parse(&read(path)?, path)? {
        Program::Policy(policy) => Ok(policy),
        Program::Query(_) | Program::Dominance(_) => {
            Err(fail(format!("{path} is not a policy: silas query {path} ...")))
        }
    }
}

/// Compile the policy read from `path`: a reader its readings name is the
/// policy file of that name beside it.
fn compile_at(program: &PolicyProgram, path: &str, registry: &Registry, physics: &Physics) -> Result<CompiledPolicy> {
    let dir = std::path::Path::new(path).parent().map(std::path::Path::to_path_buf).unwrap_or_default();
    let resolve = |name: &str| policy_program(&dir.join(format!("{name}.silas")).display().to_string());
    program.compile_reading(registry, physics, &resolve)
}

fn eval(args: &[String]) -> Result<bool> {
    let options = Options::parse(args, &[], &["hand", "trick", "seat", "policy", "book"])?;
    let path = file(&options, "silas eval POLICY --hand ID [--trick N] [--seat ROLE]")?;
    let program = policy_program(path)?;
    let (frame, heading) = position(&options)?;
    let registry = Registry::standard();
    let policy = compile_at(&program, path, &registry, frame.physics())?;
    describe(&frame, &heading);
    let legal = frame.physics().legal_plays(frame.kernel().viewer_hand(), frame.lead());
    println!("  legal: {}", tiles(legal));
    let mut budget = Budget::new(100_000_000);
    let (allowed, witnesses) = policy.explain(&frame, &mut budget)?;
    println!("policy {}", program.name);
    for (rule, why) in why_not(&policy, &allowed.provenance, &frame, &registry, &mut budget)? {
        println!("  passed: rule {rule}: {why}");
    }
    match &allowed.provenance {
        DecisionProvenance::RelationalRule { name } => {
            println!("  fired: rule {name}");
            for (answer, witness) in &witnesses {
                let answer: Vec<String> = answer.iter().map(Value::to_string).collect();
                let binding: Vec<String> = witness.iter().map(|(n, v)| format!("{n}={v}")).collect();
                let shown = if answer.is_empty() { "holds".to_owned() } else { answer.join(" ") };
                println!("    {shown}  <=  {}", binding.join(" "));
            }
        }
        DecisionProvenance::Exact => println!("  fired: the exact table"),
        DecisionProvenance::Fallback => println!("  fired: no rule; the fallback (lowest legal)"),
    }
    println!("  allowed: {}", tiles(allowed.tiles));
    println!("  plays: {}", allowed.tiles.iter().next().expect("a deciding layer allows a tile"));
    Ok(true)
}

/// The deciding layer at the auction, and the deciding rule's witnesses.
fn print_choice<T>(choice: &Choice<T>, what: &str, shown: &str) {
    match &choice.provenance {
        DecisionProvenance::RelationalRule { name } => {
            println!("  {what} {shown}, by rule {name}");
            for (answer, witness) in &choice.witnesses {
                let answer: Vec<String> = answer.iter().map(Value::to_string).collect();
                let binding: Vec<String> = witness.iter().map(|(n, v)| format!("{n}={v}")).collect();
                let held = if answer.is_empty() { "holds".to_owned() } else { answer.join(" ") };
                println!("    {held}  <=  {}", binding.join(" "));
            }
        }
        _ => println!("  {what} {shown}, by the fallback"),
    }
}

fn bid(args: &[String]) -> Result<bool> {
    let options = Options::parse(args, &[], &["hand", "seat", "calls", "scoring", "policy", "book"])?;
    let path = file(&options, "silas bid POLICY --hand ID [--seat ROLE] [--calls C,...] [--scoring S]")?;
    let program = policy_program(path)?;
    let hand = hand(&options)?;
    let role = options.get("seat").unwrap_or("bidder");
    let seat = ROLES
        .iter()
        .position(|r| *r == role)
        .map(|i| Seat::ALL[i])
        .ok_or_else(|| fail(format!("--seat is one of {}", ROLES.join(", "))))?;
    let dealt = hand.hands[seat.index()];
    if dealt.len() != 7 {
        return Err(fail(format!("{} gives only the bidder's hand", hand.id)));
    }
    let calls: Vec<Call> = match options.get("calls") {
        Some(text) => text.split([',', ' ']).filter(|w| !w.is_empty()).map(str::parse).collect::<Result<_>>()?,
        None => vec![],
    };
    if calls.len() >= Seat::COUNT {
        return Err(fail("--calls are the calls before the seat's: at most three"));
    }
    let shaker = seat.plus(Seat::COUNT - 1 - calls.len());
    let auction = Auction::new(AuctionConfig::STRAIGHT, shaker, calls)?;
    let declarations: Arc<[Physics]> = Straight42::ALL.iter().map(|d| d.physics()).collect();
    let state = match_state(&options)?;
    let scoring = state.scoring();
    let frame = AuctionFrame::new(auction.clone(), seat, dealt, state, declarations.clone())?;
    let policy = compile_at(&program, path, &Registry::standard(), &declarations[0])?;
    println!("{}: the {role} ({seat}) to call; {auction}; {scoring}", hand.id);
    println!("  hand: {}", tiles(dealt));
    let legal: Vec<String> = auction.legal_calls().iter().map(Call::to_string).collect();
    println!("  legal: {}", legal.join(" "));
    println!("policy {}", program.name);
    let mut budget = Budget::new(1_000_000_000);
    let choice = policy.explain_bid(&frame, &mut budget)?;
    print_choice(&choice, "calls", &choice.choice.to_string());
    if choice.choice != Call::Pass {
        let mut after = auction.after(choice.choice)?;
        while after.actor().is_some() && after.legal(Call::Pass) {
            after = after.after(Call::Pass)?;
        }
        if after.result().is_some_and(|(winner, _)| winner == seat) {
            let won = AuctionFrame::new(after, seat, dealt, state, declarations)?;
            let declared = policy.explain_declare(&won, &mut budget)?;
            print_choice(&declared, "if the rest pass, declares", declared.choice.name());
        }
    }
    Ok(true)
}

fn query(args: &[String]) -> Result<bool> {
    let named = [
        "hand",
        "trick",
        "seat",
        "cap",
        "book",
        "auction",
        "shaker",
        "read",
        "likelihood",
        "offered",
        "scoring",
        "law",
    ];
    let options = Options::parse(args, &[], &named)?;
    // `--law ROLE`: the belief is that hidden seat's hand law.
    let law = options
        .get("law")
        .map(|role| {
            ROLES
                .iter()
                .position(|r| *r == role)
                .map(|i| Seat::ALL[i])
                .ok_or_else(|| fail(format!("--law is one of {}", ROLES.join(", "))))
        })
        .transpose()?;
    let path = file(&options, "silas query FILE --hand ID [--trick N] [--seat ROLE] [--cap N]")?;
    let text = read(path)?;
    let query = match Program::parse(&text, path)? {
        Program::Query(query) => query.compile(&Registry::standard())?,
        Program::Dominance(dominance) => return dominate(&dominance),
        Program::Policy(_) => return Err(fail(format!("{path} is a policy: silas eval {path} --hand ID ..."))),
    };
    let (frame, heading) = position(&options)?;
    let auctioned = ["auction", "shaker", "offered", "scoring"].iter().any(|o| options.get(o).is_some());
    let (frame, reading) = match options.get("read") {
        None if auctioned || options.get("likelihood").is_some() => {
            return Err(fail(
                "--auction, --shaker, --offered, --scoring and --likelihood go with --read: a book hand's own \
                 auction is a convention (its bidder speaks first and the rest pass), not the table's, so a \
                 reading names the auction",
            ));
        }
        None => (frame, None),
        Some(_) => {
            let (frame, reading) = read_record(&options, frame)?;
            (frame, Some(reading))
        }
    };
    describe(&frame, &heading);
    let cap = options.number("cap", 100_000)?;
    let mut budget = Budget::new(1_000_000_000_000);
    let belief = match reading {
        None => match law {
            None => Belief::uniform_for(frame.clone(), &query, cap, &mut budget)?,
            Some(seat) => Belief::uniform_hand_law(frame.clone(), seat, cap, &mut budget)?,
        },
        Some(reading) => {
            if options.get("auction").is_some() {
                println!("  auction: {}", frame.auction());
            }
            let evidence = reading.evidence(&frame, &mut budget)?;
            for seat in evidence.seats() {
                let mut read = Vec::new();
                if let Some(call) = &seat.call {
                    let declares =
                        if call.declares { format!(" and declares {}", frame.physics().name()) } else { String::new() };
                    read.push(format!("{} calls {}{declares}", call.policy, call.call));
                }
                if let Some(plays) = seat.plays.as_ref().filter(|p| !p.tiles.is_empty()) {
                    let shown: Vec<String> = plays.tiles.iter().map(Domino::to_string).collect();
                    let how = match plays.likelihood {
                        Likelihood::Consistent => "",
                        Likelihood::Uniform => ", uniformly",
                    };
                    read.push(format!("{} allows its plays {}{how}", plays.policy, shown.join(" ")));
                }
                println!(
                    "  read {}: {}, from {} of the {} hands it could hold",
                    seat.seat,
                    read.join(", and "),
                    seat.agreeing,
                    seat.candidates
                );
            }
            match law {
                None => Belief::given_for(&evidence, &query, cap, &mut budget)?,
                Some(seat) => Belief::hand_law(&evidence, seat, cap)?,
            }
        }
    };
    let summary = query.summarize(&belief, &mut budget)?;
    let p = summary.event_probability();
    let resolved = frame.kernel().pool().difference(belief.unresolved());
    println!("query {path}");
    let mass = match (belief.authority().is_exact(), belief.total_mass().rational().is_integer()) {
        (true, _) => String::new(),
        (false, true) => format!(" ({} kept)", belief.total_mass()),
        (false, false) => format!(" (weighing {})", belief.total_mass()),
    };
    let what = match belief.known() {
        Some(seat) => format!("points, the hand law of {seat}"),
        None if resolved.is_empty() => "points resolving nothing".to_owned(),
        None => format!("points resolving {}", tiles(resolved)),
    };
    println!("  {} worlds{mass}, as {} {what}", frame.count_worlds(), belief.len());
    println!("  P(answer) = {} ({})", p.value(), p.authority());
    for answer in summary.answer_presence_mass.keys().filter(|a| !a.0.is_empty()) {
        let shown: Vec<String> = answer.0.iter().map(Value::to_string).collect();
        let p = summary.presence_probability(answer);
        println!("  P({}) = {} ({})", shown.join(" "), p.value(), p.authority());
    }
    Ok(true)
}

/// The reading `--read` names: every read seat's plays, under
/// `--likelihood`, and with `--auction` its calls too, the book position's
/// frame then put under the auction `--auction` names (won by the bidder at
/// the hand's bid, as the record needs). A book hand's own auction is a
/// convention, so its calls are read only when the auction is named.
fn read_record(options: &Options, frame: Frame) -> Result<(Frame, Reading)> {
    let seat = |role: &str| {
        ROLES
            .iter()
            .position(|r| *r == role)
            .map(|i| Seat::ALL[i])
            .ok_or_else(|| fail(format!("a seat is one of {}, not {role}", ROLES.join(", "))))
    };
    let words = |text: &str| text.split([',', ' ']).filter(|w| !w.is_empty()).map(str::to_owned).collect::<Vec<_>>();
    let likelihood = match options.get("likelihood").unwrap_or("consistent") {
        "consistent" => Likelihood::Consistent,
        "uniform" => Likelihood::Uniform,
        other => return Err(fail(format!("--likelihood is consistent or uniform, not {other}"))),
    };
    let mut readers = Vec::new();
    for pair in words(options.need("read")?) {
        let (role, path) = pair.split_once('=').ok_or_else(|| fail("--read takes ROLE=POLICY, ..."))?;
        let policy = compile_at(&policy_program(path)?, path, &Registry::standard(), frame.physics())?;
        readers.push((seat(role)?, policy));
    }
    let Some(calls) = options.get("auction") else {
        if frame.plays().is_empty() {
            return Err(fail(
                "nothing to read: no seat has played, and a book hand's own auction is a convention \
                 (its bidder speaks first and the rest pass), not the table's; --auction names the table's",
            ));
        }
        return Ok((frame, Reading::plays(readers, likelihood)?));
    };
    let calls: Vec<Call> = words(calls).iter().map(|c| c.parse()).collect::<Result<_>>()?;
    let shaker = seat(options.get("shaker").unwrap_or("right"))?;
    let auction = Auction::new(AuctionConfig::STRAIGHT, shaker, calls)?;
    let contract = frame.contract();
    if auction.result() != Some((contract.declarer(), contract.bid())) {
        return Err(fail(format!(
            "{auction} is not won by the bidder (S0) at {}, the hand's contract",
            contract.bid()
        )));
    }
    let frame = Frame::new(auction, frame.physics().clone(), frame.viewer(), frame.dealt(), frame.plays().to_vec())?;
    let reading = Reading::calls(readers.clone(), match_state(options)?, offered(options)?)?;
    Ok((frame, reading.with_plays(readers, likelihood)?))
}

/// The match an auction is called at: `--scoring` (marks, the default, or
/// points), at 0-0.
fn match_state(options: &Options) -> Result<MatchState> {
    let scoring = match options.get("scoring").unwrap_or("marks") {
        "marks" => Scoring::MARKS,
        "points" => Scoring::POINTS,
        other => return Err(fail(format!("--scoring is marks or points, not {other}"))),
    };
    MatchState::new(scoring, [0, 0])
}

/// The declarations offered to an auction's winner: `--offered`, by default
/// the nine straight ones in `Straight42` order. The order is a declaring
/// policy's tie-break.
fn offered(options: &Options) -> Result<Arc<[Physics]>> {
    match options.get("offered") {
        Some(text) => text
            .split([',', ' '])
            .filter(|w| !w.is_empty())
            .map(|d| d.parse::<Straight42>().map(|s| s.physics()))
            .collect(),
        None => Ok(Straight42::ALL.iter().map(|d| d.physics()).collect()),
    }
}

/// The frames coverage reads, each with the random deal it comes from (none
/// for the book's), and what they are.
struct Frames {
    frames: Vec<(Option<usize>, Frame)>,
    /// What the frames are, for the heading.
    what: String,
    /// How the random deals were made: their contracts and their play.
    how: Vec<String>,
    random: usize,
}

/// The frames coverage reads: every point of the book's narrated lines (or
/// of one hand's), seen by the seat to play, and the opening of each hand
/// the book gives alone; then `--random` seeded deals, each seen by every
/// seat at every play.
///
/// A random deal's contract is a random seat's at a random bid from 30 to
/// 42, under the nine straight declarations in turn; or, with `--bidders`,
/// the auction the policy makes from the deal: the shaker drawn, every seat
/// calling by it under `--scoring`, and the winner declaring by it among
/// `--offered`, a deal passed out dealt again. The book's contracts are not
/// the policy's, so its frames are then left out. Each seat plays a tile
/// drawn uniformly from its legal ones; or, with `--play`, from the set that
/// policy allows (every legal tile where it is silent), as the uniform
/// likelihood reads a play.
fn coverage_frames(options: &Options) -> Result<Frames> {
    let registry = Registry::standard();
    let random = options.number("random", 0)?;
    let bidders = options.get("bidders").map(policy_program).transpose()?;
    let players = options.get("play").map(policy_program).transpose()?;
    if random == 0 && (bidders.is_some() || players.is_some()) {
        return Err(fail("--bidders and --play make the random deals: give --random N"));
    }
    if bidders.is_some() && options.get("hand").is_some() {
        return Err(fail("--bidders reads its own deals, not the book's: --hand does not apply"));
    }
    let hands = match (&bidders, options.get("hand")) {
        (Some(_), _) => vec![],
        (None, Some(_)) => vec![hand(options)?],
        (None, None) => book(options)?,
    };
    // A line the record refuses is read up to the refusal, and named.
    let mut frames = Vec::new();
    let mut partial = Vec::new();
    for hand in &hands {
        let (points, refused) = hand.points();
        frames.extend(points.into_iter().map(|(_, frame)| (None, frame)));
        if refused.is_some() {
            partial.push(hand.id.clone());
        }
    }
    let book_frames = frames.len();
    let state = match_state(options)?;
    let offered = offered(options)?;
    let bidders = bidders.map(|p| Ok::<_, Error>((p.name.clone(), p.compile(&registry, &offered[0])?))).transpose()?;
    let mut players_by_physics = std::collections::BTreeMap::new();
    let mut budget = Budget::new(u64::MAX);
    let mut rng = 0x5eed_u64;
    let mut next = |n: usize| {
        rng = rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) % n as u64) as usize
    };
    let (mut dealt, mut bids) = (0, std::collections::BTreeMap::<Bid, usize>::new());
    for game in 0..random {
        let (deal, auction, physics) = loop {
            let mut deck = Domino::ALL.to_vec();
            for i in (1..28).rev() {
                deck.swap(i, next(i + 1));
            }
            let deal: [DominoSet; 4] = std::array::from_fn(|s| deck[7 * s..7 * s + 7].iter().copied().collect());
            dealt += 1;
            let Some((_, policy)) = &bidders else {
                let auction = Auction::won_by(Seat::ALL[next(4)], Bid::number(30 + next(13) as u32)?)?;
                break (deal, auction, Straight42::ALL[game % 9].physics());
            };
            let mut auction = Auction::new(AuctionConfig::STRAIGHT, Seat::ALL[next(4)], vec![])?;
            while let Some(seat) = auction.actor() {
                let frame = AuctionFrame::new(auction.clone(), seat, deal[seat.index()], state, offered.clone())?;
                auction = auction.after(policy.bid(&frame, &mut budget)?.choice)?;
            }
            let Some((declarer, _)) = auction.result() else { continue };
            let won = AuctionFrame::new(auction.clone(), declarer, deal[declarer.index()], state, offered.clone())?;
            let physics = policy.declare(&won, &mut budget)?.choice;
            break (deal, auction, physics);
        };
        *bids.entry(auction.result().expect("a contract").1).or_default() += 1;
        let player = match &players {
            None => None,
            Some(program) => {
                let id = physics.identity().to_owned();
                if !players_by_physics.contains_key(&id) {
                    let at = options.get("play").expect("--play names the players");
                    players_by_physics.insert(id.clone(), compile_at(program, at, &registry, &physics)?);
                }
                Some(&players_by_physics[&id])
            }
        };
        let mut record = Frame::new(auction.clone(), physics.clone(), Seat::S0, deal[0], vec![])?;
        let mut hands = deal;
        while let Some(viewer) = record.next_actor() {
            let plays = record.plays().to_vec();
            let frame = Frame::new(auction.clone(), physics.clone(), viewer, deal[viewer.index()], plays)?;
            let drawn = match player {
                None => physics.legal_plays(hands[viewer.index()], record.lead()),
                Some(policy) => policy.permits(&frame, &mut budget)?,
            };
            frames.push((Some(game), frame));
            let tile = drawn.iter().nth(next(drawn.len())).expect("a legal play");
            hands[viewer.index()].remove(tile);
            record = record.after(tile)?;
        }
    }
    let mut how = Vec::new();
    if let Some((name, _)) = &bidders {
        let declarations: Vec<&str> = offered.iter().map(|p| p.name().rsplit('/').next().unwrap_or("")).collect();
        let shown: Vec<String> = bids.iter().map(|(bid, n)| format!("{} x{n}", bid.value())).collect();
        how.push(format!(
            "contracts: {name} makes every call (the shaker drawn, {}) and declares from {}",
            state.scoring(),
            declarations.join(" ")
        ));
        how.push(format!(
            "  {random} contracts from {dealt} deals ({} passed out); bids {}",
            dealt - random,
            shown.join(", ")
        ));
    }
    if random > 0 {
        how.push(match &players {
            None => "play: every seat a uniformly random legal tile".to_owned(),
            Some(program) => format!(
                "play: every seat uniformly within what {} allows (every legal tile where it is silent)",
                program.name
            ),
        });
    }
    let what = match (bidders.is_some(), options.get("hand")) {
        (true, _) => format!("{random} random deals"),
        (false, Some(id)) => format!("{book_frames} frames of {id}, and {random} random deals"),
        (false, None) => {
            let refused = if partial.is_empty() {
                String::new()
            } else {
                format!(" ({} read only up to a play the record refuses)", partial.join(", "))
            };
            format!("{book_frames} frames of the book's {} hands{refused}, and {random} random deals", hands.len())
        }
    };
    Ok(Frames { frames, what, how, random })
}

/// How often each rule answered, allowed a legal tile and decided, and where
/// each pattern stops: the frames where each step of its funnel holds. A
/// count of zero is a rule or a clause that never holds on these frames.
fn coverage(args: &[String]) -> Result<bool> {
    let options = Options::parse(args, &[], &["hand", "random", "bidders", "play", "offered", "scoring", "book"])?;
    let path = file(
        &options,
        "silas coverage FILE [--hand ID] [--random N [--bidders POLICY [--offered D,...] [--scoring S]] [--play POLICY]]",
    )?;
    let registry = Registry::standard();
    let Frames { frames, what, how, random } = coverage_frames(&options)?;
    let mut budget = Budget::new(u64::MAX);
    // The random deals in which a count rose: a rule's, or a query's.
    let in_deals = |deals: &std::collections::BTreeSet<usize>| {
        if random == 0 { String::new() } else { format!(", in {} of the {random} random deals", deals.len()) }
    };
    let show = |counts: &[(String, u64)], of: u64| {
        for (label, n) in counts {
            let flag = if *n == 0 { "!" } else { " " };
            let label: String = if label.chars().count() > 72 {
                label.chars().take(71).chain(['…']).collect()
            } else {
                label.clone()
            };
            println!("    {flag}{n:>7}/{of}  {label}");
        }
    };
    match Program::parse(&read(path)?, path)? {
        Program::Policy(program) => {
            let mut compiled = std::collections::BTreeMap::new();
            let mut coverage: Option<Coverage> = None;
            let mut answered_in = Vec::new();
            for (deal, frame) in &frames {
                let id = frame.physics().identity().to_owned();
                if !compiled.contains_key(&id) {
                    compiled.insert(id.clone(), compile_at(&program, path, &registry, frame.physics())?);
                }
                let policy = &compiled[&id];
                let coverage = match &mut coverage {
                    Some(coverage) => coverage,
                    None => {
                        let made = Coverage::new(policy, &registry)?;
                        answered_in = vec![std::collections::BTreeSet::new(); made.rules.len()];
                        coverage.insert(made)
                    }
                };
                let before: Vec<u64> = coverage.rules.iter().map(|r| r.answered).collect();
                coverage.observe(policy, frame, &mut budget)?;
                for ((rule, was), deals) in coverage.rules.iter().zip(before).zip(&mut answered_in) {
                    if let Some(deal) = deal.filter(|_| rule.answered > was) {
                        deals.insert(deal);
                    }
                }
            }
            println!("coverage of policy {} over {what}", program.name);
            for line in &how {
                println!("  {line}");
            }
            let Some(coverage) = coverage else { return Ok(true) };
            let of = coverage.frames;
            for (rule, deals) in coverage.rules.iter().zip(&answered_in) {
                let flag = if rule.decided == 0 { "!" } else { " " };
                println!(
                    "{flag} rule {}: decided {}, allowed {}, answered {} of {of}{}",
                    rule.name,
                    rule.decided,
                    rule.allowed,
                    rule.answered,
                    in_deals(deals)
                );
                for (i, pattern) in rule.patterns.iter().enumerate() {
                    if rule.patterns.len() > 1 {
                        println!("    pattern {}", i + 1);
                    }
                    show(pattern, of);
                }
            }
            println!("  the fallback decided {} of {of}", coverage.fallback);
        }
        Program::Query(query) => {
            let compiled = query.compile(&registry)?;
            let funnels = Funnel::of(&query, &registry)?;
            let (mut held, mut held_in) = (0, std::collections::BTreeSet::new());
            let mut counts: Vec<Vec<(String, u64)>> =
                funnels.iter().map(|f| f.labels().into_iter().map(|l| (l.to_owned(), 0)).collect()).collect();
            for (deal, frame) in &frames {
                if !compiled.evaluate_viewer(frame, &mut budget)?.is_empty() {
                    held += 1;
                    held_in.extend(*deal);
                }
                for (funnel, counts) in funnels.iter().zip(&mut counts) {
                    for count in counts.iter_mut().take(funnel.depth(frame, &mut budget)?) {
                        count.1 += 1;
                    }
                }
            }
            println!("coverage of query {path} over {what}");
            for line in &how {
                println!("  {line}");
            }
            println!("  held at {held} of {}{}", frames.len(), in_deals(&held_in));
            for (i, pattern) in counts.iter().enumerate() {
                if counts.len() > 1 {
                    println!("    pattern {}", i + 1);
                }
                show(pattern, frames.len() as u64);
            }
        }
        Program::Dominance(_) => return Err(fail(format!("{path} is a dominance: silas query {path}"))),
    }
    Ok(true)
}

/// Decide a dominance over the 43 outcomes of a hand of 42 points (the
/// declaration enters only through its total).
fn dominate(dominance: &Dominance) -> Result<bool> {
    let declaration = Straight42::ALL[0].physics();
    let decision = dominance.decide(&declaration)?;
    let spelled = |bid: Bid| match bid {
        Bid::Points(n) => format!("(points {n})"),
        Bid::Marks(m) => format!("(marks {m})"),
    };
    let [first, second] = dominance.contracts;
    println!("dominates {}: {} over {}, bid by S0", dominance.name, spelled(first), spelled(second));
    let place = match dominance.place {
        Place::Any => "anywhere short of the target".to_owned(),
        Place::Short(n) => format!("short {n}"),
        Place::ShortAtMost(n) => format!("short {n} or less"),
        Place::Score(a, b) => format!("at {a} to {b}"),
    };
    println!(
        "  {}, the bidder's side {place}: {} states, {} outcomes",
        dominance.scoring, decision.states, decision.outcomes
    );
    let by = match dominance.criterion {
        Criterion::Marks => "the hand's net marks",
        Criterion::Points => "the hand's net points",
        Criterion::MatchWin => "the match-win order (at least as good under every match-win value)",
    };
    println!("  by {by}");
    if !decision.readings.is_empty() {
        let readings: Vec<String> = decision.readings.iter().map(SetReading::to_string).collect();
        println!("  a set marks bid read as {}", readings.join(" and as "));
    }
    match &decision.counterexample {
        None => println!("  holds ({})", decision.authority()),
        Some(c) => println!("  fails ({}): {c}", decision.authority()),
    }
    Ok(true)
}

/// Print each file's core: every definition expanded and every short form
/// spelled out, as the evaluator reads it.
fn expand(args: &[String]) -> Result<bool> {
    let options = Options::parse(args, &[], &[])?;
    if options.words.is_empty() {
        return Err(fail("silas expand FILE ..."));
    }
    for path in &options.words {
        match Program::parse(&read(path)?, path)? {
            Program::Query(query) => println!("{query}"),
            Program::Policy(policy) => println!("{policy}"),
            Program::Dominance(dominance) => println!("{dominance}"),
        }
    }
    Ok(true)
}

fn fmt(args: &[String]) -> Result<bool> {
    let options = Options::parse(args, &["check"], &[])?;
    let check = options.flags.iter().any(|f| f == "check");
    if options.words.is_empty() {
        return Err(fail("silas fmt [--check] FILE ..."));
    }
    let mut clean = true;
    for path in &options.words {
        let text = read(path)?;
        let formatted = format(&text, path)?;
        if formatted != text {
            clean = false;
            if check {
                println!("{path}: not formatted");
            } else {
                std::fs::write(path, formatted).map_err(|e| fail(format!("{path}: {e}")))?;
                println!("{path}: formatted");
            }
        }
    }
    Ok(clean || !check)
}

/// The `.silas` files under `path`, or `path` itself when it is a file.
fn silas_files(path: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        out.push(path.to_owned());
        return Ok(());
    }
    let entries = std::fs::read_dir(path).map_err(|e| fail(format!("{}: {e}", path.display())))?;
    for entry in entries {
        let path = entry.map_err(|e| fail(e.to_string()))?.path();
        if path.is_dir() {
            silas_files(&path, out)?;
        } else if path.extension().is_some_and(|x| x == "silas") {
            out.push(path);
        }
    }
    Ok(())
}

/// Compile every file (each policy under all nine straight declarations) and
/// print its warnings. A warning does not fail the check.
fn check(args: &[String]) -> Result<bool> {
    let options = Options::parse(args, &[], &[])?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dirs: Vec<PathBuf> = if options.words.is_empty() {
        vec![root.join("examples"), root.join("lib")]
    } else {
        options.words.iter().map(PathBuf::from).collect()
    };
    let mut files = Vec::new();
    for dir in &dirs {
        silas_files(dir, &mut files)?;
    }
    files.sort();
    let registry = Registry::standard();
    let (mut ok, mut failed, mut warned) = (0, 0, 0);
    for path in &files {
        let name = path.display().to_string();
        let text = read(&name)?;
        let checked = (|| -> Result<(&str, Vec<Warning>)> {
            let program = match Program::parse(&text, &name) {
                Ok(program) => program,
                Err(e) => {
                    // A library: definitions only, read as a query's would be.
                    let library = Query::parse(&format!("{text}\n(query)"), &name);
                    return library.and_then(|q| q.compile(&registry)).map(|_| ("library", vec![])).map_err(|_| e);
                }
            };
            match program {
                Program::Query(query) => {
                    query.compile(&registry)?;
                    Ok(("query", query.lint()))
                }
                Program::Policy(policy) => {
                    for rules in Straight42::ALL {
                        compile_at(&policy, &name, &registry, &rules.physics())?;
                    }
                    Ok(("policy", policy.lint()))
                }
                Program::Dominance(dominance) => {
                    for rules in Straight42::ALL {
                        dominance.contracts(&rules.physics())?;
                    }
                    Ok(("dominance", vec![]))
                }
            }
        })();
        match checked {
            Ok((kind, warnings)) => {
                ok += 1;
                println!("ok      {kind:<9} {name}");
                for warning in &warnings {
                    println!("{warning}");
                }
                warned += warnings.len();
            }
            Err(e) => {
                failed += 1;
                println!("FAILED  {name}\n{e}");
            }
        }
    }
    println!("{ok} compiled, {failed} failed, {warned} warnings");
    Ok(failed == 0)
}
