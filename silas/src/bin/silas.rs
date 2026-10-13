//! `silas`: the language at the command line, over the book's hands.
//!
//! ```text
//! silas eval     POLICY --hand ID [--trick N] [--seat ROLE] [--book FILE]
//! silas query    QUERY --hand ID [--trick N] [--seat ROLE] [--cap N] [--book FILE]
//! silas query    DOMINANCE
//! silas coverage POLICY-OR-QUERY [--hand ID] [--random N] [--book FILE]
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
//! allowed set. `query` prints the query's exact probability over the worlds
//! the seat cannot tell apart, with its authority, resolving only the tiles
//! it reads. On a `(dominates ...)` file it needs no hand: it decides the form
//! by enumerating the hand's outcomes, exactly, and prints the first
//! counterexample when there is one. `coverage` reads a policy (or a query of
//! the viewer's frame) at every point of the book's narrated lines, and of
//! `--random` seeded deals, and counts how often each rule answered, allowed a
//! legal tile and decided, and how often each step of each pattern held: a
//! zero is a rule or a clause that never holds there. `expand` prints a
//! file's core: definitions expanded, short forms spelled out. `fmt` formats
//! canonically, in place (or reports with `--check`). `check` compiles every
//! `.silas` file (policies under all nine straight declarations) and prints
//! its warnings. An option the command does not take is refused.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

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
  silas query    QUERY --hand ID [--trick N] [--seat ROLE] [--cap N] [--book FILE]
  silas query    DOMINANCE
  silas coverage POLICY-OR-QUERY [--hand ID] [--random N] [--book FILE]
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
        Some("query") => query(rest),
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

fn read(path: &str) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| fail(format!("{path}: {e}")))
}

/// One hand of the book: its contract with the bidder in S0, the four hands
/// it gives (only the bidder's, for a hand the book gives alone), and the
/// plays its narrated line names, up to the first it does not.
struct Hand {
    id: String,
    contract: Contract,
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
        let contract = Contract::new(Seat::S0, Bid::number(bid as u32)?, rules.physics())?;
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
        out.push(Hand { id, contract, hands, full: example["kind"] == "deal", narrated });
    }
    Ok(out)
}

impl Hand {
    /// The bidder's frame before the opening lead.
    fn opening(&self) -> Result<Frame> {
        Frame::new(self.contract.clone(), Seat::S0, self.hands[0], vec![])
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
            match Frame::new(self.contract.clone(), viewer, self.hands[viewer.index()], self.narrated[..i].to_vec()) {
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

fn eval(args: &[String]) -> Result<bool> {
    let options = Options::parse(args, &[], &["hand", "trick", "seat", "policy", "book"])?;
    let path = file(&options, "silas eval POLICY --hand ID [--trick N] [--seat ROLE]")?;
    let program = policy_program(path)?;
    let (frame, heading) = position(&options)?;
    let registry = Registry::standard();
    let policy = program.compile(&registry, frame.physics())?;
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

fn query(args: &[String]) -> Result<bool> {
    let options = Options::parse(args, &[], &["hand", "trick", "seat", "cap", "book"])?;
    let path = file(&options, "silas query FILE --hand ID [--trick N] [--seat ROLE] [--cap N]")?;
    let text = read(path)?;
    let query = match Program::parse(&text, path)? {
        Program::Query(query) => query.compile(&Registry::standard())?,
        Program::Dominance(dominance) => return dominate(&dominance),
        Program::Policy(_) => return Err(fail(format!("{path} is a policy: silas eval {path} --hand ID ..."))),
    };
    let (frame, heading) = position(&options)?;
    describe(&frame, &heading);
    let cap = options.number("cap", 100_000)?;
    let mut budget = Budget::new(1_000_000_000);
    let belief = Belief::uniform_for(frame.clone(), &query, cap, &mut budget)?;
    let summary = query.summarize(&belief, &mut budget)?;
    let p = summary.event_probability();
    let resolved = frame.kernel().pool().difference(belief.unresolved());
    println!("query {path}");
    println!(
        "  {} worlds, as {} points resolving {}",
        frame.count_worlds(),
        belief.len(),
        if resolved.is_empty() { "nothing".to_owned() } else { tiles(resolved) }
    );
    println!("  P(answer) = {} ({})", p.value(), p.authority());
    for answer in summary.answer_presence_mass.keys().filter(|a| !a.0.is_empty()) {
        let shown: Vec<String> = answer.0.iter().map(Value::to_string).collect();
        let p = summary.presence_probability(answer);
        println!("  P({}) = {} ({})", shown.join(" "), p.value(), p.authority());
    }
    Ok(true)
}

/// The frames coverage reads: every point of the book's narrated lines (or
/// of one hand's), seen by the seat to play, and the opening of each hand
/// the book gives alone; then `random` seeded deals, under the nine straight
/// declarations in turn, played out by uniformly random legal plays.
fn coverage_frames(options: &Options) -> Result<(Vec<Frame>, String)> {
    let hands = match options.get("hand") {
        Some(_) => vec![hand(options)?],
        None => book(options)?,
    };
    // A line the record refuses is read up to the refusal, and named.
    let mut frames = Vec::new();
    let mut partial = Vec::new();
    for hand in &hands {
        let (points, refused) = hand.points();
        frames.extend(points.into_iter().map(|(_, frame)| frame));
        if refused.is_some() {
            partial.push(hand.id.clone());
        }
    }
    let book_frames = frames.len();
    let random = options.number("random", 0)?;
    let mut rng = 0x5eed_u64;
    let mut next = |n: usize| {
        rng = rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) % n as u64) as usize
    };
    for game in 0..random {
        let physics = Straight42::ALL[game % 9].physics();
        let mut deck = Domino::ALL.to_vec();
        for i in (1..28).rev() {
            deck.swap(i, next(i + 1));
        }
        let deal: [DominoSet; 4] = std::array::from_fn(|s| deck[7 * s..7 * s + 7].iter().copied().collect());
        let contract = Contract::new(Seat::ALL[next(4)], Bid::number(30 + next(13) as u32)?, physics.clone())?;
        let mut record = Frame::new(contract.clone(), Seat::S0, deal[0], vec![])?;
        let mut hands = deal;
        while let Some(viewer) = record.next_actor() {
            frames.push(Frame::new(contract.clone(), viewer, deal[viewer.index()], record.plays().to_vec())?);
            let legal = physics.legal_plays(hands[viewer.index()], record.lead());
            let tile = legal.iter().nth(next(legal.len())).expect("a legal play");
            hands[viewer.index()].remove(tile);
            record = record.after(tile)?;
        }
    }
    let what = match options.get("hand") {
        Some(id) => id.to_owned(),
        None if partial.is_empty() => format!("the book's {} hands", hands.len()),
        None => format!(
            "the book's {} hands ({} read only up to a play the record refuses)",
            hands.len(),
            partial.join(", ")
        ),
    };
    Ok((frames, format!("{book_frames} frames of {what}, and {random} random deals")))
}

/// How often each rule answered, allowed a legal tile and decided, and where
/// each pattern stops: the frames where each step of its funnel holds. A
/// count of zero is a rule or a clause that never holds on these frames.
fn coverage(args: &[String]) -> Result<bool> {
    let options = Options::parse(args, &[], &["hand", "random", "book"])?;
    let path = file(&options, "silas coverage FILE [--hand ID] [--random N]")?;
    let registry = Registry::standard();
    let (frames, what) = coverage_frames(&options)?;
    let mut budget = Budget::new(u64::MAX);
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
            let mut coverage = None;
            for frame in &frames {
                let id = frame.physics().identity().to_owned();
                if !compiled.contains_key(&id) {
                    compiled.insert(id.clone(), program.compile(&registry, frame.physics())?);
                }
                let policy = &compiled[&id];
                if coverage.is_none() {
                    coverage = Some(Coverage::new(policy, &registry)?);
                }
                coverage.as_mut().expect("made").observe(policy, frame, &mut budget)?;
            }
            println!("coverage of policy {} over {what}", program.name);
            let Some(coverage) = coverage else { return Ok(true) };
            let of = coverage.frames;
            for rule in &coverage.rules {
                let flag = if rule.decided == 0 { "!" } else { " " };
                println!(
                    "{flag} rule {}: decided {}, allowed {}, answered {} of {of}",
                    rule.name, rule.decided, rule.allowed, rule.answered
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
            let mut held = 0;
            let mut counts: Vec<Vec<(String, u64)>> =
                funnels.iter().map(|f| f.labels().into_iter().map(|l| (l.to_owned(), 0)).collect()).collect();
            for frame in &frames {
                held += u64::from(!compiled.evaluate_viewer(frame, &mut budget)?.is_empty());
                for (funnel, counts) in funnels.iter().zip(&mut counts) {
                    for count in counts.iter_mut().take(funnel.depth(frame, &mut budget)?) {
                        count.1 += 1;
                    }
                }
            }
            println!("coverage of query {path} over {what}");
            println!("  held at {held} of {}", frames.len());
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
                        policy.compile(&registry, &rules.physics())?;
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
