//! `silas`: the language at the command line, over the book's hands.
//!
//! ```text
//! silas eval  --hand ID [--trick N] [--seat ROLE] --policy FILE [--book FILE]
//! silas query FILE --hand ID [--trick N] [--seat ROLE] [--cap N] [--book FILE]
//! silas fmt   [--check] FILE ...
//! silas check [DIR ...]
//! ```
//!
//! A position is a hand of `examples/book/hands.json` (or `--book`): a full
//! deal at the point of its narrated line (its tricks, then its tail, up to
//! the first play the book does not name) where `--seat` (bidder, left,
//! partner, right) is to play to trick `--trick` (default: the opening lead),
//! seen from that seat; or a hand the book gives alone, seen by its bidder
//! before the first lead. The bidder sits in S0.
//!
//! `eval` prints the deciding layer of the policy, the fired rule's witnesses
//! (each answer and one binding of the guard's roles) and the allowed set.
//! `query` prints the query's exact probability over the worlds the seat
//! cannot tell apart, with its authority, resolving only the tiles it reads.
//! `fmt` formats canonically, in place (or reports with `--check`). `check`
//! compiles every `.silas` file (policies under all nine straight
//! declarations).

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

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("eval") => eval(&args[1..]),
        Some("query") => query(&args[1..]),
        Some("fmt") => fmt(&args[1..]),
        Some("check") => check(&args[1..]),
        _ => Err(fail(
            "usage: silas eval --hand ID [--trick N] [--seat ROLE] --policy FILE | silas query FILE --hand ID \
             [--trick N] [--seat ROLE] [--cap N] | silas fmt [--check] FILE ... | silas check [DIR ...]",
        )),
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
    fn parse(args: &[String], flags: &[&str]) -> Result<Options> {
        let mut out = Options { named: vec![], flags: vec![], words: vec![] };
        let mut it = args.iter();
        while let Some(arg) = it.next() {
            match arg.strip_prefix("--") {
                Some(name) if flags.contains(&name) => out.flags.push(name.to_owned()),
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

fn book_path(options: &Options) -> String {
    options
        .get("book")
        .map_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/examples/book/hands.json").to_owned(), str::to_owned)
}

/// A book position: the frame, and a line describing it.
fn position(options: &Options) -> Result<(Frame, String)> {
    let path = book_path(options);
    let json: Json = serde_json::from_str(&read(&path)?).map_err(|e| fail(format!("{path}: {e}")))?;
    let id = options.need("hand")?;
    let empty = vec![];
    let examples = json["examples"].as_array().unwrap_or(&empty);
    let example = examples.iter().find(|e| e["id"] == id).ok_or_else(|| fail(format!("no hand {id} in {path}")))?;
    let lines = json["lines"].as_array().unwrap_or(&empty);
    let line = lines.iter().find(|l| l["example"] == id && l["source"] == "prose");
    let text = |v: &Json, what: &str| v.as_str().map(str::to_owned).ok_or_else(|| fail(format!("{id}: no {what}")));
    let declaration = text(&example["declaration"], "declaration")
        .or_else(|_| text(line.map_or(&Json::Null, |l| &l["declaration"]), "declaration"))?;
    let rules: Straight42 = declaration.parse()?;
    let bid = example["bid"]
        .as_u64()
        .or_else(|| line.and_then(|l| l["bid"].as_u64()))
        .ok_or_else(|| fail(format!("{id}: no numeric bid")))?;
    let contract = Contract::new(Seat::S0, Bid::number(bid as u32)?, rules.physics())?;
    let mut hands = [DominoSet::EMPTY; 4];
    for seat in example["seats"].as_array().unwrap_or(&empty) {
        let role = text(&seat["role"], "role")?;
        let at = BOOK_ROLES.iter().position(|r| *r == role).ok_or_else(|| fail(format!("{id}: a role {role}")))?;
        let tiles = seat["tiles"].as_array().unwrap_or(&empty);
        hands[at] = tiles.iter().filter_map(Json::as_str).map(str::parse).collect::<Result<_>>()?;
    }
    let trick = options.number("trick", 1)?;
    let full = example["kind"] == "deal";
    let seat = match options.get("seat") {
        Some(name) => Some(
            ROLES
                .iter()
                .position(|r| *r == name)
                .ok_or_else(|| fail(format!("--seat is one of {}", ROLES.join(", "))))?,
        ),
        None => None,
    };
    if !full {
        if trick != 1 || seat.is_some_and(|s| s != 0) {
            return Err(fail(format!("{id} gives only the bidder's hand: its position is the opening lead")));
        }
        let frame = Frame::new(contract, Seat::S0, hands[0], vec![])?;
        return Ok((frame, format!("{id}: the bidder's hand before the opening lead")));
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
    // The first point of the line where the seat is to play to the trick.
    let mut frame = Frame::new(contract.clone(), Seat::S0, hands[0], vec![])?;
    let mut at = None;
    for i in 0..=narrated.len() {
        let next = frame.next_actor();
        if i / 4 + 1 == trick && next.is_some_and(|s| seat.is_none_or(|r| s.index() == r)) {
            at = Some((i, next.expect("a seat to play")));
            break;
        }
        if i < narrated.len() {
            frame = frame.after(narrated[i]).map_err(|e| fail(format!("{id}: the narrated play {}: {e}", i + 1)))?;
        }
    }
    let (i, viewer) =
        at.ok_or_else(|| fail(format!("{id}'s narrated line does not reach that seat in trick {trick}")))?;
    let frame = Frame::new(contract, viewer, hands[viewer.index()], narrated[..i].to_vec())?;
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

fn eval(args: &[String]) -> Result<bool> {
    let options = Options::parse(args, &[])?;
    let (frame, heading) = position(&options)?;
    let path = options.need("policy")?;
    let program = PolicyProgram::parse(&read(path)?, path)?;
    let policy = program.compile(&Registry::standard(), frame.physics())?;
    describe(&frame, &heading);
    let legal = frame.physics().legal_plays(frame.kernel().viewer_hand(), frame.lead());
    println!("  legal: {}", tiles(legal));
    let (allowed, witnesses) = policy.explain(&frame, &mut Budget::new(100_000_000))?;
    println!("policy {}", program.name);
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
    let options = Options::parse(args, &[])?;
    let [path] = options.words.as_slice() else {
        return Err(fail("silas query FILE --hand ID ..."));
    };
    let (frame, heading) = position(&options)?;
    let query = Query::parse(&read(path)?, path)?.compile(&Registry::standard())?;
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

fn fmt(args: &[String]) -> Result<bool> {
    let options = Options::parse(args, &["check"])?;
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

fn silas_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(dir).map_err(|e| fail(format!("{}: {e}", dir.display())))?;
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

fn check(args: &[String]) -> Result<bool> {
    let options = Options::parse(args, &[])?;
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
    let (mut ok, mut failed) = (0, 0);
    for path in &files {
        let name = path.display().to_string();
        let text = read(&name)?;
        let checked = (|| -> Result<&str> {
            let program = match Program::parse(&text, &name) {
                Ok(program) => program,
                Err(e) => {
                    // A library: definitions only, read as a query's would be.
                    let library = Query::parse(&format!("{text}\n(query (roles) (out))"), &name);
                    return library.and_then(|q| q.compile(&registry)).map(|_| "library").map_err(|_| e);
                }
            };
            match program {
                Program::Query(query) => {
                    query.compile(&registry)?;
                    Ok("query")
                }
                Program::Policy(policy) => {
                    for rules in Straight42::ALL {
                        policy.compile(&registry, &rules.physics())?;
                    }
                    Ok("policy")
                }
            }
        })();
        match checked {
            Ok(kind) => {
                ok += 1;
                println!("ok      {kind:<7} {name}");
            }
            Err(e) => {
                failed += 1;
                println!("FAILED  {name}\n{e}");
            }
        }
    }
    println!("{ok} compiled, {failed} failed");
    Ok(failed == 0)
}
