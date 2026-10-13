//! Grade every narrated play of the book's eight full deals (Hands 28-35)
//! double-dummy: the solver sees all four hands, so a play it calls a mistake
//! is a mistake only for a seat that could see them too.
//!
//! Reads `examples/book/hands.json` (or the path given as the first argument):
//! each deal's four hands by role and its narrated (`prose`) line, the bidder
//! leading the first trick. The bidder sits in S0. A marks bid is the whole 42.
//! A play the prose does not name ends the graded line.
//!
//! `cargo run --release --example solve_grade [hands.json]`

use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;
use silas::reference::Straight42;
use silas::*;

const ROLES: [&str; 4] = ["bidder", "opponent-1", "partner", "opponent-2"];
const DEALS: [&str; 8] = ["hand-28", "hand-29", "hand-30", "hand-31", "hand-32", "hand-33", "hand-34", "hand-35"];

fn role(name: &str) -> Result<usize> {
    ROLES.iter().position(|r| *r == name).ok_or_else(|| Error(format!("unknown role {name}")))
}

fn text<'a>(v: &'a Value, what: &str) -> Result<&'a str> {
    v.as_str().ok_or_else(|| Error(format!("{what} is not a string: {v}")))
}

/// The deal and its narrated line, each play with the role the prose gives.
fn read(examples: &[Value], lines: &[Value], id: &str) -> Result<(Deal, Vec<(usize, Domino)>)> {
    let example = examples.iter().find(|e| e["id"] == id).ok_or_else(|| Error(format!("no example {id}")))?;
    let line = lines
        .iter()
        .find(|l| l["example"] == id && l["source"] == "prose")
        .ok_or_else(|| Error(format!("no narrated line for {id}")))?;
    let mut hands = [DominoSet::EMPTY; 4];
    for seat in example["seats"].as_array().into_iter().flatten() {
        let tiles = seat["tiles"].as_array().into_iter().flatten();
        let at = role(text(&seat["role"], "a role")?)?;
        hands[at] = tiles.map(|t| text(t, "a tile")?.parse()).collect::<Result<_>>()?;
    }
    let decl: Straight42 = text(&line["declaration"], "the declaration")?.parse()?;
    let bid = line["bid"].as_u64().ok_or_else(|| Error(format!("{id}: no numeric bid")))?;
    let contract = Contract { declarer: Seat::S0, bid: bid.min(42) as u32 };
    let tricks = line["tricks"].as_array().into_iter().flatten().map(|t| &t["plays"]);
    let plays = tricks.chain(std::iter::once(&line["tail"]["plays"])).flat_map(|p| p.as_array().into_iter().flatten());
    let mut narrated = Vec::new();
    for play in plays {
        match (play[0].as_str(), play[1].as_str()) {
            (Some(r), Some(t)) => narrated.push((role(r)?, t.parse()?)),
            _ => break,
        }
    }
    Ok((Deal { physics: Arc::new(decl), contract, hands }, narrated))
}

fn shown(set: DominoSet) -> String {
    set.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" ")
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/examples/book/hands.json").to_owned());
    let source = std::fs::read_to_string(&path).map_err(|e| Error(format!("{path}: {e}")))?;
    let json: Value = serde_json::from_str(&source).map_err(|e| Error(format!("{path}: {e}")))?;
    let (examples, lines) = (json["examples"].as_array().cloned(), json["lines"].as_array().cloned());
    let (examples, lines) = (examples.unwrap_or_default(), lines.unwrap_or_default());
    println!("Double-dummy grades of the narrated plays: a reference that sees all four hands.");
    println!("margin = bidders' points less the opponents', from the bidders' side; make = 1 if the bid is made.");
    for id in DEALS {
        let (deal, narrated) = read(&examples, &lines, id)?;
        let started = Instant::now();
        let physics = deal.physics.as_ref();
        let opening = OpenPosition::deal(physics, deal.contract, deal.hands)?;
        let value = Solver::new(physics).value(&opening, Objective::Margin);
        let opening_time = started.elapsed();
        // The line is graded up to its first illegal play, if it has one.
        let mut line: Vec<Domino> = narrated.iter().map(|(_, d)| *d).collect();
        let mut position = opening.clone();
        let mut broken = None;
        for (ply, d) in line.iter().enumerate() {
            match position.play(physics, *d) {
                Ok(next) => position = next,
                Err(e) => {
                    broken = Some(format!(
                        "play {} (T{}) is illegal ({e}); the line is graded to there",
                        ply + 1,
                        ply / 4 + 1
                    ));
                    line.truncate(ply);
                    break;
                }
            }
        }
        let grades = grade_line(&deal, &line)?;
        println!(
            "\n{id}: {}, the bidders need {}; double-dummy from the opening lead: margin {value:+} ({} to the bidders); \
             solved in {:.1?}, line graded in {:.1?}",
            physics.identity(),
            deal.contract.bid,
            (42 + value) / 2,
            opening_time,
            started.elapsed()
        );
        let mut decisions = 0;
        let mut misses = (0, 0);
        for (g, (r, _)) in grades.iter().zip(&narrated) {
            let seat_role = ROLES[g.seat.index()];
            if *r != g.seat.index() {
                return Err(Error(format!("{id} ply {}: the prose gives {} to {}", g.ply, g.tile, ROLES[*r])));
            }
            if g.legal.len() < 2 {
                continue;
            }
            decisions += 1;
            misses.0 += usize::from(!g.margin.optimal());
            misses.1 += usize::from(!g.make.optimal());
            let flag = match (g.margin.optimal(), g.make.optimal()) {
                (true, true) => "",
                (false, true) => "  points short",
                (true, false) => "  CONTRACT",
                (false, false) => "  points short, CONTRACT",
            };
            println!(
                "  T{} {:<10} {}  margin {:+3} best {:+3} [{}]  make {} best {} [{}]{flag}",
                g.ply / 4 + 1,
                seat_role,
                g.tile,
                g.margin.given,
                g.margin.best,
                shown(g.margin.tied),
                g.make.given,
                g.make.best,
                shown(g.make.tied),
            );
        }
        println!(
            "  {} narrated plays, {decisions} with a choice: {} short of the best margin, {} turning the contract",
            grades.len(),
            misses.0,
            misses.1
        );
        if let Some(note) = broken {
            println!("  {note}");
        }
    }
    Ok(())
}
