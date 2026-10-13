//! The book's illustrated examples (`examples/book/hands.json`), read as data:
//! every hand and deal the book draws, and the plays its prose narrates. Facts
//! and capture anchors only; see the file's `about` and `notation`.

use std::sync::OnceLock;

use serde_json::Value;
use silas::*;

const SOURCE: &str = include_str!("../../examples/book/hands.json");

/// A seat by the role the book gives it, in play order from the bidder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Bidder,
    Opponent1,
    Partner,
    Opponent2,
}

impl Role {
    pub const ALL: [Role; 4] = [Role::Bidder, Role::Opponent1, Role::Partner, Role::Opponent2];

    /// Clockwise from the bidder.
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn at(i: usize) -> Role {
        Role::ALL[i % 4]
    }

    /// 0 for the bidder's side, 1 for the setters.
    pub fn side(self) -> usize {
        self.index() % 2
    }

    fn parse(s: &str) -> Role {
        match s {
            "bidder" => Role::Bidder,
            "opponent-1" => Role::Opponent1,
            "partner" => Role::Partner,
            "opponent-2" => Role::Opponent2,
            _ => panic!("not a role: {s}"),
        }
    }
}

/// Who plays a tile in a narrated line: a role, an opponent the prose does not
/// identify, or a seat it does not name at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Who {
    Seat(Role),
    Setter,
    Unnamed,
}

impl Who {
    fn parse(v: &Value) -> Who {
        match v.as_str() {
            None => Who::Unnamed,
            Some("setter") => Who::Setter,
            Some(s) => Who::Seat(Role::parse(s)),
        }
    }

    pub fn side(self) -> Option<usize> {
        match self {
            Who::Seat(r) => Some(r.side()),
            Who::Setter => Some(1),
            Who::Unnamed => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SeatHand {
    pub role: Role,
    /// In printed order; `None` is an empty position of a later state.
    pub tiles: Vec<Option<Domino>>,
    /// Parallel to `tiles`; full deals only.
    pub marks: Vec<Option<u8>>,
}

impl SeatHand {
    pub fn set(&self) -> DominoSet {
        self.tiles.iter().flatten().copied().collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Bid {
    Points(u32),
    Pass,
    Unstated,
}

impl Bid {
    fn parse(v: &Value) -> Bid {
        match v {
            Value::Null => Bid::Unstated,
            Value::String(s) if s == "pass" => Bid::Pass,
            _ => Bid::Points(v.as_u64().expect("a bid") as u32),
        }
    }

    /// The points the bid needs: a marks bid (84 and up) needs all 42.
    pub fn contract_points(&self) -> u32 {
        match self {
            Bid::Points(b) => (*b).min(42),
            _ => panic!("no bid to play: {self:?}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Example {
    pub id: String,
    pub kind: String,
    pub seats: Vec<SeatHand>,
    pub declaration: Option<String>,
    pub bid: Bid,
    pub captures: Vec<String>,
    pub prose: Vec<String>,
}

impl Example {
    pub fn seat(&self, role: Role) -> Option<&SeatHand> {
        self.seats.iter().find(|s| s.role == role)
    }

    /// The bidder's seven tiles.
    pub fn hand(&self) -> DominoSet {
        self.seat(Role::Bidder).expect("a bidder").set()
    }

    /// A full deal in role order.
    pub fn deal(&self) -> [DominoSet; 4] {
        assert_eq!(self.kind, "deal", "{} is not a full deal", self.id);
        Role::ALL.map(|r| self.seat(r).expect("every role").set())
    }

    pub fn physics(&self) -> std::sync::Arc<dyn Physics> {
        super::physics(self.declaration.as_deref().expect("a straight declaration"))
    }
}

#[derive(Clone, Debug)]
pub struct Trick {
    pub anchor: String,
    pub leader: Who,
    pub plays: Vec<(Who, Option<Domino>)>,
    pub winner: Option<Who>,
    pub points: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct Tail {
    pub winner: Who,
    pub plays: Vec<(Who, Option<Domino>)>,
}

#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub after: usize,
    pub side: usize,
    pub points: u32,
}

#[derive(Clone, Debug)]
pub struct Line {
    pub id: String,
    pub example: String,
    pub source: String,
    pub declaration: Option<String>,
    pub bid: Bid,
    pub tricks: Vec<Trick>,
    pub tail: Option<Tail>,
    /// Stated final points, by side: [bidder's side, setters].
    pub totals: [Option<u32>; 2],
    pub checkpoints: Vec<Checkpoint>,
    /// For an annotation line, the diagram states whose marks it reads.
    pub states: Vec<String>,
    pub expect: Value,
}

impl Line {
    /// Every named tile of the line's tricks, in narrated order.
    pub fn plays(&self) -> Vec<Domino> {
        self.tricks.iter().flat_map(|t| t.plays.iter().map(|(_, d)| d.expect("a named tile"))).collect()
    }
}

pub struct Book {
    pub examples: Vec<Example>,
    pub lines: Vec<Line>,
}

impl Book {
    pub fn example(&self, id: &str) -> &Example {
        self.examples.iter().find(|e| e.id == id).unwrap_or_else(|| panic!("no example {id}"))
    }

    pub fn line(&self, id: &str) -> &Line {
        self.lines.iter().find(|l| l.id == id).unwrap_or_else(|| panic!("no line {id}"))
    }
}

pub fn book() -> &'static Book {
    static BOOK: OnceLock<Book> = OnceLock::new();
    BOOK.get_or_init(|| parse(&serde_json::from_str(SOURCE).expect("hands.json is JSON")))
}

fn tile(v: &Value) -> Option<Domino> {
    v.as_str().map(|s| s.parse().unwrap_or_else(|e| panic!("{e:?}")))
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().map(|s| s.as_str().unwrap().to_owned()).collect()).unwrap_or_default()
}

fn plays(v: &Value) -> Vec<(Who, Option<Domino>)> {
    v.as_array().unwrap().iter().map(|p| (Who::parse(&p[0]), tile(&p[1]))).collect()
}

fn parse(root: &Value) -> Book {
    let examples = root["examples"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| Example {
            id: e["id"].as_str().unwrap().to_owned(),
            kind: e["kind"].as_str().unwrap().to_owned(),
            seats: e["seats"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| {
                    let tiles: Vec<Option<Domino>> = s["tiles"].as_array().unwrap().iter().map(tile).collect();
                    let marks = match s.get("annotations") {
                        Some(a) => a.as_array().unwrap().iter().map(|m| m.as_u64().map(|m| m as u8)).collect(),
                        None => vec![None; tiles.len()],
                    };
                    SeatHand { role: Role::parse(s["role"].as_str().unwrap()), tiles, marks }
                })
                .collect(),
            declaration: e["declaration"].as_str().map(str::to_owned),
            bid: Bid::parse(&e["bid"]),
            captures: strings(&e["captures"]),
            prose: strings(&e["prose"]),
        })
        .collect();
    let lines = root["lines"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| {
            let side_total = |k: &str| l["totals"].get(k).and_then(Value::as_u64).map(|p| p as u32);
            Line {
                id: l["id"].as_str().unwrap().to_owned(),
                example: l["example"].as_str().unwrap().to_owned(),
                source: l["source"].as_str().unwrap().to_owned(),
                declaration: l.get("declaration").and_then(Value::as_str).map(str::to_owned),
                bid: l.get("bid").map(Bid::parse).unwrap_or(Bid::Unstated),
                tricks: l
                    .get("tricks")
                    .and_then(Value::as_array)
                    .map(|ts| {
                        ts.iter()
                            .map(|t| Trick {
                                anchor: t["anchor"].as_str().unwrap().to_owned(),
                                leader: Who::parse(&t["leader"]),
                                plays: plays(&t["plays"]),
                                winner: t.get("winner").map(Who::parse),
                                points: t.get("points").and_then(Value::as_u64).map(|p| p as u32),
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                tail: l.get("tail").map(|t| Tail { winner: Who::parse(&t["winner"]), plays: plays(&t["plays"]) }),
                totals: [side_total("bidder"), side_total("setters")],
                checkpoints: l
                    .get("checkpoints")
                    .and_then(Value::as_array)
                    .map(|cs| {
                        cs.iter()
                            .map(|c| Checkpoint {
                                after: c["after"].as_u64().unwrap() as usize,
                                side: if c["side"] == "bidder" { 0 } else { 1 },
                                points: c["points"].as_u64().unwrap() as u32,
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                states: strings(&l["states"]),
                expect: l["expect"].clone(),
            }
        })
        .collect();
    Book { examples, lines }
}
