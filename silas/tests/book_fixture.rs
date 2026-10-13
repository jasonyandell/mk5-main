//! The book's illustrated examples as data (`examples/book/hands.json`),
//! checked mechanically: every hand is seven distinct tiles, every full deal
//! is the whole set, and every narrated line replays under `Straight42`
//! (follow suit, the winner leads next, trick points) to the winners and
//! totals the prose states. Where a diagram's printed marks order a seat's
//! plays differently, the marks are replayed too and their outcome is
//! asserted as the fixture records it, legal or not.
mod common;

use common::fixture::{Bid, Book, Example, Line, Role, Who, book};
use serde_json::Value;
use silas::*;

/// A replay step: on, or stopped with its outcome.
type Step = std::result::Result<(), Outcome>;

const SEATS: usize = 4;
/// Seven tricks and the 35 count.
const ALL_POINTS: u32 = 42;

/// How a line ends.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Outcome {
    /// Every check passed.
    Legal,
    /// A seat plays a tile another seat's diagram holds.
    Conflict { trick: usize, seat: Role, tile: Domino },
    /// A seat does not follow suit while holding a follower.
    Illegal { trick: usize, seat: Role, tile: Domino },
    /// Annotations: no seat marks this trick.
    Ends { trick: usize },
    /// Annotations: this seat, due to play, marks no tile for the trick.
    Unmarked { trick: usize, seat: Role },
}

impl Outcome {
    fn expected(v: &Value) -> Outcome {
        let trick = || v["trick"].as_u64().unwrap() as usize;
        let seat = || role(v["seat"].as_str().unwrap());
        let tile = || v["tile"].as_str().unwrap().parse().unwrap();
        match v["result"].as_str().unwrap() {
            "legal" | "complete" => Outcome::Legal,
            "conflict" => Outcome::Conflict { trick: trick(), seat: seat(), tile: tile() },
            "illegal" => Outcome::Illegal { trick: trick(), seat: seat(), tile: tile() },
            "ends" => Outcome::Ends { trick: trick() },
            "unmarked" => Outcome::Unmarked { trick: trick(), seat: seat() },
            other => panic!("unknown result {other}"),
        }
    }
}

fn role(s: &str) -> Role {
    Role::ALL[["bidder", "opponent-1", "partner", "opponent-2"].iter().position(|r| *r == s).unwrap()]
}

/// Points banked by one side: a lower bound, exact when every trick the side
/// took is fully known (or its points stated).
#[derive(Clone, Copy, Debug, Default)]
struct Bank {
    points: u32,
    exact: bool,
}

/// A line replayed so far.
struct Replay {
    physics: Physics,
    /// The remaining tiles of each seat the book depicts.
    hands: [Option<DominoSet>; SEATS],
    /// Whether every play of the seat so far is named.
    tracked: [bool; SEATS],
    /// Every named tile played.
    played: DominoSet,
    /// The contexts a seat has shown void in.
    voids: [ContextSet; SEATS],
    banks: [Bank; 2],
    winners: Vec<Who>,
    tricks: usize,
    /// The first play against follow-suit. The replay goes on past it: the
    /// tiles still leave the hands and the points still count.
    violation: Option<Outcome>,
}

impl Replay {
    fn new(example: &Example, decl: &str) -> Replay {
        let mut hands = [None; SEATS];
        for seat in &example.seats {
            hands[seat.role.index()] = Some(seat.set());
        }
        Replay {
            physics: common::physics(decl),
            hands,
            tracked: [true; SEATS],
            played: DominoSet::EMPTY,
            voids: [ContextSet::EMPTY; SEATS],
            banks: [Bank { points: 0, exact: true }; 2],
            winners: Vec::new(),
            tricks: 0,
            violation: None,
        }
    }

    /// The seat's legal plays when its hand is known exactly.
    fn legal(&self, seat: Role, lead: Option<Domino>) -> Option<DominoSet> {
        let i = seat.index();
        match (self.hands[i], self.tracked[i]) {
            (Some(hand), true) => Some(self.physics.legal_plays(hand, lead)),
            _ => None,
        }
    }

    /// One trick of narrated plays. `stated` is the prose's winner and points.
    fn trick(&mut self, leader: Who, plays: &[(Who, Option<Domino>)], stated: (Option<Who>, Option<u32>)) -> Step {
        let n = self.tricks + 1;
        if let (Some(prev), Who::Seat(_)) = (self.winners.last(), leader) {
            assert!(*prev == leader || *prev == Who::Setter, "trick {n}: led by {leader:?}, won by {prev:?}");
        }
        let mut lead: Option<Domino> = None;
        let mut known = Vec::new();
        for (k, (who, named)) in plays.iter().enumerate() {
            let mut tile = *named;
            // A play the rules force is the play, named or not.
            if let Who::Seat(seat) = who
                && tile.is_none()
                && k > 0
                && let Some(only) = self.legal(*seat, lead).filter(|s| s.len() == 1)
            {
                tile = only.iter().next();
            }
            let Some(d) = tile else {
                if let Who::Seat(seat) = who {
                    self.tracked[seat.index()] = false;
                }
                continue;
            };
            assert!(!self.played.contains(d), "trick {n}: {d} played twice");
            let holder = Role::ALL.into_iter().find(|r| self.hands[r.index()].is_some_and(|h| h.contains(d)));
            match who {
                Who::Seat(seat) if self.hands[seat.index()].is_some() => {
                    if holder != Some(*seat) {
                        return Err(Outcome::Conflict { trick: n, seat: *seat, tile: d });
                    }
                }
                Who::Seat(seat) => {
                    if holder.is_some() {
                        return Err(Outcome::Conflict { trick: n, seat: *seat, tile: d });
                    }
                }
                _ => assert!(holder.is_none(), "trick {n}: an unnamed seat plays the depicted {d}"),
            }
            if let (Some(led), Who::Seat(seat)) = (lead, who) {
                let context = self.physics.led_context(led);
                let i = seat.index();
                let follows = self.physics.follows(d, context);
                let legal = if follows {
                    !self.voids[i].contains(context)
                } else {
                    self.legal(*seat, Some(led)).is_none_or(|l| l.contains(d))
                };
                if !legal {
                    self.violation.get_or_insert(Outcome::Illegal { trick: n, seat: *seat, tile: d });
                } else if !follows {
                    self.voids[i].insert(context);
                }
            }
            if let Who::Seat(seat) = who
                && let Some(h) = self.hands[seat.index()].as_mut()
            {
                h.remove(d);
            }
            self.played.insert(d);
            if k == 0 {
                lead = Some(d);
            }
            known.push((*who, d));
        }
        let (stated_winner, stated_points) = stated;
        let complete = known.len() == SEATS && known.iter().all(|(w, _)| matches!(w, Who::Seat(_)));
        let winner = if complete {
            let led = self.physics.led_context(known[0].1);
            let best = (1..SEATS).fold(0, |b, i| {
                if self.physics.trick_key(known[i].1, led) > self.physics.trick_key(known[b].1, led) { i } else { b }
            });
            let won = known[best].0;
            if let Some(s) = stated_winner {
                assert_eq!(s, won, "trick {n}: the stated winner");
            }
            won
        } else {
            stated_winner.unwrap_or_else(|| panic!("trick {n}: no winner stated or computable"))
        };
        let floor = 1 + known.iter().map(|(_, d)| self.physics.count(*d)).sum::<u32>();
        let points = match (complete, stated_points) {
            (true, Some(p)) => {
                assert_eq!(p, floor, "trick {n}: the stated points");
                Bank { points: p, exact: true }
            }
            (true, None) => Bank { points: floor, exact: true },
            (false, Some(p)) => {
                assert!(floor <= p, "trick {n}: {floor} named points exceed the stated {p}");
                Bank { points: p, exact: true }
            }
            (false, None) => Bank { points: floor, exact: false },
        };
        let side = winner.side().expect("a winner of known side");
        self.banks[side].points += points.points;
        self.banks[side].exact &= points.exact;
        self.winners.push(winner);
        self.tricks += 1;
        Ok(())
    }

    /// The side `side` takes every remaining trick of a fully known deal,
    /// whatever the other side plays: `forced` fixes the tail's named plays.
    fn claim(&self, leader: Role, side: usize, forced: &[(Who, Option<Domino>)]) -> bool {
        let hands = self.hands.map(|h| h.expect("a full deal"));
        let forced: Vec<(Role, Domino)> = forced
            .iter()
            .map(|(w, d)| match (w, d) {
                (Who::Seat(r), Some(d)) => (*r, *d),
                _ => panic!("a tail play names its seat and tile"),
            })
            .collect();
        wins_out(&self.physics, hands, leader, side, &forced)
    }
}

/// Whether `side` can take every trick from here (perfect information): its
/// seats choose, the others' every legal play is tried.
fn wins_out(
    physics: &Physics,
    hands: [DominoSet; SEATS],
    leader: Role,
    side: usize,
    forced: &[(Role, Domino)],
) -> bool {
    if hands.iter().all(|h| h.is_empty()) {
        return true;
    }
    fn go(
        physics: &Physics,
        hands: [DominoSet; SEATS],
        leader: Role,
        side: usize,
        forced: &[(Role, Domino)],
        trick: &mut Vec<Domino>,
    ) -> bool {
        if trick.len() == SEATS {
            let led = physics.led_context(trick[0]);
            let best = (1..SEATS).fold(0, |b, i| {
                if physics.trick_key(trick[i], led) > physics.trick_key(trick[b], led) { i } else { b }
            });
            let winner = Role::at(leader.index() + best);
            return winner.side() == side && wins_out(physics, hands, winner, side, &[]);
        }
        let seat = Role::at(leader.index() + trick.len());
        let mut options = physics.legal_plays(hands[seat.index()], trick.first().copied());
        if let Some((_, d)) = forced.iter().find(|(r, _)| *r == seat) {
            assert!(options.contains(*d), "a forced tail play {d} is not legal");
            options = DominoSet::single(*d);
        }
        let mut each = options.iter().map(|d| {
            let mut next = hands;
            next[seat.index()].remove(d);
            trick.push(d);
            let ok = go(physics, next, leader, side, forced, trick);
            trick.pop();
            ok
        });
        if seat.side() == side { each.any(|ok| ok) } else { each.all(|ok| ok) }
    }
    go(physics, hands, leader, side, forced, &mut Vec::new())
}

/// Replay a narrated line: the outcome, and the replay for its totals.
fn replay(book: &Book, line: &Line) -> (Step, Replay) {
    let example = book.example(&line.example);
    let decl = line.declaration.as_deref().expect("a straight declaration");
    let mut r = Replay::new(example, decl);
    for t in &line.tricks {
        if let Err(o) = r.trick(t.leader, &t.plays, (t.winner, t.points)) {
            return (Err(o), r);
        }
    }
    (r.violation.clone().map_or(Ok(()), Err), r)
}

fn check_totals(line: &Line, r: &mut Replay) {
    for c in &line.checkpoints {
        assert!(c.after <= r.tricks, "{}: a checkpoint past the line", line.id);
    }
    if let Some(tail) = &line.tail {
        let side = tail.winner.side().unwrap();
        let rest = ALL_POINTS - r.banks[0].points - r.banks[1].points;
        if r.hands.iter().all(Option::is_some) {
            let Who::Seat(leader) = *r.winners.last().unwrap() else { panic!("{}: a full deal's leader", line.id) };
            assert!(r.claim(leader, side, &tail.plays), "{}: the tail's claim fails", line.id);
            r.banks[side].points += rest;
        } else {
            r.banks[side].exact = false;
        }
        r.tricks = 7;
    }
    if r.tricks == 7 {
        // Every point is banked by one side or the other.
        for side in 0..2 {
            if r.banks[side].exact {
                r.banks[1 - side] = Bank { points: ALL_POINTS - r.banks[side].points, exact: true };
            }
        }
    }
    for (side, stated) in line.totals.iter().enumerate() {
        if let Some(p) = stated {
            assert_eq!(r.tricks, 7, "{}: a total for an unfinished line", line.id);
            let b = r.banks[side];
            if b.exact {
                assert_eq!(b.points, *p, "{}: side {side}'s total", line.id);
            } else {
                assert!(b.points <= *p, "{}: side {side} has {} named points, over the stated {p}", line.id, b.points);
            }
        }
    }
}

/// The checkpoints need the bank after a given trick: replay to it.
fn check_checkpoints(book: &Book, line: &Line) {
    for c in &line.checkpoints {
        let mut short = line.clone();
        short.tricks.truncate(c.after);
        let (_, r) = replay(book, &short);
        let b = r.banks[c.side];
        if b.exact {
            assert_eq!(b.points, c.points, "{}: after trick {}", line.id, c.after);
        } else {
            assert!(b.points <= c.points, "{}: after trick {}, {} named points", line.id, c.after, b.points);
        }
    }
}

/// The plays the printed marks give: trick `k` takes each seat's tile marked
/// `k`, in play order from the last winner.
fn annotation_line(book: &Book, line: &Line) -> (Outcome, Vec<Who>, Replay) {
    let deal = book.example(&line.example);
    let mut marks: Vec<(Role, Domino, u8)> = Vec::new();
    for state in &line.states {
        for seat in &book.example(state).seats {
            for (d, m) in seat.tiles.iter().zip(&seat.marks) {
                if let (Some(d), Some(m)) = (d, m) {
                    match marks.iter().find(|(_, e, _)| e == d) {
                        Some((_, _, old)) => assert_eq!(old, m, "{}: {d} marked twice", line.id),
                        None => marks.push((seat.role, *d, *m)),
                    }
                }
            }
        }
    }
    let mut r = Replay::new(deal, deal.declaration.as_deref().unwrap());
    let mut leader = Role::Bidder;
    for k in 1..=7u8 {
        let n = k as usize;
        if !marks.iter().any(|(_, _, m)| *m == k) {
            let winners = r.winners.clone();
            return (r.violation.clone().unwrap_or(Outcome::Ends { trick: n }), winners, r);
        }
        let mut plays = Vec::new();
        for j in 0..SEATS {
            let seat = Role::at(leader.index() + j);
            match marks.iter().find(|(r, _, m)| *r == seat && *m == k) {
                Some((_, d, _)) => plays.push((Who::Seat(seat), Some(*d))),
                None => {
                    let winners = r.winners.clone();
                    return (r.violation.clone().unwrap_or(Outcome::Unmarked { trick: n, seat }), winners, r);
                }
            }
        }
        if let Err(o) = r.trick(Who::Seat(leader), &plays, (None, None)) {
            let winners = r.winners.clone();
            return (o, winners, r);
        }
        let Who::Seat(w) = *r.winners.last().unwrap() else { unreachable!() };
        leader = w;
    }
    let winners = r.winners.clone();
    (r.violation.clone().unwrap_or(Outcome::Legal), winners, r)
}

fn expected_winners(v: &Value) -> Option<Vec<Who>> {
    v.get("winners").map(|ws| ws.as_array().unwrap().iter().map(|w| Who::Seat(role(w.as_str().unwrap()))).collect())
}

// ------------------------------------------------------------------ the tests

#[test]
fn the_crosswalk_counts_hold() {
    // 36 numbered hands and two samples: 39 diagram states, 450 tile
    // appearances (the archive's figure crosswalk).
    let b = book();
    let ids: Vec<&str> = b.examples.iter().map(|e| e.id.as_str()).collect();
    for n in 1..=36 {
        assert!(ids.contains(&format!("hand-{n}").as_str()), "hand-{n}");
    }
    assert!(ids.contains(&"sample-nelo") && ids.contains(&"sample-sevens") && ids.contains(&"hand-30-state"));
    assert_eq!(b.examples.len(), 39);
    let appearances: usize = b.examples.iter().flat_map(|e| &e.seats).map(|s| s.tiles.iter().flatten().count()).sum();
    assert_eq!(appearances, 450);
    let deals = b.examples.iter().filter(|e| e.kind == "deal").count();
    assert_eq!(deals, 8);
    for e in &b.examples {
        assert!(!e.captures.is_empty() && !e.prose.is_empty(), "{}", e.id);
        for anchor in e.captures.iter().chain(&e.prose) {
            assert!(anchor.len() == 9 && anchor.starts_with("shot-0"), "{}: {anchor}", e.id);
        }
    }
}

#[test]
fn every_hand_has_seven_distinct_tiles() {
    for e in book().examples.iter().filter(|e| e.kind != "state") {
        for seat in &e.seats {
            assert_eq!(seat.tiles.len(), 7, "{}", e.id);
            assert!(seat.tiles.iter().all(Option::is_some), "{}", e.id);
            assert_eq!(seat.set().len(), 7, "{}: a tile repeats", e.id);
        }
    }
}

#[test]
fn every_full_deal_partitions_the_set() {
    for e in book().examples.iter().filter(|e| e.kind == "deal") {
        for role in Role::ALL {
            assert_eq!(e.seats.iter().filter(|s| s.role == role).count(), 1, "{}: {role:?}", e.id);
        }
        let all = e.deal().iter().fold(DominoSet::EMPTY, |all, h| {
            assert!(all.is_disjoint(*h), "{}", e.id);
            all.union(*h)
        });
        assert_eq!(all, DominoSet::FULL, "{}", e.id);
        assert!(matches!(e.bid, Bid::Points(_)), "{}: a deal is played at a stated bid", e.id);
    }
}

#[test]
fn hand_30s_later_state_is_the_deal_after_three_tricks() {
    let b = book();
    let deal = b.example("hand-30").deal();
    let state = b.example("hand-30-state");
    let gone: DominoSet = b.line("hand-30/prose").plays()[..12].iter().copied().collect();
    for role in Role::ALL {
        let seat = state.seat(role).unwrap();
        assert_eq!(seat.tiles.len(), 7);
        assert_eq!(seat.set(), deal[role.index()].difference(gone), "{role:?}");
    }
}

#[test]
fn the_deals_reseat_each_other_as_the_prose_says() {
    // Hands 29 and 30 deal Hand 28's tiles anew; 31 is 30's deal, 33 is 32's.
    let b = book();
    let as_set = |id: &str| {
        let mut hands = b.example(id).deal().to_vec();
        hands.sort_by_key(|h| h.bits());
        hands
    };
    assert_eq!(as_set("hand-29"), as_set("hand-28"));
    assert_eq!(as_set("hand-30"), as_set("hand-28"));
    assert_eq!(b.example("hand-31").deal(), b.example("hand-30").deal());
    assert_eq!(as_set("hand-33"), as_set("hand-32"));
    // The small hands replayed in chapter 3 are chapter 2's.
    for (later, earlier) in [(12, 1), (13, 2), (14, 3), (15, 4), (16, 8)] {
        assert_eq!(b.example(&format!("hand-{later}")).hand(), b.example(&format!("hand-{earlier}")).hand());
    }
}

#[test]
fn every_narrated_line_replays_to_the_stated_result() {
    let b = book();
    let mut replayed = 0;
    for line in b.lines.iter().filter(|l| l.source != "annotations") {
        let (outcome, mut r) = replay(b, line);
        let expected = Outcome::expected(&line.expect);
        match outcome {
            Ok(()) => assert_eq!(expected, Outcome::Legal, "{}", line.id),
            Err(o) => assert_eq!(o, expected, "{}", line.id),
        }
        if !matches!(expected, Outcome::Conflict { .. }) {
            check_totals(line, &mut r);
            check_checkpoints(b, line);
        }
        assert!(line.bid != Bid::Pass && line.bid != Bid::Unstated, "{}: a line is played at a bid", line.id);
        replayed += 1;
    }
    assert_eq!(replayed, 21);
}

#[test]
fn the_full_deals_lines_cover_every_trick() {
    // Each of the eight full deals has a prose line whose tricks and tail
    // account for all seven tricks.
    let b = book();
    for e in b.examples.iter().filter(|e| e.kind == "deal") {
        let line = b.line(&format!("{}/prose", e.id));
        assert_eq!(line.declaration, e.declaration, "{}", e.id);
        assert_eq!(line.bid, e.bid, "{}", e.id);
        let (outcome, mut r) = replay(b, line);
        assert_eq!(outcome.err().unwrap_or(Outcome::Legal), Outcome::expected(&line.expect), "{}", e.id);
        check_totals(line, &mut r);
        assert_eq!(r.tricks, 7, "{}", e.id);
    }
}

#[test]
fn the_annotation_orders_replay_to_their_recorded_outcomes() {
    // The marks are read as trick numbers. Where they differ from the prose,
    // the archive's review records the outcome: Hand 28's swapped doubles and
    // Hand 35's swapped trumps change nothing; Hand 31 marks no bidder tile 2;
    // by Hand 32's marks the partner wins trick 3; Hand 34's marks break
    // follow-suit at trick 5.
    let b = book();
    let mut lines = 0;
    for line in b.lines.iter().filter(|l| l.source == "annotations") {
        let (outcome, winners, mut r) = annotation_line(b, line);
        assert_eq!(outcome, Outcome::expected(&line.expect), "{}", line.id);
        if let Some(w) = expected_winners(&line.expect) {
            assert_eq!(winners, w, "{}", line.id);
        }
        if let Some(setters) = line.expect.get("setters").and_then(Value::as_u64) {
            r.tricks = 7;
            assert_eq!(r.banks[1].points, setters as u32, "{}", line.id);
            assert!(r.banks[1].exact);
        }
        lines += 1;
    }
    assert_eq!(lines, 8);
}

#[test]
fn the_known_source_conflicts_are_kept() {
    let b = book();
    // Hand 24: the diagram's bidder holds the 1-1 the prose gives opponent 1.
    assert!(b.example("hand-24").hand().contains(common::tile("1-1")));
    let (outcome, _) = replay(b, b.line("hand-24/prose"));
    assert_eq!(outcome, Step::Err(Outcome::Conflict { trick: 2, seat: Role::Opponent1, tile: common::tile("1-1") }));
    // Hand 26: the 5-2 the crisis branch lets an opponent play is the bidder's.
    assert!(b.example("hand-26").hand().contains(common::tile("5-2")));
    // Hand 31: the 6-5, led second, is marked 6, and no bidder tile is marked 2.
    let bidder = b.example("hand-31").seat(Role::Bidder).unwrap();
    let mark = |t: &str| bidder.tiles.iter().position(|d| *d == Some(common::tile(t))).and_then(|i| bidder.marks[i]);
    assert_eq!(mark("6-5"), Some(6));
    assert!(!bidder.marks.contains(&Some(2)));
}

#[test]
fn image_corrections_override_the_transcript() {
    // Where the page image disagrees with the transcript, the fixture carries
    // the image's tile and records the transcript's reading beside it.
    let root: Value = serde_json::from_str(include_str!("../examples/book/hands.json")).unwrap();
    let b = book();
    let mut corrections = 0;
    for e in root["examples"].as_array().unwrap() {
        for c in e.get("image_corrections").and_then(Value::as_array).into_iter().flatten() {
            let example = b.example(e["id"].as_str().unwrap());
            let seat = example.seat(role(c["seat"].as_str().unwrap())).unwrap();
            let at = c["position"].as_u64().unwrap() as usize - 1;
            let image: Domino = c["image"].as_str().unwrap().parse().unwrap();
            let transcript: Domino = c["transcript"].as_str().unwrap().parse().unwrap();
            assert_eq!(seat.tiles[at], Some(image), "{}", example.id);
            assert!(!seat.set().contains(transcript), "{}: the transcript's reading survives", example.id);
            corrections += 1;
        }
    }
    assert_eq!(corrections, 1);
}

#[test]
fn hand_6_holds_the_blank_five_the_image_shows() {
    // shot-0031, row 2, first tile: the transcript reads 0-2, the page image
    // shows 0-5, and the prose's trump count (the three high blanks) needs it.
    let hand = book().example("hand-6").seat(Role::Bidder).unwrap();
    assert_eq!(hand.tiles[2], Some(common::tile("0-5")));
    assert!(!hand.set().contains(common::tile("0-2")));
}
