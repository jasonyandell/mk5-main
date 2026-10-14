//! STRATEGY-FUSION FENCE, sanctioned: the clairvoyant solver and
//! `(clairvoyant-dd-make)` are the upper bound the information-state solves are
//! held under, and the measure of how loose it is (Hand 26). The ruling:
//! wiki/topics/strategy-fusion-fence.md.
//!
//! Solved outcomes (README, Solved outcomes): the lay-down and what the viewer
//! can force, held to brute force on small positions and to the double-dummy
//! bound, and asked of the book's lay-downs (Roberson, *Winning 42*, Ch. 7,
//! Hands 21 and 22), of hands that look as strong and can be set, and of the
//! claim in Hand 32.
mod common;

use common::fixture::{Role, book};
use common::rules::{dealt, prose, then};
use common::*;
use silas::reference::Straight42;
use silas::*;

const LAY_DOWN: &str = include_str!("../examples/book/lay-down.silas");
const SHAPE: &str = include_str!("../examples/book/lay-down-shape.silas");
const CLAIM: &str = include_str!("../examples/book/claim-the-rest.silas");

fn big() -> Budget {
    Budget::new(2_000_000_000)
}

/// Whether a query holds at the frame: one answer, the empty tuple.
fn holds(source: &str, frame: &Frame) -> bool {
    let query = compile(source);
    !query.evaluate_viewer(frame, &mut big()).unwrap().is_empty()
}

/// The bidder's frame before the first lead, the bidder in `chair`.
fn bidders(decl: &str, bid: u32, chair: Seat, hand: DominoSet) -> Frame {
    framed(&contract(&physics(decl), chair, bid), chair, hand, vec![]).unwrap()
}

/// The book hand `id` as its bidder sees it before the first lead.
fn book_hand(id: &str, chair: Seat) -> Frame {
    let example = book().example(id);
    let decl = example.declaration.as_deref().unwrap();
    bidders(decl, example.bid.number(), chair, example.hand())
}

#[test]
fn the_books_lay_downs_lay_down_and_are_certain() {
    for (id, top) in [("hand-21", ["2-2", "6-2", "5-2"]), ("hand-22", ["4-4", "6-4", "5-4"])] {
        for chair in Seat::ALL {
            let frame = book_hand(id, chair);
            let line = lay_down(&frame, &mut big()).unwrap().expect("a lay-down");
            // The top three trumps first draw every trump out.
            let mut first: Vec<Domino> = line[..3].to_vec();
            first.sort();
            let mut want = tiles(&top).iter().collect::<Vec<_>>();
            want.sort();
            assert_eq!(first, want, "{id}: {line:?}");
            assert!(forces(&frame, Goal::Make, &mut big()).unwrap(), "{id}");
            assert!(holds(LAY_DOWN, &frame) && holds(SHAPE, &frame), "{id}");
        }
    }
    // Hand 22 leads the 6-5 after the 6-6 (S-0094-a).
    let line = lay_down(&book_hand("hand-22", Seat::S0), &mut big()).unwrap().unwrap();
    let at = |d: &str| line.iter().position(|t| *t == tile(d)).unwrap();
    assert!(at("6-6") < at("6-5"), "{line:?}");
}

#[test]
fn hands_that_look_as_strong_can_be_set() {
    // Hand 34: four fives with the double, the 6-5 out. Hands 17 and 18: the
    // book's 84s with an off.
    for id in ["hand-34", "hand-17", "hand-18"] {
        let frame = book_hand(id, Seat::S0);
        assert_eq!(lay_down(&frame, &mut big()).unwrap(), None, "{id}");
        assert!(!forces(&frame, Goal::Make, &mut big()).unwrap(), "{id}");
        assert!(!holds(LAY_DOWN, &frame) && !holds(SHAPE, &frame), "{id}");
    }
    // In Hand 34 a world where the right setter holds the 6-5 with two more
    // fives sets every line of the bidder's, though he sees all four hands.
    let frame = book_hand("hand-34", Seat::S0);
    let set = frame.worlds().find(|w| {
        let right = w.hand(Seat::S3);
        tiles(&["6-5", "5-3", "5-1"]).is_subset_of(right)
    });
    let position = OpenPosition::at(&frame, &set.unwrap()).unwrap();
    assert_eq!(ClairvoyantSolver::new(frame.physics()).value(&position, Objective::Make), 0);
}

#[test]
fn every_tile_of_a_suit_lays_down_three_marks() {
    // S-0188-a: all seven of a suit, bid at three marks.
    for decl in ["P0", "P1", "P2", "P3", "P4", "P5", "P6"] {
        let physics = physics(decl);
        let suit = physics.incidence(physics.trump().unwrap());
        let frame = bidders(decl, 126, Seat::S1, suit);
        assert!(holds(LAY_DOWN, &frame) && holds(SHAPE, &frame), "{decl}");
    }
}

#[test]
fn the_shape_is_sufficient_and_not_necessary() {
    // Every hand of the book's shape (S-0093-a) lays down: four trumps or more,
    // the top three among them, every other tile a walker.
    let mut rng = Rng(0x1a7d);
    let mut shaped = 0;
    for _ in 0..600 {
        let decl = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "DT"][rng.below(8)];
        let physics = physics(decl);
        let trump = physics.trump().unwrap();
        let mut trumps: Vec<Domino> = physics.incidence(trump).iter().collect();
        trumps.sort_by_key(|d| std::cmp::Reverse(physics.trick_key(*d, trump)));
        let (top, rest) = trumps.split_at(3);
        let more = 1 + rng.below(4);
        let mut hand: DominoSet = top.iter().copied().collect();
        let mut rest = rest.to_vec();
        for _ in 0..more {
            hand.insert(rest.swap_remove(rng.below(rest.len())));
        }
        let mut others: Vec<Domino> =
            rng.shuffled().into_iter().filter(|d| !physics.follows(*d, trump) && !hand.contains(*d)).collect();
        while hand.len() < 7 {
            hand.insert(others.pop().unwrap());
        }
        let frame = bidders(decl, 84, Seat::S0, hand);
        if holds(SHAPE, &frame) {
            shaped += 1;
            assert!(lay_down(&frame, &mut big()).unwrap().is_some(), "{decl} {hand:?}");
        }
    }
    assert!(shaped >= 50, "{shaped} hands of the shape");
    // Not necessary: all seven sixes lay down at no trump, with no trump at all.
    let sixes = tiles(&["6-6", "6-5", "6-4", "6-3", "6-2", "6-1", "6-0"]);
    let frame = bidders("NT", 84, Seat::S0, sixes);
    assert!(holds(LAY_DOWN, &frame) && !holds(SHAPE, &frame));
}

#[test]
fn hand_32_claims_the_rest_already_set() {
    // S-0143-c: opponent 2 leads the 5-4; the bidder trumps and claims.
    let at = dealt("hand-32", Role::Bidder, &then(&prose("hand-32"), &["5-4"]));
    for chair in Seat::ALL {
        let frame = at.frame(chair);
        let line = lay_down(&frame, &mut big()).unwrap().expect("a claim");
        assert!(frame.physics().follows(line[0], frame.physics().trump().unwrap()), "{line:?}");
        assert!(holds(CLAIM, &frame));
        // The setters hold 11 of the 7 they need: the claim is of tricks.
        assert!(!forces(&frame, Goal::Make, &mut big()).unwrap());
        assert!(!holds(LAY_DOWN, &frame));
    }
}

#[test]
fn a_settled_hand_answers_at_once_and_the_other_side_is_undefined() {
    // S-0113-b: an 84 is set by the setters' first trick.
    let mut rng = Rng(84);
    let (deal, plays) = loop {
        let deck = rng.shuffled();
        let deal: [DominoSet; 4] = std::array::from_fn(|i| deck[7 * i..7 * i + 7].iter().copied().collect());
        let mut position = OpenPosition::deal(contract(&physics("P6"), Seat::S0, 84), deal).unwrap();
        let mut plays = vec![];
        for _ in 0..4 {
            let tile = position.legal().iter().next().unwrap();
            plays.push(tile);
            position = position.play(tile).unwrap();
        }
        if position.leader().team() != Seat::S0.team() {
            break (deal, plays);
        }
    };
    let frame = |viewer: Seat| {
        framed(&contract(&physics("P6"), Seat::S0, 84), viewer, deal[viewer.index()], plays.clone()).unwrap()
    };
    let mut nothing = Budget::new(3);
    assert!(forces(&frame(Seat::S1), Goal::Set, &mut nothing).unwrap());
    assert!(!forces(&frame(Seat::S0), Goal::Make, &mut nothing).unwrap());
    assert!(forces(&frame(Seat::S0), Goal::Set, &mut big()).is_err());
    for (viewer, other) in [(Seat::S0, "certain-set"), (Seat::S1, "certain-make")] {
        for source in [
            format!("(query (roles) (out) (pattern ({other})))"),
            format!("(query (roles) (out) (pattern (not ({other}))))"),
        ] {
            assert!(!holds(&source, &frame(viewer)), "{viewer} {source}");
        }
    }
}

/// A random deal after `plays` random legal plays, under a random
/// declaration and bid: the contract, the deal and the record.
fn random_deal(rng: &mut Rng, plays: usize) -> (Contract, [DominoSet; 4], Vec<Domino>) {
    let physics = Straight42::ALL[rng.below(9)].physics();
    let deck = rng.shuffled();
    let deal: [DominoSet; 4] = std::array::from_fn(|i| deck[7 * i..7 * i + 7].iter().copied().collect());
    let bids = [30, 31, 32, 34, 36, 38, 41, 42, 84];
    let contract = contract(&physics, Seat::ALL[rng.below(4)], bids[rng.below(bids.len())]);
    let mut position = OpenPosition::deal(contract.clone(), deal).unwrap();
    let mut record = vec![];
    for _ in 0..plays {
        let legal: Vec<Domino> = position.legal().iter().collect();
        let tile = legal[rng.below(legal.len())];
        record.push(tile);
        position = position.play(tile).unwrap();
    }
    (contract, deal, record)
}

/// A random deal's position, seen by the seat to play.
fn random_frame(rng: &mut Rng, plays: usize) -> Frame {
    let (contract, deal, record) = random_deal(rng, plays);
    let viewer = OpenPosition::deal(contract.clone(), deal)
        .map(|p| record.iter().fold(p, |p, d| p.play(*d).unwrap()))
        .unwrap()
        .next_actor()
        .unwrap();
    framed(&contract, viewer, deal[viewer.index()], record).unwrap()
}

/// Every play of the viewer's from `line` in turn, every legal play of the
/// others: does the viewer take every trick?
fn line_takes_every_trick(position: &OpenPosition, viewer: Seat, line: &[Domino]) -> bool {
    let Some(seat) = position.next_actor() else { return true };
    let after = |tile: Domino, line: &[Domino]| {
        let next = position.play(tile).unwrap();
        let lost = next.trick().is_empty() && next.leader() != viewer;
        !lost && line_takes_every_trick(&next, viewer, line)
    };
    if seat == viewer {
        let (first, rest) = line.split_first().unwrap();
        return position.legal().contains(*first) && after(*first, rest);
    }
    position.legal().iter().all(|tile| after(tile, line))
}

fn orders(tiles: &[Domino]) -> Vec<Vec<Domino>> {
    if tiles.is_empty() {
        return vec![vec![]];
    }
    let mut out = vec![];
    for (i, first) in tiles.iter().enumerate() {
        let mut rest = tiles.to_vec();
        rest.remove(i);
        out.extend(orders(&rest).into_iter().map(|o| [vec![*first], o].concat()));
    }
    out
}

/// Some order of the viewer's tiles takes every trick in every world.
fn brute_lay_down(frame: &Frame) -> bool {
    let mine: Vec<Domino> = frame.kernel().viewer_hand().iter().collect();
    let worlds = worlds_of(frame);
    orders(&mine).iter().any(|line| worlds.iter().all(|p| line_takes_every_trick(p, frame.viewer(), line)))
}

/// Some strategy of the viewer's ends the hand in `goal` in every world
/// against every play, found over explicit sets of worlds that share a
/// history: the viewer chooses one tile for them all; another seat's tile
/// keeps the worlds in which it was legal.
fn brute_forces(worlds: &[OpenPosition], viewer: Seat, goal: Goal) -> bool {
    let first = &worlds[0];
    let Some(seat) = first.next_actor() else {
        let side = first.contract().declarer().team().index();
        let made = first.banked()[side] >= first.contract().threshold();
        return made == (goal == Goal::Make);
    };
    let after = |tile: Domino| -> Vec<OpenPosition> {
        worlds.iter().filter(|p| p.legal().contains(tile)).map(|p| p.play(tile).unwrap()).collect()
    };
    if seat == viewer {
        return first.legal().iter().any(|tile| brute_forces(&after(tile), viewer, goal));
    }
    let options = worlds.iter().fold(DominoSet::EMPTY, |all, p| all.union(p.legal()));
    options.iter().all(|tile| brute_forces(&after(tile), viewer, goal))
}

fn worlds_of(frame: &Frame) -> Vec<OpenPosition> {
    frame.worlds().map(|w| OpenPosition::at(frame, &w).unwrap()).collect()
}

fn double_dummy_reaches(frame: &Frame, goal: Goal) -> Vec<bool> {
    frame
        .worlds()
        .map(|w| {
            let position = OpenPosition::at(frame, &w).unwrap();
            (ClairvoyantSolver::new(frame.physics()).value(&position, Objective::Make) == 1) == (goal == Goal::Make)
        })
        .collect()
}

#[test]
fn agrees_with_brute_force_and_never_rises_above_the_double_dummy_bound() {
    let mut rng = Rng(0x0dd5);
    // [lay-down true, lay-down false, forces true, forces false]; and the
    // positions where every world makes double dummy and the viewer cannot
    // force it: strategy fusion, or a partner who need not help.
    let mut seen = [0; 4];
    let mut fused = 0;
    for round in 0..900 {
        let tricks_left = if round % 3 == 0 { 3 } else { 2 };
        let plays = 4 * (7 - tricks_left) + rng.below(4);
        let frame = random_frame(&mut rng, plays);
        let goal = Goal::of(&frame, frame.viewer());
        let lays_down = lay_down(&frame, &mut big()).unwrap();
        assert_eq!(lays_down.is_some(), brute_lay_down(&frame), "{frame:?}");
        seen[usize::from(lays_down.is_none())] += 1;
        if let Some(line) = &lays_down {
            let worlds = frame.worlds().map(|w| OpenPosition::at(&frame, &w).unwrap());
            assert!(worlds.into_iter().all(|p| line_takes_every_trick(&p, frame.viewer(), line)));
        }
        let forced = forces(&frame, goal, &mut big()).unwrap();
        if tricks_left == 2 {
            assert_eq!(forced, brute_forces(&worlds_of(&frame), frame.viewer(), goal), "{frame:?}");
        }
        seen[2 + usize::from(!forced)] += 1;
        // The double-dummy bound is never below the guarantee, and a lay-down
        // forces every goal still in reach.
        let bound = double_dummy_reaches(&frame, goal);
        if forced {
            assert!(bound.iter().all(|b| *b), "{frame:?}");
        } else if bound.iter().all(|b| *b) {
            fused += 1;
        }
        let in_reach = bound.iter().any(|b| *b);
        if lays_down.is_some() && in_reach {
            assert!(forced, "{frame:?}");
        }
    }
    assert!(seen.iter().all(|n| *n >= 20) && fused >= 1, "{seen:?} {fused}");
}

#[test]
fn the_viewers_information_state_is_his_frame() {
    // From every chair, whoever is to play: the and-or search over frames
    // against the same search over explicit sets of worlds.
    let mut rng = Rng(0x5e7);
    let mut seen = [0; 2];
    for round in 0..300 {
        let plays = if round % 30 == 0 { 12 } else { [16, 20, 20][round % 3] } + rng.below(4);
        let (contract, deal, record) = random_deal(&mut rng, plays);
        let viewer = Seat::ALL[rng.below(4)];
        let frame = framed(&contract, viewer, deal[viewer.index()], record).unwrap();
        let goal = Goal::of(&frame, viewer);
        let forced = forces(&frame, goal, &mut big()).unwrap();
        assert_eq!(forced, brute_forces(&worlds_of(&frame), viewer, goal), "{frame:?}");
        seen[usize::from(forced)] += 1;
    }
    assert!(seen.iter().all(|n| *n >= 50), "{seen:?}");
}

#[test]
fn the_double_dummy_bound_as_a_query_is_exact_only_on_the_whole_fiber() {
    let frame = small_frame(physics("P6"));
    let dd = compile("(query (roles) (out) (pattern (clairvoyant-dd-make)))");
    let uniform = Belief::uniform(frame.clone(), 90).unwrap();
    let p = dd.summarize(&uniform, &mut big()).unwrap().event_probability();
    let made = double_dummy_reaches(&frame, Goal::Make).iter().filter(|b| **b).count() as i64;
    assert_eq!(exact(p), r(made, 90));
    // A belief that leaves a tile unresolved refuses it; uniform_for resolves
    // the whole pool, and its cap refuses a fiber beyond it.
    let partial = Belief::uniform_resolving(frame.clone(), tiles(&["6-5"]), 90).unwrap();
    assert!(dd.summarize(&partial, &mut big()).unwrap_err().0.contains("unresolved"));
    let whole = Belief::uniform_for(frame, &dd, 90, &mut big()).unwrap();
    assert!(whole.unresolved().is_empty());
    let root = book_hand("hand-21", Seat::S0);
    assert!(Belief::uniform_for(root.clone(), &dd, 100_000, &mut big()).unwrap_err().0.contains("cap"));
    // The viewer facts read no world: one point, exact.
    let certain = compile(LAY_DOWN);
    let belief = Belief::uniform_for(root, &certain, 1, &mut big()).unwrap();
    let p = certain.summarize(&belief, &mut big()).unwrap().event_probability();
    assert_eq!(exact(p), r(1, 1));
}

/// STRATEGY-FUSION FENCE: a policy cannot read the clairvoyant predicate, nor
/// can a viewer evaluation, and each refusal names the fence. A policy that
/// chose on it would choose per world.
#[test]
fn a_policy_that_reads_the_clairvoyant_predicate_is_refused_at_the_fence() {
    let policy = "(policy peek\n  (fallback lowest-legal)\n  (rule clairvoyant (select answer)\n    \
                  (guard (query (roles (domino d)) (out d) (pattern (own-legal d) (clairvoyant-dd-make))))))";
    let error = PolicyProgram::parse(policy, "peek.silas")
        .and_then(|p| p.compile(&Registry::standard(), &physics("P6")).map(drop))
        .unwrap_err()
        .0;
    assert!(
        error.starts_with("peek.silas:3:3: rule clairvoyant uses the clairvoyant predicate clairvoyant-dd-make"),
        "{error}"
    );
    assert!(error.contains("STRATEGY-FUSION FENCE") && error.contains("wiki/topics/strategy-fusion-fence.md"));
    // Through a definition too: the guard's compiled atoms carry the access.
    let wrapped = "(define peeks (roles)
  (clairvoyant-dd-make))\n(policy peek\n  (fallback lowest-legal)\n  \
                   (rule clairvoyant (select answer)\n    (guard (query (roles (domino d)) (out d) (pattern (own-legal d) (peeks))))))";
    let error = PolicyProgram::parse(wrapped, "peek.silas")
        .and_then(|p| p.compile(&Registry::standard(), &physics("P6")).map(drop))
        .unwrap_err()
        .0;
    assert!(error.contains("STRATEGY-FUSION FENCE"), "{error}");
    // A viewer evaluation refuses it before any work, naming the fence.
    let frame = small_frame(physics("P6"));
    let dd = compile("(query (roles) (out) (pattern (clairvoyant-dd-make)))");
    let error = dd.evaluate_viewer(&frame, &mut big()).unwrap_err().0;
    assert!(error.starts_with("viewer evaluation refused the clairvoyant predicate clairvoyant-dd-make"), "{error}");
    assert!(error.contains("STRATEGY-FUSION FENCE"), "{error}");
    // A world predicate's refusal stays plain.
    let holds = compile("(query (roles) (out) (pattern (holds S1 6-5)))");
    let error = holds.evaluate_viewer(&frame, &mut big()).unwrap_err().0;
    assert_eq!(error, "viewer evaluation refused the world predicate holds");
}

#[test]
fn work_beyond_the_budget_is_refused_and_not_remembered() {
    let frame = book_hand("hand-21", Seat::S0);
    assert!(lay_down(&frame, &mut Budget::new(1_000)).is_err());
    assert!(forces(&frame, Goal::Make, &mut Budget::new(1_000)).is_err());
    let query = compile(LAY_DOWN);
    assert!(query.evaluate_viewer(&frame, &mut Budget::new(1_000)).is_err());
    assert!(!query.evaluate_viewer(&frame, &mut big()).unwrap().is_empty());
    // A frame no lay-down proves and no world refutes leaves the viewer's
    // search: given one unit less than the work it takes, it is refused.
    let mut rng = Rng(5);
    let (frame, goal) = loop {
        let frame = random_frame(&mut rng, 16);
        let goal = Goal::of(&frame, frame.viewer());
        let (to_make, to_set) = frame.to_make_and_set();
        let open = to_make > 0 && to_set > 0;
        if open
            && double_dummy_reaches(&frame, goal).iter().all(|b| *b)
            && lay_down(&frame, &mut big()).unwrap().is_none()
        {
            break (frame, goal);
        }
    };
    let mut budget = big();
    forces(&frame, goal, &mut budget).unwrap();
    assert!(forces(&frame, goal, &mut Budget::new(budget.spent() - 1)).is_err());
}

/// Hand 26 at 35, fours, after trick 2: the bidder's 4-4 and 6-4 have drawn all
/// seven trumps (one record of the book's two leads), 17 are banked, and he
/// leads the 3-3 1-1 5-5 2-2 5-2 into the 15 tiles he cannot place.
fn hand_26_trumps_cleared() -> common::fixture::Position {
    common::rules::position(
        "P4",
        Role::Bidder,
        &["3-3", "1-1", "5-5", "2-2", "5-2"],
        &["4-4", "4-3", "4-2", "4-1", "6-4", "5-4", "4-0", "6-6"],
    )
    .bid(35)
}

/// The bidder's side's final points when the bidder maximizes and the setters
/// minimize. With no tile named, his partner maximizes too (double-dummy
/// play); with one named, the partner keeps that tile to his last trick, which
/// is a legal choice at every trick before it, and minimizes otherwise.
fn bidder_side_points(position: &OpenPosition, keeps: Option<Domino>) -> u32 {
    let Some(seat) = position.next_actor() else {
        return position.banked()[Seat::S0.team().index()];
    };
    let mut legal: Vec<Domino> = position.legal().iter().collect();
    if let Some(kept) = keeps.filter(|_| seat == Seat::S2 && legal.len() > 1) {
        legal.retain(|d| *d != kept);
    }
    let values = legal.iter().map(|d| bidder_side_points(&position.play(*d).unwrap(), keeps));
    if seat == Seat::S0 || (seat == Seat::S2 && keeps.is_none()) {
        values.max().unwrap()
    } else {
        values.min().unwrap()
    }
}

#[test]
fn hand_26_the_setting_deal_is_a_world_of_the_cleared_position_and_differs_in_the_partners_play() {
    // The private source adjudication (A13) builds a deal that sets the 35
    // when the partner keeps the 3-2 to the last trick. Its tricks 1 and 2 are
    // another record (other seats play the trumps), so it does not reach the
    // position above by its own play; its hidden hands, banked points and
    // leader are that position's, and they are one of its 756,756 layouts.
    let deal = [
        ["3-3", "1-1", "4-4", "6-4", "5-5", "2-2", "5-2"],
        ["4-0", "4-1", "5-0", "5-1", "1-0", "6-1", "6-2"],
        ["4-2", "4-3", "3-2", "3-0", "2-0", "3-1", "2-1"],
        ["4-5", "6-5", "5-3", "6-3", "6-6", "6-0", "0-0"],
    ]
    .map(|hand| tiles(&hand));
    let record = ["4-4", "4-1", "4-2", "4-5", "6-4", "4-0", "4-3", "6-6"].map(tile);
    let own = common::fixture::Position::dealt(physics("P4"), 35, deal, Role::Bidder, &record).frame(Seat::S0);
    let cleared = hand_26_trumps_cleared().frame(Seat::S0);
    assert_ne!(own.plays(), cleared.plays());
    assert_eq!(
        (own.kernel().pool(), own.banked(), own.leader()),
        (cleared.kernel().pool(), cleared.banked(), cleared.leader())
    );
    assert_eq!((own.count_worlds(), own.to_make_and_set()), (756_756, (18, 8)));
    let world = cleared
        .worlds()
        .find(|w| {
            w.hand(Seat::S1) == deal[1].difference(tiles(&["4-0", "4-1"]))
                && w.hand(Seat::S2) == deal[2].difference(tiles(&["4-2", "4-3"]))
        })
        .expect("the setting deal's layout is one of the 756,756");
    assert_eq!(world.hand(Seat::S3), deal[3].difference(tiles(&["4-5", "6-6"])));
    // There double-dummy play makes the bid, the partner shedding his count on
    // the bidder's doubles: the bidder's side scores 36.
    let position = OpenPosition::at(&cleared, &world).unwrap();
    assert_eq!(ClairvoyantSolver::new(cleared.physics()).value(&position, Objective::Make), 1);
    assert_eq!(bidder_side_points(&position, None), 36);
    // The difference is the partner's play. Keeping the 3-2, always a legal
    // choice, sets the bidder even though he sees all four hands: 31, short of
    // the 35 and the 36. This one world is why he cannot force the make.
    assert_eq!(bidder_side_points(&position, Some(tile("3-2"))), 31);
}

#[test]
fn hand_26_can_be_set_once_trumps_are_cleared_though_double_dummy_makes() {
    // S-0123-b, from the position after trick 2 with the trumps gone.
    let frame = hand_26_trumps_cleared().frame(Seat::S0);
    assert_eq!((frame.count_worlds(), frame.to_make_and_set()), (756_756, (18, 8)));
    // Double dummy makes in every world: the bound says "cannot be set".
    assert!(double_dummy_reaches(&frame, Goal::Make).iter().all(|made| *made));
    // The bidder's own play cannot force it: the claim fails, exactly.
    assert_eq!(lay_down(&frame, &mut big()).unwrap(), None);
    assert!(!forces(&frame, Goal::Make, &mut big()).unwrap());
    assert!(!holds(include_str!("../examples/book/cannot-be-set.silas"), &frame));
}

#[test]
fn s0043f_hand_12_no_lead_can_hurt_the_bidder() {
    // Hand 12 after trick 3 (ours: the bidder's 6-5 on trick 2, the one six
    // that makes the trick safe for the partner's 5-0): the left setter has
    // the lead and the setters 6 of the 7 that set the 36. The bidder holds
    // two trumps and the 4-4 and 1-1: a four or an ace meets its double, any
    // other lead is trumped. He cannot be set, from his own chair.
    let plays = ["6-6", "6-1", "6-0", "1-0", "6-5", "6-2", "5-0", "3-0", "2-0", "2-2", "4-2", "3-2"];
    let at = book().bidders("hand-12", &plays.map(tile));
    for chair in Seat::ALL {
        let frame = at.frame(chair);
        assert_eq!(frame.to_make_and_set(), (29, 1));
        assert!(forces(&frame, Goal::Make, &mut big()).unwrap());
        assert!(holds(include_str!("../examples/book/cannot-be-set.silas"), &frame));
    }
}

#[test]
fn s0137d_hand_30_the_last_two_tiles_cannot_be_beaten() {
    // Hand 30 after trick 5: the bidder leads with the 6-2, the last trump,
    // and the 2-0, the last deuce; the right setter's 5-5 and 4-1 are out and
    // fall to him.
    let at = dealt("hand-30", Role::Bidder, &prose("hand-30"));
    for chair in Seat::ALL {
        let frame = at.frame(chair);
        assert!(holds(CLAIM, &frame));
        assert!(holds(LAY_DOWN, &frame));
    }
}
