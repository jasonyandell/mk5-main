//! Executable policies over a fixed, fully specified deal.
mod common;

use std::sync::Arc;

use common::*;
use silas::*;

/// A complete deal: S0's hand first, then S1, S2, S3.
const DEAL: [[&str; 7]; 4] = [
    ["0-0", "1-1", "2-0", "3-2", "5-5", "6-4", "6-6"],
    ["1-0", "2-2", "3-0", "4-1", "5-1", "6-0", "6-5"],
    ["2-1", "3-1", "4-0", "4-4", "5-2", "6-1", "6-3"],
    ["3-3", "4-2", "4-3", "5-0", "5-3", "5-4", "6-2"],
];

/// S0's view of the deal, played out from S0's lead: S0 declares, bidding 30.
struct Table {
    physics: Arc<dyn Physics>,
    hands: [DominoSet; 4],
    frame: Frame,
}

impl Table {
    fn deal(decl: &str) -> Table {
        let hands = DEAL.map(|hand| tiles(&hand));
        assert_eq!(hands.iter().fold(DominoSet::EMPTY, |all, h| all.union(*h)), DominoSet::FULL);
        let physics = physics(decl);
        let contract = Contract { declarer: Seat::S0, bid: 30 };
        let frame = Frame::from_record(physics.clone(), Seat::S0, hands[0], contract, vec![]).unwrap();
        Table { physics, hands, frame }
    }

    fn history(&self) -> &[(Seat, Domino)] {
        self.frame.record().unwrap().plays()
    }

    fn banked(&self) -> [u32; 2] {
        self.frame.record().unwrap().banked()
    }

    fn actor(&self) -> Seat {
        self.frame.leader().plus(self.frame.prefix().len())
    }

    fn legal(&self, seat: Seat) -> DominoSet {
        let held = self.hands[seat.index()].difference(self.frame.played());
        self.physics.legal_plays(held, self.frame.lead())
    }

    /// Make the next play, stepping S0's frame and its record.
    fn play(&mut self, tile: Domino) {
        let actor = self.actor();
        assert!(self.legal(actor).contains(tile));
        let class = match self.frame.led_context() {
            None => PlayClass::Lead,
            Some(q) if self.physics.follows(tile, q) => PlayClass::Follow(q),
            Some(q) => PlayClass::Slough(q),
        };
        self.frame = step_frame(&self.frame, ObservedPlay { actor, tile, class }, &mut budget()).unwrap();
    }

    /// Everyone plays their lowest legal tile until S0 is to act holding `left`.
    fn until_viewer_holds(mut self, left: usize) -> Table {
        while !(self.actor() == Seat::S0 && self.frame.kernel().viewer_hand().len() == left) {
            let lowest = self.legal(self.actor()).iter().next().unwrap();
            self.play(lowest);
        }
        self
    }

    /// S0's frame under a contract of `bid`.
    fn at_bid(&self, bid: u32) -> Frame {
        let contract = Contract { declarer: Seat::S0, bid };
        let hand = self.frame.kernel().viewer_hand();
        Frame::from_record(self.physics.clone(), Seat::S0, hand, contract, self.history().to_vec()).unwrap()
    }
}

fn program(key: PolicyKey) -> PolicyProgram {
    PolicyProgram {
        name: "table-and-relation".into(),
        initial_mode: "fresh".into(),
        bindings: vec![],
        exact_rules: vec![ExactRule { key, action: tile("6-6") }],
        rules: vec![PolicyRule {
            name: "own-legal".into(),
            in_mode: "fresh".into(),
            next_mode: "persistent".into(),
            selector: Selector::FirstOutput,
            guard: "(query (roles (domino action)) (out action) (pattern (own-legal action)))".parse().unwrap(),
        }],
        fallback: Fallback::LowestLegal,
    }
}

#[test]
fn text_roundtrip_preserves_exact_information_key() {
    for left in [7, 5] {
        let table = Table::deal("P6").until_viewer_holds(left);
        let source = program(PolicyKey::of(&table.at_bid(31)).unwrap());
        assert_eq!(source.to_string().parse::<PolicyProgram>().unwrap(), source);
    }
}

#[test]
fn exact_rule_precedes_relation_and_is_checked_for_legality() {
    let table = Table::deal("P6");
    let input = table.frame.clone();
    let mut source = program(PolicyKey::of(&input).unwrap());
    let hand = table.frame.kernel().viewer_hand();
    source.exact_rules[0].action = hand.iter().last().unwrap();
    let compiled = source.compile(&Registry::standard(), &table.physics).unwrap();
    let mut budget = Budget::new(100_000);
    let mut controller = compiled.initialize(&input, &mut budget).unwrap();
    let trace = compiled.choose_traced(&mut controller, &input, &mut budget).unwrap();
    assert_eq!((trace.action, trace.provenance), (source.exact_rules[0].action, DecisionProvenance::Exact));

    source.exact_rules[0].action = DominoSet::FULL.difference(hand).iter().next().unwrap();
    assert!(source.compile(&Registry::standard(), &table.physics).is_err());
    // An exact key is bound to the physics it was taken under.
    assert!(program(PolicyKey::of(&input).unwrap()).compile(&Registry::standard(), &physics("P5")).is_err());
}

#[test]
fn world_predicates_are_refused_and_fallback_is_total() {
    let table = Table::deal("P6");
    let input = table.frame.clone();
    let mut source = program(PolicyKey::of(&input).unwrap());
    source.exact_rules.clear();
    source.rules.clear();
    let compiled = source.compile(&Registry::standard(), &table.physics).unwrap();
    let mut budget = Budget::new(100_000);
    let mut controller = compiled.initialize(&input, &mut budget).unwrap();
    assert_eq!(compiled.choose(&mut controller, &input, &mut budget).unwrap(), tile("0-0"));

    source.rules.push(PolicyRule {
        name: "peek".into(),
        in_mode: "fresh".into(),
        next_mode: "fresh".into(),
        selector: Selector::FirstOutput,
        guard: "(query (roles (domino action)) (out action) (pattern (holds S1 action)))".parse().unwrap(),
    });
    assert!(source.compile(&Registry::standard(), &table.physics).is_err());
}

/// S0 plays its highest legal tile; everyone else plays lowest. Returns S0's
/// decisions as (key, action, trace) in order.
fn play_out(
    decl: &str,
    mut choose: impl FnMut(&Frame) -> (Domino, Option<DecisionProvenance>),
) -> Vec<(PolicyKey, Domino, Option<DecisionProvenance>)> {
    let mut table = Table::deal(decl);
    let mut decisions = Vec::new();
    while table.history().len() < 28 {
        let actor = table.actor();
        let tile = if actor == Seat::S0 {
            let input = table.frame.clone();
            let (tile, provenance) = choose(&input);
            decisions.push((PolicyKey::of(&input).unwrap(), tile, provenance));
            tile
        } else {
            table.legal(actor).iter().next().unwrap()
        };
        table.play(tile);
    }
    decisions
}

#[test]
fn a_table_policy_exports_and_replays_through_its_serialized_program() {
    let direct = play_out("P6", |input| {
        let legal = input.physics().legal_plays(input.kernel().viewer_hand(), input.lead());
        (legal.iter().last().unwrap(), None)
    });
    assert_eq!(direct.len(), 7);
    let program = PolicyProgram {
        name: "exported-table".into(),
        initial_mode: "fresh".into(),
        bindings: vec![],
        exact_rules: direct.iter().map(|(key, action, _)| ExactRule { key: key.clone(), action: *action }).collect(),
        rules: vec![],
        fallback: Fallback::LowestLegal,
    };
    let parsed: PolicyProgram = program.to_string().parse().unwrap();
    assert_eq!(parsed, program);
    let compiled = parsed.compile(&Registry::standard(), &physics("P6")).unwrap();
    let mut controller = None;
    let replayed = play_out("P6", |input| {
        let mut budget = Budget::new(100_000);
        let controller = controller.get_or_insert_with(|| compiled.initialize(input, &mut budget).unwrap());
        let trace = compiled.choose_traced(controller, input, &mut budget).unwrap();
        (trace.action, Some(trace.provenance))
    });
    assert_eq!(replayed.len(), direct.len());
    for ((key, action, _), (replayed_key, replayed_action, provenance)) in direct.iter().zip(&replayed) {
        assert_eq!((key, action), (replayed_key, replayed_action));
        assert_eq!(provenance, &Some(DecisionProvenance::Exact));
    }
}

#[test]
fn controller_provenance_and_mode_updates_are_transactional() {
    let table = Table::deal("P6");
    let input = table.frame.clone();
    let mut first = program(PolicyKey::of(&input).unwrap());
    first.exact_rules.clear();
    let mut second = first.clone();
    second.name = "other-program".into();
    let first = first.compile(&Registry::standard(), &table.physics).unwrap();
    let second = second.compile(&Registry::standard(), &table.physics).unwrap();
    let mut controller = first.initialize(&input, &mut Budget::new(10_000)).unwrap();
    assert!(second.choose(&mut controller, &input, &mut Budget::new(10_000)).is_err());
    assert_eq!(controller.mode(), "fresh");
    assert!(first.choose(&mut controller, &input, &mut Budget::new(0)).is_err());
    assert_eq!(controller.mode(), "fresh");
    let trace = first.choose_traced(&mut controller, &input, &mut Budget::new(10_000)).unwrap();
    assert_eq!(trace.provenance, DecisionProvenance::RelationalRule { name: "own-legal".into() });
    assert_eq!((trace.mode_before.as_str(), controller.mode()), ("fresh", "persistent"));
}

#[test]
fn malformed_schema_history_and_empty_binding_are_refused() {
    for source in [
        "(policy p (initial a) (initial b) (fallback lowest-legal))",
        "(policy p (initial a) (fallback lowest-legal) (mystery x))",
    ] {
        assert!(source.parse::<PolicyProgram>().is_err(), "{source}");
    }

    let table = Table::deal("P6").until_viewer_holds(6);
    let mut wrong = PolicyKey::of(&table.frame).unwrap();
    wrong.history[0].0 = wrong.history[0].0.plus(1);
    let mut source = program(wrong);
    assert!(source.compile(&Registry::standard(), &table.physics).is_err());
    source.exact_rules.clear();
    let compiled = source.compile(&Registry::standard(), &table.physics).unwrap();
    let unrecorded = Frame::new(
        table.physics.clone(),
        table.frame.kernel().clone(),
        table.frame.leader(),
        table.frame.prefix().to_vec(),
        table.frame.played(),
    )
    .unwrap();
    assert!(compiled.initialize(&unrecorded, &mut budget()).is_err());

    let input = table.frame.clone();
    let mut source = program(PolicyKey::of(&input).unwrap());
    source.exact_rules.clear();
    source
        .bindings
        .push(RigidBinding { name: "missing".into(), query: "(query (roles (domino d)) (out d))".parse().unwrap() });
    let compiled = source.compile(&Registry::standard(), &table.physics).unwrap();
    assert!(compiled.initialize(&input, &mut Budget::new(10_000)).is_err());
}

#[test]
fn contract_resolution_counts_every_point_the_physics_awards() {
    let table = Table::deal("P6").until_viewer_holds(3);
    let [ours, theirs] = table.banked();
    assert!(ours > 0 && theirs > 0, "the fixture banks points on both sides: {:?}", table.banked());
    let source = PolicyProgram {
        name: "fallback-only".into(),
        initial_mode: "fresh".into(),
        bindings: vec![],
        exact_rules: vec![],
        rules: vec![],
        fallback: Fallback::LowestLegal,
    };
    let compiled = source.compile(&Registry::standard(), &table.physics).unwrap();
    let resolved = |bid: u32| {
        let input = table.at_bid(bid);
        let mut controller = compiled.initialize(&input, &mut budget()).unwrap();
        compiled.choose_traced(&mut controller, &input, &mut budget()).unwrap().contract_resolved
    };
    // 42 points in all: a point per trick and 35 count.
    assert_eq!(resolved(ours), Some(true));
    assert_eq!(resolved(ours + 1), None);
    assert_eq!(resolved(42 - theirs), None);
    assert_eq!(resolved(42 - theirs + 1), Some(false));
}
