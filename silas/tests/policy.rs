//! Executable policies over a fixed, fully specified deal.
mod common;

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
    physics: Physics,
    hands: [DominoSet; 4],
    frame: Frame,
}

impl Table {
    fn deal(decl: &str) -> Table {
        let hands = DEAL.map(|hand| tiles(&hand));
        assert_eq!(hands.iter().fold(DominoSet::EMPTY, |all, h| all.union(*h)), DominoSet::FULL);
        let physics = physics(decl);
        let frame = framed(&contract(&physics, Seat::S0, 30), Seat::S0, hands[0], vec![]).unwrap();
        Table { physics, hands, frame }
    }

    fn banked(&self) -> [u32; 2] {
        self.frame.banked()
    }

    fn actor(&self) -> Seat {
        self.frame.next_actor().unwrap()
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
        framed(&contract(&self.physics, Seat::S0, bid), Seat::S0, self.frame.dealt(), self.frame.plays().to_vec())
            .unwrap()
    }
}

fn own_legal_rule() -> PolicyRule {
    PolicyRule {
        name: "own-legal".into(),
        selector: Selector::FirstOutput,
        guard: "(query (roles (domino action)) (out action) (pattern (own-legal action)))".parse().unwrap(),
        span: Span::default(),
    }
}

fn program(key: PolicyKey) -> PolicyProgram {
    PolicyProgram {
        name: "table-and-relation".into(),
        exact_rules: vec![ExactRule { key, action: tile("6-6"), span: Span::default() }],
        rules: vec![own_legal_rule()],
        bidding: vec![],
        declaring: vec![],
        fallback: Fallback::LowestLegal,
    }
}

#[test]
fn text_roundtrip_preserves_exact_information_key() {
    for left in [7, 5] {
        let table = Table::deal("P6").until_viewer_holds(left);
        let source = program(PolicyKey::of(&table.at_bid(31)));
        assert_eq!(source.to_string().parse::<PolicyProgram>().unwrap(), source);
        let marks = program(PolicyKey::of(&table.at_bid(84)));
        assert_eq!(marks.to_string().parse::<PolicyProgram>().unwrap(), marks);
    }
}

#[test]
fn exact_rule_precedes_relation_and_is_checked_for_legality() {
    let table = Table::deal("P6");
    let input = table.frame.clone();
    let mut source = program(PolicyKey::of(&input));
    let hand = table.frame.kernel().viewer_hand();
    source.exact_rules[0].action = hand.iter().last().unwrap();
    let compiled = source.compile(&Registry::standard(), &table.physics).unwrap();
    let mut budget = Budget::new(100_000);
    let trace = compiled.decide(&input, &mut budget).unwrap();
    assert_eq!((trace.action, trace.provenance), (source.exact_rules[0].action, DecisionProvenance::Exact));

    source.exact_rules[0].action = DominoSet::FULL.difference(hand).iter().next().unwrap();
    assert!(source.compile(&Registry::standard(), &table.physics).is_err());
    // An exact key is bound to the physics it was taken under.
    assert!(program(PolicyKey::of(&input)).compile(&Registry::standard(), &physics("P5")).is_err());
}

#[test]
fn world_predicates_are_refused_and_fallback_is_total() {
    let table = Table::deal("P6");
    let input = table.frame.clone();
    let mut source = program(PolicyKey::of(&input));
    source.exact_rules.clear();
    source.rules.clear();
    let compiled = source.compile(&Registry::standard(), &table.physics).unwrap();
    let mut budget = Budget::new(100_000);
    assert_eq!(compiled.choose(&input, &mut budget).unwrap(), tile("0-0"));

    source.rules.push(PolicyRule {
        name: "peek".into(),
        selector: Selector::FirstOutput,
        guard: "(query (roles (domino action)) (out action) (pattern (holds S1 action)))".parse().unwrap(),
        span: Span::default(),
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
    while table.frame.plays().len() < 28 {
        let actor = table.actor();
        let tile = if actor == Seat::S0 {
            let input = table.frame.clone();
            let (tile, provenance) = choose(&input);
            decisions.push((PolicyKey::of(&input), tile, provenance));
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
        exact_rules: direct
            .iter()
            .map(|(key, action, _)| ExactRule { key: key.clone(), action: *action, span: Span::default() })
            .collect(),
        rules: vec![],
        bidding: vec![],
        declaring: vec![],
        fallback: Fallback::LowestLegal,
    };
    let parsed: PolicyProgram = program.to_string().parse().unwrap();
    assert_eq!(parsed, program);
    let compiled = parsed.compile(&Registry::standard(), &physics("P6")).unwrap();
    let replayed = play_out("P6", |input| {
        let trace = compiled.decide(input, &mut Budget::new(100_000)).unwrap();
        (trace.action, Some(trace.provenance))
    });
    assert_eq!(replayed.len(), direct.len());
    for ((key, action, _), (replayed_key, replayed_action, provenance)) in direct.iter().zip(&replayed) {
        assert_eq!((key, action), (replayed_key, replayed_action));
        assert_eq!(provenance, &Some(DecisionProvenance::Exact));
    }
}

#[test]
fn a_decision_is_a_function_of_the_frame() {
    // The same frame gets the same decision, however often and in whatever
    // order it is asked, and from whichever path the frame was built.
    let table = Table::deal("P6").until_viewer_holds(4);
    let mut source = program(PolicyKey::of(&table.frame));
    source.exact_rules.clear();
    let compiled = source.compile(&Registry::standard(), &table.physics).unwrap();
    let first = compiled.decide(&table.frame, &mut Budget::new(10_000)).unwrap();
    assert_eq!(first.provenance, DecisionProvenance::RelationalRule { name: "own-legal".into() });
    let earlier = Table::deal("P6").frame;
    let other = compiled.decide(&earlier, &mut Budget::new(10_000)).unwrap();
    for _ in 0..3 {
        assert_eq!(compiled.decide(&table.frame, &mut Budget::new(10_000)).unwrap(), first);
        assert_eq!(compiled.decide(&earlier, &mut Budget::new(10_000)).unwrap(), other);
    }
    let rebuilt =
        framed(&table.frame.contract().clone(), Seat::S0, table.frame.dealt(), table.frame.plays().to_vec()).unwrap();
    assert_eq!(compiled.decide(&rebuilt, &mut Budget::new(10_000)).unwrap(), first);
    // A refusal (here, an exhausted budget) leaves nothing behind.
    assert!(compiled.choose(&table.frame, &mut Budget::new(0)).is_err());
    assert_eq!(compiled.decide(&table.frame, &mut Budget::new(10_000)).unwrap(), first);
    // The allowed set is the rule's legal answers, and the decision its least.
    let allowed = compiled.allowed(&table.frame, &mut Budget::new(10_000)).unwrap();
    let legal = table.physics.legal_plays(table.frame.kernel().viewer_hand(), table.frame.lead());
    assert_eq!(allowed, Allowed { provenance: first.provenance.clone(), tiles: legal });
    assert_eq!((first.allowed, first.action), (legal, legal.iter().next().unwrap()));
}

#[test]
fn malformed_schema_history_and_modes_are_refused() {
    for source in [
        "(policy p (fallback lowest-legal) (fallback lowest-legal))",
        "(policy p (fallback lowest-legal) (mystery x))",
        // A policy has no modes and no bindings: it is a function of the frame.
        "(policy p (initial a) (fallback lowest-legal))",
        "(policy p (fallback lowest-legal) (bind top (query (roles (domino d)) (out d))))",
        "(policy p (fallback lowest-legal) (rule r (in a) (next a) (select answer) (guard (query (roles (domino d)) (out d)))))",
        "(policy p (fallback lowest-legal) (rule exact (select answer) (guard (query (roles (domino d)) (out d)))))",
    ] {
        let refused = source.parse::<PolicyProgram>().and_then(|p| p.compile(&Registry::standard(), &physics("P6")));
        assert!(refused.is_err(), "{source}");
    }
    // The one fallback may go unsaid.
    assert_eq!("(policy p)".parse::<PolicyProgram>().unwrap(), "(policy p (fallback lowest-legal))".parse().unwrap());

    let table = Table::deal("P6").until_viewer_holds(6);
    let mut wrong = PolicyKey::of(&table.frame);
    wrong.history.swap(0, 1);
    let source = program(wrong);
    assert!(source.compile(&Registry::standard(), &table.physics).is_err());
}

#[test]
fn contract_resolution_counts_every_point_the_physics_awards() {
    // A position where the bidder's side has made some point bids and the
    // other side has set others: every bid an auction allows (30 and up) is
    // resolved by the 42 points the physics awards.
    let (table, [ours, theirs]) = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "DT", "NT"]
        .into_iter()
        .flat_map(|decl| (1..=6).map(move |left| Table::deal(decl).until_viewer_holds(left)))
        .map(|table| {
            let banked = table.banked();
            (table, banked)
        })
        .find(|(_, [ours, theirs])| (31..=40).contains(ours) && (3..=12).contains(theirs))
        .expect("a position resolving some point bids each way");
    let source = PolicyProgram {
        name: "fallback-only".into(),
        exact_rules: vec![],
        rules: vec![],
        bidding: vec![],
        declaring: vec![],
        fallback: Fallback::LowestLegal,
    };
    let compiled = source.compile(&Registry::standard(), &table.physics).unwrap();
    let resolved = |bid: u32| compiled.decide(&table.at_bid(bid), &mut budget()).unwrap().contract_resolved;
    // 42 points in all: a point per trick and 35 count.
    for bid in 30..=41 {
        let expected = if bid <= ours {
            Some(true)
        } else if theirs >= 43 - bid {
            Some(false)
        } else {
            None
        };
        assert_eq!(resolved(bid), expected, "bid {bid} with {ours} to {theirs} banked");
    }
    assert_eq!((resolved(ours), resolved(ours + 1)), (Some(true), None));
    assert_eq!((resolved(42 - theirs), resolved(42 - theirs + 1)), (None, Some(false)));
    // A marks bid needs every point: set the moment the other side scores.
    assert_eq!(resolved(84), Some(false));
}
