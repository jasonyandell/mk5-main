// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! One typed public play applied to frames, worlds and beliefs.
mod common;

use std::collections::BTreeSet;

use common::*;
use silas::*;

fn initial() -> Frame {
    small_frame(physics("P6"))
}

fn small_budget() -> Budget {
    Budget::new(10_000)
}

fn lead(t: &str) -> ObservedPlay {
    ObservedPlay { actor: Seat::S0, tile: tile(t), class: PlayClass::Lead }
}

#[test]
fn typed_play_rejects_wrong_actor_class_and_illegal_world() {
    let frame = initial();
    let world = frame.worlds().next().unwrap();
    let wrong_actor = ObservedPlay { actor: Seat::S1, tile: tile("6-6"), class: PlayClass::Lead };
    assert!(step_frame(&frame, wrong_actor, &mut small_budget()).is_err());
    let wrong_class = ObservedPlay { actor: Seat::S0, tile: tile("6-6"), class: PlayClass::Follow(Context::Called) };
    assert!(step_frame(&frame, wrong_class, &mut small_budget()).is_err());
    assert!(step_world(&frame, &world, lead("6-5")).is_err());
}

#[test]
fn frame_step_rejects_viewer_slough_while_viewer_can_follow() {
    // S1 declares and leads the 5-5; S0, last to play, holds the 5-4.
    let p = physics("P6");
    let dealt = tiles(&["5-4", "0-0", "1-1", "2-1", "3-1", "4-1", "4-0"]);
    let plays = vec![tile("5-5"), tile("5-1"), tile("5-0")];
    let frame = framed(&contract(&p, Seat::S1, 30), Seat::S0, dealt, plays).unwrap();
    let fives = "q5".parse::<Context>().unwrap();
    let illegal = ObservedPlay { actor: Seat::S0, tile: tile("0-0"), class: PlayClass::Slough(fives) };
    assert!(step_frame(&frame, illegal, &mut small_budget()).is_err());
}

#[test]
fn frame_step_rejects_hidden_tile_forbidden_by_an_existing_void() {
    // Sixes trump: S1 shows out on S0's 6-4, then cannot follow the 6-6.
    let p = physics("P6");
    let dealt = tiles(&["6-4", "6-6", "0-0", "1-1", "2-0", "3-0", "4-0"]);
    let plays = ["6-4", "3-2", "6-0", "6-1", "6-6"].map(tile).to_vec();
    let frame = framed(&contract(&p, Seat::S0, 30), Seat::S0, dealt, plays).unwrap();
    assert!(frame.shown_voids(Seat::S1).contains(Context::Called));
    let impossible = ObservedPlay { actor: Seat::S1, tile: tile("6-5"), class: PlayClass::Follow(Context::Called) };
    assert!(step_frame(&frame, impossible, &mut small_budget()).is_err());
}

#[test]
fn hidden_slough_conditions_support_and_pushes_worlds_forward() {
    let led = step_frame(&initial(), lead("6-6"), &mut small_budget()).unwrap();
    let prior = Belief::uniform(led, 90).unwrap();
    let slough = ObservedPlay { actor: Seat::S1, tile: tile("3-2"), class: PlayClass::Slough(Context::Called) };
    let stepped = step_belief(&prior, slough, "slough-posterior", 90, &mut small_budget()).unwrap();
    let p = exact(stepped.evidence_probability.clone());
    assert!(p > r(0, 1) && p < r(1, 1));
    assert_eq!(stepped.belief.authority(), &Authority::exact());
    assert_eq!(stepped.belief.frame().prefix(), &[tile("6-6"), tile("3-2")]);
    assert!(!stepped.belief.frame().kernel().pool().contains(tile("3-2")));
    let trumps = stepped.belief.frame().physics().incidence(Context::Called);
    for (world, _) in stepped.belief.worlds() {
        assert!(!world.hand(Seat::S1).contains(tile("3-2")));
        assert!(world.hand(Seat::S1).is_disjoint(trumps));
    }
}

#[test]
fn explicit_likelihood_precedes_pushforward_and_normalization() {
    let led = step_frame(&initial(), lead("6-6"), &mut small_budget()).unwrap();
    let prior = Belief::uniform(led, 90).unwrap();
    let follow = ObservedPlay { actor: Seat::S1, tile: tile("6-5"), class: PlayClass::Follow(Context::Called) };
    let unit = step_belief(&prior, follow, "unit", 90, &mut small_budget()).unwrap();
    let model = "test: a coin flip for every legal world";
    let halved = step_belief_given(&prior, follow, "half", 90, &mut small_budget(), model, |_| Ok(w(1, 2))).unwrap();
    assert_eq!(halved.evidence_probability.value(), &(unit.evidence_probability.value() * r(1, 2)));
    assert_eq!(halved.evidence_probability.authority(), &given(model));
    assert_eq!(halved.belief.authority(), &given(model));
    assert_eq!(halved.belief.len(), unit.belief.len());
    assert_eq!(halved.belief.total_mass(), &(unit.belief.total_mass() * &w(1, 2)));
}

#[test]
fn viewer_intervention_preserves_hidden_deal_odds() {
    let frame = initial();
    let entries: Vec<_> = frame.worlds().take(3).zip(1..).map(|(world, i)| (world, w(i, 1))).collect();
    let prior = Belief::from_weights(frame, "nonuniform", given("test: weights 1, 2, 3"), entries, 3).unwrap();
    let stepped = step_belief(&prior, lead("6-6"), "viewer-intervention", 3, &mut small_budget()).unwrap();
    assert_eq!(stepped.evidence_probability.value(), &r(1, 1));
    assert_eq!(stepped.belief.authority(), prior.authority());
    assert_eq!(stepped.belief.len(), prior.len());
    for ((_, before), (_, after)) in prior.worlds().zip(stepped.belief.worlds()) {
        assert_eq!(before, after);
    }
}

#[test]
fn rigid_identity_separates_persistence_extinction_and_birth() {
    let a = Answer(vec![Value::Domino(tile("6-6"))]);
    let b = Answer(vec![Value::Domino(tile("5-5"))]);
    let c = Answer(vec![Value::Domino(tile("3-2"))]);
    let prior = BTreeSet::from([a.clone(), b.clone()]);
    let fresh = BTreeSet::from([b.clone(), c.clone()]);
    let dynamics = compare_answers(&prior, &fresh);
    assert_eq!(dynamics.transported, prior);
    assert_eq!(dynamics.persistent, BTreeSet::from([b]));
    assert_eq!(dynamics.extinct, BTreeSet::from([a]));
    assert_eq!(dynamics.born, BTreeSet::from([c]));
}

#[test]
fn refusal_returns_no_partial_belief() {
    let prior = Belief::uniform(initial(), 90).unwrap();
    let mut tiny = Budget::new((prior.len() as u64).saturating_sub(1));
    assert!(step_belief(&prior, lead("6-6"), "never-built", 90, &mut tiny).is_err());
    assert_eq!(prior.frame().prefix(), &[]);
    assert_eq!(prior.id(), "uniform-full-support");
}

#[test]
fn hindsight_anchor_is_a_preimage_not_revelation() {
    let frame = initial();
    let worlds: Vec<_> = frame.worlds().take(2).collect();
    let answer = Answer(vec![Value::Chair(Seat::S1)]);
    let target = step_world(&frame, &worlds[0], lead("6-6")).unwrap();
    let later = vec![(target, answer.clone())];
    let anchored =
        anchor_back(&frame, lead("6-6"), worlds.iter().map(|w| (*w, answer.clone())), &later, &mut small_budget())
            .unwrap();
    assert_eq!(anchored, vec![(worlds[0], answer)]);
}

#[test]
fn hindsight_anchor_validates_empty_inputs_and_deduplicates_relation_rows() {
    let frame = initial();
    let malformed = ObservedPlay { actor: Seat::S1, tile: tile("6-6"), class: PlayClass::Lead };
    assert!(anchor_back(&frame, malformed, Vec::new(), &[], &mut small_budget()).is_err());
    let world = frame.worlds().next().unwrap();
    let answer = Answer(vec![Value::Chair(Seat::S1)]);
    let successor = step_world(&frame, &world, lead("6-6")).unwrap();
    let anchored = anchor_back(
        &frame,
        lead("6-6"),
        vec![(world, answer.clone()), (world, answer.clone())],
        &[(successor, answer.clone()), (successor, answer.clone())],
        &mut small_budget(),
    )
    .unwrap();
    assert_eq!(anchored, vec![(world, answer)]);
}

#[test]
fn a_resolving_belief_steps_exactly_when_the_play_reads_only_resolved_tiles() {
    let full = Belief::uniform(initial(), 90).unwrap();
    let resolving = Belief::uniform_resolving(initial(), tiles(&["6-5", "5-5"]), 90).unwrap();
    // The viewer's lead reads nothing hidden: both measures step alike.
    let a = step_belief(&full, lead("6-6"), "led", 90, &mut small_budget()).unwrap();
    let b = step_belief(&resolving, lead("6-6"), "led", 90, &mut small_budget()).unwrap();
    assert_eq!(b.belief.unresolved(), resolving.unresolved());
    let partner = compile("(query (roles) (out) (pattern (holds S2 6-5)))");
    assert_eq!(
        partner.summarize(&a.belief, &mut budget()).unwrap().event_probability(),
        partner.summarize(&b.belief, &mut budget()).unwrap().event_probability()
    );
    // S1's follow depends on every trump it might hold, and 6-4 is unresolved.
    let follow = ObservedPlay { actor: Seat::S1, tile: tile("6-5"), class: PlayClass::Follow(Context::Called) };
    assert!(step_belief(&b.belief, follow, "followed", 90, &mut small_budget()).is_err());
}
