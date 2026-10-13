//! Finite dynamics: one typed public play applied to a frame, a world, and a
//! belief. These are extensional transforms on finite states.
//!
//! A [`Belief`] is a measure over physical hands only. A caller whose field
//! policy has latent state must carry that state itself and supply the
//! resulting likelihood.

use crate::belief::{Belief, given, probability};
use crate::eval::{Answer, Answers, Budget};
use crate::frame::{Frame, HIDDEN_SEATS, Holdings, Kernel, World};
use crate::measure::{Estimate, Weight};
use crate::nouns::{Context, Domino, DominoSet, Seat};
use crate::{Result, error};

/// The public classification of a play.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayClass {
    Lead,
    Follow(Context),
    Slough(Context),
}

/// An actor-attributed, context-typed play.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservedPlay {
    pub actor: Seat,
    pub tile: Domino,
    pub class: PlayClass,
}

#[derive(Clone, Debug)]
pub struct BeliefStep {
    pub belief: Belief,
    /// The observation's prior probability under the supplied likelihood.
    pub evidence_probability: Estimate,
}

/// Check the play's actor and classification against the public frame.
fn typed(frame: &Frame, play: ObservedPlay) -> Result<()> {
    if frame.next_actor() != Some(play.actor) {
        return Err(error(format!("the observed actor {} is not the next actor {:?}", play.actor, frame.next_actor())));
    }
    let follows = |q: Context| frame.physics().follows(play.tile, q);
    match (frame.led_context(), play.class) {
        (None, PlayClass::Lead) => Ok(()),
        (Some(q), PlayClass::Follow(r)) if q == r && follows(q) => Ok(()),
        (Some(q), PlayClass::Slough(r)) if q == r && !follows(q) => Ok(()),
        _ => Err(error("the play's classification or context disagrees with the public frame")),
    }
}

fn successor_kernel(frame: &Frame, play: ObservedPlay) -> Result<Kernel> {
    let kernel = frame.kernel();
    let mut viewer_hand = kernel.viewer_hand();
    let mut pool = kernel.pool();
    let mut hidden = *kernel.hidden();
    if play.actor == kernel.viewer() {
        if !viewer_hand.remove(play.tile) {
            return Err(error("the viewer does not hold the observed tile"));
        }
    } else {
        if !pool.remove(play.tile) {
            return Err(error("the observed tile is outside the hidden pool"));
        }
        let slot = hidden.iter_mut().find(|h| h.seat == play.actor).expect("every other seat is hidden");
        if slot.capacity == 0 {
            return Err(error("the observed actor has no tiles"));
        }
        slot.capacity -= 1;
        if let PlayClass::Slough(q) = play.class {
            slot.voids.insert(q);
        }
    }
    Kernel::new(kernel.viewer(), viewer_hand, pool, hidden)
}

/// Apply a play to the public frame and its support. The play must be legal
/// in at least one predecessor world. The fourth play closes the trick and
/// seats its winner as the next leader.
pub fn step_frame(frame: &Frame, play: ObservedPlay, budget: &mut Budget) -> Result<Frame> {
    typed(frame, play)?;
    let kernel = frame.kernel();
    let physics = frame.physics();
    if play.actor == kernel.viewer() {
        if !physics.legal_plays(kernel.viewer_hand(), frame.lead()).contains(play.tile) {
            return Err(error("the viewer's play is illegal in the predecessor frame"));
        }
    } else {
        let slot = kernel.hidden().iter().position(|h| h.seat == play.actor).expect("every other seat is hidden");
        if !kernel.allowed(physics, slot).contains(play.tile) {
            return Err(error("the hidden play contradicts the actor's shown voids"));
        }
    }
    budget.spend(1)?;
    let next = successor_kernel(frame, play)?;
    if physics.count_worlds(&next) == 0 {
        return Err(error("no predecessor world makes this play legal"));
    }
    let mut prefix = frame.prefix().to_vec();
    prefix.push(play.tile);
    let mut leader = frame.leader();
    if let Ok(trick) = <[Domino; 4]>::try_from(prefix.as_slice()) {
        leader = physics.trick_winner(leader, trick);
        prefix.clear();
    }
    frame.after(next, leader, prefix, (play.actor, play.tile))
}

fn legal_in(holdings: Holdings<'_>, play: ObservedPlay) -> Result<bool> {
    let frame = holdings.frame();
    typed(frame, play)?;
    Ok(frame.physics().legal_plays(holdings.hand(play.actor)?, frame.lead()).contains(play.tile))
}

/// Whether a world of the frame is in the play's legal domain.
pub fn play_is_legal(frame: &Frame, world: &World, play: ObservedPlay) -> bool {
    typed(frame, play).is_ok()
        && frame.contains(world)
        && legal_in(Holdings::new(frame, world, DominoSet::EMPTY), play).expect("a fully resolved read")
}

/// Push one legal predecessor world through the play.
pub fn step_world(frame: &Frame, world: &World, play: ObservedPlay) -> Result<World> {
    typed(frame, play)?;
    if !play_is_legal(frame, world, play) {
        return Err(error("the world is outside the play's legal domain"));
    }
    let next = successor_kernel(frame, play)?;
    let hands: [DominoSet; HIDDEN_SEATS] =
        std::array::from_fn(|slot| world.hand(next.hidden()[slot].seat).difference(DominoSet::single(play.tile)));
    let world = next.world(hands);
    if !next.contains(frame.physics(), &world) {
        return Err(error("the successor world violates the updated support"));
    }
    Ok(world)
}

/// Condition a belief on a play whose likelihood is one in every world where
/// it is legal: a forced play, or a viewer's act chosen independently of the
/// hidden deal. Authority is preserved.
pub fn step_belief(
    prior: &Belief,
    play: ObservedPlay,
    posterior_id: impl Into<String>,
    max_worlds: usize,
    budget: &mut Budget,
) -> Result<BeliefStep> {
    step(prior, play, posterior_id.into(), max_worlds, budget, prior.authority().clone(), |_| Ok(Weight::one()))
}

/// Condition a belief on a play under a declared policy likelihood from the
/// pinned `model`: drop worlds where the play is illegal, multiply by the
/// likelihood in `[0, 1]`, push each world forward, and merge.
pub fn step_belief_given(
    prior: &Belief,
    play: ObservedPlay,
    posterior_id: impl Into<String>,
    max_worlds: usize,
    budget: &mut Budget,
    model: &str,
    likelihood: impl FnMut(Holdings<'_>) -> Result<Weight>,
) -> Result<BeliefStep> {
    let authority = prior.authority().meet(&given(model)?)?;
    step(prior, play, posterior_id.into(), max_worlds, budget, authority, likelihood)
}

fn step(
    prior: &Belief,
    play: ObservedPlay,
    posterior_id: String,
    max_worlds: usize,
    budget: &mut Budget,
    authority: crate::Authority,
    mut likelihood: impl FnMut(Holdings<'_>) -> Result<Weight>,
) -> Result<BeliefStep> {
    typed(prior.frame(), play)?;
    budget.spend(prior.len() as u64)?;
    let frame = step_frame(prior.frame(), play, budget)?;
    let mut entries = Vec::new();
    let mut evidence = Weight::zero();
    for (world, weight) in prior.worlds() {
        let holdings = prior.holdings(world);
        if !legal_in(holdings, play)? {
            continue;
        }
        let weighted = weight * &probability(likelihood(holdings)?)?;
        if weighted.is_zero() {
            continue;
        }
        evidence += &weighted;
        entries.push((step_world(prior.frame(), world, play)?, weighted));
    }
    if evidence.is_zero() {
        return Err(error("a zero-mass observation under the supplied likelihood"));
    }
    let belief = prior.rebuild(frame, posterior_id, authority, entries, max_worlds)?;
    let evidence_probability =
        Estimate::new(evidence.share_of(prior.total_mass()).rational().clone(), belief.authority().clone());
    Ok(BeliefStep { belief, evidence_probability })
}

/// Rigid transport: dominoes, chairs and contexts keep their identity.
pub fn transport_answers(prior: &Answers) -> Answers {
    prior.clone()
}

/// Transported prior answers set against a fresh evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnswerDynamics {
    pub transported: Answers,
    pub persistent: Answers,
    pub extinct: Answers,
    pub born: Answers,
}

pub fn compare_answers(prior: &Answers, fresh: &Answers) -> AnswerDynamics {
    let transported = transport_answers(prior);
    AnswerDynamics {
        persistent: transported.intersection(fresh).cloned().collect(),
        extinct: transported.difference(fresh).cloned().collect(),
        born: fresh.difference(&transported).cloned().collect(),
        transported,
    }
}

/// The preimage of explicit later `(world, answer)` pairs under the play:
/// analyst hindsight, which adds nothing to an earlier player's information.
pub fn anchor_back(
    frame: &Frame,
    play: ObservedPlay,
    prior: impl IntoIterator<Item = (World, Answer)>,
    later: &[(World, Answer)],
    budget: &mut Budget,
) -> Result<Vec<(World, Answer)>> {
    typed(frame, play)?;
    let prior: Vec<_> = prior.into_iter().collect();
    budget.spend(prior.len() as u64)?;
    let mut out: Vec<(World, Answer)> = Vec::new();
    for (world, answer) in prior {
        if play_is_legal(frame, &world, play) {
            let successor = step_world(frame, &world, play)?;
            if later.iter().any(|(w, a)| *w == successor && *a == answer) && !out.contains(&(world, answer.clone())) {
                out.push((world, answer));
            }
        }
    }
    Ok(out)
}
