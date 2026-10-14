//! Solved outcomes: what the viewer can guarantee in every world he cannot
//! tell apart, against every play of the other three seats, and the
//! double-dummy bound above it. The statement, and why each answer is exact,
//! is the README's "Solved outcomes".

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use crate::eval::Budget;
use crate::frame::{Frame, HIDDEN_SEATS, Holdings, Kernel};
use crate::nouns::{Domino, DominoSet, Seat};
use crate::physics::Physics;
use crate::registry::{Access, Phase, Predicate, PredicateContext, PredicateSpec};
use crate::solve::{Objective, OpenPosition, Solver};
use crate::syntax::Value;
use crate::{Result, error};

/// The most worlds read in looking for one where double-dummy play fails the
/// viewer's goal.
pub const OUTCOME_SCAN_CAP: u128 = 10_000;

/// What a side plays for: the declaring side the make, the other the set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Goal {
    Make,
    Set,
}

impl Goal {
    /// The goal of `seat`'s side under the frame's contract.
    pub fn of(frame: &Frame, seat: Seat) -> Goal {
        if seat.team() == frame.contract().declarer().team() { Goal::Make } else { Goal::Set }
    }

    /// Whether a hand the declaring side makes (`made`) ends in this goal.
    fn reached(self, made: bool) -> bool {
        made == (self == Goal::Make)
    }
}

/// Whether some strategy of the viewer's, a choice of tile at each of his
/// frames, ends the hand in `goal` in every world of `frame` against every
/// legal play of the other three seats, his partner's included. Refused
/// unless `goal` is his own side's, and when the work exceeds the budget.
pub fn forces(frame: &Frame, goal: Goal, budget: &mut Budget) -> Result<bool> {
    if Goal::of(frame, frame.viewer()) != goal {
        return Err(error("the viewer can be asked to force only his own side's goal"));
    }
    let mut search = Search { goal, memo: HashMap::new() };
    if let Some(answer) = search.proved(frame, budget)? {
        return Ok(answer);
    }
    if refuted(frame, goal, budget)? {
        return Ok(false);
    }
    search.expand(frame, budget)
}

/// The viewer's lay-down: a sequence of his tiles (his play to the open trick
/// if he is following, then one lead per trick) each of which wins its trick
/// in every world of `frame` against every legal play of the other three
/// seats; `None` when there is none. Refused unless the viewer is to play.
pub fn lay_down(frame: &Frame, budget: &mut Budget) -> Result<Option<Vec<Domino>>> {
    if frame.next_actor() != Some(frame.viewer()) {
        return Err(error("a lay-down is the viewer's line when he is to play"));
    }
    let physics = frame.physics();
    let kernel = frame.kernel();
    let mut states: [Vec<DominoSet>; HIDDEN_SEATS] = Default::default();
    for (slot, state) in states.iter_mut().enumerate() {
        *state = possible_hands(physics, kernel, slot, budget)?;
    }
    let mine = kernel.viewer_hand();
    let Some(lead) = frame.lead() else {
        return leads(physics, mine, &states, budget);
    };
    let led = physics.led_context(lead);
    let best = frame.prefix().iter().map(|d| physics.trick_key(*d, led)).max().expect("an open trick has a lead");
    // The hidden slots run clockwise from the viewer: the first `still` of
    // them have yet to play to the open trick.
    let still = Seat::COUNT - 1 - frame.prefix().len();
    for tile in physics.legal_plays(mine, Some(lead)).iter().filter(|d| physics.trick_key(*d, led) > best) {
        if let Some(after) = replies(physics, lead, tile, &states, still, budget)?
            && let Some(line) = leads(physics, without(mine, tile), &after, budget)?
        {
            return Ok(Some([vec![tile], line].concat()));
        }
    }
    Ok(None)
}

/// The and-or search over the viewer's information states, memoized on the
/// record: every play is public, so his state after a history is that
/// history's frame, and its fiber is the worlds still possible.
struct Search {
    goal: Goal,
    memo: HashMap<Vec<Domino>, bool>,
}

impl Search {
    /// The answer at `frame` without search: a settled hand, a goal out of
    /// reach, or the viewer to play with a lay-down.
    fn proved(&self, frame: &Frame, budget: &mut Budget) -> Result<Option<bool>> {
        if let Some(answer) = settled(frame, self.goal) {
            return Ok(Some(answer));
        }
        // An unsettled goal is in reach: taking every trick reaches it.
        let lays_down = frame.next_actor() == Some(frame.viewer()) && lay_down(frame, budget)?.is_some();
        Ok(lays_down.then_some(true))
    }

    fn value(&mut self, frame: &Frame, budget: &mut Budget) -> Result<bool> {
        if let Some(answer) = self.memo.get(frame.plays()) {
            return Ok(*answer);
        }
        let answer = match self.proved(frame, budget)? {
            Some(answer) => answer,
            None => self.expand(frame, budget)?,
        };
        self.memo.insert(frame.plays().to_vec(), answer);
        Ok(answer)
    }

    /// Some legal tile of the viewer's wins; every tile some world lets
    /// another seat play loses. A tile no world lets it play has no frame.
    fn expand(&mut self, frame: &Frame, budget: &mut Budget) -> Result<bool> {
        budget.spend(1)?;
        let kernel = frame.kernel();
        if frame.next_actor() == Some(frame.viewer()) {
            for tile in frame.physics().legal_plays(kernel.viewer_hand(), frame.lead()).iter() {
                if self.value(&frame.after(tile)?, budget)? {
                    return Ok(true);
                }
            }
            return Ok(false);
        }
        for tile in kernel.pool().iter() {
            if let Ok(next) = frame.after(tile)
                && !self.value(&next, budget)?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// The answer for `goal`'s side when the hand has decided it: true once the
/// goal is reached, false once the other side's is, or once the goal is out
/// of reach even taking every remaining trick.
fn settled(frame: &Frame, goal: Goal) -> Option<bool> {
    let (to_make, to_set) = frame.to_make_and_set();
    let [a, b] = frame.banked();
    let left = frame.physics().total_points() - a - b;
    let (ours, theirs) = match goal {
        Goal::Make => (to_make, to_set),
        Goal::Set => (to_set, to_make),
    };
    if ours == 0 {
        Some(true)
    } else if theirs == 0 || ours > left {
        Some(false)
    } else {
        None
    }
}

/// Whether one of the first [`OUTCOME_SCAN_CAP`] worlds of `frame`, in an
/// order that would visit each once, fails `goal` under double-dummy play.
/// The defence that sets every declaring line there sets every strategy of
/// the viewer's, so one such world refutes `goal`; none found proves nothing.
fn refuted(frame: &Frame, goal: Goal, budget: &mut Budget) -> Result<bool> {
    let n = frame.count_worlds();
    let step = stride(n);
    let order = (0..n.min(OUTCOME_SCAN_CAP)).map(move |i| i * step);
    for world in frame.physics().worlds_at(frame.kernel(), order) {
        if !goal.reached(double_dummy_make(&OpenPosition::at(frame, &world)?, budget)?) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// A step coprime to `n`, near `n` over the golden ratio: `i * step mod n`
/// visits every world once and spreads the first ones over the fiber. A fiber
/// deals at most 21 tiles, so `n` is below 2^64.
fn stride(n: u128) -> u128 {
    const GOLDEN: u128 = 0x9e37_79b9_7f4a_7c15;
    let gcd = |mut a: u128, mut b: u128| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let mut step = ((n * GOLDEN) >> 64).max(1);
    while gcd(step, n) != 1 {
        step += 1;
    }
    step
}

fn double_dummy_make(position: &OpenPosition, budget: &mut Budget) -> Result<bool> {
    let mut solver = Solver::new(position.physics());
    let made = solver.value(position, Objective::Make) == 1;
    budget.spend(solver.nodes())?;
    Ok(made)
}

/// Every hand hidden slot `slot` holds in some world of `kernel`, each once:
/// a hand of its capacity from the tiles its voids allow, whose rest the other
/// two slots can hold (Hall's condition for two).
pub(crate) fn possible_hands(
    physics: &Physics,
    kernel: &Kernel,
    slot: usize,
    budget: &mut Budget,
) -> Result<Vec<DominoSet>> {
    let allowed: [DominoSet; HIDDEN_SEATS] = std::array::from_fn(|j| kernel.allowed(physics, j));
    let room = kernel.hidden().map(|h| h.capacity);
    let (a, b) = ((slot + 1) % HIDDEN_SEATS, (slot + 2) % HIDDEN_SEATS);
    let tiles: Vec<Domino> = allowed[slot].iter().collect();
    let mut hands = Vec::new();
    choose(&tiles, room[slot], DominoSet::EMPTY, &mut |hand| {
        budget.spend(1)?;
        let rest = kernel.pool().difference(hand);
        if rest.is_subset_of(allowed[a].union(allowed[b]))
            && rest.difference(allowed[b]).len() <= room[a]
            && rest.difference(allowed[a]).len() <= room[b]
        {
            hands.push(hand);
        }
        Ok(())
    })?;
    Ok(hands)
}

/// Every `k` of `tiles`, each added to `chosen`.
fn choose(tiles: &[Domino], k: usize, chosen: DominoSet, visit: &mut dyn FnMut(DominoSet) -> Result<()>) -> Result<()> {
    match tiles.split_first() {
        _ if k == 0 => visit(chosen),
        Some((first, rest)) if rest.len() + 1 >= k => {
            let mut with = chosen;
            with.insert(*first);
            choose(rest, k - 1, with, visit)?;
            choose(rest, k, chosen, visit)
        }
        _ => Ok(()),
    }
}

/// The first `seats` hidden slots reply to `tile`, played to a trick led by
/// `lead`: `None` when some hand one of them may hold has a legal reply that
/// takes the trick from it, else every hand each may hold after.
fn replies(
    physics: &Physics,
    lead: Domino,
    tile: Domino,
    states: &[Vec<DominoSet>; HIDDEN_SEATS],
    seats: usize,
    budget: &mut Budget,
) -> Result<Option<[Vec<DominoSet>; HIDDEN_SEATS]>> {
    let over = physics.beats(physics.led_context(lead), tile);
    let mut after = states.clone();
    for (state, next) in states.iter().zip(after.iter_mut()).take(seats) {
        let mut hands = Vec::with_capacity(state.len());
        for hand in state {
            budget.spend(1)?;
            let legal = physics.legal_plays(*hand, Some(lead));
            if !legal.is_disjoint(over) {
                return Ok(None);
            }
            hands.extend(legal.iter().map(|d| without(*hand, d)));
        }
        hands.sort_unstable();
        hands.dedup();
        *next = hands;
    }
    Ok(Some(after))
}

/// An order of `mine`, one lead per trick, each winning its trick against
/// every reply the hands in `states` allow.
fn leads(
    physics: &Physics,
    mine: DominoSet,
    states: &[Vec<DominoSet>; HIDDEN_SEATS],
    budget: &mut Budget,
) -> Result<Option<Vec<Domino>>> {
    if mine.is_empty() {
        return Ok(Some(Vec::new()));
    }
    // The tiles fewest others can beat first: the order is speed, not answer.
    let mut order: Vec<Domino> = mine.iter().collect();
    order.sort_by_key(|d| physics.threat(*d).len());
    for tile in order {
        if let Some(after) = replies(physics, tile, tile, states, HIDDEN_SEATS, budget)?
            && let Some(line) = leads(physics, without(mine, tile), &after, budget)?
        {
            return Ok(Some([vec![tile], line].concat()));
        }
    }
    Ok(None)
}

fn without(mut hand: DominoSet, tile: Domino) -> DominoSet {
    hand.remove(tile);
    hand
}

/// The outcome predicates, version `silas-v3/outcome`.
pub(crate) fn predicates() -> [Outcome; 4] {
    [
        ("lay-down", Kind::LayDown, Access::Viewer),
        ("certain-make", Kind::Certain(Goal::Make), Access::Viewer),
        ("certain-set", Kind::Certain(Goal::Set), Access::Viewer),
        ("dd-make", Kind::DoubleDummyMake, Access::World),
    ]
    .map(|(name, kind, access)| Outcome {
        spec: PredicateSpec {
            name: name.to_owned(),
            version: "silas-v3/outcome".to_owned(),
            parameters: vec![],
            horizon_plies: Domino::COUNT as u32,
            access,
            phase: Phase::Play,
        },
        kind,
        last: Mutex::new(None),
    })
}

#[derive(Clone, Copy)]
enum Kind {
    LayDown,
    Certain(Goal),
    DoubleDummyMake,
}

/// One outcome predicate. A viewer fact keeps the last frame's answer: it is
/// asked once per binding of a query, and its work is the whole fiber's.
pub(crate) struct Outcome {
    spec: PredicateSpec,
    kind: Kind,
    last: Mutex<Option<(Frame, Option<bool>)>>,
}

impl Outcome {
    fn at_frame(&self, frame: &Frame, budget: &mut Budget) -> Result<Option<bool>> {
        let viewer = frame.viewer();
        match self.kind {
            Kind::LayDown if frame.next_actor() == Some(viewer) => Ok(Some(lay_down(frame, budget)?.is_some())),
            Kind::Certain(goal) if Goal::of(frame, viewer) == goal => forces(frame, goal, budget).map(Some),
            _ => Ok(None),
        }
    }
}

impl Predicate for Outcome {
    fn spec(&self) -> &PredicateSpec {
        &self.spec
    }

    fn evaluate(&self, context: PredicateContext<'_>, _: &[Value], budget: &mut Budget) -> Result<Option<bool>> {
        if let Kind::DoubleDummyMake = self.kind {
            let holdings = context.world().ok_or_else(|| error("a world predicate requires a world"))?;
            return double_dummy_in(holdings, budget).map(Some);
        }
        let frame = context.frame()?;
        let mut last = self.last.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((seen, answer)) = last.as_ref()
            && seen == frame
        {
            return Ok(*answer);
        }
        let answer = self.at_frame(frame, budget)?;
        *last = Some((frame.clone(), answer));
        Ok(answer)
    }
}

/// Whether the declaring side makes from the frame in this world under
/// double-dummy play. It reads every hidden hand, so a belief that leaves a
/// tile unresolved refuses it.
fn double_dummy_in(holdings: Holdings<'_>, budget: &mut Budget) -> Result<bool> {
    let mut hands = [DominoSet::EMPTY; Seat::COUNT];
    for seat in Seat::ALL {
        hands[seat.index()] = holdings.hand(seat)?;
    }
    double_dummy_make(&OpenPosition::of(holdings.frame(), hands)?, budget)
}
