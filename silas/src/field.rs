// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The field and the paired comparison (`docs/cross-game.md` § 2.3 and
//! § 2.4, slice 3). A [`Field`] is four [`Player`]s, one a seat; a hand is
//! played out by them, every seat its own player at its own frame: the
//! auction, the declaration, then all 28 plays ([`play_out`]). A [`Trial`]
//! asks how much better one player does than another in a chair of that
//! field, over a deal measure ([`Deals`]) and a set of seatings ([`Seats`]),
//! from one match state, under one [`Utility`]: the two play the same deal,
//! in the same chair, against the same three other seats, and the answer
//! ([`Paired`]) is the mean of the per-deal differences, with its flips and
//! an interval over deals. Every comparison is run against the chance
//! player too.
//!
//! A field is a model, so every answer rests on it (`exact_given(field)`),
//! and on its sample when the deals were drawn; none is `exact`.

use std::sync::Arc;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use crate::auction::{Auction, AuctionConfig, AuctionFrame, Call};
use crate::eval::Budget;
use crate::frame::{Bid, Contract, Frame, World};
use crate::measure::{Authority, Estimate, Sample, Verdict};
use crate::nouns::{Domino, DominoSet, Seat, Team};
use crate::physics::Physics;
use crate::policy::{CompiledPolicy, Trump};
use crate::reading::digest;
use crate::scoring::{MatchState, Outcome, Utility};
use crate::{Result, error};

/// Who sits in a seat: a Silas policy, which bids, declares and plays by its
/// sections (a section it lacks is its fallback), or the chance player.
///
/// The chance player is the arena's `RandomBidder` and lowest-legal play,
/// drawn exactly: at its turn to call it bids with probability 3/10 (always,
/// when it may not pass), uniformly among the legal bids of at most one
/// mark; it declares uniformly among the pip suits offered (among all
/// offered, if no pip suit is); it plays its lowest legal tile. Its draws
/// are keyed on its seed and its auction frame (its seat, its tiles, the
/// shaker and the calls so far), so it too is a function of what it sees,
/// and the same seed calls the same way at the same frame.
#[derive(Clone)]
pub enum Player {
    Policy(Arc<CompiledPolicy>),
    Chance { seed: u64 },
}

impl Player {
    /// The chance player every comparison is also run against.
    pub const CHANCE: Player = Player::Chance { seed: 0 };

    /// `NAME#DIGEST` (the policy's name and the digest of its identity, as
    /// a reading names a policy), or `chance#SEED`.
    pub fn identity(&self) -> String {
        match self {
            Player::Policy(policy) => format!("{}#{:016x}", policy.source().name, digest(policy.identity())),
            Player::Chance { seed } => format!("chance#{seed}"),
        }
    }

    fn call(&self, frame: &AuctionFrame, budget: &mut Budget) -> Result<Call> {
        let seed = match self {
            Player::Policy(policy) => return Ok(policy.bid(frame, budget)?.choice),
            Player::Chance { seed } => *seed,
        };
        let auction = frame.auction();
        let legal = auction.legal_calls();
        let bids: Vec<Call> =
            legal.iter().copied().filter(|c| matches!(c, Call::Bid(b) if *b <= Bid::Marks(1))).collect();
        let mut draws = Stream::keyed(seed, frame);
        if bids.is_empty() || (legal.contains(&Call::Pass) && draws.below(10) >= 3) {
            return Ok(Call::Pass);
        }
        Ok(bids[draws.below(bids.len())])
    }

    fn declare(&self, frame: &AuctionFrame, budget: &mut Budget) -> Result<Physics> {
        let seed = match self {
            Player::Policy(policy) => return Ok(policy.declare(frame, budget)?.choice),
            Player::Chance { seed } => *seed,
        };
        let offered = frame.declarations();
        let suits: Vec<&Physics> = offered.iter().filter(|d| matches!(Trump::of(d), Some(Trump::Suit(_)))).collect();
        let menu = if suits.is_empty() { offered.iter().collect() } else { suits };
        Ok(menu[Stream::keyed(seed, frame).below(menu.len())].clone())
    }

    fn play(&self, frame: &Frame, budget: &mut Budget) -> Result<Domino> {
        match self {
            Player::Policy(policy) => policy.choose(frame, budget),
            Player::Chance { .. } => frame
                .physics()
                .legal_plays(frame.kernel().viewer_hand(), frame.lead())
                .iter()
                .next()
                .ok_or_else(|| error("the chance player has no legal play")),
        }
    }
}

/// The four players, by seat. Its identity is theirs, seat by seat, so a
/// field cannot be named for what it is not.
#[derive(Clone)]
pub struct Field {
    seats: [Player; Seat::COUNT],
}

impl Field {
    pub fn new(seats: [Player; Seat::COUNT]) -> Field {
        Field { seats }
    }

    pub fn seat(&self, seat: Seat) -> &Player {
        &self.seats[seat.index()]
    }

    /// `S0 a#…, S1 b#…, S2 c#…, S3 d#…`.
    pub fn id(&self) -> String {
        Seat::ALL.iter().map(|s| format!("{s} {}", self.seat(*s).identity())).collect::<Vec<_>>().join(", ")
    }
}

/// What the table plays under, beyond the players: the auction's rules, the
/// declarations its winner may choose from (in order: a declaring policy's
/// tie-break), and the match state the hand is played from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Table {
    pub config: AuctionConfig,
    pub offered: Arc<[Physics]>,
    pub state: MatchState,
}

/// How one hand ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Settlement {
    /// The auction, the contract it settled under the declarer's
    /// declaration, the outcome (the points each side took, all 28 tiles
    /// played) and the plays with their seats.
    Played { auction: Auction, contract: Contract, outcome: Outcome, plays: Vec<(Seat, Domino)> },
    /// Four passes without the forced bid: no contract and no play. The
    /// next seat shakes, which is another deal, so this hand moves no score.
    PassedOut { auction: Auction },
}

impl Settlement {
    pub fn auction(&self) -> &Auction {
        match self {
            Settlement::Played { auction, .. } | Settlement::PassedOut { auction } => auction,
        }
    }

    /// What the hand is worth to `team` from `state`: the utility of its
    /// contract and outcome; for a passed-out hand, which moves no score,
    /// nothing in marks or points and the value of `state` itself.
    pub fn value(&self, utility: &Utility, team: Team, state: &MatchState) -> Result<BigRational> {
        match (self, utility) {
            (Settlement::Played { contract, outcome, .. }, _) => utility.of(team, state, contract, *outcome),
            (Settlement::PassedOut { .. }, Utility::MatchWin(value)) if value.team() != team => {
                Err(error(format!("a match-win value for {} asked for {team}", value.team())))
            }
            (Settlement::PassedOut { .. }, Utility::MatchWin(value)) => value.at(state),
            (Settlement::PassedOut { .. }, Utility::Marks | Utility::Points) => Ok(BigRational::zero()),
        }
    }
}

/// Play out the hand dealt `world`, shaken by `shaker`, by the field's
/// players at `table`: each seat calls by its player at its own auction
/// frame; the winner declares; then every one of the 28 tiles is played,
/// each seat choosing by its player at its own frame. Refused when a player
/// errs, plays a tile it may not, or exhausts the budget.
pub fn play_out(world: &World, shaker: Seat, field: &Field, table: &Table, budget: &mut Budget) -> Result<Settlement> {
    seated(world, shaker, &field.seats.each_ref(), table, budget)
}

fn seated(
    world: &World,
    shaker: Seat,
    players: &[&Player; Seat::COUNT],
    table: &Table,
    budget: &mut Budget,
) -> Result<Settlement> {
    let player = |seat: Seat| players[seat.index()];
    let at = |auction: &Auction, seat: Seat| {
        AuctionFrame::new(auction.clone(), seat, world.hand(seat), table.state, table.offered.clone())
    };
    let mut auction = Auction::new(table.config, shaker, vec![])?;
    while let Some(seat) = auction.actor() {
        auction = auction.after(player(seat).call(&at(&auction, seat)?, budget)?)?;
    }
    let Some((declarer, _)) = auction.result() else {
        return Ok(Settlement::PassedOut { auction });
    };
    let declaration = player(declarer).declare(&at(&auction, declarer)?, budget)?;
    if !table.offered.contains(&declaration) {
        return Err(error(format!("{declarer} declared {}, which is not offered", declaration.name())));
    }
    let contract = auction.contract(declaration.clone())?;
    let physics = contract.declaration();
    let mut hands: [DominoSet; Seat::COUNT] = Seat::ALL.map(|s| world.hand(s));
    let mut plays = Vec::with_capacity(Domino::COUNT);
    let mut history = Vec::with_capacity(Domino::COUNT);
    let mut banked = [0u32; 2];
    let mut leader = declarer;
    for _ in 0..Domino::COUNT / Seat::COUNT {
        let mut trick = [Domino::ALL[0]; Seat::COUNT];
        for k in 0..Seat::COUNT {
            let seat = leader.plus(k);
            let frame = Frame::new(auction.clone(), declaration.clone(), seat, world.hand(seat), plays.clone())?;
            let tile = player(seat).play(&frame, budget)?;
            if !physics.legal_plays(hands[seat.index()], (k > 0).then_some(trick[0])).contains(tile) {
                return Err(error(format!("{} played {tile}, which {seat} may not play", player(seat).identity())));
            }
            hands[seat.index()].remove(tile);
            plays.push(tile);
            history.push((seat, tile));
            trick[k] = tile;
        }
        leader = physics.trick_winner(leader, trick);
        banked[leader.team().index()] += physics.trick_points(trick);
    }
    let side = declarer.team();
    let outcome = Outcome::new(banked[side.index()], banked[side.other().index()]);
    Ok(Settlement::Played { auction, contract, outcome, plays: history })
}

/// The deals a comparison is over.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Deals {
    /// These deals, each once: the answer is exact over the list, and a
    /// claim about these deals alone.
    Listed(Vec<World>),
    /// `n` deals drawn uniformly from the stream seeded `seed`
    /// ([`Deals::worlds`]): the answer is sampled.
    Uniform(Sample),
}

impl Deals {
    /// The deals, in order. A uniform sample is a Fisher–Yates shuffle of
    /// the 28 tiles per deal, seven to each seat from S0, every draw exactly
    /// uniform (splitmix64, the high draws rejected), so each of the
    /// 472,518,347,558,400 deals is equally likely and the same seed deals
    /// the same hands.
    pub fn worlds(&self) -> Vec<World> {
        match self {
            Deals::Listed(worlds) => worlds.clone(),
            Deals::Uniform(sample) => {
                let mut stream = Stream(sample.seed);
                (0..sample.n)
                    .map(|_| {
                        let mut deck = Domino::ALL;
                        for i in (1..deck.len()).rev() {
                            deck.swap(i, stream.below(i + 1));
                        }
                        let hands = std::array::from_fn(|s| deck[7 * s..7 * s + 7].iter().copied().collect());
                        World::deal(hands).expect("a shuffle deals seven tiles to each seat")
                    })
                    .collect()
            }
        }
    }
}

/// The chairs the player under test takes, each at every shaker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seats {
    /// Every chair: sixteen seatings a deal.
    Rotate,
    /// One chair: four seatings a deal.
    Fixed(Seat),
}

impl Seats {
    /// Each seating: the chair under test and the shaker.
    pub fn seatings(self) -> Vec<(Seat, Seat)> {
        let chairs = match self {
            Seats::Rotate => Seat::ALL.to_vec(),
            Seats::Fixed(chair) => vec![chair],
        };
        chairs.into_iter().flat_map(|chair| Seat::ALL.map(|shaker| (chair, shaker))).collect()
    }
}

/// One comparison's question, but for the two players: the field, the
/// deals, the seatings, the table, the utility of the side of the chair
/// under test, and the confidence of the interval (a rational strictly
/// between zero and one: 19/20 for 95%).
#[derive(Clone)]
pub struct Trial {
    pub field: Field,
    pub deals: Deals,
    pub seats: Seats,
    pub table: Table,
    pub utility: Utility,
    pub confidence: BigRational,
}

/// How much better one player did than another in the chair under test:
/// the mean of the per-deal differences in the utility's unit (a deal's
/// difference is its mean over the seatings: one cluster), how many deals,
/// on how many each did better, and an interval over deals at the trial's
/// confidence. The verdict is `Better` (or `Worse`) only when the interval
/// clears zero. Only Silas constructs one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Paired {
    delta: Estimate,
    per_deal: Vec<BigRational>,
    flips: (usize, usize, usize),
    interval: (BigRational, BigRational),
    verdict: Verdict,
    chance: Option<Box<Paired>>,
}

impl Paired {
    /// The mean per-deal difference, with its authority.
    pub fn delta(&self) -> &Estimate {
        &self.delta
    }

    /// The deals (the clusters), not the seatings.
    pub fn n(&self) -> usize {
        self.per_deal.len()
    }

    /// Each deal's difference, in order.
    pub fn per_deal(&self) -> &[BigRational] {
        &self.per_deal
    }

    /// The deals where the first player did better, where the second did,
    /// and where they tied.
    pub fn flips(&self) -> (usize, usize, usize) {
        self.flips
    }

    /// The interval for the mean difference. Over a listed set it is the
    /// mean itself, which is exact given the field; over a sample it is the
    /// empirical Bernstein bound (Maurer and Pontil 2009, Theorem 4, both
    /// sides) on the per-deal differences, which lie within the utility's
    /// range: with probability at least the confidence, the mean over every
    /// deal lies within it. Its half-width is rounded up to a millionth.
    pub fn interval(&self) -> &(BigRational, BigRational) {
        &self.interval
    }

    pub fn verdict(&self) -> Verdict {
        self.verdict
    }

    /// The same comparison with the second player replaced by the chance
    /// player ([`Player::CHANCE`]); none on that comparison itself.
    pub fn chance(&self) -> Option<&Paired> {
        self.chance.as_deref()
    }
}

impl Trial {
    /// Compare `a` against `b` in the chair under test ([`Paired`]), and `a`
    /// against the chance player alongside. Every deal is played at every
    /// seating by each, the chair's player replacing the field's there, on
    /// every core. Refused for a confidence outside (0, 1), for no deals, for
    /// a sample of fewer than two (the interval needs two), for a match-win
    /// value of the other side than a chair's, and for any refused play-out.
    pub fn compare(&self, a: &Player, b: &Player, budget: &mut Budget) -> Result<Paired> {
        if self.confidence <= BigRational::zero() || self.confidence >= BigRational::one() {
            return Err(error(format!("a confidence lies strictly between 0 and 1, not {}", self.confidence)));
        }
        let worlds = self.deals.worlds();
        let sampled = match &self.deals {
            Deals::Listed(_) if worlds.is_empty() => return Err(error("a comparison needs a deal")),
            Deals::Uniform(sample) if sample.n < 2 => {
                return Err(error("a sampled comparison needs two deals for its interval"));
            }
            Deals::Listed(_) => None,
            Deals::Uniform(sample) => Some(*sample),
        };
        let seatings = self.seats.seatings();
        if let Utility::MatchWin(value) = &self.utility
            && let Some((chair, _)) = seatings.iter().find(|(chair, _)| chair.team() != value.team())
        {
            return Err(error(format!(
                "the match-win value is {}'s, but the chair {chair} is on {}: seat the player under test on {}",
                value.team(),
                chair.team(),
                value.team()
            )));
        }
        let chance = Player::CHANCE;
        let differences = on_every_core(&worlds, budget, |world, budget| {
            let mut ab = BigRational::zero();
            let mut ac = BigRational::zero();
            for &(chair, shaker) in &seatings {
                let mut value = |player: &Player| {
                    let mut players = self.field.seats.each_ref();
                    players[chair.index()] = player;
                    seated(world, shaker, &players, &self.table, budget)?.value(
                        &self.utility,
                        chair.team(),
                        &self.table.state,
                    )
                };
                let va = value(a)?;
                ab += &va - value(b)?;
                ac += va - value(&chance)?;
            }
            let k = BigRational::from_integer(BigInt::from(seatings.len()));
            Ok((ab / &k, ac / k))
        })?;
        let mut authority = Authority::given(self.field.id())?.meet(&self.utility.authority());
        if let Some(sample) = sampled {
            authority = authority.meet(&Authority::sampled(sample.n, sample.seed));
        }
        let (ab, ac): (Vec<_>, Vec<_>) = differences.into_iter().unzip();
        let chance = self.summarize(ac, sampled.is_some(), authority.clone(), None);
        Ok(self.summarize(ab, sampled.is_some(), authority, Some(Box::new(chance))))
    }

    /// The width of the range a per-deal difference lies in. A hand's marks
    /// or points to a side lie within plus or minus the most it can be worth
    /// (in marks, the most four calls can reach under the cap; in points, a
    /// set marks bid at that most, its value and the setters' points: 42 a
    /// mark and 42), so a difference within plus or minus twice that; a
    /// match-win value lies in [0, 1], a difference in [-1, 1].
    fn range(&self) -> BigRational {
        let marks = self.table.config.max_marks().min(5);
        let total = self.table.offered.first().map_or(42, Physics::total_points);
        let width = match self.utility {
            Utility::Marks => 4 * marks,
            Utility::Points => 4 * total * (marks + 1),
            Utility::MatchWin(_) => 2,
        };
        BigRational::from_integer(BigInt::from(width))
    }

    fn summarize(
        &self,
        per_deal: Vec<BigRational>,
        sampled: bool,
        authority: Authority,
        chance: Option<Box<Paired>>,
    ) -> Paired {
        let n = BigRational::from_integer(BigInt::from(per_deal.len()));
        let mean: BigRational = per_deal.iter().sum::<BigRational>() / &n;
        let flips = per_deal.iter().fold((0, 0, 0), |(a, b, t), d| {
            if d.is_positive() {
                (a + 1, b, t)
            } else if d.is_negative() {
                (a, b + 1, t)
            } else {
                (a, b, t + 1)
            }
        });
        let half = if sampled {
            let one = BigRational::one();
            let variance: BigRational =
                per_deal.iter().map(|d| (d - &mean) * (d - &mean)).sum::<BigRational>() / (&n - &one);
            // Each side at half the miss: ln(2 / (miss / 2)).
            let log = ln_upper(&(BigRational::from_integer(BigInt::from(4)) / (one - &self.confidence)));
            let two = BigRational::from_integer(BigInt::from(2));
            let seven_thirds = BigRational::new(BigInt::from(7), BigInt::from(3));
            sqrt_upper(&(two * variance * &log / &n)) + seven_thirds * self.range() * log / (&n - BigRational::one())
        } else {
            BigRational::zero()
        };
        let half = round_up(&half, 1_000_000);
        let interval = (&mean - &half, &mean + &half);
        let verdict = if interval.0.is_positive() {
            Verdict::Better
        } else if interval.1.is_negative() {
            Verdict::Worse
        } else {
            Verdict::Indistinguishable
        };
        Paired { delta: Estimate::new(mean, authority), per_deal, flips, interval, verdict, chance }
    }
}

/// `f` on each world, on every core, in order. Each core draws on an equal
/// share of what `budget` has left, and `budget` is charged what they spent
/// together.
fn on_every_core<T: Send>(
    worlds: &[World],
    budget: &mut Budget,
    f: impl Fn(&World, &mut Budget) -> Result<T> + Sync,
) -> Result<Vec<T>> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(worlds.len().max(1));
    let share = budget.remaining() / threads as u64;
    let parts: Vec<Result<(Vec<T>, u64)>> = std::thread::scope(|s| {
        let handles: Vec<_> = worlds
            .chunks(worlds.len().div_ceil(threads).max(1))
            .map(|part| {
                let f = &f;
                s.spawn(move || {
                    let mut budget = Budget::new(share);
                    let out = part.iter().map(|w| f(w, &mut budget)).collect::<Result<Vec<T>>>()?;
                    Ok((out, budget.spent()))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a play-out thread")).collect()
    });
    let mut out = Vec::with_capacity(worlds.len());
    for part in parts {
        let (mut some, spent) = part?;
        budget.spend(spent)?;
        out.append(&mut some);
    }
    Ok(out)
}

/// A splitmix64 stream: the deals' and the chance player's draws.
struct Stream(u64);

impl Stream {
    /// The chance player's stream at `frame`: its seed, seat, tiles, the
    /// shaker and every call so far.
    fn keyed(seed: u64, frame: &AuctionFrame) -> Stream {
        let auction = frame.auction();
        let calls = auction.calls().iter().map(|c| match c {
            Call::Pass => 0,
            Call::Bid(bid) => 1 + u64::from(bid.value()),
        });
        let key = [frame.viewer().index() as u64, u64::from(frame.dealt().bits()), auction.shaker().index() as u64];
        let mut stream = Stream(seed);
        for word in key.into_iter().chain(calls) {
            stream = Stream(stream.next() ^ word);
        }
        stream
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Exactly uniform below `n`: a draw past the last whole multiple of `n`
    /// is drawn again.
    fn below(&mut self, n: usize) -> usize {
        let n = n as u64;
        let limit = u64::MAX - u64::MAX % n;
        loop {
            let x = self.next();
            if x < limit {
                return (x % n) as usize;
            }
        }
    }
}

/// `x` rounded up to a multiple of `1 / scale`.
fn round_up(x: &BigRational, scale: u64) -> BigRational {
    let scale = BigRational::from_integer(BigInt::from(scale));
    (x * &scale).ceil() / scale
}

/// An upper bound on `sqrt(x)` for `x >= 0`, within a billionth: for
/// `x = p / q`, `sqrt(p q) / q` with the root rounded up.
fn sqrt_upper(x: &BigRational) -> BigRational {
    if x.is_zero() {
        return BigRational::zero();
    }
    let scale = BigInt::from(1_000_000_000u64);
    let root = (x.numer() * x.denom() * &scale * &scale).sqrt() + BigInt::one();
    BigRational::new(root, x.denom() * scale)
}

/// An upper bound on `ln(x)` for `x >= 1`, rounded up to a billionth:
/// `x = 2^k r` with `r` in [1, 2), and `ln(x) = k ln(2) + ln(r)`, each
/// logarithm `2 atanh((r - 1) / (r + 1))` by 25 terms of its series (the
/// argument is at most a third) and a geometric bound on the rest.
fn ln_upper(x: &BigRational) -> BigRational {
    let one = BigRational::one();
    let two = BigRational::from_integer(BigInt::from(2));
    let mut r = x.clone();
    let mut k = 0u32;
    while r >= two {
        r /= &two;
        k += 1;
    }
    let ln = |r: &BigRational| -> BigRational {
        let y = (r - &one) / (r + &one);
        let y2 = &y * &y;
        let mut power = y.clone();
        let mut sum = BigRational::zero();
        let terms = 25;
        for j in 0..terms {
            sum += &power / BigRational::from_integer(BigInt::from(2 * j + 1));
            power *= &y2;
        }
        let tail = &power / (BigRational::from_integer(BigInt::from(2 * terms + 1)) * (&one - &y2));
        &two * (sum + tail)
    };
    round_up(&(BigRational::from_integer(BigInt::from(k)) * ln(&two) + ln(&r)), 1_000_000_000)
}
