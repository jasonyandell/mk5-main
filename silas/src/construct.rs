// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! Exact uniform construction on disjoint ownership cells. The query is an
//! existence event: overlapping witnesses never multiply a world's mass.
//! No policy likelihood enters this API. A sample is a world, not an exact
//! belief or an estimate. Work/cell exhaustion refuses the whole preparation.

use std::cell::Cell;

use crate::physics::Allocation;
use crate::*;

struct Completion<const N: usize> {
    placed: [DominoSet; N],
    residual: Allocation<N>,
    end: u128,
}

struct Cells<const N: usize> {
    cells: Vec<Completion<N>>,
    count: u128,
    base_count: u128,
    resolved: DominoSet,
}

impl<const N: usize> Cells<N> {
    fn prepare(
        pool: DominoSet,
        capacities: [usize; N],
        allowed: [DominoSet; N],
        max_cells: usize,
        budget: &mut Budget,
        mut evaluate: impl FnMut([DominoSet; N], DominoSet, &Cell<DominoSet>, &mut Budget) -> Result<bool>,
    ) -> Result<Self> {
        let base_count = allocation(pool, capacities, allowed, budget)?.count();
        let mut resolved = DominoSet::EMPTY;
        if base_count == 0 {
            return Ok(Self { cells: vec![], count: 0, base_count, resolved });
        }
        loop {
            let unresolved = pool.difference(resolved);
            let tiles: Vec<_> = resolved.iter().collect();
            let needed = Cell::new(DominoSet::EMPTY);
            let mut cells = Vec::new();
            let (mut visited, mut count) = (0usize, 0u128);
            place(
                &tiles,
                capacities,
                &allowed,
                &mut [DominoSet::EMPTY; N],
                budget,
                &mut |placed, remaining, budget| {
                    if visited == max_cells {
                        return Err(error(format!("construction exceeds its cap of {max_cells} ownership cells")));
                    }
                    visited += 1;
                    let residual = allocation(unresolved, remaining, allowed, budget)?;
                    let mass = residual.count();
                    if mass == 0 {
                        return Ok(());
                    }
                    let representative = residual.unrank(0);
                    let hands = std::array::from_fn(|s| placed[s].union(representative[s]));
                    // Each refusal has its own probe. A different error must not
                    // be hidden by an unresolved read at an earlier cell.
                    let read = Cell::new(DominoSet::EMPTY);
                    let accepts = evaluate(hands, unresolved, &read, budget);
                    if !read.get().is_empty() {
                        needed.set(needed.get().union(read.get()));
                    } else if accepts? {
                        count += mass;
                        cells.push(Completion { placed: *placed, residual, end: count });
                    }
                    Ok(())
                },
            )?;
            if needed.get().is_empty() {
                return Ok(Self { cells, count, base_count, resolved });
            }
            resolved = resolved.union(needed.get());
        }
    }

    fn unrank(&self, rank: u128) -> Result<[DominoSet; N]> {
        if rank >= self.count {
            return Err(error(format!("world rank {rank} is outside 0..{}", self.count)));
        }
        let i = self.cells.partition_point(|c| c.end <= rank);
        let cell = &self.cells[i];
        let start = if i == 0 { 0 } else { self.cells[i - 1].end };
        let rest = cell.residual.unrank(rank - start);
        Ok(std::array::from_fn(|s| cell.placed[s].union(rest[s])))
    }

    fn probability(&self) -> Option<Estimate> {
        (self.base_count > 0).then(|| {
            Estimate::new(
                Weight::integer(self.count).share_of(&Weight::integer(self.base_count)).rational().clone(),
                Authority::exact(),
            )
        })
    }
}

fn allocation<const N: usize>(
    pool: DominoSet,
    capacities: [usize; N],
    allowed: [DominoSet; N],
    budget: &mut Budget,
) -> Result<Allocation<N>> {
    // Charge an upper bound on the table's entries and outgoing edges.
    let entries = (pool.len() + 1) * capacities[..N - 1].iter().map(|c| c + 1).product::<usize>();
    budget.spend((entries * N) as u64)?;
    Ok(Allocation::new(pool, capacities, allowed))
}

fn place<const N: usize>(
    tiles: &[Domino],
    remaining: [usize; N],
    allowed: &[DominoSet; N],
    placed: &mut [DominoSet; N],
    budget: &mut Budget,
    visit: &mut impl FnMut(&[DominoSet; N], [usize; N], &mut Budget) -> Result<()>,
) -> Result<()> {
    budget.spend(1)?;
    let Some((tile, rest)) = tiles.split_first() else {
        return visit(placed, remaining, budget);
    };
    for s in 0..N {
        if remaining[s] > 0 && allowed[s].contains(*tile) {
            let mut next = remaining;
            next[s] -= 1;
            placed[s].insert(*tile);
            place(rest, next, allowed, placed, budget, visit)?;
            placed[s].remove(*tile);
        }
    }
    Ok(())
}

/// An unbiased integer in `0..count`, given independent uniform u64 words.
/// Rejection, rather than `% count` alone, removes modulo bias. The budget
/// also bounds a broken random source that keeps producing rejected words.
fn rank(count: u128, next: &mut impl FnMut() -> u64, budget: &mut Budget) -> Result<u128> {
    if count == 0 {
        return Err(error("cannot sample an exactly empty construction"));
    }
    let reject_below = count.wrapping_neg() % count;
    loop {
        budget.spend(1)?;
        let word = (u128::from(next()) << 64) | u128::from(next());
        if word >= reject_below {
            return Ok(word % count);
        }
    }
}

/// The uniform legal worlds of one fixed frame in which a query has an
/// answer. Preparation resolves every ownership read the evaluator requests,
/// then retains each accepted cell's completion table for subsequent draws.
/// A zero count proves impossibility; budget/cell exhaustion returns an error.
pub struct WorldSpace {
    frame: Frame,
    query: CompiledQuery,
    cells: Cells<HIDDEN_SEATS>,
}

impl WorldSpace {
    pub fn new(frame: Frame, query: &CompiledQuery, max_cells: usize, budget: &mut Budget) -> Result<Self> {
        let kernel = frame.kernel();
        let cells = Cells::prepare(
            kernel.pool(),
            kernel.hidden().map(|h| h.capacity),
            std::array::from_fn(|s| kernel.allowed(frame.physics(), s)),
            max_cells,
            budget,
            |hands, unresolved, needed, budget| {
                let world = kernel.world(hands);
                query
                    .evaluate_in(Holdings::new(&frame, &world, unresolved).probing(needed), budget)
                    .map(|answers| !answers.is_empty())
            },
        )?;
        Ok(Self { frame, query: query.clone(), cells })
    }

    pub fn frame(&self) -> &Frame {
        &self.frame
    }
    pub fn query(&self) -> &CompiledQuery {
        &self.query
    }
    pub fn count(&self) -> u128 {
        self.cells.count
    }
    pub fn base_count(&self) -> u128 {
        self.cells.base_count
    }
    pub fn cell_count(&self) -> usize {
        self.cells.cells.len()
    }
    pub fn resolved(&self) -> DominoSet {
        self.cells.resolved
    }
    pub fn probability(&self) -> Estimate {
        self.cells.probability().expect("a frame has a nonempty legal fiber")
    }

    /// Every accepted world exactly once. Membership and the query are
    /// checked on the completed world before it is returned.
    pub fn unrank(&self, rank: u128, budget: &mut Budget) -> Result<World> {
        let world = self.frame.kernel().world(self.cells.unrank(rank)?);
        if self.query.evaluate(&self.frame, &world, budget)?.is_empty() {
            return Err(error("constructed world does not satisfy its query"));
        }
        Ok(world)
    }

    /// Uniform over complete accepted worlds, with replacement. The caller
    /// supplies independent uniform u64 words, and records its seed/provenance.
    pub fn sample(&self, next: &mut impl FnMut() -> u64, budget: &mut Budget) -> Result<World> {
        self.unrank(rank(self.count(), next, budget)?, budget)
    }
}

/// An opening family whose viewer hand varies as well. The auction and
/// declaration are fixed, legal records; no play history is invented.
///
/// This first opening fragment accepts standard predicates of tile ownership,
/// tile/seat facts, and the fixed public opening. Use `(holds S0 d)` rather
/// than `(mine d)`: reads of *every* hand then go through checked Holdings.
/// Hand-sensitive viewer predicates and custom predicates are refused before
/// counting. Quantifiers, aggregates, equality and projection keep the normal
/// evaluator's semantics. Full-deal uniformity is not hand-uniformity.
pub struct OpeningSpace {
    auction: Auction,
    physics: Physics,
    viewer: Seat,
    query: CompiledQuery,
    cells: Cells<{ Seat::COUNT }>,
}

impl OpeningSpace {
    pub fn new(
        auction: Auction,
        physics: Physics,
        viewer: Seat,
        query: &Query,
        max_cells: usize,
        budget: &mut Budget,
    ) -> Result<Self> {
        Self::restricted(auction, physics, viewer, query, [DominoSet::FULL; 4], max_cells, budget)
    }

    /// As `new`, with per-seat allowed tiles. Missing coverage or conflicting
    /// ownership constraints can give a proved empty base (probability None).
    pub fn restricted(
        auction: Auction,
        physics: Physics,
        viewer: Seat,
        query: &Query,
        allowed: [DominoSet; 4],
        max_cells: usize,
        budget: &mut Budget,
    ) -> Result<Self> {
        auction.contract(physics.clone())?;
        let query = query.compile(&Registry::standard())?;
        for spec in query.predicate_specs() {
            if !matches!(
                spec.name.as_str(),
                "holds"
                    | "void"
                    | "legal"
                    | "forced"
                    | "live"
                    | "in"
                    | "double"
                    | "beats"
                    | "boss"
                    | "master"
                    | "quota"
                    | "tile"
                    | "chair"
                    | "context"
                    | "team"
                    | "partner"
                    | "opponent"
                    | "successor"
                    | "viewer"
                    | "leader"
                    | "next-actor"
                    | "led-context"
                    | "current-winner"
                    | "played"
                    | "count"
                    | "trick-play"
                    | "leads-context"
                    | "declarer"
                    | "bid"
                    | "stake"
                    | "banked"
                    | "to-make"
                    | "to-set"
                    | "trick-number"
                    | "led"
                    | "played-to"
                    | "trick-of"
                    | "shown-void"
                    | "trump"
                    | "bears"
                    | "high-pip"
                    | "ranks-above"
                    | "shaker"
                    | "passed"
                    | "no-bid"
                    | "high-bidder"
                    | "bid-of"
                    | "high-bid"
            ) {
                return Err(error(format!(
                    "opening construction does not admit {}; express ownership with holds, or use a fixed frame",
                    spec.name
                )));
            }
        }
        let cells = Cells::prepare(
            DominoSet::FULL,
            [7; 4],
            allowed,
            max_cells,
            budget,
            |hands, unresolved, needed, budget| {
                let (frame, world) = opening(&auction, &physics, viewer, hands)?;
                query
                    .evaluate_in(Holdings::opening(&frame, &world, unresolved).probing(needed), budget)
                    .map(|answers| !answers.is_empty())
            },
        )?;
        Ok(Self { auction, physics, viewer, query, cells })
    }

    pub fn count(&self) -> u128 {
        self.cells.count
    }
    pub fn base_count(&self) -> u128 {
        self.cells.base_count
    }
    pub fn cell_count(&self) -> usize {
        self.cells.cells.len()
    }
    pub fn resolved(&self) -> DominoSet {
        self.cells.resolved
    }
    pub fn query(&self) -> &CompiledQuery {
        &self.query
    }
    pub fn probability(&self) -> Option<Estimate> {
        self.cells.probability()
    }

    /// Return the constructed viewer frame alongside the full opening deal.
    /// Replaying that frame and evaluating the query checks each result.
    pub fn unrank(&self, rank: u128, budget: &mut Budget) -> Result<(Frame, World)> {
        let (frame, world) = opening(&self.auction, &self.physics, self.viewer, self.cells.unrank(rank)?)?;
        if self.query.evaluate(&frame, &world, budget)?.is_empty() {
            return Err(error("constructed opening does not satisfy its query"));
        }
        Ok((frame, world))
    }

    pub fn sample(&self, next: &mut impl FnMut() -> u64, budget: &mut Budget) -> Result<(Frame, World)> {
        self.unrank(rank(self.count(), next, budget)?, budget)
    }
}

fn opening(auction: &Auction, physics: &Physics, viewer: Seat, hands: [DominoSet; 4]) -> Result<(Frame, World)> {
    let frame = Frame::new(auction.clone(), physics.clone(), viewer, hands[viewer.index()], vec![])?;
    let world = frame.kernel().world(frame.kernel().hidden().map(|h| hands[h.seat.index()]));
    if !frame.contains(&world) {
        return Err(error("constructed opening is outside its frame's support"));
    }
    Ok((frame, world))
}
