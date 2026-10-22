// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The decision quotient of a solved course (Jason's note *A minimal comparison
//! algebra for a declared 42 family*, 2026-10-21; Lean: `scenario/lean/Stab/Residual.lean`).
//!
//! A course is a finite acyclic graph of a viewer's decisions. Each decision has its
//! legal menu, its complete maximizing set, and its blocks: an own play, what was seen
//! until the next own decision, and that decision. A block the course lacks is REJECT,
//! and a decision with no blocks is the final tile, whose play is HALT. Two decisions
//! are one state when every block sequence gives equal outputs: the coarsest right
//! congruence preserving the interface. It is computed twice, bottom-up by
//! signatures and by Moore refinement from the output partition, and the two must
//! agree. The refinement rounds are the block-depth ladder.
//!
//! An [`Interface`] says what an output and a block label may mention: tiles as seen
//! ([`Interface::Raw`]), or no tile identity at all ([`Interface::Shape`],
//! [`Interface::Fork`]).

use std::collections::HashMap;
use std::hash::Hash;

use crate::{Result, error};

/// One block from a decision: the own play, what was seen, the next decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block<A, X> {
    pub play: A,
    pub seen: X,
    pub next: usize,
}

/// One decision: the legal menu (strictly ascending), the complete maximizing set
/// (ascending, nonempty, within the menu) and its blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision<A, X> {
    pub menu: Vec<A>,
    pub best: Vec<A>,
    pub blocks: Vec<Block<A, X>>,
}

/// What an output and a block label may mention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Interface {
    /// Menu and maximizing set as tiles; blocks labelled by play and observation.
    Raw,
    /// No tile identity: menu size, maximizing slots (rank in the menu), and under
    /// each slot the multiset of successor classes.
    Shape,
    /// As shape, but under each slot the set of successor classes: which next
    /// decisions the evidence can lead to, not how many observations lead there.
    Fork,
}

impl Interface {
    pub const ALL: [Interface; 3] = [Interface::Raw, Interface::Shape, Interface::Fork];

    pub fn name(self) -> &'static str {
        match self {
            Interface::Raw => "raw",
            Interface::Shape => "shape",
            Interface::Fork => "fork",
        }
    }
}

/// A validated course, with an order that puts every decision after its successors.
#[derive(Clone, Debug)]
pub struct Course<A, X> {
    decisions: Vec<Decision<A, X>>,
    order: Vec<usize>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Out<A> {
    Raw(Vec<A>, Vec<A>),
    Shape(usize, Vec<usize>),
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Next<A, X> {
    Raw(Vec<(A, X, usize)>),
    Slots(Vec<Vec<usize>>),
}

/// The quotient: each decision's class (dense, from zero) and the ladder, the number
/// of classes after each refinement round until a round repeats.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Quotient {
    pub class: Vec<usize>,
    pub states: usize,
    pub rounds: Vec<Vec<usize>>,
}

impl Quotient {
    pub fn ladder(&self) -> Vec<usize> {
        self.rounds.iter().map(|r| count(r)).collect()
    }

    /// The ladder restricted to some decisions (the roots, say).
    pub fn ladder_of(&self, which: &[usize]) -> Vec<usize> {
        self.rounds.iter().map(|r| distinct(which.iter().map(|&i| r[i]))).collect()
    }
}

fn count(class: &[usize]) -> usize {
    distinct(class.iter().copied())
}

fn distinct(it: impl Iterator<Item = usize>) -> usize {
    let mut v: Vec<usize> = it.collect();
    v.sort_unstable();
    v.dedup();
    v.len()
}

/// Dense class ids by first appearance.
fn intern<K: Hash + Eq>(keys: impl Iterator<Item = K>) -> Vec<usize> {
    let mut ids = HashMap::new();
    keys.map(|k| {
        let n = ids.len();
        *ids.entry(k).or_insert(n)
    })
    .collect()
}

/// Whether two class vectors are one partition.
fn same_partition(a: &[usize], b: &[usize]) -> bool {
    let (mut ab, mut ba) = (HashMap::new(), HashMap::new());
    a.iter().zip(b).all(|(&x, &y)| *ab.entry(x).or_insert(y) == y && *ba.entry(y).or_insert(x) == x)
}

impl<A: Clone + Ord + Hash, X: Clone + Ord + Hash> Course<A, X> {
    /// Refused when a menu is unsorted or repeats, the maximizing set is empty or
    /// leaves the menu, a block plays outside the menu, repeats a label or points
    /// nowhere, or the graph has a cycle.
    pub fn new(decisions: Vec<Decision<A, X>>) -> Result<Course<A, X>> {
        for (i, d) in decisions.iter().enumerate() {
            if d.menu.is_empty() || !d.menu.windows(2).all(|w| w[0] < w[1]) {
                return Err(error(format!("decision {i}: the menu is empty or not strictly ascending")));
            }
            if d.best.is_empty() || !d.best.windows(2).all(|w| w[0] < w[1]) {
                return Err(error(format!("decision {i}: the maximizing set is empty or not ascending")));
            }
            if !d.best.iter().all(|a| d.menu.binary_search(a).is_ok()) {
                return Err(error(format!("decision {i}: the maximizing set leaves the menu")));
            }
            let mut labels: Vec<(&A, &X)> = Vec::with_capacity(d.blocks.len());
            for b in &d.blocks {
                if d.menu.binary_search(&b.play).is_err() {
                    return Err(error(format!("decision {i}: a block plays outside the menu")));
                }
                if b.next >= decisions.len() {
                    return Err(error(format!("decision {i}: a block points nowhere")));
                }
                labels.push((&b.play, &b.seen));
            }
            labels.sort();
            if labels.windows(2).any(|w| w[0] == w[1]) {
                return Err(error(format!("decision {i}: two blocks share a label")));
            }
        }
        // Successors first: depth-first post-order, iteratively.
        let n = decisions.len();
        let (mut state, mut order) = (vec![0u8; n], Vec::with_capacity(n));
        for start in 0..n {
            if state[start] != 0 {
                continue;
            }
            let mut stack = vec![(start, 0usize)];
            state[start] = 1;
            while let Some(top) = stack.last_mut() {
                let v = top.0;
                if let Some(b) = decisions[v].blocks.get(top.1) {
                    top.1 += 1;
                    let w = b.next;
                    match state[w] {
                        0 => {
                            state[w] = 1;
                            stack.push((w, 0));
                        }
                        1 => return Err(error(format!("decision {v}: the course has a cycle"))),
                        _ => {}
                    }
                } else {
                    state[v] = 2;
                    order.push(v);
                    stack.pop();
                }
            }
        }
        Ok(Course { decisions, order })
    }

    pub fn decisions(&self) -> &[Decision<A, X>] {
        &self.decisions
    }

    /// Every decision, each after all of its successors.
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    fn out(&self, i: usize, interface: Interface) -> Out<A> {
        let d = &self.decisions[i];
        match interface {
            Interface::Raw => Out::Raw(d.menu.clone(), d.best.clone()),
            Interface::Shape | Interface::Fork => {
                Out::Shape(d.menu.len(), d.best.iter().map(|a| slot(&d.menu, a)).collect())
            }
        }
    }

    fn next(&self, i: usize, interface: Interface, class: &[usize]) -> Next<A, X> {
        let d = &self.decisions[i];
        match interface {
            Interface::Raw => {
                let mut v: Vec<_> = d.blocks.iter().map(|b| (b.play.clone(), b.seen.clone(), class[b.next])).collect();
                v.sort();
                Next::Raw(v)
            }
            Interface::Shape | Interface::Fork => {
                let mut per = vec![Vec::new(); d.menu.len()];
                for b in &d.blocks {
                    per[slot(&d.menu, &b.play)].push(class[b.next]);
                }
                for p in &mut per {
                    p.sort_unstable();
                    if interface == Interface::Fork {
                        p.dedup();
                    }
                }
                Next::Slots(per)
            }
        }
    }

    /// Bottom-up: a decision's signature is its output and its labelled successors'
    /// signatures, interned as the course is walked successors first.
    pub fn signatures(&self, interface: Interface) -> Vec<usize> {
        let mut class = vec![usize::MAX; self.decisions.len()];
        let mut ids = HashMap::new();
        for &i in &self.order {
            let key = (self.out(i, interface), self.next(i, interface, &class));
            let n = ids.len();
            class[i] = *ids.entry(key).or_insert(n);
        }
        class
    }

    /// Moore refinement from the output partition: every round, until one repeats.
    /// Round `k` separates exactly what block sequences of length at most `k` reveal.
    pub fn refine(&self, interface: Interface) -> Vec<Vec<usize>> {
        let n = self.decisions.len();
        let mut rounds = vec![intern((0..n).map(|i| self.out(i, interface)))];
        loop {
            let last = rounds.last().expect("round zero");
            let new = intern((0..n).map(|i| (last[i], self.next(i, interface, last))));
            let done = count(&new) == count(last);
            rounds.push(new);
            if done {
                return rounds;
            }
        }
    }

    /// The quotient under an interface, built both ways; refused if they disagree.
    pub fn quotient(&self, interface: Interface) -> Result<Quotient> {
        let class = self.signatures(interface);
        let rounds = self.refine(interface);
        if !same_partition(&class, rounds.last().expect("a round")) {
            return Err(error(format!("{}: signature and refinement partitions differ", interface.name())));
        }
        Ok(Quotient { states: count(&class), class, rounds })
    }
}

fn slot<A: Ord>(menu: &[A], a: &A) -> usize {
    menu.binary_search(a).expect("a validated play is in the menu")
}

/// A course read from JSON (schema `silas-decision-course-v1`, written by
/// `lem/quotient.py --export`), with its tags and values, and the report
/// `silas quotient` prints.
#[cfg(feature = "course")]
pub mod file {
    use std::collections::{BTreeMap, BTreeSet};

    use num_rational::BigRational;
    use num_traits::Zero;
    use serde_json::{Value, json};

    use super::{Block, Course, Decision, Interface};
    use crate::{Result, error};

    /// A block label in the file: the observed plays, each `(seat, tile)` as tokens.
    pub type Seen = Vec<(u32, u32)>;

    /// A decision's tags and values, beside its place in the course.
    #[derive(Clone, Debug)]
    pub struct Tags {
        pub group: String,
        pub split: String,
        pub stage: String,
        pub ply: u32,
        pub q: Vec<BigRational>,
        pub mass: BigRational,
        pub settled: bool,
    }

    pub struct CourseFile {
        pub course: Course<u32, Seen>,
        pub tags: Vec<Tags>,
    }

    fn field<'a>(v: &'a Value, k: &str, i: usize) -> Result<&'a Value> {
        v.get(k).ok_or_else(|| error(format!("decision {i}: no {k}")))
    }

    fn uint(v: &Value, what: &str) -> Result<u32> {
        v.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| error(format!("{what}: not a small integer")))
    }

    fn uints(v: &Value, what: &str) -> Result<Vec<u32>> {
        v.as_array().ok_or_else(|| error(format!("{what}: not a list")))?.iter().map(|x| uint(x, what)).collect()
    }

    fn text(v: &Value, what: &str) -> Result<String> {
        v.as_str().map(str::to_string).ok_or_else(|| error(format!("{what}: not a string")))
    }

    fn rational(v: &Value, what: &str) -> Result<BigRational> {
        text(v, what)?.parse().map_err(|_| error(format!("{what}: not a rational")))
    }

    impl CourseFile {
        /// Read and audit: the maximizing set is the argmax of q; the blocks under each
        /// play partition the mass; Bellman, `q[a] * mass = Σ child.mass * max child.q`;
        /// a decision without blocks has a single tile.
        pub fn from_json(text_in: &str) -> Result<CourseFile> {
            let root: Value = serde_json::from_str(text_in).map_err(|e| error(format!("course: {e}")))?;
            if root.get("schema").and_then(Value::as_str) != Some("silas-decision-course-v1") {
                return Err(error("course: schema is not silas-decision-course-v1"));
            }
            let rows = root.get("nodes").and_then(Value::as_array).ok_or_else(|| error("course: no nodes"))?;
            let (mut decisions, mut tags) = (Vec::with_capacity(rows.len()), Vec::with_capacity(rows.len()));
            for (i, r) in rows.iter().enumerate() {
                let menu = uints(field(r, "menu", i)?, "menu")?;
                let best = uints(field(r, "best", i)?, "best")?;
                let q = field(r, "q", i)?
                    .as_array()
                    .ok_or_else(|| error(format!("decision {i}: q is not a list")))?
                    .iter()
                    .map(|x| rational(x, "q"))
                    .collect::<Result<Vec<_>>>()?;
                let mut blocks = Vec::new();
                for b in field(r, "children", i)?.as_array().ok_or_else(|| error("children: not a list"))? {
                    let seen = b
                        .get("seen")
                        .and_then(Value::as_array)
                        .ok_or_else(|| error(format!("decision {i}: a block has no seen list")))?
                        .iter()
                        .map(|p| {
                            let p = uints(p, "seen")?;
                            match p[..] {
                                [s, t] => Ok((s, t)),
                                _ => Err(error("seen: not a pair")),
                            }
                        })
                        .collect::<Result<Seen>>()?;
                    let play = uint(b.get("play").unwrap_or(&Value::Null), "play")?;
                    let next = uint(b.get("next").unwrap_or(&Value::Null), "next")? as usize;
                    blocks.push(Block { play, seen, next });
                }
                tags.push(Tags {
                    group: text(field(r, "group", i)?, "group")?,
                    split: text(field(r, "split", i)?, "split")?,
                    stage: text(field(r, "stage", i)?, "stage")?,
                    ply: uint(field(r, "ply", i)?, "ply")?,
                    mass: rational(field(r, "mass", i)?, "mass")?,
                    settled: field(r, "settled", i)?.as_bool().ok_or_else(|| error("settled: not a bool"))?,
                    q,
                });
                decisions.push(Decision { menu, best, blocks });
            }
            let course = Course::new(decisions)?;
            audit(&course, &tags)?;
            Ok(CourseFile { course, tags })
        }

        /// Every count of the decision-quotient report, under every interface.
        pub fn report(&self) -> Result<Value> {
            let ds = self.course.decisions();
            let t = &self.tags;
            let n = ds.len();
            let groups: BTreeSet<&str> = t.iter().map(|x| x.group.as_str()).collect();
            let mut split_of = BTreeMap::new();
            for x in t {
                split_of.insert(x.group.as_str(), x.split.as_str());
            }
            let stages: BTreeSet<&str> = t.iter().map(|x| x.stage.as_str()).collect();
            let disc: Vec<usize> = (0..n).filter(|&i| ds[i].menu.len() > 1 && !t[i].settled).collect();
            let nonflat = |i: &usize| ds[*i].best.len() < ds[*i].menu.len();
            let tally = |it: &mut dyn Iterator<Item = usize>| {
                let v: Vec<usize> = it.collect();
                json!({"nodes": v.len(), "nonflat": v.iter().filter(|i| nonflat(i)).count()})
            };
            let splits: BTreeSet<&str> = t.iter().map(|x| x.split.as_str()).collect();
            let mut discretionary = json!({
                "nodes": disc.len(),
                "nonflat": disc.iter().filter(|i| nonflat(i)).count(),
            });
            discretionary["by_split"] = Value::Object(
                splits
                    .iter()
                    .map(|s| (s.to_string(), tally(&mut disc.iter().copied().filter(|&i| t[i].split == *s))))
                    .collect(),
            );
            discretionary["by_stage"] = Value::Object(
                stages
                    .iter()
                    .map(|s| (s.to_string(), tally(&mut disc.iter().copied().filter(|&i| t[i].stage == *s))))
                    .collect(),
            );
            let mut classes = serde_json::Map::new();
            let sizes: BTreeSet<usize> = disc.iter().map(|&i| ds[i].menu.len()).collect();
            for size in sizes {
                let group: Vec<usize> = disc.iter().copied().filter(|&i| ds[i].menu.len() == size).collect();
                let mut orders: BTreeMap<String, usize> = BTreeMap::new();
                let mut types: BTreeMap<String, usize> = BTreeMap::new();
                for &i in &group {
                    let w = weak_order(&t[i].q);
                    *orders.entry(w.iter().map(usize::to_string).collect()).or_default() += 1;
                    let mut tiers: BTreeMap<usize, usize> = BTreeMap::new();
                    for r in &w {
                        *tiers.entry(*r).or_default() += 1;
                    }
                    let mut sizes: Vec<usize> = tiers.into_values().collect();
                    sizes.sort_unstable_by(|a, b| b.cmp(a));
                    let key = sizes.iter().map(usize::to_string).collect::<Vec<_>>().join("/");
                    *types.entry(key).or_default() += 1;
                }
                classes.insert(
                    size.to_string(),
                    json!({"nodes": group.len(), "distinct_weak_orders": orders.len(),
                           "weak_orders": orders, "tier_types": types}),
                );
            }
            let mut interfaces = serde_json::Map::new();
            for interface in Interface::ALL {
                let quot = self.course.quotient(interface)?;
                let c = &quot.class;
                let of_stage = |s: &str| (0..n).filter(|&i| t[i].stage == s).collect::<Vec<_>>();
                let mut per: BTreeMap<&str, (BTreeSet<usize>, usize)> = BTreeMap::new();
                for i in 0..n {
                    let e = per.entry(t[i].group.as_str()).or_default();
                    e.0.insert(c[i]);
                    e.1 += 1;
                }
                let states: Vec<usize> = per.values().map(|(s, _)| s.len()).collect();
                let mut ratios: Vec<BigRational> =
                    per.values().map(|(s, k)| BigRational::new((*k).into(), s.len().into())).collect();
                ratios.sort();
                let mut groups_of: BTreeMap<usize, BTreeSet<&str>> = BTreeMap::new();
                for i in 0..n {
                    groups_of.entry(c[i]).or_default().insert(t[i].group.as_str());
                }
                let mut coverage = serde_json::Map::new();
                for from in &splits {
                    let held: BTreeSet<usize> = (0..n).filter(|&i| t[i].split == *from).map(|i| c[i]).collect();
                    let mut into = serde_json::Map::new();
                    for to in splits.iter().filter(|s| *s != from) {
                        let mut by = serde_json::Map::new();
                        for st in &stages {
                            let keys: Vec<usize> = (0..n).filter(|&i| t[i].split == *to && t[i].stage == *st).collect();
                            let dk: Vec<usize> =
                                keys.iter().copied().filter(|&i| ds[i].menu.len() > 1 && !t[i].settled).collect();
                            let seen = |v: &[usize]| v.iter().filter(|&&i| held.contains(&c[i])).count();
                            by.insert(
                                st.to_string(),
                                json!({"all": {"seen": seen(&keys), "of": keys.len()},
                                       "discretionary_unsettled": {"seen": seen(&dk), "of": dk.len()}}),
                            );
                        }
                        into.insert(to.to_string(), Value::Object(by));
                    }
                    coverage.insert(from.to_string(), Value::Object(into));
                }
                interfaces.insert(
                    interface.name().to_string(),
                    json!({
                        "states": quot.states,
                        "halt_and_reject": 2,
                        "ladder_by_block_depth": quot.ladder(),
                        "ladder_by_stage": stages.iter().map(|s| (s.to_string(), json!(quot.ladder_of(&of_stage(s)))))
                            .collect::<serde_json::Map<_, _>>(),
                        "states_by_stage": stages.iter()
                            .map(|s| (s.to_string(), json!(of_stage(s).iter().map(|&i| c[i]).collect::<BTreeSet<_>>().len())))
                            .collect::<serde_json::Map<_, _>>(),
                        "per_root_states": {"min": states.iter().min(), "max": states.iter().max(),
                                            "sum": states.iter().sum::<usize>()},
                        "per_root_compression_nodes_per_state": {
                            "min": ratios.first().map(ToString::to_string),
                            "median": ratios.get(ratios.len() / 2).map(ToString::to_string),
                            "max": ratios.last().map(ToString::to_string)},
                        "classes_in_more_than_one_root": groups_of.values().filter(|g| g.len() > 1).count(),
                        "coverage_by_split_classes": coverage,
                    }),
                );
            }
            let mut by_ply: BTreeMap<String, usize> = BTreeMap::new();
            let mut by_stage: BTreeMap<String, usize> = BTreeMap::new();
            let mut menus: BTreeMap<String, usize> = BTreeMap::new();
            let mut split_counts: BTreeMap<String, usize> = BTreeMap::new();
            for i in 0..n {
                *by_ply.entry(t[i].ply.to_string()).or_default() += 1;
                *by_stage.entry(t[i].stage.clone()).or_default() += 1;
                *menus.entry(ds[i].menu.len().to_string()).or_default() += 1;
            }
            for s in split_of.values() {
                *split_counts.entry(s.to_string()).or_default() += 1;
            }
            Ok(json!({
                "roots": groups.len(),
                "splits": split_counts,
                "nodes": n,
                "nodes_by_ply": by_ply,
                "nodes_by_stage": by_stage,
                "menu_sizes": menus,
                "discretionary_unsettled": discretionary,
                "decision_classes": classes,
                "interfaces": interfaces,
            }))
        }
    }

    /// Dense rank of each menu slot by q, best first.
    fn weak_order(q: &[BigRational]) -> Vec<usize> {
        let mut levels: Vec<&BigRational> = q.iter().collect();
        levels.sort_by(|a, b| b.cmp(a));
        levels.dedup();
        q.iter().map(|v| levels.iter().position(|l| *l == v).expect("a level")).collect()
    }

    fn audit(course: &Course<u32, Seen>, tags: &[Tags]) -> Result<()> {
        for (i, (d, t)) in course.decisions().iter().zip(tags).enumerate() {
            if t.q.len() != d.menu.len() {
                return Err(error(format!("decision {i}: q is not aligned with the menu")));
            }
            let top = t.q.iter().max().expect("a nonempty menu");
            let argmax: Vec<u32> = d.menu.iter().zip(&t.q).filter(|(_, v)| *v == top).map(|(a, _)| *a).collect();
            if argmax != d.best {
                return Err(error(format!("decision {i}: the maximizing set is not the argmax of q")));
            }
            if d.blocks.is_empty() {
                if d.menu.len() != 1 {
                    return Err(error(format!("decision {i}: no blocks from a real menu")));
                }
                continue;
            }
            for (slot, a) in d.menu.iter().enumerate() {
                let under: Vec<usize> = d.blocks.iter().filter(|b| b.play == *a).map(|b| b.next).collect();
                if under.is_empty() {
                    return Err(error(format!("decision {i}: play {a} is explored nowhere")));
                }
                let mass: BigRational = under.iter().map(|&k| tags[k].mass.clone()).sum();
                if mass != t.mass {
                    return Err(error(format!("decision {i}: the blocks under {a} do not partition the mass")));
                }
                let bellman: BigRational = under
                    .iter()
                    .map(|&k| &tags[k].mass * tags[k].q.iter().max().cloned().unwrap_or_else(BigRational::zero))
                    .sum();
                if bellman != &t.q[slot] * &t.mass {
                    return Err(error(format!("decision {i}: Bellman fails under play {a}")));
                }
            }
        }
        Ok(())
    }
}
