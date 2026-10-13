//! The game's nouns: pips, the 28 dominoes, seats and teams, the eight led
//! contexts, and bitsets over dominoes and contexts. Nothing here knows a
//! declaration; every rule lives behind [`crate::Physics`].

use std::fmt;
use std::str::FromStr;

/// A pip value in `0..=6`. The inner value is private, so a `Pip` cannot name
/// a value outside the double-six set.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Pip(u8);

impl Pip {
    pub const COUNT: usize = 7;
    pub const ALL: [Pip; Self::COUNT] = [Pip(0), Pip(1), Pip(2), Pip(3), Pip(4), Pip(5), Pip(6)];

    pub const fn new(value: u8) -> Option<Pip> {
        if value < Self::COUNT as u8 { Some(Pip(value)) } else { None }
    }

    pub const fn value(self) -> u8 {
        self.0
    }
}

impl fmt::Display for Pip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A domino `h-l` with `h >= l`. The order of the two ends is a naming
/// convention, not an orientation.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Domino {
    hi: Pip,
    lo: Pip,
}

const fn triangular(hi: u8, lo: u8) -> usize {
    (hi * (hi + 1) / 2 + lo) as usize
}

impl Domino {
    pub const COUNT: usize = 28;

    /// Every domino, in index order: `h-l` has index `h(h+1)/2 + l`, so 0-0 is
    /// 0 and 6-6 is 27.
    pub const ALL: [Domino; Self::COUNT] = {
        let mut out = [Domino { hi: Pip(0), lo: Pip(0) }; Self::COUNT];
        let mut hi = 0;
        while hi < Pip::COUNT as u8 {
            let mut lo = 0;
            while lo <= hi {
                out[triangular(hi, lo)] = Domino { hi: Pip(hi), lo: Pip(lo) };
                lo += 1;
            }
            hi += 1;
        }
        out
    };

    pub const fn new(a: Pip, b: Pip) -> Domino {
        if a.0 >= b.0 { Domino { hi: a, lo: b } } else { Domino { hi: b, lo: a } }
    }

    pub const fn hi(self) -> Pip {
        self.hi
    }

    pub const fn lo(self) -> Pip {
        self.lo
    }

    pub const fn is_double(self) -> bool {
        self.hi.0 == self.lo.0
    }

    pub const fn has(self, pip: Pip) -> bool {
        self.hi.0 == pip.0 || self.lo.0 == pip.0
    }

    pub const fn pip_sum(self) -> u8 {
        self.hi.0 + self.lo.0
    }

    pub const fn index(self) -> usize {
        triangular(self.hi.0, self.lo.0)
    }

    pub const fn from_index(i: usize) -> Option<Domino> {
        if i < Self::COUNT { Some(Self::ALL[i]) } else { None }
    }
}

impl fmt::Display for Domino {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.hi, self.lo)
    }
}

impl fmt::Debug for Domino {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl FromStr for Domino {
    type Err = crate::Error;

    /// `a-b` with both ends in `0..=6`, in either order.
    fn from_str(s: &str) -> crate::Result<Domino> {
        let pip = |x: &str| x.parse::<u8>().ok().and_then(Pip::new);
        let (a, b) = s.split_once('-').ok_or_else(|| crate::error(format!("not a domino: {s}")))?;
        match (pip(a), pip(b)) {
            (Some(a), Some(b)) => Ok(Domino::new(a, b)),
            _ => Err(crate::error(format!("not a domino: {s}"))),
        }
    }
}

/// The four seats, clockwise. Seat `s` and seat `s+2` are partners.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Seat {
    S0,
    S1,
    S2,
    S3,
}

/// The two partnerships: `T0 = {S0, S2}`, `T1 = {S1, S3}`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Team {
    T0,
    T1,
}

impl Seat {
    pub const COUNT: usize = 4;
    pub const ALL: [Seat; Self::COUNT] = [Seat::S0, Seat::S1, Seat::S2, Seat::S3];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn from_index(i: usize) -> Option<Seat> {
        if i < Self::COUNT { Some(Self::ALL[i]) } else { None }
    }

    /// The `n`-th seat clockwise from this one.
    pub const fn plus(self, n: usize) -> Seat {
        Self::ALL[(self.index() + n) % Self::COUNT]
    }

    pub const fn successor(self) -> Seat {
        self.plus(1)
    }

    pub const fn partner(self) -> Seat {
        self.plus(2)
    }

    pub const fn team(self) -> Team {
        match self {
            Seat::S0 | Seat::S2 => Team::T0,
            Seat::S1 | Seat::S3 => Team::T1,
        }
    }
}

impl Team {
    pub const fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for Seat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S{}", self.index())
    }
}

impl fmt::Display for Team {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T{}", self.index())
    }
}

/// A led context: one of the seven pip suits, or the called suit. `Called` is
/// a distinct inhabitant, never a pip.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Context {
    Natural(Pip),
    Called,
}

impl Context {
    pub const COUNT: usize = 8;
    pub const ALL: [Context; Self::COUNT] = [
        Context::Natural(Pip(0)),
        Context::Natural(Pip(1)),
        Context::Natural(Pip(2)),
        Context::Natural(Pip(3)),
        Context::Natural(Pip(4)),
        Context::Natural(Pip(5)),
        Context::Natural(Pip(6)),
        Context::Called,
    ];

    /// Pip suits are `0..=6`; the called suit is 7.
    pub const fn index(self) -> usize {
        match self {
            Context::Natural(p) => p.0 as usize,
            Context::Called => 7,
        }
    }

    pub const fn from_index(i: usize) -> Option<Context> {
        if i < Self::COUNT { Some(Self::ALL[i]) } else { None }
    }
}

impl fmt::Display for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Context::Natural(p) => write!(f, "q{p}"),
            Context::Called => f.write_str("q*"),
        }
    }
}

impl FromStr for Context {
    type Err = crate::Error;

    /// `q0`..`q6`, or `q*` for the called suit.
    fn from_str(s: &str) -> crate::Result<Context> {
        let parsed = match s.strip_prefix('q') {
            Some("*") => Some(Context::Called),
            Some(n) => n.parse().ok().and_then(Context::from_index),
            None => None,
        };
        parsed.ok_or_else(|| crate::error(format!("not a context: {s}")))
    }
}

/// A set of dominoes: bit `i` is the domino with index `i`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DominoSet(u32);

impl DominoSet {
    pub const EMPTY: DominoSet = DominoSet(0);
    pub const FULL: DominoSet = DominoSet((1 << Domino::COUNT) - 1);

    pub const fn from_bits(bits: u32) -> Option<DominoSet> {
        if bits & !Self::FULL.0 == 0 { Some(DominoSet(bits)) } else { None }
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn single(d: Domino) -> DominoSet {
        DominoSet(1 << d.index())
    }

    pub const fn contains(self, d: Domino) -> bool {
        self.0 & (1 << d.index()) != 0
    }

    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn union(self, other: DominoSet) -> DominoSet {
        DominoSet(self.0 | other.0)
    }

    pub const fn intersection(self, other: DominoSet) -> DominoSet {
        DominoSet(self.0 & other.0)
    }

    pub const fn difference(self, other: DominoSet) -> DominoSet {
        DominoSet(self.0 & !other.0)
    }

    pub const fn is_subset_of(self, other: DominoSet) -> bool {
        self.0 & !other.0 == 0
    }

    pub const fn is_disjoint(self, other: DominoSet) -> bool {
        self.0 & other.0 == 0
    }

    /// Returns whether `d` was absent.
    pub fn insert(&mut self, d: Domino) -> bool {
        let absent = !self.contains(d);
        self.0 |= 1 << d.index();
        absent
    }

    /// Returns whether `d` was present.
    pub fn remove(&mut self, d: Domino) -> bool {
        let present = self.contains(d);
        self.0 &= !(1 << d.index());
        present
    }

    /// Members in index order.
    pub fn iter(self) -> impl Iterator<Item = Domino> {
        Domino::ALL.into_iter().filter(move |d| self.contains(*d))
    }
}

impl FromIterator<Domino> for DominoSet {
    fn from_iter<I: IntoIterator<Item = Domino>>(iter: I) -> DominoSet {
        DominoSet(iter.into_iter().fold(0, |bits, d| bits | 1 << d.index()))
    }
}

impl fmt::Debug for DominoSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

/// A set of contexts: bit `i` is the context with index `i`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ContextSet(u8);

impl ContextSet {
    pub const EMPTY: ContextSet = ContextSet(0);

    pub const fn contains(self, q: Context) -> bool {
        self.0 & (1 << q.index()) != 0
    }

    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns whether `q` was absent.
    pub fn insert(&mut self, q: Context) -> bool {
        let absent = !self.contains(q);
        self.0 |= 1 << q.index();
        absent
    }

    /// Members in index order.
    pub fn iter(self) -> impl Iterator<Item = Context> {
        Context::ALL.into_iter().filter(move |q| self.contains(*q))
    }
}

impl FromIterator<Context> for ContextSet {
    fn from_iter<I: IntoIterator<Item = Context>>(iter: I) -> ContextSet {
        ContextSet(iter.into_iter().fold(0, |bits, q| bits | 1 << q.index()))
    }
}

impl fmt::Debug for ContextSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}
