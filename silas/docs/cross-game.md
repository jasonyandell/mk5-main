# Cross-game policy comparison (#130): design

Written 2026-10-13 against `silas/` (README, GAPS.md gap 2 and its neighbours 1, 4,
7, 8; `src/measure.rs`, `src/frame.rs`, `src/policy.rs`), the peer review
`scratch/silas-lean/REVIEW.md` § 3.2, `wiki/entities/scenario-player.md` (the arena,
the field net, paired deals, the foundation repair), `wiki/experiments/book-study-2026-10-12.md`
(the bid book, chance baselines), texas-42's `lean/Texas42/{Strategic,Belief,Play,Deal,Auction,Cells,Information}.lean`,
the arena's `auction.py`/`bidders.py`/`match.py`, and the private coverage study's
140 matrix rows that cite #130 (the study's own tally: 138, of them 121 blocked and 17
citing it for a part; paraphrased here, never quoted). Read-only; nothing was run.
This document is the spec for the gap.

**The one-paragraph verdict.** The gap is not one object but four, and only one of
them is new mathematics. (i) A *match frame*: the scoring (points to 250, the book's
default; marks to seven, the tournament's) and the score, with the hand's settlement
as a function of the contract and the 42 points. (ii) A *bid action* in policies, over
an auction frame that gap 4 will later read. (iii) A *comparison* of two hand
policies: exact where it quantifies over outcomes (contract dominance, decided by a
43-case enumeration and proved in Lean in two lines), exact-given-a-field where it
quantifies over an enumerable deal set, and sampled otherwise, imported from the
arena as a ledger of paired deals with the authority `ExactGiven(field) ⊓
Sampled(n, seed)`. (iv) An *equilibrium* claim ("keep the setters honest") that no
fixed-field comparison can state; named, bounded, and left out of scope. The book's
condition for "42, not 84" is "within 42 points of 250", not "one mark from game";
both versions are the same theorem about `award`, and neither mentions a deal, a
belief or a field. That theorem is the first Lean-checkable cross-game claim.

---

## 1. Taxonomy of the book's cross-game claims

The 140 rows cluster into seven kinds. For each: the policies compared, the
distribution over deals, seats and match states, the field, the utility, and the
claim type. Three claim types are used throughout:

- **dominance** (field-free): a pointwise inequality over outcomes or deals, decidable
  exactly and provable in Lean;
- **fixed-field** (estimable): an expectation under a deal measure and a named field,
  `ExactGiven(F)` over an enumerated set, else `Sampled`;
- **equilibrium**: the field is a function of the policy under test; needs a
  response map and a mixed policy.

Row ids are the coverage matrix's; the assignment of every row is the appendix.

| cluster | rows | claim type | what unblocks it |
|---|---:|---|---|
| A. Bid only enough; the reasoned bids; never 32 or 33 | 9 | dominance on the contract + a field hypothesis on the auction | slices 1, 2 (statement); 4 (the auction hypothesis measured) |
| B. The 84 and the match state: 42 not 84 at the end; "far behind" | 13 | 1 dominance (the 42); the rest fixed-field through a match-win utility | slice 1 (the theorem); 4 (the utility's shape) |
| C. The 84 hand: protected off, straight off, lay-down | 12 | fixed-field belief statements at the auction (`P_F(make all 42 | shape)`); 2 already exact (lay-downs) | slices 3, 4; gap 7 for the set rate |
| D. The bid book: pricing a hand, the worked bids, pass or bid | 60 | fixed-field (`P_F(make n | hand)` against a cutoff); the risk count exact (gap 5) | slice 2 (the bids as a policy); 4 (priced) |
| E. Reputation, style, adapting opponents | 14 | 4 equilibrium; 10 fixed-field against a *named non-equilibrium* field | slice 4 (the ten); slice 5 (the four) |
| F. The scoring frame and the match (definitions) | 18 | definitions; 2 one-line theorems (the 43 floor; a set outscores the make) | slice 1 |
| G. 84 play rules cited to #130 only because the 84 was unrepresentable | 14 | play rules guarded on `(bid 84)`; their *evaluation* is gap 7 | slice 0 (re-tag), no code |

### A. Bid only enough (9 rows: S-0036-e, S-0152-c+S-0153-b, S-0153-a, S-0053-a, S-0137-e, S-0034-a, S-0033-b, S-0028-d, S-0128-c)

*Policies.* `σ_min`, which at its turn bids the least legal raise within its
willingness (the arena's `HeuristicBidder` is already this, with `willing = 42 −
exposed count − caution`), against `σ_+k`, identical except that it bids `k` above
the least raise when its willingness allows. "Never 32 or 33" is `σ_min` with the
levels 32 and 33 removed from its range; "the reasoned bids are 30, 31, 35, 36" is a
claim about the histogram of `σ`'s bids over the deal.

*Distribution.* Uniform over deals; all four seats and shakers (the last bidder sees
every bid, so the seat is part of the state); match state irrelevant to the dominance
part, relevant to the field part.

*Field.* The claim decomposes:

1. **Holding the contract** (the auction over, the play the same), a lower point level
   weakly dominates a higher one in every world. Under marks, every level 30 to 41 pays
   one mark (`Contract::stake`) and `Makes` is antitone in the threshold. Under points,
   a made bid scores all the points taken whatever the level (S-0017-c), and a set
   pays the setters the bid plus their points, so the lower level loses no payoff when
   made and pays less when set. **Dominance**, provable (§ 3).
2. **Winning the auction.** The higher bid wins auctions the lower would lose; the
   claim's content is that the field rarely overcalls a 31 and not a 33, i.e.
   `P_F(outbid at n ∧ ¬outbid at n+k)` is small, and raises go to 35 anyway (S-0034-a).
   **Fixed-field**: a property of the field's bid kernel, measurable by a histogram
   of its responses.

*Utility.* Marks per hand (the dominance part holds for match-win too, since it is
pointwise on states). S-0153-a is a histogram claim, not a comparison: it is exact
over the 1,184,040 hands under a fixed bid rule (as S-0053-d, "half of all hands are
not biddable", is).

### B. The 84 and the match state (13 rows: S-0089-b, S-0170-a, S-0089-d, S-0082-a, S-0089-c+S-0167-a, S-0047-a, S-0110-a, S-0122-a, S-0127-b, S-0129-b+S-0137-g+S-0080-d, S-0155-c, S-0164-c, S-0165-b)

*The dominance (S-0089-b, S-0170-a's second half, S-0082-a).* Two contracts with the
same bidder and declaration, `M(1)` (42) and `M(2)` (84). Both have threshold 42
(`Contract.threshold`), so `Makes` is the same proposition for both in every world;
the stakes are 1 and 2. At a match state where one mark (points: 42 points) wins,
a make under either contract is a win and a set gives the opponents one mark more
under the 84 (points: 42 more, plus their own points under either reading of
S-0082-a versus S-0017-c; the inequality holds under both). With `V` the probability
of winning the match, constant at 1 on won states and antitone in the opponents'
score, `V(m + settle(M(1), p)) ≥ V(m + settle(M(2), p))` for every outcome `p`.
**Dominance**, over outcomes, not deals: no belief and no field appear. It
generalizes: needing `j` marks, `M(j)` dominates `M(k)` for every `k > j`. The
*policy* version ("bid 42, not 84") adds one hypothesis: the field's overcalls after
my `M(1)` and after my `M(2)` lead to settlements ordered the same way. S-0165-b is
the book's own warning that the field will overcall a 42 at this very score, so the
hypothesis is a field property, measurable, and the policy claim is
`ExactGiven(F)`.

*The variance claims ("far behind, gamble"; "ahead, protect"; S-0129-b, S-0122-a,
S-0110-a, S-0047-a, S-0164-c, S-0089-d, S-0127-b).* Policies `σ_cautious` and
`σ_bold` differing in willingness; the claim is `Δ(σ_bold, σ_cautious; F, D, m) > 0`
when `m` is far behind and `< 0` when ahead. In match-win utility this is a statement
about the curvature of `V` in the score difference: a mean-preserving spread of the
hand's outcome raises `V` where `V` is convex (behind) and lowers it where concave
(ahead). Given `V`, the sign is **exact** (a dynamic program over the score grid);
`V` itself is the value of the match game under a per-hand outcome model, and that
model is estimated under a field. So: **fixed-field**, with the field entering only
through a 2-parameter-per-contract hand-outcome table. S-0170-a's first half ("bid 84
any time") under linear marks is `2(2p − 1) ≥ 2p' − 1` for the 84's `p` against the
alternative's `p'`, i.e. `p > (2p' + 1)/4` (the bid book's own formula): a belief
statement, not a dominance.

*Utility.* Match-win probability from the given score; marks per hand is the wrong
unit for every row here except S-0082-a's definition.

### C. The 84 hand (12 rows: S-0082-b, S-0083-a, S-0083-b, S-0084-b, S-0086-c, S-0088-a, S-0088-b, S-0089-a+S-0101-c, S-0146-a, S-0149-a, S-0150-a, S-0188-a)

*Policies.* The bidder's `σ_84` (bid two marks on the shape) against `σ_1` (one mark
or the best point bid on the same hand). *Distribution.* The fiber of the bidder's
hand (399,072,960 worlds for a hand before the lead), uniform; seat irrelevant
before the auction except through the field's overcalls. *Field.* Needed: the
defenders' play determines `P(make all 42)`; the book's reason 2 (S-0083-b, "the
defenders do not know what to save") is a claim that the field's imperfect
information lowers its set rate below the double-dummy bound, so the comparison
must be against an information-respecting field, never the oracle. *Utility.* Marks
per hand or match-win. *Claim type.* **Fixed-field** belief statements
`P_F(make | shape) > (2p' + 1)/4`, `ExactGiven(F)` over the enumerated fiber when a
play-out over it is affordable, else `Sampled`. Two rows are already exact: the
lay-downs (S-0082-b, S-0188-a; `lay-down.silas`, `(certain-make)`), whose three
marks are this gap's settlement only. The straight-off 84's set rate "two in three"
(S-0089-a) is gap 7 (play-out over a belief) with this gap's contract.

### D. The bid book (60 rows; the list is in the appendix)

*Policies.* The book's bid rule as one policy, `risk-budget`: for each pip trump,
price the trumps and the offs (the tiles that can beat each off's exposed side,
counted once), add them, bid `42 − risk` when `risk ≤ 12`, with the book's own
overrides (one off at 16: pass, S-0026-a; two offs at 17: try 30, S-0031-a; count on
the partner for one trick with two or more offs, S-0028-b). The 30 transcribed hands
are its exact table. *Distribution.* The book's hands (a listed set, so a claim
about those hands, not about the deal), or uniform. *Field.* Needed for any
*verdict* on a bid; not for the *count*. *Utility.* `P_F(make n | hand)` against a
cutoff, or marks per hand. *Claim type.* Two parts: the risk count is **exact**
(a `card` of the tiles beating the off's exposed side: gap 5), and the bid's
rightness is **fixed-field** (`ExactGiven(F)` on the hand's fiber under a play-out,
or `Sampled`). The wave-1 bid book already measured the second part for Zeb's
forecast (25 of 30 on pass-or-bid, against "always bid" at 23). The trump-call rows
(S-0032-a, S-0104-b, S-0107-a, S-0121-b, S-0125-b, S-0134-e and the Ch. 9 doubles
rows) are the same comparison with the declaration as the policy's choice.

### E. Reputation and style (14 rows: S-0089-e, S-0081-a, S-0154-a, S-0037-a; S-0080-b, S-0080-c, S-0070-a, S-0114-b, S-0155-a, S-0205-a, S-0184-a+S-0197-a, S-0188-b, S-0190-a, S-0191-a+S-0226-a)

*Equilibrium (4: S-0089-e, S-0081-a, S-0154-a, S-0037-a).* "Keep strong defenders
honest with an occasional straight-off 84": the defenders' policy is a best response
to the frequency with which I bid 84 on a straight off; with `σ_q` the mixed policy
that does so with probability `q`, the claim is `∃ q > 0, Δ(σ_q, σ_0; BR(σ_q), D, m)
> 0`, equivalently that `σ_0` is exploitable by `BR(σ_0)`. A fixed-field comparison
cannot state it: against any fixed field a bluff that is set two times in three has
pure cost, and the comparison would "refute" a claim that was never about a fixed
field. Needs mixed policies and a response map `BR`; Lean's `bestResponse_eq` is for
a fixed field. Out of scope until the fixed-field comparison exists (slice 5).
"Become a frequent bidder" (S-0037-a) is the bid rate as the mixed parameter.

*Fixed-field against a named non-equilibrium field (10).* "Against an over-aggressive
bidder, pass and set him" (S-0080-c), "against habitual overbidders lead for the
tens" (S-0070-a), "marks do not punish frivolous bidding, points do" (S-0114-b): the
field is a *named* deviation (`HeuristicBidder(caution = −k)`, or a field whose offs
are ten-count suits), and the claim is `Δ(σ_book, σ_default; F_named, D, m)` with
that field's identity in the authority. S-0114-b is the same policy pair compared
under both scorings, with the claim that the sign differs. The Ch. 14 anecdotes
(aggressive and conservative champions) are comparisons of `willingness` settings
against the same field. All **fixed-field**, `Sampled`.

### F. The scoring frame (18 rows: S-0015-c, S-0017-c+S-0132-a, S-0020-a, S-0063-a+S-0154-b, S-0063-b, S-0072-b, S-0073-a, S-0113-a+S-0209-b, S-0114-a, S-0156-a, S-0166-a, S-0166-b, S-0168-b, S-0199-b, S-0208-b, S-0208-c, S-0208-e, S-0210-e; S-0114-b's frame is here, the row is counted in E)

Definitions, not decisions: the points game (first to 250; a made bid scores the
points taken; the setters score the bid plus their points; the bidder of the deciding
hand wins a tie at 250), the marks game (one mark a hand, two for 84, three for 126,
first to seven; the hand stops once made or set), the tournament's bidding limits,
the forced bid, the raise steps after 84. Two are one-line theorems: a set is worth
at least 43 under points for every bid `b` (`b + (43 − b)`, S-0063-b), and "a set
always outscores the make it replaced". S-0073-a ("trump sets score 60 to 70") is an
empirical histogram, fixed-field. These rows are unblocked the moment the frame
exists; they need no comparison.

### G. 84 play rules miscited (14 rows: S-0085-b, S-0087-b, S-0090-a+S-0096-a, S-0092-a+S-0097-a, S-0092-b, S-0095-a, S-0098-a, S-0098-b, S-0099-a, S-0099-b, S-0100-a, S-0100-b, S-0100-c, S-0101-a)

Discard and plan rules for playing or defending an 84 ("save your doubles", "hold
pairs", "the discard priority list"). They cite #130 because the catalogs were
written when `Contract` flattened an 84 to 42; the contract now carries
`Bid::Marks(2)` and `(bid 84)` is a guard (GAPS.md, Closed). What remains of #130 in
them is nothing: each is a play rule, statable now, blocked only by its *other*
gaps (#128 on four, #135 on three, #132 on one) and by gap 7 for any claim about the
rule's set rate. Slice 0 re-tags them.

---

## 2. The Silas design

Four additions, each the smallest that states its cluster. Silas owns the frame, the
bid action, the settlement, the deal measure, the exact comparisons and the ledger
import. The arena owns Zeb, the field nets, the paired runner and the sampling. They
connect through data (a ledger of paired outcomes with a field identity), never
through a call: Silas never links a solver or a net.

### 2.1 The match frame and the settlement

```rust
pub enum Scoring { Points { target: u32 }, Marks { target: u32 } }   // 250; 7
pub struct MatchState { scoring: Scoring, score: [u32; 2] }          // by Team
pub struct Outcome { declaring: u32, other: u32 }                    // declaring + other = total_points (42)
pub struct Transfer { to: [u32; 2] }                                 // what each team gains this hand

impl Contract {
    /// The hand's settlement under a scoring. Marks: `award` (the stake to the
    /// side that gets it). Points: made, the bidder's side scores the points it
    /// took (a marks bid: its value, 42·m) and the other side what it took; set,
    /// the bidder's side scores nothing and the other side the bid's value plus
    /// what it took.
    pub fn settle(&self, scoring: Scoring, outcome: Outcome) -> Transfer;
}
impl MatchState {
    pub fn apply(&self, contract: &Contract, outcome: Outcome) -> MatchState;
    pub fn won(&self) -> Option<Team>;        // first to the target; under points, the deciding hand's bidder wins a tie
    pub fn short(&self, team: Team) -> u32;   // what the side still needs
}
```

Points scoring is not in the engine (`gameTarget` is marks), the arena
(`marks_to_win`), or Lean (`award : Team × ℕ`). It must exist to state the book's
own version of nearly every row in B, D and F, and `settle` under points is where
the one textual ambiguity lives: whether a set marks bid under points pays the
setters their own points too (S-0082-a says "they get 84"; S-0017-c's rule says bid
plus their points). `settle` takes the S-0017-c reading and the header records the
other; every theorem below holds under both.

**Utility.** `Utility::Marks` (the signed transfer for a side), `Utility::Points`,
and `Utility::MatchWin(V)` with `V : MatchState → ℚ`. `V` is built once, exactly,
from a *hand-outcome model* `H : MatchState → Contract-choice → FinPMF Outcome`
(what contract the policy under test reaches from each state and how it ends) by a
dynamic program on the finite score grid (8 × 8 states under marks; 251 × 251 under
points, transitions in hand-sized steps). `V` carries `ExactGiven(H)`; `H` is
estimated in the arena or taken from a declared toy. Properties Silas checks on any
`V` it is handed, because the theorems need them: `V = 1` on states the side has
won, `0` on states it has lost, antitone in the opponents' score at a fixed own score.

> **Amendment (slice 1, 3e2d3135).** The match-win utility needs a fourth property: V never
> falls as the side's own score rises (monotone in own score at a fixed opponents' score).
> With only the three properties above, "bid only enough" (31 vs 33) provably fails: a made
> 31 against a set 33 changes both scores, which the order "own score equal, other's no
> larger" cannot compare. The implemented order holds iff it holds for every admissible V.
> The Lean `WinUtility` below needs the same field (`mono_own`).


### 2.2 The bid action and the auction frame

```text
policy   := (policy name (fallback lowest-legal) section ...)
section  := (bidding brule ...) | (declaring drule ...) | form       ; form as today
brule    := (rule name (select bid n | pass | least-raise) (guard query))
drule    := (rule name (select declare q) (guard query))
```

A bid rule's guard is a query over the **auction frame**: the viewer, its dealt hand,
the shaker, the bids so far in order, the legal bids and the match state. Viewer
predicates added, all literal-numbered as Silas's numbers are:

| predicate | reads |
|---|---|
| `(shaker c)` | who shook |
| `(bid-of c b)`, `(passed c)`, `(no-bid)` | the auction so far (the same record gap 4 reads; it lands here as structure, its reading is #134 and #128) |
| `(high-bid n)`, `(legal-bid n)` | the auction's state; legality is Lean's `legalBid` under an `AuctionConfig` (max mark bid, entry at `M(1)` or `M(2)`, 42-steps or 84-steps after, forced bid) |
| `(scoring marks)`, `(scoring points)` | the frame |
| `(score c n)`, `(short c n)`, `(short-at-most c n)`, `(behind c n)` | the match state for `c`'s side |

The *hand policy* is the existing play policy plus these two sections; it is still a
function of its frame (the auction frame before the lead, the play frame after), so
F1's measurability holds and gap 1's likelihood applies to bids unchanged. The
`Record` grows an `auction: Auction { shaker, bids: Vec<(Seat, Bid)> }` and
`Contract` becomes derived from it (`Contract::ofBid`, as Lean's
`DealLocalInfo.mech`); `PolicyKey.history` includes the bids. This is the structural
half of gap 4, done once for both gaps.

Examples the clusters need:

```text
; S-0036-e. The least legal raise, if the hand is worth it.
(policy bid-only-enough (fallback lowest-legal)
  (bidding
    (rule least (select least-raise) (guard (query (roles (number n)) (out) (pattern (viewer v) (least-raise n) (worth v n)))))
    (rule pass (select pass) (guard (query (roles) (out) (pattern))))))

; S-0089-b / S-0170-a. Needing one mark (or at most 42 points), one mark, not two.
(rule the-last-mark (select bid 42)
  (guard (query (roles (chair v)) (out) (pattern (viewer v) (short-at-most v 42) (eighty-four-shape v)))))
```

`worth` and `eighty-four-shape` are prelude definitions over the hand (the book's
shapes: three or four trumps, two or three doubles, one off under its double);
`(worth v n)` with a literal `n` is the risk budget written as a disjunction of
literals until gap 5's `card` lands, as `lead-sets` is today.

> **Amendment (slice 2, as built).** The grammar is `(select bid n|pass|least-raise)`
> and `(select declare answer|q0..q6|doubles|no-trump)`, with the short forms
> `(rule r (bid ...) clause ...)` and `(rule r (declare t) clause ...)`. `worth` is not
> one definition but a family per bound (`exposes-at-most-10`, `risk-at-most-7`, in
> `lib/bidding.silas`), since a bound is a literal; a bid level is a rule. The match
> state is in the auction frame only. Every choice made against this section is in
> § 4, Slice 2 as built.

### 2.3 The deal measure, the seats and the field

```rust
pub enum Deals {
    Listed(Vec<DealId>),             // the book's hands: exact over that list; a claim about those deals
    Fiber(Frame),                    // every completion of one hand: enumerable or resolving; exact
    Hands,                           // the viewer's own hand over C(28,7): exact (gap 6's deal view)
    Uniform { sample: Sample },      // the 472,518,347,558,400 deals: Sampled(n, seed)
}
pub enum Seats { Rotate, Fixed(Seat) }     // the policy under test in every chair, every shaker (paired), or one
pub struct Field { id: String, seats: [HandPolicySpec; 4] }
pub enum HandPolicySpec { Silas(CompiledPolicy), Named(String) }   // Named: an arena model, by identity (net hash, eps, seed)
```

A field of four Silas policies is one Silas can play out itself. A `Named` seat is
one only the arena can play; a comparison that needs it is a ledger import (§ 2.5).
Either way the field is a model, so its identity enters the authority:
`Authority::given(field.id)`. Self-play is a model too; nothing in a comparison is
ever `exact` except the enumerations of § 2.4.

### 2.4 The two forms, and how they compute

**`dominates`: exact, over outcomes.**

```text
(dominates name
  (contracts (marks 1) (marks 2))        ; same bidder and declaration
  (frame (scoring marks 7) (short 1))    ; or (scoring points 250) (short-at-most 42)
  (utility match-win))
```

Decided by enumerating the 43 outcomes `(declaring, other)` with `declaring + other
= 42`, and for each comparing the two states `m.apply(K₁, o)` and `m.apply(K₂, o)`:
either both are won by the side, or the side's own score is equal and the opponents'
is no larger under `K₁`. That is state dominance: it implies `V(·) ≥ V(·)` for every
`V` with the § 2.1 properties, so no `V` is needed and the answer is `exact`. The
same form with `(contracts (points 31) (points 33))` decides cluster A's contract
half under either scoring; with a concrete `V` it also decides the variance claims
of cluster B exactly given `V` (`ExactGiven(H)`).

**`compare`: the expectation of a paired difference.**

```text
(compare name
  (policies a b)                                  ; two hand policies
  (frame (scoring marks 7) (score 6 4))           ; one match state, or (score any) under a declared score measure
  (deals (listed hand-21 hand-22) | (fiber hand-21) | hands | (uniform (sample 22000 1000000)))
  (seats rotate)
  (field F)
  (utility marks | points | match-win))
```

Denotation. With `settle(σ, F, ω, s, m)` the transfer when `σ` sits in chair `s`
against `F` on deal `ω` from state `m` (the auction, then the play, each seat a
function of its own frame), and `U` the utility,

```
Δ(a, b) = E_{ω ~ D, s ~ S} [ U(m.apply(settle(a, F, ω, s, m))) − U(m.apply(settle(b, F, ω, s, m))) ].
```

It computes one of three ways, by what `deals` and `field` are:

1. **Exact given the field**: `Listed`, `Fiber` (within the cap) or `Hands`, with a
   field of Silas policies. Silas plays each world out itself: the auction by each
   seat's bid rules, then 28 frames, one per play (`Frame::after`), each seat's
   policy at its own frame. Authority `ExactGiven(F)`; `exact` on the deal part. A
   fiber of 399 million worlds is enumerable for a *query* (resolving) but not for a
   play-out; above the cap the comparison refuses, as everything in Silas does, and
   the caller samples instead.
2. **Sampled, by Silas**: `Uniform` with a Silas field, drawn from the seeded stream;
   `ExactGiven(F) ⊓ Sampled(n, seed)`.
3. **Sampled, by the arena**: any `Named` seat. Silas does not compute; it imports a
   ledger (§ 2.5) and labels it `ExactGiven(F) ⊓ Sampled(n, seed)`.

The result:

```rust
pub struct Comparison {
    delta: Estimate,                 // the mean paired difference, in the utility's unit
    n: usize,                        // deals (clusters), not seatings
    flips: (usize, usize, usize),    // deals where a won, b won, tie
    interval: (BigRational, BigRational),   // deal-clustered, at the declared confidence
    verdict: Verdict,                // Better only when the interval clears zero
    chance: Option<Box<Comparison>>, // the same comparison with b replaced by the chance baseline, always reported
}
```

The band of `measure::compare` finally has its job and is not chosen by the caller:
it is derived from `n` at a declared confidence (Hoeffding on the bounded
differential, `|Δ| ≤ 2 · max stake` in marks, `≤ 1` in match-win; or the
deal-clustered bootstrap from the ledger), per REVIEW.md F6.

**Paired deals.** The two policies play the *same* deal, in the *same* chair, against
the *same* field with the *same* dice (the arena's `run_paired` and the record-keyed
tape already give this; the E1b design is the template: 22,000 paired deals, 962
flips for and 386 against on a +0.026 marks/hand effect). The estimate is the mean
of per-deal differences, the interval is over deals, and each deal's seatings are one
cluster. This is not a variance trick layered on top of the definition; § 3 shows it
is the definition's own form (`Δ` is a single expectation of a per-deal difference).

### 2.5 The arena ledger: what the arena owns, what Silas owns

The arena writes, per comparison, one row per (deal, seating, policy):

```
ledger: { field: "<identity>", sample: { n, seed }, scoring, state, utility-free }
row:    { deal: <seed or id>, shaker, chair, policy: a|b, auction: [(seat, bid)...], contract, outcome: {declaring, other}, plays: [...] }
```

The ledger carries **outcomes, never utilities**. Silas recomputes every transfer
and utility from `(contract, outcome, state)` through its own `settle`, so the units
are Silas's by construction, and the arena cannot report marks where points were
asked. The ledger's `field` string is the arena model's identity (net content hash,
`field_eps`, seed, training pour), which becomes `Authority::given`.

On the arena side (Python, outside Silas): a `SilasBidder`/`SilasPlayer` adapter
(the book study already drove `SilasPolicy.allowed(frame)` at pour positions; the
bid half is new and reads `BidContext.marks`/`marks_to_win`, which exist), a
points-scoring option in `score_hand` or none (the arena may stay marks-only, since
Silas settles), a fixed-state runner (hands from a given score, not games to seven;
E1b's one-hand games are this at 0–0), and the ledger writer.

**Differential check.** For a field of Silas policies, the arena's play of a deal and
Silas's own play-out of it must agree tile for tile; `tests/` holds the ledger of a
Silas-only field to Silas's exact play-out. That is the referee-and-core discipline
applied across the boundary, and it is how a ledger from a `Named` field earns trust
in its plumbing without Silas trusting the net.

---

## 3. The Lean statement

Objects texas-42 already has: `Contract` with `kind`, `amount`, `declaration`,
`threshold`, `stake`, `Makes`, `award` (`Play.lean`); `Bid`, `AuctionState`,
`legalBid`, `result` (`Auction.lean`); `Deal` (finite, `Belief.lean`);
`PolicyKernel`, `likelihoodFrom`, `posterior` (`Belief.lean`); `BeliefProc`,
`latentVal`, `beliefVal`, `beliefVal_eq_exp_latentVal`, `bestResponse_eq`
(`Strategic.lean`); `DealLocalInfo` with the auction (`Information.lean`);
`terminal_scores` (the 42 points are conserved, `Play.lean`).

### 3.1 New definitions (the trust surface)

```lean
inductive Scoring | points (target : ℕ) | marks (target : ℕ)

structure MatchState where
  scoring : Scoring
  score   : Team → ℕ

/-- The hand's transfer: marks by `award`; points by the book's rule (S-0017-c). -/
def Contract.settle (K : Contract) (sc : Scoring) (pD pO : ℕ) : Team → ℕ

def MatchState.apply (m : MatchState) (K : Contract) (pD pO : ℕ) : MatchState
def MatchState.Won (m : MatchState) (t : Team) : Prop := target ≤ m.score t   -- per scoring

/-- A match-win utility for side `T`: 1 on won states, 0 on lost ones, antitone in the
other side's score at a fixed own score. -/
structure WinUtility (T : Team) (V : MatchState → ℚ) : Prop where
  won      : ∀ m, m.Won T → V m = 1
  lost     : ∀ m, m.Won T.other → V m = 0
  antitone : ∀ m m', m.scoring = m'.scoring → m.score T = m'.score T →
               m.score T.other ≤ m'.score T.other → V m' ≤ V m
```

### 3.2 Theorems (field-free; no `Deal`, no `FinPMF`)

**The first Lean-checkable cross-game claim: needing the last mark, one mark
dominates two.**

```lean
theorem mark_contracts_same_make (b : Seat) (δ : Declaration) (j k p : ℕ) :
    (⟨b, .mark, j, δ⟩ : Contract).Makes p ↔ (⟨b, .mark, k, δ⟩ : Contract).Makes p := by
  simp [Contract.Makes, Contract.threshold]      -- both thresholds are 42

theorem last_mark_dominance {T : Team} {V} (hV : WinUtility T V)
    (m : MatchState) (b : Seat) (hb : b.team = T) (δ : Declaration)
    (j k : ℕ) (hjk : j ≤ k) (hneed : m.short T ≤ j)      -- j marks (or j·42 points) win
    (pD pO : ℕ) (hsum : pD + pO = 42) :
    V (m.apply ⟨b, .mark, k, δ⟩ pD pO) ≤ V (m.apply ⟨b, .mark, j, δ⟩ pD pO) := by
  by_cases h : (⟨b, .mark, j, δ⟩ : Contract).Makes pD
  · -- made: both states are won by T (j suffices, k ≥ j), so both values are 1
    ...
  · -- set: T's score is unchanged in both; the other side gains j (+pO) under the
    -- first and k (+pO) under the second; antitone
    ...
```

Checked: both contracts need all 42 points, so the make is the same event; the stake
is the only difference; a won state is terminal for every admissible `V`. The marks
version (`Scoring.marks 7`, `hneed : 7 ≤ m.score T + j`) and the points version
(`Scoring.points 250`, `hneed : 250 ≤ m.score T + 42·j`) are the two instances of one
theorem; the book states the points one (S-0089-b) and the marks one (S-0170-a). The
proof is a case split and two lemma applications; it should also be kernel-decidable
for a concrete `V` by `decide` over `pD ∈ 0..42`. No `sorry`, no `native_decide`.

Its policy-level corollary needs a hypothesis about the field's auction response:

```lean
/-- The opponents' continuation after `M(j)` and after `M(k)` settle to states ordered
the same way, for every deal. -/
def OvercallMonotone (F : Field) (m : MatchState) (j k : ℕ) : Prop := ...

theorem last_mark_policy_dominance (hF : OvercallMonotone F m j k) ... :
    Δ (bid j) (bid k) F D m ≤ 0
```

`OvercallMonotone` is a finite property of the field's bid kernel (four bids); it is
measured, not proved, and S-0165-b says it can fail. That is the honest boundary of
the theorem.

**Cluster A's contract half.**

```lean
theorem point_level_dominance (sc : Scoring) (b : Seat) (δ : Declaration)
    (n n' pD pO : ℕ) (h : n ≤ n') (hsum : pD + pO = 42) (t : Team) (ht : t = b.team) :
    (⟨b, .point, n', δ⟩ : Contract).settle sc pD pO t ≤ (⟨b, .point, n, δ⟩ : Contract).settle sc pD pO t
  ∧ (⟨b, .point, n, δ⟩ : Contract).settle sc pD pO t.other ≤ (⟨b, .point, n', δ⟩ : Contract).settle sc pD pO t.other
```

Under marks: `stake = 1` at every level and `Makes` is antitone in `n`. Under
points: made pays `pD` whatever `n`; set pays `n + pO` to the other side, monotone
in `n`. Field-free; the auction is the hypothesis.

**Cluster F's two lines.** `set_floor : ∀ b, 30 ≤ b → b ≤ 41 → ∀ pO, (43 − b) ≤ pO →
43 ≤ b + pO` and `set_beats_make`.

### 3.3 The measured claims: definition in Lean, value as evidence (TRUST-01)

Instantiate `BeliefProc` for one hand: `S` the policy's seat's `DealLocalInfo`
(auction and plays so far); `A := Bid ⊕ Declaration ⊕ Domino`; `O` the public
segment until the seat's next decision; `L := Deal × Ξ` with `Ξ` the field's
randomness (`field_eps`'s coin); `reward := 0`; `util s l := U (m.apply (settle …))`
at the terminal; horizon the seat's decisions (at most nine). Then
`beliefVal_eq_exp_latentVal` gives

```
value(σ; F, β, m) = β.exp (latentVal σ)         and, by linearity of FinPMF.exp,
Δ(a, b; F, β, m)  = β.exp (fun ω => latentVal a ω − latentVal b ω).
```

`Δ` is one expectation of a per-deal difference. The paired estimator is its Monte
Carlo form; the deal-clustered interval is the right one because the summand is the
per-deal difference. These are theorems about the *definition* of `Δ` and are
provable now. The *value* of `Δ` for a `Named` field is never in Lean: it is a
`Comparison` with its authority, cited in the wiki at the evidence tier, exactly as
TRUST-01 keeps an external `PASS` from becoming an axiom. If a Lean statement ever
needs it, it is a hypothesis of the theorem, named with the ledger's identity.

### 3.4 What is not stated

The equilibrium claim (cluster E's four rows). Lean has `bestResponse_eq` for a fixed
field; "keep them honest" is `∃ q > 0, Δ(σ_q, σ_0; BR(σ_q), …) > 0` with `BR` a
response map and `σ_q` a `FinPMF` over pure policies. Stating it needs the mixed
policy and the response map as objects; proving anything needs the two-partnership
game. Named here so that no fixed-field `Comparison` is ever cited as its verdict.

---

## 4. Staged build plan

Each slice is testable alone and names the rows it moves. Row counts are from the
appendix. "Statable" means the row's claim becomes a Silas file with a number or a
verdict; "priced" means the number is measured.

### Slice 0: re-tag (no code; an hour)

**What.** Cluster G's 14 rows cite #130 for the unrepresentable 84 only. Re-tag each
to `encodable-now` or to its remaining gap (#128 ×4, #135 ×3, #132 ×1, gap 7 for set
rates). Same for S-0149-a (an 84 made: a settlement, slice 1).
**Tests.** None; a coverage edit in the archive. **Unblocks.** 14 rows (G) leave
#130. **Cost.** Trivial.

### Slice 1: the match frame, settlement, utility and `dominates` (low; two days)

**API.** `Scoring`, `MatchState`, `Outcome`, `Transfer`, `Contract::settle`,
`MatchState::{apply, won, short}`, `Utility::{Marks, Points, MatchWin(V)}`, the `V`
property check, the score-grid DP from a hand-outcome table, the `(dominates …)`
form in the language and `silas dominate FILE`.
**Tests.** `settle` on the catalog's own vectors (bid 35, took 30 against 12: 0 and
47; bid 30, took 33: 33 and 9; bid 35, took 33: 0 and 44); marks `settle` equals
`award`; the 43 floor for every bid; `dominates` says yes for `(marks 1)` vs `(marks
2)` at `short 1` under marks and at `short-at-most 42` under points, by enumeration;
says *no* for the same pair at `short 2` (84 is not dominated two marks out); says
yes for `(points 31)` vs `(points 33)` under both scorings; under `Utility::Marks`
the 84 wins at `p = 0.6` and loses at `p = 0.4` against a one-mark bid (the
`(2p' + 1)/4` line); a toy `V` failing antitonicity is refused; the DP's `V` at
6–6 under marks is the next hand's make probability.
**Lean.** § 3.1 definitions, `last_mark_dominance`, `point_level_dominance`,
`set_floor`.
**Unblocks.** Cluster F (18) statable and done; cluster B's dominance rows
(S-0089-b, S-0170-a's second half, S-0082-a: 3) proved; cluster A's contract half
stated as a theorem (the 9 rows move to "file (+ field)"). 21 rows closed, 9
half-closed.

### Slice 2: the bid action and the auction frame (medium; a week)

**API.** `Auction { shaker, bids }` in the `Record`; `Contract` derived from it;
`AuctionConfig` (max mark bid, entry rule, raise steps, forced bid) on the
`Physics`/frame; `Frame::auction(...)` constructor for a seat before the lead;
the predicates of § 2.2; `(bidding …)` and `(declaring …)` sections;
`CompiledPolicy::bid(&AuctionFrame) -> Bid` and `::declare`; `PolicyKey.history`
with bids; `silas bid --hand ID --seat ROLE --policy FILE`. `legal-bid` is Lean's
`legalBid` transcribed, under differential test against the arena's legal set.
**Tests.** Legality at every seat and shaker under both entry rules; a policy's bid
is a function of the auction frame (same frame, same bid; measurability by the
`Viewer`-only check); `bid-only-enough.silas` reproduces `HeuristicBidder` on 1,600
random hands at every seat (the arena's bidder is the oracle for the encoding, as
the gates were for the rules); `risk-budget.silas` on the book's 30 transcribed
hands with the bids the book prints, at all four seats, under both scorings, as
`tests/book_bids.rs`; `no-32-33` never bids 32 or 33; the bid histogram of
`risk-budget` over `Hands` (exact, 1,184,040 hands) gives the biddable share
(S-0053-d's "about half") and the level histogram (S-0153-a's "30, 31, 35, 36").
**Unblocks.** Cluster A fully statable (9); cluster D's bids statable as a policy
and an exact table (60 rows move from `blocked-by #130` to a file, their verdicts
waiting on slice 4); S-0053-d and S-0153-a priced exactly (2). Gap 4 receives its
structure for free. 69 rows move; 11 close.

#### Slice 2 as built (2026-10-13)

Built: `Call`, `AuctionConfig`, `Auction` (legal by construction), the contract
derived from the auction (`Frame::new(auction, declaration, ...)`), `AuctionFrame`,
predicate phases and the auction predicates, the `(bidding ...)` and `(declaring
...)` sections, `CompiledPolicy::{bid, declare}`, `silas bid`, `lib/bidding.silas`,
and `examples/book/bidding/` (`bid-only-enough`, `risk-budget`, `never-32-or-33`).
README § The auction is the reference. Where this spec was silent or did not fit
the code, the choice made:

1. **The configuration.** `AuctionConfig` is the marks cap and the forced bid. The
   entry at one or two marks and the one-mark overcall after a marks bid are fixed:
   rules-of-42, the TypeScript engine, the arena and Lean's `legalBid` all say so, and
   no source we hold varies them; the "42-steps or 84-steps" knob has no source and is
   left out. "Both entry rules" in the tests is read as the two all-pass rules: the
   reshake (four passes, no contract) and the forced shaker.
2. **The record.** The `Auction` carries its configuration, shaker and calls, and
   refuses an illegal call, so the contract (`Auction::contract`) needs no legality
   check of its own. `Contract::new` stays, as Lean's `Contract.ofBid`, for the
   match's settlements and `dominates`, which need no auction. A hand given with its
   contract alone (the book's `hands.json`, Zeb's record through Python) gets
   `Auction::won_by`, the shortest auction its declarer wins; the auction predicates
   read that convention there, not the table's auction. `PolicyKey` keys on the
   auction (`PolicyKey.history with bids`).
3. **Phases.** Every predicate is of the play, the auction or both; a rule naming a
   predicate of the other phase is refused at compilation, a query at evaluation.
   The auction's record (`shaker`, `bid-of`, `passed`, `no-bid`, `high-bid`,
   `high-bidder`) is read in the play too; the match state (`scoring-*`, `score`,
   `short`, `short-at-most`, `behind`) only at the auction, since the play frame does
   not yet carry it. `(behind c n)` is "trails by at least `n`". Pips are named by
   contexts (`(bears d q)`, `(high-pip d q)`), since before the declaration no suit
   is called.
4. **Decisions.** The fallback is the least legal call (a pass when legal, 30 for a
   forced shaker: the arena's forced bid) and the first declaration offered. A
   declaring rule names declarations by what their tables make trump (`Trump::of`);
   of those it allows the first offered decides, so the order offered is the
   tie-break (the arena's higher pip is the order P6 to P0). The declarations
   offered are on the auction frame and must agree on the count. Rule names are
   unique across the sections. There is no exact table for bids: "the 30
   transcribed hands are its exact table" is read as the policy's table of calls on
   those hands, pinned in `tests/bidding.rs`, not as an exact layer.
5. **`bid-only-enough` is the arena's `HeuristicBidder`**, bid and declaration,
   including its tie-breaks (length, then the trump double), so the 16 declaring
   rules compare suit lengths with scopes of up to seven roles. The role cap rose
   from 64 to 256 for it.
6. **`risk-budget` is our reading** of the book's count, since no one formula gives
   all its worked numbers: each off risks its trick and the count of the suit it
   leads (its high pip) that the hand lacks and is not trump, counted once; without
   the trump double one trump trick is at risk, with the count trumps lacked; a
   trump suit is three tiles or more; bid 42 less the risk to 12, try 30 with two
   offs to 17, and 84 on the shape. The shape requires the trump double, which every
   84 the book bids holds and the shape as stated does not say (without it Hand 10,
   a pass, is an 84). The reading matches 8 of the 12 risk counts `hands.json`
   prints. The bid book is the study's 30 hands with the tiles of `hands.json`
   (Hand 20's figure differs from the variant the study read from the prose).
7. **`never-32-or-33`** raises past them to 34 (the least level above), else passes.

Measured, exact over all 1,184,040 hands (the opening call):

| policy | pass | bids | the levels |
|---|---:|---:|---|
| `bid-only-enough` (the arena's arithmetic) | 780,843 (65.9%) | 403,197 | all 30 |
| `risk-budget` (our reading) | 647,781 (54.7%) | 536,259 | 30: 260,860; 31: 15,465; 32: 4,180; 33: 24,311; 34: 49,637; 35: 42,795; 36: 10,686; 37: 1,993; 38: 14,302; 39: 34,533; 40: 36,101; 41: 11,422; 42: 3,409; 84: 26,565 |

S-0053-d's "about half of all hands are not biddable" holds under the reading
(54.7%) and not under the arena's arithmetic (65.9%). S-0153-a's reasoned bids, 30,
31, 35 and 36, are 61.5% of the reading's bids, 30 alone 48.6%; 32 and 33 are 5.3%;
34 is bid more often than 35, and 39 and 40 nearly as often: the reading's levels
follow 42 less a count in fives and a few tricks, not the book's four. On the bid book `risk-budget`
makes the book's call on 18 of 30 (the commonest call, 30, on 8) and agrees on bid
or pass on 22 (always bid: 23; Zeb's forecast: 25): as a pass-or-bid rule our
reading does not beat bidding every hand.

Cost: one core decides `bid-only-enough`'s opening in 13 µs a hand, `risk-budget`'s
in 150 µs and the arena declaration in 210 µs; all hands take 1 s, 15 s and 20 s on
18 cores (each run alone; the three together in `tests/bidding.rs`, about 40 s). The cost is the encoding, not the arithmetic: a level is a literal, so
`risk-budget` is fifteen rules that each derive the risk again under every suit;
each bound enumerates sets of count tiles over the 28 dominoes per role; the
declaration compares suit lengths by enumeration. Of the time, about three
quarters is the scope search and clause evaluation and a fifth the predicates'
dispatch by name; scheduling a scope's clauses at compile time took a third off
(allocation had been 15%). The lever is gap 5: an aggregate and bound numbers.

**Amendment (the aggregate, 2026-10-13).** Gap 5 is built (README, Semantics:
`card`, `points`, `+` and `(<= a b)`, three-valued by intervals; numbers stay
literal-only, so a bid level is still a rule). `lib/bidding.silas` says each bound
in one clause, `bid-only-enough` declares by two rules instead of sixteen, and an
aggregate is counted once per decision across a section's rules. Every call and
declaration of the three policies is pinned equal to the encodings above on all
1,184,040 hands (`tests/bidding.rs`). One core: the opening 7.5 µs (13), the
reading 20 µs (150), the arena declaration 92 µs (210); the pinned run of all
three policies, 31 s on 18 cores against 117 s.

What of gap 4 (#134) this closes and what it leaves is `GAPS.md` § 4.

**Amendment (the bid as evidence, 2026-10-13).** The reading of a bid this slice left to
#128 is built for the auction (README, The record as evidence): a belief conditioned
on named seats' calls, and the declarer's declaration, being what their bidding
policies make from their hands, counted exactly and `exact_given` those policies. It is
the field-free half of a field: the same object a slice-3 field's seats would be, read
backwards. C4 is statable as a query; the offs readings of pp. 53–54 waited on a seat's
hand law, built 2026-10-16 (`Belief::hand_law`; `GAPS.md` § 4).

### Slice 3: the deal measure and the exact play-out (medium-high; two weeks)

**API.** `Deals::{Listed, Fiber, Hands, Uniform}`, `Seats`, `Field` of Silas
policies, `play_out(world, shaker, &Field, &MatchState) -> Settlement` (the auction
then 28 frames, every seat its own policy at its own frame, the hand stopped early
under marks once made or set), `compare` over `Listed`/`Fiber`/`Hands` with a Silas
field (`ExactGiven(F)`), over `Uniform` with Silas's seeded sample
(`⊓ Sampled`); the per-deal flips and the clustered interval; the chance baseline
(the arena's `RandomBidder` and lowest-legal play, as a Silas policy) run alongside
every comparison.
**Tests.** Conservation on every play-out (seven tricks, 28 plays, 42 points,
Lean's `terminal_scores`); the book's 36 worked deals played out under the
book-faithful policies reproduce the book's line at every decision a rule governs
(the chain of `tests/book_rules.rs`); determinism (same seed, same ledger); authority
is never `exact` (a field is a model); paired symmetry (`compare a a` is zero at
every deal; swapping `a` and `b` negates `delta` and the interval); the chance
baseline on a 500-deal sample has an interval containing zero against itself.
**Unblocks.** Cluster C's belief statements for a Silas field (12 rows statable;
the lay-downs' three marks settled); gap 7's straight-off set rate for a Silas
field (S-0089-a: "two in three"); cluster B's variance claims given a `V` built
from a Silas-field outcome table (7 rows). 19 rows priced against a Silas field; all
labelled with that field.

#### Slice 3 as built (2026-10-15)

Built: `Player`, `Field`, `Table`, `play_out` and `Settlement`, `World::deal`,
`Deals::{Listed, Uniform}`, `Seats`, `Trial::compare` and `Paired` with its
chance arm (`src/field.rs`; README § The field; `tests/field.rs`). Where this
spec was silent or did not fit the code, the choice made:

1. **Not built: `Fiber` and `Hands`.** The deal measures are the listed set
   and the uniform sample only; a comparison over one hand's completions or
   over the viewer's hands waits on its own slice, and nothing stands in for
   it (no variant, no stub).
2. **A seat is one `CompiledPolicy`.** A policy already carries its play,
   bidding and declaring sections and a fallback for each, and a reading
   already reads one policy for a seat's calls and plays; a pair would let a
   seat bid by one policy and play by another, which is a new policy, written
   as one. Beside it the chance player (`Player::Chance { seed }`), since a
   Silas policy is deterministic. `HandPolicySpec` is `Player`; `Named` is
   slice 4's.
3. **The field's identity is its players**, seat by seat (`NAME#DIGEST`, as a
   reading names a policy), not a caller's string, so a field cannot be named
   for what it is not. The auction's rules, the declarations offered and the
   match state are the `Table`, a parameter of the question, not of the model.
4. **The whole hand is played**, all 28 tiles, never stopped once made or set
   (the § 4 text said stop early under marks): the outcome is the points each
   side took, and conservation is a test.
5. **Passed out.** Four passes without the forced bid settle as
   `Settlement::PassedOut`: no contract, no play. The reshake is another deal,
   so the hand moves no score: zero in marks or points, and under a match-win
   value the value of the state itself. A comparison counts it like any other
   settlement, so a player that passes a hand out its rival bids is credited
   or charged the difference.
6. **Seatings.** `Rotate` is every chair at every shaker (16 a deal), `Fixed`
   one chair at every shaker (4): the shaker is symmetrized, not a property of
   the deal. The field's player in the chair under test sits out.
7. **The chance player** is the arena's `RandomBidder` (`p_bid` 3/10,
   uniform among the legal bids of at most one mark, a uniform pip trump) and
   lowest-legal play, drawn exactly (splitmix64 with rejection) and keyed on
   its seed and its auction frame. Where it may not pass it bids. The chance
   arm is `Player::CHANCE` (seed 0) in place of `b`, from the same play-outs
   of `a`; it has no chance arm of its own. "The self-arm (`a` against `a`
   under a reseed)" of § 5 is the chance player against a reseeded self.
8. **The interval.** Over a listed set the mean is exact given the field, so
   the interval is the mean and the verdict its sign. Over a sample it is the
   empirical Bernstein bound (Maurer and Pontil 2009, Theorem 4, both sides:
   `ln(4/δ)`), not Hoeffding's: it uses the per-deal sample variance, and
   paired differences are mostly ties. At 2,000 deals, `risk-budget` against
   `bid-only-enough` in the tests' field is −0.083 marks a deal, [−0.204,
   +0.039] at 95%: a half-width of 0.122, where Hoeffding's on the same range
   is 0.607. The range is the
   utility's a priori one, never the sample's: four times the most marks four
   calls reach under the cap (20 straight), four times a set marks bid at that
   most in points, 2 for a match-win value. `ln` and `sqrt` are bounded above
   in exact rationals (the `atanh` series with a geometric tail; an integer
   root), and the half-width rounded up to a millionth, so no float enters.
   The confidence is the trial's, a rational in (0, 1).
9. **No `(compare ...)` form and no `silas compare` yet.** The text form
   needs listed deals by name and a field of four policy files; it, and the
   command, land with slice 4's `silas compare --ledger FILE`, where the
   ledger names both.
10. **The replay check is not built.** The book's rules are 18 partial
    policies, each governing a slice of the decisions, with no stated
    precedence among them, and of `hands.json` only 8 examples are full
    deals, whose auctions are a convention (`Auction::won_by`). A play-out
    "under the book-faithful policies" needs one composed policy whose rule
    order is ours: new modelling, not a check.

Cost: release build, every core: 500 deals at one chair (2,000 seatings, three
arms) in about 0.15 s with the chance players; 2,000 deals rotated (32,000
seatings) of three bidding policies in about 2 s. `tests/field.rs` runs in
under a second.

### Slice 4: the arena ledger (medium; a week, mostly arena-side)

**API.** The ledger format of § 2.5; `Comparison::from_ledger(path, utility)`
recomputing every transfer through `settle` and labelling `ExactGiven(field) ⊓
Sampled(n, seed)`; `silas compare --ledger FILE`. Arena: `SilasBidder` and
`SilasPlayer` adapters, the fixed-state runner, the ledger writer, a named
overbidding field (`HeuristicBidder(caution = −k)`), the chance arm.
**Tests.** Round trip (a ledger from a Silas-only field equals Silas's own
play-out, tile for tile and transfer for transfer, the differential check); a
ledger that reports utilities is refused; the estimate equals the arena's paired
mean to the rational; the sign test agrees with the flips; a mislabelled unit
(marks ledger read as points) is refused by the scoring field.
**Unblocks (priced, against the named field).** Cluster D's 60 verdicts (the bid
book against Zeb and the net: the wave-1 25-of-30 becomes a `Comparison` with its
baseline); cluster C's 12 against an information-respecting field; cluster B's
variance claims with `V` from the arena's outcome table (7); cluster A's auction
hypothesis measured (`OvercallMonotone` and the raise histogram: 9); cluster E's
ten fixed-field rows against the named deviant field. About 98 rows receive a
measured verdict; none is `exact`.

#### Slice 4 as built (2026-10-15)

Built: the ledger (`Ledger`, `LedgerRow`, `Arm`, `Ledger::from_json` under a
`ledger` feature, `Paired::from_ledger`, `src/ledger.rs`) and `silas compare
--ledger FILE`; on the arena's side the fixed-state runner and ledger writer
(`arena/ledger.py`), `SilasSeat` (one Silas policy as a whole seat,
`arena/book_play.py`), `LowestPlay` (`arena/play.py`), the chance arm and
`overbidding_field(k)`. README § The arena ledger; `tests/ledger.rs`,
`arena/test_ledger.py`. Where this spec was silent, the choice made:

1. **Outcomes, checked.** A row carries the hands as dealt, the calls, the
   contract, the outcome and the plays; Silas replays every row (the auction
   under the table's rules, the contract from the auction, each play's turn,
   holding and legality, the tricks) and refuses a row whose outcome is not
   its plays'. The arena's word is checked, never taken.
2. **No other key.** Every key is required and no other allowed, so a
   utility in a ledger is refused by name, not ignored.
3. **The unit is the ledger's scoring.** A marks utility reads only a marks
   ledger, points only points, a match-win value only its own scoring: the
   players bid for one game, and reading their hands as another's measures
   players who were not playing it. `Trial::compare` refuses the same way
   (#183): the check is one, `field::check`, for both.
4. **The chance arm is in the ledger**, as rows of arm `chance` (the arena's
   `RandomBidder` and lowest-legal play), and is required: `Paired.chance` is
   never absent. Silas cannot replay it (the arena's random draws are not
   Silas's chance player's), so it is read, not replayed.
5. **Fixed-state runner.** Every hand from the ledger's score; a hand passed
   out (no forced bid) is a row with no contract, as `Settlement::PassedOut`.
   The deals are `random.Random(seed)` shuffles (the arena's stream, not
   Silas's), so a ledger's sample is `Sample::arena` and its label
   `sampled(n, seed, arena)`: the same (n, seed) in Silas's own stream is
   another sample, and `Trial::compare` refuses an arena sample (#183). Each
   seating's
   auction draws from a generator keyed on (seed, deal, chair, shaker), the
   same for every arm. Play runs in lockstep, one batched call per play
   policy a tick, so a GPU player amortizes.
6. **The field's identity** is the arena's seat specs, seat by seat, each
   play policy named with the seed it plays at and each field net (`F=`)
   with the first 16 hex digits of its SHA-256 (`arena.ledger.play_name`,
   #183), so two ledgers name one field alike only when they seated the
   same nets at the same seed. A deployed Zeb seat sits in the chair in
   `arena/test_ledger.py`.
7. **`SilasSeat`'s play frame and declaration carry the table's auction**
   (#183, closed here). `SilasFrame(..., auction=(shaker, calls))` takes the
   calls the arena ran, refused unless they end with the bidder winning at the
   bid; `SilasSeat` passes the state's dealer and bids (`table_auction`), and
   `run_auction` hands a bidder with `declare_at` the whole auction and the
   match score. A state that holds no whole legal auction (a hand set up from
   its contract) keeps `Auction::won_by`. What remains: the arena forces a
   shaker's pass to 30, so Silas sees a bid of 30 where the policy might have
   passed (`forced` is not recorded in Zeb's state); a rule reading that call
   would differ from Silas asked with the pass made illegal. No policy in the
   round trip has one, and the points ledger is forced.
8. **No `(compare ...)` text form**: the command reads the ledger, and the
   utility (the ledger's unit by default) and the confidence (19/20) are its
   options.

**The differential check found no disagreement.** Two ledgers of a field of
five Silas policies (`bid-only-enough`, `pounce`, `risk-budget`,
`protect-the-count`; `risk-budget` against `never-32-or-33` in the chair),
in marks from 0-0 with every chair rotated and the straight auction (hands
passed out), and in points from 200-180 at one chair with the forced
shaker: every a and b row (192 hands, 4 of them passed out) is Silas's own `play_out`, call for
call and tile for tile, under Zeb's engine and rules on one side and
`Straight42` on the other, and each ledger reads to the arena's paired mean
exactly (+1/8 marks a deal; −133/16 points a deal) with the arena's flips.

### Slice 5: mixed policies and a response map (high; out of scope until 4 lands)

`(mix (q σ₁) (1 − q σ₂))` as a `FinPMF` over policies; `Field::respond(σ)` by the
arena (a field trained on a pour against `σ`, or a ladder); the claim
`∃ q > 0, Δ(σ_q, σ_0; respond(σ_q)) > 0` as a curve in `q`. Unblocks cluster E's four
equilibrium rows. Not scheduled; named so that no fixed-field number is cited for them.

---

## 5. Pitfalls specific to this project's history

1. **Chance baselines, always.** The book study's rule: every agreement number ships
   with its chance baseline, and a tie is not agreement. For bids: "always bid"
   scores 23 of 30 on the bid book, so 25 of 30 is two hands; a random bidder's
   `Comparison` is reported beside every priced bid rule. For `compare`: the
   chance arm (random bids, lowest-legal play) and the self-arm (`a` against `a`
   under a reseed) bound what the interval can mean. `Comparison.chance` is not
   optional.

2. **Units: marks per hand, marks per game, match-win, points.** The arena reports
   marks per *game* to seven over paired halves, E1b reported marks per *hand* at
   0–0, the bid book spoke in make probability, and the book speaks in points. Every
   claim in cluster B flips sign between units ("84 is right" is `p > 1/2` in linear
   marks and false at the last mark in match-win). The `utility` is declared in the
   form, the ledger carries no utilities, and `settle` is the only place a unit is
   made. The field-net units bug of 2026-10-10 (a probability divided by two points)
   is the precedent.

3. **Dead and decided states.** The pour dropped 525,116 rule positions decided by
   the banked score; the match has the same: a decided match (`won` is `Some`) and a
   hand that cannot change the outcome have `V` flat, and including them dilutes `Δ`
   toward zero. Under marks the hand stops once made or set, so a points utility on
   a marks play-out is undefined past that play. Passed-out deals (all four pass) are
   a redeal in the book and `None` in the arena; a comparison must say whether they
   count as zero or are redrawn, because a passing policy "wins" them for free and a
   bid-rate claim (S-0037-a, S-0156-a's anecdote of winning 7–1 without a bid)
   turns on exactly that. Paired deals where both policies reach the same contract
   and play the same tiles are ties; the effective sample is the flips, not `n`.

4. **The field model's own bias.** The field net imitates a bid-30 Walt; every pour
   is bid 30, its bid feature is constant, and it has no match-state input. Against
   it, "bid only enough" is trivially true (nothing overcalls), "keep them honest" is
   vacuous (nothing adapts), and match-score-conditioned play is measured against a
   seat that ignores the score. The field's identity string must include its training
   pour and its `field_eps`, and a cluster-B verdict against a score-blind field is
   labelled as such. Walt's `make` utility is not the book's points utility: a field
   playing for the make under a points comparison is the wrong referent
   (scenario-player.md records a points-trained field behind on marks for this
   reason). And the oracle is *algorithmia non grata*: `(clairvoyant-dd-make)` bounds cluster C's
   `p` from above and must never be the field, because the book's reason for the 84
   (the defenders cannot tell what to save) is invisible to it.

5. **Equilibrium versus fixed field.** A fixed-field `Comparison` can show that a
   given field is exploitable; it cannot confirm or refute "keep them honest". The
   straight-off 84 is set two in three against any sensible fixed field, so the
   number will read as a refutation of a claim about adaptation. Cluster E's four
   rows carry the `equilibrium` tag in the coverage until slice 5 and no
   `Comparison` is cited for them.

6. **Selected hands are not a distribution.** The book's 30 bid-book hands and 36
   worked deals are chosen to illustrate; `Deals::Listed` says "over these deals"
   in its authority and the wiki says it in words. The Ch. 16 shapes are checked on
   two hands each for this reason (`tests/book_odds.rs`), and a bid rule is checked
   on `Hands` (all of them) before it is called a rule.

7. **The book against itself.** "With a double protecting the off, always bid 84"
   (S-0089-b) and Hand 26's "35 or 36, 84 only if far behind" with the 5-5 ahead of
   the 5-2 (S-0122-a) differ by the trump count (two trumps, a 36% chance of being
   outtrumped). The encoded shape is S-0083-a's full one (three or four trumps, two or
   three doubles, the off under its double); the header records both passages, as
   `GAPS.md`'s "The book against itself" does for the play rules.

8. **Authority discipline.** Nothing in a comparison is `exact` once a field is in
   it; self-play is a model; `Listed`/`Fiber`/`Hands` make the deal part exact and the
   field part `ExactGiven`. The meet never upgrades (`Authority::at_least`); a
   `Comparison` built from a ledger cannot reach `ExactGiven` alone because the
   ledger's `Sample` is in it. The band is derived from `n`, never chosen.

9. **The seat and the shaker.** The bid depends on the chair (the last bidder sees
   three bids), so a bid rule is tested in every chair under every shaker, as the
   play rules are at all four seatings today, and `Seats::Rotate` pairs the four
   seatings of a deal as one cluster.

---

## Appendix: row assignment

**A (9).** S-0028-d, S-0033-b, S-0034-a, S-0036-e, S-0053-a, S-0128-c, S-0137-e,
S-0152-c+S-0153-b, S-0153-a.

**B (13).** S-0047-a, S-0082-a, S-0089-b, S-0089-c+S-0167-a, S-0089-d, S-0110-a,
S-0122-a, S-0127-b, S-0129-b+S-0137-g+S-0080-d, S-0155-c, S-0164-c, S-0165-b, S-0170-a.

**C (12).** S-0082-b, S-0083-a, S-0083-b, S-0084-b, S-0086-c, S-0088-a, S-0088-b,
S-0089-a+S-0101-c, S-0146-a, S-0149-a, S-0150-a, S-0188-a.

**D (60).** S-0021-a, S-0021-c, S-0021-d, S-0024-a, S-0024-b, S-0024-c, S-0024-d,
S-0024-e, S-0025-a, S-0025-b, S-0026-a, S-0026-d, S-0027-d, S-0028-a, S-0028-b,
S-0028-c, S-0029-a, S-0030-a, S-0031-a, S-0031-b, S-0032-a, S-0033-a+S-0053-b,
S-0034-b, S-0035-a, S-0035-b, S-0036-b, S-0036-c, S-0036-d+S-0053-c, S-0046-e,
S-0050-a, S-0053-d, S-0103-a+S-0106-b, S-0103-b, S-0104-a, S-0104-b, S-0105-e,
S-0106-c, S-0107-a, S-0107-e, S-0109-a, S-0110-b, S-0121-a, S-0121-b,
S-0121-c+S-0233-h+S-0123-c, S-0124-e, S-0125-a, S-0125-b, S-0128-b, S-0129-a,
S-0129-c, S-0134-e, S-0142-a, S-0142-b, S-0142-c, S-0153-c, S-0160-c,
S-0197-b+S-0234-e, S-0232-b, S-0232-c, S-0232-d.

**E (14).** Equilibrium: S-0037-a, S-0081-a, S-0089-e, S-0154-a. Fixed-field against
a named field: S-0070-a, S-0080-b, S-0080-c, S-0114-b, S-0155-a, S-0184-a+S-0197-a,
S-0188-b, S-0190-a, S-0191-a+S-0226-a, S-0205-a.

**F (18).** S-0015-c, S-0017-c+S-0132-a, S-0020-a, S-0063-a+S-0154-b, S-0063-b,
S-0072-b, S-0073-a, S-0113-a+S-0209-b, S-0114-a, S-0156-a, S-0166-a, S-0166-b,
S-0168-b, S-0199-b, S-0208-b, S-0208-c, S-0208-e, S-0210-e (S-0114-b is counted in
E).

**G (14).** S-0085-b, S-0087-b, S-0090-a+S-0096-a, S-0092-a+S-0097-a, S-0092-b,
S-0095-a, S-0098-a, S-0098-b, S-0099-a, S-0099-b, S-0100-a, S-0100-b, S-0100-c,
S-0101-a.

Rows citing #130 and another gap keep the other: #134 on 15 (the auction's reading),
#128 on 6, #135 on 4, #132 on 1, gap 7 on 1. Those parts are not this gap's.
