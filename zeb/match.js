// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: MIT

/**
 * Match-aware bidding: the hand-only call against one more option, a mark
 * bid (84, or one mark over an 84), priced by match equity.
 *
 * A port of arena/match_equity.py (`HandLaw`, `MatchEquity`, `Odds`,
 * `one_mark_value`, `mark_bid_value`) and arena/zeb_bid.py (`price`,
 * `best_at`, the decision in `ZebBidder.bid`, mode "met"). The Python is the
 * referee: arena/test_zeb_match_parity.py drives both over thousands of
 * contexts and requires values within 1e-12 and identical calls.
 *
 * Pure JavaScript, no WebAssembly. A tied value keeps the hand-only call.
 */

const ONE_MARK = 42;
const TWO_MARKS = 84;

/**
 * The deployed field's law (Zeb bids at n160 and plays, both sides), fitted
 * on the FIT block of wiki/experiments/match-aware-bidding-2026-10-14.md
 * (384 games, 4,251 hands) and frozen before the primary was planned. Source
 * of the numbers: ZEB_LAW in arena/zeb_bid.py.
 * ZEB_LAW[w][f]: w = the first shaker's side took the mark, f = the next
 * first shaker sits on the other side.
 */
const ZEB_LAW = [
  [0.1682, 0.3312],
  [0.1701, 0.3305],
];

/**
 * What happens to a one-mark call at our turn, as the field bids it: the
 * same numbers as ZEB_ODDS in arena/zeb_bid.py (fitted on the same FIT block
 * of wiki/experiments/match-aware-bidding-2026-10-14.md). Keys are
 * `${holder}:${after}`, and for KEEP `:low` (bids 30..34) or `:high` (35..41).
 */
const KEEP = {
  'none:0:low': 1.0, 'none:0:high': 1.0, 'none:1:low': 0.9212, 'none:1:high': 0.9913,
  'none:2:low': 0.938, 'none:2:high': 0.9891, 'none:3:low': 0.8818, 'none:3:high': 0.9871,
  'opp:0:low': 1.0, 'opp:0:high': 1.0, 'opp:1:low': 0.9901, 'opp:1:high': 1.0,
  'opp:2:low': 0.9901, 'opp:2:high': 1.0, 'opp:3:low': 0.9901, 'opp:3:high': 0.9901,
};
const OVER_SET = 0.1727;
const PASS_OURS = {
  'none:0': 0.5049, 'none:1': 0.4285, 'none:2': 0.5053, 'none:3': 0.4445,
  'partner:0': 0.8254, 'partner:1': 0.7963, 'partner:2': 0.81, 'partner:3': 0.81,
  'opp:0': 0.1688, 'opp:1': 0.1746, 'opp:2': 0.2037, 'opp:3': 0.1799,
};

/** Odds.take: the chance our side takes the hand's one mark after `call` (0 = pass). */
function take(call, pMake, holder, after) {
  const key = `${holder}:${after}`;
  if (!(key in PASS_OURS)) throw new RangeError(`no odds for holder ${holder}, passesAfter ${after}`);
  if (call === 0) return PASS_OURS[key];
  const keep = call >= ONE_MARK ? 1.0 : KEEP[`${key}:${call < 35 ? 'low' : 'high'}`];
  if (keep === undefined) throw new RangeError(`no odds for a point bid of ${call} with holder ${holder} (behind a partner the hand-only call is a pass)`);
  return keep * pMake + (1.0 - keep) * OVER_SET;
}

const equities = new Map();

/** MatchEquity(ZEB_LAW, target): V(a, b, d), our side at a marks, theirs at b, d = 1 when we shake first. */
function matchEquity(target) {
  let v = equities.get(target);
  if (v !== undefined) return v;
  const memo = new Map();
  const solve = (a, b, d) => {
    if (a >= target) return 1.0;
    if (b >= target) return 0.0;
    let total = 0.0;
    for (const w of [0, 1]) {
      for (const f of [0, 1]) {
        const p = ZEB_LAW[w][f];
        if (p === 0.0) continue;
        const ours = w === d;
        const nd = f ? 1 - d : d;
        total += p * (ours ? at(a + 1, b, nd) : at(a, b + 1, nd));
      }
    }
    return total;
  };
  const at = (a, b, d) => {
    a = Math.min(a, target);
    b = Math.min(b, target);
    const k = `${a},${b},${d}`;
    let r = memo.get(k);
    if (r === undefined) memo.set(k, (r = solve(a, b, d)));
    return r;
  };
  equities.set(target, at);
  return at;
}

const oneMarkValue = (v, a, b, nd, pi) => pi * v(a + 1, b, nd) + (1.0 - pi) * v(a, b + 1, nd);
const markBidValue = (v, a, b, nd, marks, p) => p * v(a + marks, b, nd) + (1.0 - p) * v(a, b + marks, nd);

/**
 * `price` in arena/zeb_bid.py: E[V | c0], E[V | markCall] and the call with
 * the larger value (a tie keeps c0). `c0` is the hand-only bid (0 = pass)
 * made with probability `pC0`; `markCall` (84 or more) is made with
 * probability `p42`. `after` is the seats still to speak.
 */
export function matchPrice({ us, them, marksToWin = 7, weShakeNext, holder, after, c0, pC0, markCall, p42 }) {
  const v = matchEquity(marksToWin);
  const nd = weShakeNext ? 1 : 0;
  const eC0 = oneMarkValue(v, us, them, nd, take(c0, pC0, holder, after));
  const eMark = markBidValue(v, us, them, nd, Math.floor(markCall / ONE_MARK), p42);
  return { eC0, eMark, call: eMark > eC0 ? markCall : c0 };
}

/** best_at(fc, 42): the declaration most likely to take all 42 (the first listed, on a tie). */
function bestAt42(allowed) {
  let best = null;
  for (const f of allowed) {
    const p = f.pmake[ONE_MARK - 30];
    if (best === null || p > best.pmake) best = { decl: f.decl, pmake: p };
  }
  return best;
}

const HOLDERS = ['none', 'partner', 'opp'];

/** Check a `match` option; throws on a malformed one. */
export function checkMatch(match) {
  if (match === null || typeof match !== 'object') throw new TypeError('match must be an object');
  for (const k of ['us', 'them']) {
    if (!Number.isInteger(match[k]) || match[k] < 0) {
      throw new RangeError(`match.${k} must be a non-negative integer, not ${match[k]}`);
    }
  }
  const { marksToWin = 7, holder, passesAfter, maxBid } = match;
  if (!Number.isInteger(marksToWin) || marksToWin < 1) {
    throw new RangeError(`match.marksToWin must be a positive integer, not ${marksToWin}`);
  }
  if (typeof match.weShakeNext !== 'boolean') throw new TypeError('match.weShakeNext must be a boolean');
  if (!HOLDERS.includes(holder)) throw new RangeError(`match.holder must be none, partner or opp, not ${holder}`);
  if (!Number.isInteger(passesAfter) || passesAfter < 0 || passesAfter > 3) {
    throw new RangeError(`match.passesAfter must be an integer in 0..3, not ${passesAfter}`);
  }
  if (!Number.isInteger(maxBid) || maxBid < 30) {
    throw new RangeError(`match.maxBid must be the highest legal bid, not ${maxBid}`);
  }
}

/**
 * chooseBid's match-aware step. `hand` is the hand-only choice (a BidChoice
 * or null). Returns the mark-bid choice when its match equity beats the
 * hand-only call's, else `hand`.
 */
export function chooseWithMatch(allowed, hand, match) {
  checkMatch(match);
  const { us, them, marksToWin = 7, holder, passesAfter, maxBid, pointsToWin = false } = match;
  // No mark option in a points-to-win game (the equity is the marks game's),
  // when no mark bid is legal, or with nothing forecast to read a p42 from.
  if (pointsToWin || maxBid < TWO_MARKS || allowed.length === 0) return hand;
  const at42 = bestAt42(allowed);
  const priced = matchPrice({
    us, them, marksToWin, weShakeNext: match.weShakeNext, holder, after: passesAfter,
    c0: hand === null ? 0 : hand.bid, pC0: hand === null ? 0.0 : hand.pmake, markCall: maxBid, p42: at42.pmake,
  });
  return priced.call === maxBid ? { bid: maxBid, decl: at42.decl, pmake: at42.pmake } : hand;
}
