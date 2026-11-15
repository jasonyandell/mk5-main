#!/usr/bin/env node
// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

// A seat for your own bot: one `walt1 play` request on stdin, one JSON line on stdout.
// Everything here is bookkeeping; your bot goes in `choose` at the bottom. As shipped it
// plays the lowest legal tile, which is a lawful and terrible player.
import { readFileSync } from 'node:fs';

const HIGH = [], LOW = [];
for (let h = 0; h <= 6; h++) for (let l = 0; l <= h; l++) { HIGH.push(h); LOW.push(l); }
const isDouble = (t) => HIGH[t] === LOW[t];
const has = (t, pip) => HIGH[t] === pip || LOW[t] === pip;
const COUNT = (t) => (HIGH[t] + LOW[t]) % 5 === 0 && HIGH[t] + LOW[t] > 0 ? HIGH[t] + LOW[t] : 0;

// decl 0..6: that pip is trumps. 7: doubles are trumps. 9: no trumps.
const isTrump = (decl, t) => (decl <= 6 ? has(t, decl) : decl === 7 ? isDouble(t) : false);
// The suit a led tile calls for: 'T' for trumps, else its high pip.
const ledSuit = (decl, t) => (isTrump(decl, t) ? 'T' : HIGH[t]);
const follows = (decl, suit, t) => (suit === 'T' ? isTrump(decl, t) : !isTrump(decl, t) && has(t, suit));
// How strongly a tile takes a trick led in `suit`; 0 when it cannot.
function power(decl, suit, t) {
  const rank = (pip) => (isDouble(t) ? 7 : HIGH[t] === pip ? LOW[t] : HIGH[t]);
  if (isTrump(decl, t)) return 100 + (decl === 7 ? HIGH[t] : rank(decl));
  return suit !== 'T' && has(t, suit) ? 1 + rank(suit) : 0;
}

function position(text) {
  const f = {};
  for (const line of text.split('\n').slice(1)) {
    const [key, ...v] = line.trim().split(/\s+/);
    if (key) f[key] = v.map(Number);
  }
  const [decl, bid, seat, bidder] = ['decl', 'bid', 'seat', 'bidder'].map((k) => f[k][0]);
  const hand = new Set(f.hand);
  let leader = bidder, trick = [];
  const points = [0, 0], tricks = [];
  for (let i = 0; i < (f.plays ?? []).length; i += 2) {
    const [who, tile] = [f.plays[i], f.plays[i + 1]];
    if (who === seat) hand.delete(tile);
    trick.push({ seat: who, tile });
    if (trick.length < 4) continue;
    const suit = ledSuit(decl, trick[0].tile);
    const winner = trick.reduce((a, b) => (power(decl, suit, b.tile) > power(decl, suit, a.tile) ? b : a));
    points[winner.seat & 1] += 1 + trick.reduce((n, p) => n + COUNT(p.tile), 0);
    tricks.push({ leader, plays: trick, winner: winner.seat });
    leader = winner.seat;
    trick = [];
  }
  const mine = [...hand].sort((a, b) => a - b);
  const suit = trick.length ? ledSuit(decl, trick[0].tile) : null;
  const following = suit === null ? [] : mine.filter((t) => follows(decl, suit, t));
  const legal = following.length ? following : mine;
  return { decl, bid, seat, bidder, hand: mine, legal, tricks, trick, leader, points };
}

// Your bot. `p` holds the seat's own tiles and the public record, nothing else:
//   decl, bid, seat, bidder      the contract (seats 0 and 2 are partners)
//   hand, legal                  tile ids the seat still holds / may play now
//   tricks                       finished tricks: { leader, plays: [{ seat, tile }], winner }
//   trick, leader, points        the trick in progress, who led it, each side's count
// A tile id is high * (high + 1) / 2 + low; HIGH[id] and LOW[id] give the pips back.
function choose(p) {
  return p.legal[0];
}

const p = position(readFileSync(0, 'utf8'));
console.log(JSON.stringify({
  v: 1, kind: 'play', choice: p.legal.length === 1 ? p.legal[0] : choose(p), forced: p.legal.length === 1,
  leader: p.leader, points: p.points,
}));
