// A tournament seat for a pinned JavaScript/WebAssembly player, one JSON line
// per decision over stdin/stdout. The Python referee (arena/engine.py) owns the
// game; this process sees only what its seat may see: its own dealt tiles, the
// auction so far, and the public plays.
//
//   node tournament/seat/host.mjs <kind> <artifact dir> <config json>
//
// kinds
//   zeb    @jasonyandell/zeb from an unpacked, integrity-checked npm tarball.
//          Bids with the package's own rule (forecast + chooseBid), behind the
//          table conventions Plunge applies around it: pass while partner holds
//          the high bid; Zeb never bids past one mark.
//   walt   Plunge's production Walt: walt-player.wasm and the bid book at the
//          deployed Plunge commit, driven exactly as Plunge drives them
//          (src/ai/auction.ts auctionMove, src/ai/book-auction.ts,
//          src/ai/phone/auction-pool.ts, src/ai/native.ts livePlayerCall,
//          difficulty 'native-partner'). Ported line for line; the source
//          lines are cited where each rule lives.
//
// requests  {op:"bid", hand, seat, high_bid, high_seat, legal, game, hand_no}
//           {op:"declare", hand, seat, bid, game, hand_no}
//           {op:"play", hand, seat, decl, bid, bidder, plays:[[seat,tile]...],
//                       leader, points, game, hand_no}
//           {op:"version"}
// replies   {bid} | {decl} | {tile} | {version} | {error}
import { readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { Worker } from 'node:worker_threads';
import { pathToFileURL } from 'node:url';
import { join } from 'node:path';

const [kind, dir, configText] = process.argv.slice(2);
const config = JSON.parse(configText ?? '{}');
const DECLS = [0, 1, 2, 3, 4, 5, 6, 7, 9];
const cap42 = (bid) => Math.min(bid, 42);

// ---------------------------------------------------------------- Zeb

async function zebSeat() {
  const pkg = join(dir, 'package');
  const zeb = await import(pathToFileURL(join(pkg, 'index.js')).href);
  zeb.initSync({ module: readFileSync(join(pkg, 'pkg', 'zeb_bg.wasm')) });
  const worlds = config.worlds ?? 160;
  const cutoff = config.cutoff ?? 0.8;
  const forecasts = new Map();
  const forecastOf = (hand) => {
    const key = [...hand].sort((a, b) => a - b).join(',');
    if (!forecasts.has(key)) {
      if (forecasts.size > 64) forecasts.clear();
      forecasts.set(key, zeb.forecast({ hand: [...hand], worlds, seed: 0 }));
    }
    return forecasts.get(key);
  };
  return {
    version: () => ({ version: zeb.version(), net: zeb.netId() }),
    bid(r) {
      if (r.high_seat >= 0 && r.high_seat % 2 === r.seat % 2) return { bid: 0 }; // partner holds it
      const need = r.legal.find((v) => v <= 42);
      if (need === undefined) return { bid: 0 }; // only marks above one remain
      const choice = zeb.chooseBid(forecastOf(r.hand), { need, cutoff });
      return { bid: choice && r.legal.includes(choice.bid) ? choice.bid : 0 };
    },
    declare(r) {
      // The declaration chooseBid leans on at this target: most likely to make
      // it, the first listed on a tie (also what a forced 30 gets).
      const t = cap42(r.bid) - 30;
      let best = null;
      for (const f of forecastOf(r.hand)) if (!best || f.pmake[t] > best.pmake[t]) best = f;
      return { decl: best.decl };
    },
    play(r) {
      const out = zeb.play({ decl: r.decl, bidder: r.bidder, bid: cap42(r.bid), seat: r.seat,
        hand: [...r.hand], plays: r.plays, worlds, seed: 0 });
      return { tile: out.tile };
    },
  };
}

// ---------------------------------------------------------------- Walt

// Plunge src/engine/rng.ts toSeed (FNV-1a), src/ai/native.ts nativeSeed.
function toSeed(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return h >>> 0;
}
const nativeSeed = (gameId, handNumber) => toSeed(`sunshine/${gameId}/${handNumber}`);

async function waltSeat() {
  const wasm = readFileSync(join(dir, 'walt-player.wasm'));
  const module = await WebAssembly.compile(wasm);
  const book = JSON.parse(readFileSync(join(dir, 'played.json'), 'utf8'));
  const [thrNum, thrDen] = book.threshold; // [4, 5]
  const key = (hand, seat) => `${seat}:${[...hand].sort((a, b) => a - b).join(',')}`;
  const byHand = new Map(book.hands.map((h) => [key(h.hand, h.seat), h]));
  // src/ai/bid-book.ts bestPanel / qualifies
  const bestPanel = (panels, bid) =>
    panels.reduce((a, b) => (b.tails[bid - 30] * a.games > a.tails[bid - 30] * b.games ? b : a));
  const qualifies = (p, bid) => p.tails[bid - 30] * thrDen >= p.games * thrNum;

  const decoder = new TextDecoder();
  const call = (obj) => {
    let x;
    const instance = new WebAssembly.Instance(module, { walt_host: {
      now_us: () => BigInt(Math.floor(performance.now() * 1000)), checkpoint: () => {} } });
    x = instance.exports;
    const bytes = new TextEncoder().encode(JSON.stringify(obj));
    const ptr = x.walt_in_prepare(bytes.length);
    new Uint8Array(x.memory.buffer, ptr, bytes.length).set(bytes);
    const len = x.walt_call();
    const out = JSON.parse(decoder.decode(new Uint8Array(x.memory.buffer, x.walt_out_ptr(), len)));
    if (out && out.error) throw new Error(`walt: ${out.error}`);
    return out;
  };

  // src/ai/phone/auction-pool.ts: rounds 4, 12, 40, 160 worlds; each round
  // prices the nine declarations on a pool of min(4, cores − 1) workers under
  // one shared wall deadline, then the Rust merger ranks the round; the last
  // complete round is kept when time runs out.
  const POOL = 4, AUCTION_WORLDS = 160, AUCTION_BUDGET_MS = 20000;
  const workerSource = `
    const { parentPort, workerData } = require('node:worker_threads');
    const module = new WebAssembly.Module(workerData);
    const decoder = new TextDecoder();
    parentPort.on('message', (obj) => {
      try {
        let x;
        const instance = new WebAssembly.Instance(module, { walt_host: {
          now_us: () => BigInt(Math.floor(performance.now() * 1000)), checkpoint: () => {} } });
        x = instance.exports;
        const bytes = new TextEncoder().encode(JSON.stringify(obj));
        const ptr = x.walt_in_prepare(bytes.length);
        new Uint8Array(x.memory.buffer, ptr, bytes.length).set(bytes);
        const len = x.walt_call();
        parentPort.postMessage({ result: JSON.parse(decoder.decode(new Uint8Array(x.memory.buffer, x.walt_out_ptr(), len))) });
      } catch (e) { parentPort.postMessage({ error: String(e) }); }
    });`;
  const pool = Array.from({ length: POOL }, () =>
    new Worker(workerSource, { eval: true, workerData: wasm }));
  for (const w of pool) w.unref();
  const onWorker = (w, obj) => new Promise((resolve, reject) => {
    w.once('message', (m) => (m.error ? reject(new Error(m.error)) : resolve(m.result)));
    w.postMessage(obj);
  });

  async function survey(request) {
    const start = performance.now(), end = start + AUCTION_BUDGET_MS;
    let saved = call({ auction_merge: request, worlds: 0, receipts: [] });
    for (const worlds of [4, 12, 40, AUCTION_WORLDS]) {
      const queue = [...DECLS], receipts = [];
      let late = false;
      await Promise.all(pool.map(async (w) => {
        while (queue.length && !late) {
          const budget = Math.floor(end - performance.now() - 80);
          if (budget < 5) { late = true; break; }
          const r = await onWorker(w, { auction_price: request, decl: queue.shift(), worlds, budget_ms: budget });
          if (r && r.error) { late = true; break; }
          receipts.push(r);
        }
      }));
      if (late || receipts.length !== DECLS.length || performance.now() >= end) break;
      const merged = call({ auction_merge: request, worlds, receipts });
      if (merged.worlds !== worlds) break;
      saved = merged;
    }
    return saved;
  }

  const surveys = new Map(); // auctionMove's `previous`: the same request reuses its survey
  async function surveyOf(request) {
    const k = JSON.stringify([request.seat, request.bid, request.seed, request.hand]);
    if (!surveys.has(k)) {
      if (surveys.size > 64) surveys.clear();
      surveys.set(k, await survey(request));
    }
    return surveys.get(k);
  }
  const target = (v) => (v <= 41 ? v : 42); // auction.ts target(): any plain marks bid is 42
  const requestOf = (r, bid) => ({ hand: [...r.hand].sort((a, b) => a - b), seat: r.seat, bid,
    seed: nativeSeed(r.game, r.hand_no) });

  return {
    version: () => ({ version: 'walt-table-v2', wasm_bytes: wasm.length }),
    async bid(r) {
      if (r.high_seat >= 0 && r.high_seat % 2 === r.seat % 2) return { bid: 0 }; // partnerAuctionPass
      const minimum = r.legal.find((v) => target(v) !== null);
      const request = requestOf(r, target(minimum));
      const entry = byHand.get(key(request.hand, request.seat));
      if (entry) { // book-auction.ts: highest supported target; at 42 the cheapest plain-marks bid
        let action = null, bid = request.bid;
        for (const v of r.legal) {
          const t = target(v);
          if (qualifies(bestPanel(entry.panels, t), t) && (action === null || t > bid)) { action = v; bid = t; }
        }
        return { bid: action ?? 0 };
      }
      const s = await surveyOf(request);
      if (!s.eligible) return { bid: 0 };
      return { bid: r.legal.find((v) => target(v) === request.bid) };
    },
    async declare(r) {
      const request = requestOf(r, cap42(r.bid));
      const entry = byHand.get(key(request.hand, request.seat));
      if (entry) return { decl: bestPanel(entry.panels, request.bid).decl };
      return { decl: (await surveyOf(request)).decl };
    },
    play(r) {
      const request = { decl: r.decl, bid: cap42(r.bid), bidder: r.bidder, seat: r.seat,
        hand: [...r.hand].sort((a, b) => a - b), plays: r.plays.flat(), seed: nativeSeed(r.game, r.hand_no) };
      // native.ts livePlayerCall, difficulty 'native-partner'
      const c = request.seat === request.bidder && request.plays.length === 0
        ? { request, worlds: 160, partner: false, budget_ms: 20000 }
        : { request, worlds: 40, partner: true };
      const out = call(c);
      // native.ts checkedAction: Walt and the table must agree on the position.
      if (r.leader !== undefined && out.leader !== r.leader) throw new Error(`leader ${out.leader} != table ${r.leader}`);
      if (r.points !== undefined && JSON.stringify(out.points) !== JSON.stringify(r.points)) {
        throw new Error(`points ${JSON.stringify(out.points)} != table ${JSON.stringify(r.points)}`);
      }
      if (!out.legal.includes(out.choice)) throw new Error(`illegal choice ${out.choice}`);
      return { tile: out.choice };
    },
  };
}

// ---------------------------------------------------------------- the loop

const seat = kind === 'zeb' ? await zebSeat() : kind === 'walt' ? await waltSeat()
  : (() => { throw new Error(`unknown seat kind ${kind}`); })();
const lines = createInterface({ input: process.stdin });
for await (const line of lines) {
  if (!line.trim()) continue;
  let reply;
  try {
    const r = JSON.parse(line);
    if (!(r.op in seat)) throw new Error(`unknown op ${r.op}`);
    reply = await seat[r.op](r);
  } catch (e) {
    reply = { error: String(e && e.stack ? e.stack : e) };
  }
  process.stdout.write(JSON.stringify(reply) + '\n');
}
process.exit(0);
