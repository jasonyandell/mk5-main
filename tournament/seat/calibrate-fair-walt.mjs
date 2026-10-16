// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

// Measure the fair Walt's clock step (#181), by the rule registered in
// wiki/experiments/fair-walt-2026-10-16.md. Run it under `scripts/cpu-slot --exclusive`.
//
//   node tournament/seat/calibrate-fair-walt.mjs reads <artifacts> <panel.json> [passes]
//     Runs every panel call on the pinned wasm under the REAL clock, one at a time, and counts
//     the wasm's now_us reads. Prints JSON: per call {kind, reads, sec} (sec = the median of
//     `passes` passes), and the pooled rate = total reads / total seconds.
//   node tournament/seat/calibrate-fair-walt.mjs host <artifacts> <panel.json> <config json>
//     Sends the panel through tournament/seat/host.mjs under that config, and prints each
//     request's wall seconds and reply (the match check: wall per decision and the moves).
import { readFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const [mode, dir, panelFile, extra] = process.argv.slice(2);
const panel = JSON.parse(readFileSync(panelFile, 'utf8'));
const requests = [...panel.bid, ...panel.open, ...panel.mid];
const DECLS = [0, 1, 2, 3, 4, 5, 6, 7, 9];

function toSeed(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return h >>> 0;
}
const nativeSeed = (g, n) => toSeed(`sunshine/${g}/${n}`);
const target = (v) => (v <= 41 ? v : 42);

async function reads() {
  const module = await WebAssembly.compile(readFileSync(join(dir, 'walt-player.wasm')));
  const run = (obj) => {
    let n = 0;
    const x = new WebAssembly.Instance(module, { walt_host: {
      now_us: () => { n++; return BigInt(Math.floor(performance.now() * 1000)); }, checkpoint: () => {} } }).exports;
    const bytes = new TextEncoder().encode(JSON.stringify(obj));
    new Uint8Array(x.memory.buffer, x.walt_in_prepare(bytes.length), bytes.length).set(bytes);
    const t = performance.now();
    x.walt_call();
    return { reads: n, sec: (performance.now() - t) / 1000 };
  };
  // The calls a request makes: a play is one call; a bid is the survey's four rounds of nine
  // pricing jobs (the merges are negligible and read no clock to speak of).
  const callsOf = (r) => {
    if (r.op === 'play') {
      const request = { decl: r.decl, bid: Math.min(r.bid, 42), bidder: r.bidder, seat: r.seat,
        hand: [...r.hand].sort((a, b) => a - b), plays: r.plays.flat(), seed: nativeSeed(r.game, r.hand_no) };
      return r.seat === r.bidder && r.plays.length === 0
        ? [{ request, worlds: 160, partner: false, budget_ms: 20000 }]
        : [{ request, worlds: 40, partner: true }];
    }
    const minimum = r.legal.find((v) => target(v) !== null);
    const request = { hand: [...r.hand].sort((a, b) => a - b), seat: r.seat, bid: target(minimum),
      seed: nativeSeed(r.game, r.hand_no) };
    return [4, 12, 40, 160].flatMap((worlds) =>
      DECLS.map((decl) => ({ auction_price: request, decl, worlds, budget_ms: 20000 })));
  };
  const passes = Number(extra ?? 1);
  const out = [];
  for (const r of requests) {
    const kind = r.op === 'bid' ? 'bid' : r.plays.length === 0 && r.seat === r.bidder ? 'open' : 'mid';
    const cs = callsOf(r);
    const tries = Array.from({ length: passes }, () => cs.map(run));
    const secs = cs.map((_, i) => tries.map((t) => t[i].sec).sort((a, b) => a - b)[passes >> 1]);
    out.push({ kind, calls: cs.length, reads: tries[0].reduce((a, c) => a + c.reads, 0),
      sec: secs.reduce((a, b) => a + b, 0) });
  }
  const total = (f) => out.filter(f).reduce((a, c) => ({ reads: a.reads + c.reads, sec: a.sec + c.sec, n: a.n + 1 }),
    { reads: 0, sec: 0, n: 0 });
  const rate = (t) => ({ ...t, reads_per_s: t.reads / t.sec });
  console.log(JSON.stringify({ per_request: out, pooled: rate(total(() => true)),
    bid: rate(total((c) => c.kind === 'bid')), open: rate(total((c) => c.kind === 'open')),
    mid: rate(total((c) => c.kind === 'mid')) }));
}

async function host() {
  const here = dirname(fileURLToPath(import.meta.url));
  const p = spawn('node', [join(here, 'host.mjs'), 'walt', dir, extra], { stdio: ['pipe', 'pipe', 'inherit'] });
  const rl = createInterface({ input: p.stdout })[Symbol.asyncIterator]();
  const rows = [];
  for (const r of requests) {
    const t = performance.now();
    p.stdin.write(JSON.stringify(r) + '\n');
    const { value } = await rl.next();
    rows.push({ op: r.op, sec: (performance.now() - t) / 1000, reply: JSON.parse(value) });
  }
  p.stdin.end();
  console.log(JSON.stringify(rows));
}

await (mode === 'reads' ? reads() : mode === 'host' ? host() : Promise.reject(new Error('mode: reads | host')));
