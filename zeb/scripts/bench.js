// Wall time of the calls that matter, in Node on this machine: a forecast
// over the seven pip declarations, doubles and no-trump on their own, and
// an opening lead's play, at 160 and 512 worlds, over the fixtures' nine
// opening hands. `npm run bench` (after `npm run build`).
import { readFileSync } from 'node:fs';

import { forecast, initSync, play } from '../index.js';

const read = (path) => readFileSync(new URL(path, import.meta.url));
initSync({ module: read('../pkg/zeb_bg.wasm') });
const hands = JSON.parse(read('../tests/fixtures/parity.json'))
  .positions.filter((p) => p.referee.forecast)
  .map((p) => p.request.hand);

/** Milliseconds per call of `f` over every hand: [median, slowest]. */
function time(f) {
  const ms = hands.map((hand) => {
    const t0 = performance.now();
    f(hand);
    return performance.now() - t0;
  });
  ms.sort((a, b) => a - b);
  return `${ms[ms.length >> 1].toFixed(0).padStart(5)} ms median, ${ms.at(-1).toFixed(0).padStart(5)} ms slowest`;
}

/**
 * Mid-hand positions under sixes: `deals` seeded deals, each played out by
 * Zeb (8 worlds, reading off) from seat 0's lead; every unforced decision.
 */
function midhand(deals) {
  const out = [];
  for (let d = 0; d < deals; d++) {
    let s = BigInt(d + 1);
    const rand = () => Number((s = (s * 6364136223846793005n + 1442695040888963407n) % 2n ** 64n) >> 33n);
    const tiles = [...Array(28).keys()];
    for (let i = 27; i > 0; i--) {
      const j = rand() % (i + 1);
      [tiles[i], tiles[j]] = [tiles[j], tiles[i]];
    }
    const dealt = [0, 1, 2, 3].map((seat) => tiles.slice(7 * seat, 7 * seat + 7));
    const plays = [];
    let leader = 0;
    for (let n = 0; n < 28; n++) {
      const seat = (leader + (n % 4)) % 4;
      const pos = { decl: 6, bidder: 0, bid: 30, seat, hand: dealt[seat], plays: [...plays] };
      const m = play({ ...pos, worlds: 8, reading: false });
      if (!m.forced && n > 0) out.push({ ...pos, trick: (n >> 2) + 1 });
      plays.push([seat, m.tile]);
      if (n % 4 === 3) leader = winner(plays.slice(-4));
    }
  }
  return out;
}

/** The seat that took a trick under sixes (decl 6): the high six, else the high tile of the led suit. */
function winner(trick) {
  const pips = (t) => {
    const hi = Math.floor((Math.sqrt(8 * t + 1) - 1) / 2);
    return [hi, t - (hi * (hi + 1)) / 2];
  };
  const rank = (t, led) => {
    const [hi, lo] = pips(t);
    if (hi === 6 || lo === 6) return 100 + (hi === lo ? 7 : hi === 6 ? lo : hi);
    const suit = led;
    if (hi !== suit && lo !== suit) return -1;
    return hi === lo ? 50 : (hi === suit ? lo : hi);
  };
  const [lh, ll] = pips(trick[0][1]);
  const led = lh === 6 || ll === 6 ? 6 : lh;
  let best = 0;
  for (let i = 1; i < 4; i++) if (rank(trick[i][1], led) > rank(trick[best][1], led)) best = i;
  return trick[best][0];
}

forecast({ hand: hands[0], declarations: [6], worlds: 8 }); // warm up
for (const worlds of [160, 512]) {
  console.log(`worlds ${worlds}`);
  console.log(`  forecast, 7 pip declarations: ${time((hand) => forecast({ hand, declarations: [0, 1, 2, 3, 4, 5, 6], worlds }))}`);
  console.log(`  forecast, doubles-trump:      ${time((hand) => forecast({ hand, declarations: [7], worlds }))}`);
  console.log(`  forecast, no-trump:           ${time((hand) => forecast({ hand, declarations: [9], worlds }))}`);
  console.log(`  play, opening lead, sixes:    ${time((hand) => play({ decl: 6, bidder: 0, bid: 30, seat: 0, hand, plays: [], worlds }))}`);
}

// Reading's cost: the nine opening leads, and every unforced mid-hand decision of six deals under sixes,
// reading on (the default) and off, interleaved, by trick. CPU time (process.cpuUsage), so a
// busy machine skews the ratio less than it would wall time.
const positions = midhand(6);
const quantile = (xs, p) => xs.sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor(p * xs.length))];
for (const worlds of [160, 512]) {
  console.log(`play, mid-hand, sixes, worlds ${worlds}: CPU ms median / p90, reading on | off`);
  const openings = hands.map((hand) => ({ decl: 6, bidder: 0, bid: 30, seat: 0, hand, plays: [], trick: 0 }));
  for (let trick = 0; trick <= 6; trick++) {
    const at = trick === 0 ? openings : positions.filter((p) => p.trick === trick);
    const ms = { true: [], false: [] };
    for (const { trick: _, ...p } of at) {
      for (const reading of [true, false]) {
        const t0 = process.cpuUsage();
        play({ ...p, worlds, reading });
        const { user, system } = process.cpuUsage(t0);
        ms[reading].push((user + system) / 1000);
      }
    }
    const cell = (xs) => `${quantile(xs, 0.5).toFixed(0).padStart(4)} / ${quantile(xs, 0.9).toFixed(0).padStart(4)}`;
    const label = trick === 0 ? 'opening lead' : `trick ${trick}`;
    console.log(`  ${label.padEnd(12)} (${String(at.length).padStart(2)} decisions): ${cell(ms.true)} | ${cell(ms.false)}`);
  }
}
