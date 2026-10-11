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

forecast({ hand: hands[0], declarations: [6], worlds: 8 }); // warm up
for (const worlds of [160, 512]) {
  console.log(`worlds ${worlds}`);
  console.log(`  forecast, 7 pip declarations: ${time((hand) => forecast({ hand, declarations: [0, 1, 2, 3, 4, 5, 6], worlds }))}`);
  console.log(`  forecast, doubles-trump:      ${time((hand) => forecast({ hand, declarations: [7], worlds }))}`);
  console.log(`  forecast, no-trump:           ${time((hand) => forecast({ hand, declarations: [9], worlds }))}`);
  console.log(`  play, opening lead, sixes:    ${time((hand) => play({ decl: 6, bidder: 0, bid: 30, seat: 0, hand, plays: [], worlds }))}`);
}
