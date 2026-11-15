// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

// Time the wasm over the conformance corpus's unforced decisions (168 requests, six hands).
import { readFileSync } from 'node:fs';
import { initSync, ask } from '../index.js';

const here = (p) => new URL(p, import.meta.url);
initSync({ module: readFileSync(here('../pkg/walt_bg.wasm')) });
const rows = readFileSync(here('../conformance/corpus.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
for (const r of rows.slice(0, 20)) ask(r.req); // warm the engine's compiler
const ms = [];
for (const r of rows) {
  const t = performance.now();
  const out = ask(r.req);
  if (!out.forced) ms.push(performance.now() - t);
}
ms.sort((a, b) => a - b);
const at = (q) => ms[Math.min(ms.length - 1, Math.floor(q * ms.length))].toFixed(1);
const mean = (ms.reduce((a, b) => a + b, 0) / ms.length).toFixed(1);
console.log(`decisions ${ms.length}  mean ${mean} ms  median ${at(0.5)}  p90 ${at(0.9)}  max ${at(1)}`);
