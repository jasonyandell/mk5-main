// Grade a clean-room walt1 build against the reference corpus (hands 1..5 are held out).
// usage: node grade.mjs <binary> [maxRows]
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
const BIN = process.argv[2], MAX = +(process.argv[3] || 1e9);
const rows = readFileSync(new URL('./corpus.jsonl', import.meta.url), 'utf8').trim().split('\n').map(JSON.parse)
  .filter(r => r.hand >= 1).slice(0, MAX);
const t = { rows: 0, err: 0, badJson: 0, illegal: 0, leaderOk: 0, pointsOk: 0, forcedOk: 0,
  decisions: 0, choiceMatch: 0, valueAbsErrSum: 0, valueN: 0, sameBestValue: 0, ms: 0, timeouts: 0 };
for (const r of rows) {
  t.rows++;
  let out, start = Date.now();
  try { out = execFileSync(BIN, { input: r.req, timeout: 120000 }).toString().trim(); }
  catch (e) { if (e.code === 'ETIMEDOUT' || e.signal) t.timeouts++; else t.err++; continue; }
  t.ms += Date.now() - start;
  let g; try { g = JSON.parse(out.split('\n').pop()); } catch { t.badJson++; continue; }
  if (g.error) { t.err++; continue; }
  const ref = r.res;
  if (g.leader === ref.leader) t.leaderOk++;
  if (JSON.stringify(g.points) === JSON.stringify(ref.points)) t.pointsOk++;
  if (g.forced === ref.forced) t.forcedOk++;
  const legal = ref.forced ? [ref.choice] : ref.opts.map(o => o[0]);
  if (!legal.includes(g.choice)) { t.illegal++; continue; }
  if (ref.forced) continue;
  t.decisions++;
  if (g.choice === ref.choice) t.choiceMatch++;
  const refV = Object.fromEntries(ref.opts), gV = Object.fromEntries(g.opts || []);
  for (const [tile, v] of ref.opts) if (tile in gV) { t.valueAbsErrSum += Math.abs(gV[tile] - v); t.valueN++; }
  if (refV[g.choice] === refV[ref.choice]) t.sameBestValue++;  // picked a tile the reference values equally
}
const pct = (a, b) => b ? (100 * a / b).toFixed(1) + '%' : '-';
console.log(JSON.stringify({
  rows: t.rows, errors: t.err, timeouts: t.timeouts, badJson: t.badJson, illegalChoice: t.illegal,
  leader: pct(t.leaderOk, t.rows), points: pct(t.pointsOk, t.rows), forced: pct(t.forcedOk, t.rows),
  decisions: t.decisions, choiceMatch: pct(t.choiceMatch, t.decisions),
  choiceEqualValueToRef: pct(t.sameBestValue, t.decisions),
  meanAbsValueErr_bp: t.valueN ? Math.round(t.valueAbsErrSum / t.valueN) : null,
  meanMsPerRequest: t.rows ? Math.round(t.ms / t.rows) : null,
}, null, 1));
